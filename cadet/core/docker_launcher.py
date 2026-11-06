"""Launchers that start a role in a container, locally or over SSH.

What this is for:
    The orchestrator runs on the host and must start role processes wherever a
    node lives. :class:`DockerLauncher` wraps a role in ``docker run`` on this
    machine; :class:`SshDockerLauncher` does the same on another machine over
    SSH, optionally through a jump host. Both satisfy the same
    :class:`~cadet.core.launcher.Launcher` interface as the local launcher, so
    the orchestrator's ordering and teardown logic is unchanged.

What it contains:
    - :class:`ContainerSpec`: the per-node container settings (image, GPU,
      capabilities, mounts - outputs, staged model artifacts and, for a machine
      with no checkout, the run's own config) that a launcher turns into flags.
    - :class:`DockerLauncher`: ``docker run`` on this machine.
    - :class:`SshDockerLauncher`: the same command over ``ssh``, with a dry-run
      mode that returns the command instead of running it. *How* the other
      machine is reached - the ssh options, the quoting, the facts a remote
      placement has to ask it - is :class:`~cadet.core.ssh.SshTarget`, shared
      with the venv launcher; this class decides only what runs there.
    - :func:`container_spec_for`: builds a spec from a config node.

How it fits into CADET:
    Selected per node by ``nodes[*].launcher``: ``docker`` and ``ssh_docker``
    respectively. Each role still runs the same ``cadet serve|client|load``
    argv, so a node moves between host, container and remote machine by
    changing one config field.

    One container per role, named after the run and node, is what makes teardown
    reliable: stopping a run is removing those containers, and a crashed
    orchestrator leaves names that can be cleaned up by pattern.

How to configure it:
    Per node: ``launcher``, ``container`` (image), ``gpu_index``, ``user``,
    ``host``, ``ssh_port``, ``ssh_identity``. The image defaults to
    :data:`DEFAULT_IMAGE`.
"""

import os
import subprocess
import time
from typing import Dict, List, Optional

from cadet.core.launcher import Handle, Launcher, LaunchSpec, LaunchStatus, child_env
from cadet.core.ssh import SshTarget
from cadet.core.staging import CONTAINER_MODELS

# Image used when a node does not name one. `CADET_NODE_IMAGE` overrides it,
# for the same reason CADET_STATE_ROOT and CADET_SCENARIO_RUNNER_DIR exist:
# which image this machine should run is a property of the machine, not of the
# experiment, so it does not belong in a config that is shared between machines.
#
# What it is for concretely: `cadet:dev` is a moving tag that anyone can rebuild,
# so a result cannot be attributed to the code it ran on by naming it. Pointing
# this at an immutable tag (`cadet:demo-2`, say) pins every placed node in
# one place, instead of a --set per node per command - and the image's recorded
# commit then says which code it was.
DEFAULT_IMAGE_TAG = "cadet:dev"
DEFAULT_IMAGE_ENV = "CADET_NODE_IMAGE"


def default_image() -> str:
    """The image a node with no ``container`` runs, honouring the environment.

    Read per call rather than captured at import, so a test or a wrapper can set
    the variable after this module is imported and still be obeyed.
    """
    return os.environ.get(DEFAULT_IMAGE_ENV) or DEFAULT_IMAGE_TAG


# Kept as a name because callers read it; it is the unoverridden default, and
# anything choosing an image for a node should call default_image() instead.
DEFAULT_IMAGE = DEFAULT_IMAGE_TAG

# Every container (and any docker network) CADET creates starts with this, which
# is what makes a leftover recognisable as CADET's own; see
# cadet.orchestrator.recovery.
OWNED_CONTAINER_PREFIX = "cadet-"
# Where the repo lives inside the image (matches both Dockerfiles' WORKDIR).
CONTAINER_WORKDIR = "/app"
# The docker network name that means "share this machine's own namespace".
HOST_NETWORK = "host"
# Outputs are bind-mounted so results survive the container and land on the host.
CONTAINER_OUTPUTS = "/app/outputs"
# Where a staged run config is mounted, when one is. Deliberately not under
# /app: the repository the image carries is at /app, and a config mounted there
# would shadow or sit beside a file of the repository's own, which is confusing
# in exactly the case this exists for - a config that is NOT in the image.
CONTAINER_RUN_CONFIG = "/cadet/run-config.yaml"


class ContainerSpec:
    """Container settings for one node's role process.

    ``needs_net_admin`` is set for a node whose link uses the netem backend: it
    installs its own qdisc with ``tc`` and cannot do so without NET_ADMIN. The
    capability is requested per node rather than globally, so only the nodes that
    shape traffic get it.
    """

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        name: Optional[str] = None,
        gpu_index: Optional[int] = None,
        all_gpus: bool = False,
        needs_net_admin: bool = False,
        outputs_dir: Optional[str] = None,
        sysctls: Optional[Dict[str, str]] = None,
        env: Optional[Dict[str, str]] = None,
        network: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        run_as: Optional[str] = None,
        models_dir: Optional[str] = None,
        config_file: Optional[str] = None,
    ):
        self.image = image
        self.name = name
        self.gpu_index = gpu_index
        # Every GPU on the machine, for a node that asks for a GPU without
        # naming which. That is what the same node gets when its role runs
        # locally - a process sees the machine's devices - and a container that
        # saw none instead did not refuse: it loaded for twelve seconds and
        # then died inside torch with no CUDA device, which reads as a broken
        # model rather than as a container that was given no GPU.
        self.all_gpus = all_gpus
        self.needs_net_admin = needs_net_admin
        self.outputs_dir = outputs_dir
        # A directory on the machine this container runs on holding the model
        # artifacts the role will load, mounted read-only: a role reads
        # artifacts and has no business writing them, and read-only is also what
        # makes a shared per-machine cache safe to mount into several runs.
        self.models_dir = models_dir
        # A resolved config file on the machine this container runs on, mounted
        # read-only at :data:`CONTAINER_RUN_CONFIG`. Set for a node whose
        # machine has no copy of the repository, where the config path the
        # orchestrator was given names nothing: the role is handed the config
        # itself instead of a path it would have to resolve against a checkout
        # it does not have.
        self.config_file = config_file
        # Namespaced TCP settings. Docker applies these at creation, which is the
        # only workable moment: an unprivileged container's /proc/sys is
        # read-only, so they cannot be set from inside afterwards.
        self.sysctls = sysctls or {}
        self.env = env or {}
        # The docker network this container joins. A node on this machine joins
        # the run's own network and is reached at the address docker gives it.
        # A node on another machine joins that machine's own namespace
        # (:data:`HOST_NETWORK`) and is reached at that machine's address; see
        # cadet.orchestrator.placement for why publishing a port is not enough.
        self.network = network
        self.extra_args = extra_args or []
        # "uid:gid" to run the role as. Set when a host directory is bind-mounted
        # in: the image has its own unprivileged user, whose uid need not match
        # the host's, and a mismatch means the role cannot write to the mount.
        # Results are meant to land on the host owned by whoever ran the run.
        self.run_as = run_as

    def docker_args(self) -> List[str]:
        """Return the ``docker run`` flags (without the image or the command)."""
        args = ["--rm"]
        if self.name:
            args += ["--name", self.name]
        if self.gpu_index is None and self.all_gpus:
            args += ["--gpus", "all"]
        if self.gpu_index is not None:
            # Pin one device rather than passing all of them, so two nodes on one
            # machine do not silently share a GPU.
            args += ["--gpus", "device={}".format(self.gpu_index)]
            # CUDA_VISIBLE_DEVICES still has to be set, because a privileged
            # container bypasses the device cgroup and sees every GPU, so the
            # choice has to be made inside the process too. But its value is
            # *not* the host index in the ordinary case: `--gpus device=N`
            # filters the devices through the cgroup and the one that survives
            # is renumbered to 0, so naming N again hides the only GPU the
            # container has. Measured on a two-GPU host, with `torch` in the
            # image reporting what it could see:
            #   --gpus device=1, CUDA_VISIBLE_DEVICES=1 -> device_count() 0
            #   --gpus device=1, CUDA_VISIBLE_DEVICES=0 -> the host's GPU 1
            #   --gpus device=0, CUDA_VISIBLE_DEVICES=0 -> the host's GPU 0
            #   --privileged --gpus device=1, CUDA_VISIBLE_DEVICES=0 -> GPU 0,
            #     because privileged bypasses the filter and the numbering is
            #     the host's again (NVIDIA_VISIBLE_DEVICES does not override it)
            # So: 0 for a container whose devices docker filtered, and the host
            # index for one that can see all of them.
            visible = self.gpu_index if "--privileged" in self.extra_args else 0
            args += ["-e", "CUDA_VISIBLE_DEVICES={}".format(visible)]
        if self.needs_net_admin:
            args += ["--cap-add", "NET_ADMIN"]
        if self.outputs_dir:
            args += ["-v", "{}:{}".format(self.outputs_dir, CONTAINER_OUTPUTS)]
        if self.models_dir:
            args += ["-v", "{}:{}:ro".format(self.models_dir, CONTAINER_MODELS)]
        if self.config_file:
            args += ["-v", "{}:{}:ro".format(self.config_file, CONTAINER_RUN_CONFIG)]
        if self.run_as:
            args += ["--user", self.run_as]
        for key, value in sorted(self.sysctls.items()):
            args += ["--sysctl", "{}={}".format(key, value)]
        if self.network:
            args += ["--network", self.network]
        for key, value in sorted(self.env.items()):
            args += ["-e", "{}={}".format(key, value)]
        args += self.extra_args
        return args


def host_user() -> str:
    """Return this process's ``uid:gid``, for a container writing to a mount."""
    return "{}:{}".format(os.getuid(), os.getgid())


def container_spec_for(
    node,
    run_id: str,
    outputs_dir: Optional[str] = None,
    needs_net_admin: bool = False,
    run_as: Optional[str] = None,
    models_dir: Optional[str] = None,
    config_file: Optional[str] = None,
) -> ContainerSpec:
    """Build a :class:`ContainerSpec` for a config node.

    The container name ties a container to its run and node, so leftovers are
    identifiable and a second run cannot collide with a first.

    When outputs are bind-mounted the container runs as the uid that owns the
    mount, so the role can write into it and the files it leaves belong to
    whoever started the run rather than to the image's user. That uid is this
    process's by default, which is right for a container on this machine and
    wrong for one on another: pass ``run_as`` to name the account on the machine
    the container will actually run on.
    """
    from cadet.network.tuning import sysctl_settings

    return ContainerSpec(
        image=node.container or default_image(),
        name="{}{}-{}".format(OWNED_CONTAINER_PREFIX, run_id, node.name),
        gpu_index=node.gpu_index,
        all_gpus=getattr(node, "device", None) == "gpu",
        needs_net_admin=needs_net_admin,
        outputs_dir=outputs_dir,
        sysctls=sysctl_settings(node),
        run_as=run_as or (host_user() if outputs_dir else None),
        models_dir=models_dir,
        config_file=config_file,
    )


# A role in a container runs as the invoking account's uid, which the image has
# no passwd entry for, so the daemon gives it ``/`` as a home - and the image's
# unprivileged user cannot write there. Anything that caches under the home then
# fails: the CARLA client egg extracts itself there, and a containerised braking
# client died with "Perhaps your account does not have write access to this
# directory?" before it had read a frame. ``/tmp`` in the container is writable,
# per-container and thrown away with it, which is what a cache wants.
CONTAINER_HOME = "/tmp"


def container_env(spec: LaunchSpec) -> Dict[str, str]:
    """Return the environment for a role that runs in a container.

    The local environment is not inherited by a container, so anything a role
    needs has to be named here or on its argv.
    """
    env = {"HOME": CONTAINER_HOME}
    env.update(child_env(spec))
    return env


class DockerLauncher(Launcher):
    """Starts each role in its own container on this machine.

    The role's argv is passed straight through to the image's entry point, so
    the command a node runs is identical to the local launcher's apart from
    where it executes.
    """

    def __init__(self, specs: Optional[Dict[str, ContainerSpec]] = None, docker: str = "docker"):
        self.specs = specs or {}
        self.docker = docker

    def _node_of(self, spec: LaunchSpec) -> str:
        # Spec names are "<sub>:<node>", e.g. "serve:server".
        return spec.name.split(":", 1)[-1]

    def container_spec(self, spec: LaunchSpec) -> ContainerSpec:
        """Return the container settings for this role, or a default."""
        return self.specs.get(self._node_of(spec), ContainerSpec())

    def command(self, spec: LaunchSpec) -> List[str]:
        """Return the full command this launcher would run (useful for dry runs)."""
        cspec = self.container_spec(spec)
        argv = self._role_argv(spec)
        # The spec's own environment travels with it, as it does locally. The
        # container spec's env is separate and already in docker_args(): it
        # describes the node, this describes the command.
        env_args = []
        for key, value in sorted(container_env(spec).items()):
            env_args += ["-e", "{}={}".format(key, value)]
        return (
            [self.docker, "run"] + cspec.docker_args() + env_args + [cspec.image] + argv
        )

    def _role_argv(self, spec: LaunchSpec) -> List[str]:
        """Strip the host interpreter prefix: the image's entry point is cadet.

        The orchestrator builds argv as ``<python> -m cadet <sub> ...`` for local
        execution. Inside the image ``cadet`` is the entry point, so the
        interpreter and module flags are dropped and the rest is kept verbatim.
        """
        argv = list(spec.argv)
        if "-m" in argv:
            index = argv.index("-m")
            if index + 1 < len(argv) and argv[index + 1] == "cadet":
                argv = argv[index + 2 :]
        return argv

    def launch(self, spec: LaunchSpec) -> Handle:
        log_fh = open(spec.log_path, "w") if spec.log_path else subprocess.DEVNULL
        proc = subprocess.Popen(
            self.command(spec),
            stdout=log_fh if log_fh is not subprocess.DEVNULL else subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return Handle(spec, proc, log_fh if log_fh is not subprocess.DEVNULL else None)

    def wait_ready(self, handle: Handle, timeout_s: Optional[float] = None) -> bool:
        marker = handle.spec.readiness_marker
        if marker is None:
            return handle.proc.poll() is None
        limit = handle.spec.readiness_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + limit
        path = handle.spec.log_path
        while time.monotonic() < deadline:
            if handle.proc.poll() is not None:
                return False
            if path and os.path.exists(path):
                try:
                    with open(path) as fh:
                        if marker in fh.read():
                            return True
                except OSError:
                    pass
            time.sleep(0.2)
        return False

    def poll(self, handle: Handle) -> str:
        code = handle.proc.poll()
        if code is None:
            return LaunchStatus.RUNNING
        return LaunchStatus.STOPPED if code == 0 else LaunchStatus.FAILED

    def stop(self, handle: Handle, grace_s: float = 5.0) -> None:
        """Ask the container to stop, then remove it and reap the client.

        Killing the local ``docker run`` client does not necessarily stop the
        container, so the container is dealt with by name; ``--rm`` then cleans
        up the rest.

        ``docker stop`` rather than ``rm -f`` for the role's own container: the
        role is that container's main process, so a stop delivers it the same
        ``SIGTERM`` a local role gets and it runs its shutdown path. Removed
        outright it is killed, and what a role writes while finishing - a
        server's per-request metrics among it - is simply absent. The removal
        still follows, because a container that ignores the signal must not
        outlive the run.
        """
        cspec = self.container_spec(handle.spec)
        if cspec.name:
            subprocess.run(
                [self.docker, "stop", "--time", str(int(max(1, grace_s))), cspec.name],
                capture_output=True,
                text=True,
            )
            subprocess.run([self.docker, "rm", "-f", cspec.name], capture_output=True, text=True)
        if handle.proc.poll() is None:
            try:
                handle.proc.terminate()
                handle.proc.wait(timeout=grace_s)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    handle.proc.kill()
                except OSError:
                    pass
        if handle.log_fh is not None:
            try:
                handle.log_fh.close()
            except OSError:
                pass


class SshDockerLauncher(SshTarget, DockerLauncher):
    """Starts each role in a container on another machine, over SSH.

    ``dry_run`` makes every operation return the command it would have run
    without contacting anything, which is how a two-host setup can be checked
    before any remote access exists.

    The channel itself - batch mode, the explicit connect timeout, an untouched
    ``known_hosts`` - is :class:`~cadet.core.ssh.SshTarget`'s, and is shared
    with the venv launcher so there is one place that policy can be read from.
    """

    def __init__(
        self,
        host: str,
        user: Optional[str] = None,
        port: Optional[int] = None,
        identity: Optional[str] = None,
        proxy_jump: Optional[str] = None,
        specs: Optional[Dict[str, ContainerSpec]] = None,
        dry_run: bool = False,
        connect_timeout_s: int = 10,
    ):
        DockerLauncher.__init__(self, specs=specs)
        SshTarget.__init__(
            self,
            host=host,
            user=user,
            port=port,
            identity=identity,
            proxy_jump=proxy_jump,
            connect_timeout_s=connect_timeout_s,
        )
        self.dry_run = dry_run

    def command(self, spec: LaunchSpec) -> List[str]:
        """Return the ssh command wrapping the remote ``docker run``."""
        return self.remote_command(DockerLauncher.command(self, spec))

    def launch(self, spec: LaunchSpec) -> Handle:
        if self.dry_run:
            raise RuntimeError(
                "this launcher is in dry-run, which contacts nothing and so "
                "cannot start a role. Inspect the command it would have run "
                "with command(); clear dry_run to run it."
            )
        return DockerLauncher.launch(self, spec)

    def stop(self, handle: Handle, grace_s: float = 5.0) -> None:
        """Stop the remote container, remove it, then reap the local ssh client.

        The stop is what lets the role on the other machine finish: see
        :meth:`DockerLauncher.stop` for why removal alone loses what a role
        writes on its way out.
        """
        if self.dry_run:
            return
        cspec = self.container_spec(handle.spec)
        if cspec.name:
            subprocess.run(
                self.ssh_prefix()
                + ["docker", "stop", "--time", str(int(max(1, grace_s))), cspec.name],
                capture_output=True,
                text=True,
            )
            subprocess.run(
                self.ssh_prefix() + ["docker", "rm", "-f", cspec.name],
                capture_output=True,
                text=True,
            )
        if handle.proc.poll() is None:
            try:
                handle.proc.terminate()
                handle.proc.wait(timeout=grace_s)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    handle.proc.kill()
                except OSError:
                    pass
        if handle.log_fh is not None:
            try:
                handle.log_fh.close()
            except OSError:
                pass
