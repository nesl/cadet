"""Place each node where its config says, and give it an address its peers can use.

What this is for:
    The orchestrator used to build one launcher for a whole run and that
    launcher was local, so every role ran as a process on the host and
    ``nodes[*].launcher`` was a field that validated and then did nothing. This
    module turns it into the dispatch it was always meant to be: each node gets
    the launcher its config asks for, and the nodes that live in containers get
    a container, a network and an address their peers can reach.

What it contains:
    - :class:`NodeNetwork`: a run-scoped docker network for the roles' own
      traffic.
    - :class:`NodeContainer`: one node's container, plus its pid (a handle on its
      network namespace) and a runner that executes commands inside it.
    - :class:`RemoteOutputs`: the staging outputs tree for a node on another
      machine, fetched back and removed at teardown. Shared by both remote
      launchers: a role in a venv writes its outputs on that machine exactly as
      a role in a container there does.
    - the model staging in :mod:`cadet.core.staging`, which puts the artifacts
      the registry names on the machine each containerised role runs on - local
      or remote, because a container on this machine cannot see this machine's
      search directories either.
    - :class:`ExecLauncher`: starts a role *inside* an existing node container.
    - :class:`Placement`: builds all of the above for one run, hands out a
      launcher per node, and takes everything down in reverse.
    - :func:`address_overrides`: the ``--set`` strings that move a containerised
      node onto an address its peers can resolve.
    - :func:`shaping_nodes`: which nodes install a ``netem`` qdisc of their own,
      shared with the preflight check that refuses shaping a node whose
      namespace the orchestrator cannot enter.

How it fits into CADET:
    :meth:`cadet.orchestrator.runner.Orchestrator.run_one` asks for a
    :class:`Placement`, then launches each role with
    ``placement.launcher_for(node)``. Teardown stops the roles first and the
    placement second, so a container never outlives the role inside it.

    **A node container is a place, not a process.** It is started as a bare
    ``sleep`` and roles are ``docker exec``-ed into it, rather than being the
    container's own command. That costs one extra step and buys two things the
    one-shot form cannot give:

      * **Deterministic ordering for shaping.** A ``netem`` qdisc has to be on
        the sender's interface *before* it sends. If the role were the
        container's command it would start sending the moment the container
        existed, so the qdisc would be installed into a race. With the container
        up first, the order is: container, qdisc, role.
      * **A namespace that can be joined.** The emulated backend attaches a veth
        to a node's network namespace by its pid, which requires the namespace
        to exist before the role does.

    **Addresses come from the placement, not from the config.** A containerised
    server binds, and is reached at, the IP docker assigns it on the run's own
    network; the config keeps saying ``localhost``, which is what the node is
    called from the machine's point of view. Rewriting one field is the whole of
    it, exactly as the emulated backend already does for its own hosts - and it
    is why moving a node into a container, or later onto another machine, stays
    a config change.

    The address is the **IP** and not the container name, although the name is
    what docker's embedded DNS resolves. Two things need the address and cannot
    use a name: ``tc`` matches a flow with ``u32 match ip dst <addr>`` and
    rejects anything that is not numeric, and the orchestrator runs outside the
    network, where that DNS does not answer.

How to configure it:
    ``nodes[*].launcher`` selects the launcher: ``local`` (a host process),
    ``docker`` (a container on this machine), ``ssh_docker`` (a container on
    another machine, reached over SSH as ``nodes[*].user@nodes[*].host`` with
    ``ssh_port`` and ``ssh_identity``), ``ssh_venv`` (a plain process in the
    virtualenv ``nodes[*].venv`` on that machine, over the same SSH channel).
    ``nodes[*].container`` names the image, ``gpu_index`` the device.

    **A venv node is told paths and not mounts.** A containerised role is given
    the paths *inside* its container - the outputs mount, the mounted config,
    the mounted model directory - because that is where the staged trees appear
    to it. A venv role has no mount: it is given those same staged trees by
    their own paths on its own machine. Which is why the overrides and the
    placement argv are built per node rather than once per run.

    **A remote node keeps its configured address.** A containerised node on this
    machine is rewritten onto the address docker gave it, because nothing
    outside the run's network can reach it otherwise. A node on another machine
    is already configured with that machine's address, which is where its peers
    reach it and where the SSH goes, so the placement publishes the node's port
    there and tells the role to bind an address its container holds instead -
    through argv, because the config is hashed into the run id.
"""

import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Callable, Dict, List, Optional, Tuple

import yaml

from cadet.core.docker_launcher import (
    CONTAINER_OUTPUTS,
    CONTAINER_RUN_CONFIG,
    HOST_NETWORK,
    OWNED_CONTAINER_PREFIX,
    ContainerSpec,
    DockerLauncher,
    SshDockerLauncher,
    container_env,
    container_spec_for,
    default_image,
)
from cadet.core.launcher import (
    Handle,
    LaunchSpec,
    LocalLauncher,
    role_python_version,
)
from cadet.core.ssh import RemoteError, SshTarget
from cadet.core.staging import (
    ModelStaging,
    StagingError,
    TargetMachine,
    artifacts_for,
    describe_staging,
    staged_model_args,
)
from cadet.core.venv_launcher import (
    SshVenvLauncher,
    VenvError,
    describe_venv,
    venv_facts,
)

# What a node container runs so that it exists without doing anything. The
# image's entry point is the cadet CLI, so it is overridden.
IDLE_ENTRYPOINT = "sleep"
IDLE_ARGS = ["infinity"]
# The console script a role is started with. `docker exec` does not use the
# image's ENTRYPOINT, so unlike `docker run` it has to be named.
ROLE_COMMAND = "cadet"
# How long to wait for a just-created container to report a pid.
PID_TIMEOUT_S = 10.0
_POLL_S = 0.2
# What a role in a container listens on when its peers reach it at the address
# of the machine the container runs on, not at an address the container holds.
CONTAINER_BIND_HOST = "0.0.0.0"
# Directory under the remote account's home that holds one staging tree per run.
REMOTE_OUTPUTS_PARENT = "{}outputs".format(OWNED_CONTAINER_PREFIX)
# Where a staged run config goes on the machine the role runs on, under that
# account's home for the same reason the outputs tree is: it is the one
# directory the account is certain to own.
REMOTE_CONFIG_PARENT = "{}configs".format(OWNED_CONTAINER_PREFIX)
# Where a venv node's pid files go on the other machine, under that account's
# home for the same reason the outputs tree is. Deliberately *not* inside the
# outputs tree, which is fetched back into the local run directory: a pid file
# is plumbing and has no business appearing in a run's results.
REMOTE_RUN_PARENT = "{}run".format(OWNED_CONTAINER_PREFIX)
# Where a staged config goes for a container on *this* machine: beside the run's
# outputs, which is the one directory this process and the daemon are already
# relied on to name identically. Dotted so it does not look like a run.
LOCAL_CONFIG_DIRNAME = ".cadet-configs"
# A run id goes into a remote path and into the argument of a remote `rm -r`,
# so it is checked rather than trusted: ids are a short hash plus "-r<n>", and
# anything else is refused before a command is built from it.
_SAFE_RUN_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*\Z")
# How long the remote directory operations are given, beyond the connect budget.
REMOTE_OP_TIMEOUT_S = 120.0


class PlacementError(Exception):
    """A node could not be placed where its config asked."""


def _docker(argv: List[str], docker: str = "docker", check: bool = False):
    """Run a docker command; raise :class:`PlacementError` if asked and it fails."""
    try:
        result = subprocess.run([docker] + argv, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise PlacementError(
            "cannot run {!r}: {}. A node with launcher 'docker' needs a docker "
            "client, so the orchestrator has to run where docker does.".format(docker, exc)
        )
    if check and result.returncode != 0:
        raise PlacementError(
            "docker {} failed: {}".format(
                " ".join(argv[:2]), (result.stderr or result.stdout).strip()
            )
        )
    return result


# The documented way to produce the image a node defaults to, named in the
# refusal rather than left to the reader: the first thing a fresh clone gets
# wrong is running before building, and an error that says only "no such image"
# sends them looking for a registry that does not exist.
BUILD_COMMAND = (
    'docker build -f docker/Dockerfile -t {} --build-arg GIT_SHA="$(git rev-parse HEAD)" .'
)

# Where an image records the commit it was built from. The standard OCI label,
# so the answer is readable by anything and not only by CADET.
REVISION_LABEL = "org.opencontainers.image.revision"


# What ssh itself exits with when it could not run the command at all - a
# refused connection, an unusable key, a uid with no passwd entry. Docker's own
# "no such image" is 1, so the two are distinguishable and must be told apart.
SSH_FAILED_EXIT = 255


def image_present(image: str, machine=None, docker: str = "docker") -> bool:
    """True if ``image`` exists on the machine a container would run on.

    ``machine`` is a :class:`~cadet.core.staging.TargetMachine`; ``None`` means
    this one. Asked rather than assumed because the alternative is a `docker
    run` that tries to *pull* a locally built image, and the error it produces
    is about registry access - which is not the problem and does not name the
    build.

    A machine that could not be reached is not a machine without the image.
    Reporting it as one sends the reader to build or copy an image that is
    already there, so an unreachable machine raises and says so instead.
    """
    argv = ["image", "inspect", "--format", "{{.Id}}", image]
    if machine is not None:
        code, out = machine.run([docker] + argv, timeout_s=REMOTE_OP_TIMEOUT_S)
        if code == SSH_FAILED_EXIT:
            raise PlacementError(
                "could not reach {} to ask about its images: {}".format(
                    machine.describe(), (out or "").strip() or "ssh exited 255"
                )
            )
        return code == 0
    return _docker(argv, docker).returncode == 0


def require_images(images, machine=None, docker: str = "docker") -> None:
    """Refuse, naming every image that is missing and how to get it there.

    Checked for the whole placement before anything is created, so a fresh
    clone is told what to build instead of leaving a half-made run behind: the
    first image to be missing would otherwise fail after the run's network
    existed.
    """
    where = machine.describe() if machine is not None else "this machine"
    missing = sorted({i for i in images if i and not image_present(i, machine, docker)})
    if not missing:
        return
    lines = [
        "{} {} not on {}, and a locally built image cannot be pulled.".format(
            "images" if len(missing) > 1 else "image",
            ", ".join(repr(i) for i in missing),
            where,
        )
    ]
    for image in missing:
        lines.append("  build it with: {}".format(BUILD_COMMAND.format(image)))
    if machine is not None:
        lines.append(
            "  or copy one built here: docker save {} | ssh {} docker load".format(
                missing[0], where
            )
        )
    lines.append("A node names its image with nodes[*].container.")
    raise PlacementError("\n".join(lines))


def gpu_count(machine=None) -> Optional[int]:
    """How many GPUs the machine reports, or ``None`` if it cannot be asked.

    ``None`` and ``0`` are different answers and are kept apart: a machine with
    no NVIDIA tooling installed cannot say how many GPUs it has, and refusing a
    run on that basis would refuse every machine whose driver is reachable only
    from inside the container.
    """
    argv = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
    if machine is not None:
        code, out = machine.run(argv, timeout_s=REMOTE_OP_TIMEOUT_S)
    else:
        try:
            done = subprocess.run(argv, capture_output=True, text=True, check=False)
        except OSError:
            return None
        code, out = done.returncode, done.stdout
    if code != 0:
        return None
    return len([line for line in out.splitlines() if line.strip()])


def require_gpus(nodes, machine=None) -> None:
    """Refuse a node whose ``gpu_index`` names a device the machine lacks.

    Checked here because the error it replaces is unreadable: docker reports
    ``nvidia-container-cli: device error: 7: unknown device`` at the end of a
    paragraph of OCI runtime wrapping, which says nothing about which node asked
    or what the machine has.
    """
    asked = [n for n in nodes if getattr(n, "gpu_index", None) is not None]
    if not asked:
        return
    present = gpu_count(machine)
    if present is None:
        return
    where = machine.describe() if machine is not None else "this machine"
    over = [n for n in asked if n.gpu_index >= present or n.gpu_index < 0]
    if not over:
        return
    available = "0..{}".format(present - 1) if present else "none"
    raise PlacementError(
        "{} {} on {}, which reports {} GPU(s) ({}). A node's device is "
        "nodes[*].gpu_index; drop it for a node that needs no GPU.".format(
            "nodes" if len(over) > 1 else "node",
            ", ".join("{!r} asks for GPU {}".format(n.name, n.gpu_index) for n in over),
            where,
            present,
            available,
        )
    )


def image_revision(image: str, machine=None, docker: str = "docker") -> Optional[str]:
    """The commit an image records, or ``None`` if it records none."""
    argv = [
        "image",
        "inspect",
        "--format",
        "{{{{index .Config.Labels {!r}}}}}".format(REVISION_LABEL).replace("'", '"'),
        image,
    ]
    if machine is not None:
        code, out = machine.run([docker] + argv, timeout_s=REMOTE_OP_TIMEOUT_S)
    else:
        result = _docker(argv, docker)
        code, out = result.returncode, result.stdout
    if code != 0:
        return None
    revision = (out or "").strip()
    if not revision or revision in ("<no value>", "unknown", "none"):
        return None
    return revision


def image_identity(image: str, machine=None, docker: str = "docker") -> Dict[str, object]:
    """What an image *is*, for the record a run leaves behind.

    Returns the tag the node asked for, the content digest the daemon resolved
    it to, and the commit the image records, with the last two absent when the
    daemon cannot be asked. The tag alone is not provenance: a tag is rebuilt,
    so two runs naming ``cadet:dev`` need not have run the same code, and the
    digest is what tells them apart afterwards. The revision label says which
    commit that content was built from, which is the question a reader of a run
    directory actually has.
    """
    identity: Dict[str, object] = {"image": image}
    argv = ["image", "inspect", "--format", "{{.Id}}", image]
    if machine is not None:
        code, out = machine.run([docker] + argv, timeout_s=REMOTE_OP_TIMEOUT_S)
    else:
        result = _docker(argv, docker)
        code, out = result.returncode, result.stdout
    digest = (out or "").strip() if code == 0 else ""
    if digest:
        identity["id"] = digest
    revision = image_revision(image, machine, docker)
    if revision:
        identity["revision"] = revision
    return identity


def repo_revision(repo_root: Optional[str] = None):
    """Return ``(commit, dirty)`` for the repository, or ``(None, False)``.

    ``None`` for a tree that is not a git checkout, which an installed copy need
    not be. ``dirty`` matters as much as the commit: an image built from the
    current commit still lacks an uncommitted edit, and that is the case that is
    easiest to misread as a change not working.
    """
    root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        head = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if head.returncode != 0:
            return None, False
        status = subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None, False
    dirty = bool((status.stdout or "").strip()) if status.returncode == 0 else False
    return (head.stdout or "").strip() or None, dirty


def warn_stale_images(images, log, machine=None, docker: str = "docker") -> List[str]:
    """Warn, without refusing, when an image's code is not the repository's.

    A role in a container runs the image's copy of the code. An image built
    before a change does not have it, and the run that follows looks like the
    change not working rather than like a stale image - which is why this exists
    and why it names the rebuild rather than only the mismatch.

    A warning and not a refusal, deliberately: there are good reasons to run a
    deliberately older image, a reproduction against a recorded commit first
    among them, and the orchestrator has no business deciding that for an
    experiment. Returns the lines it logged, so a caller can record them.
    """
    head, dirty = repo_revision()
    if head is None:
        return []
    where = machine.describe() if machine is not None else "this machine"
    lines = []
    for image in sorted({i for i in images if i}):
        revision = image_revision(image, machine, docker)
        if revision is None:
            lines.append(
                "[place] WARNING: image {!r} on {} records no commit, so whether "
                "it carries this code cannot be told. Rebuild it with: {}".format(
                    image, where, BUILD_COMMAND.format(image)
                )
            )
        elif revision != head:
            lines.append(
                "[place] WARNING: image {!r} on {} was built from {}, and this "
                "repository is at {}. A role in a container runs the image's "
                "code, so this run will not include anything changed since. "
                "Rebuild it with: {}".format(
                    image, where, revision[:10], head[:10], BUILD_COMMAND.format(image)
                )
            )
        elif dirty:
            lines.append(
                "[place] WARNING: image {!r} on {} matches this commit, but the "
                "working tree has uncommitted changes, which the image cannot "
                "have. Rebuild it with: {}".format(
                    image, where, BUILD_COMMAND.format(image)
                )
            )
    for line in lines:
        log(line)
    return lines


# How a venv on another machine is made able to run a role, named in the
# refusal rather than left to the reader - the same reason BUILD_COMMAND is.
# There is no `pip install cadet` to offer: CADET is not published anywhere, so
# the honest instruction is a copy of this repository on that machine.
VENV_INSTALL_COMMAND = (
    "python{python} -m venv {venv} && {venv}/bin/pip install <this repository, "
    "copied to that machine>"
)
VENV_REINSTALL_COMMAND = (
    "{venv}/bin/pip install --force-reinstall <this repository, copied to that machine>"
)


def require_venv(machine, node):
    """Refuse a node whose venv cannot run a role; return what the venv is.

    Asked before anything is staged, for the same reason the images are: the
    alternative is a role that fails on ``ModuleNotFoundError: cadet`` after the
    outputs tree, the config and the weights have been put on that machine, and
    an error that names neither the venv nor the install.

    The Python version is part of the refusal and not a warning. This launcher
    kind promises a version - :func:`cadet.core.launcher.role_python_version`
    states it, and the simulator version check reasons from that statement
    rather than from the venv - so a venv that is not that version makes the
    check sound while its premise is false, which is the failure mode that
    statement exists to prevent.
    """
    if not node.venv:
        raise PlacementError(
            "node {!r} asks for the 'ssh_venv' launcher but names no "
            "virtualenv. A node names one with nodes[*].venv; it is a path on "
            "{}, where the role will run.".format(node.name, machine.describe())
        )
    wanted = role_python_version("ssh_venv")
    try:
        facts = venv_facts(machine, node.venv)
    except VenvError as exc:
        raise PlacementError(
            "node {!r} cannot run a role in {}: {}.\n"
            "  CADET is not shipped to a venv - it has to be installed in one:\n"
            "    {}\n"
            "A node names its virtualenv with nodes[*].venv.".format(
                node.name,
                node.venv,
                exc,
                VENV_INSTALL_COMMAND.format(python=wanted, venv=node.venv),
            )
        )
    if facts["python"] != wanted:
        raise PlacementError(
            "the venv {} on {} runs python {}, and a role placed with the "
            "'ssh_venv' launcher is taken to run python {} - that is what the "
            "simulator version check is told, and the CARLA client egg loads "
            "under nothing else. Recreate it with: {}".format(
                node.venv,
                machine.describe(),
                facts["python"],
                wanted,
                VENV_INSTALL_COMMAND.format(python=wanted, venv=node.venv),
            )
        )
    return facts


def warn_stale_venv(facts, log, machine, venv: str) -> List[str]:
    """Warn, without refusing, when a venv's installed CADET is not this code.

    The same hazard as a stale image and the same answer, because a venv has
    the same property an image does: the role runs the copy that is installed
    there, and an install made before a change does not have it - so the run
    that follows looks like the change not working rather than like a stale
    environment. This launcher ships no code, which is what makes the warning
    necessary rather than optional.

    A warning and not a refusal, for the reason :func:`warn_stale_images` gives:
    reproducing a recorded commit against a deliberately older environment is a
    legitimate thing to do, and the orchestrator has no business deciding that
    for an experiment. Returns the lines it logged, so a caller can record them.
    """
    head, dirty = repo_revision()
    if head is None:
        return []
    where = machine.describe()
    revision = facts.get("revision")
    reinstall = VENV_REINSTALL_COMMAND.format(venv=venv)
    lines = []
    if revision is None:
        lines.append(
            "[place] WARNING: the CADET installed in {} on {} records no "
            "commit, so whether it carries this code cannot be told. "
            "Reinstall it with: {}".format(venv, where, reinstall)
        )
    elif revision != head:
        lines.append(
            "[place] WARNING: the CADET installed in {} on {} is from {}, and "
            "this repository is at {}. A role in a venv runs that copy, so "
            "this run will not include anything changed since. Reinstall it "
            "with: {}".format(venv, where, revision[:10], head[:10], reinstall)
        )
    elif dirty:
        lines.append(
            "[place] WARNING: the CADET installed in {} on {} matches this "
            "commit, but this working tree has uncommitted changes, which that "
            "install cannot have. Reinstall it with: {}".format(
                venv, where, reinstall
            )
        )
    if facts.get("dirty"):
        lines.append(
            "[place] WARNING: the CADET installed in {} on {} has uncommitted "
            "changes of its own, so what it runs is not the commit it "
            "reports.".format(venv, where)
        )
    for line in lines:
        log(line)
    return lines


def docker_available(docker: str = "docker") -> bool:
    """True if a docker client here can reach a daemon."""
    try:
        result = subprocess.run(
            [docker, "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def node_network(run_id: str) -> str:
    """Return the run-scoped docker network name for the roles' own traffic."""
    return "{}{}-net".format(OWNED_CONTAINER_PREFIX, run_id)


def node_container_name(run_id: str, node_name: str) -> str:
    """Return the container name for one node of one run."""
    return "{}{}-{}".format(OWNED_CONTAINER_PREFIX, run_id, node_name)


class NodeNetwork:
    """A docker network that exists for exactly one run."""

    def __init__(self, run_id: str, docker: str = "docker"):
        self.name = node_network(run_id)
        self.docker = docker
        self._created = False

    def create(self) -> str:
        # A crashed run leaves this name behind and docker refuses to create it
        # twice, so the config would stay unrunnable until someone intervened.
        # Removing our own name first is the same rule the pre-run orphan
        # cleanup follows, and it is scoped to exactly the name this run would
        # create.
        _docker(["network", "rm", self.name], self.docker)
        _docker(["network", "create", self.name], self.docker, check=True)
        self._created = True
        return self.name

    def remove(self) -> bool:
        """Remove the network. Returns whether it is gone."""
        if not self._created:
            return True
        result = _docker(["network", "rm", self.name], self.docker)
        self._created = False
        return result.returncode == 0


class NodeContainer:
    """One node's container: a place for its role to run in."""

    def __init__(self, node_name: str, spec: ContainerSpec, docker: str = "docker"):
        self.node_name = node_name
        self.spec = spec
        self.docker = docker
        self._started = False

    @property
    def name(self) -> str:
        return self.spec.name or node_container_name("unknown", self.node_name)

    def start(self) -> str:
        """Create the container idle, so it exists before anything runs in it."""
        # Same reason as the network: a crashed run leaves this exact name.
        _docker(["rm", "-f", self.name], self.docker)
        argv = ["run", "-d", "--entrypoint", IDLE_ENTRYPOINT]
        argv += self.spec.docker_args()
        argv.append(self.spec.image)
        argv += IDLE_ARGS
        _docker(argv, self.docker, check=True)
        self._started = True
        return self.name

    def ip_address(self, network: str) -> Optional[str]:
        """Return the address docker gave this container on ``network``.

        The network is looked up with ``index`` and not by field access: a Go
        template cannot reach a map key that contains a hyphen, and every
        run-scoped network name has several, so the dotted form fails to parse
        and yields nothing at all.
        """
        template = '{{{{(index .NetworkSettings.Networks "{}").IPAddress}}}}'.format(network)
        deadline = time.monotonic() + PID_TIMEOUT_S
        while time.monotonic() < deadline:
            result = _docker(["inspect", "-f", template, self.name], self.docker)
            address = result.stdout.strip()
            if result.returncode == 0 and address and address != "<no value>":
                return address
            time.sleep(_POLL_S)
        return None

    def pid(self) -> Optional[int]:
        """Return the container's main pid: a handle on its network namespace."""
        deadline = time.monotonic() + PID_TIMEOUT_S
        while time.monotonic() < deadline:
            result = _docker(["inspect", "-f", "{{.State.Pid}}", self.name], self.docker)
            if result.returncode == 0:
                try:
                    pid = int(result.stdout.strip())
                except ValueError:
                    pid = 0
                if pid:
                    return pid
            time.sleep(_POLL_S)
        return None

    def runner(self) -> Callable[[List[str]], Tuple[int, str]]:
        """Return a runner that executes a command inside this container.

        Used for the ``netem`` qdisc, which has to be installed in the node's
        own namespace. It runs as root explicitly: the image drops to an
        unprivileged user, and ``tc`` needs root even with NET_ADMIN.
        """

        def _run(argv: List[str]) -> Tuple[int, str]:
            result = _docker(
                ["exec", "-u", "root", self.name] + list(argv), self.docker
            )
            return result.returncode, (result.stderr or result.stdout).strip()

        return _run

    def remove(self) -> bool:
        """Remove the container. Returns whether it is actually gone.

        No graceful stop first, deliberately. By the time a placement is torn
        down the roles inside have already been stopped by their own launchers,
        so the container is an idle ``sleep`` holding nothing - and a process
        that is PID 1 in a container gets no default signal handling, so
        ``sleep`` ignores SIGTERM and a grace period is spent purely waiting for
        the kill that follows it. Measured: two containers cost 20.6 s of
        teardown that way.

        The result is returned rather than ignored: a removal that fails leaves
        an endpoint on the run's network, which stops the next run of the same
        config from creating it, and a teardown that reports success while that
        is true is worse than one that fails.
        """
        if not self._started:
            return True
        result = _docker(["rm", "-f", self.name], self.docker)
        self._started = False
        return result.returncode == 0


class RemoteOutputs:
    """A per-run outputs tree on the machine a remote node's role runs on.

    A bind mount names a path in the filesystem of the machine the container
    runs on, so for a node on another machine the orchestrator's own output
    directory is meaningless there: docker would create it, owned by root, and
    the results would stay on that machine. This creates a staging tree under
    the remote account's own home instead, mounts that, copies it back into the
    local outputs root when the run is over, and removes it - so a remote run
    leaves nothing behind and its results land where every other run's do.

    It serves a venv node unchanged, and the reason it needs no change is that
    the problem was never the mount: the orchestrator's output path names
    nothing on that machine either way. A venv role is told this tree's own
    path instead of a mount point, and everything after that - the fetch, the
    removal, the refusal to remove a tree it could not fetch - is the same.

    The tree is the outputs **root** and not the run directory, for the same
    reason the local launcher mounts the root: the role resolves
    ``outputs.root`` itself and appends the same naming template the
    orchestrator does, so mounting the run directory would nest a second copy of
    that path inside it.
    """

    def __init__(self, launcher: SshTarget, run_id: str, log=print):
        if not _SAFE_RUN_ID.match(run_id):
            raise PlacementError(
                "run id {!r} is not a safe path component, and it would be "
                "part of a path created and removed on another "
                "machine".format(run_id)
            )
        self.launcher = launcher
        self.run_id = run_id
        self.log = log
        self.path: Optional[str] = None
        self.parent: Optional[str] = None

    def prepare(self) -> str:
        """Create the staging tree remotely and return its absolute path."""
        facts = self.launcher.remote_facts()
        self.parent = "{}/{}".format(facts["home"].rstrip("/"), REMOTE_OUTPUTS_PARENT)
        self.path = "{}/{}".format(self.parent, self.run_id)
        code, output = self.launcher.run_remote(
            ["mkdir", "-p", self.path], timeout_s=REMOTE_OP_TIMEOUT_S
        )
        if code != 0:
            raise RemoteError(
                "cannot create the remote outputs directory {}: {}".format(
                    self.path, output.strip() or "rc=" + str(code)
                )
            )
        return self.path

    def fetch(self, local_root: str) -> bool:
        """Copy the staging tree into ``local_root``. Returns whether it worked.

        Streamed through ``tar`` rather than ``scp -r``: one round trip, and the
        tree merges into the local outputs root instead of landing inside a
        directory named after itself.
        """
        if self.path is None:
            return True
        os.makedirs(local_root, exist_ok=True)
        remote = self.launcher.remote_command(["tar", "-cf", "-", "-C", self.path, "."])
        try:
            sender = subprocess.Popen(
                remote, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            receiver = subprocess.Popen(
                ["tar", "-xf", "-", "-C", local_root],
                stdin=sender.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Closed here so the receiver owns the read end and the sender sees
            # a broken pipe if the receiver dies, rather than both hanging.
            if sender.stdout is not None:
                sender.stdout.close()
            _out, recv_err = receiver.communicate(timeout=REMOTE_OP_TIMEOUT_S)
            _sent, send_err = sender.communicate(timeout=REMOTE_OP_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.log("[place] WARNING: fetching remote outputs failed: {}".format(exc))
            return False
        if sender.returncode != 0 or receiver.returncode != 0:
            self.log(
                "[place] WARNING: fetching remote outputs failed (send rc={}, "
                "receive rc={}): {}".format(
                    sender.returncode,
                    receiver.returncode,
                    (send_err or recv_err or b"").decode("utf-8", "replace").strip(),
                )
            )
            return False
        return True

    def remove(self) -> bool:
        """Remove the staging tree, and its parent if no other run is using it."""
        if self.path is None:
            return True
        code, output = self.launcher.run_remote(
            ["rm", "-rf", "--", self.path], timeout_s=REMOTE_OP_TIMEOUT_S
        )
        if code != 0:
            self.log(
                "[place] WARNING: remote outputs directory {} was not removed: "
                "{}".format(self.path, output.strip() or "rc=" + str(code))
            )
            return False
        self.path = None
        # Non-recursive on purpose: it succeeds only when this was the last run's
        # tree, so a concurrent run's staging directory is never taken with it.
        if self.parent:
            self.launcher.run_remote(["rmdir", "--", self.parent], timeout_s=REMOTE_OP_TIMEOUT_S)
        return True


class StagedRunConfig:
    """The run's resolved config, placed where a containerised role can read it.

    The orchestrator is given a config *path*, and passes it to every role so
    each one loads the same experiment. A role in a container resolves that path
    against the image, whose copy of the repository is whatever was baked into
    it, so a config written since - or edited since, or never committed - is
    simply absent. The failure is a role that exits on ``FileNotFoundError``
    before it has done anything, which reads like a broken launcher and is not
    one.

    It applies to a container on *this* machine as much as to one on another,
    which is not obvious: the working tree is right there, and the container
    still cannot see it, because nothing mounts it. That is why this is not a
    remote-only concern and why one class serves both - the only difference is
    which machine the file is written to, which is the
    :class:`~cadet.core.staging.TargetMachine`'s business and not this class's.

    So the config is **sent** rather than named. What is sent is the resolved
    config - the one the orchestrator itself is running, after ``extends``,
    after the ``--set`` overrides and after the sweep has been expanded to this
    one run - which closes three problems at once: ``extends`` has no relative
    path left to resolve, the sweep has no axes left to index, and the file is
    self-contained, so nothing about the other machine's checkout matters.

    The run id is unaffected, which is the property that makes this safe:
    :func:`cadet.config.loader.config_hash` digests the resolved config, and
    this *is* that config, so the role derives the id the orchestrator did.
    """

    def __init__(self, machine: TargetMachine, run_id: str, log=print):
        if not _SAFE_RUN_ID.match(run_id):
            raise PlacementError(
                "run id {!r} is not a safe path component, and it would be "
                "part of a path created and removed on another "
                "machine".format(run_id)
            )
        self.machine = machine
        self.run_id = run_id
        self.log = log
        self.path: Optional[str] = None
        self.parent: Optional[str] = None

    def send(self, config, base_dir: Optional[str] = None) -> str:
        """Write the resolved config where the container will read it; return its path.

        ``base_dir`` is a directory that **this process and the machine running
        the container both name the same way**, and it has to be given for a
        container on this machine. The home directory is not that for a local
        container, which is the bug this parameter exists for: a bind mount's
        source is resolved by the daemon on the *host*, and when the
        orchestrator is itself containerised - which is how `scripts/cadet`
        runs, the documented primary way - its home is a path inside its own
        container. Docker then found nothing at that path on the host and
        created a **directory** there, and the role failed with
        ``IsADirectoryError: '/cadet/run-config.yaml'``.

        The outputs root is such a directory, and is already relied on to be one
        for the same reason: a containerised run's outputs mount names it on both
        sides. So a local container stages beside the outputs, and a remote one
        stages under the remote account's home, which is host-visible there
        because nothing on that machine is containerised.
        """
        if base_dir:
            self.parent = os.path.join(os.path.abspath(base_dir), LOCAL_CONFIG_DIRNAME)
        else:
            self.parent = "{}/{}".format(
                self.machine.home().rstrip("/"), REMOTE_CONFIG_PARENT
            )
        self.path = "{}/{}.yaml".format(self.parent, self.run_id)
        code, output = self.machine.run(
            ["mkdir", "-p", self.parent], timeout_s=REMOTE_OP_TIMEOUT_S
        )
        if code != 0:
            raise RemoteError(
                "cannot create the remote config directory {}: {}".format(
                    self.parent, output.strip() or "rc=" + str(code)
                )
            )
        local = os.path.join(tempfile.mkdtemp(prefix="cadet-config-"), "run-config.yaml")
        try:
            with open(local, "w") as fh:
                yaml.safe_dump(
                    config.model_dump(mode="json", by_alias=True), fh, sort_keys=False
                )
            self.machine.send(local, self.path)
        except (OSError, StagingError) as exc:
            raise RemoteError(
                "cannot put this run's config on {}: {}".format(
                    self.machine.describe(), exc
                )
            )
        finally:
            shutil.rmtree(os.path.dirname(local), ignore_errors=True)
        return self.path

    def remove(self) -> bool:
        """Remove the staged config, and its directory if nothing else is in it."""
        if self.path is None:
            return True
        code, output = self.machine.run(
            ["rm", "-f", "--", self.path], timeout_s=REMOTE_OP_TIMEOUT_S
        )
        if code != 0:
            self.log(
                "[place] WARNING: the staged config {} was not removed: "
                "{}".format(self.path, output.strip() or "rc=" + str(code))
            )
            return False
        self.path = None
        # Non-recursive, so a config staged for a concurrent run is never taken
        # with it - the same reason the outputs parent is removed this way.
        if self.parent:
            self.machine.run(["rmdir", "--", self.parent], timeout_s=REMOTE_OP_TIMEOUT_S)
        return True


class ExecLauncher(DockerLauncher):
    """Starts a role inside a node container that already exists.

    Subclasses :class:`DockerLauncher` to reuse its readiness barrier, its
    status mapping and the argv rewriting that strips the host interpreter
    prefix; only *where* the command runs differs, which is the one thing a
    launcher is for.
    """

    def __init__(self, container: NodeContainer, docker: str = "docker"):
        super().__init__(specs={container.node_name: container.spec}, docker=docker)
        self.container = container

    @staticmethod
    def _node_argument(spec: LaunchSpec) -> Optional[str]:
        """The node a role's command line names, which identifies it in the container."""
        argv = list(getattr(spec, "argv", ()) or ())
        if "--node" in argv:
            index = argv.index("--node")
            if index + 1 < len(argv):
                return argv[index + 1]
        return None

    def command(self, spec: LaunchSpec) -> List[str]:
        """Return the ``docker exec`` that runs this role inside the container."""
        argv = self._role_argv(spec)
        exec_argv = [self.docker, "exec"]
        for key, value in sorted(container_env(spec).items()):
            exec_argv += ["-e", "{}={}".format(key, value)]
        exec_argv.append(self.container.name)
        # `docker run` takes `cadet` from the image's entry point; `docker exec`
        # does not, so the console script is named here. Without it the exec
        # would look for a binary called `serve`.
        return exec_argv + [ROLE_COMMAND] + argv

    def stop(self, handle: Handle, grace_s: float = 5.0) -> None:
        """Signal the role inside the container, then reap the exec client.

        The base class removes the container, which is wrong here: the container
        is not this launcher's to own, and several roles may share one.

        Terminating the ``docker exec`` client is not enough either, and that is
        the part that was wrong. The client is a local process; the role is a
        process in the container, and killing the client does not signal it. The
        role then runs on until the placement removes the container, which kills
        it outright, and a role killed that way never runs its shutdown path:
        measured on a containerised acceptance cell, the server's per-request
        metrics CSV - which the same app writes when placed locally - was
        absent. So the signal is delivered where the role actually is, by
        matching the ``--node`` argument that makes a role's command line its
        own, and only then is the client reaped.
        """
        proc = handle.proc
        if proc.poll() is None:
            node = self._node_argument(handle.spec)
            if node is not None:
                # The leading `[-]` keeps the pattern from matching the pkill
                # command line itself, which carries the same text.
                subprocess.run(
                    [
                        self.docker,
                        "exec",
                        self.container.name,
                        "pkill",
                        "-TERM",
                        "-f",
                        "[-]-node {}".format(node),
                    ],
                    capture_output=True,
                    text=True,
                )
                try:
                    proc.wait(timeout=grace_s)
                except (subprocess.TimeoutExpired, OSError):
                    pass
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=grace_s)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    proc.kill()
                except OSError:
                    pass
        if handle.log_fh is not None:
            try:
                handle.log_fh.close()
            except OSError:
                pass


def shaping_nodes(config) -> List[str]:
    """Return the nodes that will install a ``netem`` qdisc of their own.

    A qdisc belongs on the *sender's* interface, so a forward link shapes the
    node it comes from and a reverse (or bidirectional) one also shapes the node
    it goes to. Shared with the preflight check that refuses shaping on a node
    whose namespace the orchestrator cannot reach, so the two cannot disagree
    about which nodes shape.
    """
    shaped = set()
    for link in config.links:
        if link.network.mode == "netem":
            shaped.add(link.from_)
            if link.direction in ("reverse", "both"):
                shaped.add(link.to)
    return sorted(shaped)


def address_overrides(config, placed: Dict[str, str]) -> List[str]:
    """Return ``--set`` strings putting each placed node on a reachable address.

    A containerised node is reached by its container name, which the run's own
    docker network resolves. Nodes that stay local keep the address the config
    gave them.

    ``outputs.root`` is **not** here: see :func:`role_overrides`. It is the one
    setting whose correct value differs between the host and the container, so
    it cannot be part of a rewrite both sides share.
    """
    index_of = {node.name: i for i, node in enumerate(config.nodes)}
    return [
        "nodes.{}.host={}".format(index_of[name], address)
        for name, address in sorted(placed.items())
        if name in index_of
    ]


def role_overrides(outputs_dir: str = CONTAINER_OUTPUTS) -> List[str]:
    """Return the ``--set`` strings that apply only to a placed role.

    ``outputs.root`` is read on both sides of the bind mount and means something
    different on each: the orchestrator resolves it on the host to find a run's
    directory, while a role resolves it inside its container to write there.
    Overriding it for both made the orchestrator try to create the container's
    path on the host.

    ``outputs_dir`` is that role's view of the outputs root, and it defaults to
    the container mount because that is what every containerised role sees. A
    venv role has no mount, so its view is the staging tree's own path on its
    own machine - the same override with a different value, rather than a second
    mechanism.

    Splitting the views is safe precisely because ``config_hash`` excludes
    ``outputs`` - where results are written is deliberately not part of a run's
    identity - so the role still derives the same run id as the orchestrator and
    writes into the directory the orchestrator is reading.
    """
    return ["outputs.root={}".format(outputs_dir)]


class Placement:
    """Where every node of one run lives, and how to start a role there."""

    def __init__(
        self,
        config,
        run_id: str,
        out_dir: str,
        log=print,
        docker: str = "docker",
        override=None,
        model_dirs=None,
    ):
        self.config = config
        self.run_id = run_id
        self.out_dir = out_dir
        self.log = log
        self.docker = docker
        # An injected launcher used for every node, which is what a test harness
        # wants. When set, nothing is placed: the override decides where roles
        # run, so creating containers for them would be creating containers
        # nothing uses.
        self.override = override
        self.network: Optional[NodeNetwork] = None
        self.containers: Dict[str, NodeContainer] = {}
        # node name -> its staging outputs tree on the other machine.
        self.remote_outputs: Dict[str, RemoteOutputs] = {}
        # node name -> that node's staged config, on whichever machine its
        # role runs on.
        self.staged_configs: Dict[str, StagedRunConfig] = {}
        # node name -> the outputs staging tree a *venv* node's role writes to.
        # Kept apart from the containerised nodes because the role is told this
        # path itself, where a containerised one is told its mount point.
        self.venv_launchers: Dict[str, SshVenvLauncher] = {}
        self.venv_outputs: Dict[str, str] = {}
        # node name -> what that node's venv turned out to be, for the record.
        self.venv_facts: Dict[str, Dict[str, object]] = {}
        # node name -> the image its container ran, recorded when the container
        # is created rather than asked for afterwards: by then the tag may name
        # different content, and the run ran what it ran.
        self.image_facts: Dict[str, Dict[str, object]] = {}
        # Where the orchestrator found the model artifacts, so they can be put
        # on the machine each containerised role runs on.
        self.model_dirs = list(model_dirs or [])
        # node name -> that node's per-run model directory on its own machine.
        self.model_staging: Dict[str, ModelStaging] = {}
        self._launchers: Dict[str, object] = {}
        self._local = override or LocalLauncher()

    # -- lifecycle -----------------------------------------------------------
    def nodes_needing_containers(self) -> List[object]:
        if self.override is not None:
            return []
        return [n for n in self.config.nodes if n.launcher == "docker"]

    def remote_nodes(self) -> List[object]:
        """Nodes whose container runs on another machine, reached over SSH."""
        if self.override is not None:
            return []
        return [n for n in self.config.nodes if n.launcher == "ssh_docker"]

    def venv_nodes(self) -> List[object]:
        """Nodes whose role runs in a virtualenv on another machine, over SSH."""
        if self.override is not None:
            return []
        return [n for n in self.config.nodes if n.launcher == "ssh_venv"]

    def offmachine_nodes(self) -> List[object]:
        """Every node that runs on another machine, however it is placed there.

        What the clock measurement asks for: a node on another machine keeps
        that machine's clock whether its role is in a container or in a venv,
        and the probe goes down the same SSH channel either way.
        """
        return self.remote_nodes() + self.venv_nodes()

    def netem_nodes(self) -> List[str]:
        """Nodes that will install a qdisc, so their container gets NET_ADMIN."""
        return shaping_nodes(self.config)

    def start(self) -> Dict[str, str]:
        """Create the network and the node containers. Returns node -> address.

        A remote node is prepared here too, but it is deliberately **not** in
        the returned map. The map drives the address rewrite, and a node on
        another machine is already configured with that machine's address: its
        peers reach it there, through a port the container publishes. There is
        nothing to rewrite, and rewriting it would change the run id.
        """
        needed = self.nodes_needing_containers()
        remote = self.remote_nodes()
        venvs = self.venv_nodes()
        if not needed and not remote and not venvs:
            return {}
        for node in remote:
            self._prepare_remote(node)
        for node in venvs:
            self._prepare_venv(node)
        if not needed:
            return {}
        if not docker_available(self.docker):
            raise PlacementError(
                "a node asks for the 'docker' launcher but no docker daemon is "
                "reachable here. The orchestrator places containers, so it has "
                "to run where docker does - on the host."
            )
        require_images(
            [node.container or default_image() for node in needed], docker=self.docker
        )
        require_gpus(needed)
        warn_stale_images(
            [node.container or default_image() for node in needed],
            self.log,
            docker=self.docker,
        )
        shaped = set(self.netem_nodes())
        self.network = NodeNetwork(self.run_id, self.docker)
        self.network.create()
        self.log("[place] network {}".format(self.network.name))

        placed: Dict[str, str] = {}
        machine = TargetMachine(log=self.log)
        for node in needed:
            spec = container_spec_for(
                node,
                self.run_id,
                # The outputs ROOT, not this run's directory: the role resolves
                # outputs.root itself and appends the same naming template the
                # orchestrator does, so mounting the run directory would nest a
                # second copy of that path inside it.
                outputs_dir=os.path.abspath(self.config.outputs.root),
                needs_net_admin=node.name in shaped,
                models_dir=self._stage_models(node, machine),
                config_file=self._stage_config(node, machine),
            )
            spec.network = self.network.name
            self.image_facts[node.name] = image_identity(spec.image, docker=self.docker)
            container = NodeContainer(node.name, spec, self.docker)
            container.start()
            self.containers[node.name] = container
            address = container.ip_address(self.network.name)
            if address is None:
                raise PlacementError(
                    "container {} came up but docker reported no address for it "
                    "on {}; the roles would have nothing to bind or connect "
                    "to".format(container.name, self.network.name)
                )
            placed[node.name] = address
            self.log(
                "[place] {} -> container {} at {} (image {}{})".format(
                    node.name,
                    container.name,
                    address,
                    spec.image,
                    ", NET_ADMIN" if spec.needs_net_admin else "",
                )
            )
        return placed

    def _prepare_remote(self, node) -> None:
        """Build the launcher and the staging tree for a node on another machine.

        Three things differ from a container on this machine, and each of them
        is a property of the *other* machine rather than this one:

          * the container joins the other machine's own network namespace, so
            the role is reached at the address the config already names - the
            same one the SSH goes to - and is told to bind with
            :data:`CONTAINER_BIND_HOST` (see :meth:`role_args` for why that is
            argv and not a config override). Publishing the port instead was
            tried first and does not travel: on the machine this was built
            against, a published port was unreachable from here while a host
            process on a port in the same range was reachable, because reaching
            a published port depends on that machine's forwarding policy and
            nothing in a config can promise it. Joining its namespace depends on
            nothing, and the hazard it would otherwise add - a netem qdisc
            landing on that machine's real interface - is already refused in
            preflight for exactly these nodes;
          * the outputs mount is a staging tree in the remote account's home,
            fetched back and removed in :meth:`stop`;
          * the config is sent rather than named, because the path the
            orchestrator was given does not exist on that machine - though that
            is not remote-specific, see :class:`StagedRunConfig`;
          * the container runs as the *remote* account's uid:gid, because this
            process's uid need not exist there and would not own the mount.
        """
        launcher = SshDockerLauncher(
            host=node.host,
            user=node.user,
            port=node.ssh_port,
            identity=node.ssh_identity,
        )
        machine = TargetMachine(launcher, log=self.log)
        require_images([node.container or default_image()], machine=machine)
        require_gpus([node], machine=machine)
        warn_stale_images([node.container or default_image()], self.log, machine=machine)
        outputs = RemoteOutputs(launcher, self.run_id, log=self.log)
        remote_dir = outputs.prepare()
        facts = launcher.remote_facts()
        config_file = self._stage_config(node, machine)
        spec = container_spec_for(
            node,
            self.run_id,
            outputs_dir=remote_dir,
            run_as="{}:{}".format(facts["uid"], facts["gid"]),
            models_dir=self._stage_models(node, machine),
            config_file=config_file,
        )
        spec.network = HOST_NETWORK
        self.image_facts[node.name] = image_identity(
            spec.image, machine=machine, docker=self.docker
        )
        launcher.specs = {node.name: spec}
        self.remote_outputs[node.name] = outputs
        self._launchers[node.name] = launcher
        self.log(
            "[place] {} -> remote container {} (image {}, network {}{}), "
            "outputs staged remotely".format(
                node.name,
                spec.name,
                spec.image,
                spec.network,
                ", port {}".format(node.port) if node.port is not None else "",
            )
        )

    def _prepare_venv(self, node) -> None:
        """Build the launcher and the staging for a node in a venv elsewhere.

        The same four problems the remote container case has, answered the same
        way and by the same classes - the outputs tree staged in that account's
        home and fetched back, the config sent rather than named, the model
        artifacts shipped to the machine that will read them - plus the one that
        is this launcher's own: **the code**. A container carried it in its
        image; a venv has only what is installed in it, and this launcher ships
        nothing. So the venv is asked what it is before anything else is done
        with it, which is both the refusal (no venv, no CADET in it, or the
        wrong Python) and the staleness warning that an image gets from its
        recorded commit.

        What differs from the container case, and why:

          * **nothing is mounted**, so the role is told the staged paths
            themselves: the outputs root through a ``--set`` of its own, the
            config by its path on that machine, the weights with a
            ``--model-dir`` naming that machine's staging directory;
          * **no uid is needed.** A container has to be told which account to
            run as because the image has a user of its own; a role started over
            SSH is already that account, and the files it writes already belong
            to it;
          * **the GPU is this machine's numbering as the config states it.** A
            container's devices are filtered by docker and renumbered, which is
            why the container launchers set ``CUDA_VISIBLE_DEVICES``; nothing
            filters a plain process, so the host index the config names is the
            index torch sees (see
            :func:`cadet.core.model_registry.visible_gpu_index`) and the
            variable is deliberately left unset. ``gpu_index`` is still checked
            against what the machine reports.
        """
        launcher = SshVenvLauncher(
            host=node.host,
            venv=node.venv or "",
            user=node.user,
            port=node.ssh_port,
            identity=node.ssh_identity,
        )
        machine = TargetMachine(launcher, log=self.log)
        facts = require_venv(machine, node)
        warn_stale_venv(facts, self.log, machine=machine, venv=node.venv)
        require_gpus([node], machine=machine)
        # The pid-file directory, which the launcher needs before it starts
        # anything and which depends on an answer this machine has now given.
        launcher.run_dir = "{}/{}/{}".format(
            machine.home().rstrip("/"), REMOTE_RUN_PARENT, self.run_id
        )
        # The directory the role runs in, so an --app-module naming a module in
        # that checkout can be found; the probe has already reported it.
        launcher.install_root = facts.get("root") or ""
        outputs = RemoteOutputs(launcher, self.run_id, log=self.log)
        remote_dir = outputs.prepare()
        self._stage_config(node, machine)
        self._stage_models(node, machine)
        self.remote_outputs[node.name] = outputs
        self.venv_outputs[node.name] = remote_dir
        self.venv_launchers[node.name] = launcher
        self.venv_facts[node.name] = facts
        self._launchers[node.name] = launcher
        self.log(
            "[place] {} -> remote venv {} on {} ({}{}), outputs staged "
            "remotely".format(
                node.name,
                node.venv,
                machine.describe(),
                describe_venv(facts),
                ", port {}".format(node.port) if node.port is not None else "",
            )
        )

    def _stage_config(self, node, machine: TargetMachine) -> str:
        """Put this run's config on the machine ``node``'s container runs on.

        Called for every containerised node, local or remote, for the reason in
        :class:`StagedRunConfig`: a container on this machine cannot see this
        machine's working tree either.

        A local container stages beside the outputs rather than under the home
        directory - see :meth:`StagedRunConfig.send` for why the home directory
        is the wrong answer when the orchestrator is itself in a container.
        """
        staged = StagedRunConfig(machine, self.run_id, log=self.log)
        base = None if machine.is_remote else os.path.abspath(self.config.outputs.root)
        path = staged.send(self.config, base_dir=base)
        self.staged_configs[node.name] = staged
        return path

    def _stage_models(self, node, machine: TargetMachine) -> Optional[str]:
        """Put this node's model artifacts on its machine; return the mount source.

        Called for every containerised node, local or remote. The local case is
        not a special case that could be skipped: a container on this machine
        cannot see this machine's search directories either, which is why an app
        that loads a model from the registry had never run under the ``docker``
        launcher.
        """
        artifacts = artifacts_for(self.config, self.model_dirs)
        if not artifacts:
            return None
        staging = ModelStaging(machine, self.run_id, log=self.log)
        try:
            staged = staging.stage(artifacts)
        except StagingError as exc:
            raise PlacementError(str(exc))
        if staged is not None:
            self.model_staging[node.name] = staging
        return staged

    def role_overrides_for(self, node) -> List[str]:
        """Return the ``--set`` strings for *this node's* role process.

        Per node and not per run: ``outputs.root`` is only redirected for a role
        that runs in a container, and a run can mix placements - a client here
        with a server on another machine is the whole point of ``ssh_docker``.
        Telling that local client to write into the container's mount path would
        send its results to a directory that does not exist on this machine.
        """
        if node.name in self.venv_outputs:
            # No mount: the role writes into the staging tree by its own path.
            return role_overrides(self.venv_outputs[node.name])
        if node.name in self.containers or node.name in self.remote_outputs:
            return role_overrides()
        return []

    def role_config_path(self, node) -> Optional[str]:
        """Return the config path *this node's* role should load, if not ours.

        ``None`` means "the path the orchestrator was given", which is right for
        a role that runs as a process on this machine. A role in a container
        gets the staged copy's mount path instead - local or remote alike, since
        neither can see this machine's working tree; because that copy is the
        resolved single-run config, the role that loads it is also told
        ``--run 0``, there being no sweep left in it to index.

        A role in a venv is handed the staged copy's **own** path on its
        machine, nothing being mounted for it: the same file, named the way that
        role can open it.
        """
        staged = self.staged_configs.get(node.name)
        if staged is None:
            return None
        if node.name in self.venv_outputs:
            return staged.path
        return CONTAINER_RUN_CONFIG

    def role_args(self, node) -> List[str]:
        """Return extra argv for this node's role process, beyond the config.

        The bind address and the staged model directory, both argv rather than a
        ``--set`` on purpose: ``nodes[*].host`` is hashed into the run id, so
        overriding it for the role alone would make the role derive a different
        run id from the orchestrator and write its records somewhere else. Where
        a process listens, and where it looks for weights, are placement
        decisions and not part of the experiment's identity.
        """
        args: List[str] = []
        if node.role not in ("server", "sim"):
            # Only these take placement argv: `cadet client` and `cadet load`
            # accept neither --bind-host nor --model-dir, because neither
            # listens and neither loads a model from the registry today. A
            # containerised client that needed weights would need that flag
            # first; nothing asks for it yet, and passing one it does not
            # declare would fail the run on argument parsing.
            return args
        if node.name in self.remote_outputs:
            # Both remote placements bind this: the address a node is configured
            # with is how its peers and the SSH reach that machine, and need not
            # be an address the role can bind there verbatim.
            args += ["--bind-host", CONTAINER_BIND_HOST]
        if node.name in self.model_staging:
            if node.name in self.venv_outputs:
                # Nothing is mounted, so the directory the role is told to look
                # in is the staging directory itself.
                args += ["--model-dir", self.model_staging[node.name].path]
            else:
                args += staged_model_args()
        return args

    def launcher_for(self, node):
        """Return the launcher this node's config asks for."""
        if self.override is not None:
            return self.override
        if node.name in self._launchers:
            return self._launchers[node.name]
        if node.launcher == "local":
            launcher = self._local
        elif node.launcher == "docker":
            container = self.containers.get(node.name)
            if container is None:
                raise PlacementError(
                    "node {!r} asks for the 'docker' launcher but no container "
                    "was placed for it".format(node.name)
                )
            launcher = ExecLauncher(container, self.docker)
        elif node.launcher == "ssh_docker":
            # Built in _prepare_remote, which had to ask the other machine for
            # its home and uid first. Absent means the placement was never
            # started, exactly as for a local container.
            raise PlacementError(
                "node {!r} asks for the 'ssh_docker' launcher but no remote "
                "container was prepared for it".format(node.name)
            )
        elif node.launcher == "ssh_venv":
            # Built in _prepare_venv, for the same reason: it had to ask that
            # machine where its home is and what its venv contains first.
            raise PlacementError(
                "node {!r} asks for the 'ssh_venv' launcher but no remote venv "
                "was prepared for it".format(node.name)
            )
        else:
            raise PlacementError(
                "node {!r} asks for the {!r} launcher, which has no "
                "implementation".format(node.name, node.launcher)
            )
        self._launchers[node.name] = launcher
        return launcher

    def address_overrides_for(self, placed: Dict[str, str]) -> List[str]:
        """Return the ``--set`` strings for the addresses this placement assigned."""
        return address_overrides(self.config, placed)

    def runner_for(self, node_name: str):
        """Return the command runner for a node's own namespace, or None."""
        container = self.containers.get(node_name)
        return container.runner() if container is not None else None

    def stop(self) -> List[str]:
        """Remove the containers and then the network. Returns what was removed.

        Containers first and the network second, in that order and not the
        other: docker refuses to remove a network that still has an attached
        endpoint, so a container left behind takes the network with it.
        """
        removed = []
        # The per-run model directories, on whichever machine each one is. The
        # per-machine cache underneath is kept on purpose: re-shipping a model
        # on every run of a sweep would cost more than the run.
        for name, staging in sorted(self.model_staging.items()):
            if staging.remove():
                removed.append("models:{}".format(name))
        self.model_staging.clear()
        # The remote side first: the results have to be fetched before the
        # staging tree is removed, and both are independent of the local
        # containers. A fetch that fails leaves the tree in place and says so,
        # because losing a run's output silently is worse than a leftover.
        for name, outputs in sorted(self.remote_outputs.items()):
            fetched = outputs.fetch(os.path.abspath(self.config.outputs.root))
            if fetched and outputs.remove():
                removed.append("remote-outputs:{}".format(name))
            elif not fetched:
                self.log(
                    "[place] WARNING: outputs for remote node {} were not "
                    "fetched; the staging tree is left on that machine so the "
                    "results are not lost".format(name)
                )
        self.remote_outputs.clear()
        # The pid-file directory of a venv node, after its role's own launcher
        # has removed the pid file inside it. Non-recursive, so it is removed
        # only once this run has nothing left in it; a role still running leaves
        # it, which is the report that matters.
        for name, launcher in sorted(self.venv_launchers.items()):
            code, _out = launcher.run_remote(
                ["rmdir", "--", launcher.run_dir], timeout_s=REMOTE_OP_TIMEOUT_S
            )
            if code == 0:
                removed.append("venv-run-dir:{}".format(name))
        self.venv_launchers.clear()
        self.venv_outputs.clear()
        for name, staged in sorted(self.staged_configs.items()):
            if staged.remove():
                removed.append("staged-config:{}".format(name))
        self.staged_configs.clear()
        for _name, container in sorted(self.containers.items()):
            if container.remove():
                removed.append("container:{}".format(container.name))
            else:
                self.log(
                    "[place] WARNING: container {} could not be removed; it still "
                    "holds an endpoint on {}".format(
                        container.name, self.network.name if self.network else "?"
                    )
                )
        self.containers.clear()
        if self.network is not None:
            if self.network.remove():
                removed.append("network:{}".format(self.network.name))
            else:
                self.log(
                    "[place] WARNING: network {} could not be removed".format(
                        self.network.name
                    )
                )
            self.network = None
        return removed

    def status(self) -> Dict[str, object]:
        """Return a snapshot safe to serialise into a run record."""
        return {
            "network": self.network.name if self.network else None,
            "placed": {
                node.name: node.launcher for node in self.config.nodes
            },
            "containers": sorted(c.name for c in self.containers.values()),
            "remote": sorted(self.remote_outputs),
            "venvs": {
                name: describe_venv(facts)
                for name, facts in sorted(self.venv_facts.items())
            },
            "images": dict(sorted(self.image_facts.items())),
            "models": describe_staging(self.model_staging),
        }
