"""Start and stop the simulator side of a run: CARLA, and the scenario driver.

The simulator used to be a long-lived shared container started by hand; a
crashed simulator was discovered only when the next run failed. This module
makes the simulator part of the run's lifecycle: the orchestrator brings it up,
waits for it, and takes it down. It provides :class:`SimSide`
(``start`` / ``wait_ready`` / ``start_scenario`` / ``stop`` / ``status``),
mirroring :class:`cadet.network.netwaggle.emulator.Emulator` so both halves of
a run are driven the same way, plus :func:`plumbing_network`,
:func:`carla_container`, and :func:`scenario_container` (run-scoped names).

The scenario driver gets a container for the same reason the simulator does: it
used to be started by ``docker exec`` into a long-lived container that also held
the application code, so the scenario's dependencies and the app's were one set
and neither could be pinned without the other. In its own container it carries
only ScenarioRunner's pinned dependencies, and the ScenarioRunner tree and the
CARLA PythonAPI are mounted read-only rather than baked in, because the version
pin checks them where they are mounted.

All objects carry the ``cadet-`` prefix so
:func:`cadet.orchestrator.recovery.sweep` can remove them after a crash. It
requires a docker client on the host; role containers and the emulation
container do not have one. The plumbing network is run-scoped and unshaped:
CARLA control traffic must not cross an emulated link, or the experiment is
measuring the simulator's own control loop, and a network per run prevents two
runs from seeing each other's simulator by name. A hand-started simulator is
left alone: every object here is named after the run id, so a launch collides
with nothing of anyone else's except on the RPC port, which defaults to the
original's and is what a run changes when it must not collide. Configured via
``source.carla.image``, ``port``, ``gpu_index``, ``quality``, ``offscreen``,
and ``source.scenario.runner.path`` / ``.image``.
"""

import hashlib
import socket
import subprocess
import time
from typing import Dict, List, Optional, Sequence, Tuple

# Everything created here carries this prefix, so the recovery sweep can
# recognise it as CADET's own. Kept in one place with the launcher's.
from cadet.core.docker_launcher import OWNED_CONTAINER_PREFIX

# CARLA listens on the RPC port and the two above it (the streaming and
# secondary ports). All three are published so the run can be reached and
# polled from the host.
CARLA_PORT_SPAN = 3
# How long to wait for the server to accept a connection on its RPC port.
CARLA_READY_TIMEOUT_S = 120.0
# How often to poll for it.
_POLL_S = 0.5
# How long docker is given to stop a container before it is killed.
STOP_TIMEOUT_S = 10
# Where the scenario container finds the two external trees it is given. These
# are the image's own conventions, not a machine's paths.
CONTAINER_SCENARIO_RUNNER = "/opt/scenario_runner"
# Where a mounted CARLA client distribution goes. Deliberately not /opt/carla:
# the image bakes the client egg at /opt/carla/PythonAPI and puts it on the path
# with a .pth, so mounting a tree at /opt/carla hides the egg and the scenario
# fails on `import carla` - which it did, once, for exactly this reason. A mount
# point must not be a parent of cadet.sim.versions.CONTAINER_CARLA_EGG_DIR.
CONTAINER_CARLA_API = "/opt/carla-client"
# What the scenario container runs so that it exists without doing anything;
# the scenario is then started inside it. Same place-first arrangement the node
# containers use, and for the same reason: the container has to exist before
# anything is run in or attached to it.
IDLE_ENTRYPOINT = "sleep"
IDLE_ARGS = ["infinity"]
# What CUDA_VISIBLE_DEVICES must say inside a container that was given one
# device by `--gpus device=N`: that device is renumbered to 0, whatever N was.
CONTAINER_GPU_INDEX = 0


def carla_launch_settings(config):
    """Return the CARLA settings a launch needs, or None if there is none.

    A launch is declared by ``source.carla.image``: with it set the
    orchestrator starts the server, and without it the simulator is somebody
    else's to run. The kind is checked rather than the attribute, because only
    a CARLA source has the block at all.
    """
    source = getattr(config, "source", None)
    if source is None or source.kind != "carla":
        return None
    return source.carla if source.carla.image else None


class SimSideError(Exception):
    """The simulator side could not be brought up or torn down."""


# A shared server is named after what it was launched with rather than after a
# run, because the whole point is that the next run finds it. Two runs reuse one
# server exactly when they would otherwise have started the same one.
SHARED_SCOPE_PREFIX = "sim-"
SHARED_SCOPE_LEN = 10


def shared_scope(carla) -> str:
    """Return the name scope a reusable simulator is found again under.

    A fingerprint of the launch settings, not of the run. The port is in it
    because that is what a run changes when it must not collide with another;
    the GPU index because a server on the wrong device is the one failure this
    must never quietly produce; and the image because a reused server of the
    wrong version is a client error a long way from its cause.
    """
    fields = (carla.image, carla.port, carla.gpu_index, carla.quality, carla.offscreen)
    digest = hashlib.sha256("|".join(repr(field) for field in fields).encode("utf-8"))
    return "{}{}".format(SHARED_SCOPE_PREFIX, digest.hexdigest()[:SHARED_SCOPE_LEN])


def plumbing_network(run_id: str) -> str:
    """Return the run-scoped docker network name for framework plumbing."""
    return "{}{}-plumbing".format(OWNED_CONTAINER_PREFIX, run_id)


def carla_container(run_id: str) -> str:
    """Return the run-scoped CARLA container name."""
    return "{}{}-carla".format(OWNED_CONTAINER_PREFIX, run_id)


def scenario_container(run_id: str) -> str:
    """Return the run-scoped scenario-driver container name."""
    return "{}{}-scenario".format(OWNED_CONTAINER_PREFIX, run_id)


def carla_command(carla) -> List[str]:
    """Return the CARLA server command line for this config.

    Ported from the arrangement the hand-started simulator uses: offscreen
    rendering with OpenGL, no sound, and an explicit RPC port. Offscreen is not
    optional in practice - there is no display in a container, and a server that
    tries to open one fails in a way that reads as a timeout rather than as a
    missing display.
    """
    argv = ["./CarlaUE4.sh", "-opengl", "-nosound"]
    argv.append("-quality-level={}".format(carla.quality))
    argv.append("-carla-rpc-port={}".format(carla.port))
    return argv


def _docker(argv: List[str], docker: str = "docker", check: bool = False):
    """Run a docker command, returning the CompletedProcess."""
    try:
        result = subprocess.run([docker] + argv, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise SimSideError(
            "cannot run {!r}: {}. The simulator side needs a docker client, so "
            "it runs on the host, not inside a container.".format(docker, exc)
        )
    if check and result.returncode != 0:
        raise SimSideError(
            "docker {} failed: {}".format(
                " ".join(argv[:2]), (result.stderr or result.stdout).strip()
            )
        )
    return result


def docker_available(docker: str = "docker") -> bool:
    """True if a docker client here can talk to a daemon."""
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


def port_open(host: str, port: int, timeout_s: float = 1.0) -> bool:
    """True if something accepts a TCP connection on ``host:port``."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(timeout_s)
    try:
        probe.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


class SimSide:
    """One run's simulator: a CARLA server and the scenario driver beside it."""

    def __init__(self, config, run_id: str, log=print, docker: str = "docker"):
        self.config = config
        self.run_id = run_id
        self.log = log
        self.docker = docker
        carla = carla_launch_settings(config)
        # Reuse is the default, so the server and the network it sits on are
        # named after the launch settings. The scenario driver stays run-scoped
        # whatever happens: it is the run, and a stale one is never reusable.
        self.shared = carla is not None and not carla.per_run
        scope = shared_scope(carla) if self.shared else run_id
        self.network = plumbing_network(scope)
        self.carla_name = carla_container(scope)
        self.scenario_name = scenario_container(run_id)
        self._created: List[str] = []  # for teardown, in reverse order
        self._state = "empty"
        self._detail = ""

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> "SimSide":
        """Create the plumbing network and start the CARLA server on it."""
        carla = carla_launch_settings(self.config)
        if carla is None:
            return self
        if not docker_available(self.docker):
            raise SimSideError(
                "source.carla.image is declared but no docker daemon is reachable "
                "here. The orchestrator owns the simulator containers, so it has "
                "to run where docker does - on the host."
            )

        if self.shared and self._reuse(carla):
            return self
        # A shared server that exists and is not answering is this code's to
        # replace: it cannot be reused, and nothing else is going to remove it.
        if self.shared:
            _docker(["rm", "-f", self.carla_name], self.docker)
        if not self._network_exists():
            _docker(["network", "create", self.network], self.docker, check=True)
            self.log("[orch] network {}".format(self.network))
            if not self.shared:
                # A shared network outlives the run that created it, so it is
                # not this run's to tear down. `cadet down` removes it.
                self._created.append("network:" + self.network)

        argv = ["run", "-d", "--name", self.carla_name, "--network", self.network]
        # Published on the loopback only: the run's server is for this run, and
        # the host needs it reachable to poll readiness.
        for offset in range(CARLA_PORT_SPAN):
            port = carla.port + offset
            argv += ["-p", "127.0.0.1:{}:{}".format(port, port)]
        if carla.gpu_index is not None:
            # --gpus selects which host device is passed in, and
            # CUDA_VISIBLE_DEVICES then constrains the runtime inside the
            # container - but in the container's own numbering, which starts at
            # 0 for the single device passed in. The host index names a device
            # the container does not have: measured as `--gpus device=1` with
            # `CUDA_VISIBLE_DEVICES=1` giving a device_count of 0, against 1
            # when it is 0.
            argv += ["--gpus", "device={}".format(carla.gpu_index)]
            argv += ["-e", "CUDA_VISIBLE_DEVICES={}".format(CONTAINER_GPU_INDEX)]
        if carla.offscreen:
            argv += ["-e", "SDL_VIDEODRIVER=offscreen"]
        argv.append(carla.image)
        argv += ["/bin/bash", "-c", " ".join(carla_command(carla))]

        _docker(argv, self.docker, check=True)
        if not self.shared:
            self._created.append("container:" + self.carla_name)
        self._state = "started"
        self.log(
            "[orch] carla {} on {} (rpc {})".format(
                self.carla_name, self.network, carla.port
            )
        )
        return self

    def start_scenario(
        self,
        argv: List[str],
        log_path: str,
        image: str,
        mounts: Sequence[Tuple[str, str]] = (),
        env: Optional[Dict[str, str]] = None,
        workdir: Optional[str] = None,
    ) -> subprocess.Popen:
        """Run the scenario driver inside its own container, beside the server.

        ``argv`` is the app's own sim-hook command, unchanged: the scenario
        driver is the app's business and this only decides *where* it runs.

        The container is given **CARLA's own network namespace** rather than an
        address on the plumbing network. The simulator's address is one config
        value that every part of a run has to agree on, and from the plumbing
        network the answer would be the server's container address while from
        the host - where the roles run - it is the published loopback port. One
        config value cannot be both. Sharing the namespace makes ``127.0.0.1``
        correct inside this container too, which is also the address the
        original used: its ScenarioRunner and its server were one host.

        ``workdir`` is where the command runs, and it matters because a
        scenario's ``--additionalScenario`` and ``--configFile`` are written
        relative to the repository root.

        ``mounts`` are ``(host path, container path)`` pairs. They are passed in
        rather than derived from the config because what a scenario container
        needs mounted - the ScenarioRunner tree, and an egg the image was not
        built with - is the caller's knowledge, and a mount's two halves are a
        host path and a container path that must not be confused for each other.

        Returns the ``docker exec`` process, whose lifetime tracks the
        scenario's, so the caller's existing readiness barrier and guard rules
        apply to it unchanged.
        """
        create = [
            "run",
            "-d",
            "--name",
            self.scenario_name,
            # Share the server's namespace; see the note above on why the
            # plumbing network would give the simulator two addresses.
            "--network",
            "container:" + self.carla_name,
            "--entrypoint",
            IDLE_ENTRYPOINT,
        ]
        if workdir is not None:
            create += ["-w", workdir]
        for host_path, container_path in mounts:
            create += ["-v", "{}:{}:ro".format(host_path, container_path)]
        for key, value in sorted((env or {}).items()):
            create += ["-e", "{}={}".format(key, value)]
        # No display in a container, and a scenario that tries to open one fails
        # in a way that reads as a timeout.
        create += ["-e", "SDL_VIDEODRIVER=dummy", "-e", "SDL_AUDIODRIVER=dummy"]
        # Unbuffered for the same reason the launchers set it: the readiness
        # barrier reads this container's log, and a buffered marker reads as a
        # startup timeout. Set at creation because `docker exec` inherits it.
        create += ["-e", "PYTHONUNBUFFERED=1"]
        create.append(image)
        create += IDLE_ARGS
        # A crashed run leaves this exact name; clearing our own name first is
        # the rule the pre-run cleanup already follows.
        _docker(["rm", "-f", self.scenario_name], self.docker)
        _docker(create, self.docker, check=True)
        self._created.append("container:" + self.scenario_name)
        self.log("[orch] scenario container {}".format(self.scenario_name))

        handle = open(log_path, "w")
        exec_argv = ["exec"]
        if workdir is not None:
            # `docker run -w` sets the container's default; an exec does not
            # inherit it, so it is given again here.
            exec_argv += ["-w", workdir]
        exec_argv.append(self.scenario_name)
        return subprocess.Popen(
            [self.docker] + exec_argv + list(argv),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _reuse(self, carla) -> bool:
        """True if a server of this name is already up and answering.

        Both halves are checked because either alone is misleading: a running
        container whose server died inside it answers nothing, and an open port
        with no container of ours behind it is somebody else's server, which
        preflight is the thing that refuses.
        """
        if not self.container_running(self.carla_name):
            return False
        if not port_open("127.0.0.1", carla.port):
            return False
        self._state = "started"
        self.log(
            "[orch] carla {} reused (rpc {}); the scenario reloads the world".format(
                self.carla_name, carla.port
            )
        )
        return True

    def _network_exists(self) -> bool:
        """True if the plumbing network is already there, as a reused one is."""
        result = _docker(
            ["network", "inspect", "-f", "{{.Name}}", self.network], self.docker
        )
        return result.returncode == 0 and result.stdout.strip() == self.network

    def wait_ready(self, timeout_s: float = CARLA_READY_TIMEOUT_S) -> bool:
        """Block until the CARLA server accepts a connection on its RPC port.

        Readiness is a TCP accept rather than a log line: CARLA's startup output
        varies between builds, and what the client actually needs is a server
        that will answer. The container is polled too, so a server that dies
        during startup fails immediately instead of waiting out the timeout -
        the same rule the role readiness barrier follows.
        """
        if carla_launch_settings(self.config) is None:
            return True
        port = carla_launch_settings(self.config).port
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self.container_running(self.carla_name):
                self._detail = "carla container exited during startup"
                self.log("[orch] {}; see docker logs {}".format(self._detail, self.carla_name))
                return False
            if port_open("127.0.0.1", port):
                self._state = "running"
                self.log("[orch] carla ready on rpc {}".format(port))
                return True
            time.sleep(_POLL_S)
        self._detail = "carla did not accept a connection on {} within {:.0f}s".format(
            port, timeout_s
        )
        return False

    def stop(self) -> Dict[str, object]:
        """Remove everything this run created, in reverse order.

        A reused server and its network are not among them: they were not this
        run's to create and the next run expects to find them. `cadet down`
        removes them, which is also what makes them visible as leftovers.
        """
        removed = []
        for item in reversed(self._created):
            kind, _, name = item.partition(":")
            if kind == "container":
                # `docker rm` has no timeout flag; that belongs to `docker stop`.
                _docker(["stop", "-t", str(STOP_TIMEOUT_S), name], self.docker)
                _docker(["rm", "-f", name], self.docker)
                removed.append(item)
            elif kind == "network":
                # The network can only go once nothing is attached to it, which
                # the container removals above have just ensured.
                _docker(["network", "rm", name], self.docker)
                removed.append(item)
        self._created = []
        self._state = "stopped"
        if self.shared and self.container_running(self.carla_name):
            self.log(
                "[orch] carla {} left running for the next run "
                "(cadet down removes it)".format(self.carla_name)
            )
        return {"state": self._state, "removed": removed, "detail": self._detail}

    # -- introspection -------------------------------------------------------
    def container_running(self, name: str) -> bool:
        """True if a container of this name exists and is running."""
        result = _docker(
            ["inspect", "-f", "{{.State.Running}}", name], self.docker
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    def container_pid(self, name: str) -> Optional[int]:
        """Return a container's main pid, which is how its netns is reached.

        This is what the emulated backend attaches a veth to when a node lives
        in its own container: a pid is a handle on a network namespace, and
        ``ip link set ... netns <pid>`` takes it directly.
        """
        result = _docker(["inspect", "-f", "{{.State.Pid}}", name], self.docker)
        if result.returncode != 0:
            return None
        try:
            pid = int(result.stdout.strip())
        except ValueError:
            return None
        return pid or None

    def simulator_host(self, from_network: bool = False) -> str:
        """Return the address to reach this run's simulator from.

        Two answers, because there are two vantage points. A process on the
        plumbing network reaches it at its container address; a process on the
        host reaches it at the published loopback port. Returning the wrong one
        is a connection refused rather than anything subtler, so the caller says
        which it wants.
        """
        if not from_network:
            return "127.0.0.1"
        address = self.container_address(self.carla_name)
        return address or "127.0.0.1"

    def container_address(self, name: str) -> Optional[str]:
        """Return a container's address on the plumbing network.

        Looked up with ``index`` rather than field access: a Go template cannot
        reach a map key containing a hyphen, and every run-scoped network name
        has several.
        """
        template = '{{{{(index .NetworkSettings.Networks "{}").IPAddress}}}}'.format(self.network)
        result = _docker(["inspect", "-f", template, name], self.docker)
        address = result.stdout.strip()
        if result.returncode != 0 or not address or address == "<no value>":
            return None
        return address

    def status(self) -> Dict[str, object]:
        """Return a snapshot safe to serialise into a run record."""
        carla = carla_launch_settings(self.config)
        return {
            "state": self._state,
            "network": self.network,
            "carla": {
                "container": self.carla_name,
                "image": carla.image if carla else None,
                "port": carla.port if carla else None,
                "gpu_index": carla.gpu_index if carla else None,
            },
            "detail": self._detail,
        }
