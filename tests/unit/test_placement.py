"""Tests for per-node launcher dispatch and placement.

These are the parts that can be checked without a docker daemon: the naming the
recovery sweep depends on, which launcher each node's config selects, and the
two config views a containerised run needs. The end-to-end behaviour is covered
by the lifecycle matrix, which needs a daemon and so cannot live here.
"""

import os
import subprocess

import pytest

from cadet.core.docker_launcher import DockerLauncher, SshDockerLauncher, host_user
from cadet.core.launcher import LaunchSpec, LocalLauncher
from cadet.orchestrator.placement import (
    CONTAINER_OUTPUTS,
    ExecLauncher,
    NodeContainer,
    Placement,
    PlacementError,
    address_overrides,
    node_container_name,
    node_network,
    require_gpus,
    require_images,
    role_overrides,
)
from cadet.orchestrator.recovery import OWNED_RUN_OBJECT_RE


class _Node:
    def __init__(self, name, launcher="local", role="client", host="localhost", port=None):
        self.name = name
        self.launcher = launcher
        self.role = role
        self.host = host
        self.port = port
        self.container = None
        self.venv = None
        self.gpu_index = None
        self.user = None
        self.ssh_port = None
        self.ssh_identity = None
        self.transport_tuning = None


class _Network:
    def __init__(self, mode):
        self.mode = mode


class _Link:
    def __init__(self, name, from_, to, mode, direction="forward"):
        self.name = name
        self.from_ = from_
        self.to = to
        self.direction = direction
        self.network = _Network(mode)


class _Outputs:
    def __init__(self, root="outputs"):
        self.root = root


class _Config:
    def __init__(self, nodes, links=(), outputs_root="outputs"):
        self.nodes = list(nodes)
        self.links = list(links)
        self.outputs = _Outputs(outputs_root)


# --- naming the recovery sweep relies on --------------------------------
def test_names_are_run_scoped_and_recognised_by_the_recovery_sweep():
    # The sweep matches these by shape, so a name it cannot recognise is a
    # leftover nothing will ever clean up.
    net = node_network("0123456789-r0")
    container = node_container_name("0123456789-r0", "server")
    assert net == "cadet-0123456789-r0-net"
    assert container == "cadet-0123456789-r0-server"
    assert OWNED_RUN_OBJECT_RE.match(net)
    assert OWNED_RUN_OBJECT_RE.match(container)


# --- which launcher a node gets -----------------------------------------
def _placement(nodes, links=()):
    return Placement(_Config(nodes, links), "0123456789-r0", "/tmp/out", log=lambda *_a: None)


def test_a_local_node_gets_the_local_launcher():
    place = _placement([_Node("client")])
    assert isinstance(place.launcher_for(_Node("client")), LocalLauncher)


def test_a_docker_node_without_a_container_is_an_error():
    # Reached only if start() was skipped; it is a programming error rather than
    # a config one, and silently falling back to local would run the role in the
    # wrong place.
    place = _placement([_Node("server", launcher="docker")])
    with pytest.raises(PlacementError, match="no container was placed"):
        place.launcher_for(_Node("server", launcher="docker"))


def test_an_unprepared_ssh_node_is_refused():
    """The launcher is built in _prepare_remote, which has to ask that machine
    for its home and uid first; asking for one before the placement started is
    the same mistake as asking for an unplaced local container."""
    node = _Node("remote", launcher="ssh_docker", host="elsewhere")
    place = _placement([node])
    with pytest.raises(PlacementError, match="no remote container was prepared"):
        place.launcher_for(node)


def test_an_unprepared_venv_node_is_refused():
    """Built in _prepare_venv, which has to ask that machine what its venv
    contains first; asking for one before the placement started is the same
    mistake as asking for an unplaced local container."""
    node = _Node("remote", launcher="ssh_venv", host="elsewhere")
    place = _placement([node])
    with pytest.raises(PlacementError, match="no remote venv was prepared"):
        place.launcher_for(node)


def test_an_unknown_launcher_is_refused():
    # Every kind the schema accepts now has an implementation, so the rule under
    # test - "a kind this dispatch has never heard of is refused rather than
    # quietly run here" - is checked with a kind the schema would reject. It was
    # written against ssh_venv and that was implemented under it, which is
    # exactly the drift this form avoids.
    place = _placement([_Node("x", launcher="ssh_carrier_pigeon")])
    with pytest.raises(PlacementError, match="no .*implementation"):
        place.launcher_for(_Node("x", launcher="ssh_carrier_pigeon"))


def test_an_injected_launcher_overrides_every_node_and_places_nothing():
    # What a test harness wants: nothing is placed, so no container is created
    # for a role the override will run elsewhere.
    injected = LocalLauncher()
    config = _Config([_Node("server", launcher="docker")])
    place = Placement(config, "0123456789-r0", "/tmp/out", log=lambda *_a: None, override=injected)
    assert place.nodes_needing_containers() == []
    assert place.launcher_for(_Node("server", launcher="docker")) is injected


# --- which nodes need NET_ADMIN -----------------------------------------
def test_only_the_shaping_side_of_a_netem_link_needs_net_admin():
    links = [_Link("l", "client", "server", "netem", direction="forward")]
    place = _placement([_Node("client"), _Node("server")], links)
    assert place.netem_nodes() == ["client"]


def test_both_sides_shape_a_bidirectional_netem_link():
    links = [_Link("l", "client", "server", "netem", direction="both")]
    place = _placement([_Node("client"), _Node("server")], links)
    assert place.netem_nodes() == ["client", "server"]


def test_a_delay_link_shapes_nothing_in_the_kernel():
    links = [_Link("l", "client", "server", "delay")]
    place = _placement([_Node("client"), _Node("server")], links)
    assert place.netem_nodes() == []


# --- the two config views -----------------------------------------------
def test_address_overrides_move_only_the_placed_nodes():
    config = _Config([_Node("client"), _Node("server", launcher="docker", port=15040)])
    overrides = address_overrides(config, {"server": "172.19.0.3"})
    assert overrides == ["nodes.1.host=172.19.0.3"]


def test_address_overrides_do_not_touch_outputs():
    # outputs.root means a different directory on each side of the bind mount,
    # so it must not be in the rewrite both sides share.
    config = _Config([_Node("server", launcher="docker", port=1)])
    assert all("outputs" not in o for o in address_overrides(config, {"server": "10.0.0.2"}))


def test_role_overrides_point_outputs_at_the_mount():
    assert role_overrides() == ["outputs.root={}".format(CONTAINER_OUTPUTS)]


def test_role_overrides_are_empty_for_a_node_that_is_not_in_a_container():
    """Per node, not per run: a run mixes placements, and a local client told to
    write into a container's mount path would write where nothing exists."""
    place = _placement([_Node("client")])
    assert place.role_overrides_for(_Node("client")) == []


def test_role_args_are_empty_unless_the_node_is_remote():
    place = _placement([_Node("server", launcher="docker", port=15040)])
    assert place.role_args(_Node("server", launcher="docker", port=15040)) == []


# --- the exec launcher ---------------------------------------------------
class _Spec:
    def __init__(self, name):
        self.image = "cadet:dev"
        self.name = name
        self.network = None
        self.needs_net_admin = False
        self.outputs_dir = None
        self.gpu_index = None
        self.sysctls = {}
        self.env = {}
        self.extra_args = []
        self.run_as = None

    def docker_args(self):
        return ["--rm", "--name", self.name]


def test_exec_launcher_names_the_console_script():
    # `docker exec` does not use the image ENTRYPOINT, unlike `docker run`, so
    # the command has to name `cadet` or it would look for a binary called
    # `serve`.
    container = NodeContainer("server", _Spec("cadet-0123456789-r0-server"))
    launcher = ExecLauncher(container)
    spec = LaunchSpec(
        name="serve:server",
        argv=["/usr/bin/python3", "-m", "cadet", "serve", "cfg.yaml", "--node", "server"],
    )
    command = launcher.command(spec)
    assert command[:2] == ["docker", "exec"]
    # The env flags sit between; the container name is what follows them.
    at = command.index("cadet-0123456789-r0-server")
    assert command[at + 1] == "cadet"
    assert command[at + 2 :] == ["serve", "cfg.yaml", "--node", "server"]


def test_every_launcher_forces_unbuffered_child_output():
    # A readiness marker sitting in a block buffer reads as a startup timeout,
    # and all three launchers decide readiness by watching a log, so none of
    # them may leave the child buffered.
    spec = LaunchSpec(
        name="scenario_runner",
        argv=["python3", "scenario_runner.py"],
        env={"CADET_SCENARIO_VEHICLE": "vehicle.audi.tt"},
    )
    exec_command = ExecLauncher(NodeContainer("server", _Spec("c"))).command(spec)
    run_command = DockerLauncher(specs={"server": _Spec("c")}).command(spec)
    for command in (exec_command, run_command):
        assert "PYTHONUNBUFFERED=1" in command
        # And the spec's own variables reach the container: a scenario reads
        # them, and dropping them silently changed which vehicle was spawned.
        assert "CADET_SCENARIO_VEHICLE=vehicle.audi.tt" in command


def test_docker_run_does_not_repeat_the_console_script():
    # The contrast that makes the above necessary: `docker run` gets it from the
    # image's entry point.
    launcher = DockerLauncher(specs={"server": _Spec("c")})
    spec = LaunchSpec(
        name="serve:server", argv=["/usr/bin/python3", "-m", "cadet", "serve", "cfg.yaml"]
    )
    assert "cadet" not in launcher.command(spec)[-2:]


def test_a_mounted_container_runs_as_the_host_user():
    # Otherwise the role cannot write into the bind mount: the image's user need
    # not share the host's uid, and here it does not.
    assert host_user() == "{}:{}".format(os.getuid(), os.getgid())


def test_children_keep_their_own_signal_handlers():
    """SDL would otherwise install its own and swallow the teardown signal.

    Measured on the braking client with the ported visualiser: teardown took
    5.2 s (the full termination grace, so the client was killed) with SDL's
    handlers in place, against 0.2 s without them.
    """
    from cadet.core.launcher import LaunchSpec, child_env

    env = child_env(LaunchSpec(name="client", argv=["cadet", "client"]))
    assert env["SDL_NO_SIGNAL_HANDLERS"] == "1"
    assert env["PYTHONUNBUFFERED"] == "1"

    # An app's own value still wins, as for every other variable here.
    own = child_env(
        LaunchSpec(name="client", argv=["cadet", "client"], env={"SDL_NO_SIGNAL_HANDLERS": "0"})
    )
    assert own["SDL_NO_SIGNAL_HANDLERS"] == "0"


# --- what a fresh clone gets wrong first ----------------------------------
class _Machine:
    """A :class:`TargetMachine` at the one operation these checks use."""

    def __init__(self, answers):
        self.answers = answers  # first argv word -> (returncode, output)

    def describe(self):
        return "the other machine"

    def run(self, argv, timeout_s=None):
        return self.answers.get(argv[0], (1, ""))


def test_a_missing_image_names_the_build_command_not_a_registry():
    """The error it replaces is about pull access, which is not the problem."""
    machine = _Machine({"docker": (1, "Error: No such image")})
    with pytest.raises(PlacementError) as err:
        require_images(["cadet:dev"], machine=machine)
    message = str(err.value)
    assert "cannot be pulled" in message
    assert "docker build -f docker/Dockerfile -t cadet:dev" in message
    # The build arg is part of the documented command: without it the image
    # records no commit and the staleness check can only say "cannot tell".
    assert "GIT_SHA" in message
    # And, for a second machine, the way to get one there without building.
    assert "docker save cadet:dev" in message


def test_every_missing_image_is_named_at_once():
    # Otherwise the first one fails after the run's network exists, and the
    # second is only discovered on the next attempt.
    machine = _Machine({"docker": (1, "")})
    with pytest.raises(PlacementError) as err:
        require_images(["a:1", "b:2", "a:1"], machine=machine)
    message = str(err.value)
    assert "'a:1', 'b:2'" in message
    assert message.count("docker build") == 2


def test_images_that_are_present_are_not_refused():
    machine = _Machine({"docker": (0, "sha256:abc")})
    require_images(["cadet:dev"], machine=machine)


def test_a_gpu_index_the_machine_does_not_have_is_refused_by_name():
    machine = _Machine({"nvidia-smi": (0, "0\n1\n")})
    node = _Node("server", launcher="docker")
    node.gpu_index = 7
    with pytest.raises(PlacementError) as err:
        require_gpus([node], machine=machine)
    message = str(err.value)
    assert "'server' asks for GPU 7" in message
    assert "2 GPU(s) (0..1)" in message


def test_a_gpu_index_within_range_is_accepted():
    machine = _Machine({"nvidia-smi": (0, "0\n1\n")})
    node = _Node("server", launcher="docker")
    node.gpu_index = 1
    require_gpus([node], machine=machine)


def test_a_machine_that_cannot_be_asked_about_gpus_is_not_refused():
    """None and zero are different answers.

    A machine whose driver is reachable only from inside the container cannot
    answer, and refusing on that basis would refuse a working arrangement.
    """
    machine = _Machine({})  # nvidia-smi is absent
    node = _Node("server", launcher="docker")
    node.gpu_index = 7
    require_gpus([node], machine=machine)


def test_a_node_without_a_gpu_index_asks_nothing_of_the_machine():
    class _Never:
        def describe(self):
            return "unused"

        def run(self, argv, timeout_s=None):
            raise AssertionError("the machine should not be asked")

    require_gpus([_Node("client")], machine=_Never())


# --- an image whose code is not the repository's ---------------------------
def test_an_image_from_another_commit_is_a_warning_naming_the_rebuild(monkeypatch):
    """A warning and not a refusal: reproducing against an older image is valid.

    The failure this catches is silent. A role in a container runs the image's
    code, so a run against a stale image looks like the change not working.
    """
    from cadet.orchestrator import placement as place

    monkeypatch.setattr(place, "repo_revision", lambda *_a: ("a" * 40, False))
    monkeypatch.setattr(place, "image_revision", lambda *_a, **_k: "b" * 40)
    lines = []
    place.warn_stale_images(["cadet:dev"], lines.append)
    assert len(lines) == 1
    assert "was built from bbbbbbbbbb" in lines[0]
    assert "this repository is at aaaaaaaaaa" in lines[0]
    assert "docker build" in lines[0] and "GIT_SHA" in lines[0]


def test_an_image_with_no_recorded_commit_says_it_cannot_tell(monkeypatch):
    from cadet.orchestrator import placement as place

    monkeypatch.setattr(place, "repo_revision", lambda *_a: ("a" * 40, False))
    monkeypatch.setattr(place, "image_revision", lambda *_a, **_k: None)
    lines = []
    place.warn_stale_images(["cadet:dev"], lines.append)
    assert "records no commit" in lines[0]


def test_a_matching_image_with_a_dirty_tree_is_still_warned_about(monkeypatch):
    """The case easiest to misread: the commit agrees and the edit is not in it."""
    from cadet.orchestrator import placement as place

    monkeypatch.setattr(place, "repo_revision", lambda *_a: ("a" * 40, True))
    monkeypatch.setattr(place, "image_revision", lambda *_a, **_k: "a" * 40)
    lines = []
    place.warn_stale_images(["cadet:dev"], lines.append)
    assert "uncommitted changes" in lines[0]


def test_a_matching_image_and_a_clean_tree_say_nothing(monkeypatch):
    from cadet.orchestrator import placement as place

    monkeypatch.setattr(place, "repo_revision", lambda *_a: ("a" * 40, False))
    monkeypatch.setattr(place, "image_revision", lambda *_a, **_k: "a" * 40)
    assert place.warn_stale_images(["cadet:dev"], lambda _m: None) == []


def test_a_tree_that_is_not_a_checkout_is_not_warned_about(monkeypatch):
    """An installed copy need not be a git checkout, and that is not a fault."""
    from cadet.orchestrator import placement as place

    monkeypatch.setattr(place, "repo_revision", lambda *_a: (None, False))
    called = []
    monkeypatch.setattr(place, "image_revision", lambda *_a, **_k: called.append(1))
    assert place.warn_stale_images(["cadet:dev"], lambda _m: None) == []
    assert called == []


def test_a_role_in_a_container_is_signalled_where_it_runs(monkeypatch):
    """Killing a `docker exec` client does not signal the role it started.

    The client is a local process and the role is a process in the container,
    so reaping the client leaves the role running until the container is
    removed, which kills it outright. A role killed that way never runs its
    shutdown path. Measured on a containerised acceptance cell: the server's
    per-request metrics CSV, which the same app writes when placed locally,
    was absent from the run directory.
    """
    from cadet.core.launcher import Handle
    from cadet.orchestrator import placement as placement_mod

    calls = []

    def _fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(placement_mod.subprocess, "run", _fake_run)

    class _Proc(object):
        def __init__(self):
            self.waits = 0
            self.terminated = False

        def poll(self):
            # Still running until the signal has been delivered and waited on.
            return 0 if self.waits else None

        def wait(self, timeout=None):
            self.waits += 1
            return 0

        def terminate(self):
            self.terminated = True

    launcher = ExecLauncher(NodeContainer("server", _Spec("cadet-0123456789-r0-server")))
    spec = LaunchSpec(
        name="serve:server",
        argv=["/usr/bin/python3", "-m", "cadet", "serve", "cfg.yaml", "--node", "server"],
    )
    proc = _Proc()
    launcher.stop(Handle(spec=spec, proc=proc, log_fh=None))

    signalled = [c for c in calls if "pkill" in c]
    assert signalled, calls
    argv = signalled[0]
    assert argv[:3] == ["docker", "exec", "cadet-0123456789-r0-server"]
    assert argv[3:6] == ["pkill", "-TERM", "-f"]
    # The leading [-] keeps the pattern from matching pkill's own command line.
    assert argv[6] == "[-]-node server"
    # The container is not this launcher's to remove: the placement owns it.
    assert not [c for c in calls if c[:2] == ["docker", "rm"]]
    # The client is reaped by the wait that follows the signal, so no terminate.
    assert proc.terminated is False


def test_a_roles_own_container_is_stopped_before_it_is_removed(monkeypatch):
    """Where the role *is* the container's main process, a stop signals it."""
    from cadet.core import docker_launcher as docker_mod
    from cadet.core.launcher import Handle

    calls = []

    def _fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(docker_mod.subprocess, "run", _fake_run)

    class _Dead(object):
        def poll(self):
            return 0

    launcher = DockerLauncher(specs={"server": _Spec("cadet-0123456789-r0-server")})
    spec = LaunchSpec(name="serve:server", argv=["python3", "-m", "cadet", "serve", "c.yaml"])
    launcher.stop(Handle(spec=spec, proc=_Dead(), log_fh=None), grace_s=7.0)

    verbs = [c[1] for c in calls if c[:1] == ["docker"]]
    assert verbs[:2] == ["stop", "rm"], calls
    assert "--time" in calls[0] and calls[0][calls[0].index("--time") + 1] == "7"


def test_a_roles_argv_carries_the_stage_modules(monkeypatch):
    """A containerised role inherits no environment, so the argv must carry them.

    `--stage-module` used to travel only through CADET_STAGE_MODULES. A local
    role is a child of this process and inherits it; a role in a container is
    given only the variables its launcher passes, so it started with an empty
    registry and died on the first stage its config named. Measured on an
    acceptance cell: `KeyError: 'camera'` from a server whose config named four
    registered stages.
    """
    from cadet.orchestrator.runner import Orchestrator

    monkeypatch.setenv("CADET_STAGE_MODULES", "cadet.pipeline.stages.braking")
    orch = Orchestrator.__new__(Orchestrator)
    argv = orch._base_argv("serve", "cfg.yaml", [], 0, ["examples.braking.pipeline.app"])
    assert "--stage-module" in argv
    assert argv[argv.index("--stage-module") + 1] == "cadet.pipeline.stages.braking"

    # Nothing published, nothing added: a non-pipeline run's argv is unchanged.
    monkeypatch.delenv("CADET_STAGE_MODULES", raising=False)
    plain = orch._base_argv("serve", "cfg.yaml", [], 0, [])
    assert "--stage-module" not in plain


def test_a_containerised_role_is_given_a_writable_home():
    """The uid a role runs as has no home in the image, and `/` is not writable.

    A role in a container runs as the invoking account's uid, which the image
    has no passwd entry for, so the daemon gives it `/` as a home. Anything
    that caches under the home then fails: the CARLA client egg extracts itself
    there, and a containerised braking client died before it had read a frame.
    """
    from cadet.core.docker_launcher import CONTAINER_HOME

    spec = LaunchSpec(
        name="client:client",
        argv=["python3", "-m", "cadet", "client", "c.yaml", "--node", "client"],
    )
    run_command = DockerLauncher(specs={"client": _Spec("c")}).command(spec)
    exec_command = ExecLauncher(NodeContainer("client", _Spec("c"))).command(spec)
    for command in (run_command, exec_command):
        assert "HOME={}".format(CONTAINER_HOME) in command
    # A local role keeps the account's own home: this is a container's problem.
    local = LocalLauncher()
    assert not hasattr(local, "container_spec")


def test_a_gpu_node_without_an_index_gets_the_machines_gpus():
    """A container that saw no GPU did not refuse: it died inside torch.

    A node that asks for a GPU without naming which one gets what the same
    node gets when its role runs locally, which is the machine's devices. The
    container used to be given none, and the run then loaded for twelve seconds
    and failed with no CUDA device - which reads as a broken model rather than
    as a container that was handed no GPU.
    """
    from cadet.core.docker_launcher import container_spec_for

    class _Node(object):
        def __init__(self, device, gpu_index=None):
            self.name = "server"
            self.container = None
            self.device = device
            self.gpu_index = gpu_index
            self.transport_tuning = None

    unnamed = container_spec_for(_Node("gpu"), "0123456789-r0").docker_args()
    assert "--gpus" in unnamed
    assert unnamed[unnamed.index("--gpus") + 1] == "all"

    # A named index still pins one device, so two nodes on one machine do not
    # silently share a GPU.
    named = container_spec_for(_Node("gpu", 1), "0123456789-r0").docker_args()
    assert named[named.index("--gpus") + 1] == "device=1"

    # A CPU node asks for no GPU and is given none.
    assert "--gpus" not in container_spec_for(_Node("cpu"), "0123456789-r0").docker_args()
