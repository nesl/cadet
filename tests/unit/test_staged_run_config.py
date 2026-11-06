"""Tests for putting a run's config where a containerised role can read it.

The problem these pin is one that looks like a broken launcher and is not. The
orchestrator passes every role a config *path*; a role in a container resolves
that path against the image, so it names whatever copy of the repository the
image was built with, and a config written, edited or never committed since is
simply absent there. The role then exits on ``FileNotFoundError`` before it has
done anything.

It is not a remote-only problem, which is the part that is easy to get wrong: a
container on *this* machine cannot see this machine's working tree either,
because nothing mounts it. One staging class serves both, and the tests below
fake the machine rather than caring which it is.

So three properties are under test:

* the **resolved** config is what gets sent - after ``extends``, after the
  ``--set`` overrides and after the sweep is expanded - which is what makes the
  file self-contained on a machine with no checkout;
* the **run id is unchanged** by sending it. ``config_hash`` digests the
  resolved config, and the staged file is that config, so the role derives the
  id the orchestrator did. A staged config that changed the id would make the
  role write its records into a different directory from the one the
  orchestrator reads, which is a silent failure rather than a loud one;
* the role is told **index 0** with it, because the staged file is one run with
  no sweep left to index. A sweep's second run would otherwise ask for index 1
  of a one-run file.

No daemon and no second machine are needed for any of it: the machine is faked
at the two operations the staging uses.
"""

import os

import pytest
import yaml

from cadet.config.loader import config_hash, load_run_configs
from cadet.core.docker_launcher import CONTAINER_RUN_CONFIG, ContainerSpec
from cadet.orchestrator.placement import (
    REMOTE_CONFIG_PARENT,
    PlacementError,
    StagedRunConfig,
)


class _FakeMachine:
    """A :class:`TargetMachine` at the three operations the staging uses."""

    def __init__(self, home="/home/elsewhere", fail_mkdir=False):
        self._home = home
        self.fail_mkdir = fail_mkdir
        self.commands = []
        self.sent = {}

    def home(self):
        return self._home

    def describe(self):
        return "the other machine"

    def run(self, argv, timeout_s=None):
        self.commands.append(list(argv))
        if self.fail_mkdir and argv[0] == "mkdir":
            return 1, "mkdir: cannot create directory"
        return 0, ""

    def send(self, local_path, dest_path):
        with open(local_path) as fh:
            self.sent[dest_path] = fh.read()


def _config(path="examples/distributed_sensing/quick.yaml"):
    from conftest import REPO_ROOT

    runs, _base = load_run_configs(os.path.join(REPO_ROOT, path), [])
    return runs[0]


def test_what_is_sent_is_the_resolved_config(tmp_path):
    config = _config()
    machine = _FakeMachine()
    staged = StagedRunConfig(machine, "0123456789-r0", log=lambda *_a: None)
    path = staged.send(config)

    assert path == "/home/elsewhere/{}/0123456789-r0.yaml".format(REMOTE_CONFIG_PARENT)
    data = yaml.safe_load(machine.sent[path])
    # Self-contained: nothing left for the other machine's checkout to resolve.
    assert "extends" not in data
    assert (data.get("sweep") or {}).get("axes") in (None, {}, [])
    assert data["experiment"]["name"] == config.experiment.name


def test_sending_the_config_does_not_change_the_run_id(tmp_path):
    """The property that makes this safe: the role derives the same id."""
    config = _config()
    machine = _FakeMachine()
    staged = StagedRunConfig(machine, "0123456789-r0", log=lambda *_a: None)
    path = staged.send(config)

    local = str(tmp_path / "staged.yaml")
    with open(local, "w") as fh:
        fh.write(machine.sent[path])
    reloaded, _base = load_run_configs(local, [])
    assert len(reloaded) == 1
    assert config_hash(reloaded[0]) == config_hash(config)


def test_the_staged_config_is_removed_and_its_directory_tried(tmp_path):
    machine = _FakeMachine()
    staged = StagedRunConfig(machine, "0123456789-r0", log=lambda *_a: None)
    path = staged.send(_config())
    assert staged.remove() is True
    assert ["rm", "-f", "--", path] in machine.commands
    # Non-recursive, so a config staged for a concurrent run is not taken too.
    assert ["rmdir", "--", os.path.dirname(path)] in machine.commands
    # Idempotent: a second teardown is not an error.
    assert staged.remove() is True


def test_an_unsafe_run_id_is_refused_before_any_path_is_built():
    machine = _FakeMachine()
    with pytest.raises(PlacementError, match="not a safe path component"):
        StagedRunConfig(machine, "../../etc", log=lambda *_a: None)
    assert machine.commands == []


def test_a_machine_that_cannot_make_the_directory_says_so():
    from cadet.core.ssh import RemoteError

    machine = _FakeMachine(fail_mkdir=True)
    staged = StagedRunConfig(machine, "0123456789-r0", log=lambda *_a: None)
    with pytest.raises(RemoteError, match="remote config directory"):
        staged.send(_config())


def test_the_spec_mounts_the_staged_config_read_only():
    spec = ContainerSpec(config_file="/home/elsewhere/cadet-configs/abc-r0.yaml")
    args = spec.docker_args()
    assert "-v" in args
    assert "/home/elsewhere/cadet-configs/abc-r0.yaml:{}:ro".format(
        CONTAINER_RUN_CONFIG
    ) in args


def test_a_spec_without_a_staged_config_mounts_nothing_extra():
    assert not any(CONTAINER_RUN_CONFIG in arg for arg in ContainerSpec().docker_args())


def test_the_mount_point_is_outside_the_image_repository():
    """A config mounted under /app would sit beside the repository's own."""
    assert not CONTAINER_RUN_CONFIG.startswith("/app")


# --- which node is handed the staged config, and with which index ----------
class _Node:
    def __init__(self, name, launcher="local"):
        self.name = name
        self.launcher = launcher
        self.role = "server"
        self.host = "localhost"
        self.port = 15010


class _Placement:
    """The one question the runner asks the placement about a config path."""

    def __init__(self, remote=()):
        self.remote = set(remote)

    def role_config_path(self, node):
        return CONTAINER_RUN_CONFIG if node.name in self.remote else None


def test_a_local_node_keeps_the_path_and_index_the_orchestrator_was_given():
    from cadet.orchestrator.runner import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    path, index = orch._role_config(_Placement(), _Node("here"), "examples/x.yaml", 3)
    assert (path, index) == ("examples/x.yaml", 3)


def test_a_remote_node_is_handed_the_staged_config_at_index_zero():
    """Index 0 because the staged file is one run with no sweep left in it."""
    from cadet.orchestrator.runner import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    place = _Placement(remote=["there"])
    path, index = orch._role_config(place, _Node("there", "ssh_docker"), "examples/x.yaml", 3)
    assert (path, index) == (CONTAINER_RUN_CONFIG, 0)


def test_a_container_on_this_machine_is_staged_for_too():
    """The easy mistake: the working tree is right there and still unreachable.

    Nothing mounts the repository into a node container, so a `docker` node has
    the same problem as an `ssh_docker` one and takes the same answer.
    """
    place = _Placement(remote=["local_container"])
    assert place.role_config_path(_Node("local_container", "docker")) == CONTAINER_RUN_CONFIG


def test_a_local_container_stages_beside_the_outputs_not_under_home(tmp_path):
    """The home directory is the wrong answer when the orchestrator is a container.

    A bind mount's source is resolved by the daemon on the *host*. Run through
    `scripts/cadet` - the documented primary way - the orchestrator's own home
    is a path inside its container, so docker found nothing there, created a
    **directory** at the mount source, and the role died with
    `IsADirectoryError: '/cadet/run-config.yaml'`. Found by running the netem
    example through the wrapper; `launcher: docker` was unusable that way.
    """
    from cadet.orchestrator.placement import LOCAL_CONFIG_DIRNAME

    machine = _FakeMachine(home="/not/a/host/path")
    staged = StagedRunConfig(machine, "0123456789-r0", log=lambda *_a: None)
    outputs = str(tmp_path / "outputs")
    path = staged.send(_config(), base_dir=outputs)

    assert path.startswith(os.path.join(outputs, LOCAL_CONFIG_DIRNAME))
    assert "/not/a/host/path" not in path
    # And the remote case is unchanged: the home directory is right there,
    # because nothing on that machine is containerised.
    remote = StagedRunConfig(_FakeMachine(), "0123456789-r0", log=lambda *_a: None)
    assert remote.send(_config()).startswith("/home/elsewhere/")
