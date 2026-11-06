"""Tests for running the scenario driver in a container of its own.

The scenario is driven through the launcher interface so that readiness,
supervision and teardown stay one code path whichever side of a container the
scenario runs on. These tests pin the three decisions that are not obvious from
the interface: where the container gets its network, what has to be mounted
into it, and which interpreter runs the tree.
"""

import pytest

from cadet.config.schema import CarlaSource
from cadet.sim import containers
from cadet.sim.containers import SimSideError
from cadet.sim.scenario_command import CONTAINER_PYTHON, scenario_runner_command
from cadet.sim.scenario_launcher import (
    CONTAINER_REPO,
    TREE_HOST_PATH_ENV,
    scenario_launcher_for,
    scenario_mounts,
)

SCENARIO = {
    "name": "AtSpeedBraking_1",
    "definition": "examples/braking/flat/scenarios/at_speed_braking_scenario.py",
    "runner": {"path": "/opt/scenario_runner"},
}


def _config(image=None, **carla):
    """A CARLA source declaring a launch, and a scenario with or without an image."""
    scenario = dict(SCENARIO)
    scenario["runner"] = dict(scenario["runner"])
    if image is not None:
        scenario["runner"]["image"] = image
    settings = dict({"image": "carlasim/carla:0.9.10.1"}, **carla)
    return type("_C", (), {"source": CarlaSource(carla=settings, scenario=scenario)})()


@pytest.fixture
def captured(monkeypatch):
    """Capture docker argv instead of running it."""
    calls = []

    class _Result:
        returncode = 0
        stdout = "true"
        stderr = ""

    monkeypatch.setattr(
        containers, "_docker", lambda argv, docker="docker", check=False: (
            calls.append(list(argv)), _Result()
        )[1]
    )
    monkeypatch.setattr(containers, "docker_available", lambda docker="docker": True)

    # The exec that starts the scenario is a Popen of its own rather than a
    # _docker call, because its lifetime has to track the scenario it runs.
    class _Proc:
        pid = 4321

        def poll(self):
            return None

    monkeypatch.setattr(
        containers.subprocess,
        "Popen",
        lambda argv, **kw: (calls.append(list(argv)), _Proc())[1],
    )
    return calls


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A ScenarioRunner tree on this machine, named the way a host names one."""
    path = tmp_path / "scenario_runner"
    path.mkdir()
    monkeypatch.setenv(TREE_HOST_PATH_ENV, str(path))
    return str(path)


def test_no_image_leaves_the_scenario_on_the_host(tree):
    """Without an image the hook runs where it always did."""
    sim = containers.SimSide(_config(), "abc123", log=lambda *a: None)
    assert scenario_launcher_for(_config(), sim, "/repo") is None


def test_an_image_asks_for_a_container(tree):
    """With an image the hook is launched into one, with both mounts."""
    config = _config(image="cadet:sim")
    sim = containers.SimSide(config, "abc123", log=lambda *a: None)
    launcher = scenario_launcher_for(config, sim, "/repo")
    assert launcher is not None
    assert launcher.image == "cadet:sim"
    assert launcher.mounts == [(tree, "/opt/scenario_runner"), ("/repo", CONTAINER_REPO)]
    assert launcher.workdir == CONTAINER_REPO


def test_an_unnamed_tree_is_refused_by_name(monkeypatch):
    """The refusal names the variable, because that is the thing to set."""
    monkeypatch.delenv(TREE_HOST_PATH_ENV, raising=False)
    with pytest.raises(SimSideError) as caught:
        scenario_mounts(_config(image="cadet:sim"), "/repo")
    assert TREE_HOST_PATH_ENV in str(caught.value)


def test_a_tree_that_is_not_there_is_refused(monkeypatch, tmp_path):
    """A named directory that does not exist is a refusal, not an empty mount.

    A missing mount source makes docker create the path as an empty directory,
    so the scenario would fail at its first import instead of here.
    """
    monkeypatch.setenv(TREE_HOST_PATH_ENV, str(tmp_path / "absent"))
    with pytest.raises(SimSideError) as caught:
        scenario_mounts(_config(image="cadet:sim"), "/repo")
    assert "not a directory" in str(caught.value)


def test_the_container_shares_the_servers_network(captured, tree, tmp_path):
    """The scenario container gets CARLA's namespace, so 127.0.0.1 is the server.

    The alternative - its own address on the plumbing network - would make the
    simulator's address depend on who is asking: the server's container address
    from inside the network, the published loopback port from the host where the
    roles run. One config value cannot be both, and the symptom of getting it
    wrong is an ego that never attaches rather than a connection error.
    """
    config = _config(image="cadet:sim")
    sim = containers.SimSide(config, "abc123", log=lambda *a: None)
    sim.start()
    sim.start_scenario(
        argv=["python3", "scenario_runner.py"],
        log_path=str(tmp_path / "scenario.log"),
        image="cadet:sim",
        workdir=CONTAINER_REPO,
    )
    create = [argv for argv in captured if argv[0] == "run"][-1]
    assert "container:" + sim.carla_name in create
    assert sim.network not in create, "sharing a namespace and joining a network differ"
    assert create[create.index("-w") + 1] == CONTAINER_REPO


def test_the_containerised_scenario_runs_the_containers_interpreter():
    """An image in the config picks the interpreter that exists in that image."""
    assert scenario_runner_command(_config(image="cadet:sim"))[0] == CONTAINER_PYTHON


def test_the_host_scenario_still_runs_this_interpreter():
    """Without an image nothing changes: the default is this process's."""
    import sys

    assert scenario_runner_command(_config())[0] == sys.executable


def test_the_client_distribution_is_mounted_when_named(tree, tmp_path, monkeypatch):
    """ScenarioRunner needs `agents`, which the egg does not carry.

    The egg provides the `carla` module and the image bakes it, which is why
    this looked unnecessary; `agents` ships in the PythonAPI tree beside it and
    ScenarioRunner imports it for its own behaviours, so a scenario without it
    fails at import - which reads as a startup timeout.
    """
    from cadet.sim.containers import CONTAINER_CARLA_API
    from cadet.sim.scenario_launcher import (
        API_HOST_PATH_ENV,
        API_PACKAGE_SUBDIR,
        scenario_pythonpath,
    )

    api = tmp_path / "carla-PythonAPI"
    (api / API_PACKAGE_SUBDIR / "agents").mkdir(parents=True)
    monkeypatch.setenv(API_HOST_PATH_ENV, str(api))

    config = _config(image="cadet:sim")
    assert (str(api), CONTAINER_CARLA_API) in scenario_mounts(config, "/repo")
    # The importable parent of `agents` is the carla subdirectory, not the root.
    path = scenario_pythonpath(config)
    assert path.split(":")[0] == CONTAINER_CARLA_API + "/" + API_PACKAGE_SUBDIR
    assert CONTAINER_REPO in path.split(":")


def test_an_unnamed_client_distribution_is_not_mounted(tree):
    """Unset means the image carries the whole distribution, so nothing is added."""
    config = _config(image="cadet:sim")
    assert len(scenario_mounts(config, "/repo")) == 2


def test_the_client_mount_does_not_hide_the_baked_egg():
    """The mount point must not be a parent of where the image bakes the egg.

    The image puts the egg at a fixed path under /opt/carla and adds it to the
    path with a .pth file. A tree mounted at /opt/carla replaces that directory,
    so the egg stops existing and the scenario fails on `import carla` with the
    client distribution mounted and apparently correct.
    """
    import os

    from cadet.sim.containers import CONTAINER_CARLA_API
    from cadet.sim.versions import CONTAINER_CARLA_EGG_DIR

    shadowed = os.path.commonpath([CONTAINER_CARLA_API, CONTAINER_CARLA_EGG_DIR])
    assert shadowed != CONTAINER_CARLA_API
