"""Tests for the simulator-side container launch.

Every case here is a bug that shipped unnoticed because nothing constructed
:class:`cadet.sim.containers.SimSide`: the launch was written, merged and never
run, so a renamed config key and a mis-numbered GPU index sat in it as syntax
errors would have sat in an unimported module. The docker calls are captured
rather than made, which is what lets the command line be asserted on a machine
with no daemon.
"""

import pytest

from cadet.config.schema import CarlaSource
from cadet.sim import containers


@pytest.fixture
def captured(monkeypatch):
    """Capture the docker argv SimSide would run, and make every call succeed."""
    calls = []

    class _Result:
        returncode = 0
        stdout = "true"
        stderr = ""

    def _docker(argv, docker="docker", check=False):
        calls.append(list(argv))
        return _Result()

    monkeypatch.setattr(containers, "_docker", _docker)
    monkeypatch.setattr(containers, "docker_available", lambda docker="docker": True)
    # No reusable server, so these cases exercise the launch. Without this the
    # result depends on whether anything happens to hold the port on the machine
    # running the tests, which is how a shared default turns a unit test into a
    # question about the host.
    monkeypatch.setattr(containers, "port_open", lambda host, port, timeout_s=1.0: False)
    return calls


def _config(**carla):
    """A CARLA source that declares a launch, built through the schema."""
    settings = dict({"image": "carlasim/carla:0.9.10.1"}, **carla)
    return type("_C", (), {"source": CarlaSource(carla=settings)})()


def _run_argv(calls):
    """The `docker run` call among the captured ones."""
    return next(argv for argv in calls if argv[0] == "run")


def test_the_server_is_told_the_configured_port(captured):
    """The RPC port on the command line is the one the config set.

    This read ``carla.rpc_port`` until the first launch was attempted - a name
    the key fold had replaced with ``port`` - so the launch raised
    AttributeError before starting anything.
    """
    containers.SimSide(_config(port=2345), "abc123", log=lambda *a: None).start()
    argv = _run_argv(captured)
    assert "-carla-rpc-port=2345" in argv[-1]


def test_the_port_defaults_to_the_originals(captured):
    """An unconfigured launch uses the port the original's client dialled."""
    from cadet.config.schema import CarlaSettings

    containers.SimSide(_config(), "abc123", log=lambda *a: None).start()
    argv = _run_argv(captured)
    assert "-carla-rpc-port={}".format(CarlaSettings().port) in argv[-1]


def test_the_gpu_index_is_the_containers_not_the_hosts(captured):
    """`--gpus device=N` renumbers that device to 0 inside the container.

    Measured: ``--gpus device=1`` with ``CUDA_VISIBLE_DEVICES=1`` gives a
    device_count of 0, because the container has exactly one device and it is
    index 0. Passing the host index hides the only GPU the container has, and
    the symptom is a CPU-speed run rather than an error.
    """
    containers.SimSide(_config(gpu_index=1), "abc123", log=lambda *a: None).start()
    argv = _run_argv(captured)
    assert "device=1" in argv, "the host index selects which device is passed in"
    assert "CUDA_VISIBLE_DEVICES=0" in argv, "inside the container it is index 0"
    assert "CUDA_VISIBLE_DEVICES=1" not in argv


def test_no_image_is_an_attach_not_a_launch(captured):
    """Without an image the simulator is somebody else's, so nothing is started."""
    config = type("_C", (), {"source": CarlaSource()})()
    assert containers.carla_launch_settings(config) is None
    containers.SimSide(config, "abc123", log=lambda *a: None).start()
    assert captured == []


def test_the_published_ports_cover_the_servers_span(captured):
    """CARLA listens on the RPC port and the two above it; all three are published."""
    containers.SimSide(_config(port=2345), "abc123", log=lambda *a: None).start()
    argv = _run_argv(captured)
    published = [argv[i + 1] for i, item in enumerate(argv) if item == "-p"]
    assert published == [
        "127.0.0.1:{0}:{0}".format(port) for port in (2345, 2346, 2347)
    ]


# --- reuse: one server for a sweep, not one per cell ------------------------
# A CARLA start is tens of seconds, and the original ran one long-lived server
# for a whole study, clearing the previous run's actors with --reloadWorld and a
# pause between experiments. So a run reuses a server it finds, and these fix
# what "finds" means and what reuse must never silently do.


def test_a_running_server_is_reused_rather_than_relaunched(captured, monkeypatch):
    """The point of the default: the second cell of a sweep starts nothing."""
    monkeypatch.setattr(containers, "port_open", lambda host, port, timeout_s=1.0: True)
    sim = containers.SimSide(_config(), "abc123", log=lambda *a: None).start()

    assert not [argv for argv in captured if argv[0] == "run"]
    assert not [argv for argv in captured if argv[:2] == ["network", "create"]]
    assert sim._state == "started"


def test_a_server_that_is_up_but_not_answering_is_replaced(captured):
    """A container whose server died inside it is not reusable, and nothing else
    is going to remove it."""
    containers.SimSide(_config(), "abc123", log=lambda *a: None).start()

    names = [argv for argv in captured if argv[:2] == ["rm", "-f"]]
    assert names and names[0][2].endswith("-carla")
    assert _run_argv(captured)


def test_a_reused_server_is_named_after_the_launch_not_the_run():
    """Two runs reuse one server exactly when they would have started the same
    one, so the name is a fingerprint of the launch settings."""
    first = containers.SimSide(_config(), "aaaaaaaaaa-r0", log=lambda *a: None)
    second = containers.SimSide(_config(), "bbbbbbbbbb-r3", log=lambda *a: None)
    other_port = containers.SimSide(_config(port=2345), "aaaaaaaaaa-r0", log=lambda *a: None)

    assert first.carla_name == second.carla_name
    assert first.network == second.network
    assert other_port.carla_name != first.carla_name
    # The scenario driver *is* the run, and a stale one is never reusable.
    assert first.scenario_name != second.scenario_name


def test_per_run_puts_the_run_back_in_the_name():
    """What a run asks for when it must not inherit any simulator state."""
    sim = containers.SimSide(_config(per_run=True), "aaaaaaaaaa-r0", log=lambda *a: None)

    assert not sim.shared
    assert "aaaaaaaaaa-r0" in sim.carla_name
    assert "aaaaaaaaaa-r0" in sim.network


def test_stopping_a_run_leaves_a_reused_server_and_its_network(captured, monkeypatch):
    """The next run expects to find them; `cadet down` is what removes them."""
    monkeypatch.setattr(containers, "port_open", lambda host, port, timeout_s=1.0: False)
    sim = containers.SimSide(_config(), "abc123", log=lambda *a: None).start()
    del captured[:]
    result = sim.stop()

    assert result["removed"] == []
    assert not [argv for argv in captured if argv[0] == "rm"]
    assert not [argv for argv in captured if argv[:2] == ["network", "rm"]]


def test_a_per_run_server_is_torn_down_with_its_run(captured):
    """The opposite setting, and the behaviour that used to be the only one."""
    sim = containers.SimSide(_config(per_run=True), "abc123", log=lambda *a: None).start()
    result = sim.stop()

    assert "container:" + sim.carla_name in result["removed"]
    assert "network:" + sim.network in result["removed"]


def test_cadet_down_recognises_the_reusable_simulators_objects():
    """Reuse makes a deliberate leftover, and nothing else would remove it."""
    from cadet.orchestrator.recovery import OWNED_SIM_OBJECT_RE

    sim = containers.SimSide(_config(), "abc123", log=lambda *a: None)
    assert OWNED_SIM_OBJECT_RE.match(sim.carla_name)
    assert OWNED_SIM_OBJECT_RE.match(sim.network)
    # And a hand-started server of a similar name stays out of the sweep.
    assert not OWNED_SIM_OBJECT_RE.match("cadet-sim-carla")
    assert not OWNED_SIM_OBJECT_RE.match("carla-server")
