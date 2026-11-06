"""Tests for the defaults that make an unconfigured run the original's run.

Each default here was a deviation at some point, so each is pinned: a default
that drifts silently is the failure the fidelity rule exists to prevent.
"""

import json
import os
import sys
from types import SimpleNamespace

import yaml

from cadet.config.schema import Camera, Experiment, Node
from cadet.orchestrator import lifecycle
from cadet.sim.scenario_command import ORIGINAL_TIMEOUT_S, scenario_runner_command

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BRAKING_CONFIGS = [
    os.path.join(REPO, "examples", "braking", flavour, name)
    for flavour in ("flat", "pipeline")
    for name in ("atspeed_braking.yaml", "openloop_braking.yaml")
]


def test_the_defaults_are_the_originals():
    experiment = Experiment(name="x", app="y")
    # The original judged a run by the client's exit code alone.
    assert experiment.require_output is False
    # The original slept five seconds between experiments.
    assert experiment.inter_run_pause_s == 5.0
    # The original re-ran every experiment; resume is the deviation.
    assert experiment.resume is False
    # The original started the client after the scenario.
    assert experiment.client_start == "after_scenario"


class _Scenario:
    name = "AtSpeedBraking_1"
    definition = "examples/braking/flat/scenarios/at_speed_braking_scenario.py"
    config_file = "examples/braking/flat/scenarios/AtSpeed.xml"


def _scenario_config(**carla):
    """Return a real config with a scenario, built through the schema.

    Built by the schema rather than by hand: a stand-in that carried its own
    ``sim`` attribute is how this test went on passing after the key fold moved
    the ScenarioRunner tree under ``source.scenario``, while every config the
    loader could actually produce returned no command at all.
    """
    from cadet.config.schema import CarlaSource

    return SimpleNamespace(
        source=CarlaSource(
            carla=dict({"host": "10.0.0.5", "port": 2200}, **carla),
            scenario={
                "name": "AtSpeedBraking_1",
                "definition": "examples/braking/flat/scenarios/at_speed_braking_scenario.py",
                "config_file": "examples/braking/flat/scenarios/AtSpeed.xml",
                "runner": {"path": "/opt/scenario_runner"},
            },
        )
    )


def test_the_scenario_command_is_the_original_script():
    # run_scenario.sh is three lines and ends:
    #   python3 scenario_runner.py --scenario $1 --reloadWorld --timeout 10000 --output
    argv = scenario_runner_command(_scenario_config())
    assert argv[1] == "/opt/scenario_runner/scenario_runner.py"
    assert "--reloadWorld" in argv, "a run that does not reload keeps the last run's actors"
    assert "--output" in argv
    assert argv[argv.index("--timeout") + 1] == str(ORIGINAL_TIMEOUT_S)
    assert argv[argv.index("--scenario") + 1] == "AtSpeedBraking_1"
    # The one addition to the original's flags: it hard-coded localhost:2000 by
    # omitting these, and a scenario on a different simulator than the client
    # spawns the ego where the client cannot see it. Measured: the client then
    # waits out its ego timeout and the run produces nothing.
    assert argv[argv.index("--host") + 1] == "10.0.0.5"
    assert argv[argv.index("--port") + 1] == "2200"


def test_no_scenario_means_no_hook():
    """A source with no scenario yields no hook, which is what a replay wants."""
    bare = SimpleNamespace(source=SimpleNamespace(scenario=None))
    assert scenario_runner_command(bare) is None


def test_no_runner_tree_means_no_hook():
    """A scenario named with no tree to run it from is not a command either."""
    config = _scenario_config()
    config.source.scenario.runner = None
    assert scenario_runner_command(config) is None


def test_the_interpreter_is_the_callers_to_name():
    """The default is this process's; a scenario container passes its own.

    Pinned because the command used to hard-code ``sys.executable``, which is
    the one interpreter that is certainly absent from the scenario container.
    """
    from cadet.sim.scenario_command import CONTAINER_PYTHON

    assert scenario_runner_command(_scenario_config())[0] == sys.executable
    argv = scenario_runner_command(_scenario_config(), python=CONTAINER_PYTHON)
    assert argv[0] == CONTAINER_PYTHON == "python3"
    # The tree is unchanged by the interpreter: it is a container path already.
    assert argv[1] == "/opt/scenario_runner/scenario_runner.py"


class _Experiment:
    def __init__(self, require_output):
        self.require_output = require_output


class _RunConfig:
    def __init__(self, require_output):
        self.experiment = _Experiment(require_output)


def _write_outcome(directory):
    from cadet.roles.client import client_outcome_file

    with open(os.path.join(directory, client_outcome_file("client")), "w") as handle:
        json.dump({"ok": False, "detail": "client transmitted no frames"}, handle)


def test_no_output_does_not_fail_a_run_by_default(tmp_path):
    _write_outcome(str(tmp_path))
    lines = []
    failures = lifecycle.client_output_failures(
        str(tmp_path), _RunConfig(require_output=False), log=lines.append
    )
    assert failures == []
    # Reported even though it does not fail the run: the count is the client's
    # own and worth having.
    assert any("no output" in line for line in lines), lines


def test_no_output_fails_a_run_when_asked_for(tmp_path):
    _write_outcome(str(tmp_path))
    failures = lifecycle.client_output_failures(
        str(tmp_path), _RunConfig(require_output=True), log=lambda *_a: None
    )
    assert failures == ["client transmitted no frames"]


def test_the_app_side_defaults_are_the_originals():
    # The original client left the camera free-running at the asynchronous
    # server's own rate; a fixed sensor_tick is the deviation.
    assert Camera().fixed_rate is False
    node = Node(name="server", role="server", host="127.0.0.1", port=10000)
    # The original server held the reply back with a real time.sleep.
    assert node.server_delay_mode == "blocking"
    # The original read every frame from one unbounded FIFO.
    assert node.enforce_queue_policy is False

    from examples.braking.flat.app import BrakingParams

    params = BrakingParams()
    # The original sent a control command every client loop iteration.
    assert params.control_rate == "per_loop"
    # The original wrote an annotated JPEG per processed frame.
    assert params.save_annotated_frames is True
    # The original rendered a window every control tick and saved a PNG per
    # captured frame, which is what set its client loop rate.
    assert params.client_visualizer is True

    from cadet.pipeline.app import PipelineAppParams

    assert PipelineAppParams().client_visualizer is True

    from cadet.pipeline.stages.braking import YoloDetectorStage

    # The staged app has the same default, read from the stage's own params.
    assert YoloDetectorStage(name="detect")._save_annotated is True


def test_the_braking_examples_turn_the_warm_up_off():
    # models.warmup defaults on, which is the deviation: the original paid the
    # first frame's JIT cost inside the measured run. Every braking example
    # therefore says so, and the measured brake time depends on it.
    for path in BRAKING_CONFIGS:
        with open(path) as handle:
            config = yaml.safe_load(handle)
        assert config["models"]["warmup"] is False, path


def test_a_live_ego_changes_nothing_about_when_the_run_ends():
    """The ending a normal run reaches is the one it always reached.

    The client ends when the ego has stopped. It now also ends when the ego has
    gone from the world, which is a case it used to wait out until the active
    limit. That second path must be invisible to a run where the ego is there:
    nothing checked before the client is waiting, nothing recorded while the
    actor exists.
    """
    from examples.braking.flat.app import BrakingClientApp

    app = BrakingClientApp.__new__(BrakingClientApp)
    app._done = False
    app._ego_gone = False
    app._ego_checked_at = 0.0
    logged = []
    app._log = logged.append

    class _Source(object):
        def __init__(self, alive):
            self._alive = alive

        def ego_is_alive(self):
            return self._alive

    # A live ego: not done, nothing logged, nothing marked.
    app.source = _Source(True)
    assert app.is_done(None) is False
    assert app._ego_gone is False
    assert logged == []

    # Asked at most once a second: the simulator is not questioned on every
    # frame of the path the latency numbers come from.
    asked = []

    class _Counting(object):
        @staticmethod
        def ego_is_alive():
            asked.append(1)
            return True

    app.source = _Counting()
    app._ego_checked_at = 0.0
    for _ in range(50):
        assert app.is_done(None) is False
    assert len(asked) == 1

    # The ego having stopped is still what ends a normal run, and the new path
    # is not consulted once it has.
    app._done = True
    app.source = _Source(False)
    assert app.is_done(None) is True
    assert app._ego_gone is False
    assert logged == []


def test_an_ego_removed_before_it_stopped_ends_the_run_and_is_not_a_success():
    """A scenario that ends first leaves the second half of the measurement absent.

    Measured on the distributed tier: the server issued the brake, the scenario
    exited and destroyed the ego, and the client waited 120 s for a stop that
    could not arrive. The run now ends there, and says why rather than
    reporting a brake with no stop as a success.
    """
    from examples.braking.flat.app import BrakingApp, BrakingClientApp

    app = BrakingClientApp.__new__(BrakingClientApp)
    app._done = False
    app._ego_gone = False
    app._ego_checked_at = 0.0
    logged = []
    app._log = logged.append

    class _Gone(object):
        @staticmethod
        def ego_is_alive():
            return False

    app.source = _Gone()
    assert app.is_done(None) is True
    assert app._ego_gone is True
    assert any("no longer in the world" in line for line in logged)

    ok, detail = BrakingApp.client_success(app)
    assert ok is False
    assert "removed from the world" in detail


def test_the_ego_is_looked_up_in_the_world_not_asked_about_itself():
    """A handle whose actor the server destroyed goes on answering.

    `is_alive` stayed true and `get_location` kept returning the last position,
    so a client trusting either kept braking a vehicle that was no longer
    there, and in this version a lookup by id keeps answering for it too. The
    world's own actor list is asked instead, which is the only answer the
    simulator itself stands behind.
    """
    from cadet.sources.carla import CarlaSource

    source = CarlaSource.__new__(CarlaSource)

    class _StaleHandle(object):
        id = 1639
        is_alive = True  # the lie this exists to ignore

    class _World(object):
        def __init__(self, known):
            self._known = known

        def get_actors(self):
            return [_StaleHandle() for actor_id in self._known if actor_id == 1639]

    source.vehicle = _StaleHandle()
    source._world = _World({1639})
    assert source.ego_is_alive() is True

    # The server has destroyed it: the handle still says otherwise, the world
    # does not, and the world is what counts.
    source._world = _World(set())
    assert source.ego_is_alive() is False

    # A world that cannot be asked is not evidence that the ego has gone.
    class _Unreachable(object):
        @staticmethod
        def get_actors():
            raise RuntimeError("time-out while waiting for the simulator")

    source._world = _Unreachable()
    assert source.ego_is_alive() is True


def test_the_ego_is_found_by_role_name_as_the_original_found_it():
    """The original matched `role_name == 'hero'` and nothing else.

    `wait_for_ego_vehicle` in the reference client's simulation module looped
    over `world.get_actors().filter('vehicle.*')` and returned the actor whose
    `role_name` attribute was `hero` - which is what ScenarioRunner tags an ego
    with when it spawns one. So that is the default here, and a config that
    names no role name still behaves as the original did.
    """
    from cadet.config.schema import Scenario
    from cadet.sources.carla import CarlaSource

    assert Scenario().ego_role_name == "hero"

    source = CarlaSource.__new__(CarlaSource)
    source._ego_role_name = "hero"
    source._ego_name = "audi.tt"
    source._warned_blueprint_match = False
    source._log = None

    class _Actor(object):
        def __init__(self, role, ident):
            self.attributes = {"role_name": role}
            self.id = ident

    wanted = _Actor("hero", 7)

    class _Actors(list):
        def filter(self, _pattern):
            return list(self)

    class _World(object):
        def __init__(self, actors):
            self._actors = _Actors(actors)

        def get_actors(self):
            return self._actors

    # The ego is picked out of a world holding other vehicles of its own model.
    source._world = _World([_Actor("scenario", 5), wanted, _Actor("autopilot", 9)])
    assert source._find_ego() is wanted

    # Not there yet is not an error: the caller polls.
    source._world = _World([_Actor("autopilot", 9)])
    assert source._find_ego() is None


def test_matching_the_ego_by_blueprint_is_a_deviation_that_says_so():
    """Reached only by emptying the role name, and it warns once.

    A blueprint is not unique, so which actor it returns depends on what else
    is in the world and on when the question is asked. Measured on the
    distributed tier: the client attached to the wrong vehicle, its camera was
    elsewhere on the map and the ego was never driven, which reads as a working
    run rather than as a misattachment. The warning is what makes it visible.
    """
    from cadet.sources.carla import CarlaSource

    logged = []
    source = CarlaSource.__new__(CarlaSource)
    source._ego_role_name = ""  # emptied on purpose: the opt-in
    source._ego_name = "audi.tt"
    source._warned_blueprint_match = False
    source._log = logged.append

    class _Actor(object):
        id = 11
        attributes = {"role_name": "whatever"}

    class _World(object):
        @staticmethod
        def get_actors():
            class _Match(list):
                def filter(self, pattern):
                    assert pattern == "vehicle.audi.tt", pattern
                    return [_Actor()]

            return _Match()

    source._world = _World()
    assert isinstance(source._find_ego(), _Actor)
    assert len(logged) == 1
    assert "blueprint" in logged[0]
    assert "role_name 'hero'" in logged[0]

    # Said once, not once per poll: the caller polls until the ego appears.
    source._find_ego()
    source._find_ego()
    assert len(logged) == 1
