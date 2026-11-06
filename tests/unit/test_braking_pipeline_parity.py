"""Offline exactness: braking pipeline stages == flat braking logic (parity L1).

The decision and control stages in ``cadet.pipeline.stages.braking`` are ported
verbatim from the flat braking app. These tests drive identical input sequences
through the flat logic and through the stages and assert the outputs match exactly
(same brake frame, same per-tick commands + flags) — the stage-level form of
parity level 1, with no CARLA. Recording real CARLA runs and replaying them is
the higher-fidelity form of the same check.
"""

import random

from cadet.pipeline.messages import Detections
from cadet.pipeline.stages.braking import (
    SPEED_KICK_Y,
    STOP_SPEED_MPS,
    BrakeControlStage,
    ControlInput,
    RoiConsecutiveStage,
    frame_triggers_brake,
    target_speed_mps,
)
from examples.braking.flat.app import BrakeControllerState as FlatCtrl
from examples.braking.flat.app import frame_triggers_brake as flat_trigger
from examples.braking.flat.app import pid_throttle as flat_pid

TARGET_IDS = [2, 0]  # e.g. car, person
ROI = {"cx_min": 300, "cx_max": 500}
CONF = 0.3
THRESH = 3


def test_frame_trigger_helper_identical_to_flat():
    rng = random.Random(0)
    for _ in range(500):
        boxes = [
            (rng.randint(0, 5), rng.random(), rng.uniform(0, 800)) for _ in range(rng.randint(0, 4))
        ]
        assert frame_triggers_brake(boxes, TARGET_IDS, 300, 500, CONF) == flat_trigger(
            boxes, TARGET_IDS, 300, 500, CONF
        )


def _flat_brake_frame(box_seq):
    """Reference: the flat server's consecutive-detection logic (handle())."""
    consecutive = 0
    for i, boxes in enumerate(box_seq):
        found = flat_trigger(boxes, TARGET_IDS, 300, 500, CONF)
        consecutive = consecutive + 1 if found else 0
        if consecutive >= THRESH:
            return i
    return None


def test_roi_consecutive_stage_matches_flat_brake_frame():
    # A sequence: two hits, a miss (resets the counter), then three hits -> brake.
    hit = [(2, 0.9, 400)]  # in-ROI car above threshold
    miss = [(2, 0.9, 700)]  # out of ROI
    none = []
    box_seq = [hit, hit, miss, hit, hit, hit, hit]
    stage = RoiConsecutiveStage(
        "decide",
        {"target_ids": TARGET_IDS, "roi": ROI, "conf_thresh": CONF, "detection_threshold": THRESH},
    )
    emit_frame = None
    for i, boxes in enumerate(box_seq):
        out = stage.process(Detections(detections=boxes, frame_id=i))
        if out is not None and emit_frame is None:
            emit_frame = i
            assert out.brake is True and out.frame_id == i
            break
    assert emit_frame == _flat_brake_frame(box_seq)
    assert emit_frame == 5  # frames 3,4,5 are the three consecutive hits
    # a never-triggering sequence emits nothing
    stage2 = RoiConsecutiveStage(
        "d",
        {"target_ids": TARGET_IDS, "roi": ROI, "conf_thresh": CONF, "detection_threshold": THRESH},
    )
    assert all(
        stage2.process(Detections(detections=m, frame_id=i)) is None
        for i, m in enumerate([none, miss, none, miss])
    )


def _flat_control_oracle(ticks, target_mps, gains):
    """Reference control branch, mirroring the flat client's translate_action."""
    ctrl = FlatCtrl(gains["target_decel"], gains["max_decel"], gains["kp"], gains["max_ramp"])
    brake_engaged = stopped = set_speed_once = speed_forced = False
    out = []
    for speed, loc_y, t, brake_pending in ticks:
        throttle = steer = brake = 0.0
        engage = kick = stop = False
        if brake_pending and not brake_engaged:
            brake_engaged = True
            engage = True
        if brake_engaged:
            ctrl.update_decel(speed, t)
            brake = ctrl.step()
            throttle = 0.0
            if not stopped and speed < STOP_SPEED_MPS:
                stopped = True
                stop = True
        else:
            if loc_y < SPEED_KICK_Y and not speed_forced:
                set_speed_once = True
                speed_forced = True
            if set_speed_once:
                kick = True
                set_speed_once = False
            else:
                throttle = flat_pid(speed, target_mps)
                brake = 0.0
        out.append((throttle, steer, brake, kick, engage, stop))
    return out


def test_brake_control_stage_matches_flat_control_exactly():
    gains = {"target_decel": 7.0, "max_decel": 7.3, "kp": 0.2, "max_ramp": 0.05}
    target_mps = target_speed_mps("40mph")
    # ticks: (speed_mps, loc_y, t, brake_pending)
    # pre-brake: approach the speed-kick gate, cruise; then brake at tick 5; decelerate to a stop.
    ticks = [
        (18.0, 95.0, 0.0, False),  # above gate, cruise
        (18.0, 92.0, 0.1, False),
        (18.0, 89.0, 0.2, False),  # crosses gate -> speed kick
        (18.1, 85.0, 0.3, False),  # cruise again
        (18.0, 80.0, 0.4, False),
        (17.5, 75.0, 0.5, True),  # brake decision arrives -> engage
        (15.0, 72.0, 0.6, True),
        (10.0, 70.0, 0.7, True),
        (5.0, 69.0, 0.8, True),
        (0.5, 68.5, 0.9, True),
        (0.05, 68.5, 1.0, True),  # below stop speed -> stop
    ]
    expected = _flat_control_oracle(ticks, target_mps, gains)

    stage = BrakeControlStage(
        "control",
        {"ego_vehicle": "audi.tt", "ego_speed": "40mph", "controllers": {"audi.tt": gains}},
    )
    got = []
    for speed, loc_y, t, bp in ticks:
        cmd = stage.process(ControlInput(speed_mps=speed, loc_y=loc_y, t=t, brake_pending=bp))
        got.append(
            (cmd.throttle, cmd.steer, cmd.brake, cmd.speed_kick, cmd.engage_brake, cmd.stopped)
        )

    assert got == expected
    # sanity on the structure: exactly one speed kick, one engage, one stop
    assert sum(g[3] for g in got) == 1
    assert sum(g[4] for g in got) == 1
    assert sum(g[5] for g in got) == 1


def test_the_pipeline_client_reports_the_same_link_measurement(tmp_path):
    """A staged pipeline on a shaped link wrote no link report at all.

    The report is written when the client app reports message statistics, and
    the pipeline's client did not, so a cell with an emulated link had the
    standalone client's report and not the staged one's. That was a gap in this
    client rather than a property of staged pipelines: it keeps the same
    per-frame records the flat client does, so it can answer the same question.
    """
    from cadet.metrics.link_report import REPLY_ON_EVENT, REPLY_PER_MESSAGE
    from cadet.pipeline.app import PipelineClientApp

    app = PipelineClientApp.__new__(PipelineClientApp)

    class _Metrics(object):
        records = {
            1: {"frame_send_time": 1.0, "server_response_receive_time": 1.025},
            2: {"frame_send_time": 2.0, "server_response_receive_time": None},
            3: {"frame_send_time": None, "server_response_receive_time": None},
            4: {"frame_send_time": 4.0, "server_response_receive_time": 4.010},
        }

    app._metrics = _Metrics()

    class _Stages(object):
        stages = []

    class _Cfg(object):
        pipeline = _Stages()

    app.cfg = _Cfg()

    stats = app.message_stats()
    # Three frames were transmitted, two answered; the one never transmitted is
    # not a link measurement.
    assert stats.sent == 3
    assert stats.received == 2
    assert [round(x, 1) for x in stats.latencies_ms] == [25.0, 10.0]
    # With no terminus named, unanswered frames are the normal case.
    assert stats.reply_policy == REPLY_ON_EVENT

    # A terminus that answers every frame makes an unanswered one a measurement.
    class _Spec(object):
        impl = "decision_log"

    class _AnswersAll(object):
        answers_every_message = True

    import cadet.pipeline.registry as registry

    registry._REGISTRY.setdefault("decision_log", _AnswersAll)
    _Stages.stages = [_Spec()]
    assert app.message_stats().reply_policy == REPLY_PER_MESSAGE


def test_a_scenario_command_in_the_environment_does_not_break_an_open_loop_run(monkeypatch):
    """One shell runs many configs, and the variable is the shell's, not the run's.

    A scenario drives a simulator, and a source that is not a simulator has
    none to drive, so there is no hook to start. Refusing the run because
    CADET_SCENARIO_RUNNER_CMD happened to be set made an unrelated environment
    variable break every pipeline over replay, live, synthetic or dataset
    frames - twenty acceptance cells at once.
    """
    from cadet.config import build_config
    from cadet.core.apps import get_app

    monkeypatch.setenv("CADET_SCENARIO_RUNNER_CMD", "python scenario_runner.py --host {host}")

    config = build_config(
        {
            "experiment": {"name": "ol", "app": "pipeline", "repetitions": 1},
            "source": {"kind": "synthetic", "count": 2},
            "app": {"plugin": "pipeline", "params": {}},
            "nodes": [
                {"name": "client", "host": "localhost", "role": "client"},
                {"name": "server", "host": "127.0.0.1", "port": 15099, "role": "server"},
            ],
        }
    )
    assert list(get_app("pipeline").sim_hooks(config, [])) == []

    # A CARLA source still gets the hook the variable names.
    carla_config = build_config(
        {
            "experiment": {"name": "cl", "app": "pipeline", "repetitions": 1},
            "source": {"kind": "carla", "carla": {"host": "198.51.100.7", "port": 2000}},
            "app": {"plugin": "pipeline", "params": {}},
            "nodes": [
                {"name": "client", "host": "localhost", "role": "client"},
                {"name": "server", "host": "127.0.0.1", "port": 15099, "role": "server"},
            ],
        }
    )
    hooks = list(get_app("pipeline").sim_hooks(carla_config, []))
    assert len(hooks) == 1
    assert "198.51.100.7" in " ".join(hooks[0].argv)
