"""Offline check of the parity-L1 recorder format + replay harness (no CARLA).

Writes synthetic detection/control recordings in the exact shape the flat app
produces, then replays them through the pipeline stages via the parity script's
functions and asserts the decision frame and every control command match. This
validates the record/replay I/O contract; the real check feeds recorded CARLA runs.
"""

import os
import sys

from cadet.pipeline.record import JsonlRecorder
from cadet.pipeline.stages.braking import BrakeControlStage, frame_triggers_brake

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"
    ),
)
import braking_l1_parity as L1  # noqa: E402

TARGET_IDS = [2]
CTRL_PARAMS = {
    "ego_vehicle": "audi.tt",
    "ego_speed": "40mph",
    "controllers": {
        "audi.tt": {"target_decel": 7.0, "max_decel": 7.3, "kp": 0.2, "max_ramp": 0.05}
    },
}


def _write_detections(path):
    rec = JsonlRecorder(path)
    rec.record(
        meta=True,
        target_ids=TARGET_IDS,
        roi_cx_min=300,
        roi_cx_max=500,
        conf_thresh=0.3,
        detection_threshold=3,
    )
    hit = [[2, 0.9, 400.0]]
    miss = [[2, 0.9, 700.0]]
    seq = [hit, hit, miss, hit, hit, hit, miss]  # three consecutive hits at frames 3,4,5
    consec = 0
    for fid, boxes in enumerate(seq):
        found = frame_triggers_brake([tuple(b) for b in boxes], TARGET_IDS, 300, 500, 0.3)
        consec = consec + 1 if found else 0
        rec.record(frame_id=fid, boxes=boxes, consecutive=consec, brake=bool(consec >= 3))
    rec.close()


def _write_control(path):
    """Generate recorded control by running the (verbatim) control stage forward."""
    rec = JsonlRecorder(path)
    stage = BrakeControlStage("gen", CTRL_PARAMS)
    ticks = [
        (18.0, 95.0, 0.0, False),
        (18.0, 89.0, 0.1, False),
        (18.0, 80.0, 0.2, False),
        (17.0, 75.0, 0.3, True),
        (10.0, 72.0, 0.4, True),
        (3.0, 70.0, 0.5, True),
        (0.05, 69.0, 0.6, True),
    ]
    from cadet.pipeline.messages import Message

    for speed, loc_y, t, bp in ticks:
        m = Message()
        m.speed_mps, m.loc_y, m.t, m.brake_pending = speed, loc_y, t, bp
        cmd = stage.process(m)
        rec.record(
            t=t,
            speed_mps=speed,
            loc_y=loc_y,
            brake_pending=bp,
            throttle=cmd.throttle,
            steer=cmd.steer,
            brake=cmd.brake,
        )
    rec.close()


def test_l1_replay_decisions_and_control_match(tmp_path):
    det = str(tmp_path / "detections.jsonl")
    ctrl = str(tmp_path / "control.jsonl")
    _write_detections(det)
    _write_control(ctrl)

    flat_brake, pipeline_brake, n = L1.replay_decisions(det)
    assert flat_brake == pipeline_brake == 5 and n == 7

    n_ctrl, diffs = L1.replay_control(ctrl, CTRL_PARAMS)
    assert n_ctrl == 7 and diffs == []


def test_l1_replay_detects_a_divergence(tmp_path):
    ctrl = str(tmp_path / "control.jsonl")
    _write_control(ctrl)
    # Corrupt one recorded command: replay must flag the mismatch.
    lines = open(ctrl).read().splitlines()
    import json

    rec = json.loads(lines[4])
    rec["brake"] = rec["brake"] + 0.5
    lines[4] = json.dumps(rec, sort_keys=True)
    open(ctrl, "w").write("\n".join(lines) + "\n")
    n_ctrl, diffs = L1.replay_control(ctrl, CTRL_PARAMS)
    assert len(diffs) == 1 and diffs[0][0] == 4
