"""Offline check of the V2V parity-L1 recorder format + replay harness (no CARLA).

Writes synthetic lead-range/control recordings in the exact shape the flat V2V app
produces, then replays them through the pipeline stages via the parity script and
asserts the brake frame and every control command match. The "flat" decisions come
from the *example's* V2VBrakeGate and are replayed through the *stage's* gate,
so this also proves the two verbatim copies of the gate agree (the port is faithful).
"""

import os
import sys

from cadet.pipeline.record import JsonlRecorder
from cadet.pipeline.stages.braking import BrakeControlStage
from examples.v2v.app import V2VBrakeGate

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"
    ),
)
import v2v_l1_parity as L1  # noqa: E402

GATE_PARAMS = {
    "ttc_thresh_s": 2.0,
    "k_ttc_frames": 2,
    "alpha_ttc": 0.30,
    "d_thresh_m": 24.0,
    "d_gate_after_s": 12.0,
}

CTRL_PARAMS = {
    "ego_vehicle": "audi.tt",
    "ego_speed": "20mph",
    "controllers": {
        "audi.tt": {"target_decel": 7.0, "max_decel": 7.3, "kp": 0.2, "max_ramp": 0.05}
    },
}

# (sim_time, best_z): a lead closing after the distance gate arms (sim_time > 12);
# includes a no-detection frame (None) to exercise the "advance no state" path.
SEQ = [
    (11.0, 40.0),
    (11.5, None),
    (12.0, 35.0),
    (12.5, 30.0),
    (13.0, 22.0),  # within 24 m, closing, sim_time > 12 -> distance brake here
    (13.5, 18.0),
]


def _write_detections(path):
    rec = JsonlRecorder(path)
    rec.record(meta=True, target_ids=[2], roi_cx_min=300, roi_cx_max=500, conf_thresh=0.3,
               **GATE_PARAMS)
    gate = V2VBrakeGate(**GATE_PARAMS)
    for fid, (sim_time, best_z) in enumerate(SEQ):
        brake = gate.update(best_z, sim_time, sim_time)
        rec.record(frame_id=fid, best_z=best_z, sim_time=sim_time, brake=bool(brake))
    rec.close()


def _write_control(path):
    rec = JsonlRecorder(path)
    stage = BrakeControlStage("gen", CTRL_PARAMS)
    ticks = [
        (9.0, 95.0, 0.0, False),
        (9.0, 89.0, 0.1, False),
        (9.0, 80.0, 0.2, False),
        (8.0, 75.0, 0.3, True),
        (5.0, 72.0, 0.4, True),
        (2.0, 70.0, 0.5, True),
        (0.05, 69.0, 0.6, True),
    ]
    from cadet.pipeline.messages import Message

    for speed, loc_y, t, bp in ticks:
        m = Message()
        m.speed_mps, m.loc_y, m.t, m.brake_pending = speed, loc_y, t, bp
        cmd = stage.process(m)
        rec.record(t=t, speed_mps=speed, loc_y=loc_y, brake_pending=bp,
                   throttle=cmd.throttle, steer=cmd.steer, brake=cmd.brake)
    rec.close()


def test_v2v_l1_replay_decisions_and_control_match(tmp_path):
    det = str(tmp_path / "detections.jsonl")
    ctrl = str(tmp_path / "control.jsonl")
    _write_detections(det)
    _write_control(ctrl)

    flat_brake, pipeline_brake, n = L1.replay_decisions(det)
    assert flat_brake == pipeline_brake == 4 and n == len(SEQ)

    n_ctrl, diffs = L1.replay_control(ctrl, CTRL_PARAMS)
    assert n_ctrl == 7 and diffs == []


def test_v2v_l1_replay_detects_a_divergence(tmp_path):
    ctrl = str(tmp_path / "control.jsonl")
    _write_control(ctrl)
    lines = open(ctrl).read().splitlines()
    import json

    rec = json.loads(lines[4])
    rec["brake"] = rec["brake"] + 0.5
    lines[4] = json.dumps(rec, sort_keys=True)
    open(ctrl, "w").write("\n".join(lines) + "\n")
    n_ctrl, diffs = L1.replay_control(ctrl, CTRL_PARAMS)
    assert len(diffs) == 1 and diffs[0][0] == 4
