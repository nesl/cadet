#!/usr/bin/env python3
"""Parity level 1: replay a recorded flat V2V run through the pipeline stages.

Reads a recording made by the flat V2V app with ``$CADET_RECORD_DIR`` set
(``detections.jsonl`` from the server, ``control.jsonl`` from the client) and
replays it offline (no CARLA) through the ported ``v2v_ttc_gate`` and the shared
``brake_controller`` stages, asserting every decision and command matches the flat
run exactly. The V2V gate is time-dependent, so each recorded frame carries its
``best_z`` and ``sim_time`` and the gate is rerun on exactly those values.

Usage:
    python scripts/v2v_l1_parity.py --dir <record_dir> --config <v2v.yaml>
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cadet.pipeline.messages import Message  # noqa: E402
from cadet.pipeline.record import read_jsonl  # noqa: E402
from cadet.pipeline.stages.braking import BrakeControlStage  # noqa: E402
from cadet.pipeline.stages.v2v import V2VLeadRange, V2VTtcGateStage  # noqa: E402


def replay_decisions(detfile):
    """Replay recorded lead ranges through v2v_ttc_gate; return (flat, pipeline) brake frames."""
    recs = read_jsonl(detfile)
    meta = recs[0]
    frames = recs[1:]
    stage = V2VTtcGateStage(
        "decide",
        {
            "ttc_thresh_s": meta["ttc_thresh_s"],
            "k_ttc_frames": meta["k_ttc_frames"],
            "alpha_ttc": meta["alpha_ttc"],
            "d_thresh_m": meta["d_thresh_m"],
            "d_gate_after_s": meta["d_gate_after_s"],
        },
    )
    pipeline_brake = None
    for r in frames:
        msg = V2VLeadRange(best_z=r["best_z"], frame_id=r["frame_id"])
        msg.sim_time = r["sim_time"]
        out = stage.process(msg)
        if out is not None and pipeline_brake is None:
            pipeline_brake = r["frame_id"]
    flat_brake = next((r["frame_id"] for r in frames if r["brake"]), None)
    return flat_brake, pipeline_brake, len(frames)


def replay_control(ctrlfile, control_params):
    """Replay recorded ego telemetry through brake_controller; return (n, diffs)."""
    recs = read_jsonl(ctrlfile)
    stage = BrakeControlStage("control", control_params)
    diffs = []
    for i, r in enumerate(recs):
        telem = Message()
        telem.speed_mps = r["speed_mps"]
        telem.loc_y = r["loc_y"]
        telem.t = r["t"]
        telem.brake_pending = r["brake_pending"]
        cmd = stage.process(telem)
        got = (round(cmd.throttle, 9), round(cmd.steer, 9), round(cmd.brake, 9))
        exp = (round(r["throttle"], 9), round(r["steer"], 9), round(r["brake"], 9))
        if got != exp:
            diffs.append((i, r.get("t"), exp, got))
    return len(recs), diffs


def _control_params_from_config(config_path):
    from cadet.config.loader import build_config, load_config_data
    from examples.v2v.app import V2VParams

    cfg = build_config(load_config_data(config_path))
    params = V2VParams.model_validate(cfg.app.params)
    controllers = {k: v.model_dump() for k, v in params.controllers.items()}
    return {
        "ego_vehicle": cfg.source.scenario.ego_vehicle,
        "ego_speed": cfg.source.scenario.ego_speed,
        "controllers": controllers,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dir", required=True, help="record directory ($CADET_RECORD_DIR of the flat run)"
    )
    ap.add_argument("--config", required=True, help="the flat V2V config (for control params)")
    args = ap.parse_args()

    detfile = os.path.join(args.dir, "detections.jsonl")
    ctrlfile = os.path.join(args.dir, "control.jsonl")

    flat_brake, pipe_brake, n_frames = replay_decisions(detfile)
    dec_ok = flat_brake == pipe_brake
    print(
        "decisions: frames={} flat_brake_frame={} pipeline_brake_frame={} -> {}".format(
            n_frames, flat_brake, pipe_brake, "MATCH" if dec_ok else "MISMATCH"
        )
    )

    n_ctrl, diffs = replay_control(ctrlfile, _control_params_from_config(args.config))
    ctrl_ok = not diffs
    print(
        "control: ticks={} mismatches={} -> {}".format(
            n_ctrl, len(diffs), "MATCH" if ctrl_ok else "MISMATCH"
        )
    )
    if diffs:
        i, t, exp, got = diffs[0]
        print("  first divergence: tick {} (t={}) expected {} got {}".format(i, t, exp, got))

    return 0 if (dec_ok and ctrl_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
