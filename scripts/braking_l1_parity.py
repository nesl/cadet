#!/usr/bin/env python3
"""Parity level 1: replay a recorded flat braking run through the pipeline stages.

Reads a recording made by the flat app with ``$CADET_RECORD_DIR`` set
(``detections.jsonl`` from the server, ``control.jsonl`` from the client) and
replays it offline (no CARLA) through the ported ``roi_consecutive`` and
``brake_controller`` stages, asserting every decision and command matches the flat
run exactly. Reports the first diverging (frame/tick, stage) on any mismatch.

Usage:
    python scripts/braking_l1_parity.py --dir <record_dir> --config <flat.yaml>
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cadet.pipeline.messages import Detections, Message  # noqa: E402
from cadet.pipeline.record import read_jsonl  # noqa: E402
from cadet.pipeline.stages.braking import BrakeControlStage, RoiConsecutiveStage  # noqa: E402


def replay_decisions(detfile):
    """Replay recorded detections through roi_consecutive; return (flat, pipeline) brake frames."""
    recs = read_jsonl(detfile)
    meta = recs[0]
    frames = recs[1:]
    stage = RoiConsecutiveStage(
        "decide",
        {
            "target_ids": meta["target_ids"],
            "roi": {"cx_min": meta["roi_cx_min"], "cx_max": meta["roi_cx_max"]},
            "conf_thresh": meta["conf_thresh"],
            "detection_threshold": meta["detection_threshold"],
        },
    )
    pipeline_brake = None
    for r in frames:
        boxes = [(int(c), float(cf), float(cx)) for c, cf, cx in r["boxes"]]
        out = stage.process(Detections(detections=boxes, frame_id=r["frame_id"]))
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
    from examples.braking.flat.app import BrakingParams

    cfg = build_config(load_config_data(config_path))
    params = BrakingParams.model_validate(cfg.app.params)
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
    ap.add_argument("--config", required=True, help="the flat braking config (for control params)")
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
