#!/usr/bin/env python3
"""GPU-contention load worker: run YOLO over a folder of images, repeatedly.

Ported from ``carla_wrapper/energy_monitoring/multi-tenancy/run_yolo_extra_server.py``
(the original's "extra client"): load the model on the GPU, warm it up, then run
inference on every PNG in ``--images`` in sorted order. The original made a single
pass; a V2I run lasts far longer than one pass, so this loops the folder until
``--duration`` seconds have elapsed (or forever when ``--duration`` is 0), which is
what ``scripts/multi_tenant.py`` needs to hold GPU pressure across a whole run. The
per-image inference call (``model(img_path)``) is unchanged from the original, so one
worker's GPU footprint matches the original's extra client exactly.

A worker self-terminates when ``--duration`` elapses and also exits cleanly on
SIGTERM/SIGINT, so the launcher never has to kill it. It produces no output (it
exists only to occupy the GPU); timing of the measured process is the main server's
job (``run_yolo_main_server``), left to the standalone benchmark.

Usage:
    python scripts/gpu_load_worker.py --model yolo11x.pt --images <dir> --duration 40
"""

import argparse
import signal
import time
from pathlib import Path

_STOP = False


def _on_signal(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument(
        "--duration", type=float, default=0.0,
        help="seconds to keep loading (0 = until signalled)",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    import torch
    from ultralytics import YOLO

    model = YOLO(args.model).to(args.device)
    model(torch.zeros(1, 3, 640, 640).to(args.device))  # warm-up, as the original

    images = sorted(Path(args.images).glob("*.png"))
    if not images:
        raise RuntimeError("No images found in {}".format(args.images))

    deadline = (time.monotonic() + args.duration) if args.duration > 0 else None
    while not _STOP:
        for img_path in images:
            if _STOP or (deadline is not None and time.monotonic() >= deadline):
                return
            _ = model(img_path)  # identical to the original extra client's call


if __name__ == "__main__":
    main()
