#!/usr/bin/env python3
"""Measured YOLO pass for the multi-tenant benchmark (ported, verbatim behaviour).

Ported from ``carla_wrapper/energy_monitoring/multi-tenancy/run_yolo_main_server.py``:
load the model, warm it up, then time ``model(img_path)`` over every PNG in ``--images``
and write the per-frame times to ``--out``. Run by ``scripts/multi_tenant.py`` as the
measured tenant while N background workers contend for the GPU; the inference call and
timing are unchanged from the original so the numbers are comparable.

Usage:
    python scripts/run_yolo_main_server.py --model yolo11x.pt --images <dir> --out <csv>
"""

import argparse
import csv
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    model = YOLO(args.model).to("cuda:0")
    model(torch.zeros(1, 3, 640, 640).cuda())  # warm-up

    images = sorted(Path(args.images).glob("*.png"))
    if not images:
        raise RuntimeError("No images found in {}".format(args.images))

    timings = []
    for i, img_path in enumerate(images):
        start = time.perf_counter()
        _ = model(img_path)
        duration = time.perf_counter() - start
        timings.append(duration)
        print("[INFO][Main] Frame {:03d}: {:.4f} sec".format(i + 1, duration))

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["detection_time_sec"])
        for t in timings:
            w.writerow([t])
    avg = sum(timings) / len(timings)
    print("[INFO][Main] Output saved to {}. Avg time: {:.4f} sec".format(args.out, avg))


if __name__ == "__main__":
    main()
