#!/usr/bin/env python3
"""Multi-tenant GPU-contention launcher (ported from the original V2I load study).

Ports ``carla_wrapper/energy_monitoring/multi-tenancy/multi_tenant_launcher.py``: for
each tenant count it starts that many background YOLO-on-local-images workers to load
the GPU, logs GPU utilisation with ``nvidia-smi`` (the original's logger, verbatim),
and then runs the measured workload. It has two modes:

  benchmark (default, = the original): the measured workload is one timed
    ``run_yolo_main_server`` pass over the images; per-frame times go to a CSV, so you
    get the main server's latency vs tenant count, exactly as the original did.

  wrap (``--wrap "<command>"``): the measured workload is an external command - a V2I
    run (``cadet run examples/v2i/v2i_braking.yaml ...``) - so its RSU brakes under N
    tenants of GPU pressure. This is the default V2I load mechanism: extra
    YOLO-on-images *processes* contending for the GPU, NOT extra network clients to the
    RSU (that fan-in path is ``cadet/roles/load.py`` and a documented opt-in deviation).

Workers are duration-bounded (``--worker-seconds``) and also terminated when the
measured workload returns, so a sweep never leaves GPU load running between cells.

Usage:
    # Faithful standalone benchmark (sweep 0,5,10):
    python scripts/multi_tenant.py --model yolo11x.pt --images <dir> \
        --out-dir <dir> --counts 0,5,10

    # A V2I run under each tenant count:
    python scripts/multi_tenant.py --model yolo11x.pt --images <dir> \
        --out-dir <dir> --counts 0,5,10 --worker-seconds 240 \
        --wrap "python -m cadet run examples/v2i/v2i_braking.yaml --app-module examples.v2i.app ..."
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(REPO, "scripts", "gpu_load_worker.py")


def start_gpu_logger(out_dir, count):
    """nvidia-smi utilisation logger, verbatim from the original launcher."""
    log_path = os.path.join(out_dir, "gpu_util_{}_clients.csv".format(count))
    proc = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,utilization.gpu",
            "--format=csv,noheader,nounits",
            "-l", "1",
        ],
        stdout=open(log_path, "w"),
    )
    return proc, log_path


def stop_gpu_logger(proc):
    proc.send_signal(signal.SIGINT)
    proc.wait()


def start_workers(count, model, images, worker_seconds):
    """Start ``count`` background GPU-load workers; return the process handles."""
    procs = []
    for _ in range(count):
        procs.append(subprocess.Popen([
            sys.executable, WORKER,
            "--model", model,
            "--images", images,
            "--duration", str(worker_seconds),
        ]))
    return procs


def stop_workers(procs):
    """Ask workers to stop (they also self-exit at --duration as a backstop)."""
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()


def run_main_server(count, model, images, out_dir):
    """The original's measured workload: one timed YOLO pass over the images."""
    out_file = os.path.join(out_dir, "main_client_{}_clients.csv".format(count))
    subprocess.run([
        sys.executable, os.path.join(REPO, "scripts", "run_yolo_main_server.py"),
        "--model", model, "--images", images, "--out", out_file,
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11x.pt")
    parser.add_argument("--images", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--counts", default="0,5,10",
        help="comma-separated tenant counts to sweep (original used 0..10)",
    )
    parser.add_argument(
        "--worker-seconds", type=float, default=300.0,
        help="max seconds each load worker runs (backstop; also terminated on finish)",
    )
    parser.add_argument(
        "--wrap", default=None,
        help="command to run as the measured workload instead of the benchmark pass",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    counts = [int(c) for c in args.counts.split(",") if c.strip() != ""]

    for count in counts:
        print("\n[INFO] ==== RUN: measured workload + {} extra tenants ====".format(count))
        start = datetime.now()
        gpu_proc, gpu_log = start_gpu_logger(args.out_dir, count)
        print("[INFO] GPU logging to: {}".format(gpu_log))

        workers = start_workers(count, args.model, args.images, args.worker_seconds)
        if count > 0:
            print("[INFO] waiting for {} tenants to initialise...".format(count))
            time.sleep(2)  # let tenants occupy the GPU first, as the original did

        if args.wrap:
            rc = subprocess.run(args.wrap, shell=True).returncode
            print("[INFO] wrapped command rc={}".format(rc))
        else:
            run_main_server(count, args.model, args.images, args.out_dir)

        stop_workers(workers)
        stop_gpu_logger(gpu_proc)
        print("[INFO] {} tenants: done in {}".format(count, datetime.now() - start))


if __name__ == "__main__":
    main()
