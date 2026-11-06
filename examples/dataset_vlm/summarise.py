"""Summarise one or more dataset-VLM runs from the CSVs the run wrote.

What this is for:
    A run leaves one per-sample log on the dataset node (``samples_<node>.csv``).
    This script reduces it to the quantities the example exists to report - the
    task score (VQA accuracy or detection F1), end-to-end latency and the served
    sample count - so two placements can be read side by side without a plotting
    stack.

What it contains:
    - :func:`summarise`: the per-run numbers, as a mapping.
    - :func:`format_rows`: a fixed-width table of several runs.
    - :func:`write_summary`: the same numbers as ``dataset_vlm_summary.json`` in
      the run directory, which is how they reach the dashboard (it reads every
      ``*_summary.json`` and takes the first numeric leaf as the headline metric).
    - a command line taking run directories.

How it fits into CADET:
    Reads only files under ``outputs/`` that the app wrote; it never writes into
    the repository. Point it at ``outputs/<experiment>/dataset_vlm/<run_id>``.
"""

import argparse
import csv
import glob
import json
import os

SAMPLE_GLOB = "samples_*.csv"
SUMMARY_FILE = "dataset_vlm_summary.json"
COLUMNS = [
    "run",
    "task",
    "placement",
    "samples",
    "answered",
    "score",
    "e2e_p50_ms",
    "e2e_p99_ms",
    "infer_p50_ms",
]


def _floats(rows, column):
    values = []
    for row in rows:
        raw = row.get(column)
        if raw not in (None, ""):
            values.append(float(raw))
    return values


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _mean(values):
    return sum(values) / len(values) if values else None


def _read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def read_run(run_dir):
    """Return the sample rows for one run directory."""
    rows = []
    for path in sorted(glob.glob(os.path.join(run_dir, SAMPLE_GLOB))):
        rows += _read_csv(path)
    return rows


def settings_of(run_dir):
    """Return the run's task and placement, from the config saved beside it."""
    import yaml

    path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(path):
        return {"task": "unknown", "placement": "unknown"}
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    params = ((config.get("app") or {}).get("params")) or {}
    return {"task": params.get("task", "unknown"), "placement": params.get("placement", "unknown")}


def summarise(run_dir):
    """Return the summary numbers for one run directory."""
    rows = read_run(run_dir)
    settings = settings_of(run_dir)
    e2e = _floats(rows, "e2e_latency_ms")
    infer = _floats(rows, "server_infer_ms")
    if settings["task"] == "detection":
        score = _mean(_floats(rows, "det_f1"))
    else:
        score = _mean(_floats(rows, "vqa_score"))
    return {
        "run": os.path.basename(os.path.normpath(run_dir)),
        "task": settings["task"],
        "placement": settings["placement"],
        "samples": len(rows),
        "answered": len(e2e),
        "score": round(score, 4) if score is not None else None,
        "e2e_p50_ms": _percentile(e2e, 0.50),
        "e2e_p99_ms": _percentile(e2e, 0.99),
        "infer_p50_ms": _percentile(infer, 0.50),
    }


def write_summary(run_dir, summary=None):
    """Write ``dataset_vlm_summary.json`` into ``run_dir``; return its path.

    The task score is first, so the dashboard's first-numeric-leaf rule makes it
    the headline metric; every key keeps its unit in its name.
    """
    summary = summary if summary is not None else summarise(run_dir)
    ordered = {}
    for name in ("score", "e2e_p50_ms", "e2e_p99_ms", "infer_p50_ms", "samples", "answered"):
        if summary.get(name) is not None:
            ordered[name] = summary[name]
    path = os.path.join(run_dir, SUMMARY_FILE)
    with open(path, "w") as handle:
        json.dump(ordered, handle, indent=2)
    return path


def _cell(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "{:.3g}".format(value) if abs(value) < 1 else "{:.1f}".format(value)
    return str(value)


def format_rows(rows):
    widths = {name: max(len(name), *(len(_cell(row[name])) for row in rows)) for name in COLUMNS}
    lines = ["  ".join(name.ljust(widths[name]) for name in COLUMNS)]
    for row in rows:
        lines.append("  ".join(_cell(row[name]).ljust(widths[name]) for name in COLUMNS))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", nargs="+", help="a run's output directory")
    parser.add_argument(
        "--write",
        action="store_true",
        help="also write {} into each run directory, for the dashboard".format(SUMMARY_FILE),
    )
    args = parser.parse_args(argv)
    summaries = [summarise(path) for path in args.run_dir]
    if args.write:
        for path, summary in zip(args.run_dir, summaries):
            print("wrote {}".format(write_summary(path, summary)))
    print(format_rows(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
