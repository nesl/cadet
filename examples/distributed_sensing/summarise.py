"""Summarise one or more distributed sensing runs from the CSVs the run wrote.

What this is for:
    A run leaves one event log per sensor and one arrival log on the processing
    node. This script reduces them to the four quantities the example exists to
    compare - end-to-end latency, age of information at the processing node,
    detection rate and bytes per link - so two placements can be read side by
    side without a plotting stack.

What it contains:
    - :func:`read_run`: load one run directory's sensor logs and the matching
      arrival log.
    - :func:`summarise`: the per-run numbers, as a mapping.
    - :func:`format_rows`: a fixed-width table of several runs.
    - :func:`write_summary`: the same numbers as ``sensing_summary.json`` inside
      the run directory, which is how they reach a reader that does not know
      this application.
    - a command line taking run directories.

How it fits into CADET:
    Reads only files under ``outputs/`` that the app wrote; it never writes into
    the repository. Point it at ``outputs/<experiment>/distributed_sensing/<run_id>``.

    With ``--write``, it also writes ``sensing_summary.json`` into each run
    directory. That name is not arbitrary: the dashboard reads every
    ``*_summary.json`` in a run directory and flattens its numeric leaves, which
    is how an application's own metrics reach the sweep matrix and the crossover
    plot without the dashboard knowing anything about this example. The
    reduction stays offline and after the fact, because the quantities need both
    the per-sensor logs and the processing node's arrival log, and no single
    role has both.

How to configure it:
    Nothing to configure. Every number it prints comes from a named column of a
    named CSV, listed in the example's README.
"""

import argparse
import csv
import glob
import os

# Columns the summary reads, so a change to the app's logs fails here loudly.
# Two globs, not one: the per-sensor log is `events_<node>.csv`, and the first
# pattern is what it was called before the sensors were named after their nodes.
# Runs recorded under the old name still summarise, which is the only reason to
# keep reading it.
SENSOR_GLOBS = ("events_*.csv", "sensor_*_events.csv")
ARRIVAL_PREFIX = "arrivals_"
COLUMNS = [
    "run",
    "placement",
    "access_p50_ms",
    "access_loss",
    "sensors",
    "answered",
    "lost",
    "e2e_p50_ms",
    "e2e_p99_ms",
    "aoi_p50_ms",
    "aoi_p99_ms",
    "frames_with_box",
    "uplink_kb",
]


def _floats(rows, column):
    """Return the numeric values of one column, skipping blanks."""
    values = []
    for row in rows:
        raw = row.get(column)
        if raw not in (None, ""):
            values.append(float(raw))
    return values


def _percentile(values, fraction):
    """Return the value at ``fraction`` of the sorted sample, or ``None``."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def read_run(run_dir):
    """Return ``(sensor_rows, arrival_rows)`` for one run directory."""
    sensor_rows = []
    for pattern in SENSOR_GLOBS:
        for path in sorted(glob.glob(os.path.join(run_dir, pattern))):
            sensor_rows += _read_csv(path)
    # The processing node writes its arrival log beside the run directories,
    # named by the config hash, because the server role is not told the run id.
    parent = os.path.dirname(os.path.normpath(run_dir))
    digest = os.path.basename(os.path.normpath(run_dir)).split("-r")[0]
    arrival_path = os.path.join(parent, ARRIVAL_PREFIX + digest + ".csv")
    arrival_rows = _read_csv(arrival_path) if os.path.exists(arrival_path) else []
    return sensor_rows, arrival_rows


def settings_of(run_dir):
    """Return what this run was configured to do, from the config saved beside it.

    The runtime writes the resolved config into the run directory, so the axes a
    sweep varied can be read back from the results rather than recomputed from
    the sweep definition.
    """
    import yaml

    path = os.path.join(run_dir, "config.yaml")
    unknown = {"placement": "unknown", "access_p50_ms": None, "access_loss": None}
    if not os.path.exists(path):
        return unknown
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    links = config.get("links") or []
    access = (links[0].get("network") or {}) if links else {}
    distribution = access.get("distribution") or {}
    return {
        "placement": ((config.get("app") or {}).get("params") or {}).get("placement", "unknown"),
        "access_p50_ms": distribution.get("p50_ms"),
        "access_p99_ms": distribution.get("p99_ms"),
        "access_loss": access.get("loss"),
        "access_bandwidth_mbps": access.get("bandwidth_mbps"),
        "sweep": (config.get("experiment") or {}).get("name"),
    }


def summarise(run_dir):
    """Return the summary numbers for one run directory."""
    sensor_rows, arrival_rows = read_run(run_dir)
    e2e = _floats(sensor_rows, "e2e_latency_ms")
    aoi = _floats(arrival_rows, "aoi_ms") or _floats(sensor_rows, "aoi_ms")
    detections = _floats(sensor_rows, "detections")
    uplink_bytes = sum(_floats(sensor_rows, "bytes_sent"))
    sensors = len({row["sensor"] for row in sensor_rows}) if sensor_rows else 0
    with_box = sum(1 for value in detections if value >= 1)
    summary = {
        "run": os.path.basename(os.path.normpath(run_dir)),
        "sensors": sensors,
        "answered": len(e2e),
        "lost": len(sensor_rows) - len(e2e),
        "e2e_p50_ms": _percentile(e2e, 0.50),
        "e2e_p99_ms": _percentile(e2e, 0.99),
        "aoi_p50_ms": _percentile(aoi, 0.50),
        "aoi_p99_ms": _percentile(aoi, 0.99),
        "frames_with_box": (
            round(with_box / len(detections), 3) if detections else None
        ),
        "uplink_kb": round(uplink_bytes / 1024.0, 1),
    }
    summary.update(settings_of(run_dir))
    return summary


# The file name the dashboard looks for. One definition, because the dashboard
# matches on the suffix and a rename here would silently stop the metric
# appearing rather than fail.
SUMMARY_FILE = "sensing_summary.json"


def write_summary(run_dir, summary=None):
    """Write ``sensing_summary.json`` into ``run_dir``; return its path.

    The headline metric is first in the file for a reader that takes the first
    numeric leaf, and every key keeps its unit in its name, because a number in
    a matrix cell with no unit is not a measurement.
    """
    import json

    summary = summary if summary is not None else summarise(run_dir)
    path = os.path.join(run_dir, SUMMARY_FILE)
    ordered = {}
    for name in ("e2e_p50_ms", "e2e_p99_ms", "aoi_p50_ms", "aoi_p99_ms"):
        if summary.get(name) is not None:
            ordered[name] = summary[name]
    for name in ("answered", "lost", "sensors", "frames_with_box", "uplink_kb"):
        if summary.get(name) is not None:
            ordered[name] = summary[name]
    with open(path, "w") as handle:
        json.dump(ordered, handle, indent=2)
    return path


def format_rows(rows):
    """Render the summaries as a fixed-width table."""
    widths = {
        name: max(len(name), *(len(_cell(row[name])) for row in rows)) for name in COLUMNS
    }
    lines = ["  ".join(name.ljust(widths[name]) for name in COLUMNS)]
    for row in rows:
        lines.append("  ".join(_cell(row[name]).ljust(widths[name]) for name in COLUMNS))
    return "\n".join(lines)


def _cell(value):
    """Render one value for the table."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        # A loss fraction and a latency in milliseconds share this column
        # formatter, and one decimal place would print every loss as zero.
        return "{:.3g}".format(value) if abs(value) < 1 else "{:.1f}".format(value)
    return str(value)


def main(argv=None):
    """Summarise every run directory named on the command line."""
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
