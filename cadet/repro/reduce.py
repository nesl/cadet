"""Reduce N repetitions of a config into one variance record.

What this is for:
    The reduction half of the reproducibility harness: find the repetition
    directories under a run tree, group them by the config cell they belong to,
    and per metric report the median, a defensible spread, the count, and **how
    many repetitions produced the metric at all**. Per-repetition values are
    reported beside the summary, because a bimodal set of twenty runs has a
    median that describes neither mode and the braking numbers here have shown
    that shape.

What it contains:
    - :func:`discover_repetitions` / :func:`reduce_tree`: discovery and the
      whole reduction.
    - :class:`MetricSummary`: one metric across repetitions, including the
      repetitions that lack it and a bimodality hint.
    - :func:`findings_for`: the things a reader must not have to notice -
      metrics missing from some repetitions, failed runs, context that differs
      between repetitions, a latency paced near the shaped round trip with
      Nagle left at the original's default, warm-up off.
    - :func:`render_json` / :func:`render_markdown` / :func:`write_reports`.

How it fits into CADET:
    Driven by ``cadet variance`` (:mod:`cadet.cli`). It reads only what a run
    already writes (via :mod:`cadet.repro.extract`) and writes its report into
    the run tree, never into the repository: a variance report is a result.

How to configure it:
    Nothing to configure. **Spread is reported as the inter-quartile range
    first** and the standard deviation second: the IQR is what matches a median,
    it survives the one slow first inference that an un-warmed model produces,
    and it does not imply a dispersion about a mean that no run sits near. The
    standard deviation is kept because that is what a paper's variance check
    usually quotes.
"""

import json
import os
import platform
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence

from cadet.repro.extract import RUN_MARKER, Repetition, read_repetition

# Report file names, written into the directory being reduced.
REPORT_JSON = "variance_report.json"
REPORT_MD = "variance_report.md"

# A run directory holds a per-run log directory; nothing in it is a repetition.
SKIP_DIRS = ("logs", "client_frames", "server_detections")

# Context keys that are expected to differ between repetitions (the repetition
# index is in the run id) and so are not reported as varying context.
CONTEXT_PER_REP = ("client_outcome", "limit")

# Bimodality hint: with fewer than this many values a gap is not evidence of
# two modes, it is just a small sample.
MIN_FOR_MODES = 4


class MetricSummary:
    """One metric across the repetitions of one cell.

    ``count`` is how many repetitions produced the metric and ``missing`` names
    the ones that did not, so a metric absent from three of twenty runs reads as
    that rather than as a 17-run average.
    """

    def __init__(self, name: str, values: Dict[str, float], missing: Sequence[str]):
        self.name = name
        self.values = dict(values)
        self.missing = list(missing)
        numbers = sorted(self.values.values())
        self.count = len(numbers)
        self.median = statistics.median(numbers) if numbers else None
        self.minimum = numbers[0] if numbers else None
        self.maximum = numbers[-1] if numbers else None
        # A single value has no deviation: reporting 0.0 would claim a
        # tightness that was never measured.
        self.stdev = statistics.stdev(numbers) if len(numbers) > 1 else None
        self.q1, self.q3, self.iqr = _quartiles(numbers)
        self.modes = _modes(numbers)

    def as_dict(self) -> Dict[str, Any]:
        """JSON-ready view, per-repetition values included."""
        return {
            "metric": self.name,
            "count": self.count,
            "missing_from": self.missing,
            "median": self.median,
            "iqr": self.iqr,
            "q1": self.q1,
            "q3": self.q3,
            "stdev": self.stdev,
            "min": self.minimum,
            "max": self.maximum,
            "bimodal": self.modes is not None,
            "modes": self.modes,
            "per_repetition": self.values,
        }


def _quartiles(numbers: Sequence[float]):
    """Inclusive quartiles and the IQR, or ``(None, None, None)`` if undefined.

    ``statistics.quantiles`` needs two points; with one repetition there is no
    spread to report and saying so is better than reporting zero.
    """
    if len(numbers) < 2:
        return None, None, None
    q1, _, q3 = statistics.quantiles(numbers, n=4, method="inclusive")
    return q1, q3, q3 - q1


def _modes(numbers: Sequence[float]) -> Optional[Dict[str, Any]]:
    """Flag a set that splits into two clusters, and describe them.

    A heuristic, deliberately: the largest gap between consecutive sorted values
    must be at least half the whole range and leave at least two values on each
    side. It is a hint that the median describes neither mode - the per-metric
    values are always reported, so the reader checks it rather than trusting it.
    """
    if len(numbers) < MIN_FOR_MODES:
        return None
    span = numbers[-1] - numbers[0]
    if span <= 0:
        return None
    gaps = [(b - a, i) for i, (a, b) in enumerate(zip(numbers, numbers[1:]))]
    gap, index = max(gaps)
    low, high = numbers[: index + 1], numbers[index + 1 :]
    if gap < 0.5 * span or len(low) < 2 or len(high) < 2:
        return None
    return {
        "gap": gap,
        "low": {"n": len(low), "median": statistics.median(low)},
        "high": {"n": len(high), "median": statistics.median(high)},
    }


def discover_repetitions(root: str) -> List[str]:
    """Return every repetition directory under ``root``, sorted.

    A repetition is a directory holding a run marker, which is what the
    orchestrator writes per ``(run_index, rep)`` cell. ``root`` itself counts,
    so a single run directory reduces to a one-repetition record.
    """
    found: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        if RUN_MARKER in filenames:
            found.append(dirpath)
    return sorted(found)


def reduce_tree(root: str) -> Dict[str, Any]:
    """Reduce every cell under ``root`` into one report record."""
    paths = discover_repetitions(root)
    reps = [read_repetition(path) for path in paths]
    cells: Dict[str, List[Repetition]] = {}
    for item in reps:
        cells.setdefault(item.base, []).append(item)
    return {
        "root": os.path.abspath(root),
        "repetition_dirs": len(paths),
        "spread": (
            "IQR first (matches the median, survives one slow first inference); "
            "stdev reported beside it"
        ),
        "environment": _environment(),
        "cells": [_reduce_cell(base, items) for base, items in sorted(cells.items())],
    }


def _environment() -> Dict[str, Any]:
    """Where the *reduction* ran, labelled as such.

    This is not the measurement environment unless the reduction runs on the
    same host and image as the runs did. Image provenance matters here - the
    project has compared numbers taken in different images before - so the
    versions that a measurement would depend on are read live rather than taken
    from a pin, and the caveat is carried in the report.
    """
    env: Dict[str, Any] = {
        "note": (
            "read where the reduction ran, not necessarily where the runs ran; "
            "do not compare numbers taken in different images"
        ),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": platform.node(),
    }
    try:
        import cv2

        env["cv2"] = cv2.__version__
    except Exception as exc:  # noqa: BLE001 - a missing optional dep is a fact to record
        env["cv2"] = "unavailable: {}".format(exc)
    return env


def _reduce_cell(base: str, items: List[Repetition]) -> Dict[str, Any]:
    """Summarise one config cell's repetitions."""
    items = sorted(items, key=lambda i: (i.rep if i.rep is not None else -1, i.run_id))
    labels = [item.label for item in items]
    names = sorted({name for item in items for name in item.metrics})
    summaries = []
    for name in names:
        values = {item.label: item.metrics[name] for item in items if name in item.metrics}
        missing = [label for label in labels if label not in values]
        summaries.append(MetricSummary(name, values, missing))
    statuses: Dict[str, int] = {}
    for item in items:
        statuses[item.status] = statuses.get(item.status, 0) + 1
    return {
        "cell": base,
        "repetitions": len(items),
        "statuses": statuses,
        "context": _context_for(items),
        "findings": findings_for(items, summaries),
        "metrics": [summary.as_dict() for summary in summaries],
        "per_repetition": [item.as_dict() for item in items],
    }


def _context_for(items: List[Repetition]) -> Dict[str, Any]:
    """Shared measurement context, with any key that differs marked as varying.

    Repetitions of one cell should share their configuration; a key that does
    not is reported as ``{"varies": {...}}`` rather than reduced away, because
    it means the repetitions are not repetitions of the same thing.
    """
    keys = sorted({key for item in items for key in item.context})
    shared: Dict[str, Any] = {}
    for key in keys:
        if key in CONTEXT_PER_REP:
            continue
        seen = {item.label: item.context.get(key) for item in items}
        distinct = {json.dumps(v, sort_keys=True, default=str) for v in seen.values()}
        if len(distinct) == 1:
            shared[key] = next(iter(seen.values()))
        else:
            shared[key] = {"varies": seen}
    return shared


def findings_for(items: List[Repetition], summaries: List[MetricSummary]) -> List[str]:
    """The things a reader of these numbers must not have to notice for himself."""
    out: List[str] = []
    total = len(items)

    failed = [item.label for item in items if item.status != "complete"]
    if failed:
        out.append(
            "{} of {} repetitions did not complete ({}); their metrics are reported as "
            "measured and might not be comparable with the rest".format(
                len(failed), total, ", ".join(failed)
            )
        )
    for item in items:
        if item.status != "complete" and item.reason:
            out.append("{}: {} ({})".format(item.label, item.status, item.reason))

    for summary in summaries:
        if summary.missing:
            out.append(
                "{} present in {} of {} repetitions; missing from {} - a finding, "
                "not a gap to average over".format(
                    summary.name, summary.count, total, ", ".join(summary.missing)
                )
            )
        if summary.modes:
            out.append(
                "{} is bimodal: {} repetitions near {:.4g} and {} near {:.4g}; the median "
                "{:.4g} describes neither mode".format(
                    summary.name,
                    summary.modes["low"]["n"],
                    summary.modes["low"]["median"],
                    summary.modes["high"]["n"],
                    summary.modes["high"]["median"],
                    summary.median,
                )
            )

    out.extend(_context_findings(items, summaries))
    out.extend(_note_findings(items))
    return out


def _note_findings(items: List[Repetition]) -> List[str]:
    """Per-repetition notes, with one that applies to every repetition said once.

    With twenty repetitions the same absent file otherwise fills the report
    twenty times and buries the note that applies to only three of them - which
    is the one that matters, because that is the uneven case.
    """
    total = len(items)
    seen: Dict[str, List[str]] = {}
    for item in items:
        for note in item.notes:
            seen.setdefault(note, []).append(item.label)
    out = []
    for note, labels in seen.items():
        if len(labels) == total and total > 1:
            out.append("all {} repetitions: {}".format(total, note))
        else:
            out.append("{}: {}".format(", ".join(labels), note))
    return out


def _context_findings(items: List[Repetition], summaries: List[MetricSummary]) -> List[str]:
    """Context-driven caveats: warm-up, Nagle versus the send interval, varying config."""
    out: List[str] = []
    context = _context_for(items)

    for key, value in sorted(context.items()):
        if isinstance(value, dict) and "varies" in value:
            out.append(
                "context {} differs between repetitions ({}): they are not repetitions "
                "of one configuration".format(key, json.dumps(value["varies"], default=str))
            )

    if context.get("warmup") is False:
        out.append(
            "models.warmup is false (the braking examples' faithful default), so the first "
            "inference pays the model's start-up cost inside the measured run; with N "
            "repetitions that lands in the spread rather than the median - see the per-run "
            "server inference max beside its median"
        )

    intervals = context.get("send_interval_ms")
    configured = context.get("configured_p50_ms")
    nodelay = context.get("tcp_nodelay")
    if isinstance(intervals, dict) and isinstance(configured, dict):
        both_ends_on = bool(nodelay) and isinstance(nodelay, dict) and all(
            value is True for value in nodelay.values()
        )
        worst = max(
            (p50 for values in configured.values() if isinstance(values, list) for p50 in values),
            default=0.0,
        )
        fastest = min((v for v in intervals.values() if v), default=None)
        if worst and fastest and not both_ends_on and fastest < 2.0 * worst:
            out.append(
                "send interval {:.0f} ms is within twice the configured one-way p50 "
                "({:.0f} ms) and tcp_nodelay is not on at both ends: on a packet-level "
                "link a latency measured here reads as the link plus an extra round trip. "
                "State the send interval beside every latency".format(fastest, worst)
            )
    if any(summary.name.startswith("link.") for summary in summaries):
        out.append(
            "send interval beside every latency: {}".format(
                json.dumps(intervals, default=str) if intervals else "not recorded in the config"
            )
        )
    return out


def render_json(report: Dict[str, Any]) -> str:
    """The whole record, per-repetition values included."""
    return json.dumps(report, indent=2, sort_keys=False, default=str) + "\n"


def _fmt(value: Any) -> str:
    """Format a number for the Markdown table, leaving an absent one visible."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return "{:.4g}".format(value)
    return str(value)


def render_markdown(report: Dict[str, Any]) -> str:
    """A report a person reads: context, findings, summary, then every value."""
    lines = ["# Variance over repetitions", ""]
    lines.append("Run tree: `{}`".format(report["root"]))
    lines.append("")
    lines.append("Repetition directories found: {}".format(report["repetition_dirs"]))
    lines.append("")
    lines.append("Spread: {}.".format(report["spread"]))
    lines.append("")
    env = report.get("environment") or {}
    lines.append(
        "Reduction environment (not necessarily the measurement environment): "
        "python {}, cv2 {}, {}.".format(env.get("python"), env.get("cv2"), env.get("platform"))
    )
    lines.append("")

    for cell in report["cells"]:
        lines.append("## Cell `{}` - {} repetitions".format(cell["cell"], cell["repetitions"]))
        lines.append("")
        lines.append(
            "Statuses: {}".format(
                ", ".join("{}={}".format(k, v) for k, v in sorted(cell["statuses"].items()))
            )
        )
        lines.append("")
        lines.append("### Measurement context")
        lines.append("")
        for key, value in sorted((cell.get("context") or {}).items()):
            lines.append("- `{}`: {}".format(key, json.dumps(value, default=str)))
        lines.append("")
        lines.append("### Findings and caveats")
        lines.append("")
        findings = cell.get("findings") or []
        if not findings:
            lines.append("- none: every metric present in every repetition, all runs complete")
        for finding in findings:
            lines.append("- {}".format(finding))
        lines.append("")
        lines.append("### Summary per metric")
        lines.append("")
        lines.append("| metric | n / N | median | IQR | stdev | min | max | bimodal |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for metric in cell["metrics"]:
            lines.append(
                "| `{}` | {} / {} | {} | {} | {} | {} | {} | {} |".format(
                    metric["metric"],
                    metric["count"],
                    cell["repetitions"],
                    _fmt(metric["median"]),
                    _fmt(metric["iqr"]),
                    _fmt(metric["stdev"]),
                    _fmt(metric["min"]),
                    _fmt(metric["max"]),
                    "yes" if metric["bimodal"] else "",
                )
            )
        lines.append("")
        lines.append("### Per-repetition values")
        lines.append("")
        labels = [item["run_id"] for item in cell["per_repetition"]]
        short = [
            "r{}".format(item["rep"]) if item["rep"] is not None else item["run_id"]
            for item in cell["per_repetition"]
        ]
        lines.append("| metric | {} |".format(" | ".join(short)))
        lines.append("| --- |{}".format(" --- |" * len(short)))
        for metric in cell["metrics"]:
            values = metric["per_repetition"]
            lines.append(
                "| `{}` | {} |".format(
                    metric["metric"],
                    " | ".join(_fmt(values.get(label)) for label in short),
                )
            )
        lines.append("")
        lines.append("Repetition directories: {}".format(", ".join(labels)))
        lines.append("")
    return "\n".join(lines)


def write_reports(report: Dict[str, Any], out_dir: str) -> List[str]:
    """Write both renderings into ``out_dir`` and return the paths written.

    ``out_dir`` is part of the run tree: a variance report is a result, so it
    never goes in the repository.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, text in ((REPORT_JSON, render_json(report)), (REPORT_MD, render_markdown(report))):
        path = os.path.join(out_dir, name)
        with open(path, "w") as handle:
            handle.write(text)
        written.append(path)
    return written
