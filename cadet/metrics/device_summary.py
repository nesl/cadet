"""Per-node roll-up of the device collectors' samples, for one run.

What this is for:
    The collectors each write a per-node CSV of samples
    (:mod:`cadet.metrics.collectors`), which is the right record to keep and
    the wrong thing to read when the question is "what did this node draw".
    This reduces those files to a few numbers per node - mean and peak
    utilisation, peak memory, mean and peak power, the run's energy - so a
    report, a dashboard or a person at a terminal can answer that without
    parsing thousands of rows.

    It is a reduction and never a source: every number here is derived from a
    CSV that stays in the run directory, and a run whose summary is missing or
    stale loses nothing.

What it contains:
    - :data:`SPECS`: per collector, which columns are averaged and peaked,
      which are cumulative (read from the last row, not summed), and what the
      samples are grouped by. Adding a collector is adding an entry.
    - :func:`summarise_run`: the whole run directory, as
      ``{"nodes": {<node>: {<collector>: {...}}}}``.
    - :func:`write_summary`: that, written to
      ``<run_dir>/device_summary.json``, or nothing when the run collected
      nothing.
    - :func:`summary_lines`: one readable line per node, for the run log.

How it fits into CADET:
    Called by :mod:`cadet.orchestrator.runner` after teardown, which is the
    first moment every node's samples are present: a node placed on another
    machine writes into a staging tree there, and the placement fetches that
    tree back as it stops. Running before the teardown would summarise the
    local nodes and silently omit the remote ones.

How to configure it:
    Nothing to configure. It summarises whatever ``metrics.collectors`` asked
    for; a run that asked for nothing gets no file, which is the same default
    the collectors have.
"""

import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

# The file this writes, beside the CSVs it reduces.
SUMMARY_FILE = "device_summary.json"

# The time column every collector stamps - one of
# ``cadet.metrics.collectors.CLOCK_COLUMNS``, pinned against it by a test rather
# than imported, so a reduction of an old run's files does not depend on the
# collectors still being importable. The one to measure a span with:
# it is monotonic, so it is not moved by a clock step mid-run, and the span
# wanted here is a duration on one node rather than an instant comparable
# across them.
_SPAN_COLUMN = "t_monotonic"

# The marker a collector writes beside its CSV when it had nothing to measure
# on that node - ``cadet.metrics.collectors.NOT_APPLICABLE_SUFFIX``, pinned
# against it by a test rather than imported, for the same reason as the column
# above.
_NOT_APPLICABLE_SUFFIX = ".not_applicable"


class _Spec:
    """How one collector's rows reduce.

    ``averaged`` columns get a ``_mean`` and a ``_peak``; ``peaked`` columns get
    a ``_peak`` alone, for a quantity whose average says little (memory in use);
    ``cumulative`` columns are read from the **last** row, because the collector
    already accumulated them and summing the column would count every sample's
    total again; ``last`` columns are constants restated every sample (a device's
    total memory, a core count).

    ``group_by`` names the columns that make a row's identity beyond the node:
    a GPU collector writes one row per device per sample and a bandwidth
    collector one per link and direction, so reducing either without grouping
    would average unrelated series together. ``group_label`` is what the group
    is called in the output.
    """

    def __init__(
        self,
        averaged: Sequence[str] = (),
        peaked: Sequence[str] = (),
        cumulative: Sequence[str] = (),
        last: Sequence[str] = (),
        group_by: Sequence[str] = (),
        group_label: str = "",
    ):
        self.averaged = tuple(averaged)
        self.peaked = tuple(peaked)
        self.cumulative = tuple(cumulative)
        self.last = tuple(last)
        self.group_by = tuple(group_by)
        self.group_label = group_label


SPECS = {
    "cpu": _Spec(
        averaged=("cpu_pct_host", "mem_pct_host", "cpu_pct_process"),
        peaked=("mem_used_mb_host", "rss_mb_process", "threads_process"),
        last=("cpu_count",),
    ),
    "gpu": _Spec(
        averaged=("util_gpu_pct", "util_mem_pct", "power_w"),
        peaked=("mem_used_mb",),
        last=("mem_total_mb",),
        group_by=("gpu_index",),
        group_label="gpus",
    ),
    # energy_j_total is the run integral the collector carries forward, so the
    # run's energy is the last row's value. energy_j is that one interval's
    # energy and is deliberately not summed here: the total already is that sum,
    # computed by the sampler that knows its own intervals.
    "energy": _Spec(
        averaged=("power_w",),
        cumulative=("energy_j_total",),
    ),
    "bandwidth": _Spec(
        averaged=("mbps",),
        cumulative=("bytes_total", "messages_total"),
        group_by=("link", "direction"),
        group_label="links",
    ),
}


def _number(cell) -> Optional[float]:
    """Parse one CSV cell as a float, or ``None`` for anything unreadable.

    An empty cell is normal rather than exceptional: the collectors share one
    file per collector across sources that report different columns, so a rail
    a source does not have is written empty. ``None`` keeps it out of the
    statistics instead of entering them as a zero.
    """
    if cell is None:
        return None
    text = str(cell).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


class _Series:
    """Accumulates one column's readable values: count, sum, low, peak, last."""

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.low: Optional[float] = None
        self.peak: Optional[float] = None
        self.last: Optional[float] = None

    def add(self, value: Optional[float]) -> None:
        if value is None:
            return
        self.count += 1
        self.total += value
        self.last = value
        if self.low is None or value < self.low:
            self.low = value
        if self.peak is None or value > self.peak:
            self.peak = value

    @property
    def mean(self) -> Optional[float]:
        return self.total / self.count if self.count else None

    @property
    def span(self) -> Optional[float]:
        """Last minus first, for a column that is a clock rather than a load."""
        if self.count < 2 or self.low is None or self.peak is None:
            return None
        return self.peak - self.low


def _round(value: Optional[float]) -> Optional[float]:
    """Round for a record meant to be read, keeping small values legible."""
    if value is None:
        return None
    return round(value, 3)


class _Group:
    """One series set: a whole collector, or one GPU or one link within it."""

    def __init__(self, spec: _Spec):
        self.spec = spec
        self.rows = 0
        self.series: Dict[str, _Series] = {}
        self.span = _Series()

    def add(self, row: Dict[str, str]) -> None:
        self.rows += 1
        self.span.add(_number(row.get(_SPAN_COLUMN)))
        columns = (
            self.spec.averaged + self.spec.peaked + self.spec.cumulative + self.spec.last
        )
        for column in columns:
            self.series.setdefault(column, _Series()).add(_number(row.get(column)))

    def as_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {"samples": self.rows}
        if self.span.span is not None:
            out["duration_s"] = _round(self.span.span)
        for column in self.spec.averaged:
            series = self.series.get(column)
            if series is None or not series.count:
                continue
            out[column + "_mean"] = _round(series.mean)
            out[column + "_peak"] = _round(series.peak)
        for column in self.spec.peaked:
            series = self.series.get(column)
            if series is not None and series.count:
                out[column + "_peak"] = _round(series.peak)
        for column in self.spec.cumulative + self.spec.last:
            series = self.series.get(column)
            if series is not None and series.count:
                out[column] = _round(series.last)
        return out


def summarise_file(path: str, collector: str) -> Dict[str, object]:
    """Reduce one ``<collector>_<node>.csv``. Returns ``{}`` for no samples.

    A file that exists with a header and no rows is a node whose role started
    and died before it sampled, which is a real outcome and reads here as zero
    samples rather than as an error.
    """
    spec = SPECS.get(collector)
    if spec is None:
        return {}
    whole = _Group(spec)
    groups: Dict[Tuple[str, ...], _Group] = {}
    try:
        with open(path, "r") as fh:
            for row in csv.DictReader(fh):
                if spec.group_by:
                    key = tuple(str(row.get(name, "")) for name in spec.group_by)
                    member = groups.get(key)
                    if member is None:
                        member = groups[key] = _Group(spec)
                    member.add(row)
                    whole.rows += 1
                else:
                    whole.add(row)
    except (OSError, csv.Error, UnicodeDecodeError):
        # A truncated or unreadable file is reported as nothing measured. The
        # CSV itself stays in the run directory, so the rows that did land are
        # not lost by this returning empty.
        return {}
    if spec.group_by:
        if not groups:
            return {}
        return {
            "samples": whole.rows,
            spec.group_label: {
                "/".join(key): member.as_dict() for key, member in sorted(groups.items())
            },
        }
    return whole.as_dict() if whole.rows else {}


def _node_of(filename: str, collector: str) -> str:
    """Return the node name in ``<collector>_<node>.csv``."""
    return filename[len(collector) + 1 : -len(".csv")]


def summarise_run(out_dir: str) -> Dict[str, object]:
    """Reduce every collector CSV in a run directory, per node.

    Returns ``{}`` when the run collected nothing, so a caller can tell "this
    run measured no devices" from "this run measured devices that drew zero".
    """
    nodes: Dict[str, Dict[str, object]] = {}
    try:
        names = sorted(os.listdir(out_dir))
    except OSError:
        return {}
    for name in names:
        if not name.endswith(".csv"):
            continue
        for collector in SPECS:
            if not name.startswith(collector + "_"):
                continue
            node = _node_of(name, collector)
            if not node:
                continue
            marker = os.path.join(out_dir, name + _NOT_APPLICABLE_SUFFIX)
            if os.path.exists(marker):
                # Stated rather than omitted. A node that had nothing to measure
                # and a node nobody asked about both produce no numbers, and
                # only one of them is worth a reader's attention.
                try:
                    with open(marker) as handle:
                        reason = handle.read().strip()
                except OSError:
                    reason = "not applicable"
                nodes.setdefault(node, {})[collector] = {
                    "not_applicable": reason or "not applicable"
                }
                break
            reduced = summarise_file(os.path.join(out_dir, name), collector)
            if reduced:
                nodes.setdefault(node, {})[collector] = reduced
            break
    return {"nodes": nodes} if nodes else {}


def write_summary(out_dir: str) -> Optional[str]:
    """Write ``device_summary.json``; return its path, or ``None`` if nothing.

    No file rather than an empty one, for the same reason the collectors write
    nothing when none is configured: a file saying ``{}`` is indistinguishable
    from a run whose devices reported nothing.
    """
    summary = summarise_run(out_dir)
    if not summary:
        return None
    path = os.path.join(out_dir, SUMMARY_FILE)
    try:
        with open(path, "w") as fh:
            json.dump(summary, fh, indent=2)
    except OSError:
        return None
    return path


def summary_lines(summary: Dict[str, object]) -> List[str]:
    """Return one line per node, for the run log.

    The run log is where this is read while a run is being watched, so the line
    carries the few numbers a person checks at that moment - load, power, energy
    - and the file carries the rest.
    """
    lines = []
    for node, collectors in sorted((summary.get("nodes") or {}).items()):
        parts = []
        cpu = collectors.get("cpu") or {}
        if "cpu_pct_host_mean" in cpu:
            parts.append(
                "cpu {}%/{}% mean/peak".format(
                    cpu.get("cpu_pct_host_mean"), cpu.get("cpu_pct_host_peak")
                )
            )
        for index, gpu in sorted((collectors.get("gpu") or {}).get("gpus", {}).items()):
            parts.append(
                "gpu{} {}% {}W peak {}MB".format(
                    index,
                    gpu.get("util_gpu_pct_mean"),
                    gpu.get("power_w_mean"),
                    gpu.get("mem_used_mb_peak"),
                )
            )
        energy = collectors.get("energy") or {}
        if energy.get("energy_j_total") is not None:
            parts.append(
                "energy {}J at {}W mean".format(
                    energy.get("energy_j_total"), energy.get("power_w_mean")
                )
            )
        if parts:
            lines.append("[metrics] {}: {}".format(node, ", ".join(parts)))
    return lines
