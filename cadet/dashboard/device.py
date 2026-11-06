"""Read the per-node device series a run's metric collectors wrote.

What this is for:
    The device view plots GPU, CPU and energy over time, per node. The series
    are the collector CSVs :mod:`cadet.metrics.collectors` writes into the run
    directory; this module turns them into plottable series and headline
    numbers, and says "no device samples in this run" for everything else.

What it contains:
    - :data:`COLLECTORS`: the collector files read, and the quantity each one
      contributes to a panel.
    - :class:`Series`: one quantity of one node over time.
    - :func:`read_device_series`: the whole read, per node.
    - :func:`read_device_summary`: the orchestrator's roll-up, when present.

How it fits into CADET:
    Reads ``<run_dir>/<collector>_<node>.csv`` as written by
    :func:`cadet.metrics.collectors.csv_path_for`, whose header is the three
    clock columns ``t_wall,t_monotonic,node`` followed by that collector's own.
    Called by :mod:`cadet.dashboard.render` for the device view. Writes nothing.

    Two absences are normal and neither is an error. A collector file is missing
    because ``metrics.collectors`` did not name it, which is the default; and a
    file can carry a header with no rows because the role died before its first
    sample. Both read as no samples.

    The time axis is ``t_monotonic`` minus the series' own first value, so each
    series starts at zero. ``t_wall`` is the only clock comparable across nodes,
    and two nodes on different machines have an offset the ``clock_offset``
    collector measures, so this module does not align nodes to each other and
    does not pretend that offset is zero.

How to configure it:
    Nothing here. Which collectors wrote anything is decided by the run's
    ``metrics.collectors``.
"""

import glob
import os
import statistics
from typing import Any, Dict, List, Optional

from cadet.dashboard.runs import DEVICE_SUMMARY as _DEVICE_SUMMARY
from cadet.dashboard.runs import LEGACY_NODE, _number, load_rows, node_suffix

# The per-node roll-up the orchestrator writes after teardown, so it covers
# remote nodes whose CSVs are fetched back. A convenience for the headline
# numbers only: the CSVs stay the source of truth for the series, so the view
# still works against a run directory written before the summary existed.
#
# Taken from :mod:`cadet.dashboard.runs` rather than spelled again here, because
# that module has to skip this file when it globs the application sidecars and
# the two names drifting apart would silently reintroduce that collision.
DEVICE_SUMMARY = _DEVICE_SUMMARY

# The quantity columns each collector contributes, as
# ``collector -> [(column, label, unit)]``. Only these are plotted; the clock and
# identity columns are the series' axis and key, not quantities.
#
# ``gpu`` and ``bandwidth`` write one row per device / per link+direction per
# sample, so their key column splits a node's file into several series rather
# than labelling one. ``energy_j_total`` is the run integral and cumulative, so
# it is a last-row headline rather than a per-sample quantity.
COLLECTORS = (
    (
        "gpu",
        "gpu_index",
        (
            ("util_gpu_pct", "GPU utilisation", "%"),
            ("mem_used_mb", "GPU memory", "MB"),
            ("power_w", "GPU power", "W"),
        ),
    ),
    (
        "cpu",
        None,
        (
            ("cpu_pct_host", "CPU (host)", "%"),
            ("cpu_pct_process", "CPU (process)", "%"),
            ("rss_mb_process", "resident memory", "MB"),
        ),
    ),
    (
        "energy",
        None,
        (
            ("power_w", "power", "W"),
            ("energy_j", "energy per sample", "J"),
        ),
    ),
    (
        "bandwidth",
        "link",
        (
            ("mbps", "throughput", "Mbps"),
            ("messages_delta", "messages per sample", ""),
        ),
    ),
)

# Columns whose last value is the run's total rather than a sample, so they are
# reported as a headline and never summed over the column.
CUMULATIVE = {
    "energy": (("energy_j_total", "energy over the run", "J"),),
    "bandwidth": (("bytes_total", "bytes over the run", "B"), ),
}


class Series(object):
    """One quantity of one node over time: the x/y a plot panel draws.

    ``t`` is seconds since this series' own first sample, never a wall clock:
    the panel's question is how a quantity moved during the run, and the run's
    own start is the only origin both a local and a fetched-back CSV agree on.
    """

    def __init__(self, collector: str, label: str, unit: str, key: str = "") -> None:
        self.collector = collector
        self.label = label
        self.unit = unit
        self.key = key
        self.t = []  # type: List[float]
        self.y = []  # type: List[float]

    @property
    def name(self) -> str:
        """The panel's title: the quantity, and which device or link it is."""
        return "{} [{}]".format(self.label, self.key) if self.key else self.label

    @property
    def samples(self) -> int:
        """How many samples this series holds."""
        return len(self.y)

    def stats(self) -> Dict[str, Optional[float]]:
        """Mean, peak and last, which is what a headline tile shows."""
        if not self.y:
            return {"mean": None, "peak": None, "last": None}
        return {"mean": statistics.mean(self.y), "peak": max(self.y), "last": self.y[-1]}


def read_device_summary(run_dir: str) -> Dict[str, Any]:
    """The orchestrator's per-node roll-up, or an empty mapping.

    Shape: ``{"nodes": {"<node>": {"<collector>": {...}}}}``. Partial or absent
    is expected rather than an error - the view falls back to computing its
    headline numbers from the series - so this returns the ``nodes`` mapping and
    nothing more, and an unreadable file reads as absent.
    """
    from cadet.dashboard.runs import load_json

    data = load_json(os.path.join(run_dir, DEVICE_SUMMARY)) or {}
    nodes = data.get("nodes")
    return nodes if isinstance(nodes, dict) else {}


def read_device_series(run_dir: str) -> Dict[str, List[Series]]:
    """Every device series in a run directory, grouped by node.

    Returns ``{node: [Series, ...]}``, empty when the run has no collector file
    with any sample in it. A collector file that exists with no rows contributes
    no series, which is the same answer as a collector that never ran: in both
    cases there is nothing to plot, and the view says so.
    """
    found: Dict[str, List[Series]] = {}
    for collector, key_column, quantities in COLLECTORS:
        prefix = "{}_".format(collector)
        for path in sorted(glob.glob(os.path.join(run_dir, "{}*.csv".format(prefix)))):
            node = node_suffix(path, prefix, ".csv")
            if node == LEGACY_NODE:
                continue
            rows = load_rows(path)
            if not rows:
                continue
            series = _series_from_rows(collector, key_column, quantities, rows)
            if series:
                found.setdefault(node, []).extend(series)
    # A node key with no series would read to a caller as a node that has
    # samples, and the device view would render it an empty heading. A file whose
    # every value was unparseable is the same answer as no file: no samples.
    return {node: series for node, series in found.items() if series}


def _series_from_rows(
    collector: str,
    key_column: Optional[str],
    quantities: Any,
    rows: List[Dict[str, str]],
) -> List[Series]:
    """Turn one collector CSV's rows into one series per (key, quantity).

    The key column is part of the series identity, not a label on it: one sample
    of a two-GPU node is two rows, and merging them would plot one card's
    utilisation as if it were the node's.
    """
    groups = {}  # type: Dict[str, List[Dict[str, str]]]
    for row in rows:
        key = str(row.get(key_column) or "") if key_column else ""
        if key_column and collector == "bandwidth":
            # A link is measured in each direction, so the direction is part of
            # the key too; a bidirectional link is two series, not one.
            key = "{} {}".format(key, row.get("direction") or "").strip()
        groups.setdefault(key, []).append(row)

    out = []  # type: List[Series]
    for key in sorted(groups):
        group = groups[key]
        clocks = [_number(row.get("t_monotonic")) for row in group]
        origin = next((c for c in clocks if c is not None), None)
        for column, label, unit in quantities:
            series = Series(collector, label, unit, key=key)
            for clock, row in zip(clocks, group):
                value = _number(row.get(column))
                if clock is None or value is None or origin is None:
                    continue
                series.t.append(clock - origin)
                series.y.append(value)
            if series.samples:
                out.append(series)
    return out


def cumulative_totals(run_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Per-node run totals from the cumulative collector columns.

    ``energy_j_total`` and ``bytes_total`` are running totals, so the run's value
    is the **last** row's, not the column's sum. Read separately from the series
    for that reason.
    """
    found = {}  # type: Dict[str, List[Dict[str, Any]]]
    for collector, columns in sorted(CUMULATIVE.items()):
        prefix = "{}_".format(collector)
        for path in sorted(glob.glob(os.path.join(run_dir, "{}*.csv".format(prefix)))):
            node = node_suffix(path, prefix, ".csv")
            if node == LEGACY_NODE:
                continue
            rows = load_rows(path)
            if not rows:
                continue
            for column, label, unit in columns:
                values = [v for v in (_number(row.get(column)) for row in rows) if v is not None]
                if values:
                    found.setdefault(node, []).append(
                        {"label": label, "unit": unit, "value": values[-1]}
                    )
    return found
