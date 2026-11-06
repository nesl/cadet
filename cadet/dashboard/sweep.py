"""Group run directories into a sweep of placement x network condition.

What this is for:
    A sweep is a matrix of cells, and the dashboard's sweep view is that matrix
    plus the crossover plot. The sweep's own output format is not written yet, so
    this derives the matrix from the run directories that exist, by reading each
    run's resolved ``config.yaml``. When the sweep format lands, only
    :func:`group_runs` changes: it is deliberately the single function the whole
    view goes through, and everything downstream sees :class:`SweepTable`.

What it contains:
    - :func:`placement_of`, :func:`network_of`: the two cell coordinates, each
      derived from one part of the config.
    - :func:`swept_value`: the numeric position of a cell on the swept network
      axis, which is the crossover plot's x.
    - :class:`SweepCell` / :class:`SweepTable`: the grouped result.
    - :func:`group_runs`: the grouping rule, in one place.

How it fits into CADET:
    Reads :class:`cadet.dashboard.runs.RunRecord` values and nothing else.
    Called by :mod:`cadet.dashboard.render` for the sweep view and by
    :mod:`cadet.dashboard.server` to decide whether a sweep is still in
    progress. Writes nothing.

How to configure it:
    Nothing. The grouping follows the two axes CADET's sweep configs actually
    sweep (``app.params.placement`` or a pipeline stage's ``node``, crossed with
    the whole ``links`` list), so a tree produced by one of those configs groups
    without being told the axes.
"""

import os
import statistics
from typing import Any, Dict, List, Optional, Tuple

from cadet.dashboard.runs import RunRecord, load_json

# A link with no network model shaped at all. Named rather than blank so a cell
# whose runs are unshaped is distinguishable from one whose config was unread.
NO_NETWORK = "unshaped"

# Node roles that can host the compute a placement names. Used only when nothing
# states the placement, to fall back on "where the server is".
COMPUTE_ROLE = "server"


def placement_of(record: RunRecord) -> str:
    """The cell's placement coordinate: where this run put its compute.

    Three sources, in the order they are authoritative:

    1. ``app.params.placement``, when the app declares one. This is the escape
       hatch a flat app uses to carry placement as a parameter, and the axis the
       sweep configs actually sweep.
    2. The pipeline stages' nodes, when the run is a pipeline: the placement
       *is* which node each stage runs on, so the coordinate is the stage-to-node
       mapping with the origin stage dropped (it is on the source by definition).
    3. The compute node's name and device, for a flat app that declares neither.
    """
    declared = record.app_params.get("placement")
    if declared not in (None, ""):
        return str(declared)
    if record.stages:
        pairs = [
            "{}@{}".format(stage.get("name"), stage.get("node"))
            for stage in record.stages[1:]
            if stage.get("node")
        ]
        if pairs:
            return ", ".join(pairs)
    for node in record.nodes:
        if node.get("role") == COMPUTE_ROLE:
            device = node.get("device")
            return "{}/{}".format(node.get("name"), device) if device else str(node.get("name"))
    return "unknown"


def _distribution_label(distribution: Dict[str, Any]) -> str:
    """One-line label for a link's delay distribution.

    Each kind is labelled by the parameters that *define* it rather than by a
    single derived number, because two distributions with the same median are a
    different network condition and must not collapse into one cell.
    """
    kind = str(distribution.get("kind") or "")
    if kind == "constant":
        return "{:g}ms".format(float(distribution.get("value_ms") or 0.0))
    if kind == "normal":
        return "{:g}+-{:g}ms".format(
            float(distribution.get("mean_ms") or 0.0), float(distribution.get("std_ms") or 0.0)
        )
    if kind == "lognormal":
        return "p50 {:g}ms p99 {:g}ms".format(
            float(distribution.get("p50_ms") or 0.0), float(distribution.get("p99_ms") or 0.0)
        )
    if kind == "trace":
        return "trace"
    return kind or "?"


def access_links(record: RunRecord) -> List[Dict[str, Any]]:
    """The links the run's *sources* send over, which is the condition it ran in.

    Not the whole ``links`` list, which was the first rule here and put a sweep's
    cells in different columns depending on its topology. A cloud placement
    declares a backhaul hop as well as an access hop, so labelling by every link
    gave a cloud run at a given access quality a different coordinate from an
    edge run at the same one, and a matrix meant to be three columns wide came
    out nine and mostly empty.

    A backhaul is a property of the topology; the access network is the
    condition. So the links out of a client node are what name the cell, and
    both placements at one access quality land in one column. When no node is a
    client, or no link leaves one, every link is used - a record that cannot
    identify its sources should still be grouped by something it did state.
    """
    clients = {
        node.get("name")
        for node in record.nodes
        if isinstance(node, dict) and node.get("role") == "client"
    }
    if clients:
        from_clients = [link for link in record.links if link.get("from") in clients]
        if from_clients:
            return from_clients
    return list(record.links)


def network_of(record: RunRecord) -> str:
    """The cell's network-condition coordinate: the shaping this run ran under.

    Built from the access links (see :func:`access_links`): they are declared one
    per client with identical parameters, so the condition is the set of distinct
    access qualities rather than one link's fields. A run with no links reads as
    :data:`NO_NETWORK`, which is a real condition (the unshaped baseline) and not
    a missing value.
    """
    if not record.links:
        return NO_NETWORK
    qualities = []
    for link in access_links(record):
        network = link.get("network") if isinstance(link.get("network"), dict) else {}
        if not network:
            qualities.append(NO_NETWORK)
            continue
        parts = [_distribution_label(network.get("distribution") or {})]
        loss = network.get("loss")
        if loss:
            parts.append("{:g}% loss".format(float(loss) * 100.0))
        bandwidth = network.get("bandwidth_mbps")
        if bandwidth:
            parts.append("{:g} Mbps".format(float(bandwidth)))
        medium = network.get("medium")
        if medium and medium != "wired":
            parts.append(str(medium))
        qualities.append(", ".join(parts))
    distinct = sorted(set(qualities))
    return " | ".join(distinct)


def swept_value(record: RunRecord) -> Optional[float]:
    """The cell's position on the swept network axis, in milliseconds.

    The crossover plot needs a number to put a cell at, and the axis a CADET
    sweep varies is the link's delay. The representative value is the median of
    each access link's central delay: the mean for a normal, the p50 for a
    lognormal, the value for a constant. Access links only, for the reason
    :func:`access_links` gives - including a backhaul here would move a cloud
    run along the x axis relative to an edge run at the same access quality, so
    the two would not be above each other on the plot that compares them.
    ``None`` when nothing in the run states a delay, which the plot renders as a
    categorical position rather than guessing zero.
    """
    centres = []
    for link in access_links(record):
        network = link.get("network") if isinstance(link.get("network"), dict) else {}
        distribution = network.get("distribution") if isinstance(network, dict) else None
        if not isinstance(distribution, dict):
            continue
        for key in ("mean_ms", "p50_ms", "value_ms"):
            if distribution.get(key) is not None:
                centres.append(float(distribution[key]))
                break
    return statistics.median(centres) if centres else None


class SweepCell(object):
    """One cell of the matrix: the repetitions of one config at one coordinate.

    A cell holds several runs because a sweep repeats each config, so every
    number it reports is a median across repetitions and ``runs`` is kept so the
    view can link to each one.
    """

    def __init__(self, placement: str, network: str, runs: List[RunRecord]) -> None:
        self.placement = placement
        self.network = network
        self.runs = runs

    @property
    def axis(self) -> Optional[float]:
        """The cell's position on the swept axis; see :func:`swept_value`."""
        values = [v for v in (swept_value(run) for run in self.runs) if v is not None]
        return statistics.median(values) if values else None

    @property
    def in_progress(self) -> bool:
        """Whether any repetition of this cell has not finished."""
        return any(run.in_progress for run in self.runs)

    def metric(self) -> Tuple[str, Optional[float], str]:
        """The cell's task metric as ``(name, median, unit)`` across repetitions."""
        name, _value, unit = self.runs[0].task_metric() if self.runs else ("task metric", None, "")
        values = [run.task_metric()[1] for run in self.runs]
        values = [v for v in values if v is not None]
        return name, (statistics.median(values) if values else None), unit

    def latency_ms(self) -> Optional[float]:
        """The cell's measured latency, median across repetitions."""
        values = [v for v in (run.end_to_end_ms() for run in self.runs) if v is not None]
        return statistics.median(values) if values else None


class SweepTable(object):
    """The grouped matrix: placements down, network conditions across.

    ``cells`` is sparse on purpose. A sweep whose matrix has a hole - a cell
    nothing was run for, or one whose runs were lost - shows the hole rather
    than an invented number.
    """

    def __init__(
        self,
        root: str,
        placements: List[str],
        networks: List[str],
        cells: Dict[Tuple[str, str], SweepCell],
        notes: List[str],
        name: str = "",
        progress: Optional[Dict[str, Any]] = None,
    ) -> None:
        # What the sweep itself says about whether it is still going, read from
        # the marker beside its run directories. ``None`` for a tree whose runs
        # were not produced by `cadet sweep`, or one written before the marker
        # existed, and the views then fall back to the runs alone.
        self.progress = progress or {}
        # Which sweep this is, when a tree holds more than one; see
        # :func:`group_sweeps`. Empty for a tree read as a single sweep.
        self.name = name
        self.root = root
        self.placements = placements
        self.networks = networks
        self.cells = cells
        self.notes = notes

    def cell(self, placement: str, network: str) -> Optional[SweepCell]:
        """The cell at a coordinate, or None when the matrix has a hole there."""
        return self.cells.get((placement, network))

    @property
    def runs(self) -> List[RunRecord]:
        """Every run in the table, sorted by run id."""
        found = [run for cell in self.cells.values() for run in cell.runs]
        return sorted(found, key=lambda run: run.run_id)

    @property
    def in_progress(self) -> bool:
        """Whether this sweep has not finished.

        What the live view polls, and two questions rather than one. A run is in
        progress when its directory exists with no marker yet. A *sweep* is in
        progress for longer than that: between two cells every run it has
        produced is complete and more are coming, which from the runs alone is
        indistinguishable from a finished sweep. So the sweep's own marker is
        asked first, and the runs are the fallback for a tree that has none.
        """
        state = self.progress.get("state")
        if state:
            return state == "running"
        return any(run.in_progress for run in self.runs)

    @property
    def progress_detail(self) -> str:
        """``"3 of 6 cells"`` when the sweep recorded its progress, else empty."""
        cells = self.progress.get("cells")
        done = self.progress.get("completed")
        if not cells or done is None:
            return ""
        return "{} of {} cells".format(done, cells)

    def series(self) -> List[Dict[str, Any]]:
        """One series per placement for the crossover plot, sorted by the axis.

        Each point is ``(x, y)`` where x is the cell's swept value (its index
        among the network conditions when no run states a delay) and y is the
        cell's task metric. A cell with no metric contributes no point, so a
        hole in the matrix is a gap in the line rather than a drop to zero.
        """
        axes = {}  # type: Dict[str, Optional[float]]
        for index, network in enumerate(self.networks):
            found = [
                self.cells[(p, network)].axis
                for p in self.placements
                if (p, network) in self.cells and self.cells[(p, network)].axis is not None
            ]
            axes[network] = found[0] if found else float(index)
        out = []
        for placement in self.placements:
            points = []
            for network in self.networks:
                cell = self.cells.get((placement, network))
                if cell is None:
                    continue
                _name, value, _unit = cell.metric()
                if value is None:
                    continue
                points.append((axes[network] or 0.0, value, network))
            if points:
                out.append({"name": placement, "points": sorted(points)})
        return out


def group_runs(root: str, records: List[RunRecord], name: str = "") -> SweepTable:
    """Group run records into the placement x network matrix. The grouping rule.

    One function on purpose: the sweep's own output format is not written yet,
    and when it is, the rule moves to reading that file and nothing downstream
    changes. Until then the rule is:

    * a run's cell coordinate is ``(placement_of(run), network_of(run))``;
    * runs sharing a coordinate are that cell's repetitions, which for a real
      sweep is exactly the set sharing a config hash, since the hash is a
      function of the config the coordinate is derived from;
    * the axis order is the swept value where the runs state one and the first
      appearance otherwise, so a matrix of conditions with no declared delay
      still comes out in a stable order.

    A run whose ``config.yaml`` was unreadable has no coordinate to derive, so it
    lands in no cell and is named in ``notes`` instead of being guessed at.
    """
    cells = {}  # type: Dict[Tuple[str, str], List[RunRecord]]
    notes = []  # type: List[str]
    for record in records:
        if not record.nodes and not record.links and not record.app:
            notes.append("{}: no readable config, not placed in the matrix".format(record.run_id))
            continue
        key = (placement_of(record), network_of(record))
        cells.setdefault(key, []).append(record)

    placements = sorted({key[0] for key in cells})
    # Network conditions order by the swept delay when the runs state one, so
    # the crossover plot's x axis is monotone; by first appearance otherwise.
    order = {}  # type: Dict[str, Tuple[float, float]]
    for index, key in enumerate(sorted(cells)):
        network = key[1]
        if network in order:
            continue
        values = [v for v in (swept_value(r) for r in cells[key]) if v is not None]
        order[network] = (values[0] if values else float("inf"), float(index))
    networks = sorted(order, key=lambda name: order[name])

    built = {
        key: SweepCell(key[0], key[1], sorted(runs, key=lambda run: run.run_id))
        for key, runs in cells.items()
    }
    return SweepTable(root, placements, networks, built, notes, name=name)


# What a sweep with nothing to name it by is called, so it is still grouped.
UNNAMED_SWEEP = "(unnamed)"


def group_sweeps(root: str, records: List[RunRecord]) -> List[SweepTable]:
    """One table per sweep in the tree, in a stable order. **A sweep is a
    directory of runs.**

    Why a matrix per sweep at all: a tree can hold several, and they are
    separate experiments rather than more cells of one. Merged into a single
    matrix they averaged together, because the coordinate is (placement,
    network) and the thing that differed between them - the sensor count, the
    link backend, a queue policy - is not part of it. A cell then held a
    two-sensor run and a ten-sensor one and reported the mean as a measurement,
    and nothing said so.

    Why the grouping key is the containing directory and not the experiment
    name, which was the first rule here: a wrapper that runs several `cadet
    sweep` invocations and gathers their runs into one directory does so exactly
    because they belong in one matrix - the placement sweep gathers an edge
    sweep and a cloud sweep so the three placements can be compared, and those
    carry two different experiment names. Keying on the name split them back
    apart and undid the wrapper's whole purpose. Keying on the directory follows
    whoever laid the tree out: `cadet sweep` writes one experiment per
    directory, and a wrapper that wants them together puts them together.
    """
    by_directory = {}  # type: Dict[str, List[RunRecord]]
    for record in records:
        parent = os.path.dirname(os.path.normpath(record.path))
        try:
            key = os.path.relpath(parent, root)
        except ValueError:
            key = parent
        by_directory.setdefault(key, []).append(record)
    tables = []
    for key in sorted(by_directory):
        found = by_directory[key]
        # The label is the directory, except when the root *is* the sweep's
        # directory ("." has nothing to say), where the experiment the runs
        # declare is the better name.
        label = key if key not in (".", "") else (found[0].experiment or UNNAMED_SWEEP)
        directory = os.path.dirname(os.path.normpath(found[0].path))
        table = group_runs(root, found, name=label)
        table.progress = read_progress(directory)
        tables.append(table)
    return tables


# Written by `cadet sweep` beside its run directories; see
# cadet.orchestrator.runner.SWEEP_MARKER.
SWEEP_MARKER = "sweep.json"


def read_progress(directory: str) -> Dict[str, Any]:
    """Read a sweep's own progress marker, or ``{}`` when there is none.

    Absent is the ordinary case and not an error: a tree of runs that were not
    produced by one `cadet sweep`, or produced before the marker existed, has
    none, and the views fall back to asking the runs.
    """
    data = load_json(os.path.join(directory, SWEEP_MARKER))
    return data if isinstance(data, dict) else {}
