"""Read a run directory into the record the dashboard views render.

What this is for:
    The dashboard reads run directories it did not produce, on a machine with no
    network, and is demonstrated live. So every read here degrades: a file that
    is absent, truncated or malformed becomes a note on the record and never an
    exception. A view that says "no link report in this run" is correct; a
    traceback in front of an audience is not.

What it contains:
    - :data:`RUN_MARKER`, :data:`TERMINAL_STATUSES`: what makes a directory a
      run and when that run has finished.
    - :func:`discover_runs`: every run directory under a root.
    - :class:`RunRecord` / :func:`read_run`: one run directory, read file by
      file, with ``notes`` carrying whatever was missing.
    - the per-file readers (``_read_marker``, ``_read_config``,
      ``_read_link_reports``, ``_read_outcomes``, ``_read_server_metrics``,
      ``_read_frames``, ``_read_sidecars``).

How it fits into CADET:
    Reads only what the roles and the orchestrator already wrote
    (``run.json``, ``config.yaml``, ``client_outcome_<node>.json``,
    ``link_report_<node>.json``, ``server_metrics_<node>.csv``,
    ``*_latency_location_log.csv``, ``*_summary.json``). Called by
    :mod:`cadet.dashboard.sweep` for the sweep grouping and by
    :mod:`cadet.dashboard.render` for the run view. Writes nothing: the
    dashboard reads a run, it does not run or change one.

    The per-node file names carry the node suffix the roles now write. The old
    unsuffixed ``link_report.json`` / ``client_outcome.json`` are still accepted
    from run directories produced before the rename, attributed to the node
    ``"-"``, so an existing output tree does not read as a run with no link.

How to configure it:
    Nothing. The file names are the ones the roles write.
"""

import csv
import glob
import json
import os
import re
import statistics
from typing import Any, Dict, List, Optional, Tuple

import yaml

# The marker the orchestrator writes when a run reaches its teardown. Its
# absence is what "in progress" means to the live view, so it is also the test
# for whether a run has finished.
RUN_MARKER = "run.json"

# The statuses the orchestrator writes into the marker. Anything else - and a
# marker that is absent entirely - counts as still in progress.
TERMINAL_STATUSES = ("complete", "failed")

# A directory is a run when it holds either of these: the marker (a finished
# run) or the resolved config the orchestrator saves before starting (a run in
# progress, which the live view has to be able to show).
RUN_EVIDENCE = (RUN_MARKER, "config.yaml")

# Directories under an output root that are not runs: the shared sinks the roles
# write frames and logs into. Skipped by name because they sit beside the run
# directories, and a directory of JPEGs is not a run.
SKIP_DIRS = ("server_detections", "client_frames", "logs")

# ``<base>-r<rep>``: the orchestrator run id. The base is the config cell hash,
# which is what repetitions of one cell group by.
REP_SUFFIX = re.compile(r"^(?P<base>.+)-r(?P<rep>\d+)$")

# Node attributed to a legacy unsuffixed per-node file. Not a real node name, and
# deliberately not one, so a view never prints it as a topology node.
LEGACY_NODE = "-"

# The orchestrator's per-node device roll-up. It matches the ``*_summary.json``
# glob the application sidecars use, so it is named here and skipped there: its
# leaves are device statistics, and flattening them in beside an application's
# own metrics would file a node's mean wattage as a task metric.
DEVICE_SUMMARY = "device_summary.json"

# Columns of the server's per-request CSV that make up the per-stage latency the
# run view shows, in the order a request passes through them.
SERVER_STAGES = ("queue_wait_s", "inference_s")

_LOC = re.compile(r"-?\d+(?:\.\d+)?")


def parse_run_id(run_id: str) -> Tuple[str, Optional[int]]:
    """Split ``<base>-r<rep>`` into its config cell and repetition index.

    A run id without the suffix is its own cell with no repetition index, which
    is what a single hand-started run looks like.
    """
    match = REP_SUFFIX.match(run_id or "")
    if not match:
        return run_id or "", None
    return match.group("base"), int(match.group("rep"))


def discover_runs(root: str) -> List[str]:
    """Return every run directory under ``root``, sorted.

    ``root`` itself counts, so pointing the dashboard at a single run directory
    shows that one run. A directory qualifies on :data:`RUN_EVIDENCE` rather
    than on the marker alone: a run in progress has no marker yet, and the live
    view exists to show exactly those.
    """
    found = []  # type: List[str]
    if not os.path.isdir(root):
        return found
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        if any(name in filenames for name in RUN_EVIDENCE):
            found.append(dirpath)
            # A run directory holds no nested runs; not descending also keeps a
            # role's own output subdirectory from reading as one.
            dirnames[:] = []
    return sorted(found)


def _number(value: Any) -> Optional[float]:
    """Coerce a CSV/JSON field to a float, or None when it is not a number.

    An empty CSV cell is normal - a collector source that does not produce a
    column writes one - so it reads as absent rather than as zero.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _mapping(value: Any) -> Dict[str, Any]:
    """A config section as a mapping, or an empty one.

    Every section is checked rather than trusted. The config in a run directory
    is the resolved one the orchestrator saved, so it is normally well formed -
    but it is a file on disk in a tree the dashboard did not produce, and a
    section that is a scalar or a list must read as an absent section rather
    than reach an attribute access.
    """
    return value if isinstance(value, dict) else {}


def _sequence(value: Any) -> List[Any]:
    """A config section as a list, or an empty one. See :func:`_mapping`."""
    return value if isinstance(value, list) else []


def _loc_y(value: Any) -> Optional[float]:
    """Second coordinate of a ``"(x, y, z)"`` location cell, or None."""
    numbers = _LOC.findall(str(value or ""))
    return float(numbers[1]) if len(numbers) >= 2 else None


def load_json(path: str) -> Optional[Dict[str, Any]]:
    """Load a JSON object, or None when it is absent or unreadable."""
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def load_rows(path: str) -> Optional[List[Dict[str, str]]]:
    """Read a CSV as dict rows, or None when it is absent or unreadable.

    An empty list is a real answer, and a different one from None: a file with a
    header and no rows means the writer died before its first sample, which is
    "no samples" rather than "no file".
    """
    try:
        with open(path, newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error):
        return None


def node_suffix(path: str, prefix: str, ext: str) -> str:
    """Node name out of ``<prefix><node><ext>``, or :data:`LEGACY_NODE`."""
    name = os.path.basename(path)
    if name.startswith(prefix) and name.endswith(ext):
        return name[len(prefix):-len(ext)] or LEGACY_NODE
    return LEGACY_NODE


def per_node_paths(run_dir: str, prefix: str, ext: str) -> List[Tuple[str, str]]:
    """``(node, path)`` for every ``<prefix><node><ext>`` in the run directory.

    Falls back to the legacy unsuffixed ``<prefix><ext>`` when no suffixed file
    is there, so a run directory written before the per-node rename still reads.
    The suffixed files win when both exist, because they are the ones that carry
    whose record is whose.
    """
    paths = sorted(glob.glob(os.path.join(run_dir, "{}*{}".format(prefix, ext))))
    found = [(node_suffix(p, prefix, ext), p) for p in paths]
    if found:
        return found
    legacy = os.path.join(run_dir, "{}{}".format(prefix.rstrip("_"), ext))
    return [(LEGACY_NODE, legacy)] if os.path.exists(legacy) else []


class RunRecord(object):
    """One run directory: its identity, status, config facts and measurements.

    ``notes`` is the degradation channel. Every reader that finds nothing
    appends a sentence here, and the views print those sentences instead of the
    panel they could not fill, which is what keeps a run directory missing a
    file from being an error.
    """

    def __init__(self, path: str, run_id: str, base: str, rep: Optional[int]) -> None:
        self.path = path
        self.run_id = run_id
        self.base = base
        self.rep = rep
        self.status = "in-progress"
        self.reason = ""
        self.detail = ""
        self.phases = {}  # type: Dict[str, Optional[float]]
        self.experiment = ""
        self.app = ""
        self.nodes = []  # type: List[Dict[str, Any]]
        self.links = []  # type: List[Dict[str, Any]]
        self.routes = []  # type: List[Dict[str, Any]]
        self.channels = []  # type: List[Dict[str, Any]]
        self.stages = []  # type: List[Dict[str, Any]]
        self.app_params = {}  # type: Dict[str, Any]
        self.transport = {}  # type: Dict[str, Any]
        self.collectors = []  # type: List[str]
        self.link_reports = {}  # type: Dict[str, Dict[str, Any]]
        self.outcomes = {}  # type: Dict[str, Dict[str, Any]]
        self.server = {}  # type: Dict[str, Dict[str, Any]]
        self.frames = {}  # type: Dict[str, Any]
        self.sidecars = {}  # type: Dict[str, Any]
        self.notes = []  # type: List[str]

    @property
    def in_progress(self) -> bool:
        """Whether this run has not reached a terminal status.

        The live view's whole predicate: no marker, or a marker whose status is
        not one the orchestrator writes at teardown.
        """
        return self.status not in TERMINAL_STATUSES

    @property
    def label(self) -> str:
        """Short name for this run in a table (``r3``, or its run id)."""
        return "r{}".format(self.rep) if self.rep is not None else self.run_id

    def task_metric(self) -> Tuple[str, Optional[float], str]:
        """The run's headline task metric as ``(name, value, unit)``.

        Generic by construction: the first of the candidates the run actually
        wrote, rather than a metric named for one application. ``None`` for the
        value means the run wrote none of them, which a cell renders as a dash.

        An application with no per-frame record of the braking shape still has a
        headline, in its own ``*_summary.json``: the **first** numeric entry
        there is taken, with its unit read off the key's suffix. First rather
        than a name this module knows, because the point of the sidecar is that
        the dashboard does not know the application - so the application decides
        which of its numbers leads by writing it first.
        """
        candidates = [
            ("brake reaction", self.frames.get("task_latency_s"), "s"),
            ("stop time", self.frames.get("stop_t_s"), "s"),
        ]
        for name, value, unit in candidates:
            if value is not None:
                return name, value, unit
        for key, value in self.sidecars.items():
            return _metric_name(key), value, _metric_unit(key)
        return "task metric", None, ""

    def end_to_end_ms(self) -> Optional[float]:
        """Median measured one-way latency across the run's link reports.

        The sweep table's latency column. Taken from the link reports because
        that is the number measured on the wire; a run with none falls back to
        the server's own request latency, and a run with neither has none.
        """
        values = []
        for report in self.link_reports.values():
            value = _number(((report.get("measured") or {}).get("latency_ms") or {}).get("p50"))
            if value is not None:
                values.append(value)
        if values:
            return statistics.median(values)
        totals = [
            node["total_s"] for node in self.server.values() if node.get("total_s") is not None
        ]
        return statistics.median(totals) * 1000.0 if totals else None


def read_run(path: str) -> RunRecord:
    """Read one run directory. Never raises on a missing or malformed file."""
    marker = load_json(os.path.join(path, RUN_MARKER)) or {}
    run_id = str(marker.get("run_id") or os.path.basename(path.rstrip(os.sep)))
    base, rep = parse_run_id(run_id)
    record = RunRecord(path, run_id, base, rep)
    _read_marker(record, marker)
    _read_config(record)
    _read_link_reports(record)
    _read_outcomes(record)
    _read_server_metrics(record)
    _read_frames(record)
    _read_sidecars(record)
    return record


def _read_marker(record: RunRecord, marker: Dict[str, Any]) -> None:
    """Status, reason and the run's phase durations from ``run.json``."""
    if not marker:
        record.notes.append("no {}: this run has not reached its teardown".format(RUN_MARKER))
        return
    record.status = str(marker.get("status") or "unknown")
    record.reason = str(marker.get("reason") or "")
    record.detail = str(marker.get("detail") or "")
    for key in ("startup_s", "active_s", "teardown_s"):
        record.phases[key] = _number(marker.get(key))


def _read_config(record: RunRecord) -> None:
    """Topology and the config facts a measurement has to be read against.

    The resolved config the run saved, not the one a user wrote: it is already
    expanded, so what is here is what this run actually ran.
    """
    path = os.path.join(record.path, "config.yaml")
    try:
        with open(path) as handle:
            config = yaml.safe_load(handle)
    except (OSError, ValueError, UnicodeDecodeError, yaml.YAMLError):
        record.notes.append("no readable config.yaml: no topology for this run")
        return
    if not isinstance(config, dict):
        record.notes.append("config.yaml is not a mapping: no topology for this run")
        return
    record.experiment = str(_mapping(config.get("experiment")).get("name") or "")
    app = _mapping(config.get("app"))
    record.app = str(app.get("plugin") or "")
    record.app_params = _mapping(app.get("params"))
    record.nodes = [n for n in _sequence(config.get("nodes")) if isinstance(n, dict)]
    record.links = [link for link in _sequence(config.get("links")) if isinstance(link, dict)]
    record.routes = [r for r in _sequence(config.get("routes")) if isinstance(r, dict)]
    record.channels = [c for c in _sequence(config.get("channels")) if isinstance(c, dict)]
    stages = _sequence(_mapping(config.get("pipeline")).get("stages"))
    record.stages = [s for s in stages if isinstance(s, dict)]
    record.transport = _mapping(config.get("transport"))
    collectors = _sequence(_mapping(config.get("metrics")).get("collectors"))
    record.collectors = [str(name) for name in collectors]


def _read_link_reports(record: RunRecord) -> None:
    """``link_report_<node>.json``: what was configured and what was measured."""
    found = per_node_paths(record.path, "link_report_", ".json")
    if not found:
        record.notes.append("no link report in this run")
        return
    for node, path in found:
        data = load_json(path)
        if data is None:
            record.notes.append("link report for node {} is unreadable".format(node))
            continue
        record.link_reports[node] = data


def _read_outcomes(record: RunRecord) -> None:
    """``client_outcome_<node>.json``: each client's own verdict on its run."""
    found = per_node_paths(record.path, "client_outcome_", ".json")
    if not found:
        record.notes.append("no client outcome in this run")
        return
    for node, path in found:
        data = load_json(path)
        if data is None:
            record.notes.append("client outcome for node {} is unreadable".format(node))
            continue
        record.outcomes[node] = {
            "ok": bool(data.get("ok")),
            "detail": str(data.get("detail") or ""),
        }


def _read_server_metrics(record: RunRecord) -> None:
    """``server_metrics_<node>.csv``: per-stage latency, reduced within the run.

    Median and p99 per stage, because one run holds hundreds of requests and the
    question the run view answers is about the run rather than the request. The
    max is kept separately as ``inference_s`` peak: with ``models.warmup`` off
    (the examples' faithful default) the first inference pays the load cost, so
    a single mean would hide both it and the steady state.
    """
    found = per_node_paths(record.path, "server_metrics_", ".csv")
    if not found:
        record.notes.append("no server per-request record in this run")
        return
    for node, path in found:
        rows = load_rows(path)
        if rows is None:
            record.notes.append("server record for node {} is unreadable".format(node))
            continue
        entry = {"requests": len(rows), "stages": [], "total_s": None}  # type: Dict[str, Any]
        if not rows:
            record.notes.append("server record for node {} has no requests".format(node))
            record.server[node] = entry
            continue
        entry["replies"] = sum(
            1 for row in rows if str(row.get("replied", "")).strip().lower() == "true"
        )
        totals = []
        for column in SERVER_STAGES:
            values = [v for v in (_number(row.get(column)) for row in rows) if v is not None]
            if not values:
                continue
            ordered = sorted(values)
            entry["stages"].append(
                {
                    "name": column[:-2].replace("_", " "),
                    "median_s": statistics.median(ordered),
                    "p99_s": ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))],
                    "max_s": ordered[-1],
                    "samples": len(ordered),
                }
            )
            totals.append(statistics.median(ordered))
        if totals:
            entry["total_s"] = sum(totals)
        record.server[node] = entry


def _read_frames(record: RunRecord) -> None:
    """Per-frame CSV: the task metric, the send pacing and the frame count.

    The task columns carry one sample per run (the server replies on the event),
    so they are read as the run's value rather than reduced within it.
    """
    matches = sorted(glob.glob(os.path.join(record.path, "*_latency_location_log.csv")))
    if not matches:
        record.notes.append("no per-frame record in this run")
        return
    if len(matches) > 1:
        record.notes.append(
            "several per-frame records; read {}".format(os.path.basename(matches[0]))
        )
    rows = load_rows(matches[0])
    if rows is None:
        record.notes.append("per-frame record is unreadable")
        return
    record.frames["rows"] = len(rows)
    if not rows:
        record.notes.append("per-frame record has no rows")
        return

    latencies = [
        v for v in (_number(r.get("brake_reaction_latency_s")) for r in rows) if v is not None
    ]
    if latencies:
        record.frames["task_latency_s"] = statistics.median(latencies)
    triggered = [r for r in rows if str(r.get("brake_issued", "")).strip().lower() == "true"]
    record.frames["triggered"] = bool(triggered)
    if triggered:
        first = triggered[0]
        record.frames["trigger_t_s"] = _number(first.get("server_response_receive_time"))
        record.frames["stop_t_s"] = _number(first.get("stopping_time"))
        record.frames["stop_y"] = _loc_y(first.get("stopping_loc"))
        record.frames["server_processing_s"] = _number(first.get("server_processing_time"))
    else:
        record.notes.append("per-frame record shows no triggered event")
    speeds = [v for v in (_number(r.get("vehicle_speed_mph")) for r in rows) if v is not None]
    if speeds:
        record.frames["speed_mean"] = statistics.mean(speeds)
    # Send pacing as measured, beside the configured interval: the gap between
    # consecutive sends is what a latency number has to be read against.
    sends = [v for v in (_number(r.get("frame_send_time")) for r in rows) if v is not None]
    gaps = [b - a for a, b in zip(sends, sends[1:]) if b > a]
    if gaps:
        record.frames["send_gap_ms"] = statistics.median(gaps) * 1000.0


# Units a summary key can declare in its own name, longest suffix first so
# "_ms" is not read as "_s".
_KEY_UNITS = (("_ms", "ms"), ("_kb", "kB"), ("_s", "s"), ("_pct", "%"))


def _metric_unit(key: str) -> str:
    """The unit a summary key names in its suffix, or none."""
    for suffix, unit in _KEY_UNITS:
        if key.endswith(suffix):
            return unit
    return ""


def _metric_name(key: str) -> str:
    """The readable name of a summary key, with its unit suffix removed."""
    for suffix, _unit in _KEY_UNITS:
        if key.endswith(suffix):
            key = key[: -len(suffix)]
            break
    return key.replace("_", " ")


def _read_sidecars(record: RunRecord) -> None:
    """App summary sidecars (``*_summary.json``): every numeric leaf, flattened.

    How an application's own metrics reach the run view without the view knowing
    the application. Nothing here is required, so finding none is not a note.
    """
    for path in sorted(glob.glob(os.path.join(record.path, "*_summary.json"))):
        if os.path.basename(path) == DEVICE_SUMMARY:
            continue
        data = load_json(path)
        if data is None:
            record.notes.append("summary {} is unreadable".format(os.path.basename(path)))
            continue
        # Not sorted: the order in the file is the application's statement of
        # which number leads, and task_metric reads the first one.
        for key, value in data.items():
            number = _number(value)
            if number is not None:
                record.sidecars[str(key)] = number
