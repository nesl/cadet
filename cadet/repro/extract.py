"""Read one repetition directory into named metrics plus its measurement context.

What this is for:
    One repetition of a config writes a fixed set of files into its own output
    directory. This module turns that directory into a flat ``{metric: value}``
    mapping and a context mapping, without instrumenting anything: every number
    here is already on disk when the run ends.

What it contains:
    - :class:`Repetition`: one repetition's identity, status, metrics, context
      and the per-file notes explaining anything absent.
    - :func:`read_repetition`: the whole read, file by file.
    - the per-file readers (``_read_run_marker``, ``_read_frame_csv``,
      ``_read_summary_sidecars``, ``_read_link_reports``,
      ``_read_client_outcomes``, ``_read_server_metrics``,
      ``_read_collector_csvs``) and the context reader ``_read_context``.

How it fits into CADET:
    Called by :mod:`cadet.repro.reduce` once per repetition; the reduction only
    ever sees these mappings, so adding a file to a run means adding a reader
    here and nothing else.

How to configure it:
    Nothing: the file names are the ones the roles write
    (``run.json``, ``*_latency_location_log.csv``, ``*_summary.json``,
    ``link_report_<node>.json``, ``client_outcome_<node>.json``,
    ``server_metrics_<node>.csv``, ``<collector>_<node>.csv``) and the resolved
    ``config.yaml`` the run saved beside them.
"""

import csv
import glob
import json
import os
import re
import statistics
from typing import Any, Dict, List, Optional, Tuple

import yaml

# The run marker the orchestrator writes last; its presence is what makes a
# directory a repetition rather than an arbitrary folder.
RUN_MARKER = "run.json"

# ``<base>-r<rep>``: the run id the orchestrator derives per repetition. The
# base is the config cell (its hash), so it is what repetitions group by.
REP_SUFFIX = re.compile(r"^(?P<base>.+)-r(?P<rep>\d+)$")

# Phase durations in the run marker worth reducing; the rest of the marker is
# status, not measurement.
MARKER_NUMERIC = ("startup_s", "active_s", "teardown_s")

# Columns of the server's per-request CSV that are measurements. Reduced within
# a repetition first (median and max), because one repetition holds hundreds of
# requests and the across-repetition question is about the run, not the request.
SERVER_NUMERIC = (
    "queue_wait_s",
    "inference_s",
    "queue_depth_at_enqueue",
    "avg_power_w",
    "energy_j",
)

# Collector CSVs are ``<collector>_<node>.csv``; these are the collectors that
# exist, so a file named for one is read as a sample stream and anything else in
# the directory is left alone.
COLLECTOR_NAMES = ("gpu", "cpu", "energy", "bandwidth")

# Clock/identity columns every collector row carries: not measurements.
COLLECTOR_SKIP = ("t_wall", "t_monotonic", "node", "source")

_LOC = re.compile(r"-?\d+(?:\.\d+)?")


class Repetition:
    """One repetition: where it is, whether it completed, and what it measured.

    ``metrics`` is flat and dotted (``link.client.latency_ms.p50``) so the
    reduction needs no knowledge of which file a number came from. ``notes``
    records what was absent and why, because a metric missing from some
    repetitions is a finding and has to survive into the report.
    """

    def __init__(self, path: str, run_id: str, base: str, rep: Optional[int]):
        self.path = path
        self.run_id = run_id
        self.base = base
        self.rep = rep
        self.status: str = "unknown"
        self.reason: str = ""
        self.detail: str = ""
        self.metrics: Dict[str, float] = {}
        self.context: Dict[str, Any] = {}
        self.notes: List[str] = []

    @property
    def label(self) -> str:
        """Short name for this repetition in a report (``r3``, or its run id)."""
        return "r{}".format(self.rep) if self.rep is not None else self.run_id

    def as_dict(self) -> Dict[str, Any]:
        """JSON-ready view of this repetition."""
        return {
            "run_id": self.run_id,
            "rep": self.rep,
            "path": self.path,
            "status": self.status,
            "reason": self.reason,
            "detail": self.detail,
            "metrics": self.metrics,
            "context": self.context,
            "notes": self.notes,
        }


def parse_run_id(run_id: str) -> Tuple[str, Optional[int]]:
    """Split ``<base>-r<rep>`` into its cell and repetition index.

    A run id without the suffix is its own cell with no repetition index, which
    is what a single hand-started run looks like.
    """
    match = REP_SUFFIX.match(run_id or "")
    if not match:
        return run_id or "", None
    return match.group("base"), int(match.group("rep"))


def _number(value: Any) -> Optional[float]:
    """Coerce a CSV/JSON field to a float, or None if it is not a number.

    Booleans are not numbers here: a ``True`` in ``brake_issued`` is a flag, and
    averaging flags across repetitions would invent a metric.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in ("none", "nan", "null", "-"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _loc_y(value: Any) -> Optional[float]:
    """Pull the Y component out of a ``(x, y, z)`` location field.

    Y is the axis the braking and V2V parity numbers are quoted on (brake y,
    stop y), and the CSV carries the location as the original's string.
    """
    if value is None:
        return None
    parts = _LOC.findall(str(value))
    if len(parts) < 2:
        return None
    try:
        return float(parts[1])
    except ValueError:
        return None


def _put(metrics: Dict[str, float], name: str, value: Optional[float]) -> None:
    """Record a metric only if it has a value (absence must stay visible)."""
    if value is not None:
        metrics[name] = float(value)


def _median_max(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Median and max of a within-repetition sample stream."""
    if not values:
        return None, None
    return statistics.median(values), max(values)


def read_repetition(path: str) -> Repetition:
    """Read one repetition directory. Never raises on a malformed file."""
    marker = _load_json(os.path.join(path, RUN_MARKER)) or {}
    run_id = str(marker.get("run_id") or os.path.basename(path.rstrip(os.sep)))
    base, rep = parse_run_id(run_id)
    item = Repetition(path, run_id, base, rep)
    _read_run_marker(item, marker)
    _read_context(item)
    _read_frame_csv(item)
    _read_summary_sidecars(item)
    _read_link_reports(item)
    _read_client_outcomes(item)
    _read_server_metrics(item)
    _read_collector_csvs(item)
    return item


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    """Load a JSON object, returning None when it is absent or unreadable."""
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _rows(path: str) -> Optional[List[Dict[str, str]]]:
    """Read a CSV as dict rows, or None when it is absent or unreadable."""
    try:
        with open(path, newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error):
        return None


def _read_run_marker(item: Repetition, marker: Dict[str, Any]) -> None:
    """Status, reason and the run's phase durations."""
    if not marker:
        item.notes.append("no {}: status unknown".format(RUN_MARKER))
        return
    item.status = str(marker.get("status") or "unknown")
    item.reason = str(marker.get("reason") or "")
    item.detail = str(marker.get("detail") or "")
    item.context["allow_experimental"] = bool(marker.get("allow_experimental"))
    if marker.get("limit"):
        item.context["limit"] = marker.get("limit")
    for key in MARKER_NUMERIC:
        _put(item.metrics, "run.{}".format(key), _number(marker.get(key)))


def _read_context(item: Repetition) -> None:
    """Read the resolved ``config.yaml`` for what has to be stated beside a number.

    The four things that have cost measurement rounds in this project are all
    config facts, not measurements: the **send interval** (a packet-level link
    measured at or below the shaped round trip reads as the link plus a round
    trip unless ``tcp_nodelay`` is on at both ends), **warm-up** (off in the
    braking examples, so the first inference is slow and lands in the spread),
    the **approach speed** default, and the client loop rate. They are carried
    into the report rather than left for a reader to look up.
    """
    path = os.path.join(item.path, "config.yaml")
    try:
        with open(path) as handle:
            config = yaml.safe_load(handle)
    except (OSError, ValueError):
        item.notes.append("no config.yaml: measurement context unavailable")
        return
    if not isinstance(config, dict):
        item.notes.append("config.yaml is not a mapping: context unavailable")
        return

    experiment = config.get("experiment") or {}
    source = config.get("source") or {}
    item.context["experiment"] = experiment.get("name")
    item.context["app"] = (config.get("app") or {}).get("plugin")
    item.context["seed"] = experiment.get("seed")
    item.context["configured_repetitions"] = experiment.get("repetitions")
    item.context["warmup"] = (config.get("models") or {}).get("warmup")
    item.context["source_kind"] = source.get("kind")
    item.context["source_fps"] = source.get("fps")
    scenario = source.get("scenario") or {}
    if scenario:
        item.context["ego_speed"] = scenario.get("ego_speed")
        item.context["scenario"] = scenario.get("name")
    camera = source.get("camera") or {}
    if camera:
        item.context["camera"] = "{}x{}".format(camera.get("width"), camera.get("height"))

    # Send interval: a channel's stream_fps paces the sends; without one the
    # source's fps does. Reported per channel so a run with two channels does
    # not get one number standing for both.
    intervals = {}
    for channel in config.get("channels") or []:
        if not isinstance(channel, dict):
            continue
        fps = channel.get("stream_fps") or source.get("fps")
        value = _number(fps)
        if value:
            intervals[str(channel.get("name"))] = round(1000.0 / value, 3)
    if intervals:
        item.context["send_interval_ms"] = intervals

    # Nagle: the default is the original's (unset), and the option only helps
    # with both ends on. Recorded per node so "client only" is visible.
    nodelay = {}
    for node in config.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        tuning = node.get("transport_tuning") or {}
        nodelay[str(node.get("name"))] = tuning.get("tcp_nodelay")
    if nodelay:
        item.context["tcp_nodelay"] = nodelay
    item.context["server_delay_ms"] = [
        node.get("extra_processing_ms")
        for node in (config.get("nodes") or [])
        if isinstance(node, dict) and node.get("role") == "server"
    ]


def _read_frame_csv(item: Repetition) -> None:
    """Per-frame CSV: frames sent, the brake/stop sample, and approach speed.

    The brake columns carry one sample per run (the server replies only on
    brake), so they are read as the run's value rather than reduced within it.
    """
    matches = sorted(glob.glob(os.path.join(item.path, "*_latency_location_log.csv")))
    if not matches:
        item.notes.append("no per-frame CSV")
        return
    if len(matches) > 1:
        item.notes.append("several per-frame CSVs; read {}".format(os.path.basename(matches[0])))
    rows = _rows(matches[0])
    if rows is None:
        item.notes.append("per-frame CSV unreadable")
        return
    item.metrics["frames.rows"] = float(len(rows))
    if not rows:
        item.notes.append("per-frame CSV has no rows")
        return

    speeds = [v for v in (_number(r.get("vehicle_speed_mph")) for r in rows) if v is not None]
    if speeds:
        _put(item.metrics, "frames.speed_mph.mean", statistics.fmean(speeds))
        _put(item.metrics, "frames.speed_mph.max", max(speeds))

    latencies = [
        v for v in (_number(r.get("brake_reaction_latency_s")) for r in rows) if v is not None
    ]
    if latencies:
        _put(item.metrics, "frames.brake_reaction_latency_s", statistics.median(latencies))

    braked = [r for r in rows if str(r.get("brake_issued")).strip().lower() == "true"]
    if braked:
        first = braked[0]
        _put(item.metrics, "frames.brake_t_s", _number(first.get("server_response_receive_time")))
        _put(item.metrics, "frames.brake_y", _loc_y(first.get("server_response_loc")))
        _put(item.metrics, "frames.stop_t_s", _number(first.get("stopping_time")))
        _put(item.metrics, "frames.stop_y", _loc_y(first.get("stopping_loc")))
        _put(
            item.metrics,
            "frames.server_processing_s",
            _number(first.get("server_processing_time")),
        )
    else:
        item.notes.append("per-frame CSV records no braked frame")

    # Send-side pacing as measured, beside the configured interval: the gap
    # between consecutive sends is what a latency number has to be read against.
    sends = [v for v in (_number(r.get("frame_send_time")) for r in rows) if v is not None]
    gaps = [b - a for a, b in zip(sends, sends[1:]) if b > a]
    if gaps:
        _put(item.metrics, "frames.send_gap_ms.median", statistics.median(gaps) * 1000.0)


def _read_summary_sidecars(item: Repetition) -> None:
    """App summary sidecars (``*_summary.json``): every numeric leaf, flattened.

    This is how an app's own parity metrics reach the reduction without the
    reduction knowing the app: the V2V client writes leader/follower brake and
    stop times, the closest gap, the loop rate and the frame count here.
    """
    for path in sorted(glob.glob(os.path.join(item.path, "*_summary.json"))):
        data = _load_json(path)
        if data is None:
            item.notes.append("{} unreadable".format(os.path.basename(path)))
            continue
        for key, value in _flatten(data):
            number = _number(value)
            if number is not None:
                item.metrics["summary.{}".format(key)] = number
            elif isinstance(value, str) and value:
                item.context.setdefault("summary", {})[key] = value


def _flatten(data: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    """Flatten nested mappings into dotted keys; lists are left alone.

    A list is not flattened because an index is not a name: a two-element list
    would silently become two metrics whose meaning depends on ordering.
    """
    out: List[Tuple[str, Any]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            name = "{}.{}".format(prefix, key) if prefix else str(key)
            out.extend(_flatten(value, name))
    elif prefix:
        out.append((prefix, data))
    return out


def _node_suffix(path: str, prefix: str, ext: str) -> str:
    """Node name out of ``<prefix><node><ext>`` (the per-node record naming)."""
    name = os.path.basename(path)
    return name[len(prefix) : -len(ext)] if name.endswith(ext) else name


def _read_link_reports(item: Repetition) -> None:
    """``link_report_<node>.json``: the measured link, and what was configured.

    The configured one-way p50 is kept in the context beside the send interval,
    because the pair is what says whether a packet-level measurement sat near
    the shaped round trip.
    """
    paths = sorted(glob.glob(os.path.join(item.path, "link_report_*.json")))
    if not paths:
        item.notes.append("no link report")
        return
    for path in paths:
        node = _node_suffix(path, "link_report_", ".json")
        report = _load_json(path)
        if report is None:
            item.notes.append("link_report_{} unreadable".format(node))
            continue
        measured = report.get("measured") or {}
        for key in ("messages_sent", "messages_received", "unanswered", "unanswered_pct"):
            _put(item.metrics, "link.{}.{}".format(node, key), _number(measured.get(key)))
        latency = measured.get("latency_ms") or {}
        for key in ("samples", "p50", "p99", "mean", "min", "max"):
            _put(item.metrics, "link.{}.latency_ms.{}".format(node, key), _number(latency.get(key)))
        configured = [
            link.get("p50_ms") if isinstance(link, dict) else None
            for link in (report.get("configured_links") or [])
        ]
        configured += [
            route.get("p50_ms") if isinstance(route, dict) else None
            for route in (report.get("configured_routes") or [])
        ]
        configured = [v for v in (_number(c) for c in configured) if v is not None]
        if configured:
            item.context.setdefault("configured_p50_ms", {})[node] = configured
        if measured.get("reply_policy"):
            item.context.setdefault("reply_policy", {})[node] = measured.get("reply_policy")


def _read_client_outcomes(item: Repetition) -> None:
    """``client_outcome_<node>.json``: each client's own verdict on its run."""
    paths = sorted(glob.glob(os.path.join(item.path, "client_outcome_*.json")))
    if not paths:
        item.notes.append("no client outcome record")
        return
    for path in paths:
        node = _node_suffix(path, "client_outcome_", ".json")
        outcome = _load_json(path)
        if outcome is None:
            item.notes.append("client_outcome_{} unreadable".format(node))
            continue
        item.context.setdefault("client_outcome", {})[node] = {
            "ok": bool(outcome.get("ok")),
            "detail": outcome.get("detail"),
        }


def _read_server_metrics(item: Repetition) -> None:
    """``server_metrics_<node>.csv``: per-request rows, reduced within the run.

    Both the median and the max of ``inference_s`` are kept: with
    ``models.warmup`` false (the braking examples' faithful default) the first
    inference pays the JIT cost, so the max is that warm-up and the median is
    the steady state. Averaging them together would hide both.
    """
    paths = sorted(glob.glob(os.path.join(item.path, "server_metrics_*.csv")))
    if not paths:
        item.notes.append("no server per-request CSV")
        return
    for path in paths:
        node = _node_suffix(path, "server_metrics_", ".csv")
        rows = _rows(path)
        if rows is None:
            item.notes.append("server_metrics_{} unreadable".format(node))
            continue
        item.metrics["server.{}.requests".format(node)] = float(len(rows))
        replied = [r for r in rows if str(r.get("replied")).strip().lower() == "true"]
        item.metrics["server.{}.replies".format(node)] = float(len(replied))
        for column in SERVER_NUMERIC:
            values = [v for v in (_number(r.get(column)) for r in rows) if v is not None]
            median, maximum = _median_max(values)
            _put(item.metrics, "server.{}.{}.median".format(node, column), median)
            _put(item.metrics, "server.{}.{}.max".format(node, column), maximum)
        drops = [v for v in (_number(r.get("stale_drops_so_far")) for r in rows) if v is not None]
        if drops:
            _put(item.metrics, "server.{}.stale_drops".format(node), max(drops))


def _read_collector_csvs(item: Repetition) -> None:
    """Collector CSVs, when a config enabled them: the median sample per column.

    The collectors are off by default, so finding none is not a finding; a
    collector file that exists for some repetitions and not others is, and
    shows up as a differing metric count in the report.
    """
    for name in COLLECTOR_NAMES:
        for path in sorted(glob.glob(os.path.join(item.path, "{}_*.csv".format(name)))):
            node = _node_suffix(path, "{}_".format(name), ".csv")
            rows = _rows(path)
            if rows is None:
                item.notes.append("{}_{} unreadable".format(name, node))
                continue
            item.metrics["collector.{}.{}.samples".format(name, node)] = float(len(rows))
            if not rows:
                continue
            for column in rows[0]:
                if column in COLLECTOR_SKIP:
                    continue
                values = [v for v in (_number(r.get(column)) for r in rows) if v is not None]
                if values:
                    _put(
                        item.metrics,
                        "collector.{}.{}.{}.median".format(name, node, column),
                        statistics.median(values),
                    )
