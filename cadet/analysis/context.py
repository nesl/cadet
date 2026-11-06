"""Reduce a run directory, or a tree of them, into the record a model reads.

What this is for:
    ``cadet summarize`` and ``cadet ask`` send a model what a run already wrote.
    A run directory holds hundreds of CSV rows per node and a sweep holds that
    many times its cells, so what is sent is the *reduction* the dashboard
    already computes and never the raw files: a sweep of ten runs must not
    become ten thousand rows. This module is the whole of that reduction and of
    its rendering into text.

What it contains:
    - :func:`build_context`: the entry point; decides whether a path is one run
      or a tree of sweeps, and returns the record either way.
    - :func:`run_context` / :func:`sweep_context`: the two shapes.
    - :func:`render_context`: the record as text, at the largest detail level
      that fits a byte budget.
    - :data:`NODE_FIELDS`, :data:`LINK_FIELDS`, :data:`SECRET_HINTS`: what is
      carried, and what is dropped.

How it fits into CADET:
    Reads no file itself. Every read goes through :mod:`cadet.dashboard.runs`,
    :mod:`cadet.dashboard.sweep` and :mod:`cadet.dashboard.device`, which
    already read every file a run writes, already group a tree into sweeps, and
    already degrade to a note rather than raising on a malformed file. So a
    context built over a half-written run tree says what is missing instead of
    failing, and a run directory has one reader rather than two.

    Called by :mod:`cadet.analysis.session`, which wraps the text in a prompt,
    and by ``cadet summarize --context-only``, which prints it and sends
    nothing. Writes nothing.

How to configure it:
    ``analysis.max_context_bytes`` bounds the rendered text; see
    :func:`render_context` for what is dropped to meet it. Nothing else.

What never reaches the record, by construction:
    * **No credential.** The schema has no field for a key:
      ``analysis.api_key_env`` names an environment variable, and the key is
      read from the environment at the moment of the request and held nowhere
      else. The ``analysis`` section of a run's config is not read here at all,
      and in any free mapping that is read (an app's own ``params``, the
      transport block) a key whose name hints at a secret is dropped - see
      :data:`SECRET_HINTS`.
    * **No address or account.** A node's ``host``, ``port``, ``user``,
      ``ssh_port``, ``ssh_identity``, ``container`` and ``venv`` say how the
      orchestrator reached a machine; they are not facts a measurement is read
      against, and a lab address is not something to hand to a third party. The
      device roll-up keys its bandwidth series by peer address, so those keys
      are rewritten to ``peerN`` (:func:`_scrub_peer`).
"""

import copy
import csv
import math
import os
import re
from typing import Any, Dict, List, Optional

from cadet.dashboard import device as device_reader
from cadet.dashboard.runs import discover_runs, read_run
from cadet.dashboard.sweep import group_sweeps, network_of, placement_of, swept_value

# Node fields the record carries: what the run *was*, never how it was reached.
# See the module docstring for why the connection fields are absent.
NODE_FIELDS = (
    "name",
    "role",
    "device",
    "launcher",
    "scheduler",
    "extra_processing_ms",
    "server_delay_mode",
    "enforce_queue_policy",
    "inbound_block_at",
    "clients",
    "rate_hz",
    "arrival",
    "frames",
)

# Link fields the record carries, and the network-model fields under them. The
# distribution is carried whole rather than as one derived number: two
# distributions with the same median are a different network condition.
LINK_FIELDS = ("name", "from", "to")
NETWORK_FIELDS = ("mode", "medium", "direction", "loss", "bandwidth_mbps")

# The measured fields of a link report. The configured side of that file is
# already in the config facts, so only the measurement is carried: what was
# measured is exactly what the configuration cannot tell a reader.
MEASURED_FIELDS = (
    "reply_policy",
    "messages_sent",
    "messages_received",
    "unanswered",
    "unanswered_pct",
)
LATENCY_FIELDS = ("samples", "p50", "p99", "mean", "min", "max")

# Substrings that make a mapping key look like a credential. An app's ``params``
# is a free mapping, so a key could be carried there by an app this module knows
# nothing about; the name is the only signal available, and dropping on it costs
# nothing because a credential is never an experiment fact.
SECRET_HINTS = ("key", "token", "secret", "password", "credential", "passwd")

# A series key that is an address rather than a node name: anything with a dot
# or a colon in it. Rewritten rather than dropped, because which series is which
# still matters; see :func:`_scrub_peer`.
_ADDRESS = re.compile(r"[.:]")

# The detail levels :func:`render_context` renders at, largest first. Each level
# drops the next largest part of the record rather than cutting mid-number:
#   2 - everything,
#   1 - no per-node device roll-up (the largest block, and the one a question
#       about latency or loss never reaches for),
#   0 - the sweep matrix plus each run's status and headline, nothing per node.
DETAIL_FULL = 2
DETAIL_NO_DEVICE = 1
DETAIL_OVERVIEW = 0

# What the text says about itself when it had to be reduced to fit the budget.
# Stated rather than silent: the one thing worse than a record that does not fit
# is one that quietly lost the number an answer leaned on.
DETAIL_NOTES = {
    DETAIL_NO_DEVICE: (
        "(reduced to fit analysis.max_context_bytes: the per-node device "
        "roll-up is not included)"
    ),
    DETAIL_OVERVIEW: (
        "(reduced to fit analysis.max_context_bytes: only the sweep matrix and "
        "each run's status and headline are included)"
    ),
}


class ContextError(Exception):
    """The path given is not a run directory and holds none.

    The one failure this module raises rather than notes: with no run to read
    there is no record to reduce, and a model asked about an empty record can
    only invent one.
    """


def _scrub_secrets(mapping: Any) -> Dict[str, Any]:
    """A mapping with every credential-looking key dropped, recursively."""
    if not isinstance(mapping, dict):
        return {}
    out = {}  # type: Dict[str, Any]
    for key, value in mapping.items():
        name = str(key)
        if any(hint in name.lower() for hint in SECRET_HINTS):
            continue
        out[name] = _scrub_secrets(value) if isinstance(value, dict) else value
    return out


def _scrub_peer(key: str, seen: Dict[str, str]) -> str:
    """Rewrite an address-shaped series key to a stable ``peerN``.

    The bandwidth collector names a series by the peer it measured, which on a
    two-machine run is an address and a port. The series' identity has to
    survive - a node's sent and received bytes are two series - but its peer's
    address does not.
    """
    head, _, tail = key.partition("/")
    if not _ADDRESS.search(head):
        return key
    if head not in seen:
        seen[head] = "peer{}".format(len(seen) + 1)
    return "{}/{}".format(seen[head], tail) if tail else seen[head]


def _scrub_series_keys(block: Any, seen: Dict[str, str]) -> Dict[str, Any]:
    """A roll-up with its address-shaped keys rewritten; see :func:`_scrub_peer`."""
    if not isinstance(block, dict):
        return {}
    out = {}  # type: Dict[str, Any]
    for key, value in block.items():
        name = _scrub_peer(str(key), seen)
        out[name] = _scrub_series_keys(value, seen) if isinstance(value, dict) else value
    return out


def _node_facts(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Each node reduced to :data:`NODE_FIELDS`, with unset fields left out."""
    out = []  # type: List[Dict[str, Any]]
    for node in nodes:
        facts = {f: node.get(f) for f in NODE_FIELDS if node.get(f) is not None}
        if facts:
            out.append(facts)
    return out


def _link_facts(links: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Each link reduced to its endpoints and its network model."""
    out = []  # type: List[Dict[str, Any]]
    for link in links:
        facts = {f: link.get(f) for f in LINK_FIELDS if link.get(f) is not None}
        network = link.get("network") if isinstance(link.get("network"), dict) else {}
        model = {f: network.get(f) for f in NETWORK_FIELDS if network.get(f) is not None}
        distribution = network.get("distribution")
        if isinstance(distribution, dict):
            model["distribution"] = dict(distribution)
        if model:
            facts["network"] = model
        out.append(facts)
    return out


def _link_measurements(record: Any) -> Dict[str, Dict[str, Any]]:
    """Per-node measured latency and delivery, out of the link reports."""
    out = {}  # type: Dict[str, Dict[str, Any]]
    for node, report in sorted(record.link_reports.items()):
        measured = report.get("measured") if isinstance(report, dict) else None
        if not isinstance(measured, dict):
            continue
        latency = measured.get("latency_ms")
        latency = latency if isinstance(latency, dict) else {}
        block = {f: measured.get(f) for f in MEASURED_FIELDS if measured.get(f) is not None}
        block["latency_ms"] = {f: latency.get(f) for f in LATENCY_FIELDS if f in latency}
        out[node] = block
    return out


def _device_facts(run_dir: str) -> Dict[str, Dict[str, Any]]:
    """The per-node device roll-up, or headline numbers off the series.

    ``device_summary.json`` is the orchestrator's own reduction and covers nodes
    whose CSVs were fetched back from another machine, so it is preferred. A run
    written before it existed, or one whose summary is unreadable, falls back to
    each series' own mean, peak and last - the same fallback the device view
    makes, for the same reason.
    """
    summary = device_reader.read_device_summary(run_dir)
    if summary:
        seen = {}  # type: Dict[str, str]
        return {node: _scrub_series_keys(block, seen) for node, block in sorted(summary.items())}
    out = {}  # type: Dict[str, Dict[str, Any]]
    for node, series in sorted(device_reader.read_device_series(run_dir).items()):
        block = {}  # type: Dict[str, Any]
        for item in series:
            stats = item.stats()
            block[item.name] = {
                "unit": item.unit,
                "samples": item.samples,
                "mean": stats["mean"],
                "peak": stats["peak"],
                "last": stats["last"],
            }
        if block:
            out[node] = block
    for node, totals in sorted(device_reader.cumulative_totals(run_dir).items()):
        for total in totals:
            out.setdefault(node, {})[total["label"]] = {
                "unit": total["unit"],
                "total": total["value"],
            }
    return out


def run_context(record: Any, detail: int = DETAIL_FULL) -> Dict[str, Any]:
    """One run reduced: its identity, the config it ran, and its measurements.

    ``detail`` below :data:`DETAIL_FULL` leaves out the device roll-up, which is
    the largest block in the record.
    """
    name, value, unit = record.task_metric()
    stage_fields = ("name", "impl", "node", "queue_policy")
    context = {
        "run_id": record.run_id,
        # Where the run's files are, so a reduction can read one the record
        # does not carry in full - the per-frame latencies, for p95.
        "path": record.path,
        "cell": record.base,
        "rep": record.rep,
        "status": record.status,
        "reason": record.reason,
        "detail": record.detail,
        "phases": dict(record.phases),
        "experiment": record.experiment,
        "app": record.app,
        "placement": placement_of(record),
        "network": network_of(record),
        "swept_ms": swept_value(record),
        "app_params": _scrub_secrets(record.app_params),
        "nodes": _node_facts(record.nodes),
        "links": _link_facts(record.links),
        "routes": len(record.routes),
        "stages": [
            {f: stage.get(f) for f in stage_fields if stage.get(f)} for stage in record.stages
        ],
        "transport": _scrub_secrets(record.transport),
        "collectors": list(record.collectors),
        "task_metric": {"name": name, "value": value, "unit": unit},
        "latency_ms": record.end_to_end_ms(),
        "links_measured": _link_measurements(record),
        "outcomes": copy.deepcopy(record.outcomes),
        "server": copy.deepcopy(record.server),
        "frames": dict(record.frames),
        "sidecars": dict(record.sidecars),
        "device": _device_facts(record.path) if detail >= DETAIL_FULL else {},
        "notes": list(record.notes),
    }
    return context


def sweep_context(table: Any, detail: int = DETAIL_FULL) -> Dict[str, Any]:
    """One sweep reduced: the matrix, each cell's medians, and its runs.

    The matrix is the sweep's own answer - which placement won at which network
    condition - and it is computed here rather than left to the model, because a
    median over a cell's repetitions is arithmetic and not a matter of judgement.
    """
    cells = []  # type: List[Dict[str, Any]]
    for placement in table.placements:
        for network in table.networks:
            cell = table.cell(placement, network)
            if cell is None:
                continue
            name, value, unit = cell.metric()
            cells.append(
                {
                    "placement": placement,
                    "network": network,
                    "axis_ms": cell.axis,
                    "runs": [run.run_id for run in cell.runs],
                    "metric": {"name": name, "median": value, "unit": unit},
                    "latency_ms": cell.latency_ms(),
                    "in_progress": cell.in_progress,
                }
            )
    return {
        "name": table.name,
        "placements": list(table.placements),
        "networks": list(table.networks),
        "cells": cells,
        "in_progress": table.in_progress,
        "progress": table.progress_detail,
        "notes": list(table.notes),
        "runs": [run_context(run, detail) for run in table.runs],
    }


def build_context(path: str, detail: int = DETAIL_FULL) -> Dict[str, Any]:
    """The record for a path: one run, or every sweep in a tree of them.

    A path that *is* a run directory reads as that one run. Anything else is
    read as a tree and grouped by :func:`cadet.dashboard.sweep.group_sweeps`,
    whose rule is that a sweep is a directory of runs - so a tree holding
    several sweeps comes out as several matrices rather than one that averaged
    them together.
    """
    found = discover_runs(path)
    if not found:
        raise ContextError(
            "{}: not a run directory, and no run directory under it (a run "
            "directory holds a run.json or a config.yaml)".format(path)
        )
    records = [read_run(p) for p in found]
    root = os.path.normpath(path)
    if len(records) == 1 and os.path.normpath(records[0].path) == root:
        return {"kind": "run", "root": root, "run": run_context(records[0], detail)}
    return {
        "kind": "sweep",
        "root": root,
        "sweeps": [sweep_context(table, detail) for table in group_sweeps(path, records)],
    }


# --- rendering ------------------------------------------------------------
def _num(value: Any) -> str:
    """A number for a record line, or ``-`` when the run wrote none."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "{:.3f}".format(value).rstrip("0").rstrip(".") or "0"
    return str(value)


def _pairs(mapping: Dict[str, Any]) -> str:
    """``k=v`` for a mapping, in the mapping's own order."""
    return " ".join("{}={}".format(k, _num(v)) for k, v in mapping.items())


def _flat_lines(prefix: str, block: Any, out: List[str]) -> None:
    """Flatten a nested roll-up into ``prefix: a=1 b=2`` lines.

    Generic on purpose: the roll-up's shape is the collectors' business, and
    this module should not need editing when one of them grows a field.
    """
    if not isinstance(block, dict):
        return
    scalars = {k: v for k, v in block.items() if not isinstance(v, dict)}
    if scalars:
        out.append("    {}: {}".format(prefix, _pairs(scalars)))
    for key, value in block.items():
        if isinstance(value, dict):
            _flat_lines("{} {}".format(prefix, key), value, out)


def _p95_from_frame_log(run_path: Optional[str]) -> Optional[float]:
    """Return the 95th percentile of the per-frame latencies, in ms, or None.

    ``None`` for a run that wrote no per-frame record, which is the honest
    answer for a run that did not measure per-frame latency.
    """
    if not run_path or not os.path.isdir(run_path):
        return None
    for name in sorted(os.listdir(run_path)):
        if not name.endswith("_latency_location_log.csv"):
            continue
        try:
            with open(os.path.join(run_path, name), newline="") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, csv.Error):
            return None
        values = []
        for row in rows:
            try:
                values.append(float(row.get("latency_s")) * 1000.0)
            except (TypeError, ValueError):
                continue
        if not values:
            return None
        values.sort()
        # Nearest-rank: a percentile of a measured sample, with nothing assumed
        # about the distribution between two of them.
        index = max(0, min(len(values) - 1, int(math.ceil(0.95 * len(values))) - 1))
        return round(values[index], 3)
    return None


def run_aggregates(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Answer the questions a reader asks first, from one run's own record.

    The record holds the tables a question has to be answered *from*: energy
    per node per collector, latency per client, a message count and an outcome
    each. Working across them is arithmetic, and a reader - a person skimming
    or a model answering - does it with whatever attention is left after
    parsing the tables. Measured across this repository's acceptance
    matrix, an analysis model answered 138 of 140 such questions correctly - so
    the arithmetic is not usually the problem. The two it missed were enough to
    flip those cells' verdicts, and one of them was a cell that answered the
    same question correctly in other runs, on identical data. Stating the
    reductions removes the step where that variance enters.

    So the arithmetic is done here, once, and stated above the tables: totals
    sorted, the top named, percentiles per client, frames sent against frames
    answered, and what each client reported. Nothing here is a new
    measurement; every figure is a reduction of one already in the record.
    """
    energy = []
    for node, collectors in sorted((ctx.get("device") or {}).items()):
        record = (collectors or {}).get("energy")
        if not isinstance(record, dict):
            continue
        if "not_applicable" in record:
            energy.append({"node": node, "energy_j": None, "why": record["not_applicable"]})
            continue
        total = record.get("energy_j_total")
        if total is None:
            total = record.get("energy_j")
        if total is not None:
            energy.append({"node": node, "energy_j": float(total)})
    measured = [e for e in energy if e.get("energy_j") is not None]
    measured.sort(key=lambda e: e["energy_j"], reverse=True)

    latency = []
    for node, record in sorted((ctx.get("links_measured") or {}).items()):
        stats = (record or {}).get("latency_ms") or {}
        if not stats.get("samples"):
            continue
        entry = {
            "client": node,
            "samples": stats.get("samples"),
            "p50_ms": stats.get("p50"),
            "p99_ms": stats.get("p99"),
            "mean_ms": stats.get("mean"),
            "max_ms": stats.get("max"),
        }
        # p95 is not among the figures the link report keeps, so it is computed
        # from the per-frame record where there is one, and left out where
        # there is not. Interpolating it between p50 and p99 would be a number
        # nobody measured.
        p95 = _p95_from_frame_log(ctx.get("path"))
        if p95 is not None:
            entry["p95_ms"] = p95
        latency.append(entry)

    messages = []
    for node, record in sorted((ctx.get("links_measured") or {}).items()):
        if not record:
            continue
        messages.append(
            {
                "client": node,
                "reply_policy": record.get("reply_policy"),
                "sent": record.get("messages_sent"),
                "answered": record.get("messages_received"),
                "unanswered": record.get("unanswered"),
            }
        )

    outcomes = [
        {"client": node, "ok": bool((rec or {}).get("ok")), "detail": (rec or {}).get("detail", "")}
        for node, rec in sorted((ctx.get("outcomes") or {}).items())
    ]

    return {
        "energy_by_node": measured + [e for e in energy if e.get("energy_j") is None],
        "top_energy_node": measured[0]["node"] if measured else None,
        # The run's one latency figure, across clients, as well as each
        # client's own. A run with several clients has no single p50 - the
        # sensing example has one per sensor - and "the median end-to-end
        # latency of this run" is the median over their link reports. Stating
        # only the per-client figures left that question answerable only by
        # combining them, which is the step these reductions exist to remove.
        "median_e2e_ms": ctx.get("latency_ms"),
        "latency_by_client": latency,
        "messages_by_client": messages,
        "outcomes": outcomes,
    }


def _render_aggregates(ctx: Dict[str, Any], out: List[str]) -> None:
    """State the reductions above the tables they come from."""
    agg = run_aggregates(ctx)
    reductions = ("energy_by_node", "latency_by_client", "messages_by_client", "outcomes")
    if not any(agg[key] for key in reductions):
        return
    out.append("  answers (computed from the tables below, not re-measured):")
    if agg["energy_by_node"]:
        top = agg["top_energy_node"]
        out.append(
            "    highest-energy node: {}".format(top if top else "none (no node measured power)")
        )
        for item in agg["energy_by_node"]:
            if item.get("energy_j") is None:
                out.append("      {}: not applicable - {}".format(item["node"], item["why"]))
            else:
                out.append("      {}: {} J".format(item["node"], _num(item["energy_j"])))
    if agg["median_e2e_ms"] is not None:
        out.append(
            "    median end-to-end latency for the run: {} ms{}".format(
                _num(agg["median_e2e_ms"]),
                " (median over the {} clients' link reports)".format(
                    len(agg["latency_by_client"])
                ) if len(agg["latency_by_client"]) > 1 else "",
            )
        )
    for item in agg["latency_by_client"]:
        out.append(
            "    latency {} (n={}): p50 {} ms,{} p99 {} ms, mean {} ms, max {} ms".format(
                item["client"], item["samples"], _num(item["p50_ms"]),
                " p95 {} ms,".format(_num(item["p95_ms"])) if "p95_ms" in item else "",
                _num(item["p99_ms"]), _num(item["mean_ms"]), _num(item["max_ms"]),
            )
        )
    for item in agg["messages_by_client"]:
        out.append(
            "    messages {}: {} sent, {} answered, {} unanswered ({})".format(
                item["client"], _num(item["sent"]), _num(item["answered"]),
                _num(item["unanswered"]), item["reply_policy"] or "reply policy not recorded",
            )
        )
    for item in agg["outcomes"]:
        out.append(
            "    outcome {}: {}{}".format(
                item["client"], "ok" if item["ok"] else "FAILED",
                " - {}".format(item["detail"]) if item["detail"] else "",
            )
        )
    out.append("")


def _render_run(ctx: Dict[str, Any], detail: int, out: List[str]) -> None:
    """One run's section of the record."""
    head = "run {}".format(ctx["run_id"])
    if ctx["rep"] is not None:
        head += " (cell {} rep {})".format(ctx["cell"], ctx["rep"])
    out.append(head)
    reason = " - {}".format(ctx["reason"]) if ctx["reason"] else ""
    out.append("  status: {}{}".format(ctx["status"], reason))
    if ctx["detail"]:
        out.append("  detail: {}".format(ctx["detail"]))
    out.append("  experiment: {}  app: {}".format(ctx["experiment"] or "-", ctx["app"] or "-"))
    out.append(
        "  placement: {}  network: {}  swept axis: {} ms".format(
            ctx["placement"], ctx["network"], _num(ctx["swept_ms"])
        )
    )
    if any(value is not None for value in ctx["phases"].values()):
        out.append("  phases (s): {}".format(_pairs(ctx["phases"])))
    metric = ctx["task_metric"]
    out.append(
        "  headline: {} = {} {}".format(
            metric["name"], _num(metric["value"]), metric["unit"]
        ).rstrip()
    )
    out.append(
        "  measured latency (median over link reports): {} ms".format(_num(ctx["latency_ms"]))
    )
    _render_aggregates(ctx, out)
    if detail <= DETAIL_OVERVIEW:
        if ctx["notes"]:
            out.append("  notes: {}".format("; ".join(ctx["notes"])))
        return

    if ctx["app_params"]:
        out.append("  app params: {}".format(_pairs(ctx["app_params"])))
    if ctx["transport"]:
        out.append("  transport: {}".format(_pairs(ctx["transport"])))
    out.append("  collectors: {}".format(", ".join(ctx["collectors"]) or "(none)"))
    if ctx["nodes"]:
        out.append("  nodes:")
        for node in ctx["nodes"]:
            out.append("    - {}".format(_pairs(node)))
    if ctx["links"]:
        out.append("  links:")
        for link in ctx["links"]:
            network = dict(link.get("network") or {})
            distribution = network.pop("distribution", None)
            line = "    - {}".format(_pairs({f: link[f] for f in LINK_FIELDS if f in link}))
            if network:
                line += "  {}".format(_pairs(network))
            if distribution:
                line += "  distribution[{}]".format(_pairs(distribution))
            out.append(line)
    if ctx["routes"]:
        out.append("  routes: {} multi-hop path(s)".format(ctx["routes"]))
    if ctx["stages"]:
        out.append("  pipeline stages:")
        for stage in ctx["stages"]:
            out.append("    - {}".format(_pairs(stage)))
    if ctx["links_measured"]:
        out.append("  per-node link measurement:")
        for node, block in ctx["links_measured"].items():
            body = {f: block[f] for f in MEASURED_FIELDS if f in block}
            out.append(
                "    {}: {}  latency_ms[{}]".format(
                    node, _pairs(body), _pairs(block.get("latency_ms") or {})
                )
            )
    if ctx["server"]:
        out.append("  server per-request record:")
        for node, block in ctx["server"].items():
            out.append(
                "    {}: requests={} replies={} total_s={}".format(
                    node,
                    _num(block.get("requests")),
                    _num(block.get("replies")),
                    _num(block.get("total_s")),
                )
            )
            for stage in block.get("stages") or []:
                out.append(
                    "      {}: median_s={} p99_s={} max_s={} samples={}".format(
                        stage["name"],
                        _num(stage["median_s"]),
                        _num(stage["p99_s"]),
                        _num(stage["max_s"]),
                        _num(stage["samples"]),
                    )
                )
    if ctx["frames"]:
        out.append("  per-frame record: {}".format(_pairs(ctx["frames"])))
    if ctx["outcomes"]:
        out.append(
            "  client outcomes: {}".format(
                ", ".join(
                    "{}={}".format(node, "ok" if block.get("ok") else "not ok")
                    for node, block in sorted(ctx["outcomes"].items())
                )
            )
        )
    if ctx["sidecars"]:
        out.append("  app summary: {}".format(_pairs(ctx["sidecars"])))
    if ctx["device"]:
        out.append("  device (per node):")
        for node, block in ctx["device"].items():
            _flat_lines(node, block, out)
    if ctx["notes"]:
        out.append("  notes:")
        for note in ctx["notes"]:
            out.append("    - {}".format(note))


def _render_sweep(ctx: Dict[str, Any], detail: int, out: List[str]) -> None:
    """One sweep's section: the matrix first, then each run."""
    out.append(
        "sweep {}: {} placement(s) x {} network condition(s), {} cell(s)".format(
            ctx["name"] or "(unnamed)",
            len(ctx["placements"]),
            len(ctx["networks"]),
            len(ctx["cells"]),
        )
    )
    if ctx["in_progress"]:
        detail_text = " ({})".format(ctx["progress"]) if ctx["progress"] else ""
        out.append("  still in progress{}".format(detail_text))
    out.append("  cells (each number a median over the cell's repetitions):")
    for cell in ctx["cells"]:
        metric = cell["metric"]
        out.append(
            "    placement={!r} network={!r} axis={} ms reps={} "
            "{}={} {} measured latency={} ms".format(
                cell["placement"],
                cell["network"],
                _num(cell["axis_ms"]),
                len(cell["runs"]),
                metric["name"],
                _num(metric["median"]),
                metric["unit"],
                _num(cell["latency_ms"]),
            )
        )
    if ctx["notes"]:
        out.append("  notes:")
        for note in ctx["notes"]:
            out.append("    - {}".format(note))
    for run in ctx["runs"]:
        out.append("")
        _render_run(run, detail, out)


def render_one(context: Dict[str, Any], detail: int = DETAIL_FULL) -> str:
    """Render one already-built record, with no byte budget."""
    out = []  # type: List[str]
    if context["kind"] == "run":
        out.append("CADET run record, read from {}".format(context["root"]))
        out.append("")
        _render_run(context["run"], detail, out)
    else:
        out.append(
            "CADET sweep record, read from {} ({} sweep(s))".format(
                context["root"], len(context["sweeps"])
            )
        )
        for sweep in context["sweeps"]:
            out.append("")
            _render_sweep(sweep, detail, out)
    out.append("")
    return "\n".join(out)


def render_context(path: str, max_bytes: int) -> str:
    """The record for ``path`` as text, at the largest detail level that fits.

    The budget is met by dropping whole parts of the record, largest first (see
    the ``DETAIL_*`` constants), and the level reached is stated in the text
    itself. A record that will not fit even at the smallest level is cut, and
    its last line says so.
    """
    text = ""
    for detail in (DETAIL_FULL, DETAIL_NO_DEVICE, DETAIL_OVERVIEW):
        text = render_one(build_context(path, detail), detail)
        if detail in DETAIL_NOTES:
            text = "{}\n{}".format(DETAIL_NOTES[detail], text)
        if len(text.encode("utf-8")) <= max_bytes:
            return text
    # The note is part of the budget, so a record that says it was cut is still
    # within the bound the caller asked for.
    note = "\n(record cut at {} bytes: analysis.max_context_bytes)\n".format(max_bytes)
    room = max(0, max_bytes - len(note.encode("utf-8")))
    return text.encode("utf-8")[:room].decode("utf-8", "ignore") + note
