"""Render the dashboard's four views as self-contained HTML pages.

What this is for:
    The page is shown live on a laptop that may have no network, so every page
    this module returns is complete on its own: the stylesheet is inline, the
    plots are the SVG :mod:`cadet.dashboard.svg` emitted, the typeface is
    whatever the machine already has, and no script runs. Nothing is fetched at
    page load, from a CDN or anywhere else.

What it contains:
    - :data:`STYLE`: the whole stylesheet, including the dark theme and the
      categorical slots the plots address.
    - :func:`page`: the shell every view shares.
    - :func:`sweep_view`, :func:`run_view`, :func:`device_view`,
      :func:`live_view`: the four views.
    - :func:`not_found`: the page a bad link gets, which is still a page.

How it fits into CADET:
    Called by :mod:`cadet.dashboard.server` with the records
    :mod:`cadet.dashboard.runs`, :mod:`cadet.dashboard.sweep` and
    :mod:`cadet.dashboard.device` read. Writes nothing and runs nothing: every
    number on a page was already on disk.

    A view whose data is missing renders a message in place of its panel. That
    is the behaviour the whole module is arranged around, so each view asks what
    it has rather than assuming, and ``record.notes`` is printed rather than
    swallowed.

How to configure it:
    Nothing. :func:`page` takes the refresh interval the live view uses, which
    the server decides from whether anything is still running.
"""

import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from cadet.dashboard import device as device_mod
from cadet.dashboard import svg
from cadet.dashboard.runs import RunRecord
from cadet.dashboard.sweep import SweepTable

escape = svg.escape
fmt = svg.fmt

# The whole stylesheet. Inline because a fetched stylesheet is a stylesheet that
# is missing at the meeting, and because one file holding both themes is how the
# plots and the page stay in step: the plots address --series-N and the ink
# tokens, so redefining them under the dark theme restyles the same markup.
#
# The typeface is a stack of families the machine already has. No web font is
# named, since naming one is either a fetch or a silent fallback.
STYLE = """
:root {
  color-scheme: light;
  --surface: #fcfcfb;
  --surface-raised: #ffffff;
  --border: #dedcd5;
  --ink: #0b0b0b;
  --ink-secondary: #52514e;
  --ink-muted: #76746d;
  --ok: #008300;
  --bad: #e34948;
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --series-3: #1baf7a;
  --series-4: #eda100;
  --series-5: #e87ba4;
  --series-6: #008300;
  --series-7: #6250d6;
  --series-8: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19;
    --surface-raised: #232322;
    --border: #3a3a37;
    --ink: #f0efec;
    --ink-secondary: #c3c2b7;
    --ink-muted: #96948a;
    --series-1: #3987e5;
    --series-2: #d95926;
    --series-3: #199e70;
    --series-4: #c98500;
    --series-5: #d55181;
    --series-6: #008300;
    --series-7: #9085e9;
    --series-8: #e66767;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--surface);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Cantarell, Ubuntu,
    "Helvetica Neue", Arial, sans-serif;
  font-size: 15px;
  line-height: 1.5;
}
header {
  border-bottom: 1px solid var(--border);
  padding: 14px 16px;
  display: flex;
  flex-wrap: wrap;
  gap: 10px 20px;
  align-items: baseline;
}
header h1 { font-size: 17px; margin: 0; font-weight: 600; }
nav a { color: var(--series-1); text-decoration: none; margin-right: 14px; }
nav a:hover, a:focus { text-decoration: underline; }
main { padding: 16px; max-width: 1180px; }
section { margin-bottom: 30px; }
h2 { font-size: 15px; margin: 0 0 10px; font-weight: 600; }
h3 { font-size: 14px; margin: 18px 0 6px; font-weight: 600; color: var(--ink-secondary); }
p, li { color: var(--ink-secondary); }
table { border-collapse: collapse; width: 100%; margin-bottom: 8px; }
th, td {
  border: 1px solid var(--border);
  padding: 6px 9px;
  text-align: left;
  vertical-align: top;
  font-size: 13px;
}
th { background: var(--surface-raised); font-weight: 600; color: var(--ink); }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.hole { color: var(--ink-muted); font-style: italic; }
.tiles { display: flex; flex-wrap: wrap; gap: 10px; }
.tile {
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-raised);
  padding: 9px 13px;
  min-width: 128px;
}
.tile .k { font-size: 12px; color: var(--ink-muted); }
.tile .v { font-size: 19px; font-variant-numeric: tabular-nums; }
.panel {
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-raised);
  padding: 8px;
  margin-bottom: 14px;
}
.notes { border-left: 3px solid var(--series-4); padding: 2px 0 2px 11px; }
.notes li { font-size: 13px; }
.pill {
  display: inline-block;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0 8px;
  font-size: 12px;
  color: var(--ink-secondary);
}
.pill.live { border-color: var(--series-2); color: var(--series-2); }
.ok { color: var(--ok); }
.bad { color: var(--bad); }
.viz { display: block; max-width: 100%; height: auto; }
.viz-grid { stroke: var(--border); stroke-width: 1; }
.viz-axis { stroke: var(--ink-muted); stroke-width: 1; }
.viz-tick { fill: var(--ink-secondary); font-size: 11px; }
.viz-axis-label { fill: var(--ink-muted); font-size: 12px; }
.viz-legend { fill: var(--ink-secondary); font-size: 12px; }
.viz-direct { fill: var(--ink-secondary); font-size: 11px; font-variant-numeric: tabular-nums; }
.viz-empty { fill: var(--ink-muted); font-size: 13px; font-style: italic; }
.viz-marker { stroke: var(--surface-raised); }
.viz-line { stroke-linejoin: round; stroke-linecap: round; }
.viz-swatch { stroke: none; }
@media (forced-colors: active) {
  .viz-line, .viz-marker, .viz-bar { forced-color-adjust: none; }
}
"""


def page(title: str, body: str, refresh: Optional[int] = None, nav_root: str = "/") -> str:
    """The shell every view shares: one document, nothing fetched.

    ``refresh`` is the live view's whole mechanism. A ``meta refresh`` makes the
    *page* re-request on its own interval, so the server answers one short
    request at a time and holds no connection open - which is what the brief
    asks for and also what survives a laptop whose network comes and goes.
    """
    meta = (
        '<meta http-equiv="refresh" content="{}">'.format(int(refresh))
        if refresh and refresh > 0
        else ""
    )
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "{meta}<title>{title}</title><style>{style}</style></head><body>"
        "<header><h1>CADET dashboard</h1><nav>"
        '<a href="{root}">sweep</a><a href="{root}?view=runs">runs</a>'
        '<a href="{root}?view=live">live</a>'
        "</nav></header><main>{body}</main></body></html>".format(
            meta=meta, title=escape(title), style=STYLE, root=escape(nav_root), body=body
        )
    )


def _notes(notes: List[str]) -> str:
    """Print what was missing. The degradation is shown, never swallowed."""
    if not notes:
        return ""
    items = "".join("<li>{}</li>".format(escape(note)) for note in notes)
    return '<section><h2>what is not in this record</h2><ul class="notes">{}</ul></section>'.format(
        items
    )


def _tile(key: str, value: Any, unit: str = "") -> str:
    """One headline number with its name under it."""
    text = value if isinstance(value, str) else fmt(value)
    suffix = " {}".format(escape(unit)) if unit and text != "-" else ""
    return '<div class="tile"><div class="k">{k}</div><div class="v">{v}{u}</div></div>'.format(
        k=escape(key), v=escape(text), u=suffix
    )


def _status_pill(record: RunRecord) -> str:
    """The run's status as a pill, marked live while it has not finished."""
    if record.in_progress:
        return '<span class="pill live">in progress</span>'
    css = "ok" if record.status == "complete" else "bad"
    return '<span class="pill {c}">{s}</span>'.format(c=css, s=escape(record.status))


def _run_link(root: str, record: RunRecord, view: str = "run") -> str:
    """A link to a run's own view, keyed by its path relative to the root.

    Relative rather than absolute so the link says nothing about where the tree
    sits on this machine, and so the server has a key it can check back against
    the runs it discovered rather than opening whatever a query string names.
    """
    relative = os.path.relpath(record.path, root)
    return "/{}?run={}".format(view, quote(relative))


# ---------------------------------------------------------------------------
# 1. Sweep view


def sweep_view(tables, root: str) -> str:
    """One placement x network matrix per sweep in the tree, with its plot.

    A list rather than one table, because a directory can hold several sweeps and
    they are different experiments rather than more cells of one. Each gets its
    own matrix, its own crossover and its own run list, under its experiment
    name, so two sweeps are read side by side rather than averaged together.

    Each table is also its plot's table view: every number the crossover draws is
    readable here, which is what keeps the plot from being the only way to get at
    the data.
    """
    if isinstance(tables, SweepTable):
        tables = [tables]
    present = [t for t in tables if t.cells]
    if not present:
        return page(
            "CADET dashboard",
            "<section><h2>no runs</h2><p>No run directory under "
            "<code>{}</code>. A run directory is one holding a "
            "<code>run.json</code> or a <code>config.yaml</code>.</p></section>".format(
                escape(root)
            ),
        )
    body = "".join(_sweep_section(table, root, len(present) > 1) for table in present)
    if len(present) > 1:
        body = (
            "<section><h2>sweeps</h2><p>{n} sweeps under <code>{root}</code>, "
            "one matrix each. They are separate experiments: a cell holds only "
            "the repetitions of its own sweep.</p></section>".format(
                n=len(present), root=escape(root)
            )
        ) + body
    return page("CADET dashboard: sweep", body, refresh=None)


def _sweep_section(table: SweepTable, root: str, titled: bool) -> str:
    """One sweep's matrix, crossover, notes and run list."""
    if not table.cells:
        return page(
            "CADET dashboard",
            "<section><h2>no runs</h2><p>No run directory under "
            "<code>{}</code>. A run directory is one holding a "
            "<code>run.json</code> or a <code>config.yaml</code>.</p></section>".format(
                escape(root)
            ),
        )

    name, _value, unit = next(
        (cell.metric() for cell in table.cells.values()), ("task metric", None, "")
    )
    head = "".join("<th>{}</th>".format(escape(n)) for n in table.networks)
    rows = []
    for placement in table.placements:
        cells = []
        for network in table.networks:
            cell = table.cell(placement, network)
            if cell is None:
                cells.append('<td class="hole">not run</td>')
                continue
            _n, metric, metric_unit = cell.metric()
            latency = cell.latency_ms()
            live = " &middot; live" if cell.in_progress else ""
            cells.append(
                '<td class="num">{m}{mu}<br><span class="pill">{lat}</span>'
                "<br><small>{reps} rep(s){live}</small></td>".format(
                    m=escape(fmt(metric)),
                    mu=" " + escape(metric_unit) if metric is not None and metric_unit else "",
                    # The unit rides with the number: a bare "- ms" reads as a
                    # measurement of nothing rather than as no measurement.
                    lat=("{} ms".format(fmt(latency)) if latency is not None else "no latency"),
                    reps=len(cell.runs),
                    live=live,
                )
            )
        rows.append(
            "<tr><th>{p}</th>{cells}</tr>".format(p=escape(placement), cells="".join(cells))
        )

    axis_known = any(cell.axis is not None for cell in table.cells.values())
    plot = svg.crossover_plot(
        table.series(),
        "swept link delay (ms)" if axis_known else "network condition (index)",
        "{}{}".format(name, " ({})".format(unit) if unit else ""),
        "{} by placement".format(name),
    )

    title = escape(table.name) if (titled and table.name) else "sweep"
    return (
        "<section><h2>{title}</h2>"
        "<p>{cells} cell(s) over {runs} run(s) under <code>{root}</code>. "
        "Each cell shows the task metric, the measured latency and how many "
        "repetitions it holds.</p>"
        "<table><thead><tr><th>placement \\ network</th>{head}</tr></thead>"
        "<tbody>{rows}</tbody></table></section>"
        '<section><h3>crossover{named}</h3><div class="panel">{plot}</div></section>'
        "{notes}"
        "{runs_table}".format(
            title=title,
            named=" &mdash; {}".format(escape(table.name)) if (titled and table.name) else "",
            cells=len(table.cells),
            runs=len(table.runs),
            root=escape(root),
            head=head,
            rows="".join(rows),
            plot=plot,
            notes=_notes(table.notes),
            runs_table=_runs_table(table, root),
        )
    )


def _merged(tables, root: str) -> SweepTable:
    """One table over every sweep's runs, for the views that are tree-wide.

    The live view and the plain run list ask about the tree rather than about
    one experiment, and both only read ``runs``; the matrix is never built from
    this, which is the whole reason the sweeps are separated elsewhere.
    """
    if isinstance(tables, SweepTable):
        return tables
    cells = {}
    for index, table in enumerate(tables):
        for key, cell in table.cells.items():
            cells[(index,) + key] = cell
    notes = [note for table in tables for note in table.notes]
    return SweepTable(root, [], [], cells, notes)


def runs_table(tables, root: str) -> str:
    """Every run in the tree, across its sweeps, with a link into each one's views.

    Public because the server routes a view to it; the private helper below takes
    one table and is what the per-sweep sections use.
    """
    return _runs_table(_merged(tables, root), root)


def _runs_table(table: SweepTable, root: str) -> str:
    """Every run in the sweep, with a link into its own two views."""
    rows = []
    for record in table.runs:
        name, value, unit = record.task_metric()
        rows.append(
            "<tr><td><a href=\"{href}\">{rid}</a></td><td>{status}</td>"
            '<td>{placement}</td><td>{network}</td><td class="num">{metric}</td>'
            '<td class="num">{lat}</td><td><a href="{dhref}">device</a></td></tr>'.format(
                href=escape(_run_link(root, record)),
                rid=escape(record.run_id),
                status=_status_pill(record),
                placement=escape(_placement_label(record)),
                network=escape(_network_label(record)),
                metric="{} {}".format(escape(fmt(value)), escape(unit)) if value else "-",
                lat=escape(fmt(record.end_to_end_ms())),
                dhref=escape(_run_link(root, record, view="device")),
            )
        )
    return (
        "<section><h2>runs</h2><table><thead><tr><th>run</th><th>status</th>"
        "<th>placement</th><th>network</th><th>metric</th><th>latency (ms)</th>"
        "<th></th></tr></thead><tbody>{}</tbody></table></section>".format("".join(rows))
    )


def _placement_label(record: RunRecord) -> str:
    """The run's placement coordinate, through the grouping rule's own function."""
    from cadet.dashboard.sweep import placement_of

    return placement_of(record)


def _network_label(record: RunRecord) -> str:
    """The run's network coordinate, through the grouping rule's own function."""
    from cadet.dashboard.sweep import network_of

    return network_of(record)


# ---------------------------------------------------------------------------
# 2. Run view


def run_view(record: RunRecord, root: str) -> str:
    """Topology, per-stage latency, the task metric and the link report."""
    name, value, unit = record.task_metric()
    tiles = [
        _tile("status", record.status),
        _tile(name, value, unit),
        _tile("latency", record.end_to_end_ms(), "ms"),
        _tile("frames", record.frames.get("rows")),
    ]
    for key in ("startup_s", "active_s", "teardown_s"):
        tiles.append(_tile(key.replace("_s", "").replace("_", " "), record.phases.get(key), "s"))

    body = (
        '<section><h2>run {rid} {pill}</h2><p>{exp} / {app} &middot; '
        "<code>{path}</code></p>"
        '<div class="tiles">{tiles}</div></section>'
        "{topology}{stages}{task}{link}{nav}{notes}".format(
            rid=escape(record.run_id),
            pill=_status_pill(record),
            exp=escape(record.experiment or "?"),
            app=escape(record.app or "?"),
            path=escape(os.path.relpath(record.path, root)),
            tiles="".join(tiles),
            topology=_topology_section(record),
            stages=_stage_section(record),
            task=_task_section(record),
            link=_link_section(record),
            nav='<section><p><a href="{}">device view for this run</a></p></section>'.format(
                escape(_run_link(root, record, view="device"))
            ),
            notes=_notes(record.notes),
        )
    )
    return page(
        "CADET run {}".format(record.run_id),
        body,
        refresh=5 if record.in_progress else None,
    )


def _topology_section(record: RunRecord) -> str:
    """Nodes, links and routes as the config declared them, plus the backend.

    The backend is the one topology fact that is not a node or an edge and that
    changes what every other number means: a packet-level link and an in-process
    delay model shape the same configured delay in different places.
    """
    if not record.nodes:
        return "<section><h2>topology</h2><p>no topology in this run</p></section>"
    node_rows = "".join(
        "<tr><td>{n}</td><td>{role}</td><td>{device}</td><td>{host}</td>"
        '<td class="num">{port}</td><td>{launcher}</td></tr>'.format(
            n=escape(node.get("name")),
            role=escape(node.get("role") or "-"),
            device=escape(node.get("device") or "-"),
            host=escape(node.get("host") or "-"),
            port=escape(node.get("port") if node.get("port") is not None else "-"),
            launcher=escape(node.get("launcher") or "-"),
        )
        for node in record.nodes
    )
    edges = []
    for link in record.links:
        network = link.get("network") if isinstance(link.get("network"), dict) else {}
        edges.append(
            "<tr><td>link</td><td>{n}</td><td>{frm} &rarr; {to}</td><td>{direction}</td>"
            "<td>{mode}</td><td>{shape}</td></tr>".format(
                n=escape(link.get("name")),
                frm=escape(link.get("from")),
                to=escape(link.get("to")),
                direction=escape(link.get("direction") or "forward"),
                mode=escape(network.get("mode") or "-"),
                shape=escape(_network_label(record) if len(record.links) == 1 else "see sweep"),
            )
        )
    for route in record.routes:
        edges.append(
            "<tr><td>route</td><td>{n}</td><td>{frm} &rarr; {to}</td><td>via {via}</td>"
            "<td>{realise}</td><td>hops: {hops}</td></tr>".format(
                n=escape(route.get("name")),
                frm=escape(route.get("from")),
                to=escape(route.get("to")),
                via=escape(", ".join(str(v) for v in (route.get("via") or [])) or "-"),
                realise=escape(route.get("realise") or "-"),
                hops=escape(", ".join(str(h) for h in (route.get("hops") or [])) or "-"),
            )
        )
    edge_table = (
        "<h3>links and routes</h3><table><thead><tr><th>kind</th><th>name</th>"
        "<th>between</th><th>direction</th><th>mode</th><th>shaping</th></tr></thead>"
        "<tbody>{}</tbody></table>".format("".join(edges))
        if edges
        else "<h3>links and routes</h3><p>this run declared none: the path is unshaped</p>"
    )
    backend = record.transport or {}
    return (
        "<section><h2>topology</h2>"
        "<p>backend: <code>{proto}</code> / <code>{mode}</code>"
        " &middot; encoding <code>{enc}</code></p>"
        "<h3>nodes</h3><table><thead><tr><th>node</th><th>role</th><th>device</th>"
        "<th>host</th><th>port</th><th>launcher</th></tr></thead><tbody>{nodes}</tbody>"
        "</table>{edges}</section>".format(
            proto=escape(backend.get("protocol") or "-"),
            mode=escape(backend.get("mode") or "-"),
            enc=escape(backend.get("encoding") or "-"),
            nodes=node_rows,
            edges=edge_table,
        )
    )


def _stage_section(record: RunRecord) -> str:
    """Per-stage latency: the server's own breakdown of a request, per node."""
    if not record.server:
        return (
            "<section><h2>per-stage latency</h2>"
            "<p>no server per-request record in this run</p></section>"
        )
    panels = []
    for node in sorted(record.server):
        entry = record.server[node]
        stages = entry.get("stages") or []
        if not stages:
            panels.append(
                '<div class="panel">{}</div>'.format(
                    svg.empty_plot(
                        "node {}".format(node),
                        "the record for this node holds no timed request",
                    )
                )
            )
            continue
        values = [(stage["name"], stage["median_s"]) for stage in stages]
        values.append(("total (median)", entry.get("total_s")))
        panels.append(
            '<div class="panel">{}</div>'.format(
                svg.bar_panel(
                    values,
                    "node {}: per-stage latency, median of {} request(s)".format(
                        node, entry.get("requests")
                    ),
                    "seconds",
                )
            )
        )
        rows = "".join(
            '<tr><td>{n}</td><td class="num">{med}</td><td class="num">{p99}</td>'
            '<td class="num">{mx}</td><td class="num">{s}</td></tr>'.format(
                n=escape(stage["name"]),
                med=escape(fmt(stage["median_s"])),
                p99=escape(fmt(stage["p99_s"])),
                mx=escape(fmt(stage["max_s"])),
                s=stage["samples"],
            )
            for stage in stages
        )
        panels.append(
            "<table><thead><tr><th>stage ({})</th><th>median s</th><th>p99 s</th>"
            "<th>max s</th><th>samples</th></tr></thead><tbody>{}</tbody>"
            "</table>".format(escape(node), rows)
        )
    return "<section><h2>per-stage latency</h2>{}</section>".format("".join(panels))


def _task_section(record: RunRecord) -> str:
    """The task metric, the client's verdict and the app's own summary numbers."""
    name, value, unit = record.task_metric()
    outcome_rows = "".join(
        '<tr><td>{n}</td><td class="{c}">{ok}</td><td>{detail}</td></tr>'.format(
            n=escape(node),
            c="ok" if data["ok"] else "bad",
            ok="ok" if data["ok"] else "not ok",
            detail=escape(data["detail"]),
        )
        for node, data in sorted(record.outcomes.items())
    )
    outcome = (
        "<h3>client outcome</h3><table><thead><tr><th>node</th><th>verdict</th>"
        "<th>detail</th></tr></thead><tbody>{}</tbody></table>".format(outcome_rows)
        if outcome_rows
        else "<h3>client outcome</h3><p>no client outcome in this run</p>"
    )
    sidecar = ""
    if record.sidecars:
        rows = "".join(
            '<tr><td>{k}</td><td class="num">{v}</td></tr>'.format(
                k=escape(key), v=escape(fmt(val))
            )
            for key, val in sorted(record.sidecars.items())
        )
        sidecar = (
            "<h3>application summary</h3><table><thead><tr><th>metric</th>"
            "<th>value</th></tr></thead><tbody>{}</tbody></table>".format(rows)
        )
    tiles = [
        _tile(name, value, unit),
        _tile("triggered", "yes" if record.frames.get("triggered") else "no"),
        _tile("send gap", record.frames.get("send_gap_ms"), "ms"),
        _tile("mean speed", record.frames.get("speed_mean")),
    ]
    return (
        '<section><h2>task metric</h2><div class="tiles">{tiles}</div>'
        "{outcome}{sidecar}</section>".format(
            tiles="".join(tiles), outcome=outcome, sidecar=sidecar
        )
    )


def _link_section(record: RunRecord) -> str:
    """The network link report: configured intent beside what was measured."""
    if not record.link_reports:
        return "<section><h2>link report</h2><p>no link report in this run</p></section>"
    blocks = []
    for node in sorted(record.link_reports):
        report = record.link_reports[node]
        measured = report.get("measured") or {}
        latency = measured.get("latency_ms") or {}
        tiles = [
            _tile("sent", measured.get("messages_sent")),
            _tile("received", measured.get("messages_received")),
            _tile("unanswered", measured.get("unanswered")),
            _tile("samples", latency.get("samples")),
        ]
        # min/p50/mean/p99/max side by side rather than one number: a measured
        # p99 far above the p50 is the shaped tail, and a single figure hides it.
        bars = [(key, latency.get(key)) for key in ("min", "p50", "mean", "p99", "max")]
        blocks.append(
            "<h3>node {node} &middot; reply policy <code>{policy}</code></h3>"
            '<div class="tiles">{tiles}</div><div class="panel">{plot}</div>'
            "{configured}".format(
                node=escape(node),
                policy=escape(measured.get("reply_policy") or "-"),
                tiles="".join(tiles),
                plot=svg.bar_panel(
                    [(k, _as_float(v)) for k, v in bars],
                    "measured one-way latency as reported by node {}".format(node),
                    "milliseconds",
                ),
                configured=_configured_paths(report),
            )
        )
    return "<section><h2>link report</h2>{}</section>".format("".join(blocks))


def _as_float(value: Any) -> Optional[float]:
    """A report field as a float, or None - a null latency is a real answer."""
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _configured_paths(report: Dict[str, Any]) -> str:
    """What the config asked the path to be, beside what was measured.

    Printed even when it is empty, because "configured nothing" is the fact that
    explains an unshaped measurement.
    """
    links = report.get("configured_links") or []
    routes = report.get("configured_routes") or []
    if not links and not routes:
        return "<p>this node had no configured link or route: the path is unshaped</p>"
    rows = []
    for item in links + routes:
        if not isinstance(item, dict):
            continue
        rows.append(
            "<tr><td>{n}</td><td>{detail}</td></tr>".format(
                n=escape(item.get("name") or "-"),
                detail=escape(
                    ", ".join(
                        "{}={}".format(k, v)
                        for k, v in sorted(item.items())
                        if k != "name" and not isinstance(v, (dict, list))
                    )
                    or "-"
                ),
            )
        )
    return (
        "<table><thead><tr><th>configured path</th><th>intent</th></tr></thead>"
        "<tbody>{}</tbody></table>".format("".join(rows))
    )


# ---------------------------------------------------------------------------
# 3. Device view


def device_view(record: RunRecord, root: str) -> str:
    """GPU, CPU and energy over time, per node, with the run's totals.

    An empty device view says so. The collectors are off unless a config names
    them, so a run with no series is the ordinary case rather than a fault, and
    the view distinguishes "this run asked for no collector" from "it asked and
    the file is not there".
    """
    series_by_node = device_mod.read_device_series(record.path)
    summary = device_mod.read_device_summary(record.path)
    totals = device_mod.cumulative_totals(record.path)

    if not series_by_node:
        if record.collectors:
            message = (
                "no device samples in this run. Its config asked for the "
                "collector(s) {}, so either the role died before its first "
                "sample or the per-node file was not written.".format(
                    ", ".join(record.collectors)
                )
            )
        else:
            message = (
                "no device samples in this run. Its config enabled no metric "
                "collector, which is the default, so none were written."
            )
        sections = "<section><h2>device</h2><p>{}</p></section>".format(escape(message))
        return page(
            "CADET device {}".format(record.run_id),
            '<section><h2>run {rid} {pill}</h2><p><a href="{href}">back to the run '
            "view</a></p></section>{s}{notes}".format(
                rid=escape(record.run_id),
                pill=_status_pill(record),
                href=escape(_run_link(root, record)),
                s=sections,
                notes=_notes(record.notes),
            ),
            refresh=5 if record.in_progress else None,
        )

    blocks = []
    for node in sorted(series_by_node):
        tiles = []
        for item in totals.get(node, []):
            tiles.append(_tile(item["label"], item["value"], item["unit"]))
        node_summary = summary.get(node) if isinstance(summary.get(node), dict) else None
        if node_summary:
            tiles.append(_tile("summary samples", node_summary.get("samples")))
            tiles.append(_tile("summary duration", node_summary.get("duration_s"), "s"))
        for series in series_by_node[node]:
            stats = series.stats()
            tiles.append(_tile("{} peak".format(series.name), stats["peak"], series.unit))

        panels = []
        # One panel per quantity, one y axis each: GPU percent and GPU megabytes
        # are different scales, and a second y axis is the mistake this avoids.
        for series in series_by_node[node]:
            panels.append(
                '<div class="panel">{}</div>'.format(
                    svg.series_panel(
                        [{"name": series.name, "t": series.t, "y": series.y}],
                        "{} on node {} ({} samples)".format(series.name, node, series.samples),
                        "{} ({})".format(series.label, series.unit)
                        if series.unit
                        else series.label,
                    )
                )
            )
        blocks.append(
            '<section><h2>node {node}</h2><div class="tiles">{tiles}</div>{panels}'
            "</section>".format(node=escape(node), tiles="".join(tiles), panels="".join(panels))
        )

    caveat = (
        "<section><p>Each series starts at its own first sample, so the time "
        "axis is seconds into that series rather than a shared clock. Two nodes "
        "on different machines have a clock offset that the <code>clock_offset"
        "</code> collector measures, so the panels here are not aligned across "
        "nodes.</p></section>"
    )
    return page(
        "CADET device {}".format(record.run_id),
        '<section><h2>device: run {rid} {pill}</h2><p><a href="{href}">back to the '
        "run view</a></p></section>{blocks}{caveat}{notes}".format(
            rid=escape(record.run_id),
            pill=_status_pill(record),
            href=escape(_run_link(root, record)),
            blocks="".join(blocks),
            caveat=caveat,
            notes=_notes(record.notes),
        ),
        refresh=5 if record.in_progress else None,
    )


# ---------------------------------------------------------------------------
# 4. Live view


def live_view(tables, root: str, interval: int) -> str:
    """What is running now, re-requested on an interval while anything is.

    The refresh is attached only while something is in progress, so a finished
    tree stops re-requesting and a laptop left on the page overnight is not
    serving itself a page every few seconds.

    Across every sweep in the tree, because "is anything running" is a question
    about the tree and not about one experiment in it.
    """
    if isinstance(tables, SweepTable):
        tables = [tables]
    table = _merged(tables, root)
    running = [record for record in table.runs if record.in_progress]
    finished = len(table.runs) - len(running)
    # A sweep between two cells has no run in progress and is not finished, so
    # the sweeps' own markers are what decide whether this page keeps asking.
    sweeping = [t for t in tables if t.progress.get("state") == "running"]
    if running:
        rows = "".join(
            '<tr><td><a href="{href}">{rid}</a></td><td>{status}</td>'
            "<td>{exp}</td><td>{detail}</td></tr>".format(
                href=escape(_run_link(root, record)),
                rid=escape(record.run_id),
                status=_status_pill(record),
                exp=escape(record.experiment or "?"),
                detail=escape(record.notes[0] if record.notes else record.detail),
            )
            for record in running
        )
        state = (
            "<p>{n} run(s) in progress, {f} finished. This page re-requests "
            "itself every {i} s while anything is running.</p>"
            "<table><thead><tr><th>run</th><th>status</th><th>experiment</th>"
            "<th>note</th></tr></thead><tbody>{rows}</tbody></table>".format(
                n=len(running), f=finished, i=interval, rows=rows
            )
        )
    elif sweeping:
        # The case the sweep marker exists for: no run is mid-flight, and the
        # sweep is between cells with more to come. Without the marker this page
        # said "nothing is in progress" and stopped refreshing, in the middle of
        # a sweep.
        state = (
            "<p>No run is in flight, and {n} sweep(s) are still going - between "
            "cells, with more to come. This page re-requests itself every {i} s "
            "while anything is running.</p><ul>{items}</ul>".format(
                n=len(sweeping),
                i=interval,
                items="".join(
                    "<li><code>{name}</code>: {detail}</li>".format(
                        name=escape(t.name or "sweep"),
                        detail=escape(t.progress_detail or "in progress"),
                    )
                    for t in sweeping
                ),
            )
        )
    else:
        state = (
            "<p>Nothing is in progress: all {n} run(s) under <code>{root}</code> "
            "have reached a terminal status. This page is not "
            "re-requesting.</p>".format(n=len(table.runs), root=escape(root))
        )
    return page(
        "CADET dashboard: live",
        "<section><h2>live</h2>{}</section>{}".format(state, _runs_table(table, root)),
        # The same predicate as the text above: a sweep between cells keeps the
        # page asking, or it would stop refreshing mid-sweep.
        refresh=interval if (running or sweeping) else None,
    )


def not_found(message: str, root: str) -> str:
    """A bad or stale link still gets a page, with a way back to the sweep."""
    return page(
        "CADET dashboard",
        '<section><h2>not found</h2><p>{m}</p><p><a href="/">back to the sweep '
        "view</a></p></section>".format(m=escape(message)),
    )
