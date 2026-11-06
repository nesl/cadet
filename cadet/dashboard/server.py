"""Serve the dashboard from the standard library's HTTP server.

What this is for:
    ``cadet dashboard <runs-dir>`` binds a local HTTP server and answers the
    four views. The standard library is the whole dependency: no framework, no
    template engine, nothing fetched at page load, so a fresh clone on a laptop
    with no network serves the page.

What it contains:
    - :class:`DashboardState`: the runs-dir, re-read per request.
    - :class:`DashboardHandler`: the request handler and its routes.
    - :func:`serve`: bind and serve until interrupted.

How it fits into CADET:
    Called by ``cadet dashboard`` in :mod:`cadet.cli`. Reads through
    :mod:`cadet.dashboard.runs`, groups through :mod:`cadet.dashboard.sweep`,
    renders through :mod:`cadet.dashboard.render`. It never starts, changes or
    tears down a run; the only filesystem access is reading a run directory.

    The tree is re-read on every request rather than cached, which is what makes
    the live view work without holding a connection open: the page re-requests
    on its own interval and each request sees the tree as it is now. A sweep of a
    few hundred runs re-reads in well under the refresh interval, and the cost of
    being wrong about staleness during a demonstration is higher than the cost of
    the read.

    Binding: the default host is loopback, because this serves a local
    measurement tree and binding every interface by default would publish it.

How to configure it:
    ``cadet dashboard <runs-dir> [--host H] [--port P] [--refresh S]``.
"""

import os
import posixpath
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

from cadet.dashboard import render
from cadet.dashboard.runs import RunRecord, discover_runs, read_run
from cadet.dashboard.sweep import SweepTable, group_sweeps

# Loopback, not every interface: a run tree is measurement data on someone's
# machine, and the default must not publish it to the network the laptop is on.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787

# How often the live view re-requests itself while anything is in progress.
DEFAULT_REFRESH_S = 5


class DashboardState(object):
    """The runs directory the server reads, and the reads over it.

    Holds no parsed data between requests on purpose; see the module docstring
    for why the tree is re-read instead of cached.
    """

    def __init__(self, root: str, refresh: int = DEFAULT_REFRESH_S) -> None:
        self.root = os.path.abspath(root)
        self.refresh = refresh

    def records(self) -> List[RunRecord]:
        """Every run under the root, read as it is now."""
        return [read_run(path) for path in discover_runs(self.root)]

    def tables(self) -> List[SweepTable]:
        """The tree's sweeps, one table each, through the one grouping function.

        A list because a directory can hold several sweeps and they are separate
        experiments; see :func:`cadet.dashboard.sweep.group_sweeps`.
        """
        return group_sweeps(self.root, self.records())

    def resolve(self, relative: str) -> Optional[RunRecord]:
        """The run at a path relative to the root, or None.

        Checked against the runs actually discovered rather than opened by name:
        a query string is input, and joining it onto a root would let a link read
        a directory outside the tree. Matching it to a discovered run means the
        only readable paths are the ones the server already found.
        """
        if not relative:
            return None
        # Normalise through the POSIX rules the URL used, then compare as a real
        # path, so "a/./b" and "a/b" are the same run and "../.." is neither.
        wanted = os.path.normpath(os.path.join(self.root, posixpath.normpath(relative)))
        for path in discover_runs(self.root):
            if os.path.normpath(path) == wanted:
                return read_run(path)
        return None


class DashboardHandler(BaseHTTPRequestHandler):
    """Answer the four views, and answer something for everything else.

    ``state`` is set on the subclass the server builds, since
    :class:`~http.server.HTTPServer` constructs a handler per request.
    """

    state = None  # type: Optional[DashboardState]
    server_version = "cadet-dashboard"
    # The default would add the Python version to every response header; the
    # dashboard says what it is and not what it is built on.
    sys_version = ""

    def log_message(self, format: str, *args) -> None:
        """One line per request on stderr, in the CLI's voice."""
        sys.stderr.write("[dashboard] {} {}\n".format(self.address_string(), format % args))

    def do_GET(self) -> None:
        """Route the request, and turn any failure into a page.

        The blanket handler is the acceptance criterion, not laziness: this is
        demonstrated live, and a reader that trips over an unforeseen file shape
        has to put a message on the screen rather than a traceback. The traceback
        still goes to stderr, where whoever is demonstrating can find it after.
        """
        try:
            body, status = self._dispatch()
        except Exception:  # noqa: BLE001 - see the docstring
            traceback.print_exc()
            body = render.not_found(
                "Something in this run tree could not be read. The details are on "
                "the console of the process serving this page.",
                self.state.root if self.state else "",
            )
            status = 500
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        # The live view re-requests on an interval and must not be answered from
        # a cache, or it would re-render the tree as it was when it started.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _dispatch(self) -> Tuple[str, int]:
        """Pick the view for this request path and query."""
        state = self.state
        assert state is not None  # set by serve(); a handler without one is a bug
        parts = urlsplit(self.path)
        query = parse_qs(parts.query)
        path = parts.path.rstrip("/") or "/"
        view = (query.get("view") or [""])[0]
        relative = (query.get("run") or [""])[0]

        if path == "/run" or path == "/device":
            record = state.resolve(relative)
            if record is None:
                return (
                    render.not_found(
                        "No run at {!r} under this tree. It may have been removed, "
                        "or the link may be from another tree.".format(relative),
                        state.root,
                    ),
                    404,
                )
            if path == "/device":
                return render.device_view(record, state.root), 200
            return render.run_view(record, state.root), 200

        if path == "/":
            tables = state.tables()
            if view == "live":
                return render.live_view(tables, state.root, state.refresh), 200
            if view == "runs":
                return (
                    render.page(
                        "CADET dashboard: runs",
                        render.runs_table(tables, state.root),
                    ),
                    200,
                )
            return render.sweep_view(tables, state.root), 200

        return (
            render.not_found("No view at {!r}.".format(parts.path), state.root),
            404,
        )


def serve(
    root: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    refresh: int = DEFAULT_REFRESH_S,
) -> int:
    """Bind and serve the dashboard until interrupted. Returns an exit code."""
    state = DashboardState(root, refresh=refresh)
    handler = type("BoundDashboardHandler", (DashboardHandler,), {"state": state})
    try:
        httpd = HTTPServer((host, port), handler)
    except OSError as exc:
        print("error: cannot serve on {}:{}: {}".format(host, port, exc), file=sys.stderr)
        return 2
    runs = len(discover_runs(state.root))
    print("serving {} run(s) from {}".format(runs, state.root))
    print("open http://{}:{}/".format(host, port))
    if not runs:
        print(
            "note: no run directory found yet; the page will show them as they appear",
            file=sys.stderr,
        )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0
