"""The CADET dashboard: a local page over a tree of run directories.

What this is for:
    ``cadet dashboard <runs-dir>`` serves four views over run directories the
    dashboard did not produce - a sweep matrix with its crossover plot, a run
    view, a device view, and a live view that refreshes while anything is
    running. It is demonstrated on a laptop that may have no network, so nothing
    is fetched at page load: the stylesheet is inline, the plots are SVG this
    package emits, and the only dependency is the standard library plus the
    ``PyYAML`` the config layer already requires.

What it contains:
    - :mod:`cadet.dashboard.runs`: read one run directory, degrading on anything
      missing or malformed.
    - :mod:`cadet.dashboard.sweep`: the grouping rule, in one function.
    - :mod:`cadet.dashboard.device`: the per-node device series.
    - :mod:`cadet.dashboard.svg`: the plots, emitted as markup.
    - :mod:`cadet.dashboard.render`: the four views as whole HTML pages.
    - :mod:`cadet.dashboard.server`: the standard-library HTTP server.

How it fits into CADET:
    It reads. It does not run, change or tear down a run, and it writes nothing
    into a run directory. Everything it shows was already on disk when the run
    ended. Entry point: ``cmd_dashboard`` in :mod:`cadet.cli`.

How to configure it:
    ``cadet dashboard <runs-dir> [--host H] [--port P] [--refresh S]``.
"""

from cadet.dashboard.runs import RunRecord, discover_runs, read_run
from cadet.dashboard.server import serve
from cadet.dashboard.sweep import SweepTable, group_runs, network_of, placement_of

__all__ = [
    "RunRecord",
    "SweepTable",
    "discover_runs",
    "group_runs",
    "network_of",
    "placement_of",
    "read_run",
    "serve",
]
