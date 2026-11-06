"""CADET orchestrator: run and sweep lifecycle management.

Turns a declarative experiment config into actual processes. The orchestrator
owns the lifecycle of a single run (preflight, start sim hooks, start servers,
start clients, collect outputs, tear down) and of a sweep (enumerate the matrix,
run each cell, skip completed ones on resume).

Builds on :mod:`cadet.core.launcher` (the local/docker/ssh launcher interface)
and the role entry points in :mod:`cadet.roles`. The CLI (:mod:`cadet.cli`)
exposes it as ``cadet run`` / ``cadet sweep`` / ``cadet status`` / ``cadet down``.
Driven entirely by the experiment config (``nodes``, ``nodes.launcher``,
``outputs``, ``sweep``); this package defines orchestration behaviour, not values.
"""

from cadet.orchestrator.runner import Orchestrator, PreflightError, RunResult, preflight

__all__ = ["Orchestrator", "PreflightError", "RunResult", "preflight"]
