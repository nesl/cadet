"""CADET metrics: structured measurement of a run.

What this is for:
    Turning a run into comparable numbers. The paper's three-level taxonomy
    (model / system / task) is realised as per-frame records plus a run summary,
    written to CSV so sweeps can be aggregated downstream.

What it contains:
    - :mod:`cadet.metrics.frame_log` — :class:`MetricsLog`, the per-frame record
      log with a stable column order (original AV columns first, CADET columns
      appended) and a :meth:`MetricsLog.summary` roll-up.
    - :mod:`cadet.metrics.link_report` — the configured link parameters beside
      the measured per-message latency, per run.
    - :mod:`cadet.metrics.collectors` — the system-level collectors, one per
      node, each sampling its own node while the run happens on it, plus
      :mod:`~cadet.metrics.gpu`, :mod:`~cadet.metrics.cpu`,
      :mod:`~cadet.metrics.energy` and :mod:`~cadet.metrics.bandwidth`.
    - :mod:`cadet.metrics.clock_offset` — the per-run, per-node offset between
      a node's clock and the orchestrator's, with its uncertainty. Measured by
      the orchestrator rather than by a role, because a clock difference is a
      property of a pair of machines, and recorded in ``run.json``; it corrects
      nothing.

How it fits into CADET:
    The client runtime (:mod:`cadet.roles.client`) owns a :class:`MetricsLog` and
    writes it to the run's output directory. Server-side per-request metrics are
    written by :mod:`cadet.roles.server`. Both role runtimes additionally run a
    :class:`~cadet.metrics.collectors.CollectorSet` for their own node when the
    config asks for one.

How to configure it:
    Collectors and outputs are declared under ``metrics`` / ``outputs`` in the
    experiment config; this package defines the record format, not the values.
    ``metrics.collectors`` is empty by default, so an unconfigured run records
    exactly what it always did.
"""

from cadet.metrics.collectors import (
    CollectorError,
    CollectorSet,
    build_collectors,
)
from cadet.metrics.frame_log import (
    FIELDNAMES,
    NEW_COLUMNS,
    ORIGINAL_COLUMNS,
    MetricsLog,
)

__all__ = [
    "FIELDNAMES",
    "NEW_COLUMNS",
    "ORIGINAL_COLUMNS",
    "MetricsLog",
    "CollectorError",
    "CollectorSet",
    "build_collectors",
]
