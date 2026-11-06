"""Per-stage deadlines: the time-to-decision budget handed to a stage.

Ported from the original ``deadline_enforcement`` semantics. A stage can be
given a time-to-decision (ttd) budget so it can trade fidelity for latency:
the Frenet planner, for example, picks a cheaper configuration when little time
remains. This module computes the budget and attaches it to the message; the
stage reads it only if it adapts its work. Policy ``none`` passes ``None``
(full fidelity); ``static`` passes a fixed ``budget_ms`` every frame; ``dynamic``
passes the time remaining until the end-to-end deadline (``budget_ms - elapsed``),
which can go negative. The executor in :mod:`cadet.pipeline.builder` calls
:func:`ttd_ms` before each stage. Configuration is per stage in the config
(``deadline: {policy, budget_ms}``).
"""

import time
from typing import Optional

NONE = "none"
STATIC = "static"
DYNAMIC = "dynamic"


def ttd_ms(policy: Optional[str], budget_ms: Optional[float], msg, now: Optional[float] = None):
    """Return the time-to-decision budget (ms) for ``msg`` under ``policy``.

    ``None`` means "no budget" (full fidelity). For ``dynamic`` the message's
    ``wall_time`` is the frame's capture time; if it is missing, the remaining
    budget cannot be computed and the fixed ``budget_ms`` is returned instead.
    """
    if not policy or policy == NONE:
        return None
    if policy == STATIC:
        return budget_ms
    if policy == DYNAMIC:
        if budget_ms is None:
            return None
        if msg is None or getattr(msg, "wall_time", None) is None:
            return budget_ms
        now = time.time() if now is None else now
        elapsed_ms = (now - msg.wall_time) * 1000.0
        return budget_ms - elapsed_ms
    raise ValueError("unknown deadline policy {!r}".format(policy))
