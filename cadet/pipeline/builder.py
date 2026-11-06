"""Pipeline builder: placement planning and the stage executor.

Turns an ordered list of placed stages into something runnable. Consecutive
stages on the same node form a :class:`Segment` (in-process); a node change is
a :class:`Boundary` the runtime realises as a transport link. The executor runs
a message through the chain in order, stamping entry/exit and handing each
stage its deadline budget. Provides :class:`PlacedStage`, :class:`Segment`,
:class:`Boundary`, :func:`plan_placement`, :class:`Pipeline` and
:func:`build_pipeline`. Boundaries are planned here but wired at run time
through :mod:`cadet.core.transport`, so unit tests need no network. Driven by
``pipeline.stages`` ({name, impl, node, params, queue_policy, deadline}).
"""

from typing import List, Optional, Tuple

from cadet.pipeline.deadline import ttd_ms
from cadet.pipeline.messages import Message
from cadet.pipeline.queue import LATEST_ONLY
from cadet.pipeline.registry import get_stage
from cadet.pipeline.stage import Stage


class PlacedStage:
    """A stage instance paired with its spec (node, queue policy, deadline)."""

    def __init__(self, stage: Stage, spec):
        self.stage = stage
        self.spec = spec

    @property
    def node(self) -> str:
        return self.spec.node

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def queue_policy(self) -> str:
        return getattr(self.spec, "queue_policy", LATEST_ONLY) or LATEST_ONLY


class Segment:
    """A maximal run of consecutive stages placed on the same node (in-process)."""

    def __init__(self, node: str):
        self.node = node
        self.stages: List[PlacedStage] = []

    @property
    def names(self) -> List[str]:
        return [s.name for s in self.stages]


class Boundary:
    """A node-to-node hop between two segments, realised as a transport link.

    ``queue_policy`` is the receiving stage's policy: what the downstream node keeps
    when messages arrive faster than it consumes them.
    """

    def __init__(self, from_node: str, to_node: str, queue_policy: str):
        self.from_node = from_node
        self.to_node = to_node
        self.queue_policy = queue_policy


def plan_placement(placed: List[PlacedStage]) -> Tuple[List[Segment], List[Boundary]]:
    """Group placed stages into same-node segments with boundaries between them."""
    segments: List[Segment] = []
    boundaries: List[Boundary] = []
    for ps in placed:
        if segments and segments[-1].node == ps.node:
            segments[-1].stages.append(ps)
        else:
            if segments:
                boundaries.append(Boundary(segments[-1].node, ps.node, ps.queue_policy))
            seg = Segment(ps.node)
            seg.stages.append(ps)
            segments.append(seg)
    return segments, boundaries


class Pipeline:
    """An ordered chain of placed stages, runnable message by message."""

    def __init__(self, placed: List[PlacedStage]):
        self.placed = placed
        self.segments, self.boundaries = plan_placement(placed)

    def setup(self, ctx=None) -> None:
        for ps in self.placed:
            ps.stage.setup(ctx)

    def teardown(self) -> None:
        for ps in reversed(self.placed):
            ps.stage.teardown()

    def run_once(self, msg: Message) -> Optional[Message]:
        """Run ``msg`` through every stage in order; return the final message.

        Returns ``None`` if a stage dropped the message. Each stage is stamped on
        entry and exit, handed its deadline budget via ``msg.deadline_ttd_ms``, and
        its output inherits the upstream timing. This is the synchronous path used
        for in-process segments and for offline replay (parity level 1).
        """
        cur = msg
        for ps in self.placed:
            deadline = getattr(ps.spec, "deadline", None)
            if deadline is not None:
                cur.deadline_ttd_ms = ttd_ms(
                    getattr(deadline, "policy", None), getattr(deadline, "budget_ms", None), cur
                )
            else:
                cur.deadline_ttd_ms = None
            cur.stamp(ps.stage.name, "enter")
            out = ps.stage.process(cur)
            if out is None:
                return None
            out.carry_from(cur)
            out.stamp(ps.stage.name, "exit")
            cur = out
        return cur

    def placement_summary(self) -> str:
        """One-line-per-segment description of the placement (for READMEs/reports)."""
        lines = []
        for i, seg in enumerate(self.segments):
            lines.append("[{}] {}: {}".format(i, seg.node, " -> ".join(seg.names)))
            if i < len(self.boundaries):
                b = self.boundaries[i]
                lines.append(
                    "      || boundary {} -> {} (queue={})".format(
                        b.from_node, b.to_node, b.queue_policy
                    )
                )
        return "\n".join(lines)


def build_pipeline(specs) -> Pipeline:
    """Instantiate stages from the registry for an iterable of stage specs."""
    placed = []
    for spec in specs:
        cls = get_stage(spec.impl)
        stage = cls(spec.name, dict(getattr(spec, "params", {}) or {}))
        placed.append(PlacedStage(stage, spec))
    return Pipeline(placed)
