"""Live pipeline runtime: drive transport connections from the boundary plan.

What this is for:
    Turn a placed pipeline into running processes connected by real transport links.
    Connections are derived from :func:`cadet.pipeline.builder.plan_placement`'s
    boundary list — NOT a fixed client/server pair — so the same code runs a
    two-node braking split, a three-node placement chain (P6), and (later) longer
    chains, by config alone.

What it contains:
    - :class:`ChainPlan` + :func:`plan_chain`: from the boundaries, work out the
      node chain — the origin (runs the source, and the final sink when the chain
      returns to it), the ordered "away" compute nodes, and for each node its local
      stages, its upstream/downstream neighbour and the inbound queue policy.
    - :func:`serialize_message` / :func:`deserialize_message`: the wire format
      across a boundary (:mod:`cadet.core.codec`, matching the flat apps, so
      ``transport.encoding`` selects it).
    - :func:`run_segment`: run a node's local stages over one message (the generic,
      app-agnostic compute step used on every hop).
    - :func:`serve_hop`: the compute-node loop — receive, run the local segment,
      forward downstream if there is a next node, reply upstream.

How it fits into CADET:
    Linear chains map onto the existing roles (the orchestrator launches ``cadet
    serve`` for each compute node and ``cadet client`` for the origin), so the
    runtime needs only the public :class:`cadet.core.transport` API
    (``connect``/``serve_one``/``send``/``recv``). A compute node with a downstream
    neighbour opens one outbound connection to it; a node that is the fan-in point of
    *several* inbound streams (V2I) would need multi-accept, which the public API
    does not offer, so this does not work around it.

How to configure it:
    Driven entirely by ``pipeline.stages`` placement and ``nodes`` host/port.
"""

from typing import List, Optional

from cadet.core.codec import decode as wire_decode
from cadet.core.codec import encode as wire_encode
from cadet.pipeline.builder import plan_placement
from cadet.pipeline.messages import Message
from cadet.pipeline.queue import LATEST_ONLY


def serialize_message(msg: Optional[Message]) -> bytes:
    """Wire-encode a message (or ``None`` for "no output this frame")."""
    return wire_encode(msg)


def deserialize_message(payload: bytes) -> Optional[Message]:
    """Decode a wire message produced by :func:`serialize_message`."""
    return wire_decode(payload)


class ChainPlan:
    """The per-node view of a linear pipeline chain, derived from the boundaries.

    ``node_chain`` is the ordered list of segment nodes (origin first). The origin
    runs the head segment (the source) and, when the chain returns to it, the tail
    segment (the sink). The nodes between are the ``away`` compute hops.
    """

    def __init__(self, segments, boundaries):
        self.segments = segments
        self.boundaries = boundaries
        self.node_chain: List[str] = [s.node for s in segments]
        self.origin: Optional[str] = self.node_chain[0] if segments else None
        # The chain "returns" when its last segment is back on the origin (a
        # request/reply round trip, e.g. braking's control stage on the client).
        self.origin_has_tail = len(segments) > 1 and segments[-1].node == self.origin
        self.head = segments[0] if segments else None
        self.tail = segments[-1] if self.origin_has_tail else None
        self.away_segments = segments[1:-1] if self.origin_has_tail else segments[1:]

    def head_stages(self):
        """Stages the origin runs on each source message before the first boundary."""
        return list(self.head.stages) if self.head else []

    def tail_stages(self):
        """Stages the origin runs on the returned result (empty if no return)."""
        return list(self.tail.stages) if self.tail else []

    def local_stages(self, node: str):
        """Stages of the away segment placed on ``node`` (compute hop)."""
        for seg in self.away_segments:
            if seg.node == node:
                return list(seg.stages)
        return []

    def first_away_node(self) -> Optional[str]:
        """The node the origin connects to (first compute hop), or ``None``."""
        return self.away_segments[0].node if self.away_segments else None

    def downstream_of(self, node: str) -> Optional[str]:
        """The next away node after ``node`` (its forward connect target), or ``None``."""
        names = [seg.node for seg in self.away_segments]
        if node in names:
            i = names.index(node)
            if i + 1 < len(names):
                return names[i + 1]
        return None

    def inbound_policy(self, node: str) -> str:
        """Queue policy the given node applies to its inbound boundary."""
        for b in self.boundaries:
            if b.to_node == node:
                return b.queue_policy
        return LATEST_ONLY


def plan_chain(pipeline) -> ChainPlan:
    """Build a :class:`ChainPlan` from a :class:`cadet.pipeline.builder.Pipeline`."""
    segments, boundaries = plan_placement(pipeline.placed)
    return ChainPlan(segments, boundaries)


def run_segment(stages, msg: Message) -> Optional[Message]:
    """Run ``msg`` through a node's local ``stages`` in order.

    Returns the final message, or ``None`` as soon as a stage drops it (e.g. the
    braking decision stage emits nothing until it triggers). Each stage is stamped
    on entry/exit and its output inherits the upstream timing.
    """
    cur = msg
    for ps in stages:
        cur.stamp(ps.stage.name, "enter")
        out = ps.stage.process(cur)
        if out is None:
            return None
        out.carry_from(cur)
        out.stamp(ps.stage.name, "exit")
        cur = out
    return cur


def serve_hop(inbound, stages, downstream=None) -> None:
    """Compute-node loop: receive, run the local segment, forward/reply.

    ``inbound`` is the transport serving the upstream node. For each message: run
    the local ``stages``; if a stage drops it, reply ``None`` (upstream treats that
    as "no output this frame" and keeps streaming). If there is a ``downstream``
    transport, forward the local output and relay its reply back upstream; otherwise
    reply with the local output. Returns when the upstream closes.
    """
    while True:
        raw = inbound.recv()
        if raw is None:
            return
        msg = deserialize_message(raw)
        out = run_segment(stages, msg)
        if out is None:
            inbound.send(serialize_message(None))
            continue
        if downstream is not None:
            downstream.send(serialize_message(out))
            back = downstream.recv()
            out = deserialize_message(back) if back is not None else None
        inbound.send(serialize_message(out))
