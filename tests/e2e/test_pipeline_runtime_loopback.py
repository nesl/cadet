"""End-to-end-ish test of the pipeline runtime over real sockets (no CARLA).

Uses socketpairs as the boundary transports and runs serve_hop in threads, so the
transport bridging is exercised for real while staying deterministic: a two-node
round trip (reply-on-event, as braking does) and a three-node forward chain
(origin -> mid -> terminal, result relayed back).
"""

import socket
import threading

from cadet.config.schema import StageSpec
from cadet.core.transport import FramedSocketTransport
from cadet.pipeline.builder import PlacedStage
from cadet.pipeline.messages import Detections, Frame, Message
from cadet.pipeline.registry import register_stage
from cadet.pipeline.runtime import deserialize_message, run_segment, serialize_message, serve_hop
from cadet.pipeline.stage import Stage


# --- synthetic stages ---------------------------------------------------------
@register_stage("rt_detect")
class _Detect(Stage):
    def process(self, msg):
        return Detections(detections=[("car", 0.9, 400)])


@register_stage("rt_emit_at3")
class _EmitAt3(Stage):
    """Emit a message only on the 3rd consecutive call (reply-on-event)."""

    def __init__(self, name, params=None):
        super().__init__(name, params)
        self.n = 0

    def process(self, msg):
        self.n += 1
        if self.n >= 3:
            out = Message()
            out.decided_on = self.n
            return out
        return None


@register_stage("rt_tag")
class _Tag(Stage):
    """Append this stage's name to msg.trail (to prove multi-hop ordering)."""

    def process(self, msg):
        out = Message()
        out.trail = list(getattr(msg, "trail", [])) + [self.name]
        return out


def _placed(name, impl, node, stage):
    return PlacedStage(stage, StageSpec(name=name, impl=impl, node=node))


def _pair():
    a, b = socket.socketpair()
    return FramedSocketTransport(a), FramedSocketTransport(b)


def test_two_node_round_trip_reply_on_event():
    # origin <-> server; server runs detect -> emit_at3 (brake-like reply-on-event).
    up_origin, up_server = _pair()
    stages = [
        _placed("detect", "rt_detect", "server", _Detect("detect")),
        _placed("decide", "rt_emit_at3", "server", _EmitAt3("decide")),
    ]
    t = threading.Thread(target=serve_hop, args=(up_server, stages), daemon=True)
    t.start()

    decisions = []
    for i in range(5):
        up_origin.send(serialize_message(Frame(image=0, frame_id=i, wall_time=float(i))))
        reply = deserialize_message(up_origin.recv())
        decisions.append(reply)
    up_origin.close()
    t.join(timeout=5)

    # Frames 0,1 -> None (counter < 3); frame 2 onwards -> a decision.
    assert decisions[0] is None and decisions[1] is None
    assert decisions[2] is not None and decisions[2].decided_on == 3
    # timing propagated across the boundary: the reply carries the origin frame id
    assert decisions[2].frame_id == 2


def test_three_node_forward_chain_relays_result_back():
    # origin -> mid -> terminal; each tags the message, result relayed back to origin.
    o_up, mid_in = _pair()  # origin <-> mid
    mid_out, term_in = _pair()  # mid <-> terminal

    mid_stages = [_placed("mid", "rt_tag", "edge", _Tag("mid"))]
    term_stages = [_placed("term", "rt_tag", "server", _Tag("term"))]

    tt = threading.Thread(target=serve_hop, args=(term_in, term_stages), daemon=True)
    tm = threading.Thread(
        target=serve_hop, args=(mid_in, mid_stages), kwargs={"downstream": mid_out}, daemon=True
    )
    tt.start()
    tm.start()

    o_up.send(serialize_message(Message(frame_id=1)))
    result = deserialize_message(o_up.recv())
    o_up.close()
    tm.join(timeout=5)
    tt.join(timeout=5)

    # The message passed through mid then terminal, in order.
    assert result.trail == ["mid", "term"]


def test_run_segment_drops_on_none():
    stages = [_placed("e", "rt_emit_at3", "n", _EmitAt3("e"))]
    # first two calls drop (return None), third emits
    assert run_segment(stages, Message(frame_id=0)) is None
    assert run_segment(stages, Message(frame_id=1)) is None
    assert run_segment(stages, Message(frame_id=2)) is not None
