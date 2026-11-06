"""The server's inbound queue honors per-connection freshness policies.

What is under test:
    A model slower than the frame rate must not work through a backlog of stale
    frames. ``latest_only`` keeps only the newest pending frame per connection
    (dropping the rest); a bounded ``fifo`` (a flat channel's ``max_queue``) keeps
    the newest ``max_queue`` and drops the oldest when full; the unbounded default
    keeps every frame (the original behavior). Disconnect markers are never
    dropped. Drops are counted for the server metrics.

These exercise the schedulers directly (no sockets): ``put`` with no intervening
``get`` is exactly the slow-consumer case the policy exists for.
"""

from types import SimpleNamespace

from cadet.roles.server import (
    LATEST_ONLY,
    Request,
    _FifoScheduler,
    _inbound_max_pending,
    _RoundRobinScheduler,
)

TIMEOUT = 0.01


def _config(pipeline_stages=None, channels=None):
    pipeline = None
    if pipeline_stages is not None:
        pipeline = SimpleNamespace(
            stages=[SimpleNamespace(node=n, queue_policy=p) for n, p in pipeline_stages]
        )
    chans = [SimpleNamespace(to=t, max_queue=q) for t, q in (channels or [])]
    return SimpleNamespace(pipeline=pipeline, channels=chans)


def _server(enforce=True):
    # Enforcement is opt-in (default = original unbounded); these tests enable it.
    return SimpleNamespace(name="server", enforce_queue_policy=enforce)


def test_resolver_default_is_original_unbounded_even_with_a_policy():
    # The deviation is opt-in: with enforce_queue_policy off (the default), a
    # declared policy is ignored and the queue stays unbounded (original).
    cfg = _config(pipeline_stages=[("server", "latest_only")], channels=[("server", 2)])
    assert _inbound_max_pending(cfg, _server(enforce=False)) is None


def test_resolver_pipeline_latest_only_is_bound_one():
    cfg = _config(pipeline_stages=[("server", "latest_only"), ("client", "fifo")])
    assert _inbound_max_pending(cfg, _server()) == LATEST_ONLY


def test_resolver_pipeline_fifo_is_unbounded():
    cfg = _config(pipeline_stages=[("server", "fifo")])
    assert _inbound_max_pending(cfg, _server()) is None


def test_resolver_flat_channel_max_queue_is_the_bound():
    cfg = _config(channels=[("server", 2)])
    assert _inbound_max_pending(cfg, _server()) == 2


def test_resolver_defaults_to_unbounded_when_no_policy():
    cfg = _config()
    assert _inbound_max_pending(cfg, _server()) is None


def test_resolver_pipeline_takes_precedence_over_channels():
    cfg = _config(pipeline_stages=[("server", "latest_only")], channels=[("server", 5)])
    assert _inbound_max_pending(cfg, _server()) == LATEST_ONLY


def _req(cid, payload):
    return Request(cid, payload, transport=None)


def _drain(sched):
    out = []
    while True:
        r = sched.get(TIMEOUT)
        if r is None:
            break
        out.append(r.payload)
    return out


def test_latest_only_keeps_only_the_newest_pending_frame():
    # A slow consumer: three frames arrive before a single get.
    sched = _FifoScheduler(max_pending=LATEST_ONLY)
    for frame in (1, 2, 3):
        sched.put(_req("c0", frame))
    # It never processes a frame older than the newest available.
    assert _drain(sched) == [3]
    assert sched.drops == 2


def test_fifo_max_queue_drops_oldest_when_full():
    sched = _FifoScheduler(max_pending=2)
    for frame in (1, 2, 3, 4):
        sched.put(_req("c0", frame))
    # Keeps the two newest, in order; the two oldest were dropped.
    assert _drain(sched) == [3, 4]
    assert sched.drops == 2


def test_unbounded_default_keeps_every_frame_in_order():
    sched = _FifoScheduler(max_pending=None)
    for frame in (1, 2, 3):
        sched.put(_req("c0", frame))
    assert _drain(sched) == [1, 2, 3]
    assert sched.drops == 0


def test_latest_only_is_per_connection():
    # Each connection keeps its own newest; one busy stream does not evict another.
    sched = _FifoScheduler(max_pending=LATEST_ONLY)
    sched.put(_req("a", "a1"))
    sched.put(_req("b", "b1"))
    sched.put(_req("a", "a2"))  # drops a1, not b1
    assert sorted(_drain(sched)) == ["a2", "b1"]
    assert sched.drops == 1


def test_disconnect_marker_is_never_dropped():
    # The None marker must survive the bound so the server sees the disconnect.
    sched = _FifoScheduler(max_pending=LATEST_ONLY)
    sched.put(_req("c0", 1))
    sched.put(_req("c0", 2))  # drops frame 1
    sched.put(_req("c0", None))  # disconnect marker, not bounded
    payloads = _drain(sched)
    assert 2 in payloads and None in payloads
    assert sched.drops == 1


def test_round_robin_also_bounds_per_client():
    sched = _RoundRobinScheduler(max_pending=LATEST_ONLY)
    sched.put(_req("a", "a1"))
    sched.put(_req("a", "a2"))  # drops a1
    sched.put(_req("b", "b1"))
    assert sorted(_drain(sched)) == ["a2", "b1"]
    assert sched.drops == 1


def test_block_when_full_never_drops_and_keeps_fifo_order():
    # The V2I RSU's blocking Queue(maxsize=10): a full bound blocks the producer
    # until a get() drains, so every frame survives in order (no drop-oldest). A
    # background producer fills past the bound; the consumer releases it.
    import threading
    import time as _t

    sched = _FifoScheduler(max_pending=2, block_when_full=True)
    produced = []

    def produce():
        for frame in (1, 2, 3, 4):
            sched.put(_req("c0", frame))
            produced.append(frame)

    t = threading.Thread(target=produce)
    t.start()
    # The producer blocks after filling the bound (two frames pending).
    _t.sleep(0.05)
    assert produced == [1, 2], "producer should block once the bound is full"
    # Draining one frame releases exactly one blocked put.
    assert sched.get(TIMEOUT).payload == 1
    out = [1] + _drain(sched)
    t.join(timeout=1.0)
    assert not t.is_alive()
    # Contiguous FIFO, nothing dropped - skipping frames would change TTC deltas.
    assert out == [1, 2, 3, 4]
    assert sched.drops == 0


def test_block_when_full_does_not_block_disconnect_marker():
    # A disconnect (payload=None) must never be held by the bound, even when full.
    sched = _FifoScheduler(max_pending=1, block_when_full=True)
    sched.put(_req("c0", 1))  # fills the bound
    sched.put(_req("c0", None))  # must not block
    payloads = _drain(sched)
    assert payloads == [1, None]
    assert sched.drops == 0
