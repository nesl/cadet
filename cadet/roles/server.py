"""Generic multi-connection server runtime.

What this is for:
    Accepts many concurrent client connections, each with its own reader + delay
    line, all feeding ONE shared request queue in front of a single model. A
    scheduler (``fifo`` default, or ``round_robin`` across clients) picks the next
    request; the model processes it and the reply goes back on the originating
    connection (through that connection's reverse delay line, if any).

What it contains:
    - :class:`Request`, :class:`_FifoScheduler`, :class:`_RoundRobinScheduler`.
    - :class:`ServerRuntime`: accept loop + feeder-per-connection + scheduler loop;
      records per request the queue wait, inference time and queue depth at enqueue.
    - :func:`build_server`.

How it fits into CADET:
    `cadet serve <config>`; the server node's ``scheduler`` field picks the policy.
    Per-request metrics are written to ``outputs/<experiment>/<app>/server_metrics.csv``
    (used by the concurrent-load experiments).

    **System metrics.** When ``metrics.collectors`` names them, this runtime also
    runs the per-node collectors (:mod:`cadet.metrics.collectors`) for its own
    node, around the serving loop, writing ``<collector>_<node>.csv`` into the
    run directory. If the ``energy`` collector is among them, each per-request
    record additionally carries that inference's own energy: the collector is
    asked for the average power over the window the request was being processed
    in, on the same monotonic clock the record's ``inference_s`` is measured on,
    so the attribution needs no second alignment step. Default is no collectors,
    and then nothing here changes.
"""

import csv
import os
import random
import socket
import time
from collections import OrderedDict, deque
from threading import Condition, Lock, Thread
from typing import List, Optional

from cadet.config.routes import (
    hops_in_direction,
    is_route,
    path_into_node,
    paths_into_node,
    route_hops,
)
from cadet.core.apps import get_app
from cadet.core.codec import set_wire_encoding_from_config
from cadet.core.network import build_path_network_model
from cadet.core.transport import (
    MODE_FRAME,
    PROTOCOL_TCP,
    FramedSocketTransport,
    MultiConnectionEndpoint,
)
from cadet.metrics.collectors import CollectorSet, build_collectors
from cadet.roles.common import StopSignal, node_by_role, output_dir, select_run_config


class Request:
    """One client request waiting for the model, tagged with its client id."""

    __slots__ = ("client_id", "payload", "transport", "enqueue_time", "depth", "drops")

    def __init__(self, client_id, payload, transport):
        self.client_id = client_id
        self.payload = payload  # None = disconnect marker
        self.transport = transport  # reply here
        self.enqueue_time = None
        self.depth = 0  # queue depth at enqueue
        self.drops = 0  # cumulative stale-frame drops at enqueue (for metrics)


# Per-connection inbound-queue freshness bound. ``max_pending`` is the most
# un-processed frames one connection may have waiting before the oldest is
# dropped. ``None`` keeps every frame (the original behavior); ``latest_only``
# (keep only the newest, drop stale) is this with a bound of 1. The same
# mechanism carries a flat channel's ``max_queue`` and a pipeline stage's
# ``queue_policy``. A model slower than the frame rate then never works through a
# backlog of out-of-date frames: it always sees one of the freshest frames.
LATEST_ONLY = 1


class _FifoScheduler:
    """First-in, first-out across all clients, with a per-connection freshness bound.

    ``max_pending`` bounds the un-processed frames one connection may have queued;
    when a new frame would exceed it, that connection's oldest pending frame is
    dropped and counted in :attr:`drops`. ``None`` is unbounded (keep every
    frame). Disconnect markers (``payload is None``) are never bounded or dropped.
    """

    def __init__(self, max_pending=None, block_when_full=False):
        self._dq = deque()
        self._cv = Condition()
        self._pending = {}  # client_id -> count of queued data requests
        self._max_pending = max_pending
        # When True, a full per-connection bound BLOCKS the producer (the feeder's
        # recv->put), creating TCP backpressure to the client - the V2I RSU's original
        # behavior (``Queue(maxsize=10).put()``, which blocks). When False (the default),
        # a full bound drops the connection's oldest frame instead (braking's policy).
        self._block_when_full = block_when_full
        self.drops = 0

    def put(self, req: Request) -> None:
        with self._cv:
            cid = req.client_id
            if req.payload is not None and self._max_pending is not None:
                if self._block_when_full:
                    # Backpressure: wait for a get() to drain this connection, exactly
                    # as the original's blocking Queue.put did. Never block a disconnect
                    # marker (payload is None, handled above).
                    while self._pending.get(cid, 0) >= self._max_pending:
                        self._cv.wait()
                else:
                    while self._pending.get(cid, 0) >= self._max_pending:
                        self._drop_oldest(cid)
            req.depth = len(self._dq)
            req.enqueue_time = time.monotonic()
            req.drops = self.drops
            self._dq.append(req)
            if req.payload is not None:
                self._pending[cid] = self._pending.get(cid, 0) + 1
            self._cv.notify_all()

    def _drop_oldest(self, cid) -> None:
        """Remove this connection's oldest queued data request (a stale-frame drop)."""
        for i, r in enumerate(self._dq):
            if r.client_id == cid and r.payload is not None:
                del self._dq[i]
                self._pending[cid] -= 1
                self.drops += 1
                return

    def get(self, timeout: float) -> Optional[Request]:
        with self._cv:
            deadline = time.monotonic() + timeout
            while not self._dq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(remaining)
            req = self._dq.popleft()
            if req.payload is not None:
                self._pending[req.client_id] = max(0, self._pending.get(req.client_id, 1) - 1)
            # Wake any producer blocked on a full per-connection bound (backpressure mode).
            self._cv.notify_all()
            return req


class _RoundRobinScheduler:
    """Round-robin across clients: one request per client in turn.

    Takes the same per-connection ``max_pending`` freshness bound as
    :class:`_FifoScheduler` (applied to each client's own queue).
    """

    def __init__(self, max_pending=None):
        self._clients = OrderedDict()  # client_id -> deque
        self._order = []
        self._next = 0
        self._pending = 0
        self._max_pending = max_pending
        self._cv = Condition()
        self.drops = 0

    def put(self, req: Request) -> None:
        with self._cv:
            if req.client_id not in self._clients:
                self._clients[req.client_id] = deque()
                self._order.append(req.client_id)
            dq = self._clients[req.client_id]
            if req.payload is not None and self._max_pending is not None:
                while sum(1 for r in dq if r.payload is not None) >= self._max_pending:
                    self._drop_oldest(dq)
            req.depth = self._pending
            req.enqueue_time = time.monotonic()
            req.drops = self.drops
            dq.append(req)
            self._pending += 1
            self._cv.notify()

    def _drop_oldest(self, dq) -> None:
        """Remove the oldest queued data request from one client's queue."""
        for i, r in enumerate(dq):
            if r.payload is not None:
                del dq[i]
                self._pending -= 1
                self.drops += 1
                return

    def get(self, timeout: float) -> Optional[Request]:
        with self._cv:
            deadline = time.monotonic() + timeout
            while self._pending == 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(remaining)
            n = len(self._order)
            for i in range(n):
                cid = self._order[(self._next + i) % n]
                dq = self._clients[cid]
                if dq:
                    self._next = (self._next + i + 1) % n
                    self._pending -= 1
                    return dq.popleft()
            return None


class PeerModels:
    """The network models for each connection into one node, by its peer.

    A node can be the far end of several paths at once - N sensors into one
    gateway, an ego and a roadside feed into one server - and those paths are
    not the same link. The server used to resolve one path and hand the same
    hops to every connection, varying only the RNG stream, so a cell whose
    uplinks differed could not be expressed: the difference was configured and
    then ignored.

    Call it as ``(client_id, peer_ip)`` for a plain accept loop, or pass
    :meth:`for_source` to :meth:`cadet.core.transport.MultiConnectionEndpoint.serve`
    as its ``model_for``, which is the fan-in form. Each connection gets fresh
    models with their own RNG stream, seeded from the experiment seed so a run
    stays reproducible.

    **Peers are told apart by address**, which is what a connection carries. Two
    client nodes on one host therefore cannot be distinguished, and they share
    the first path declared into this node; that is logged rather than hidden,
    because the alternative - the client naming itself in band - is a wire
    change and this is not the place to make one.
    """

    def __init__(self, config, node_name, seed=None, log=None):
        self._seed = seed
        self._log = log or (lambda m: None)
        default = path_into_node(config, node_name)
        self._default_hops = _hops_of(config, default)
        by_name = {node.name: node for node in config.nodes}
        self._hops_by_peer = {}
        self._peer_by_ip = {}
        shared = set()
        for peer, path in paths_into_node(config, node_name):
            self._hops_by_peer[peer] = _hops_of(config, path)
            address = _resolve_host(getattr(by_name.get(peer), "host", None))
            if address is None:
                continue
            if self._peer_by_ip.get(address, peer) != peer:
                shared.add(address)
                continue
            self._peer_by_ip[address] = peer
        if shared:
            self._log(
                "[server] peers share an address ({}); those connections use "
                "the first path declared into this node".format(
                    ", ".join(sorted(shared))
                )
            )

    def __call__(self, client_id, peer_ip=None):
        """Return ``(recv_model, send_model)`` for one connection."""
        hops = self._default_hops
        peer = self._peer_by_ip.get(peer_ip) if peer_ip is not None else None
        if peer is not None:
            hops = self._hops_by_peer.get(peer, hops)
        if not hops:
            return None, None
        base = None if self._seed is None else self._seed + client_id
        forward = [hop.network for hop in hops_in_direction(hops, "forward")]
        reverse = [hop.network for hop in hops_in_direction(hops, "reverse")]
        recv = send = None
        if forward:
            recv = build_path_network_model(forward, rng=random.Random(base))
        if reverse:
            send = build_path_network_model(
                reverse, rng=random.Random(None if base is None else base + 100000)
            )
        return recv, send

    def for_source(self, source_id):
        """``model_for`` for :class:`MultiConnectionEndpoint`: ``"ip:port"`` in."""
        host = str(source_id).rsplit(":", 1)[0] if source_id else None
        # The connection's ordinal is not available here, so the stream is
        # seeded from the source id instead: still reproducible, still
        # independent per connection.
        client_id = abs(hash(source_id)) % 100000 if source_id else 0
        return self(client_id, host)


def _addr_host(addr):
    """Return the host part of a socket address, or None."""
    if not addr:
        return None
    return addr[0]


def _resolve_host(host):
    """Resolve a config host to an address, or None if it cannot be resolved.

    A config names a node by host, and a connection arrives with an address, so
    the two have to be compared in the same form. An unresolvable host is not an
    error here: it only means that peer cannot be told apart by address, which
    the caller reports.
    """
    if not host:
        return None
    try:
        return socket.gethostbyname(host)
    except OSError:
        return None


def _hops_of(config, path):
    """Return the hops a path contributes to the connection that arrives on it."""
    if path is None:
        return []
    if not is_route(path):
        return [path]
    if path.realise == "relay":
        return route_hops(config, path)[-1:]
    return route_hops(config, path)


# How long a stopped server keeps *starting* queued requests before it writes
# its records and exits.
#
# The bound on the whole drain is this plus one request, because a request that
# starts just inside the budget runs to completion: an inference cannot be
# interrupted half way without throwing away the record it was about to make.
# Measured in the lifecycle matrix with a deliberately slow server
# (``extra_processing_ms: 1500``): teardown 2.0-3.5 s, against 0.1-0.4 s for a
# run whose client exits by itself and leaves nothing queued. The launcher's
# termination grace is five seconds, so a server whose per-request cost is
# above about four seconds would still be killed - and would lose the records
# this exists to keep. Nothing configures such a server today; the braking
# default is 13 ms.
STOP_DRAIN_S = 1.0


class _EndpointReply:
    """A reply handle for one fan-in connection.

    The shared processing loop replies with ``req.transport.send(reply)``. For a
    single-accept connection that transport is the connection's FramedSocketTransport;
    for a MultiConnectionEndpoint fan-in there is no per-connection transport object in
    the loop, so this routes the reply back to the originating source via the endpoint,
    keeping the loop identical for both paths.
    """

    def __init__(self, endpoint, source_id):
        self._endpoint = endpoint
        self._source_id = source_id

    def send(self, payload) -> None:
        self._endpoint.send(self._source_id, payload)


class ServerRuntime:
    """Multi-connection server: concurrent readers -> scheduler -> one model."""

    def __init__(
        self,
        host,
        port,
        model_factory,
        app,
        logger,
        warmup=True,
        extra_processing_s=0.0,
        node=None,
        scheduler="fifo",
        metrics_path=None,
        inbound_max_pending=None,
        server_delay_mode="blocking",
        collectors=None,
        expect=1,
        protocol=PROTOCOL_TCP,
        mode=MODE_FRAME,
        inbound_block_at=None,
    ):
        self.host = host
        self.port = port
        # The wire every connection into this server uses, from the run's one
        # `transport` section. Not per-connection: a peer cannot negotiate it.
        self.protocol = protocol
        self.mode = mode
        # How many client connections this server serves. 1 (the default) keeps the
        # original single-accept-loop behaviour; >1 means fan-in - the server accepts
        # ``expect`` clients through one MultiConnectionEndpoint (an ego + a roadside
        # feed into one RSU, N sensors into one gateway), each with its own per-source
        # network model (model_for) and the expect-N readiness barrier.
        self.expect = max(1, int(expect))
        # (client_id, peer_ip) -> (recv_model, send_model). The peer matters
        # because a node can be the far end of several paths at once and those
        # paths are not the same link.
        self.model_factory = model_factory
        # Fixed per-request compute cost, kept apart from the model's own time
        # so a config can state the two separately.
        self.extra_processing_s = extra_processing_s
        # How extra_processing_s is applied: "blocking" (original sleep) or "link"
        # (no server sleep; the delay is modelled on a configured reverse link).
        self.server_delay_mode = server_delay_mode
        # This server's own node, for its transport tuning.
        self.node = node
        self.app = app
        self.log = logger
        self.warmup = warmup
        self.scheduler = scheduler
        self.metrics_path = metrics_path
        # Per-connection inbound freshness bound (None = keep every frame). Set
        # from the config's queue policy; see :func:`_inbound_max_pending`.
        self.inbound_max_pending = inbound_max_pending
        # When set, the inbound bound BLOCKS the producer (backpressure) at this depth
        # instead of dropping - the V2I RSU's original Queue(maxsize=N).put() behavior.
        # Takes precedence over inbound_max_pending's drop policy when both are present.
        self.inbound_block_at = inbound_block_at
        # The per-node system-metric collectors for this node, if the config
        # asked for any. An empty set is the default and costs nothing.
        self.collectors = collectors if collectors is not None else CollectorSet()
        self.drops = 0  # stale frames dropped by the inbound queue (for metrics)

    def run(self) -> None:
        self.app.setup()
        if self.warmup and hasattr(self.app, "warmup"):
            self.log("[server] warming up model...")
            self.app.warmup()

        # Started after the warm-up so a warm-up inference is not integrated
        # into the run's energy, and before the socket so the record covers
        # every request this server answered.
        self.collectors.start()
        energy = self.collectors.get("energy")

        # inbound_block_at (V2I RSU): a bounded BLOCKING fifo queue (backpressure) at that
        # depth, matching the original's Queue(maxsize=N).put(); otherwise the configured
        # drop-oldest/unbounded policy.
        if self.inbound_block_at is not None:
            sched = _FifoScheduler(max_pending=self.inbound_block_at, block_when_full=True)
        elif self.scheduler == "fifo":
            sched = _FifoScheduler(max_pending=self.inbound_max_pending)
        else:
            sched = _RoundRobinScheduler(max_pending=self.inbound_max_pending)
        state = {"active": 0, "started": False, "next_id": 0, "stop": False}
        lock = Lock()
        records = []
        closers = []  # sockets/endpoints to close on stop (set per path below)

        # Being stopped is part of this loop, not an exception to it: the
        # scheduler wait below is bounded, so the flag is noticed within one
        # turn and the shutdown path at the bottom - teardown, and the
        # per-request metrics - runs as it does on a normal exit.
        def request_stop():
            state["stop"] = True
            for close in closers:
                try:
                    close()  # unblocks the acceptor / stops the endpoint
                except Exception:
                    pass

        stop_signal = StopSignal()
        stop_signal.install(request_stop, log=self.log)

        if self.expect > 1:
            # Fan-in: accept `expect` clients through one MultiConnectionEndpoint,
            # each with its own per-source network model (model_for). The readiness
            # marker is printed BEFORE serve() - serve() binds then blocks accepting
            # all clients, and the clients retry-connect, so the brief bind race is
            # harmless; printing after would deadlock (serve waits for clients, the
            # clients wait for the marker).
            self.log(
                "[server] Waiting for connection... on {}:{} (fan-in expect={})".format(
                    self.host, self.port, self.expect
                )
            )
            policy = (
                MultiConnectionEndpoint.POLICY_ROUND_ROBIN
                if self.scheduler == "round_robin"
                else MultiConnectionEndpoint.POLICY_FIFO
            )
            endpoint = MultiConnectionEndpoint.serve(
                self.host,
                self.port,
                expect=self.expect,
                policy=policy,
                node=self.node,
                model_for=self.model_factory.for_source,
                protocol=self.protocol,
                mode=self.mode,
            )
            closers.append(endpoint.close)
            with lock:
                state["started"] = True
                state["active"] = self.expect
            self.log("[server] {} client(s) connected (fan-in)".format(self.expect))

            # Optional per-source ingress hold (e.g. a V2I network hop in blocking
            # hop_mode). In link hop_mode the endpoint's own per-connection delay line
            # carries the delay instead, so apps leave on_ingress a no-op there.
            ingress = getattr(self.app, "on_ingress", None)

            def endpoint_feeder():
                # One drain of the merged endpoint queue into the shared scheduler,
                # reusing the same processing loop. The endpoint already applied each
                # connection's own delay line and tagged the source.
                while True:
                    sm = endpoint.recv(timeout_s=0.5)
                    if sm is None:  # every connection has closed
                        with lock:
                            state["active"] = 0
                        return
                    if ingress is not None:
                        ingress(sm.source_id, sm.payload)
                    sched.put(
                        Request(sm.source_id, sm.payload, _EndpointReply(endpoint, sm.source_id))
                    )

            Thread(target=endpoint_feeder, daemon=True).start()
        else:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.port))
            srv.listen(128)
            closers.append(srv.close)
            self.log(
                "[server] Waiting for connection... on {}:{} ({})".format(
                    self.host, self.port, self.scheduler
                )
            )

            # Optional per-source ingress hold (e.g. a V2I network hop): applied here
            # in the feeder thread, before the frame enters the scheduler, so it runs
            # concurrently with the worker's inference instead of serializing with it -
            # exactly as the original RSU slept in its per-connection receiver thread.
            ingress = getattr(self.app, "on_ingress", None)

            def feeder(transport, cid):
                while True:
                    try:
                        payload = transport.recv()
                    except Exception:
                        payload = None
                    if payload is not None and ingress is not None:
                        ingress(cid, payload)
                    sched.put(Request(cid, payload, transport))
                    if payload is None:
                        return

            def acceptor():
                while not state["stop"]:
                    try:
                        conn, _addr = srv.accept()
                    except OSError:
                        return
                    with lock:
                        cid = state["next_id"]
                        state["next_id"] += 1
                        state["active"] += 1
                        state["started"] = True
                    peer_ip = _addr_host(_addr)
                    recv_model, send_model = self.model_factory(cid, peer_ip)
                    transport = FramedSocketTransport(
                        conn,
                        recv_model,
                        send_model,
                        node=self.node,
                        protocol=self.protocol,
                        mode=self.mode,
                        # This side accepted the connection, which is the half of
                        # a websocket handshake it has to perform.
                        server_side=True,
                        # Names this connection in bandwidth_<node>.csv. The client
                        # id, not the peer address, because an ephemeral source port
                        # says nothing to a reader of the CSV.
                        label="client{}".format(cid),
                    )
                    Thread(target=feeder, args=(transport, cid), daemon=True).start()
                    self.log("[server] client {} connected".format(cid))

            Thread(target=acceptor, daemon=True).start()

        drain_until = None
        drain_started = None
        try:
            while True:
                if state["stop"] and drain_until is None:
                    # Stopped: answer what is already queued, then go. Bounded,
                    # because a client that is still sending would otherwise
                    # keep this loop alive past the orchestrator's grace and be
                    # killed anyway - with the records lost, which is the whole
                    # point of stopping gracefully.
                    drain_started = time.monotonic()
                    drain_until = drain_started + STOP_DRAIN_S
                    self.log("[server] draining for up to {}s".format(STOP_DRAIN_S))
                req = sched.get(timeout=0.05 if drain_until is not None else 0.5)
                if req is None:
                    if drain_until is not None:
                        break  # nothing left queued
                    with lock:
                        if state["started"] and state["active"] == 0:
                            break
                    continue
                if drain_until is not None and time.monotonic() > drain_until:
                    break
                if req.payload is None:  # disconnect marker
                    with lock:
                        state["active"] -= 1
                    self.log("[server] client {} disconnected".format(req.client_id))
                    continue
                queue_wait = time.monotonic() - req.enqueue_time
                t0 = time.monotonic()
                # Wall clock too: the per-frame log is on it, and it is the only
                # clock comparable across nodes.
                t0_wall = time.time()
                reply = self.app.handle(req.payload)
                if self.extra_processing_s and self.server_delay_mode == "blocking":
                    # Original behavior: a real blocking sleep on the server thread.
                    # Charged after the model, so this is processing time and not a
                    # delay on the wire: it holds the reply back without changing when
                    # the request arrived or what the model saw. In ``link`` mode the
                    # server does not sleep -- the delay is left to a configured
                    # reverse link instead.
                    time.sleep(self.extra_processing_s)
                t1 = time.monotonic()
                inference = t1 - t0
                # Energy over exactly the window ``inference_s`` measures, so
                # the row's energy and its latency are the same event.
                per_inference = energy.window(t0, t1) if energy is not None else None
                records.append(
                    (
                        req.client_id,
                        round(queue_wait, 6),
                        round(inference, 6),
                        req.depth,
                        req.drops,
                        reply is not None,
                        round(t0_wall, 6),
                        _round(getattr(per_inference, "avg_power_w", None), 4),
                        _round(getattr(per_inference, "energy_j", None), 6),
                        getattr(per_inference, "samples", None),
                    )
                )
                if reply is not None:
                    try:
                        req.transport.send(reply)
                    except Exception:
                        pass
                    if self.app.stop_after_reply():
                        break
        finally:
            if drain_until is not None:
                # What the drain actually cost, which is the number to read when
                # a teardown looks slow.
                self.log(
                    "[server] drained in {:.2f}s".format(time.monotonic() - drain_started)
                )
            state["stop"] = True
            self.drops = sched.drops
            for close in closers:
                try:
                    close()
                except Exception:
                    pass
            self.app.teardown()
            self._write_metrics(records)
            # Last, so the final window covers as much of the teardown as this
            # role can account for. The per-request energy above is already
            # computed from the live collector, so writing the records first
            # costs nothing; what it buys is that a reply still going out, or a
            # frame still arriving on a feeder thread, is inside this node's
            # totals rather than counted only by the node at the other end.
            self.collectors.stop()
            if self.inbound_max_pending is not None:
                self.log(
                    "[server] inbound queue bound={} stale-frame drops={}".format(
                        self.inbound_max_pending, self.drops
                    )
                )

        # The records are written; give the process back to the signal so its
        # exit status still says it was stopped rather than finished.
        stop_signal.finish()

    def _write_metrics(self, records) -> None:
        if not self.metrics_path:
            return
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        with open(self.metrics_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "client_id",
                    "queue_wait_s",
                    "inference_s",
                    "queue_depth_at_enqueue",
                    "stale_drops_so_far",
                    "replied",
                    # Appended at the end only (the rule the per-frame CSV
                    # follows too). Empty unless the energy collector ran.
                    "inference_start_wall",
                    "avg_power_w",
                    "energy_j",
                    "power_samples",
                ]
            )
            for r in records:
                w.writerow(r)
        self.log(
            "[server] per-request metrics -> {} ({} requests)".format(
                self.metrics_path, len(records)
            )
        )


def _round(value, digits):
    """Round a measurement for the CSV, leaving an absent one absent."""
    return None if value is None else round(value, digits)


def build_server(
    config_path: str,
    overrides: List[str],
    run_index: int,
    model_dirs: List[str],
    logger=None,
    node_name=None,
    rep: int = 0,
    bind_host: Optional[str] = None,
):
    """Build a :class:`ServerRuntime` for one run from a config file.

    ``node_name`` selects which server node to bind (needed when a config has more
    than one server); it falls back to the first node with role ``server``.

    ``rep`` names the repetition, because the per-request metrics and the
    per-node metrics CSVs are written into that repetition's run directory and
    two reps of one config would otherwise write to one file.

    ``bind_host`` overrides the address this server listens on, leaving the
    config alone. It exists for a role whose container does not hold the address
    its peers use: a node on another machine is reached at that machine's
    address and its container has to listen on one of its own. It cannot be a
    config override, because the config is hashed into the run id and the
    orchestrator and the role would then disagree about which run this is.
    """
    log = logger or (lambda m: print(m))
    config, run_id_base = select_run_config(config_path, overrides, run_index)
    # The wire encoding is a per-process choice the apps read through
    # cadet.core.codec, so it is selected once here, before any transport
    # exists. Selecting it later would leave a payload encoded one way and
    # decoded another.
    set_wire_encoding_from_config(config)
    run_id = "{}-r{}".format(run_id_base, rep)

    if node_name is not None and node_name not in ("server",):
        server_node = next((n for n in config.nodes if n.name == node_name), None)
        if server_node is None or server_node.role != "server":
            raise RuntimeError("no server node named {!r}".format(node_name))
    else:
        server_node = node_by_role(config, "server")
    if server_node is None:
        raise RuntimeError("config has no node with role 'server'")

    # What shapes each connection into this server: a link, a composed route
    # whose hops become one end-to-end model, or a relayed route - where the
    # connection this server accepts is only the route's last hop, because the
    # earlier hops are other processes' connections and are shaped where they
    # are sent. A server that is the far end of several paths gets one per peer;
    # see :class:`PeerModels`.
    seed = config.experiment.seed

    model_factory = PeerModels(config, server_node.name, seed=seed, log=log)

    # How many clients connect to this server. One distinct peer path in = one
    # expected connection; >1 (an ego + a roadside feed into one RSU, N sensors into
    # one gateway) switches the server to MultiConnectionEndpoint fan-in with the
    # expect-N barrier. One or zero keeps the original single-accept loop (the default
    # for every existing config).
    expect = max(1, len({peer for peer, _ in paths_into_node(config, server_node.name)}))

    # Everything this node records goes in the run directory, beside the
    # client's records. The per-request file used to sit one level up, in the
    # experiment directory, named with the config hash - which is why it was
    # reported as never written when it was being written all along, and why two
    # reps of one config wrote to the same file.
    run_dir = output_dir(config, run_id)
    metrics_path = os.path.join(run_dir, "server_metrics_{}.csv".format(server_node.name))
    scheduler = getattr(server_node, "scheduler", None) or "fifo"

    # Created here rather than waited for: the orchestrator starts the servers
    # before the client that would otherwise create it.
    collectors = build_collectors(config, server_node.name, run_dir, logger=log)
    if len(collectors):
        os.makedirs(run_dir, exist_ok=True)

    app = get_app(config.app.plugin).build_server(config, model_dirs, logger=log)
    return ServerRuntime(
        bind_host or server_node.host,
        server_node.port,
        model_factory,
        app,
        log,
        warmup=config.models.warmup,
        extra_processing_s=(getattr(server_node, "extra_processing_ms", None) or 0.0) / 1000.0,
        node=server_node,
        scheduler=scheduler,
        metrics_path=metrics_path,
        inbound_max_pending=_inbound_max_pending(config, server_node),
        server_delay_mode=getattr(server_node, "server_delay_mode", "blocking"),
        collectors=collectors,
        expect=expect,
        protocol=config.transport.protocol,
        mode=config.transport.mode,
        inbound_block_at=getattr(server_node, "inbound_block_at", None),
    )


def _inbound_max_pending(config, server_node):
    """Per-connection pending-frame bound for this server node's inbound queue.

    Opt-in deviation (default = original). Unless the server node sets
    ``enforce_queue_policy: true``, this returns ``None`` (keep every frame -- the
    original unbounded FIFO). When enforced, the same freshness mechanism carries
    both config forms (see :class:`ServerRuntime` / :data:`LATEST_ONLY`):

    - a placed pipeline: the first stage on this node receives from the upstream
      boundary, so its ``queue_policy`` sets the bound -- ``latest_only`` -> 1,
      ``fifo`` -> unbounded;
    - a flat ``channels`` config: the channel into this node sets the bound to its
      ``max_queue`` (a bounded fifo, drop-oldest);
    - neither: ``None``.

    Pipeline placement takes precedence when both are present.
    """
    if not getattr(server_node, "enforce_queue_policy", False):
        return None  # default: original unbounded behavior
    pipeline = getattr(config, "pipeline", None)
    stages = getattr(pipeline, "stages", []) if pipeline is not None else []
    local = [s for s in stages if s.node == server_node.name]
    if local:
        return LATEST_ONLY if local[0].queue_policy == "latest_only" else None
    inbound = [c for c in getattr(config, "channels", []) or [] if c.to == server_node.name]
    if inbound:
        return inbound[0].max_queue
    return None
