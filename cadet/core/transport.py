"""Transport interface: how roles exchange length-framed payloads over a link.

What this is for:
    One abstract way to ``send``/``recv`` byte payloads between roles over a real
    TCP socket, optionally shaped by a :class:`~cadet.core.network.NetworkModel`
    so network effects (delay, jitter, loss, bandwidth) can be emulated.

What it contains:
    - :class:`Transport`: the abstract send/recv contract.
    - the wire layers ``protocol`` and ``mode`` select: :class:`_SocketChannel`
      and :class:`_WebSocketChannel` carry bytes; :class:`_LengthFraming` and
      :class:`_StreamFraming` find a payload's boundaries in them.
    - :class:`_RecvDelayLine` / :class:`_SendDelayLine`: per-direction delay lines
      (reader/sender threads with a non-decreasing release schedule).
    - :class:`FramedSocketTransport`: the real TCP transport used by the runtimes.
    - :class:`SourceMessage` and :class:`MultiConnectionEndpoint`: a server side
      that accepts several clients at once and merges them into one queue.
    - :class:`LinkCounters` and :func:`counters_for` / :func:`counter_snapshot`:
      per-link, per-direction byte and message counters, which is where the
      ``bandwidth`` collector (:mod:`cadet.metrics.bandwidth`) reads from.

How it fits into CADET:
    Built from ``config.links``; the client and server runtimes hold a transport
    each. The link's forward/reverse ``network`` models attach to the recv/send
    delay lines.

    **Two independent choices, from ``transport``.** ``protocol`` picks what
    carries the bytes - ``tcp`` is the socket itself, ``websocket`` puts the same
    bytes inside RFC 6455 binary frames after one HTTP upgrade
    (:mod:`cadet.core.websocket`), which is what lets a link cross
    infrastructure that forwards HTTP and nothing else. ``mode`` picks how a
    payload's boundaries are found on top of that - ``frame`` prefixes each
    payload with its length (:mod:`cadet.core.service`, what every example has
    always used), ``stream`` sends no prefix at all and lets the receiver read
    each record's extent out of the record itself
    (:func:`cadet.core.codec.record_length`), which is why stream mode needs
    ``transport.encoding: safe``: a pickle payload carries no length and cannot
    be delimited in a byte stream.

    The two choices compose, and everything above them is unchanged: the delay
    lines, the counters, :class:`MultiConnectionEndpoint` and the apps see the
    same payloads whichever pair is configured.

    **One connection or many.** :meth:`FramedSocketTransport.serve_one` is the
    single-client server every example used, and it keeps working unchanged.
    :class:`MultiConnectionEndpoint` is the general form: it accepts N clients,
    gives each its own reader and its own delay line, and merges them into one
    inbound queue with every message tagged by the connection it came from. Three
    things need that shape - V2I fan-in (an ego and an infrastructure stream into
    one roadside node), the ``load`` role (N synthetic clients into one server),
    and a pipeline fan-in boundary.

    Each connection keeping **its own delay line** is the point, not an
    implementation detail: a link's delay is a property of that connection, so
    merging first and delaying afterwards would attribute one client's latency to
    another and make per-connection statistics meaningless.

    **Counted where the bytes are.** A transport knows how much it carried and
    which connection carried it, so the byte counters live here rather than
    being inferred from a payload size somewhere else. They count payload bytes
    and messages per direction; the 4-byte length prefix from
    :mod:`cadet.core.service`, the websocket frame header and everything TCP adds
    are excluded, so framing overhead stays derivable from the message count and
    the configured protocol instead of being folded into the byte count.

How to configure it:
    Link network models come from the config; this module only defines behaviour,
    not values. ``transport.protocol`` and ``transport.mode`` select the wire
    layers above, and the role runtimes pass them to :meth:`connect`,
    :meth:`serve_one` and :meth:`MultiConnectionEndpoint.serve`. The counters are
    always maintained (an integer add per message) and are only *recorded* when
    ``metrics.collectors`` names ``bandwidth``.
"""

import queue
import socket
import time
from abc import ABC, abstractmethod
from threading import Lock, Thread
from typing import Any, Dict, List, Optional

from cadet.core import websocket as ws
from cadet.core.codec import record_length
from cadet.core.network import NetworkModel, NullNetworkModel
from cadet.core.service import recv_msg, send_msg

# What `transport.protocol` and `transport.mode` select. Named here because the
# transport is where they take effect; the schema declares the same spellings.
PROTOCOL_TCP = "tcp"
PROTOCOL_WEBSOCKET = "websocket"
PROTOCOLS = (PROTOCOL_TCP, PROTOCOL_WEBSOCKET)

MODE_FRAME = "frame"
MODE_STREAM = "stream"
MODES = (MODE_FRAME, MODE_STREAM)

# How much a stream-mode reader asks the socket for at a time. A read size, not
# a message size: stream mode has no message size to know in advance, and the
# reader keeps reading until a record is complete.
STREAM_READ_BYTES = 65536


class LinkCounters:
    """Payload bytes and messages one connection carried, per direction.

    ``sent`` is what this node handed to the link and ``recv`` what the link
    delivered to it, so the pair read from the two ends of one link describes
    both directions without either end guessing about the other.
    """

    __slots__ = ("label", "bytes_sent", "bytes_recv", "messages_sent", "messages_recv")

    def __init__(self, label: str):
        self.label = label
        self.bytes_sent = 0
        self.bytes_recv = 0
        self.messages_sent = 0
        self.messages_recv = 0

    def add_sent(self, size: int) -> None:
        """Count one outbound message of ``size`` payload bytes."""
        self.bytes_sent += size
        self.messages_sent += 1

    def add_recv(self, size: int) -> None:
        """Count one inbound message of ``size`` payload bytes."""
        self.bytes_recv += size
        self.messages_recv += 1

    def snapshot(self) -> "LinkCounters":
        """A detached copy, so a sampler reads consistent totals."""
        copy = LinkCounters(self.label)
        copy.bytes_sent = self.bytes_sent
        copy.bytes_recv = self.bytes_recv
        copy.messages_sent = self.messages_sent
        copy.messages_recv = self.messages_recv
        return copy

    def __repr__(self) -> str:
        return "LinkCounters({!r}, sent={}B/{}, recv={}B/{})".format(
            self.label, self.bytes_sent, self.messages_sent, self.bytes_recv, self.messages_recv
        )


# Per-node registry, keyed by link label. Module level because the counters have
# to outlive the transport that filled them: a connection can close before the
# collector takes its last sample, and the bytes it carried are still part of
# the run. One role process is one node, so this registry is that node's.
_COUNTERS: Dict[str, LinkCounters] = {}
_COUNTERS_LOCK = Lock()


def counters_for(label: str) -> LinkCounters:
    """Return (creating if needed) the counters for the link called ``label``."""
    with _COUNTERS_LOCK:
        counters = _COUNTERS.get(label)
        if counters is None:
            counters = LinkCounters(label)
            _COUNTERS[label] = counters
        return counters


def counter_snapshot() -> List[LinkCounters]:
    """Return a detached copy of every link's counters, in label order."""
    with _COUNTERS_LOCK:
        labels = sorted(_COUNTERS)
        return [_COUNTERS[label].snapshot() for label in labels]


def reset_counters() -> None:
    """Forget every link's counters (used between tests, and by a re-run role)."""
    with _COUNTERS_LOCK:
        _COUNTERS.clear()


class Transport(ABC):
    """Abstract bidirectional payload transport."""

    @abstractmethod
    def send(self, payload: Any) -> None:
        """Enqueue/transmit a payload."""

    @abstractmethod
    def recv(self) -> Optional[Any]:
        """Return the next payload, or ``None`` if none is available."""


class _SocketChannel:
    """Bytes straight on the socket: ``transport.protocol: tcp``.

    A channel presents ``sendall`` and ``recv`` so that
    :mod:`cadet.core.service`'s framing reads it exactly as it reads a socket,
    plus ``recv_some`` for the stream framing, which wants whatever has arrived
    rather than a count.
    """

    def __init__(self, conn):
        self._conn = conn
        # One payload is one `sendall`, but a transport can be written to from
        # more than one thread (a reply path and a broadcast), and two
        # interleaved `sendall` calls would splice two payloads together.
        self._send_lock = Lock()

    def sendall(self, data: bytes) -> None:
        with self._send_lock:
            self._conn.sendall(data)

    def recv(self, n: int) -> bytes:
        return self._conn.recv(n)

    def recv_some(self) -> Optional[bytes]:
        data = self._conn.recv(STREAM_READ_BYTES)
        return bytes(data) if data else None

    def finish(self) -> None:
        """Say goodbye at the protocol's own level (nothing to say over TCP)."""


class _WebSocketChannel:
    """Bytes inside RFC 6455 binary frames: ``transport.protocol: websocket``.

    The handshake has already happened when this is constructed (the connecting
    and accepting sides each do their half, which is the one thing they cannot
    share). After it, a websocket is a reliable ordered byte carrier like the
    socket under it, so the framings above work unchanged - the buffer here is
    only because a frame boundary need not fall where a reader asked for bytes.
    """

    def __init__(self, conn, server_side: bool):
        self._conn = conn
        # The RFC: a client masks what it sends, a server does not.
        self._mask_out = not server_side
        self._buffer = bytearray()
        self._eof = False
        self._send_lock = Lock()

    def sendall(self, data: bytes) -> None:
        with self._send_lock:
            ws.send_binary(self._conn, data, self._mask_out)

    def _fill(self) -> bool:
        """Pull one more payload into the buffer; False once the peer is done."""
        if self._eof:
            return False
        payload = ws.recv_payload(self._conn, self._mask_out)
        if payload is None:
            self._eof = True
            return False
        self._buffer.extend(payload)
        return True

    def recv(self, n: int) -> bytes:
        """Up to ``n`` buffered bytes, or empty at close (a socket's contract)."""
        if not self._buffer and not self._fill():
            return b""
        if n >= len(self._buffer):
            # The whole buffer: handed over without slicing it, because the
            # common case is a frame-sized read of a megabyte payload and the
            # slice plus the front deletion would be two more copies of it.
            out = self._buffer
            self._buffer = bytearray()
            return bytes(out)
        out = bytes(self._buffer[:n])
        del self._buffer[:n]
        return out

    def recv_some(self) -> Optional[bytes]:
        if not self._buffer and not self._fill():
            return None
        out = bytes(self._buffer)
        del self._buffer[:]
        return out

    def finish(self) -> None:
        """Send a close frame, so the peer's reader ends rather than seeing EOF."""
        try:
            ws.send_close(self._conn, self._mask_out)
        except OSError:
            pass


class _LengthFraming:
    """One length-prefixed payload per message: ``transport.mode: frame``.

    The framing every example has used. The prefix is :mod:`cadet.core.service`'s,
    so the bytes on the wire are unchanged from before this module knew about
    modes at all.
    """

    def __init__(self, channel):
        self._channel = channel

    def send_frame(self, payload: bytes) -> None:
        send_msg(self._channel, payload)

    def recv_frame(self) -> Optional[bytes]:
        data = recv_msg(self._channel)
        return None if data is None else bytes(data)

    def finish(self) -> None:
        self._channel.finish()


class _StreamFraming:
    """No transport prefix; the records delimit themselves: ``mode: stream``.

    The sender writes a payload's bytes into the connection and nothing else, so
    the connection is one continuous stream of wire records. The reader buffers
    whatever arrives and asks the encoding how long the record at the front is
    (:func:`cadet.core.codec.record_length`), which is why this mode requires
    ``transport.encoding: safe`` - a pickle payload has no length in it and
    cannot be found in a stream.

    Why have it at all: a prefix written by the transport is a second length
    beside the one the record already carries, and a reader that has lost its
    place in a prefixed stream cannot recover, while one reading self-delimited
    records can resynchronise on the next record's magic.
    """

    def __init__(self, channel):
        self._channel = channel
        self._buffer = bytearray()

    def send_frame(self, payload: bytes) -> None:
        self._channel.sendall(payload)

    def recv_frame(self) -> Optional[bytes]:
        while True:
            size = record_length(self._buffer)
            if size is not None:
                out = bytes(self._buffer[:size])
                del self._buffer[:size]
                return out
            chunk = self._channel.recv_some()
            if chunk is None:
                # A partial record at EOF is a truncated stream, not a message:
                # the caller gets the same "peer closed" answer either way, and
                # the bytes are dropped rather than decoded as something else.
                return None
            self._buffer.extend(chunk)

    def finish(self) -> None:
        self._channel.finish()


def _build_wire(conn, protocol: str, mode: str, server_side: bool):
    """Return the framing a connection should use, handshaking if it must.

    One place decides the pair, so a transport, a server endpoint and a test all
    get the same wire for the same config.
    """
    if protocol not in PROTOCOLS:
        raise ValueError("unknown transport protocol {!r}; expected one of {}".format(
            protocol, PROTOCOLS
        ))
    if mode not in MODES:
        raise ValueError("unknown transport mode {!r}; expected one of {}".format(mode, MODES))
    if protocol == PROTOCOL_WEBSOCKET:
        # The upgrade happens before any payload, and only once per connection.
        if server_side:
            ws.server_handshake(conn)
        else:
            host, port = _peer_address(conn)
            ws.client_handshake(conn, host, port)
        channel = _WebSocketChannel(conn, server_side)
    else:
        channel = _SocketChannel(conn)
    return _StreamFraming(channel) if mode == MODE_STREAM else _LengthFraming(channel)


def _peer_address(conn):
    """The ``(host, port)`` a connected socket points at, for the Host header."""
    try:
        peer = conn.getpeername()
        return peer[0], peer[1]
    except (OSError, AttributeError, IndexError):
        return "localhost", 0


class _RecvDelayLine:
    """A proper delay line on the receive path.

    A reader thread pulls framed messages off the socket AS THEY ARRIVE and
    stamps each with ``release_time = arrival + sampled_delay (+ serialization)``,
    clamped to be non-decreasing so TCP ordering is preserved. Messages wait in a
    FIFO release queue; :meth:`recv` returns each one only at its release time.
    Reception never blocks on processing, so delays do not accumulate, and a slow
    consumer does not change the delay stamped on messages it has not yet read.
    """

    _EOF = object()

    def __init__(self, wire, model):
        self._wire = wire
        self._model = model
        self._queue = queue.Queue()
        self._prev_release = 0.0
        # Test instrumentation: per-delivered-message (arrival, release) stamps.
        self.stamps = []
        t = Thread(target=self._reader, daemon=True)
        t.start()

    def _reader(self) -> None:
        while True:
            try:
                data = self._wire.recv_frame()
            except OSError:
                data = None
            except Exception:
                # Anything else is a defect in the wire below, not a closed
                # peer. The consumer is told the stream ended all the same:
                # leaving the queue silent would hang whoever is reading it,
                # which turns a visible error into an unexplained stall.
                self._queue.put((0.0, self._EOF))
                raise
            if data is None:
                self._queue.put((0.0, self._EOF))
                return
            arrival = time.monotonic()
            if self._model.drops():
                continue  # lost in transit
            delay = self._model.sample_delay_s() + self._model.serialization_s(len(data))
            release = max(arrival + delay, self._prev_release)  # preserve ordering
            self._prev_release = release
            self.stamps.append((arrival, release))
            self._queue.put((release, bytes(data)))

    def recv(self) -> Optional[bytes]:
        release, payload = self._queue.get()
        if payload is self._EOF:
            return None
        now = time.monotonic()
        if release > now:
            time.sleep(release - now)
        return payload


class _SendDelayLine:
    """A delay line on the send (reply) path, for reverse-direction links.

    :meth:`send` stamps and enqueues the message; a sender thread transmits each
    one at its release time (sender side of replies). Ordering is preserved.
    """

    _STOP = object()

    def __init__(self, wire, model):
        self._wire = wire
        self._model = model
        self._queue = queue.Queue()
        self._prev_release = 0.0
        self._thread = Thread(target=self._sender, daemon=True)
        self._thread.start()

    def _sender(self) -> None:
        while True:
            release, payload = self._queue.get()
            if payload is self._STOP:
                return
            now = time.monotonic()
            if release > now:
                time.sleep(release - now)
            try:
                self._wire.send_frame(payload)
            except OSError:
                return

    def send(self, payload: bytes) -> None:
        if self._model.drops():
            return
        arrival = time.monotonic()
        delay = self._model.sample_delay_s() + self._model.serialization_s(len(payload))
        release = max(arrival + delay, self._prev_release)
        self._prev_release = release
        self._queue.put((release, payload))

    def stop(self) -> None:
        self._queue.put((0.0, self._STOP))


class FramedSocketTransport(Transport):
    """Real socket transport carrying byte payloads over the configured wire.

    The default wire is the one every example has used: plain TCP with the
    4-byte length framing in ``cadet.core.service``. ``protocol`` and ``mode``
    select the alternatives (``websocket``, and a self-delimited ``stream``); see
    this module's docstring. Optional ``recv_model`` (forward direction) and
    ``send_model`` (reverse direction) attach proper delay lines so network
    effects are emulated per-message without serializing on processing.

    ``server_side`` is the one thing the two ends cannot share: a websocket
    handshake has a requesting and an accepting half, and the RFC's masking rule
    is the other way round on each. It is False for a connecting transport and
    True for an accepted one, and is ignored over plain TCP.

    ``label`` names the link this connection is, for the byte counters; it
    defaults to the peer's address, which is the only identity a transport has
    on its own. A caller that knows the link or node name passes it, so a
    ``bandwidth_<node>.csv`` names links rather than ports.
    """

    def __init__(
        self,
        conn: socket.socket,
        recv_model: Optional[NetworkModel] = None,
        send_model: Optional[NetworkModel] = None,
        node=None,
        label: Optional[str] = None,
        protocol: str = PROTOCOL_TCP,
        mode: str = MODE_FRAME,
        server_side: bool = False,
    ):
        self._conn = conn
        self.counters = counters_for(label or _peer_label(conn))
        # Per-node socket tuning, before any traffic flows. Applied only when the
        # config asked for it; otherwise the kernel default stands. Before the
        # handshake too, so the upgrade itself travels under the same options.
        if node is not None:
            from cadet.network.tuning import apply_socket_options

            apply_socket_options(conn, node)
        self.protocol = protocol
        self.mode = mode
        self._wire = _build_wire(conn, protocol, mode, server_side)
        line = self._wire
        self._recv_line = _RecvDelayLine(line, recv_model) if _is_active(recv_model) else None
        self._send_line = _SendDelayLine(line, send_model) if _is_active(send_model) else None

    @classmethod
    def connect(
        cls,
        host: str,
        port: int,
        recv_model=None,
        send_model=None,
        retry_s: float = 0.0,
        node=None,
        label: Optional[str] = None,
        protocol: str = PROTOCOL_TCP,
        mode: str = MODE_FRAME,
    ) -> "FramedSocketTransport":
        """Open a client connection to ``host:port`` (retrying up to ``retry_s``)."""
        deadline = time.monotonic() + max(0.0, retry_s)
        while True:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.connect((host, port))
                return cls(
                    sock,
                    recv_model,
                    send_model,
                    node=node,
                    label=label,
                    protocol=protocol,
                    mode=mode,
                )
            except OSError:
                sock.close()
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.5)

    @classmethod
    def serve_one(
        cls,
        host: str,
        port: int,
        recv_model=None,
        send_model=None,
        backlog: int = 1,
        node=None,
        label: Optional[str] = None,
        protocol: str = PROTOCOL_TCP,
        mode: str = MODE_FRAME,
    ) -> "FramedSocketTransport":
        """Bind ``host:port``, accept one client, and return its transport."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(backlog)
        conn, _addr = srv.accept()
        srv.close()  # single-client server; stop listening once connected
        return cls(
            conn,
            recv_model,
            send_model,
            node=node,
            label=label,
            protocol=protocol,
            mode=mode,
            server_side=True,
        )

    def send(self, payload: bytes) -> None:
        # Counted as offered to the link, before the delay line may drop it: on
        # the message-level backend a configured loss discards a message the
        # application did send, and the two numbers are different measurements.
        self.counters.add_sent(len(payload))
        if self._send_line is not None:
            self._send_line.send(payload)
        else:
            self._wire.send_frame(payload)

    def recv(self) -> Optional[bytes]:
        if self._recv_line is not None:
            data = self._recv_line.recv()
        else:
            data = self._wire.recv_frame()
        if data is not None:
            self.counters.add_recv(len(data))
        return data

    def close(self) -> None:
        """Close the underlying socket, saying goodbye on the wire first."""
        if self._send_line is not None:
            self._send_line.stop()
        try:
            self._wire.finish()
        except OSError:
            pass
        try:
            self._conn.close()
        except OSError:
            pass


def _is_active(model: Optional[NetworkModel]) -> bool:
    """A model is active if present and not the pass-through NullNetworkModel."""
    return model is not None and not isinstance(model, NullNetworkModel)


def _peer_label(conn: socket.socket) -> str:
    """A connection's fallback counter label: the peer it is connected to."""
    try:
        peer = conn.getpeername()
    except (OSError, AttributeError):
        return "unknown"
    if isinstance(peer, tuple) and len(peer) >= 2:
        return "{}:{}".format(peer[0], peer[1])
    return str(peer)


class SourceMessage:
    """One inbound payload and the connection it arrived on.

    ``source_id`` is what makes a merged queue usable: once several clients feed
    one queue, a payload without its origin cannot be replied to, attributed to
    a stream, or counted per connection.
    """

    __slots__ = ("source_id", "payload", "received_at")

    def __init__(self, source_id: str, payload: bytes, received_at: float):
        self.source_id = source_id
        self.payload = payload
        self.received_at = received_at

    def __repr__(self) -> str:
        return "SourceMessage({!r}, {} bytes)".format(self.source_id, len(self.payload))


class MultiConnectionEndpoint:
    """A server endpoint that serves several clients through one queue.

    Each accepted connection gets its own :class:`FramedSocketTransport`, and so
    its own reader thread and its own delay line; a drain thread per connection
    moves released messages into the inbound queue. The delay a message waited is
    therefore the delay configured for *its* connection, which is what keeps
    per-connection latency statistics meaningful.

    ``policy`` decides the order messages leave the endpoint:

    ``fifo``
        One shared queue. A message is returned as soon as its own delay line
        releases it, so the order across connections is release order - the
        order the messages actually became available.
    ``round_robin``
        A queue per connection, served in rotation, at most one message per
        connection per turn. A client sending fast cannot starve a slow one,
        which is what makes a load experiment's scheduling the experiment's
        rather than an accident of arrival order.

    A connection that closes is removed from service and the others continue;
    :meth:`recv` returns ``None`` only once every connection has closed, so one
    client dying does not end the run for the rest.
    """

    POLICY_FIFO = "fifo"
    POLICY_ROUND_ROBIN = "round_robin"
    POLICIES = (POLICY_FIFO, POLICY_ROUND_ROBIN)

    # How long recv() waits on an empty queue before looking again. It exists so
    # recv() can notice a connection closing rather than blocking on a queue
    # nothing will ever post to.
    _POLL_S = 0.05

    def __init__(self, policy: str = POLICY_FIFO):
        if policy not in self.POLICIES:
            raise ValueError(
                "unknown policy {!r}; expected one of {}".format(policy, self.POLICIES)
            )
        self.policy = policy
        self._server: Optional[socket.socket] = None
        self._transports: Dict[str, FramedSocketTransport] = {}
        self._queues: Dict[str, "queue.Queue"] = {}
        self._shared: "queue.Queue" = queue.Queue()
        self._open: Dict[str, bool] = {}
        self._order: List[str] = []
        self._turn = 0
        self._lock = Lock()

    # -- construction --------------------------------------------------------
    @classmethod
    def serve(
        cls,
        host: str,
        port: int,
        expect: int = 1,
        recv_model=None,
        send_model=None,
        policy: str = POLICY_FIFO,
        accept_timeout_s: float = 60.0,
        node=None,
        model_for=None,
        protocol: str = PROTOCOL_TCP,
        mode: str = MODE_FRAME,
    ) -> "MultiConnectionEndpoint":
        """Bind ``host:port`` and accept ``expect`` clients, then serve them.

        ``recv_model``/``send_model`` apply to every connection. ``model_for``
        overrides that per connection: given the source id it returns
        ``(recv_model, send_model)``, which is what a config with a different
        link per stream needs - an ego and an infrastructure feed into one
        roadside node are not the same link.

        ``protocol``/``mode`` are the wire every accepted connection uses; they
        are one endpoint-wide choice because they come from one ``transport``
        section, and a peer cannot negotiate its own.

        All ``expect`` clients are accepted before serving begins, which matches
        what the roles need (they are started together by the orchestrator) and
        makes the readiness barrier meaningful. A client that connects later is
        not accepted; that is a deliberate limitation, not an oversight.
        """
        endpoint = cls(policy=policy)
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(max(1, expect))
        endpoint._server = server
        deadline = time.monotonic() + max(0.0, accept_timeout_s)
        try:
            for index in range(max(1, expect)):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "accepted {} of {} expected connections on {}:{} before the "
                        "{:.0f}s timeout".format(
                            index, expect, host, port, accept_timeout_s
                        )
                    )
                server.settimeout(remaining)
                conn, addr = server.accept()
                source_id = "{}:{}".format(addr[0], addr[1]) if addr else "conn{}".format(index)
                recv, send = (
                    model_for(source_id) if model_for is not None else (recv_model, send_model)
                )
                endpoint._add(
                    source_id,
                    FramedSocketTransport(
                        conn,
                        recv,
                        send,
                        node=node,
                        label=source_id,
                        protocol=protocol,
                        mode=mode,
                        server_side=True,
                    ),
                )
        finally:
            # Stop listening once the expected clients are in: the endpoint
            # serves a fixed set, so a lingering listener would only hold the
            # port and accept something nothing will read.
            server.settimeout(None)
            server.close()
            endpoint._server = None
        return endpoint

    def _add(self, source_id: str, transport: "FramedSocketTransport") -> None:
        """Register a connection and start draining it."""
        with self._lock:
            self._transports[source_id] = transport
            self._queues[source_id] = queue.Queue()
            self._open[source_id] = True
            self._order.append(source_id)
        thread = Thread(target=self._drain, args=(source_id, transport), daemon=True)
        thread.start()

    def _drain(self, source_id: str, transport: "FramedSocketTransport") -> None:
        """Move released messages from one connection onto the inbound queue(s).

        The transport's own ``recv`` blocks until its delay line releases a
        message, so the wait happens here, per connection, and not in whatever
        thread is consuming the merged queue.
        """
        while True:
            try:
                payload = transport.recv()
            except OSError:
                payload = None
            if payload is None:
                with self._lock:
                    self._open[source_id] = False
                return
            message = SourceMessage(source_id, payload, time.monotonic())
            if self.policy == self.POLICY_FIFO:
                self._shared.put(message)
            else:
                self._queues[source_id].put(message)

    # -- use -----------------------------------------------------------------
    def recv(self, timeout_s: Optional[float] = None) -> Optional[SourceMessage]:
        """Return the next message, or ``None`` when every connection has closed.

        ``timeout_s`` bounds the wait; ``None`` waits until a message arrives or
        all connections close.
        """
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            message = self._next()
            if message is not None:
                return message
            if not self.active_sources():
                return None
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(self._POLL_S)

    def _next(self) -> Optional[SourceMessage]:
        """Take one message if one is ready, honouring the policy."""
        if self.policy == self.POLICY_FIFO:
            try:
                return self._shared.get_nowait()
            except queue.Empty:
                return None
        with self._lock:
            order = list(self._order)
        if not order:
            return None
        # Rotate the starting point so no connection is consistently first.
        for offset in range(len(order)):
            source_id = order[(self._turn + offset) % len(order)]
            try:
                message = self._queues[source_id].get_nowait()
            except (queue.Empty, KeyError):
                continue
            self._turn = (self._turn + offset + 1) % len(order)
            return message
        return None

    def send(self, source_id: str, payload: bytes) -> None:
        """Reply to one connection, by the id its message carried."""
        transport = self._transports.get(source_id)
        if transport is None:
            raise KeyError("no connection {!r} on this endpoint".format(source_id))
        transport.send(payload)

    def broadcast(self, payload: bytes) -> None:
        """Send to every connection still open."""
        for source_id in self.active_sources():
            try:
                self.send(source_id, payload)
            except (OSError, KeyError):
                continue

    # -- introspection -------------------------------------------------------
    def sources(self) -> List[str]:
        """Every connection ever accepted, in accept order."""
        with self._lock:
            return list(self._order)

    def active_sources(self) -> List[str]:
        """Connections still open, in accept order."""
        with self._lock:
            return [s for s in self._order if self._open.get(s)]

    def transport_for(self, source_id: str) -> Optional["FramedSocketTransport"]:
        """Return one connection's transport, for per-connection measurement."""
        return self._transports.get(source_id)

    def delay_stamps(self, source_id: str) -> List[tuple]:
        """Return one connection's ``(arrival, release)`` stamps.

        Per connection on purpose: this is how a test or a metric checks that
        the delay a stream saw is the delay its own link configured, rather than
        an average over whatever else shared the queue.
        """
        transport = self._transports.get(source_id)
        line = getattr(transport, "_recv_line", None) if transport else None
        return list(getattr(line, "stamps", []) or [])

    def close(self) -> None:
        """Close every connection and the listener, if one is still open."""
        for transport in list(self._transports.values()):
            try:
                transport.close()
            except OSError:
                pass
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None
