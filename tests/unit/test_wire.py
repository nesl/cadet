"""Tests for the wire: the safe encoding, the pickle opt-in, and the two carriers.

What this covers:
    - the safe record round-trips every shape the apps put on it (dicts, tuples,
      bytes, numpy arrays with dtype and shape, the pipeline message family),
    - a payload that is not a safe record is refused unless ``transport.encoding:
      pickle`` was asked for, so a tree that never opts in cannot be made to
      unpickle anything off the wire,
    - the transport defaults are the originals' wire (tcp, frame) and the
      encoding default is pinned, so neither drifts unnoticed,
    - every ``protocol`` x ``mode`` pair carries payloads whole, through a delay
      line and through the fan-in endpoint,
    - ``mode: stream`` with ``encoding: pickle`` is refused by the schema, since
      a pickle payload cannot be delimited in a byte stream.

Where it fits:
    Unit tests over :mod:`cadet.core.codec`, :mod:`cadet.core.websocket` and
    :mod:`cadet.core.transport`, on loopback sockets only - nothing here needs a
    simulator or a configured link.
"""

import socket
import struct
import threading
import time
import zlib

import numpy as np
import pytest

from cadet.config.schema import TransportConfig
from cadet.core import codec
from cadet.core import websocket as ws
from cadet.core.codec import (
    MAGIC,
    PICKLE,
    SAFE,
    WireEncodingError,
    decode,
    decode_safe,
    encode,
    encode_pickle,
    encode_safe,
    record_length,
    registered_wire_types,
    set_wire_encoding,
    wire_encoding,
)
from cadet.core.network import NullNetworkModel, SampledNetworkModel, _Constant
from cadet.core.transport import (
    MODE_FRAME,
    MODE_STREAM,
    PROTOCOL_TCP,
    PROTOCOL_WEBSOCKET,
    FramedSocketTransport,
    MultiConnectionEndpoint,
)
from cadet.pipeline.messages import ControlCommand, Frame
from cadet.pipeline.stages.braking import BrakeDecision
from cadet.pipeline.stages.v2v import V2VLeadRange


@pytest.fixture(autouse=True)
def _restore_encoding():
    """The selection is per process, so a test that changes it puts it back."""
    before = wire_encoding()
    yield
    set_wire_encoding(before)


def _port():
    """A free loopback port, taken the way the other transport tests take one."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# --------------------------------------------------------------------------
# the safe record
# --------------------------------------------------------------------------
def test_the_safe_record_round_trips_the_payloads_the_apps_send():
    image = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    msg = {
        "frame_id": 7,
        "timestamp": 1234.5,
        "rgb_frame": image,
        "frame_type": "I",
        "ego_pose": (1.5, -2.0, 0.25),
        "image_size": (64, 48),
        "detections": [[2, 0.91, 311.5], [0, 0.42, 480.0]],
        "jpeg": b"\x00\xff binary",
        "nothing": None,
        "flag": True,
    }
    out = decode_safe(encode_safe(msg))
    assert out["frame_id"] == 7 and out["frame_type"] == "I" and out["flag"] is True
    assert out["nothing"] is None
    # A tuple stays a tuple: the apps index these and a list would be a silent
    # type change on the wire.
    assert out["ego_pose"] == (1.5, -2.0, 0.25)
    assert out["image_size"] == (64, 48)
    assert out["detections"] == msg["detections"]
    assert out["jpeg"] == msg["jpeg"]
    assert out["rgb_frame"].dtype == image.dtype
    assert out["rgb_frame"].shape == image.shape
    assert np.array_equal(out["rgb_frame"], image)
    # A model backend may write into the frame it is given, which a view over
    # the record's own bytes would refuse.
    assert out["rgb_frame"].flags.writeable


def test_arrays_keep_their_dtype_and_non_contiguous_layout():
    for dtype in (np.uint8, np.int32, np.float32, np.float64):
        arr = (np.arange(24).reshape(2, 3, 4) % 7).astype(dtype)
        out = decode_safe(encode_safe({"a": arr}))["a"]
        assert out.dtype == arr.dtype and np.array_equal(out, arr)
    view = np.arange(24, dtype=np.uint8).reshape(4, 6)[:, ::2]
    assert not view.flags.c_contiguous
    assert np.array_equal(decode_safe(encode_safe({"a": view}))["a"], view)


def test_the_pipeline_message_family_round_trips():
    frame = Frame(
        image=np.zeros((4, 4, 3), np.uint8),
        meta={"capture_time": 1.0, "capture_loc": (1.0, 2.0, 3.0)},
        frame_id=3,
        sim_time=1.0,
        wall_time=2.0,
    )
    frame.stamp("camera", "enter")
    frame.stamp("camera", "exit")
    out = decode_safe(encode_safe(frame))
    assert type(out) is Frame
    assert out.frame_id == 3 and out.sim_time == 1.0 and out.wall_time == 2.0
    assert out.meta["capture_loc"] == (1.0, 2.0, 3.0)
    assert [(s.stage, s.phase) for s in out.stamps] == [("camera", "enter"), ("camera", "exit")]
    assert "camera" in out.stage_latencies_ms()
    # None is "no output this frame" at a pipeline boundary.
    assert decode_safe(encode_safe(None)) is None
    assert decode_safe(encode_safe(ControlCommand(brake=1.0, frame_id=4))).brake == 1.0
    assert decode_safe(encode_safe(BrakeDecision(brake=True, frame_id=5))).frame_id == 5
    assert decode_safe(encode_safe(V2VLeadRange(best_z=12.5))).best_z == 12.5


def test_every_boundary_message_type_has_a_wire_name():
    # A pipeline message that is not registered cannot cross a boundary at all,
    # so the list is pinned rather than left to be discovered by a live run.
    # It is the set registered once the modules that own the types are imported,
    # which this module's imports above are.
    assert set(registered_wire_types()) == {
        "BrakeControlCommand",
        "BrakeDecision",
        "ControlCommand",
        "ControlInput",
        "Detections",
        "Frame",
        "Message",
        "Plan",
        "Predictions",
        "StageStamp",
        "Tracks",
        "V2VLeadRange",
    }


def test_an_object_with_no_safe_representation_is_named_not_silently_dropped():
    class Handle:
        """Stands in for a simulator handle (a carla.Location and its kind)."""

    with pytest.raises(WireEncodingError) as exc:
        encode_safe({"loc": Handle()})
    assert "Handle" in str(exc.value)
    assert "transport.encoding: pickle" in str(exc.value)


def test_an_unregistered_class_cannot_be_rebuilt_from_a_payload():
    payload = encode_safe(Frame(frame_id=1))
    body = zlib.decompress(payload[codec.RECORD_PREFIX_BYTES :])
    tampered = body.replace(b'"Frame"', b'"Other"')
    assert tampered != body
    packed = zlib.compress(tampered)
    record = MAGIC + struct.pack(">I", len(packed)) + packed
    with pytest.raises(WireEncodingError) as exc:
        decode_safe(record)
    assert "not registered" in str(exc.value)


def test_a_corrupt_or_truncated_record_is_an_error_not_a_decode():
    payload = encode_safe({"a": 1})
    with pytest.raises(WireEncodingError):
        decode_safe(payload[:-4])
    with pytest.raises(WireEncodingError):
        decode_safe(b"NOPE" + payload[4:])


# --------------------------------------------------------------------------
# the pickle opt-in
# --------------------------------------------------------------------------
def test_a_pickle_payload_is_refused_unless_pickle_was_asked_for():
    set_wire_encoding(SAFE)
    with pytest.raises(WireEncodingError) as exc:
        decode(encode_pickle({"anything": 1}))
    # The message has to name the setting, because refusing a peer's payload is
    # otherwise indistinguishable from a broken link.
    assert "transport.encoding" in str(exc.value)


def test_the_pickle_opt_in_accepts_both_formats():
    set_wire_encoding(PICKLE)
    assert decode(encode_pickle({"a": 1}))["a"] == 1
    # A reader that opted into pickle still reads a safe record: the magic says
    # which format arrived, so a mixed pair of ends is not a silent corruption.
    assert decode(encode_safe({"a": 2}))["a"] == 2


def test_a_safe_reader_still_reads_a_safe_record():
    set_wire_encoding(SAFE)
    assert decode(encode({"a": 3}))["a"] == 3


def test_the_selection_decides_what_encode_produces():
    set_wire_encoding(SAFE)
    assert encode({"a": 1})[:4] == MAGIC
    set_wire_encoding(PICKLE)
    assert encode({"a": 1})[:4] != MAGIC


def test_the_encoding_comes_from_the_config():
    assert codec.set_wire_encoding_from_config(_Cfg("pickle")) == PICKLE
    assert wire_encoding() == PICKLE
    assert codec.set_wire_encoding_from_config(_Cfg("safe")) == SAFE


class _Cfg:
    def __init__(self, encoding):
        self.transport = TransportConfig(encoding=encoding)


# --------------------------------------------------------------------------
# the defaults
# --------------------------------------------------------------------------
def test_the_transport_defaults_are_pinned():
    transport = TransportConfig()
    # The original's carrier and framing: a plain socket, one length-prefixed
    # message per payload.
    assert transport.protocol == "tcp"
    assert transport.mode == "frame"
    # The encoding default is NOT the original's; it is the safe record, which
    # was measured neutral for payload size and round-trip timing. A drift back
    # would silently reopen the unpickling of socket data.
    assert transport.encoding == "safe"
    assert codec.DEFAULT_ENCODING == SAFE


def test_stream_mode_refuses_the_pickle_encoding():
    # Refused in the schema rather than at the first byte, where it would look
    # like a corrupt stream.
    with pytest.raises(ValueError) as exc:
        TransportConfig(mode="stream", encoding="pickle")
    assert "stream" in str(exc.value)
    TransportConfig(mode="stream", encoding="safe")


# --------------------------------------------------------------------------
# record_length: what makes stream mode possible
# --------------------------------------------------------------------------
def test_record_length_reports_a_whole_record_and_waits_for_a_partial_one():
    payload = encode_safe({"a": "x" * 1000})
    assert record_length(payload) == len(payload)
    assert record_length(payload + b"more") == len(payload)
    for cut in (0, 1, 4, 7, len(payload) - 1):
        assert record_length(payload[:cut]) is None
    with pytest.raises(WireEncodingError):
        record_length(b"PICK")


# --------------------------------------------------------------------------
# the carriers and the framings
# --------------------------------------------------------------------------
def _echo(port, protocol, mode, count, box, recv_model=None):
    """Serve one client: decode, answer, and record what arrived."""
    transport = FramedSocketTransport.serve_one(
        "127.0.0.1", port, protocol=protocol, mode=mode, recv_model=recv_model
    )
    box["server"] = transport
    seen = []
    while len(seen) < count:
        payload = transport.recv()
        if payload is None:
            break
        message = decode_safe(payload)
        seen.append(message["i"])
        transport.send(encode_safe({"i": message["i"], "sum": int(message["a"].sum())}))
    box["seen"] = seen


@pytest.mark.parametrize("protocol", [PROTOCOL_TCP, PROTOCOL_WEBSOCKET])
@pytest.mark.parametrize("mode", [MODE_FRAME, MODE_STREAM])
def test_every_protocol_and_mode_pair_carries_payloads_whole(protocol, mode):
    port, box, count = _port(), {}, 6
    thread = threading.Thread(
        target=_echo, args=(port, protocol, mode, count, box), daemon=True
    )
    thread.start()
    client = FramedSocketTransport.connect(
        "127.0.0.1", port, protocol=protocol, mode=mode, retry_s=5.0
    )
    # Big enough that a payload spans more than one read, which is where a
    # framing that does not reassemble would fail.
    array = np.arange(200 * 200, dtype=np.uint8).reshape(200, 200) % 251
    for i in range(count):
        client.send(encode_safe({"i": i, "a": array}))
        reply = decode_safe(client.recv())
        assert reply == {"i": i, "sum": int(array.sum())}
    thread.join(timeout=10)
    assert box["seen"] == list(range(count))
    client.close()
    box["server"].close()


@pytest.mark.parametrize("protocol", [PROTOCOL_TCP, PROTOCOL_WEBSOCKET])
@pytest.mark.parametrize("mode", [MODE_FRAME, MODE_STREAM])
def test_a_delay_line_still_delays_its_own_connection_on_every_wire(protocol, mode):
    delay_ms = 40.0
    port, box, count = _port(), {}, 3
    thread = threading.Thread(
        target=_echo,
        args=(port, protocol, mode, count, box),
        kwargs={"recv_model": SampledNetworkModel(_Constant(delay_ms))},
        daemon=True,
    )
    thread.start()
    client = FramedSocketTransport.connect(
        "127.0.0.1", port, protocol=protocol, mode=mode, retry_s=5.0
    )
    array = np.zeros((8, 8), dtype=np.uint8)
    for i in range(count):
        start = time.monotonic()
        client.send(encode_safe({"i": i, "a": array}))
        decode_safe(client.recv())
        # The configured one-way delay, not a multiple of it: the carrier must
        # not add a round trip of its own.
        elapsed_ms = (time.monotonic() - start) * 1000.0
        assert delay_ms <= elapsed_ms < delay_ms * 2
    thread.join(timeout=10)
    client.close()
    box["server"].close()


def test_the_counters_count_payload_bytes_whatever_the_wire_adds():
    """Framing overhead stays out of the byte counters, on both carriers."""
    sizes = {}
    for protocol in (PROTOCOL_TCP, PROTOCOL_WEBSOCKET):
        port, box = _port(), {}
        thread = threading.Thread(
            target=_echo, args=(port, protocol, MODE_FRAME, 2, box), daemon=True
        )
        thread.start()
        client = FramedSocketTransport.connect(
            "127.0.0.1", port, protocol=protocol, mode=MODE_FRAME, retry_s=5.0
        )
        array = np.zeros((16, 16), dtype=np.uint8)
        for i in range(2):
            client.send(encode_safe({"i": i, "a": array}))
            client.recv()
        thread.join(timeout=10)
        sizes[protocol] = (client.counters.bytes_sent, client.counters.bytes_recv)
        client.close()
        box["server"].close()
    assert sizes[PROTOCOL_TCP] == sizes[PROTOCOL_WEBSOCKET]


def test_fan_in_works_on_the_websocket_carrier():
    """The endpoint accepts its own connections, so it handshakes its own half."""
    port, box = _port(), {}

    def serve():
        endpoint = MultiConnectionEndpoint.serve(
            "127.0.0.1",
            port,
            expect=2,
            protocol=PROTOCOL_WEBSOCKET,
            mode=MODE_STREAM,
            recv_model=NullNetworkModel(),
        )
        box["endpoint"] = endpoint
        seen = []
        while len(seen) < 2:
            message = endpoint.recv(timeout_s=10.0)
            if message is None:
                break
            seen.append((message.source_id, decode_safe(message.payload)["who"]))
            endpoint.send(message.source_id, encode_safe({"ack": True}))
        box["seen"] = seen

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    clients = [
        FramedSocketTransport.connect(
            "127.0.0.1", port, protocol=PROTOCOL_WEBSOCKET, mode=MODE_STREAM, retry_s=5.0
        )
        for _ in range(2)
    ]
    for i, client in enumerate(clients):
        client.send(encode_safe({"who": i}))
    for client in clients:
        assert decode_safe(client.recv())["ack"] is True
    thread.join(timeout=10)
    assert sorted(who for _source, who in box["seen"]) == [0, 1]
    # Two connections, two sources: the endpoint tells them apart on either
    # carrier, which is what a reply address depends on.
    assert len({source for source, _who in box["seen"]}) == 2
    for client in clients:
        client.close()
    box["endpoint"].close()


def test_an_unknown_protocol_or_mode_is_refused_before_any_traffic():
    left, right = socket.socketpair()
    with pytest.raises(ValueError):
        FramedSocketTransport(left, protocol="smoke-signal")
    with pytest.raises(ValueError):
        FramedSocketTransport(left, mode="telepathy")
    left.close()
    right.close()


# --------------------------------------------------------------------------
# the websocket handshake itself
# --------------------------------------------------------------------------
def test_the_accept_digest_is_checked_so_any_101_will_not_do():
    """The digest is the only proof the peer read the request."""
    client_sock, server_sock = socket.socketpair()

    def wrong_server():
        # Read the request head, then answer 101 with a digest of nothing.
        buf = bytearray()
        while b"\r\n\r\n" not in buf:
            chunk = server_sock.recv(1)
            if not chunk:
                return
            buf.extend(chunk)
        server_sock.sendall(
            b"HTTP/1.1 101 Switching Protocols\r\n"
            b"Upgrade: websocket\r\n"
            b"Sec-WebSocket-Accept: not-the-digest\r\n\r\n"
        )

    thread = threading.Thread(target=wrong_server, daemon=True)
    thread.start()
    with pytest.raises(ws.WebSocketError):
        ws.client_handshake(client_sock, "localhost", 1)
    thread.join(timeout=5)
    client_sock.close()
    server_sock.close()


def test_a_request_that_is_not_an_upgrade_is_refused():
    client_sock, server_sock = socket.socketpair()
    client_sock.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
    with pytest.raises(ws.WebSocketError):
        ws.server_handshake(server_sock)
    client_sock.close()
    server_sock.close()


def test_a_ping_is_answered_and_does_not_look_like_data():
    """A keepalive must not be delivered to the application as a payload."""
    client_sock, server_sock = socket.socketpair()
    # Client-side frames are masked; this stands in for a peer's keepalive.
    ws._send_control(client_sock, ws.OP_PING, b"are you there", True)
    ws.send_binary(client_sock, b"the payload", True)
    assert ws.recv_payload(server_sock, False) == b"the payload"
    # And the keepalive was answered: a pong, unmasked, from the server side.
    client_sock.settimeout(5.0)
    head = client_sock.recv(2)
    assert head[0] & 0x0F == ws.OP_PONG
    assert not head[1] & 0x80  # a server does not mask
    client_sock.close()
    server_sock.close()


def test_a_close_frame_ends_the_stream_rather_than_raising():
    client_sock, server_sock = socket.socketpair()
    ws.send_close(client_sock, True)
    assert ws.recv_payload(server_sock, False) is None
    client_sock.close()
    server_sock.close()


def test_a_payload_over_the_extended_length_boundaries_round_trips():
    # 125/126 and 65535/65536 are where the RFC changes the header width.
    for size in (0, 125, 126, 65535, 65536):
        client_sock, server_sock = socket.socketpair()
        payload = bytes(size)
        threading.Thread(
            target=ws.send_binary, args=(client_sock, payload, True), daemon=True
        ).start()
        assert ws.recv_payload(server_sock, False) == payload
        client_sock.close()
        server_sock.close()
