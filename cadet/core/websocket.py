"""Minimal RFC 6455 WebSocket framing for the transport's ``websocket`` protocol.

What this is for:
    ``transport.protocol: websocket`` carries the same payloads the TCP protocol
    does, but inside WebSocket binary frames after one HTTP upgrade, so a CADET
    link can be put through infrastructure that forwards HTTP and WebSocket and
    nothing else (a reverse proxy, a gateway, a browser-facing endpoint). Only
    what a point-to-point payload carrier needs is implemented here.

What it contains:
    - :func:`client_handshake` / :func:`server_handshake`: the HTTP upgrade, both
      ends, including the ``Sec-WebSocket-Accept`` digest that proves the peer
      understood the request.
    - :func:`send_binary`: one payload as one binary frame (masked from the
      client, unmasked from the server, as the RFC requires).
    - :func:`recv_payload`: the next application payload, reassembling
      continuation frames and answering ping with pong; ``None`` once the peer
      sends close or the socket ends.
    - :class:`WebSocketError`: a protocol violation or a refused handshake.

    Deliberately not implemented, because nothing in a CADET run uses it: the
    permessage-deflate extension (payloads are already compressed by the wire
    encoding), subprotocol negotiation, and fragmenting an outgoing payload
    (one payload is one frame).

How it fits into CADET:
    Used only by :mod:`cadet.core.transport`, which wraps a socket in these
    functions when the config selects the protocol. The framing above it is
    unchanged: a length prefix or a codec-delimited stream still runs inside the
    binary frames, so the delay lines, the counters and the apps see exactly what
    they see over plain TCP.

How to configure it:
    Nothing here is configurable. ``transport.protocol`` selects it; the HTTP
    path is fixed (:data:`WS_PATH`) because both ends are CADET roles and the
    path carries no information.
"""

import base64
import os
import struct
from hashlib import sha1

# The GUID RFC 6455 fixes for the accept digest. Not a secret and not tunable:
# both ends must use this exact value or the handshake cannot be checked.
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# The request path a CADET role asks for. Fixed: both ends are roles of the
# same run, so the path distinguishes nothing.
WS_PATH = "/cadet"

# Opcodes, from the RFC.
OP_CONTINUATION = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA

# Header limits, from the RFC: a 7-bit length up to 125, then a 16-bit length,
# then a 64-bit one.
_LEN_16 = 126
_LEN_64 = 127
_MAX_7BIT = 125
_MAX_16BIT = 0xFFFF

# Largest HTTP handshake this will read. A peer that sends more than this before
# the blank line is not a CADET role.
_MAX_HANDSHAKE_BYTES = 8192


class WebSocketError(Exception):
    """A refused handshake or a frame this implementation will not accept."""


def _accept_digest(key: str) -> str:
    """The ``Sec-WebSocket-Accept`` value for a client's ``Sec-WebSocket-Key``."""
    return base64.b64encode(sha1((key + WS_GUID).encode("ascii")).digest()).decode("ascii")


def _read_http_head(sock) -> str:
    """Read up to the blank line that ends an HTTP message head."""
    buf = bytearray()
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(1)
        if not chunk:
            raise WebSocketError("peer closed during the websocket handshake")
        buf.extend(chunk)
        if len(buf) > _MAX_HANDSHAKE_BYTES:
            raise WebSocketError("websocket handshake head exceeded {} bytes".format(
                _MAX_HANDSHAKE_BYTES
            ))
    return buf.decode("latin-1")


def client_handshake(sock, host: str, port: int) -> None:
    """Perform the client side of the upgrade on an already-connected socket."""
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    request = (
        "GET {path} HTTP/1.1\r\n"
        "Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    ).format(path=WS_PATH, host=host, port=port, key=key)
    sock.sendall(request.encode("latin-1"))
    head = _read_http_head(sock)
    status = head.split("\r\n", 1)[0]
    if "101" not in status:
        raise WebSocketError("websocket upgrade refused: {!r}".format(status))
    headers = _headers(head)
    expected = _accept_digest(key)
    if headers.get("sec-websocket-accept") != expected:
        # The digest is the only thing that proves the peer read the request
        # rather than echoing a 101 at anything that connects.
        raise WebSocketError("websocket accept digest mismatch")


def server_handshake(sock) -> None:
    """Perform the server side of the upgrade on an accepted socket."""
    head = _read_http_head(sock)
    headers = _headers(head)
    key = headers.get("sec-websocket-key")
    if headers.get("upgrade", "").lower() != "websocket" or not key:
        sock.sendall(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n")
        raise WebSocketError("not a websocket upgrade request")
    response = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Accept: {accept}\r\n"
        "\r\n"
    ).format(accept=_accept_digest(key))
    sock.sendall(response.encode("latin-1"))


def _headers(head: str):
    """Return the lower-cased header map of an HTTP message head."""
    out = {}
    for line in head.split("\r\n")[1:]:
        if not line:
            break
        name, _, value = line.partition(":")
        out[name.strip().lower()] = value.strip()
    return out


def send_binary(sock, payload: bytes, mask: bool) -> None:
    """Send ``payload`` as one binary frame.

    ``mask`` is True on the client side and False on the server side; the RFC
    requires exactly that, and a peer that sees it the other way round must
    close the connection.
    """
    size = len(payload)
    header = bytearray()
    header.append(0x80 | OP_BINARY)  # FIN set: one payload is one frame
    flag = 0x80 if mask else 0x00
    if size <= _MAX_7BIT:
        header.append(flag | size)
    elif size <= _MAX_16BIT:
        header.append(flag | _LEN_16)
        header.extend(struct.pack(">H", size))
    else:
        header.append(flag | _LEN_64)
        header.extend(struct.pack(">Q", size))
    if not mask:
        sock.sendall(bytes(header) + payload)
        return
    key = os.urandom(4)
    header.extend(key)
    sock.sendall(bytes(header) + _mask(payload, key))


def _mask(payload: bytes, key: bytes) -> bytes:
    """XOR ``payload`` with the 4-byte masking key (the RFC's transformation).

    Done as one big-integer XOR rather than a loop over the bytes. A camera
    frame is over a megabyte, and a Python-level loop over it costs tens of
    milliseconds per frame in each direction - enough to show up in a run's
    active phase, which is how it was found. The integer XOR is one C-level
    operation on the whole payload.
    """
    size = len(payload)
    if not size:
        return b""
    repeats = -(-size // len(key))  # ceiling division: enough key to cover it
    stream = (key * repeats)[:size]
    return (
        int.from_bytes(payload, "big") ^ int.from_bytes(stream, "big")
    ).to_bytes(size, "big")


def _recv_exact(sock, n: int):
    """Read exactly ``n`` bytes, or ``None`` if the peer closes first."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_payload(sock, mask_out: bool):
    """Return the next application payload, or ``None`` at close.

    Continuation frames are reassembled, a ping is answered with a pong (so a
    peer's keepalive does not look like data), and a close frame ends the
    stream. ``mask_out`` says whether a pong this side sends must be masked,
    which is the same rule as for data.
    """
    parts = bytearray()
    while True:
        head = _recv_exact(sock, 2)
        if head is None:
            return None
        fin = bool(head[0] & 0x80)
        opcode = head[0] & 0x0F
        masked = bool(head[1] & 0x80)
        size = head[1] & 0x7F
        if size == _LEN_16:
            extended = _recv_exact(sock, 2)
            if extended is None:
                return None
            (size,) = struct.unpack(">H", extended)
        elif size == _LEN_64:
            extended = _recv_exact(sock, 8)
            if extended is None:
                return None
            (size,) = struct.unpack(">Q", extended)
        key = None
        if masked:
            key = _recv_exact(sock, 4)
            if key is None:
                return None
        data = _recv_exact(sock, size) if size else b""
        if data is None:
            return None
        if key is not None:
            data = _mask(data, key)
        if opcode == OP_CLOSE:
            return None
        if opcode == OP_PING:
            _send_control(sock, OP_PONG, data, mask_out)
            continue
        if opcode == OP_PONG:
            continue
        if opcode == OP_TEXT:
            raise WebSocketError("text frame on a binary payload connection")
        if opcode not in (OP_BINARY, OP_CONTINUATION):
            raise WebSocketError("unexpected websocket opcode {}".format(opcode))
        if fin and not parts:
            # The common case: one payload is one frame, so it is returned as it
            # came rather than copied through the reassembly buffer twice.
            return data
        parts.extend(data)
        if fin:
            return bytes(parts)


def _send_control(sock, opcode: int, payload: bytes, mask: bool) -> None:
    """Send a control frame (payload is at most 125 bytes, per the RFC)."""
    payload = payload[:_MAX_7BIT]
    header = bytearray([0x80 | opcode])
    if not mask:
        header.append(len(payload))
        sock.sendall(bytes(header) + payload)
        return
    header.append(0x80 | len(payload))
    key = os.urandom(4)
    header.extend(key)
    sock.sendall(bytes(header) + _mask(payload, key))


def send_close(sock, mask: bool) -> None:
    """Send a close frame, so the peer's reader ends rather than seeing EOF."""
    # 1000 is the RFC's "normal closure"; a CADET role closing is never an error.
    _send_control(sock, OP_CLOSE, struct.pack(">H", 1000), mask)
