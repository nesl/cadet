"""Length-framed message framing over a byte stream (TCP socket).

TCP is a byte stream with no message boundaries. These helpers put a 4-byte
big-endian length prefix in front of every payload so the receiver can read
one message at a time. :class:`cadet.core.transport.FramedSocketTransport`
uses this framing for all client/server traffic; the delay lines operate on
whole framed messages. This is fixed wire framing with no user-facing
configuration.

Functions: :func:`send_msg` (length-prefix and send), :func:`recv_msg` (read
one framed payload, ``None`` at EOF), :func:`recvall` (read ``n`` bytes,
``None`` if the peer closes early).
"""

import struct

# Length prefix: 4-byte big-endian unsigned int. The byte width is derived from
# the struct format so the two can never drift apart.
_LEN_FORMAT = ">I"
_LEN_BYTES = struct.calcsize(_LEN_FORMAT)


def send_msg(sock, msg):
    """Send ``msg`` (bytes) prefixed with its ``_LEN_BYTES``-byte length."""
    msg = struct.pack(_LEN_FORMAT, len(msg)) + msg
    sock.sendall(msg)


def recv_msg(sock):
    """Receive one length-framed message, or ``None`` if the peer closed first."""
    raw_msglen = recvall(sock, _LEN_BYTES)
    if not raw_msglen:
        return None
    msglen = struct.unpack(_LEN_FORMAT, raw_msglen)[0]
    return recvall(sock, msglen)


def recvall(sock, n):
    """Read ``n`` bytes from ``sock``, or ``None`` if it closes early."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data
