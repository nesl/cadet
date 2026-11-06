"""The server role fans in several clients via MultiConnectionEndpoint (expect>1).

Drives a real ServerRuntime with ``expect=2`` over loopback sockets: two clients
connect, each sends a frame, and each must get back its OWN reply (routed to the
originating source via the endpoint). Also checks that ``expect=1`` keeps the original
single-accept path. The server runs in a thread; ``StopSignal.install`` is a no-op off
the main thread, so the run ends when both clients close (all connections closed).
"""

import socket
import threading
import time

from cadet.core.transport import FramedSocketTransport
from cadet.roles.server import ServerRuntime


class _Models:
    """Trivial model factory: no network shaping, both accept-loop and fan-in forms."""

    def for_source(self, source_id):
        return (None, None)

    def __call__(self, cid, peer_ip=None):
        return (None, None)


class _EchoApp:
    """Replies ``re:<payload>`` to every frame; never stops after a reply."""

    def __init__(self):
        self.handled = []

    def setup(self):
        pass

    def handle(self, payload):
        self.handled.append(payload)
        return b"re:" + payload

    def stop_after_reply(self):
        return False

    def teardown(self):
        pass


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_fanin_two_clients_each_get_their_own_reply():
    port = _free_port()
    app = _EchoApp()
    srv = ServerRuntime(
        "127.0.0.1", port, _Models(), app, logger=lambda m: None, warmup=False, expect=2
    )
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    # serve() blocks until both clients connect; connect retries over the brief
    # bind race.
    c1 = FramedSocketTransport.connect("127.0.0.1", port, retry_s=5.0)
    c2 = FramedSocketTransport.connect("127.0.0.1", port, retry_s=5.0)
    try:
        c1.send(b"a")
        c2.send(b"b")
        r1 = c1.recv()
        r2 = c2.recv()
        assert r1 == b"re:a"       # each client gets its own reply, routed by source
        assert r2 == b"re:b"
        assert set(app.handled) == {b"a", b"b"}
    finally:
        c1.close()
        c2.close()
    t.join(timeout=5.0)
    assert not t.is_alive()         # closing both connections ends the run


def test_expect_one_keeps_single_accept_path():
    port = _free_port()
    app = _EchoApp()
    srv = ServerRuntime(
        "127.0.0.1", port, _Models(), app, logger=lambda m: None, warmup=False, expect=1
    )
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    time.sleep(0.2)  # single-accept binds immediately; no expect barrier
    c = FramedSocketTransport.connect("127.0.0.1", port, retry_s=5.0)
    try:
        c.send(b"x")
        assert c.recv() == b"re:x"
        assert app.handled == [b"x"]
    finally:
        c.close()
    t.join(timeout=5.0)
    assert not t.is_alive()
