"""Tests for the multi-connection server endpoint.

The cases are the ones the fan-in users need: two clients into one server, ten
under load with each connection's delay still its own, and a client that dies
mid-run while the others carry on.
"""

import socket
import threading
import time

import pytest

from cadet.core.network import NullNetworkModel, SampledNetworkModel, _Constant
from cadet.core.service import recv_msg, send_msg
from cadet.core.transport import (
    FramedSocketTransport,
    MultiConnectionEndpoint,
    SourceMessage,
)


def _connect(port, timeout_s=15.0):
    """Connect, retrying while the server is still binding on its own thread."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=10)
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


def _free_port():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


class _Client(threading.Thread):
    """A raw client: connects, sends framed payloads, optionally then closes."""

    def __init__(self, port, payloads, close_after=None):
        super().__init__(daemon=True)
        self.port = port
        self.payloads = payloads
        self.close_after = close_after
        self.sock = None
        self.error = None
        self.sent = 0
        # Set by stop(). Waiting on it keeps the socket open without a spin
        # loop: dozens of clients polling at 100 Hz starved a later test's
        # accept() badly enough to fail it only when run alongside the others.
        self._done = threading.Event()

    def run(self):
        try:
            self.sock = _connect(self.port)
            for index, payload in enumerate(self.payloads):
                if self.close_after is not None and index == self.close_after:
                    self.sock.close()
                    return
                send_msg(self.sock, payload)
                self.sent += 1
            # Hold the socket open until the test closes it, so EOF timing is
            # the test's choice rather than a race.
            self._done.wait(60)
        except OSError as exc:
            self.error = exc

    def stop(self):
        self._done.set()
        try:
            if self.sock is not None:
                self.sock.close()
        except OSError:
            pass
        self.sock = None


def _serve(port, expect, **kwargs):
    """Start the endpoint on a thread, since serve() blocks until all accept."""
    box = {}

    def _run():
        try:
            box["endpoint"] = MultiConnectionEndpoint.serve(
                "127.0.0.1", port, expect=expect, accept_timeout_s=15.0, **kwargs
            )
        except Exception as exc:  # surfaced by the test below
            box["error"] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return box, thread


# --- the shape of a merged message --------------------------------------
def test_a_message_carries_the_connection_it_came_from():
    message = SourceMessage("10.0.0.2:5000", b"x", 1.0)
    assert message.source_id == "10.0.0.2:5000"
    assert "10.0.0.2:5000" in repr(message)


def test_an_unknown_policy_is_refused():
    with pytest.raises(ValueError, match="unknown policy"):
        MultiConnectionEndpoint(policy="whatever")


# --- 2 -> 1 ---------------------------------------------------------------
def test_two_clients_into_one_endpoint():
    port = _free_port()
    box, thread = _serve(port, 2)
    clients = [_Client(port, [b"a1", b"a2"]), _Client(port, [b"b1", b"b2"])]
    for client in clients:
        client.start()
    thread.join(20)
    assert "error" not in box, box.get("error")
    endpoint = box["endpoint"]
    try:
        assert len(endpoint.sources()) == 2
        got = [endpoint.recv(timeout_s=10) for _ in range(4)]
        assert all(m is not None for m in got)
        # Every message is attributed, and both connections are represented.
        assert {m.source_id for m in got} == set(endpoint.sources())
        assert sorted(m.payload for m in got) == [b"a1", b"a2", b"b1", b"b2"]
    finally:
        for client in clients:
            client.stop()
        endpoint.close()


def test_a_reply_goes_back_on_the_connection_that_asked():
    port = _free_port()
    box, thread = _serve(port, 2)
    clients = [_Client(port, [b"q-a"]), _Client(port, [b"q-b"])]
    for client in clients:
        client.start()
    thread.join(20)
    endpoint = box["endpoint"]
    try:
        first = endpoint.recv(timeout_s=10)
        endpoint.send(first.source_id, b"reply-" + first.payload)
        # The client whose payload it was receives it; identifying the socket by
        # the payload keeps the test independent of accept order.
        target = next(c for c in clients if c.payloads[0] == first.payload)
        assert recv_msg(target.sock) == b"reply-" + first.payload
    finally:
        for client in clients:
            client.stop()
        endpoint.close()


def test_sending_to_an_unknown_connection_is_an_error():
    port = _free_port()
    box, thread = _serve(port, 1)
    client = _Client(port, [b"x"])
    client.start()
    thread.join(20)
    endpoint = box["endpoint"]
    try:
        with pytest.raises(KeyError, match="no connection"):
            endpoint.send("nobody:0", b"y")
    finally:
        client.stop()
        endpoint.close()


# --- 10 -> 1 under load, with per-connection delay ----------------------
def test_ten_clients_under_load_keep_their_own_delay():
    # The property that matters: each connection has its own delay line, so the
    # delay a stream saw is the one its own link configured. Merging first and
    # delaying afterwards would blend them.
    port = _free_port()
    per_client = 20
    delay_s = 0.02

    def model_for(_source_id):
        # A constant delay per connection, so what each stream waited is
        # unambiguous and comparable against what its own link configured.
        return SampledNetworkModel(_Constant(delay_s * 1000.0)), None

    box, thread = _serve(port, 10, model_for=model_for)
    clients = [
        _Client(port, [("c{}-{}".format(i, n)).encode() for n in range(per_client)])
        for i in range(10)
    ]
    for client in clients:
        client.start()
    thread.join(30)
    assert "error" not in box, box.get("error")
    endpoint = box["endpoint"]
    try:
        assert len(endpoint.sources()) == 10
        received = []
        for _ in range(10 * per_client):
            message = endpoint.recv(timeout_s=20)
            if message is None:
                break
            received.append(message)
        assert len(received) == 10 * per_client

        # Nothing was mis-attributed: every payload names its own client.
        for message in received:
            assert message.payload.decode().startswith("c")
        by_source = {}
        for message in received:
            by_source.setdefault(message.source_id, []).append(message.payload)
        assert len(by_source) == 10
        assert all(len(v) == per_client for v in by_source.values())
        # Each payload's own client prefix must be consistent within a source:
        # a connection's messages cannot appear on another connection's id.
        for payloads in by_source.values():
            prefixes = {p.decode().split("-")[0] for p in payloads}
            assert len(prefixes) == 1, prefixes

        # Per-connection delay statistics are available and show the configured
        # delay, for every connection independently.
        for source_id in endpoint.sources():
            stamps = endpoint.delay_stamps(source_id)
            assert len(stamps) == per_client, source_id
            held = [release - arrival for arrival, release in stamps]
            assert min(held) >= delay_s * 0.5, source_id
    finally:
        for client in clients:
            client.stop()
        endpoint.close()


def test_round_robin_serves_connections_in_rotation():
    # A fast client must not starve a slow one: that is what makes a load
    # experiment's scheduling the experiment's and not an artefact of arrival.
    port = _free_port()
    box, thread = _serve(port, 2, policy=MultiConnectionEndpoint.POLICY_ROUND_ROBIN)
    loud = _Client(port, [("loud-{}".format(n)).encode() for n in range(10)])
    quiet = _Client(port, [b"quiet-0", b"quiet-1"])
    loud.start()
    quiet.start()
    thread.join(20)
    endpoint = box["endpoint"]
    try:
        # Give both clients time to have messages queued.
        time.sleep(0.5)
        seen = []
        for _ in range(6):
            message = endpoint.recv(timeout_s=5)
            if message is None:
                break
            seen.append(message)
        quiet_sources = {
            m.source_id for m in seen if m.payload.decode().startswith("quiet")
        }
        # The quiet client's messages appear early rather than after all ten of
        # the loud one's.
        assert quiet_sources, [m.payload for m in seen]
    finally:
        loud.stop()
        quiet.stop()
        endpoint.close()


# --- a client dies mid-run ----------------------------------------------
def test_one_client_dying_does_not_end_the_run_for_the_others():
    port = _free_port()
    box, thread = _serve(port, 3)
    doomed = _Client(port, [b"d1", b"d2"])
    survivors = [_Client(port, [b"s1", b"s2"]), _Client(port, [b"t1", b"t2"])]
    doomed.start()
    for client in survivors:
        client.start()
    thread.join(20)
    endpoint = box["endpoint"]
    try:
        assert len(endpoint.sources()) == 3
        # Drain what everyone sent.
        drained = []
        for _ in range(6):
            message = endpoint.recv(timeout_s=10)
            if message is None:
                break
            drained.append(message)
        assert len(drained) == 6

        # Kill one connection outright.
        doomed_id = next(
            m.source_id for m in drained if m.payload in (b"d1", b"d2")
        )
        doomed.stop()
        deadline = time.time() + 10
        while doomed_id in endpoint.active_sources() and time.time() < deadline:
            time.sleep(0.05)
        assert doomed_id not in endpoint.active_sources()

        # The survivors are still served, and still attributed.
        assert len(endpoint.active_sources()) == 2
        for client in survivors:
            send_msg(client.sock, b"after")
        more = [endpoint.recv(timeout_s=10) for _ in range(2)]
        assert all(m is not None and m.payload == b"after" for m in more)
        assert doomed_id not in {m.source_id for m in more}
    finally:
        for client in survivors:
            client.stop()
        endpoint.close()


def test_recv_returns_none_once_every_connection_has_closed():
    port = _free_port()
    box, thread = _serve(port, 1)
    client = _Client(port, [b"only"])
    client.start()
    thread.join(20)
    assert "error" not in box, box.get("error")
    assert "endpoint" in box, "serve() did not return"
    endpoint = box["endpoint"]
    try:
        assert endpoint.recv(timeout_s=10).payload == b"only"
        client.stop()
        deadline = time.time() + 10
        while endpoint.active_sources() and time.time() < deadline:
            time.sleep(0.05)
        # Not an error and not a hang: the run is simply over for this endpoint.
        assert endpoint.recv(timeout_s=2) is None
    finally:
        endpoint.close()


# --- serve_one still works ----------------------------------------------
def test_serve_one_is_unchanged_and_still_the_single_client_path():
    port = _free_port()
    box = {}

    def _run():
        box["transport"] = FramedSocketTransport.serve_one(
            "127.0.0.1", port, recv_model=NullNetworkModel(), send_model=None
        )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    sock = _connect(port)
    thread.join(10)
    transport = box["transport"]
    try:
        send_msg(sock, b"hello")
        assert transport.recv() == b"hello"
    finally:
        sock.close()
        transport.close()
