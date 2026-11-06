"""Exact-timing unit tests for the delay-line link model.

Uses a socketpair to feed framed messages into a real receive delay line and
asserts the applied per-message delay, non-accumulation, ordering, loss rate and
slow-consumer invariance. No network, no CARLA.

The delay line reads through the wire the transport configured rather than
straight off the socket, so these build the default wire (a plain socket with
the length prefix) around the socketpair - which is the same wire every example
has always used.
"""

import random
import socket
import struct
import threading
import time

import pytest

from cadet.core.network import SampledNetworkModel, _Constant, _Lognormal, _Normal
from cadet.core.service import send_msg
from cadet.core.transport import _LengthFraming, _RecvDelayLine, _SocketChannel


def _writer(sock, n, interval=0.0, size=16):
    """Send ``n`` framed messages (payload = index) at ``interval`` spacing."""
    for i in range(n):
        send_msg(sock, struct.pack(">I", i) + b"x" * (size - 4))
        if interval:
            time.sleep(interval)


def _default_wire(sock):
    """The wire `transport.protocol: tcp` + `transport.mode: frame` builds."""
    return _LengthFraming(_SocketChannel(sock))


def _const_model(value_ms, loss=0.0, bw=None, seed=0):
    return SampledNetworkModel(
        _Constant(value_ms), loss=loss, bandwidth_mbps=bw, rng=random.Random(seed)
    )


def test_constant_delay_per_message_not_cumulative():
    a, b = socket.socketpair()
    line = _RecvDelayLine(_default_wire(b), _const_model(60.0))
    # 30 fps feed; read each message as it releases.
    t = threading.Thread(target=_writer, args=(a, 40, 1 / 30.0), daemon=True)
    t.start()
    for _ in range(40):
        line.recv()
    t.join()
    delays = [rel - arr for arr, rel in line.stamps]
    assert len(delays) == 40
    # Each message delayed ~60 ms; the last is NOT larger than the first.
    assert all(abs(d - 0.060) < 0.01 for d in delays), (min(delays), max(delays))
    assert abs(delays[-1] - delays[0]) < 0.01
    a.close()
    b.close()


def test_ordering_preserved():
    a, b = socket.socketpair()
    line = _RecvDelayLine(_default_wire(b), _const_model(20.0))
    threading.Thread(target=_writer, args=(a, 30, 0.0), daemon=True).start()
    ids = []
    for _ in range(30):
        payload = line.recv()
        ids.append(struct.unpack(">I", payload[:4])[0])
    assert ids == list(range(30))
    a.close()
    b.close()


def test_slow_consumer_does_not_change_stamped_delay():
    a, b = socket.socketpair()
    line = _RecvDelayLine(_default_wire(b), _const_model(50.0))
    _writer(a, 20, 0.0)  # send all 20 as fast as possible
    time.sleep(0.5)  # consumer idle while the reader stamps them
    # Reader stamped all 20 at arrival; the stamped delay is ~50 ms for each,
    # unaffected by the consumer being slow.
    assert len(line.stamps) == 20
    delays = [rel - arr for arr, rel in line.stamps]
    assert all(abs(d - 0.050) < 0.015 for d in delays), (min(delays), max(delays))
    a.close()
    b.close()


def test_loss_rate_within_tolerance():
    a, b = socket.socketpair()
    line = _RecvDelayLine(_default_wire(b), _const_model(1.0, loss=0.3, seed=1))
    n = 400
    _writer(a, n, 0.0)
    a.shutdown(socket.SHUT_WR)
    received = 0
    while True:
        if line.recv() is None:
            break
        received += 1
    kept = received / n
    assert abs(kept - 0.7) < 0.07, kept  # ~30% dropped
    a.close()
    b.close()


def _pct(samples, q):
    s = sorted(samples)
    return s[min(len(s) - 1, int(q * len(s)))]


def test_normal_distribution_p50_p99():
    m = SampledNetworkModel(_Normal(50.0, 8.0), rng=random.Random(3))
    s = [m.sample_delay_s() for _ in range(8000)]
    assert abs(_pct(s, 0.50) - 0.050) < 0.004  # p50 ~ mean
    assert abs(_pct(s, 0.99) - (0.050 + 2.33 * 0.008)) < 0.006  # p99 ~ mean+2.33*std


def test_lognormal_hits_both_configured_percentiles():
    # The two parameters are the two percentiles, so both are checked.
    m = SampledNetworkModel(_Lognormal(40.0, 120.0), rng=random.Random(5))
    s = [m.sample_delay_s() for _ in range(8000)]
    assert abs(_pct(s, 0.50) - 0.040) < 0.006
    assert abs(_pct(s, 0.99) - 0.120) < 0.020


# --- which modes are realised in-process ---------------------------------
@pytest.mark.parametrize("mode", ["none", "netem", "netwaggle"])
def test_only_the_delay_backend_shapes_in_process(mode):
    """netem and netwaggle shape in the kernel; sampling here too would double it."""
    from cadet.config.schema import ConstantDist
    from cadet.config.schema import NetworkModel as NetworkModelCfg
    from cadet.core.network import NullNetworkModel, build_network_model

    cfg = NetworkModelCfg(mode=mode, distribution=ConstantDist(value_ms=60.0))
    model = build_network_model(cfg)
    assert isinstance(model, NullNetworkModel)
    assert model.sample_delay_s() == 0.0


def test_delay_backend_is_realised_in_process():
    from cadet.config.schema import ConstantDist
    from cadet.config.schema import NetworkModel as NetworkModelCfg
    from cadet.core.network import build_network_model

    cfg = NetworkModelCfg(mode="delay", distribution=ConstantDist(value_ms=60.0))
    assert build_network_model(cfg).sample_delay_s() == pytest.approx(0.06)
