"""Network model for the message-level link backend (``mode: delay``).

Defines how a link perturbs traffic at the message level: a delay sampled per
application message, message loss, and bandwidth serialization. The queuing
and release is done by the delay line in :mod:`cadet.core.transport`; this
module only supplies the samples.

This is one of two emulation layers (see "Link backend semantics" in
``docs/config-reference.md``). At the message level (``mode: delay``), whole
application messages are held and dropped. ``direction: forward`` delays
requests and leaves replies untouched. The sampled distribution is reproduced
at the application level because nothing between the two processes reacts to
it. At the packet level (``mode: netem``, ``mode: netwaggle``), a kernel qdisc
holds packets and the transport reacts. A configured 60 ms forward link
measures p50 60.4 ms here and about 149 ms through a packet-level backend;
that gap is TCP's behaviour rather than emulator error.

Contains: :class:`NetworkModel` (``sample_delay_s`` / ``drops`` /
``serialization_s``), :class:`NullNetworkModel` (pass-through), distribution
samplers (constant / normal / lognormal / trace), and
:func:`build_network_model`. Built from ``config.links[*].network``; only
``mode: delay`` is realised here. ``none``, ``netem`` and ``netwaggle`` all
yield a :class:`NullNetworkModel` so kernel-level backends are not counted
twice.
"""

import math
import os
import random
from abc import ABC, abstractmethod
from typing import List, Optional

# The standard normal's 99th percentile. A lognormal's sigma follows from the
# ratio of its p50 and p99 through this, and cadet.network.shaping converts the
# same way for the kernel backends, so it is defined once, here, beside the
# sampling it parameterises.
Z99 = 2.3263478740408408

# --- distribution samplers (return a delay in seconds) --------------------


class _Distribution(ABC):
    @abstractmethod
    def sample_s(self, rng: random.Random) -> float:
        """Return a non-negative delay sample in seconds."""


class _Constant(_Distribution):
    def __init__(self, value_ms: float):
        self._s = max(0.0, value_ms) / 1000.0

    def sample_s(self, rng: random.Random) -> float:
        return self._s


class _Normal(_Distribution):
    def __init__(self, mean_ms: float, std_ms: float):
        self._mean = mean_ms / 1000.0
        self._std = std_ms / 1000.0

    def sample_s(self, rng: random.Random) -> float:
        return max(0.0, rng.gauss(self._mean, self._std))


class _Lognormal(_Distribution):
    """Log-normal parameterised by its median and its 99th percentile (in ms).

    The median is the distribution's scale, so mu is its log; sigma follows from
    the ratio of the two percentiles alone.
    """

    def __init__(self, p50_ms: float, p99_ms: float):
        self._mu = math.log(max(1e-6, p50_ms / 1000.0))
        self._sigma = math.log(max(1.0, p99_ms / max(1e-6, p50_ms))) / Z99

    def sample_s(self, rng: random.Random) -> float:
        return max(0.0, rng.lognormvariate(self._mu, self._sigma))


class _Trace(_Distribution):
    """Replays delays (one per line, in ms) from a trace file, cycling."""

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise RuntimeError("delay trace not found: {}".format(path))
        with open(path) as fh:
            self._vals: List[float] = [float(x) / 1000.0 for x in fh.read().split() if x.strip()]
        if not self._vals:
            raise RuntimeError("empty delay trace: {}".format(path))
        self._i = 0

    def sample_s(self, rng: random.Random) -> float:
        v = self._vals[self._i % len(self._vals)]
        self._i += 1
        return max(0.0, v)


def _build_distribution(dist) -> _Distribution:
    kind = getattr(dist, "kind", "constant")
    if kind == "constant":
        return _Constant(getattr(dist, "value_ms", 0.0))
    if kind == "normal":
        return _Normal(dist.mean_ms, dist.std_ms)
    if kind == "lognormal":
        return _Lognormal(dist.p50_ms, dist.p99_ms)
    if kind == "trace":
        return _Trace(dist.path)
    raise ValueError("unknown distribution kind {!r}".format(kind))


# --- network models -------------------------------------------------------
class NetworkModel(ABC):
    """Per-message network perturbation sampler."""

    @abstractmethod
    def sample_delay_s(self) -> float:
        """Sample a propagation delay (seconds) for the next message."""

    @abstractmethod
    def drops(self) -> bool:
        """Return ``True`` if the current message should be dropped (loss)."""

    @abstractmethod
    def serialization_s(self, size_bytes: int) -> float:
        """Transmission (serialization) delay for ``size_bytes`` at the link rate."""


class NullNetworkModel(NetworkModel):
    """Pass-through: zero delay, no loss, no bandwidth limit."""

    def sample_delay_s(self) -> float:
        return 0.0

    def drops(self) -> bool:
        return False

    def serialization_s(self, size_bytes: int) -> float:
        return 0.0


class SampledNetworkModel(NetworkModel):
    """Delay from a distribution, plus optional loss and bandwidth."""

    def __init__(
        self,
        distribution: _Distribution,
        loss: float = 0.0,
        bandwidth_mbps: Optional[float] = None,
        rng: Optional[random.Random] = None,
    ):
        self._dist = distribution
        self._loss = loss
        self._bw = bandwidth_mbps
        self._rng = rng if rng is not None else random.Random()

    def sample_delay_s(self) -> float:
        return self._dist.sample_s(self._rng)

    def drops(self) -> bool:
        return self._loss > 0.0 and self._rng.random() < self._loss

    def serialization_s(self, size_bytes: int) -> float:
        if not self._bw or self._bw <= 0:
            return 0.0
        return (size_bytes * 8.0) / (self._bw * 1e6)


class CompositeNetworkModel(NetworkModel):
    """Several hops in series, as one per-message model.

    A composed multi-hop path (``routes[*].realise: compose``) keeps one
    end-to-end connection, so the hops cannot each hold the message in turn:
    their effects are added up instead. Per message that is

    * **delay**: the sum of the hops' samples. Each hop samples from its own
      distribution with its own RNG stream, so two hops configured alike still
      vary independently, which is the point of modelling them as two hops.
    * **loss**: lost if any hop loses it. A message does not arrive by surviving
      the worst hop; it has to survive all of them.
    * **bandwidth**: the slowest hop's serialisation time, not the sum. The
      hops are a pipeline, not a queue: the bottleneck sets the rate, and a
      store-and-forward chain that really did add them up is
      ``realise: relay``, which is a relay process per hop and not this.
    """

    def __init__(self, models: List[NetworkModel]):
        if not models:
            raise ValueError("a composite network model needs at least one hop")
        self._models = list(models)

    def sample_delay_s(self) -> float:
        return sum(model.sample_delay_s() for model in self._models)

    def drops(self) -> bool:
        return any(model.drops() for model in self._models)

    def serialization_s(self, size_bytes: int) -> float:
        return max(model.serialization_s(size_bytes) for model in self._models)


# The only mode realised inside the process. `none` has nothing to do, while
# `netem` and `netwaggle` shape in the kernel, so adding an in-process delay for
# them would apply the configured values twice.
IN_PROCESS_MODES = ("delay",)


def build_network_model(net, rng: Optional[random.Random] = None) -> NetworkModel:
    """Build a runtime :class:`NetworkModel` from a schema ``NetworkModel`` config.

    Only ``mode: delay`` yields a :class:`SampledNetworkModel`; every other mode
    gives a pass-through. ``none`` has no emulation at all, and the ``netem`` and
    ``netwaggle`` backends shape traffic outside this process (in a qdisc or in
    an emulated network), so sampling a delay here as well would apply the
    configured delay twice.

    ``rng`` makes the samples reproducible (seed it from the experiment seed in
    runs; fix it in tests).
    """
    mode = getattr(net, "mode", "none")
    if mode not in IN_PROCESS_MODES:
        return NullNetworkModel()
    dist = _build_distribution(getattr(net, "distribution", None))
    return SampledNetworkModel(
        dist,
        loss=getattr(net, "loss", 0.0) or 0.0,
        bandwidth_mbps=getattr(net, "bandwidth_mbps", None),
        rng=rng,
    )


def build_path_network_model(nets, rng: Optional[random.Random] = None) -> NetworkModel:
    """Build one model for a path of hop ``network`` configs, in order.

    One hop gives exactly what :func:`build_network_model` gives, so a direct
    link and a one-hop path cannot behave differently. Several hops give a
    :class:`CompositeNetworkModel`, each hop seeded from ``rng`` so the streams
    are independent but the run is still reproducible from one seed.
    """
    nets = list(nets)
    if not nets:
        return NullNetworkModel()
    if len(nets) == 1:
        return build_network_model(nets[0], rng=rng)
    models = []
    for index, net in enumerate(nets):
        hop_rng = None if rng is None else random.Random(rng.random() + index)
        models.append(build_network_model(net, rng=hop_rng))
    if all(isinstance(model, NullNetworkModel) for model in models):
        return NullNetworkModel()
    return CompositeNetworkModel(models)
