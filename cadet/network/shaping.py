"""Shared description of how one direction of a link perturbs traffic.

Both packet-level backends (``netem`` and ``netwaggle``) need the same numbers
(delay, jitter, loss, bandwidth) for one direction of one link, and both need
the same answer to "what does this distribution mean in the kernel?". Keeping
that here means the two backends agree on what a link's network model means
and differ only in where the qdisc is installed.

These are packet-level parameters: they are rendered into ``tc`` arguments and
TCLink keywords so they act on packets and the transport reacts to them. The
message-level backend (``mode: delay``) does not use this object; it samples
its distribution per application message in :mod:`cadet.core.network`.

Two measured facts about netem drive the mapping:

1. **netem's own jitter is uniform**, not Gaussian: ``delay 30ms 15ms``
   produces delays spread evenly over 15-45 ms. Its p99 sits at
   ``delay + 0.98 * jitter``, and it cannot represent a long tail.
2. **netem reorders packets when it jitters them.** Each packet is given an
   independent delay and released when that delay expires, so a packet behind
   can overtake one in front. Measured on a veth pair with 200 paced
   datagrams: ``delay 30ms`` gave 0 inversions, ``delay 30ms 15ms`` gave 65.
   Adding a ``rate`` gave 0 inversions, because netem serialises its output
   through a rate stage. A jittered link is therefore only installed with a
   ``rate``, and a config that asks for jitter without a bandwidth is refused.

Contains: :data:`Z99` (re-exported from :mod:`cadet.core.network`, which
samples the distributions) and :func:`percentiles_ms` (p50/p99 of a configured
distribution), :class:`Shaping` (parameters plus renderers for
Mininet's TCLink keywords and ``tc`` arguments), and :func:`shaping_from_network`
(the mapping from a config ``network`` block to a :class:`Shaping`).
Built from ``config.links[*].network`` via ``cadet.network.netwaggle.generator``
and ``cadet.network.netem``. Nothing here imports Mininet or runs commands.
"""

from typing import Dict, List, Optional, Tuple

# The standard normal's 99th percentile, for converting between a p50/p99
# intent and a distribution's parameters. It is defined beside the sampling it
# parameterises and re-exported here, so the message-level and packet-level
# backends cannot drift apart.
from cadet.core.network import Z99  # noqa: F401  (re-exported)

# netem spreads `delay M J` uniformly over [M-J, M+J], so its 99th percentile is
# this fraction of the way from M to M+J.
UNIFORM_P99_FRACTION = 0.98


class UnrepresentableDelay(ValueError):
    """A delay distribution cannot be installed in the kernel as configured."""


def percentiles_ms(dist) -> Tuple[float, float]:
    """Return ``(p50_ms, p99_ms)`` for a configured delay distribution.

    This is what lets a link report state the intent next to the achieved
    shaping. ``lognormal`` states its two percentiles directly, so for that kind
    this only reads them back.
    """
    kind = getattr(dist, "kind", "constant")
    if kind == "constant":
        value = float(getattr(dist, "value_ms", 0.0))
        return value, value
    if kind == "normal":
        return float(dist.mean_ms), float(dist.mean_ms) + Z99 * float(dist.std_ms)
    if kind == "lognormal":
        return float(dist.p50_ms), float(dist.p99_ms)
    raise UnrepresentableDelay(
        "cannot compute percentiles for distribution kind {!r}".format(kind)
    )


class Shaping:
    """How one direction of a link perturbs traffic.

    All fields are optional, and ``None`` means "not configured", which is not
    the same as zero: netem with an explicit 0 ms delay still installs a qdisc
    and so adds a little latency and reordering risk. An empty :class:`Shaping`
    therefore installs nothing at all.
    """

    def __init__(
        self,
        delay_ms: Optional[float] = None,
        jitter_ms: Optional[float] = None,
        loss_pct: Optional[float] = None,
        bandwidth_mbps: Optional[float] = None,
    ):
        self.delay_ms = delay_ms
        self.jitter_ms = jitter_ms
        self.loss_pct = loss_pct
        self.bandwidth_mbps = bandwidth_mbps

    @property
    def is_empty(self) -> bool:
        """True when nothing is configured, so no qdisc need be installed."""
        return all(
            v is None for v in (self.delay_ms, self.jitter_ms, self.loss_pct, self.bandwidth_mbps)
        )

    @property
    def jitters(self) -> bool:
        """True when this direction varies its per-packet delay."""
        return bool(self.jitter_ms)

    @property
    def preserves_order(self) -> bool:
        """True when this shaping cannot reorder packets within a flow.

        Either it does not jitter at all, or it jitters behind a ``rate`` that
        serialises netem's output. :func:`shaping_from_network` refuses anything
        else, so this should hold for every installed link.
        """
        return (not self.jitters) or self.bandwidth_mbps is not None

    @property
    def achieved_percentiles_ms(self) -> Tuple[float, float]:
        """The ``(p50, p99)`` this shaping will actually produce, in ms.

        netem's jitter is uniform, so the p99 is a fixed fraction of the way to
        ``delay + jitter`` rather than a Gaussian tail.
        """
        delay = self.delay_ms or 0.0
        if not self.jitters:
            return delay, delay
        return delay, delay + UNIFORM_P99_FRACTION * float(self.jitter_ms)

    def as_mininet_params(self) -> Dict[str, object]:
        """Return TCLink keyword parameters for this direction.

        Mininet wants ``delay``/``jitter`` as strings with units, ``loss`` as a
        percentage number and ``bw`` in Mbit/s. Only configured fields appear.

        A jittered direction cannot be expressed here in an order-preserving
        way: TCLink puts its bandwidth stage (htb) *before* netem, so netem is
        last in the egress path and its reordering survives. The emulator
        therefore replaces the qdisc on such an interface after the topology is
        up, using :meth:`netem_args`; see
        :meth:`cadet.network.netwaggle.emulator.Emulator` .
        """
        params: Dict[str, object] = {}
        if self.delay_ms is not None:
            params["delay"] = "{}ms".format(self.delay_ms)
        if self.jitter_ms is not None:
            params["jitter"] = "{}ms".format(self.jitter_ms)
        if self.loss_pct is not None:
            params["loss"] = self.loss_pct
        if self.bandwidth_mbps is not None:
            params["bw"] = self.bandwidth_mbps
        return params

    def netem_args(self) -> List[str]:
        """Return the ``netem`` arguments for delay, jitter, loss and rate.

        Jitter requires a delay: ``netem delay`` takes the jitter as its second
        positional value, so a jitter without a delay is rendered against a zero
        base rather than silently dropped.

        A ``rate`` is emitted whenever this direction jitters, because that is
        what keeps netem from reordering the flow (see the module docstring).
        """
        args: List[str] = []
        if self.delay_ms is not None or self.jitter_ms is not None:
            args += ["delay", "{}ms".format(self.delay_ms or 0.0)]
            if self.jitter_ms is not None:
                args.append("{}ms".format(self.jitter_ms))
        if self.loss_pct is not None:
            args += ["loss", "{}%".format(self.loss_pct)]
        if self.jitters and self.bandwidth_mbps is not None:
            args += ["rate", "{}mbit".format(self.bandwidth_mbps)]
        return args

    def __repr__(self) -> str:
        return "Shaping({})".format(self.as_mininet_params() or "unshaped")


def shaping_from_network(net) -> Shaping:
    """Build a :class:`Shaping` from a link's ``network`` block.

    The one place a configured distribution becomes kernel parameters, shared by
    both packet-level backends so they cannot drift apart.

    ``constant`` is exact. ``normal`` and ``lognormal`` are matched on their p50
    and p99 onto netem's uniform jitter, which is the most a single netem can
    represent; the achieved shape is recorded in the run's link report so the
    approximation is visible. Two cases are refused outright rather than
    approximated into something that measures itself:

    * a spread too wide for a symmetric jitter (``p99 > 2 * p50``), which would
      need negative delays that netem clamps to zero, moving the p50;
    * a jittered link with no ``bandwidth_mbps``, which netem can only deliver
      by reordering the flow.
    """
    p50_ms, p99_ms = _percentiles_for_kernel(net)
    return shaping_from_percentiles(
        p50_ms,
        p99_ms,
        loss_fraction=getattr(net, "loss", 0.0) or 0.0,
        bandwidth_mbps=getattr(net, "bandwidth_mbps", None),
    )


def shaping_from_path(nets) -> Shaping:
    """Build one :class:`Shaping` for a composed path of hop ``network`` blocks.

    A composed route (``routes[*].realise: compose``) keeps one end-to-end
    connection, so there is one sender and therefore **one qdisc**: the hops
    have to be collapsed into a single delay, jitter, loss and rate. The
    composition is stated in percentiles, which is what the mapping already
    works in:

    * the p50 and the p99 **add**. Adding the p99s is an upper bound, not a
      prediction: independent hops' samples add, but the tail of a sum is
      narrower than the sum of the tails. It is the conservative direction - the
      installed spread is at least what the hops ask for - and it is what makes
      a two-hop path with hops 30 ms +/- a little representable at all.
    * loss is the chance of **not** surviving every hop.
    * bandwidth is the **slowest** hop's: hops in series are a pipeline, and the
      bottleneck sets the rate.

    The refusals are then evaluated on the **composed** shaping, because the
    composition is what gets installed. That is not the same as checking each
    hop, and it is worth knowing which way it differs:

    * the spread rule is linear, so hops that are each representable always
      compose into a representable path (each hop's jitter is at most its own
      delay, so the sums keep that relation). Composing cannot turn acceptable
      hops into a refusal.
    * the bandwidth rule is **weaker** on a path than on a hop. A jittered hop
      with no bandwidth of its own is refused as a link, because netem would
      reorder its flow; in a composed path it is accepted when *any* hop sets a
      bandwidth, because there is one qdisc and that rate serialises the whole
      path's output. The reordering the rule exists to prevent cannot happen.
    """
    nets = list(nets)
    if not nets:
        return Shaping()
    if len(nets) == 1:
        return shaping_from_network(nets[0])

    p50_total = p99_total = 0.0
    survives = 1.0
    bandwidths = []
    for net in nets:
        hop_p50, hop_p99 = _percentiles_for_kernel(net)
        p50_total += hop_p50
        p99_total += hop_p99
        survives *= 1.0 - (getattr(net, "loss", 0.0) or 0.0)
        bandwidth = getattr(net, "bandwidth_mbps", None)
        if bandwidth:
            bandwidths.append(bandwidth)
    return shaping_from_percentiles(
        p50_total,
        p99_total,
        loss_fraction=1.0 - survives,
        bandwidth_mbps=min(bandwidths) if bandwidths else None,
        subject="path",
    )


def _percentiles_for_kernel(net) -> Tuple[float, float]:
    """Return a hop's ``(p50, p99)``, refusing a distribution netem cannot hold."""
    dist = getattr(net, "distribution", None)
    if getattr(dist, "kind", "constant") == "trace":
        raise UnrepresentableDelay(
            "distribution 'trace' has no netem equivalent: netem takes a delay "
            "and a uniform jitter, not a sequence of delays. Use "
            "network.mode: delay, whose in-process delay line replays the trace "
            "exactly."
        )
    return percentiles_ms(dist)


def shaping_from_percentiles(
    p50_ms: float,
    p99_ms: float,
    loss_fraction: float = 0.0,
    bandwidth_mbps: Optional[float] = None,
    subject: str = "link",
) -> Shaping:
    """Map a ``(p50, p99)`` intent onto netem, or refuse it.

    One link and a composed path differ only in how their percentiles were
    arrived at, so both are checked here and ``subject`` only names which one a
    refusal is talking about.
    """
    bandwidth = bandwidth_mbps
    if p99_ms > p50_ms:
        # netem's jitter is symmetric about the delay, so the spread it can
        # represent is bounded by the delay itself.
        jitter_ms = (p99_ms - p50_ms) / UNIFORM_P99_FRACTION
        if jitter_ms > p50_ms:
            raise UnrepresentableDelay(
                "a {} with p50 {:.1f} ms and p99 {:.1f} ms needs a jitter of "
                "{:.1f} ms, which is wider than the delay: netem's jitter is "
                "symmetric, so the low side would be negative and get clamped "
                "to zero, moving the p50. Keep p99 below {:.1f} ms for a kernel "
                "backend, or use network.mode: delay, which samples the "
                "distribution exactly.".format(
                    subject, p50_ms, p99_ms, jitter_ms, p50_ms * (1.0 + UNIFORM_P99_FRACTION)
                )
            )
        if bandwidth is None:
            raise UnrepresentableDelay(
                "a jittered kernel link needs network.bandwidth_mbps. netem "
                "gives each packet an independent delay, so without a rate to "
                "serialise its output it reorders the flow (measured: 65 "
                "inversions in 200 packets at 30 ms +/- 15 ms), which collapses "
                "the TCP window and measures the reordering rather than the "
                "link. Set a bandwidth, or use a constant delay, or "
                "network.mode: delay for message-level emulation."
            )
    else:
        jitter_ms = None

    return Shaping(
        delay_ms=p50_ms if p50_ms else None,
        jitter_ms=jitter_ms,
        loss_pct=(loss_fraction * 100.0) if loss_fraction > 0 else None,
        bandwidth_mbps=bandwidth,
    )
