"""Per-run record of what a link was configured to do and what it actually did.

What this is for:
    The three link backends realise a configuration in different ways, and the
    gap between "configured 60 ms" and "the application saw X ms" is the thing an
    experiment needs to be able to inspect. Every run therefore writes both: the
    link parameters as configured, and the effective per-message latency and
    unanswered counts as measured. Without this the gap is invisible in outputs
    and has to be reconstructed by hand from logs.

What it contains:
    - :class:`MessageStats`: what a client role observed (sent, received, and the
      per-message latencies in milliseconds).
    - :func:`link_summary`: a link's configuration, flattened for the record.
    - :func:`write_link_report`: writes ``link_report.json`` into a run's output
      directory and returns the structure it wrote.

How it fits into CADET:
    :class:`~cadet.roles.client.ClientRuntime` calls :func:`write_link_report`
    after the app finishes, for any app that exposes ``message_stats()``. The
    file sits beside the per-frame CSV and the run marker.

    Latency here is *message* latency: the time from handing a payload to the
    transport until its reply is taken off the wire. It is deliberately recorded
    at reply arrival rather than when the application next looks, because a
    polling loop would otherwise fold its own tick interval into the number.

    A **route** is recorded the same way, with one addition: a composed path has
    no observable hop (one connection, one timestamp pair), so the record states
    each hop's configured percentiles *and* the end-to-end composition, and the
    measured half stays what it was - what the application saw end to end. The
    composed p50 and p99 are the sums of the hops', which is the intent rather
    than a prediction: delays add, but a sum of independent samples has a
    narrower tail than the sum of their p99s, so the composed p99 is an upper
    bound. Per-hop *measurement* is a property of ``realise: relay``, where each
    hop is a real connection.

How to configure it:
    Nothing to configure. The configured half comes from ``config.links`` and
    ``config.routes``; the measured half from the app.
"""

import json
import os
from typing import Dict, List, Optional

# One report per client node: the measured half is that client's, so a single
# fixed name meant N clients in one run directory overwrote each other.
REPORT_PREFIX = "link_report"


def report_name(node_name) -> str:
    """Return the link-report file name for a client node."""
    return "{}_{}.json".format(REPORT_PREFIX, node_name)


# How an app answers messages, which decides whether unanswered counts mean
# anything about the link.
#   per-message  every message is answered, so an unanswered one was lost.
#   on-event     the server answers only when something happens (the braking app
#                replies when it issues a brake), so most messages are
#                unanswered by design and the count is not a loss measure.
REPLY_PER_MESSAGE = "per-message"
REPLY_ON_EVENT = "on-event"


class MessageStats:
    """What a client observed at the message level over one run."""

    def __init__(
        self,
        sent: int,
        received: int,
        latencies_ms: Optional[List[float]] = None,
        reply_policy: str = REPLY_PER_MESSAGE,
    ):
        self.sent = sent
        self.received = received
        self.latencies_ms = [v for v in (latencies_ms or []) if v is not None]
        self.reply_policy = reply_policy

    @property
    def unanswered(self) -> int:
        """Messages sent that never got a reply."""
        return max(0, self.sent - self.received)

    @property
    def unanswered_pct(self) -> Optional[float]:
        """Unanswered share, or None when the app does not answer every message.

        Returning a percentage for an ``on-event`` app would invite reading it as
        a loss rate, which it is not.
        """
        if not self.sent or self.reply_policy != REPLY_PER_MESSAGE:
            return None
        return 100.0 * self.unanswered / self.sent

    def percentile(self, pct: float) -> Optional[float]:
        """Return the ``pct`` percentile latency in ms, or None with no samples."""
        if not self.latencies_ms:
            return None
        ordered = sorted(self.latencies_ms)
        index = int(round(pct / 100.0 * (len(ordered) - 1)))
        return ordered[min(max(index, 0), len(ordered) - 1)]

    def as_dict(self) -> Dict[str, object]:
        mean = sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else None
        return {
            "reply_policy": self.reply_policy,
            "messages_sent": self.sent,
            "messages_received": self.received,
            "unanswered": self.unanswered,
            # None unless every message is meant to be answered; see reply_policy.
            "unanswered_pct": self.unanswered_pct,
            "latency_ms": {
                "samples": len(self.latencies_ms),
                "p50": self.percentile(50),
                "p99": self.percentile(99),
                "mean": mean,
                "min": min(self.latencies_ms) if self.latencies_ms else None,
                "max": max(self.latencies_ms) if self.latencies_ms else None,
            },
        }


def link_summary(link) -> Dict[str, object]:
    """Flatten one configured link into the record.

    ``semantics`` states what the numbers mean for the chosen backend, because
    the same ``delay`` and ``loss`` fields are realised at different layers:
    ``delay`` perturbs application messages, while ``netem`` and ``netwaggle``
    perturb packets and let the transport react.
    """
    net = link.network
    dist = net.distribution
    return {
        "name": link.name,
        "from": link.from_,
        "to": link.to,
        "direction": link.direction,
        "mode": net.mode,
        "medium": getattr(net, "medium", "wired"),
        "semantics": "message-level"
        if net.mode == "delay"
        else ("packet-level" if net.mode in ("netem", "netwaggle") else "none"),
        "distribution": dist.model_dump() if hasattr(dist, "model_dump") else None,
        "loss": net.loss,
        "bandwidth_mbps": net.bandwidth_mbps,
    }


def route_summary(config, route) -> Dict[str, object]:
    """Flatten one configured route into the record, hop by hop.

    ``end_to_end`` is the composition of the hops that act on each direction:
    the delay percentiles add, the loss is the chance of surviving every hop,
    and the bandwidth is the slowest hop's, because hops in series are a
    pipeline whose bottleneck sets the rate.
    """
    from cadet.config.routes import hops_in_direction, route_hops

    hops = route_hops(config, route)
    composed = {}
    for direction in ("forward", "reverse"):
        acting = hops_in_direction(hops, direction)
        if not acting:
            continue
        composed[direction] = _compose_intent(acting)
    return {
        "name": route.name,
        "from": route.from_,
        "to": route.to,
        "realise": route.realise,
        "via": [hop.to for hop in hops[:-1]],
        "hops": [link_summary(hop) for hop in hops],
        "end_to_end": composed,
    }


def _compose_intent(hops) -> Dict[str, object]:
    """Return the composed configured behaviour of ``hops``, in order."""
    from cadet.network.shaping import percentiles_ms

    p50 = p99 = 0.0
    representable = True
    survives = 1.0
    bandwidths = []
    for hop in hops:
        try:
            hop_p50, hop_p99 = percentiles_ms(hop.network.distribution)
        except Exception:
            # A trace distribution has no closed-form percentiles; the hops are
            # still recorded, so say so rather than guess a number.
            representable = False
            continue
        p50 += hop_p50
        p99 += hop_p99
        survives *= 1.0 - (hop.network.loss or 0.0)
        if hop.network.bandwidth_mbps:
            bandwidths.append(hop.network.bandwidth_mbps)
    return {
        "hops": [hop.name for hop in hops],
        "p50_ms": p50 if representable else None,
        # An upper bound: independent hops' samples add, their p99s do not.
        "p99_ms_upper_bound": p99 if representable else None,
        "loss": 1.0 - survives,
        "bandwidth_mbps": min(bandwidths) if bandwidths else None,
    }


def write_link_report(
    config, out_dir: str, stats: MessageStats, node_name: str = "client"
) -> Dict[str, object]:
    """Write ``link_report_<node>.json`` for one client and return its content."""
    routes = config.routes
    hop_names = {name for route in routes for name in route.hops}
    report = {
        "experiment": config.experiment.name,
        "app": config.app.plugin,
        # Whose measurement this is. With several clients the configured half is
        # shared and the measured half is not, so the record says which node.
        "node": node_name,
        # A hop is reported inside its route, not twice.
        "configured_links": [
            link_summary(link) for link in config.links if link.name not in hop_names
        ],
        "configured_routes": [route_summary(config, route) for route in routes],
        "measured": stats.as_dict(),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, report_name(node_name)), "w") as fh:
        json.dump(report, fh, indent=2)
    return report
