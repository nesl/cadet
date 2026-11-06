"""Bandwidth collector: bytes per link and per direction, on the node itself.

What this is for:
    What the run actually put on each link, as opposed to what the link was
    configured to allow. ``link_report.json`` already records the configured
    parameters beside the measured per-message latency; this is the same idea for
    volume, sampled over the run so a rate can be read per interval rather than
    only as a total.

What it contains:
    - :class:`BandwidthCollector`: samples the transport's byte counters and
      writes one row per link per direction per sample, with the delta and the
      rate since the previous sample.
    - :func:`build_bandwidth_collector`.

How it fits into CADET:
    The counters live in :mod:`cadet.core.transport`, incremented by the
    transport that carried the bytes, so they count this node's own traffic and
    nothing else. Built by
    :func:`cadet.metrics.collectors.build_collectors` when
    ``metrics.collectors`` names ``bandwidth``; writes
    ``bandwidth_<node>.csv`` into the run directory.

    **What the numbers are.** Payload bytes handed to, or delivered by, the
    transport - not wire bytes. The 4-byte length prefix each message carries
    (:mod:`cadet.core.service`) is excluded, and so is everything TCP adds, so
    the framing overhead is derivable from ``messages_total`` rather than
    folded into the byte count. On the message-level ``delay`` backend a
    configured loss drops a message after it has been counted as sent, because
    the counter is the application's offered load; on the packet-level backends
    nothing is dropped at this layer at all.

How to configure it:
    ``metrics.bandwidth.interval_ms`` (default 100 ms; a byte counter only
    becomes a rate over a window - see
    :data:`cadet.metrics.collectors.COUNTER_SAMPLE_INTERVAL_MS`).
"""

from typing import Dict, List, Tuple

from cadet.core.transport import counter_snapshot
from cadet.metrics.collectors import COUNTER_SAMPLE_INTERVAL_MS, Collector

_BITS_PER_BYTE = 8
_MEGA = 1e6


class BandwidthCollector(Collector):
    """Samples per-link byte counters into ``bandwidth_<node>.csv``.

    ``snapshot`` and ``clock`` are injectable so the collector can be exercised
    against a fake counter source on a fake clock: what needs testing is the
    differencing and the rate, not the socket underneath nor how long a test
    slept.
    """

    NAME = "bandwidth"
    # The last row carries the cumulative byte and message counters, so it is
    # the row a reader takes totals from. Flushed per sample so those totals are
    # on disk even if this role is killed rather than stopped.
    FLUSH_EVERY_SAMPLE = True
    COLUMNS = (
        "link",
        "direction",
        "bytes_total",
        "bytes_delta",
        "messages_total",
        "messages_delta",
        "interval_s",
        "mbps",
    )

    def __init__(self, csv_path, interval_s, node="", logger=None, snapshot=None, clock=None):
        super().__init__(csv_path, interval_s, node=node, logger=logger)
        self._snapshot = snapshot or counter_snapshot
        self._clock = clock
        # (link, direction) -> (bytes, messages, t_monotonic) as last written.
        self._previous: Dict[Tuple[str, str], Tuple[int, int, float]] = {}

    def _now(self) -> float:
        if self._clock is not None:
            return self._clock()
        import time  # noqa: PLC0415 - kept local so the clock stays injectable

        return time.monotonic()

    def read(self) -> List[Dict[str, object]]:
        now = self._now()
        rows = []
        for counters in self._snapshot():
            for direction, byte_count, message_count in (
                ("sent", counters.bytes_sent, counters.messages_sent),
                ("recv", counters.bytes_recv, counters.messages_recv),
            ):
                rows.append(self._row(counters.label, direction, byte_count, message_count, now))
        return rows

    def _row(self, link, direction, byte_count, message_count, now) -> Dict[str, object]:
        """One link/direction row, differenced against the previous sample."""
        key = (link, direction)
        previous = self._previous.get(key)
        self._previous[key] = (byte_count, message_count, now)
        if previous is None:
            # The first sample of a link has no window to divide by. Its totals
            # are the traffic before the collector saw it, so they are reported
            # without a rate rather than charged to an interval of this length.
            bytes_delta = byte_count
            messages_delta = message_count
            interval = None
            mbps = None
        else:
            bytes_delta = byte_count - previous[0]
            messages_delta = message_count - previous[1]
            interval = now - previous[2]
            mbps = (
                round((bytes_delta * _BITS_PER_BYTE) / interval / _MEGA, 6)
                if interval > 0
                else None
            )
        return {
            "link": link,
            "direction": direction,
            "bytes_total": byte_count,
            "bytes_delta": bytes_delta,
            "messages_total": message_count,
            "messages_delta": messages_delta,
            "interval_s": None if interval is None else round(interval, 6),
            "mbps": mbps,
        }


def build_bandwidth_collector(config, node: str, csv_path: str, logger=None):
    """Build the bandwidth collector for one node from ``metrics.bandwidth``."""
    options = config.metrics.bandwidth
    interval_ms = getattr(options, "interval_ms", COUNTER_SAMPLE_INTERVAL_MS)
    return BandwidthCollector(csv_path, interval_ms / 1000.0, node=node, logger=logger)
