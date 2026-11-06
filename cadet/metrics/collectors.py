"""System-metric collectors: sampling a node while the run happens on it.

What this is for:
    The paper reports CPU and GPU utilization, memory, per-link bandwidth and
    energy per inference. Each of those is a property of one node, measured on
    that node, so a collector runs inside the role process on the node it
    measures, writes its own per-node CSV into the run directory, and stamps
    every sample with the clocks the frame and inference logs already use. Energy
    can then be attributed to an inference without a second alignment step.

What it contains:
    - :data:`ORIGINAL_SAMPLE_INTERVAL_MS` and :data:`COUNTER_SAMPLE_INTERVAL_MS`:
      the two sampling-interval defaults, and where each one comes from.
    - :class:`CollectorError`: a collector that was asked for and cannot read.
    - :class:`Collector`: the sampling thread, the CSV writer and the two clock
      stamps every collector shares. A subclass supplies ``NAME``, ``COLUMNS``
      and :meth:`Collector.read`.
    - :class:`CollectorSet`: the collectors one role process runs, started and
      stopped together.
    - :func:`build_collectors`: the config-to-collectors step the roles call, and
      :func:`csv_path_for`, the per-node output name.
    - :data:`ALWAYS_ON` and :data:`ORCHESTRATOR_MEASURED`: the collector names
      that are not sampling loops here - one written by the server runtime
      regardless, one measured by the orchestrator
      (:mod:`cadet.metrics.clock_offset`).

How it fits into CADET:
    :mod:`cadet.roles.server` and :mod:`cadet.roles.client` build a
    :class:`CollectorSet` for their own node and start it around the run loop.
    **Nothing is enabled unless ``metrics.collectors`` names it**, so an
    unconfigured run behaves exactly as it did before this module existed; that
    default is pinned in ``tests/unit/test_collectors.py``.

How to configure it:
    ``metrics.collectors`` lists which collectors run on every node of the run.
    ``metrics.gpu``, ``metrics.cpu``, ``metrics.energy`` and
    ``metrics.bandwidth`` carry each one's options (sampling interval, reader
    source, device selection).
"""

import csv
import os
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

# The original power sampler (``energy_monitoring/profile_pipeline.sh``) ran
#   nvidia-smi --query-gpu=timestamp,power.draw --format=csv -lms 10
# so 10 ms is the interval every GPU/power reading here inherits. It is the
# original's number, not a chosen one.
ORIGINAL_SAMPLE_INTERVAL_MS = 10

# Host and process counters have no original to inherit from. 100 ms is used
# instead because both are rates over a window rather than instantaneous
# readings: psutil divides CPU time by the gap between two calls, and a byte
# counter only becomes a bandwidth over one. Below roughly this, the quotient is
# dominated by the sampling jitter rather than by the load being measured.
COUNTER_SAMPLE_INTERVAL_MS = 100

# How many rows are buffered before the CSV is flushed, and how long a row may
# wait to be flushed. A collector's process can be SIGKILLed (the orchestrator
# escalates after its grace) or die with its container, and what is still in the
# buffer is then lost; a row count alone bounds that loss in *rows*, which at a
# 100 ms interval is two and a half seconds of record. The time bound is what
# makes the loss on a hard kill a known quantity rather than a function of the
# sampling rate, and it costs one flush a second at any rate below _FLUSH_EVERY
# rows per second.
_FLUSH_EVERY = 100
_FLUSH_AFTER_S = 1.0

# Collectors named in the config that are not sampled here. ``queue`` is the
# server's per-request record, which is written by the server runtime for every
# run whether or not it is listed; it has no sampling loop to start.
ALWAYS_ON = ("queue",)

# Collectors named in the config that are measured by the orchestrator instead
# of by a role. ``clock_offset`` is the difference between two machines' clocks,
# which no single node can see: the orchestrator measures it and records it in
# run.json (:mod:`cadet.metrics.clock_offset`). A role skips it knowingly
# rather than falling through the "no such collector" branch below.
ORCHESTRATOR_MEASURED = ("clock_offset",)

# The clock columns every row carries, ahead of the collector's own.
#
# Two clocks, deliberately. ``t_monotonic`` is the clock the server's per-request
# records and the transport's delay lines use, so an inference window joins to a
# power sample exactly and in-process. ``t_wall`` is the clock the per-frame log
# and the client's ``brake_reaction_latency_s`` use, and the only one that is
# comparable across nodes. Writing both is what lets one CSV answer to either
# log without the reader having to know which clock produced it.
CLOCK_COLUMNS = ("t_wall", "t_monotonic", "node")


# Written beside a collector's CSV when the collector had nothing to measure on
# this node, holding the reason as its only line. A separate file rather than a
# row in the CSV: every reader of these files expects samples, and a row that
# is not a sample would have to be excluded by each of them.
NOT_APPLICABLE_SUFFIX = ".not_applicable"


class CollectorError(RuntimeError):
    """A collector the config asked for cannot read on this node.

    Raised from :meth:`Collector.start`, and deliberately not swallowed: a run
    that was told to measure energy and silently produced an empty file is worse
    than one that refuses, because the empty file is indistinguishable from a
    node that drew no power.
    """


class Collector:
    """A sampling loop that writes one per-node CSV.

    A subclass declares :attr:`NAME` (the collector name in the config and the
    CSV's filename stem), :attr:`COLUMNS` (its own columns, appended after
    :data:`CLOCK_COLUMNS`) and :meth:`read`, which returns zero or more rows per
    sample - zero or more because one sample of a two-GPU node is two rows, and
    one sample of a node with no open link is none.

    :meth:`available` reports whether this node can be read at all, and is
    checked by :meth:`start` before anything is written.
    """

    NAME = "collector"
    COLUMNS: Sequence[str] = ()

    # Whether every sample is flushed as it is written, rather than on the row
    # or time bound. True for the collectors whose **last row is authoritative**
    # - :mod:`~cadet.metrics.bandwidth`, whose last row carries the cumulative
    # byte counters, and :mod:`~cadet.metrics.energy`, whose last row carries the
    # run integral. For those two, a lost tail is not a gap in a series of
    # independent readings but an **undercount of a total**, and a role can die
    # without running :meth:`stop` at all - killed after its grace, or with its
    # container. That is how the two ends of a link came to disagree on totals
    # across two machines while agreeing on every message. One flush per sample
    # of one short row is a cheap price for a total that is right even then.
    FLUSH_EVERY_SAMPLE = False

    def __init__(self, csv_path: str, interval_s: float, node: str = "", logger=None):
        self.csv_path = csv_path
        # A zero or negative interval would spin; the floor is one millisecond,
        # which is already finer than any reader here resolves.
        self.interval_s = max(0.001, float(interval_s))
        self.node = node
        self.log = logger or (lambda message: None)
        self.rows_written = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        self._writer = None
        self._since_flush = 0
        self._last_flush = 0.0

    # -- subclass contract ---------------------------------------------------
    def available(self) -> Tuple[bool, str]:
        """Return ``(can_read, reason)`` for this node. Default: yes."""
        return True, ""

    def applicable(self) -> Tuple[bool, str]:
        """Return ``(has something to measure, reason)``. Default: yes.

        Distinct from :meth:`available`, and the distinction is the point. "I
        cannot read the thing" is a failure: the node was meant to produce this
        measurement and cannot, so the run stops rather than reporting a gap as
        a zero. "There is nothing here to read" is not a failure: a node with no
        GPU has no GPU figures, and a run that asked every node for GPU power
        should not fall over on the one that was never going to have any.

        A collector that is not applicable writes its file with the header and
        no samples, and records why beside it, so the absence is stated rather
        than looking like a collector that silently wrote nothing.
        """
        return True, ""

    def setup(self) -> None:
        """Open whatever the reader needs, after :meth:`available` passed."""

    def read(self) -> List[Dict[str, object]]:
        """Return this sample's rows, keyed by :attr:`COLUMNS`."""
        raise NotImplementedError

    def teardown(self) -> None:
        """Release the reader. Called once, after the sampling thread stops."""

    # -- lifecycle -----------------------------------------------------------
    @property
    def fieldnames(self) -> List[str]:
        """The CSV header: the shared clock columns, then the collector's own."""
        return list(CLOCK_COLUMNS) + list(self.COLUMNS)

    def start(self) -> None:
        """Check the reader, write the header and start sampling."""
        applies, why = self.applicable()
        if not applies:
            self._write_not_applicable(why)
            return
        ok, reason = self.available()
        if not ok:
            raise CollectorError(
                "metrics collector {!r} cannot read on node {!r}: {}".format(
                    self.NAME, self.node, reason
                )
            )
        self.setup()
        directory = os.path.dirname(self.csv_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._handle = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames)
        self._writer.writeheader()
        self._handle.flush()
        self._last_flush = time.monotonic()
        self._thread = threading.Thread(
            target=self._loop, name="collector-{}".format(self.NAME), daemon=True
        )
        self._thread.start()
        self.log(
            "[metrics] {} every {:.0f} ms -> {}".format(
                self.NAME, self.interval_s * 1000.0, self.csv_path
            )
        )

    def _write_not_applicable(self, reason: str) -> None:
        """Write the header-only CSV and the marker saying why it is empty.

        The CSV is still written: a reader looking for this node's file finds a
        well-formed one rather than a missing path, and "asked for, nothing to
        measure" stays distinguishable from "never asked for".
        """
        directory = os.path.dirname(self.csv_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.csv_path, "w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self.fieldnames).writeheader()
        with open(self.csv_path + NOT_APPLICABLE_SUFFIX, "w") as handle:
            handle.write(reason + "\n")
        self.log(
            "[metrics] {} not applicable on node {!r}: {}".format(
                self.NAME, self.node, reason
            )
        )

    def stop(self) -> None:
        """Stop sampling, close the final window, and close the CSV.

        The final sample is the point of this method as much as the stopping is.
        A collector stops mid-interval - the run ends when it ends, not on a
        sampling boundary - so without it the record stops at the last whole
        interval and everything after it is missing. For :mod:`~cadet.metrics.cpu`
        and :mod:`~cadet.metrics.gpu` that costs one reading. For
        :mod:`~cadet.metrics.energy` and :mod:`~cadet.metrics.bandwidth` it is an
        **undercount of a total**, because their last row carries the run
        integral and the cumulative byte counters: the energy of that final
        partial window, and every byte sent in it, would be absent from the
        totals a reader takes off the file. Two nodes that stop at different
        moments then disagree about how much they exchanged while agreeing about
        every message - which is exactly how this was found, across two machines.
        """
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_s * 10))
        self._thread = None
        self._sample()
        try:
            self.teardown()
        finally:
            if self._handle is not None:
                self._flush_if_due(force=True)
                self._handle.close()
                self._handle = None
        self.log(
            "[metrics] {} wrote {} samples -> {}".format(
                self.NAME, self.rows_written, self.csv_path
            )
        )

    def __enter__(self) -> "Collector":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    # -- the loop ------------------------------------------------------------
    def _loop(self) -> None:
        """Sample on a fixed schedule, resynchronising if a read overruns it."""
        base = time.monotonic()
        taken = 0
        while not self._stop.is_set():
            self._sample()
            taken += 1
            wait = (base + taken * self.interval_s) - time.monotonic()
            if wait <= 0:
                # A read took longer than the interval. Rebasing rather than
                # catching up keeps the loop from busy-spinning through the
                # samples it already missed, which would make the record denser
                # exactly where the node was busiest.
                base = time.monotonic()
                taken = 0
                continue
            self._stop.wait(wait)

    def _sample(self) -> None:
        """Stamp the clocks, read once, and write the rows."""
        if self._writer is None:
            return
        t_wall = time.time()
        t_monotonic = time.monotonic()
        try:
            rows = self.read()
        except Exception as exc:  # noqa: BLE001 - a bad read must not end the run
            self.log("[metrics] {} read failed (ignored): {}".format(self.NAME, exc))
            return
        for row in rows:
            record = {"t_wall": round(t_wall, 6), "t_monotonic": round(t_monotonic, 6)}
            record["node"] = self.node
            record.update(row)
            self._writer.writerow(record)
            self.rows_written += 1
            self._since_flush += 1
        self._flush_if_due()

    def _flush_if_due(self, force: bool = False) -> None:
        """Flush when enough rows, or enough time, has passed - or when forced."""
        if self._handle is None or not self._since_flush:
            return
        due = (
            force
            or self.FLUSH_EVERY_SAMPLE
            or self._since_flush >= _FLUSH_EVERY
            or (time.monotonic() - self._last_flush) >= _FLUSH_AFTER_S
        )
        if not due:
            return
        self._handle.flush()
        self._since_flush = 0
        self._last_flush = time.monotonic()

    def flush(self) -> None:
        """Write out whatever is buffered. Safe to call at any time."""
        self._flush_if_due(force=True)


class CollectorSet:
    """The collectors one role process runs, started and stopped together."""

    def __init__(self, collectors: Sequence[Collector] = (), logger=None):
        self._collectors: List[Collector] = list(collectors)
        self.log = logger or (lambda message: None)
        self._started: List[Collector] = []

    def __len__(self) -> int:
        return len(self._collectors)

    def names(self) -> List[str]:
        """The collector names in this set, in config order."""
        return [c.NAME for c in self._collectors]

    def add(self, collector: Collector) -> None:
        """Append a collector to the set (before it is started)."""
        self._collectors.append(collector)

    def get(self, name: str) -> Optional[Collector]:
        """Return the collector called ``name``, or ``None``.

        The server uses this to find the energy collector: per-inference energy
        is computed in the process that timed the inference, which is what keeps
        the two on one clock.
        """
        for collector in self._collectors:
            if collector.NAME == name:
                return collector
        return None

    def start(self) -> None:
        """Start every collector; stop the ones already started if one refuses."""
        try:
            for collector in self._collectors:
                collector.start()
                self._started.append(collector)
        except Exception:
            self.stop()
            raise

    def flush(self) -> None:
        """Flush every started collector's buffered rows."""
        for collector in self._started:
            try:
                collector.flush()
            except Exception as exc:  # noqa: BLE001 - a record must not fail a run
                self.log("[metrics] {} did not flush: {}".format(collector.NAME, exc))

    def stop(self) -> None:
        """Stop every started collector, in reverse order."""
        while self._started:
            collector = self._started.pop()
            try:
                collector.stop()
            except Exception as exc:  # noqa: BLE001 - runs in a run's finally
                self.log("[metrics] {} did not stop cleanly: {}".format(collector.NAME, exc))

    def __enter__(self) -> "CollectorSet":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()


def csv_path_for(out_dir: str, name: str, node: str) -> str:
    """Return ``<run_dir>/<collector>_<node>.csv``.

    One naming rule for every per-node file a run writes, matching
    ``client_outcome_<node>.json``, ``link_report_<node>.json`` and
    ``server_metrics_<node>.csv``, so the run directory stays readable by one
    rule rather than one per writer.
    """
    return os.path.join(out_dir, "{}_{}.csv".format(name, node or "node"))


def build_collectors(config, node: str, out_dir: str, logger=None) -> CollectorSet:
    """Build the collectors ``config`` asks for, for one node.

    Returns an empty set when ``metrics.collectors`` is empty, which is the
    default: a config that says nothing about metrics starts no threads, opens no
    readers and writes no extra files.

    The readers are imported here rather than at module import because each one
    pulls in something the framework does not otherwise need (psutil, NVML),
    and a config that asks for no collectors must not require any of it.
    """
    log = logger or (lambda message: None)
    requested = list(getattr(getattr(config, "metrics", None), "collectors", None) or [])
    collected = CollectorSet(logger=log)
    for name in requested:
        if name in ALWAYS_ON or name in ORCHESTRATOR_MEASURED:
            continue
        factory = _factory(name)
        if factory is None:
            # Unreachable for a validated config: the feature gate refuses a
            # collector with no implementation, and the two that have no
            # sampling loop are skipped above, before a role is ever started.
            log("[metrics] no collector named {!r}; skipped".format(name))
            continue
        collected.add(factory(config, node, csv_path_for(out_dir, name, node), log))
    return collected


def _factory(name: str):
    """Return the builder for one collector name, or ``None``."""
    if name == "gpu":
        from cadet.metrics.gpu import build_gpu_collector

        return build_gpu_collector
    if name == "cpu":
        from cadet.metrics.cpu import build_cpu_collector

        return build_cpu_collector
    if name == "energy":
        from cadet.metrics.energy import build_energy_collector

        return build_energy_collector
    if name == "bandwidth":
        from cadet.metrics.bandwidth import build_bandwidth_collector

        return build_bandwidth_collector
    return None
