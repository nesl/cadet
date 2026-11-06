"""Per-run clock offset: what a timestamp from another machine is worth.

What this is for:
    A run whose roles sit on two machines reads its timestamps from two clocks,
    so any interval computed across them carries the difference between those
    clocks. This module measures that difference once per run per node, with an
    uncertainty, and the orchestrator records both in ``run.json``. It does not
    correct anything: the per-frame and per-request records keep the clocks they
    were written with, and a reader who wants a cross-node interval corrected
    subtracts the recorded offset and carries the recorded uncertainty with it.
    That matters because the uncertainty is the same order as the offset - half
    a round trip on a local network is a tenth of a millisecond, and so is the
    offset - so an offset quoted without one invites a correction the
    measurement cannot resolve.

What it contains:
    - :data:`PROBE_COUNT_DEFAULT`, :data:`PROBE_WARMUP`: how many exchanges an
      estimate is made of, and why the first is discarded.
    - :class:`OffsetEstimate`: one estimate - offset, uncertainty, the method
      that produced it, and the round-trip spread behind it.
    - :func:`estimate_from_round_trips`: the round-trip estimator (the minimum
      round trip bounds the one-way error), against any callable that returns
      the other machine's clock.
    - :class:`TimeServiceReading` and :func:`parse_time_service`: what a
      machine's own time daemon reports about its offset and dispersion, read
      from ``chronyc tracking``, ``ntpq`` or ``timedatectl timesync-status``,
      and :func:`estimate_from_time_service`, which differences two of them.
    - :class:`SshClockSource` and :class:`LocalClockSource`: the two ways to
      ask a machine what time it is - over the SSH channel the placement
      already has, or on this host.
    - :func:`same_clock_estimate` for a node on this machine, and
      :func:`measure_nodes` / :func:`run_record`, which the orchestrator calls.

How it fits into CADET:
    :mod:`cadet.orchestrator.runner` calls :func:`measure_nodes` before the run
    starts and again after the roles have stopped, and :func:`run_record` turns
    the pair into the ``clock_offset`` block of ``run.json``. Measuring twice
    is what makes the offset's constancy over the run an observation rather
    than an assumption: the recorded uncertainty is widened to cover the drift
    between the two if the drift is the larger of the two. Nodes launched with
    ``local`` or ``docker`` run on this machine - a container shares its host's
    kernel clock - so their offset is zero by construction rather than by
    measurement, and the record says so.

How to configure it:
    Nothing happens unless ``metrics.collectors`` lists ``clock_offset``.
    ``metrics.clock_offset.method`` chooses the estimator (``auto`` measures
    round trips and records a time-service reading beside them as a
    cross-check), ``probes`` how many exchanges each estimate is made of, and
    ``timeout_s`` how long the orchestrator waits for the other machine.
"""

import math
import re
import shlex
import subprocess
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# How many exchanges one estimate is made of. The estimator keeps the one with
# the smallest round trip, so more probes can only narrow the bound; eleven is
# enough for the minimum to settle on a quiet local network and still costs
# milliseconds.
PROBE_COUNT_DEFAULT = 11

# Exchanges discarded before any are kept. The first one through a freshly
# opened channel pays for the remote interpreter starting up - measured at
# hundreds of milliseconds against a tenth of a millisecond for the rest - and
# an outlier that large is not a round trip anyone should reason about. It
# cannot bias the result (the minimum would reject it anyway) but it would
# dominate the reported spread.
PROBE_WARMUP = 1

# How long the orchestrator waits for the other machine before giving up.
PROBE_TIMEOUT_S_DEFAULT = 15.0

# The responder run on the other machine: it answers every line on its standard
# input with its own wall clock, so an exchange costs one round trip over an
# already-open channel rather than a new connection. ``repr`` is used because
# ``str`` of a float loses nothing in Python 3 but this is explicit about it,
# and the clock is the same ``time.time()`` the roles stamp their records with.
REMOTE_RESPONDER = (
    "import sys, time\n"
    "for _ in iter(sys.stdin.readline, ''):\n"
    "    sys.stdout.write(repr(time.time()) + '\\n')\n"
    "    sys.stdout.flush()\n"
)

# The commands a machine's own time daemon answers, in the order they are tried.
# Each is asked of whichever daemon is installed; a machine running none of them
# reports no reading, which is recorded rather than treated as an error.
TIME_SERVICE_COMMANDS = (
    ["chronyc", "-n", "tracking"],
    ["ntpq", "-c", "rv 0"],
    ["timedatectl", "timesync-status"],
)

# Method names recorded in run.json.
METHOD_ROUND_TRIP = "round_trip"
METHOD_TIME_SERVICE = "time_service"
METHOD_SAME_CLOCK = "same_clock"

# The sign convention, recorded in the run record so a reader does not have to
# guess which way to apply it.
OFFSET_NOTE = (
    "offset_s is the node's clock minus the orchestrator's clock, so a "
    "positive offset means the node's timestamps read ahead; recover an "
    "orchestrator-clock time as node_time - offset_s. uncertainty_s is a "
    "half-width, not a standard deviation: for the round-trip method it is "
    "half the smallest round trip observed, which bounds the error on the "
    "assumption only that neither direction of the exchange is faster than "
    "instantaneous. It is the same order as the offset itself on a local "
    "network, so a correction smaller than uncertainty_s is not resolved by "
    "this measurement. Nothing in the run has been corrected: every record "
    "keeps the clock it was written with."
)


class ClockOffsetError(RuntimeError):
    """Raised when an offset was asked for and could not be measured."""


class OffsetEstimate:
    """One clock-offset estimate, with the uncertainty and evidence behind it.

    ``offset_s`` is the measured clock minus the reference clock and
    ``uncertainty_s`` is a half-width on it (see :data:`OFFSET_NOTE`).
    ``method`` names what produced the number, which is the thing a reader
    needs most: a round trip and a time daemon are not equally precise, and on
    the machines this was built against they differ by an order of magnitude.
    """

    def __init__(
        self,
        method: str,
        offset_s: float,
        uncertainty_s: float,
        probes: int = 0,
        rtt_min_s: Optional[float] = None,
        rtt_max_s: Optional[float] = None,
        offset_spread_s: Optional[float] = None,
        detail: str = "",
        cross_check: Optional["OffsetEstimate"] = None,
    ):
        self.method = method
        self.offset_s = float(offset_s)
        self.uncertainty_s = abs(float(uncertainty_s))
        self.probes = int(probes)
        self.rtt_min_s = rtt_min_s
        self.rtt_max_s = rtt_max_s
        self.offset_spread_s = offset_spread_s
        self.detail = detail
        self.cross_check = cross_check

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "OffsetEstimate({}: {:+.6f}s +/- {:.6f}s)".format(
            self.method, self.offset_s, self.uncertainty_s
        )

    def agrees_with(self, other: "OffsetEstimate") -> bool:
        """True when two estimates' uncertainty intervals overlap.

        Two methods that disagree by more than the sum of their half-widths are
        not measuring the same thing, and that is worth recording next to the
        numbers rather than leaving to whoever reads them.
        """
        return abs(self.offset_s - other.offset_s) <= (
            self.uncertainty_s + other.uncertainty_s
        )

    def as_dict(self) -> Dict[str, object]:
        record: Dict[str, object] = {
            "method": self.method,
            "offset_s": self.offset_s,
            "uncertainty_s": self.uncertainty_s,
        }
        if self.probes:
            record["probes"] = self.probes
        for key, value in (
            ("rtt_min_s", self.rtt_min_s),
            ("rtt_max_s", self.rtt_max_s),
            ("offset_spread_s", self.offset_spread_s),
        ):
            if value is not None:
                record[key] = value
        if self.detail:
            record["detail"] = self.detail
        if self.cross_check is not None:
            record["cross_check"] = self.cross_check.as_dict()
            record["cross_check"]["agrees"] = self.agrees_with(self.cross_check)
        return record


def same_clock_estimate(reason: str) -> OffsetEstimate:
    """Return the zero offset of a node that shares the orchestrator's clock.

    A host process and a container on this machine both read this kernel's
    clock, so there is no offset to measure and nothing an estimate could add.
    It is recorded anyway, with its reason, so every node of a run has an entry
    and a reader never has to infer a missing one.
    """
    return OffsetEstimate(
        method=METHOD_SAME_CLOCK,
        offset_s=0.0,
        uncertainty_s=0.0,
        detail=reason,
    )


# ---------------------------------------------------------------------------
# the round-trip estimator
# ---------------------------------------------------------------------------
def estimate_from_round_trips(
    read_remote_time: Callable[[], float],
    probes: int = PROBE_COUNT_DEFAULT,
    clock: Callable[[], float] = time.time,
    warmup: int = PROBE_WARMUP,
) -> OffsetEstimate:
    """Estimate a clock offset from round trips to ``read_remote_time``.

    Each exchange reads this clock, asks the other machine for its clock, and
    reads this clock again. If the remote reading happened at the midpoint of
    the two local ones the offset is exactly their difference, and the error in
    that assumption is bounded by half the round trip - so the exchange with
    the smallest round trip gives both the estimate and its half-width. This is
    the standard estimator (Cristian's algorithm, and the NTP offset with one
    sample); the alternative of averaging all the exchanges is worse here,
    because a delayed exchange is delayed in one direction and biases the mean
    while leaving the minimum alone.

    ``clock`` is the local clock to compare against, injectable so the
    estimator can be tested against a fake pair of clocks. ``warmup``
    exchanges are made and discarded first; see :data:`PROBE_WARMUP`.
    """
    if probes < 1:
        raise ClockOffsetError("a clock offset needs at least one probe")
    samples: List[Tuple[float, float]] = []  # (round trip, offset)
    for index in range(probes + max(0, warmup)):
        before = clock()
        remote = read_remote_time()
        after = clock()
        if remote is None:
            raise ClockOffsetError("the other machine returned no clock reading")
        rtt = after - before
        if rtt < 0:
            raise ClockOffsetError(
                "the local clock went backwards during a probe ({:.9f}s); "
                "an offset measured against it would mean nothing".format(rtt)
            )
        if index < warmup:
            continue
        samples.append((rtt, float(remote) - (before + after) / 2.0))
    rtt_min, offset = min(samples, key=lambda sample: sample[0])
    rtts = [sample[0] for sample in samples]
    offsets = [sample[1] for sample in samples]
    return OffsetEstimate(
        method=METHOD_ROUND_TRIP,
        offset_s=offset,
        uncertainty_s=rtt_min / 2.0,
        probes=len(samples),
        rtt_min_s=rtt_min,
        rtt_max_s=max(rtts),
        offset_spread_s=max(offsets) - min(offsets),
        detail=(
            "minimum of {} exchanges over the orchestrator's own channel to the "
            "node's machine; uncertainty is half that round trip".format(len(samples))
        ),
    )


# ---------------------------------------------------------------------------
# the time-service estimator
# ---------------------------------------------------------------------------
class TimeServiceReading:
    """What one machine's time daemon says about its own clock.

    ``offset_s`` is normalised to *this machine's clock minus its reference*,
    whichever daemon reported it, because the three daemons do not agree on the
    sign. ``dispersion_s`` is the daemon's own bound on how far that can be
    out.
    """

    def __init__(self, source: str, offset_s: float, dispersion_s: float, detail: str = ""):
        self.source = source
        self.offset_s = float(offset_s)
        self.dispersion_s = abs(float(dispersion_s))
        self.detail = detail

    def as_dict(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "offset_s": self.offset_s,
            "dispersion_s": self.dispersion_s,
            "detail": self.detail,
        }


def _first_float(pattern: str, text: str) -> Optional[float]:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def parse_time_service(text: str) -> Optional[TimeServiceReading]:
    """Parse a time daemon's own report of its offset and dispersion.

    Understands the three reports the orchestrator asks for
    (:data:`TIME_SERVICE_COMMANDS`) and returns None for anything else,
    including the empty output of a machine that runs no time daemon - which is
    a thing to record, not an error.

    The sign conventions differ and are the only subtle part:

    * ``chronyc tracking`` says the system clock is so many seconds "fast of"
      or "slow of" NTP time, which is already this-clock-minus-reference;
    * ``ntpq`` and ``timedatectl`` report the NTP offset, which is the
      correction to *add* to this clock, i.e. reference-minus-this-clock, so it
      is negated here.

    Dispersion is taken as the daemon's own distance bound: chrony's and ntpq's
    root dispersion plus half the root delay, and ``timedatectl``'s root
    distance, which is that same quantity already combined.
    """
    if not text or not text.strip():
        return None
    # chronyc tracking
    chrony = re.search(
        r"System time\s*:\s*([0-9.eE+-]+)\s*seconds\s+(fast|slow)\s+of", text, re.IGNORECASE
    )
    if chrony:
        magnitude = float(chrony.group(1))
        offset = magnitude if chrony.group(2).lower() == "fast" else -magnitude
        dispersion = (_first_float(r"Root dispersion\s*:\s*([0-9.eE+-]+)", text) or 0.0) + (
            _first_float(r"Root delay\s*:\s*([0-9.eE+-]+)", text) or 0.0
        ) / 2.0
        return TimeServiceReading(
            "chrony", offset, dispersion, "chronyc tracking, seconds; sign as reported"
        )
    # ntpq -c "rv 0": milliseconds, NTP sign
    if re.search(r"\boffset=", text) and re.search(r"\brootdisp=", text):
        offset_ms = _first_float(r"\boffset=([0-9.eE+-]+)", text)
        if offset_ms is not None:
            dispersion_ms = (_first_float(r"\brootdisp=([0-9.eE+-]+)", text) or 0.0) + (
                _first_float(r"\brootdelay=([0-9.eE+-]+)", text) or 0.0
            ) / 2.0
            return TimeServiceReading(
                "ntpd",
                -offset_ms / 1e3,
                dispersion_ms / 1e3,
                "ntpq rv 0, milliseconds; NTP sign negated",
            )
    # timedatectl timesync-status: suffixed units, NTP sign
    if re.search(r"Root distance\s*:", text, re.IGNORECASE):
        offset = _parse_suffixed(r"Offset\s*:\s*([0-9.eE+-]+)\s*(s|ms|us|\u00b5s|ns)", text)
        distance = _parse_suffixed(
            r"Root distance\s*:\s*([0-9.eE+-]+)\s*(s|ms|us|\u00b5s|ns)", text
        )
        if offset is not None:
            return TimeServiceReading(
                "systemd-timesyncd",
                -offset,
                distance if distance is not None else 0.0,
                "timedatectl timesync-status; NTP sign negated, root distance "
                "as the dispersion",
            )
    return None


_UNIT_SECONDS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "\u00b5s": 1e-6, "ns": 1e-9}


def _parse_suffixed(pattern: str, text: str) -> Optional[float]:
    """Return a value whose unit is a suffix on the same line, in seconds."""
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except (TypeError, ValueError):
        return None
    return value * _UNIT_SECONDS.get(match.group(2).lower(), 1.0)


def estimate_from_time_service(
    local: Optional[TimeServiceReading], remote: Optional[TimeServiceReading]
) -> Optional[OffsetEstimate]:
    """Difference two time-daemon readings into one relative offset.

    Both machines' daemons measure themselves against a reference, so the
    difference of their offsets is the offset between them and the sum of their
    dispersions bounds it. Returns None unless both ends reported: one end's
    reading says nothing about the pair.

    This is the weaker of the two methods wherever the machines sit on the same
    network and poll a distant server: the dispersion each daemon reports is
    dominated by the path to *its* server, not by the link between the two
    machines, so the bound it gives can be an order of magnitude wider than
    half a local round trip - wide enough to be consistent with zero while the
    round trip resolves a real offset. It is kept because it is independent of
    the round trip's symmetry assumption, which makes it worth recording as a
    cross-check even when it is too wide to use as the estimate.
    """
    if local is None or remote is None:
        return None
    return OffsetEstimate(
        method=METHOD_TIME_SERVICE,
        offset_s=remote.offset_s - local.offset_s,
        uncertainty_s=remote.dispersion_s + local.dispersion_s,
        detail=(
            "difference of each machine's own time daemon reading "
            "({} here, {} there); uncertainty is the sum of the two "
            "dispersions, which bounds the path to each daemon's server rather "
            "than the link between the machines".format(local.source, remote.source)
        ),
    )


# ---------------------------------------------------------------------------
# asking a machine what time it is
# ---------------------------------------------------------------------------
class LocalClockSource:
    """This machine: the reference clock, and its own time daemon's report."""

    name = "orchestrator"

    def __init__(self, timeout_s: float = PROBE_TIMEOUT_S_DEFAULT):
        self.timeout_s = timeout_s

    def time_service(self) -> Optional[TimeServiceReading]:
        """Return this machine's time-daemon reading, or None if it runs none."""
        for argv in TIME_SERVICE_COMMANDS:
            try:
                result = subprocess.run(
                    list(argv), capture_output=True, text=True, timeout=self.timeout_s
                )
            except (OSError, subprocess.SubprocessError):
                continue
            reading = parse_time_service((result.stdout or "") + (result.stderr or ""))
            if reading is not None:
                return reading
        return None


class SshClockSource:
    """Another machine, reached over the SSH channel the placement already uses.

    Built from either launcher that reaches another machine - they share
    :class:`~cadet.core.ssh.SshTarget` - so the identity, port and host-key
    policy are the run's own and nothing new is configured, nothing listens and
    nothing is left on the other machine. The node's role either runs in a
    container that joins that machine's network and so its kernel, or is a
    process on that machine outright, which is why measuring the machine
    measures the node.

    The responder is kept open for the whole set of exchanges: a fresh SSH
    connection per probe would measure connection setup, which is hundreds of
    times the quantity being estimated.
    """

    def __init__(
        self,
        launcher,
        timeout_s: float = PROBE_TIMEOUT_S_DEFAULT,
        python: str = "python3",
    ):
        self._launcher = launcher
        self.timeout_s = timeout_s
        self.python = python
        self._process = None

    def __enter__(self) -> "SshClockSource":
        self.open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def open(self) -> None:
        """Start the remote responder and leave the channel open."""
        if self._process is not None:
            return
        argv = self._launcher.ssh_prefix() + [
            "{} -u -c {}".format(self.python, shlex.quote(REMOTE_RESPONDER))
        ]
        try:
            self._process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise ClockOffsetError("cannot open a channel to the node's machine: {}".format(exc))

    def read_remote_time(self) -> float:
        """One exchange: ask the other machine for its clock and return it."""
        if self._process is None:
            raise ClockOffsetError("the channel to the node's machine is not open")
        try:
            self._process.stdin.write("\n")
            self._process.stdin.flush()
            line = self._process.stdout.readline()
        except (OSError, ValueError) as exc:
            raise ClockOffsetError("the channel to the node's machine failed: {}".format(exc))
        if not line:
            raise ClockOffsetError(
                "the node's machine closed the channel without answering; it may "
                "have no {} on its path".format(self.python)
            )
        try:
            return float(line.strip())
        except ValueError:
            raise ClockOffsetError(
                "the node's machine answered with something that is not a clock "
                "reading: {!r}".format(line.strip()[:80])
            )

    def close(self) -> None:
        """Shut the responder down. Nothing is left running on either machine."""
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            process.wait(timeout=self.timeout_s)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            process.kill()
            process.wait(timeout=self.timeout_s)

    def time_service(self) -> Optional[TimeServiceReading]:
        """Return the other machine's time-daemon reading, or None if it runs none."""
        for argv in TIME_SERVICE_COMMANDS:
            code, output = self._launcher.run_remote(list(argv), timeout_s=self.timeout_s)
            if code != 0 and not output:
                continue
            reading = parse_time_service(output)
            if reading is not None:
                return reading
        return None


# ---------------------------------------------------------------------------
# what the orchestrator calls
# ---------------------------------------------------------------------------
def measure(
    source: SshClockSource,
    local: Optional[LocalClockSource] = None,
    method: str = "auto",
    probes: int = PROBE_COUNT_DEFAULT,
) -> OffsetEstimate:
    """Measure one machine's offset by ``method``.

    ``auto`` and ``round_trip`` both estimate from round trips; ``auto`` also
    reads both machines' time daemons and records that as a cross-check beside
    the estimate, so a reader can see whether two independent methods agree
    without the weaker one setting the number. ``time_service`` uses the
    daemons as the estimate and fails if either machine runs none, because an
    estimate that quietly fell back to a different method is the thing this
    record exists to prevent.

    The channel to the other machine is opened and closed here, so no caller
    can leave a responder running on it.
    """
    reference = local or LocalClockSource(timeout_s=source.timeout_s)
    if method == METHOD_TIME_SERVICE:
        estimate = estimate_from_time_service(reference.time_service(), source.time_service())
        if estimate is None:
            raise ClockOffsetError(
                "metrics.clock_offset.method is 'time_service' but at least one "
                "of the two machines reports no time daemon (chrony, ntpd or "
                "systemd-timesyncd); use 'round_trip' or 'auto'"
            )
        return estimate
    source.open()
    try:
        estimate = estimate_from_round_trips(source.read_remote_time, probes=probes)
    finally:
        source.close()
    if method != METHOD_ROUND_TRIP:
        estimate.cross_check = estimate_from_time_service(
            reference.time_service(), source.time_service()
        )
    return estimate


def measure_nodes(
    sources: Dict[str, Optional[SshClockSource]],
    local: Optional[LocalClockSource] = None,
    method: str = "auto",
    probes: int = PROBE_COUNT_DEFAULT,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, OffsetEstimate]:
    """Measure every node's offset. A ``None`` source means this machine's clock.

    A node that cannot be measured is recorded as a failure rather than
    omitted, and never raises: the offset is a property *of* the run, and
    losing it must not lose the run.
    """
    emit = log or (lambda _message: None)
    reference = local or LocalClockSource()
    out: Dict[str, OffsetEstimate] = {}
    for node, source in sources.items():
        if source is None:
            out[node] = same_clock_estimate(
                "this node runs on the orchestrator's machine, whose clock the "
                "offset is measured against; a container shares its host's clock"
            )
            continue
        try:
            out[node] = measure(source, local=reference, method=method, probes=probes)
        except ClockOffsetError as exc:
            emit("[clock] {}: no offset measured: {}".format(node, exc))
            out[node] = OffsetEstimate(
                method="unavailable", offset_s=float("nan"), uncertainty_s=float("inf"),
                detail=str(exc),
            )
    return out


def _mergeable(estimate: OffsetEstimate) -> bool:
    return not math.isnan(estimate.offset_s) and math.isfinite(estimate.uncertainty_s)


def run_record(phases: Sequence[Tuple[str, Dict[str, OffsetEstimate]]]) -> Dict[str, object]:
    """Build the ``clock_offset`` block of ``run.json`` from the measured phases.

    Each node gets the offset to use, the uncertainty to carry with it, and
    every measurement behind them. Where a node was measured more than once the
    offset is the mean and the uncertainty is **widened to cover the spread**
    when the clocks moved apart by more than the measurements' own half-widths:
    a clock that drifted during the run is not described by either end of the
    drift, and the uncertainty is the only field that can say so.
    """
    nodes: Dict[str, object] = {}
    names = []
    for _phase, measured in phases:
        for name in measured:
            if name not in names:
                names.append(name)
    for name in names:
        taken = [
            (phase, measured[name]) for phase, measured in phases if name in measured
        ]
        usable = [estimate for _phase, estimate in taken if _mergeable(estimate)]
        entry: Dict[str, object] = {
            "measurements": [
                dict(estimate.as_dict(), phase=phase) for phase, estimate in taken
            ]
        }
        if not usable:
            entry["method"] = "unavailable"
            entry["offset_s"] = None
            entry["uncertainty_s"] = None
            entry["detail"] = taken[-1][1].detail if taken else "not measured"
        else:
            offsets = [estimate.offset_s for estimate in usable]
            mean = sum(offsets) / len(offsets)
            half_spread = (max(offsets) - min(offsets)) / 2.0
            widest = max(estimate.uncertainty_s for estimate in usable)
            entry["method"] = usable[0].method
            entry["offset_s"] = mean
            entry["uncertainty_s"] = max(widest, half_spread)
            if half_spread > widest:
                entry["detail"] = (
                    "the offset moved by {:.9f}s between measurements, which is "
                    "more than their own uncertainty; the uncertainty reported "
                    "here covers that drift".format(max(offsets) - min(offsets))
                )
            if len(usable) > 1:
                entry["drift_s"] = offsets[-1] - offsets[0]
        nodes[name] = entry
    # The note is written once, beside the nodes rather than inside each of
    # them: it is the convention the whole block is read by.
    return {"reference": LocalClockSource.name, "note": OFFSET_NOTE, "nodes": nodes}
