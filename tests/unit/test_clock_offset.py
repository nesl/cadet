"""Tests for the per-run clock-offset estimator and the record it produces.

The estimator is tested against fake clocks rather than real machines, because
the property that matters is one a real pair of machines cannot demonstrate: a
known offset must come back as the measured one, and the uncertainty must be
wide enough to contain the truth whichever direction of the exchange was slow.
Both are checked by driving a fake remote clock whose offset is chosen here.

Covered: the round-trip estimator (recovery of a known offset, the bound on it,
asymmetric paths, the discarded warm-up exchange, a failing channel), the
parsers for the three time daemons and the absence of any of them, the
difference of two daemon readings, and the ``run.json`` block - including that
a drift between the run's two measurements widens the recorded uncertainty
rather than being averaged away.
"""

import math
import os

import pytest

from cadet.metrics import clock_offset

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REFERENCE_CONFIG = os.path.join(REPO, "examples", "reference", "synthetic.yaml")


class FakeRemote:
    """A remote clock ``offset_s`` ahead of a local clock we also control.

    ``forward_s`` and ``back_s`` are the one-way delays. The local clock
    advances by them as the exchange happens, which is what makes an
    asymmetric path testable: the estimator cannot see the split and the
    reported uncertainty has to cover it.
    """

    def __init__(self, offset_s, forward_s=0.0005, back_s=0.0005, start=1000.0):
        self.offset_s = offset_s
        self.forward_s = forward_s
        self.back_s = back_s
        self.now = start
        self.reads = 0

    def clock(self):
        return self.now

    def read_remote_time(self):
        self.reads += 1
        self.now += self.forward_s
        remote = self.now + self.offset_s
        self.now += self.back_s
        return remote


# --- the round-trip estimator --------------------------------------------
def test_a_known_offset_comes_back_on_a_symmetric_path():
    remote = FakeRemote(offset_s=0.000212)
    estimate = clock_offset.estimate_from_round_trips(
        remote.read_remote_time, probes=5, clock=remote.clock
    )
    assert estimate.method == "round_trip"
    assert estimate.offset_s == pytest.approx(0.000212, abs=1e-9)
    # Half the round trip, which on a symmetric path is the whole of the error
    # the estimator could have made.
    assert estimate.uncertainty_s == pytest.approx(0.0005, abs=1e-9)
    assert estimate.probes == 5


def test_the_uncertainty_contains_the_truth_on_an_asymmetric_path():
    # All the delay in one direction: the midpoint assumption is wrong by as
    # much as it can be, so this is the case the half-width has to survive.
    remote = FakeRemote(offset_s=0.0002, forward_s=0.001, back_s=0.0)
    estimate = clock_offset.estimate_from_round_trips(
        remote.read_remote_time, probes=5, clock=remote.clock
    )
    assert estimate.offset_s != pytest.approx(0.0002, abs=1e-6)
    assert abs(estimate.offset_s - 0.0002) <= estimate.uncertainty_s


def test_the_smallest_round_trip_sets_the_estimate_and_the_bound():
    # One quick exchange among slow ones: the minimum must be what is kept,
    # because a slow exchange is slow in one direction and biases the mean.
    calls = {"n": 0}
    state = {"now": 0.0}
    offset = 0.00025

    def clock():
        return state["now"]

    def read_remote_time():
        calls["n"] += 1
        rtt = 0.0001 if calls["n"] == 4 else 0.05
        state["now"] += rtt / 2.0
        remote = state["now"] + offset
        state["now"] += rtt / 2.0
        return remote

    estimate = clock_offset.estimate_from_round_trips(
        read_remote_time, probes=6, clock=clock, warmup=0
    )
    assert estimate.rtt_min_s == pytest.approx(0.0001, abs=1e-9)
    assert estimate.rtt_max_s == pytest.approx(0.05, abs=1e-9)
    assert estimate.uncertainty_s == pytest.approx(0.00005, abs=1e-9)
    assert estimate.offset_s == pytest.approx(offset, abs=1e-9)


def test_the_warmup_exchange_is_made_and_discarded():
    remote = FakeRemote(offset_s=0.0)
    estimate = clock_offset.estimate_from_round_trips(
        remote.read_remote_time, probes=3, clock=remote.clock, warmup=2
    )
    assert remote.reads == 5
    assert estimate.probes == 3


def test_a_channel_that_cannot_answer_is_an_error_not_a_zero():
    def broken():
        raise clock_offset.ClockOffsetError("channel closed")

    with pytest.raises(clock_offset.ClockOffsetError):
        clock_offset.estimate_from_round_trips(broken, probes=3)


def test_a_backwards_local_clock_is_refused():
    # An offset measured against a clock that jumped backwards mid-probe is
    # not an offset; better to report nothing than a number.
    state = {"now": 10.0}

    def clock():
        state["now"] -= 1.0
        return state["now"]

    with pytest.raises(clock_offset.ClockOffsetError):
        clock_offset.estimate_from_round_trips(lambda: 10.0, probes=2, clock=clock, warmup=0)


def test_zero_probes_is_refused():
    with pytest.raises(clock_offset.ClockOffsetError):
        clock_offset.estimate_from_round_trips(lambda: 1.0, probes=0)


# --- a node on this machine ----------------------------------------------
def test_a_node_on_this_machine_is_zero_by_construction():
    estimate = clock_offset.same_clock_estimate("shares this kernel's clock")
    assert estimate.method == "same_clock"
    assert estimate.offset_s == 0.0
    assert estimate.uncertainty_s == 0.0
    assert "clock" in estimate.detail


# --- the time daemons -----------------------------------------------------
CHRONY = """Reference ID    : 0A0A0A0A (ntp.example)
Stratum         : 3
System time     : 0.000123456 seconds fast of NTP time
Last offset     : +0.000001000 seconds
RMS offset      : 0.000002000 seconds
Root delay      : 0.002000000 seconds
Root dispersion : 0.000500000 seconds
"""

NTPQ = (
    "associd=0 status=0615 leap_none, sync_ntp, 1 event, clock_sync,\n"
    "version=\"ntpd 4.2.8p15\", processor=\"x86_64\",\n"
    "rootdelay=2.000, rootdisp=0.500, offset=-0.123456, sys_jitter=0.010\n"
)

TIMESYNCD = """       Server: 203.0.113.1 (ntp.example)
Poll interval: 34min 8s (min: 32s; max 34min 8s)
      Stratum: 2
    Precision: 1us (-25)
Root distance: 3.959ms (max: 5s)
       Offset: +1.655ms
        Delay: 130.266ms
       Jitter: 2.912ms
"""


def test_chrony_is_read_with_its_own_sign():
    reading = clock_offset.parse_time_service(CHRONY)
    assert reading.source == "chrony"
    # "fast of NTP time" means this clock is ahead of the reference.
    assert reading.offset_s == pytest.approx(0.000123456, abs=1e-12)
    # Root dispersion plus half the root delay.
    assert reading.dispersion_s == pytest.approx(0.0005 + 0.001, abs=1e-12)


def test_ntpq_has_the_opposite_sign_and_milliseconds():
    reading = clock_offset.parse_time_service(NTPQ)
    assert reading.source == "ntpd"
    # ntpd reports the correction to add, so a negative offset means this
    # clock is ahead.
    assert reading.offset_s == pytest.approx(0.000123456, abs=1e-12)
    assert reading.dispersion_s == pytest.approx((0.5 + 1.0) / 1e3, abs=1e-12)


def test_timesyncd_is_read_with_its_suffixed_units():
    reading = clock_offset.parse_time_service(TIMESYNCD)
    assert reading.source == "systemd-timesyncd"
    assert reading.offset_s == pytest.approx(-0.001655, abs=1e-12)
    assert reading.dispersion_s == pytest.approx(0.003959, abs=1e-12)


@pytest.mark.parametrize(
    "text", ["", "   \n", "Unit systemd-timesyncd.service could not be found."]
)
def test_an_absent_time_service_reads_as_nothing(text):
    # The case that matters most: a machine running no time daemon must be
    # recorded as unmeasured, never parsed into a zero offset.
    assert clock_offset.parse_time_service(text) is None


def test_two_daemon_readings_difference_into_a_relative_offset():
    local = clock_offset.TimeServiceReading("chrony", -0.001655, 0.003959)
    remote = clock_offset.TimeServiceReading("chrony", 0.002587, 0.003676)
    estimate = clock_offset.estimate_from_time_service(local, remote)
    assert estimate.method == "time_service"
    assert estimate.offset_s == pytest.approx(0.004242, abs=1e-9)
    assert estimate.uncertainty_s == pytest.approx(0.007635, abs=1e-9)
    # The point of recording the uncertainty: this estimate does not resolve
    # its own value, so it cannot be used as a correction.
    assert estimate.uncertainty_s > abs(estimate.offset_s)


def test_one_end_without_a_daemon_yields_no_relative_offset():
    reading = clock_offset.TimeServiceReading("chrony", 0.0, 0.001)
    assert clock_offset.estimate_from_time_service(reading, None) is None
    assert clock_offset.estimate_from_time_service(None, reading) is None
    assert clock_offset.estimate_from_time_service(None, None) is None


def test_the_time_service_method_refuses_rather_than_falling_back():
    class NoDaemon:
        timeout_s = 1.0

        def time_service(self):
            return None

    with pytest.raises(clock_offset.ClockOffsetError) as excinfo:
        clock_offset.measure(NoDaemon(), local=NoDaemon(), method="time_service")
    assert "time_service" in str(excinfo.value)


# --- agreement between methods -------------------------------------------
def test_estimates_agree_when_their_intervals_overlap():
    tight = clock_offset.OffsetEstimate("round_trip", -0.000212, 0.000061)
    wide = clock_offset.OffsetEstimate("time_service", 0.004242, 0.007635)
    assert tight.agrees_with(wide)
    assert wide.agrees_with(tight)
    far = clock_offset.OffsetEstimate("time_service", 0.05, 0.001)
    assert not tight.agrees_with(far)


# --- the run.json block ---------------------------------------------------
def _estimate(offset_s, uncertainty_s=0.00006):
    return clock_offset.OffsetEstimate("round_trip", offset_s, uncertainty_s, probes=11)


def _here():
    return clock_offset.same_clock_estimate("shares this kernel's clock")


def test_the_record_carries_both_measurements_and_the_convention():
    record = clock_offset.run_record(
        [
            (
                "pre_run",
                {"server": _estimate(-0.000212), "client": _here()},
            ),
            (
                "post_run",
                {"server": _estimate(-0.000216), "client": _here()},
            ),
        ]
    )
    assert record["reference"] == "orchestrator"
    assert "node_time - offset_s" in record["note"]
    server = record["nodes"]["server"]
    assert server["offset_s"] == pytest.approx(-0.000214, abs=1e-9)
    assert len(server["measurements"]) == 2
    assert [m["phase"] for m in server["measurements"]] == ["pre_run", "post_run"]
    assert server["drift_s"] == pytest.approx(-0.000004, abs=1e-9)
    # The uncertainty is unchanged here: the two measurements agree to well
    # inside their own half-widths.
    assert server["uncertainty_s"] == pytest.approx(0.00006, abs=1e-9)
    assert record["nodes"]["client"]["offset_s"] == 0.0


def test_a_drift_between_the_measurements_widens_the_uncertainty():
    # A clock that moved during the run is not described by either end of the
    # move, and the uncertainty is the only field that can say so.
    record = clock_offset.run_record(
        [
            ("pre_run", {"server": _estimate(0.0)}),
            ("post_run", {"server": _estimate(0.002)}),
        ]
    )
    server = record["nodes"]["server"]
    assert server["offset_s"] == pytest.approx(0.001, abs=1e-9)
    assert server["uncertainty_s"] == pytest.approx(0.001, abs=1e-9)
    assert "moved" in server["detail"]


def test_an_unmeasured_node_is_recorded_as_unmeasured_not_as_zero():
    unavailable = clock_offset.OffsetEstimate(
        "unavailable", float("nan"), float("inf"), detail="no python3 there"
    )
    record = clock_offset.run_record([("pre_run", {"server": unavailable})])
    server = record["nodes"]["server"]
    assert server["method"] == "unavailable"
    assert server["offset_s"] is None
    assert server["uncertainty_s"] is None
    assert "no python3" in server["detail"]


def test_a_cross_check_is_recorded_beside_the_estimate_with_whether_it_agrees():
    estimate = _estimate(-0.000212)
    estimate.cross_check = clock_offset.OffsetEstimate("time_service", 0.004242, 0.007635)
    entry = estimate.as_dict()
    assert entry["cross_check"]["method"] == "time_service"
    assert entry["cross_check"]["agrees"] is True


def test_measure_nodes_never_raises_and_records_the_failure():
    class Failing:
        timeout_s = 1.0

        def open(self):
            raise clock_offset.ClockOffsetError("no route")

        def close(self):
            pass

        def time_service(self):
            return None

    measured = clock_offset.measure_nodes({"server": Failing(), "client": None})
    assert math.isnan(measured["server"].offset_s)
    assert measured["server"].method == "unavailable"
    assert measured["client"].method == "same_clock"


# --- what the rest of the system does with it -----------------------------
def test_asking_for_it_starts_no_sampling_collector():
    # It is in metrics.collectors with the samplers, but there is nothing for a
    # role to sample: the role must skip it knowingly rather than build a
    # collector or warn about an unknown name.
    from cadet.config.loader import load_run_configs
    from cadet.metrics.collectors import build_collectors

    runs, _base = load_run_configs(str(REFERENCE_CONFIG), ["metrics.collectors=[clock_offset]"])
    lines = []
    built = build_collectors(runs[0], "client", str(REPO), logger=lines.append)
    assert len(built) == 0
    assert lines == []


def test_the_options_default_to_the_round_trip_with_a_cross_check():
    from cadet.config.schema import MetricsConfig

    options = MetricsConfig().clock_offset
    assert options.method == "auto"
    assert options.probes == clock_offset.PROBE_COUNT_DEFAULT
    # And nothing is measured at all unless the collector is asked for.
    assert MetricsConfig().collectors == []


def test_an_unasked_run_records_no_offset_at_all():
    # A run that did not measure its clocks must have no clock_offset field,
    # rather than one that could be read as a measured zero.
    from cadet.config.loader import load_run_configs
    from cadet.orchestrator.runner import Orchestrator

    runs, _base = load_run_configs(str(REFERENCE_CONFIG), [])
    orch = Orchestrator(log=lambda _m: None)
    assert orch._measure_clock_offsets(runs[0], _StubPlacement(), "pre_run") is None


def test_every_node_of_a_single_machine_run_is_recorded_as_same_clock():
    from cadet.config.loader import load_run_configs
    from cadet.orchestrator.runner import Orchestrator

    runs, _base = load_run_configs(str(REFERENCE_CONFIG), ["metrics.collectors=[clock_offset]"])
    config = runs[0]
    orch = Orchestrator(log=lambda _m: None)
    measured = orch._measure_clock_offsets(config, _StubPlacement(), "pre_run")
    assert sorted(measured) == sorted(node.name for node in config.nodes)
    assert all(estimate.method == "same_clock" for estimate in measured.values())


class _StubPlacement:
    """A placement with nothing on another machine, which is the common case."""

    def remote_nodes(self):
        return []

    def venv_nodes(self):
        return []

    def offmachine_nodes(self):
        """What the measurement asks: every node not on this machine, by any
        launcher. A node in a venv elsewhere keeps that machine's clock as much
        as one in a container there does."""
        return self.remote_nodes() + self.venv_nodes()

    def launcher_for(self, node):  # pragma: no cover - never reached
        raise AssertionError("a local node must not be probed")
