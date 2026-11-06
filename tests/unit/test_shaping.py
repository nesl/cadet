"""Tests for the packet-level shaping parameters and the distribution mapping.

Covers the two netem facts the mapping exists to respect: its jitter is a
uniform half-width (so a p50/p99 intent converts, rather than copying a standard
deviation across), and it reorders a jittered flow unless its output is
rate-serialised (so a jittered link without a bandwidth is refused).
"""

import math

import pytest

from cadet.config.schema import ConstantDist, LognormalDist, NormalDist, TraceDist
from cadet.network.shaping import (
    Z99,
    Shaping,
    UnrepresentableDelay,
    percentiles_ms,
    shaping_from_network,
)


class _Net:
    """The few fields of a link's ``network`` block the mapping reads."""

    def __init__(self, distribution, loss=0.0, bandwidth_mbps=None):
        self.distribution = distribution
        self.loss = loss
        self.bandwidth_mbps = bandwidth_mbps


# --- percentiles of a configured distribution ----------------------------
def test_constant_has_equal_percentiles():
    assert percentiles_ms(ConstantDist(value_ms=40.0)) == (40.0, 40.0)


def test_normal_p99_is_the_gaussian_tail():
    p50, p99 = percentiles_ms(NormalDist(mean_ms=50.0, std_ms=8.0))
    assert p50 == pytest.approx(50.0)
    assert p99 == pytest.approx(50.0 + Z99 * 8.0)


def test_lognormal_reports_the_percentiles_it_was_given():
    # The two fields are the two percentiles, so this is a read-back.
    p50, p99 = percentiles_ms(LognormalDist(p50_ms=30.0, p99_ms=100.0))
    assert (p50, p99) == (pytest.approx(30.0), pytest.approx(100.0))


def test_lognormal_rejects_a_tail_below_the_median():
    with pytest.raises(ValueError):
        LognormalDist(p50_ms=50.0, p99_ms=20.0)


# --- mapping a distribution onto netem -----------------------------------
def test_constant_needs_no_jitter_and_keeps_order():
    shaping = shaping_from_network(_Net(ConstantDist(value_ms=60.0)))
    assert shaping.delay_ms == 60.0
    assert shaping.jitter_ms is None
    assert shaping.preserves_order
    assert shaping.netem_args() == ["delay", "60.0ms"]


def test_jitter_is_the_uniform_half_width_matching_p99():
    net = _Net(NormalDist(mean_ms=50.0, std_ms=8.0), bandwidth_mbps=50.0)
    shaping = shaping_from_network(net)
    _p50, want_p99 = percentiles_ms(net.distribution)
    got_p50, got_p99 = shaping.achieved_percentiles_ms
    assert got_p50 == pytest.approx(50.0)
    assert got_p99 == pytest.approx(want_p99)


def test_jittered_link_emits_a_rate():
    shaping = shaping_from_network(
        _Net(NormalDist(mean_ms=50.0, std_ms=8.0), bandwidth_mbps=50.0)
    )
    args = shaping.netem_args()
    assert "rate" in args
    assert args[args.index("rate") + 1] == "50.0mbit"


def test_jitter_without_bandwidth_is_refused():
    with pytest.raises(UnrepresentableDelay) as err:
        shaping_from_network(_Net(NormalDist(mean_ms=50.0, std_ms=8.0)))
    assert "bandwidth_mbps" in str(err.value)


def test_spread_wider_than_the_delay_is_refused():
    # p50 30 / p99 100 needs a jitter of ~71 ms about a 30 ms delay.
    with pytest.raises(UnrepresentableDelay) as err:
        shaping_from_network(_Net(LognormalDist(p50_ms=30.0, p99_ms=100.0), bandwidth_mbps=100.0))
    assert "wider than the delay" in str(err.value)


def test_narrow_lognormal_is_representable():
    shaping = shaping_from_network(
        _Net(LognormalDist(p50_ms=30.0, p99_ms=55.0), bandwidth_mbps=100.0)
    )
    assert shaping.preserves_order
    p50, p99 = shaping.achieved_percentiles_ms
    assert p50 == pytest.approx(30.0)
    assert p99 == pytest.approx(55.0, rel=1e-2)


def test_trace_has_no_kernel_equivalent():
    with pytest.raises(UnrepresentableDelay) as err:
        shaping_from_network(_Net(TraceDist(path="/nonexistent.txt")))
    assert "mode: delay" in str(err.value)


def test_loss_becomes_a_percentage():
    shaping = shaping_from_network(_Net(ConstantDist(value_ms=10.0), loss=0.01))
    assert shaping.loss_pct == pytest.approx(1.0)
    assert "loss" in shaping.netem_args()


# --- the Shaping value object -------------------------------------------
def test_empty_shaping_installs_nothing():
    assert Shaping().is_empty
    assert Shaping().netem_args() == []
    assert Shaping().preserves_order


def test_unrated_jitter_does_not_claim_to_preserve_order():
    # Reachable only by constructing Shaping directly; the mapping refuses it.
    assert not Shaping(delay_ms=30.0, jitter_ms=15.0).preserves_order


def test_achieved_p99_uses_the_uniform_fraction():
    shaping = Shaping(delay_ms=30.0, jitter_ms=10.0, bandwidth_mbps=100.0)
    p50, p99 = shaping.achieved_percentiles_ms
    assert p50 == 30.0
    assert p99 == pytest.approx(30.0 + 0.98 * 10.0)
    assert not math.isnan(p99)
