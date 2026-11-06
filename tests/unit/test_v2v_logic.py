"""Unit tests for the V2V braking decision core (offline, no CARLA).

These pin the pure logic ported from ``carla_wrapper/v2v_braking_server.py``: the
monocular range estimate, the nearest-in-ROI lead selection, and the TTC +
distance brake gate. They are the stage-level parity check for V2V; the
recorded-run L1 check and the live run sit above it.
"""

from examples.v2v.app import (
    V2VBrakeGate,
    estimate_range_m,
    focal_length_pixels,
    nearest_lead_range_m,
)

TARGET_IDS = [2, 0]  # car, person
ROI_MIN, ROI_MAX = 300, 500
CONF = 0.30


def _gate(**kw):
    """Build a V2VBrakeGate with the original defaults, overriding only what a test needs."""
    base = dict(
        ttc_thresh_s=3.0, k_ttc_frames=2, alpha_ttc=0.3, d_thresh_m=24.0, d_gate_after_s=12.0
    )
    base.update(kw)
    return V2VBrakeGate(**base)


def test_focal_length_pixels():
    # 90 deg FOV, 800 px wide: f = (800/2) / tan(45 deg) = 400.
    assert abs(focal_length_pixels(800, 90.0) - 400.0) < 1e-6


def test_estimate_range_inverse_in_width():
    f, real = 400.0, 2.0
    near = estimate_range_m(200, f, real)  # wider box -> nearer
    far = estimate_range_m(50, f, real)
    assert near < far
    assert abs(estimate_range_m(100, f, real) - (f * real / 100)) < 1e-6
    # zero/negative width is clamped to 1 px (no divide-by-zero)
    assert estimate_range_m(0, f, real) == f * real


def test_nearest_lead_range_picks_nearest_in_roi_target():
    f = 400.0
    boxes = [
        (2, 0.9, 400, 100),  # car, in ROI, width 100 -> range 400*1.85/100 = 7.4
        (2, 0.9, 400, 200),  # car, in ROI, width 200 -> nearer (range 3.7)
        (2, 0.9, 700, 300),  # car, OUT of ROI -> ignored
        (5, 0.9, 400, 400),  # non-target class -> ignored
        (2, 0.1, 400, 500),  # below conf -> ignored
    ]
    z = nearest_lead_range_m(boxes, TARGET_IDS, ROI_MIN, ROI_MAX, CONF, f, 1.85)
    assert z is not None
    assert abs(z - (f * 1.85 / 200)) < 1e-6  # the width-200 box wins


def test_nearest_lead_range_none_when_no_target_in_roi():
    f = 400.0
    boxes = [(2, 0.9, 700, 200), (5, 0.9, 400, 200)]  # out of ROI / non-target
    assert nearest_lead_range_m(boxes, TARGET_IDS, ROI_MIN, ROI_MAX, CONF, f, 1.85) is None


def test_ttc_gate_brakes_after_k_confirmations():
    # Isolate the TTC gate: distance gate off (d_thresh=0 -> best_z<0 never true).
    g = _gate(d_thresh_m=0.0, d_gate_after_s=0.0)
    assert g.update(10.0, 0.0, 100.0) is False  # first frame: no prev, no decision
    assert g.update(5.0, 1.0, 100.0) is False   # ttc=1.0<3 -> counter 1 (<2)
    assert g.update(2.5, 2.0, 100.0) is True     # ttc stays <3 -> counter 2 -> brake


def test_ttc_counter_resets_when_not_confirmed():
    g = _gate(k_ttc_frames=3, alpha_ttc=1.0, d_thresh_m=0.0, d_gate_after_s=0.0)
    g.update(100.0, 0.0, 100.0)
    g.update(99.0, 1.0, 100.0)   # v_rel=1, ttc=99 -> above thresh, counter 0
    assert g.update(50.0, 2.0, 100.0) is False  # v_rel=49, ttc~1 -> counter 1 only


def test_distance_gate_brakes_when_close_and_closing_after_guard():
    # Isolate the distance gate: TTC off (ttc_thresh=0 -> ema<0 never).
    g = _gate(ttc_thresh_s=0.0)
    assert g.update(30.0, 0.0, 13.0) is False  # first frame, no prev
    assert g.update(20.0, 1.0, 13.0) is True    # within 24 m, closing, sim>12 -> brake


def test_distance_gate_held_until_sim_guard():
    g = _gate(ttc_thresh_s=0.0)
    g.update(30.0, 0.0, 5.0)
    assert g.update(20.0, 1.0, 5.0) is False  # close + closing but sim time 5 < 12 guard


def test_no_brake_when_not_closing():
    g = _gate(k_ttc_frames=1, d_gate_after_s=0.0)
    g.update(10.0, 0.0, 100.0)
    assert g.update(12.0, 1.0, 100.0) is False  # receding (range grows) -> v_rel=0, no brake


def test_none_frames_preserve_state_and_do_not_brake():
    g = _gate(alpha_ttc=1.0, d_thresh_m=0.0, d_gate_after_s=0.0)
    assert g.update(10.0, 0.0, 100.0) is False  # sets prev=(10,0)
    assert g.update(None, 1.0, 100.0) is False  # no lead this frame: no decision, no state change
    # next real frame measures closing against the pre-None prev (10 m @ t=0)
    assert g.update(5.0, 2.0, 100.0) is False   # ttc=5/((10-5)/2)=2.0<3 -> counter 1 (<2)
    assert g.update(2.5, 3.0, 100.0) is True     # counter 2 -> brake
