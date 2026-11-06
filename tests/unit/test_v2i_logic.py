"""Unit tests for the V2I decision core (pure; no CARLA).

Covers the ported pieces: ROI membership, ego ETA to the crossing, image-space ped
ETA, the centroid tracker, the ego consecutive-detection gate, and the infra fusion
gate. These are the decision functions the RSU server app and the pipeline form reuse.
"""

from examples.v2i.app import (
    CentroidTracker,
    EgoConsecGate,
    RoiRect,
    V2IFusionGate,
    V2IParams,
    center_in_roi,
    ped_eta_from_history,
    ttc_to_crossing,
)


def test_params_defaults_are_the_active_values():
    p = V2IParams()
    assert p.conf_thresh == 0.40                 # not the server getattr fallback 0.30
    assert (p.ego_roi.cx_min, p.ego_roi.cx_max) == (150, 700)
    assert (p.infra_roi.cx_min, p.infra_roi.cx_max) == (850, 1150)
    assert p.consecutive_n == 1                  # use_consecutive False -> 1
    assert p.prediction_horizon_s == 7.0
    assert p.collision_window_s == 2.0
    assert p.min_ped_speed_pxps == 1.5
    assert p.target_classes == ["person"]


def test_center_in_roi_rect_and_xband():
    rect = RoiRect(cx_min=150, cx_max=700, cy_min=50, cy_max=700)
    assert center_in_roi(400, 300, rect) is True
    assert center_in_roi(400, 800, rect) is False   # cy outside
    assert center_in_roi(100, 300, rect) is False   # cx outside
    xband = RoiRect(cx_min=300, cx_max=500)          # no cy bounds -> x-band only
    assert center_in_roi(400, 9999, xband) is True


def test_ttc_to_crossing():
    # 10 m away at 5 m/s -> 2 s; crawling -> None.
    assert abs(ttc_to_crossing(0.0, 0.0, 5.0, 0.0, 10.0) - 2.0) < 1e-9
    assert ttc_to_crossing(0.0, 0.0, 0.0, 0.0, 10.0) is None


def test_ped_eta_from_history():
    # centroid y moving 100 -> 160 over 0.3 s (200 px/s) toward y_target=220 (60 px away)
    hist = [(0.0, 900, 100.0), (0.15, 900, 130.0), (0.3, 900, 160.0)]
    eta = ped_eta_from_history(hist, y_target=220.0, min_speed_pxps=1.5, horizon_s=7.0)
    assert eta is not None and abs(eta - 0.3) < 1e-6   # 60 px / 200 px/s
    # fewer than 3 samples -> None
    assert ped_eta_from_history(hist[:2], 220.0, 1.5, 7.0) is None
    # moving away (target behind) -> None
    assert ped_eta_from_history(hist, y_target=50.0, min_speed_pxps=1.5, horizon_s=7.0) is None
    # too slow -> None
    slow = [(0.0, 900, 100.0), (1.0, 900, 100.5), (2.0, 900, 101.0)]
    assert ped_eta_from_history(slow, 220.0, 1.5, 7.0) is None


def test_centroid_tracker_track_and_drop():
    tr = CentroidTracker(max_miss=2, hist_len=5)
    det = [("person", 0.9, (380, 200, 420, 360))]  # centre (400, 280)
    assert tr.update(det, t_now=0.0) is True
    assert tr.has_target and abs(tr.cx - 400) < 1e-6
    # a second detection EMA-smooths toward the new centre
    tr.update([("person", 0.9, (480, 200, 520, 360))], t_now=0.1)  # centre 500
    assert 400 < tr.cx < 500
    # misses beyond max_miss drop the target
    tr.update([], t_now=0.2)
    tr.update([], t_now=0.3)
    assert tr.has_target is True       # still within max_miss
    tr.update([], t_now=0.4)
    assert tr.has_target is False and len(tr.hist) == 0


def test_ego_consec_gate_requires_count_and_y_gate():
    g = EgoConsecGate(consecutive_n=3, ego_brake_gate_y=-30.0)
    assert g.update(True, ego_y=-40.0) is False   # consec 1
    assert g.update(True, ego_y=-40.0) is False   # consec 2
    # consec 3 reached but ego not past the gate -> no brake
    assert g.update(True, ego_y=-20.0) is False
    # past the gate with consec >= 3 -> brake, once only
    assert g.update(True, ego_y=-40.0) is True
    assert g.update(True, ego_y=-40.0) is False


def test_ego_consec_gate_resets_on_miss():
    g = EgoConsecGate(consecutive_n=2, ego_brake_gate_y=-30.0)
    assert g.update(True, -40.0) is False
    assert g.update(False, -40.0) is False   # reset
    assert g.update(True, -40.0) is False    # consec 1 again
    assert g.update(True, -40.0) is True     # consec 2 -> brake


def test_fusion_gate_hard_ttc_brakes_without_ped():
    g = V2IFusionGate(ttc_thresh_s=3.0, collision_window_s=2.0, prediction_horizon_s=7.0)
    assert g.update(ped_eta_s=None, ego_eta_s=2.5) is True   # ego inside hard TTC
    assert g.update(None, 2.5) is False                      # fires once only


def test_fusion_gate_collision_window():
    g = V2IFusionGate(ttc_thresh_s=1.0, collision_window_s=1.0, prediction_horizon_s=7.0)
    # neither inside hard TTC (1.0), but within the collision window (|4.0-4.5|=0.5)
    assert g.update(ped_eta_s=4.0, ego_eta_s=4.5) is True


def test_fusion_gate_ignores_beyond_horizon_and_wide_gap():
    g = V2IFusionGate(ttc_thresh_s=1.0, collision_window_s=0.5, prediction_horizon_s=7.0)
    assert g.update(ped_eta_s=10.0, ego_eta_s=4.0) is False   # ped beyond horizon
    assert g.update(ped_eta_s=6.0, ego_eta_s=4.0) is False    # |6-4|=2 > window, neither < TTC
