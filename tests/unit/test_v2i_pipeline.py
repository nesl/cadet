"""The V2I pipeline gate stages decide exactly as the flat V2I app's gates.

Core must not import examples, so ``cadet/pipeline/stages/v2i.py`` keeps its own copy
of the V2I decision logic (gates + ETA helpers), as V2V does. These tests feed the
stage gates and the flat app's gates (``examples/v2i/app.py``) the same per-frame
inputs and assert they brake on exactly the same frame - parity by construction, and a
guard against the two copies drifting apart. They also check the stage wiring: a gate
stage emits a ``BrakeDecision`` only on the brake frame and nothing afterwards.
"""

from cadet.pipeline.stages.braking import BrakeDecision
from cadet.pipeline.stages.v2i import (
    V2IEgoGateStage,
    V2IEgoObs,
    V2IFusionGateStage,
    V2IInfraObs,
)
from examples.v2i.app import CentroidTracker as FlatTracker
from examples.v2i.app import EgoConsecGate as FlatEgoGate
from examples.v2i.app import V2IFusionGate as FlatFusionGate
from examples.v2i.app import ped_eta_from_history as flat_ped_eta


def _first_brake_frame(decisions):
    """Index of the first True in a list of brake flags, or None."""
    return next((i for i, d in enumerate(decisions) if d), None)


def test_ego_gate_stage_matches_flat_ego_gate():
    # Ego approaches (Y decreasing past the -30 gate) with an intermittent detection
    # that becomes steady; both gates must brake on the same frame.
    seq = [
        (False, -10.0), (True, -20.0), (False, -25.0),  # before the Y gate
        (True, -31.0), (True, -33.0), (True, -35.0),    # past the gate, steady
        (True, -40.0),
    ]
    flat = FlatEgoGate(consecutive_n=2, ego_brake_gate_y=-30.0)
    stage = V2IEgoGateStage("ego", {"consecutive_n": 2, "ego_brake_gate_y": -30.0})

    flat_brakes, stage_brakes = [], []
    for i, (in_roi, ego_y) in enumerate(seq):
        flat_brakes.append(flat.update(in_roi, ego_y))
        out = stage.process(V2IEgoObs(in_roi=in_roi, ego_y=ego_y, frame_id=i))
        stage_brakes.append(isinstance(out, BrakeDecision))

    assert _first_brake_frame(flat_brakes) is not None
    assert _first_brake_frame(flat_brakes) == _first_brake_frame(stage_brakes)
    # Fires at most once.
    assert sum(stage_brakes) == 1


def test_ego_gate_stage_never_brakes_before_the_y_gate():
    # Steady detections but the ego never passes the Y gate -> no brake, both agree.
    seq = [(True, -5.0)] * 6
    flat = FlatEgoGate(consecutive_n=1, ego_brake_gate_y=-30.0)
    stage = V2IEgoGateStage("ego", {"consecutive_n": 1, "ego_brake_gate_y": -30.0})
    for i, (in_roi, ego_y) in enumerate(seq):
        assert flat.update(in_roi, ego_y) is False
        assert stage.process(V2IEgoObs(in_roi=in_roi, ego_y=ego_y, frame_id=i)) is None


def _ped_box(cy):
    """A detection tuple (label, conf, (x1,y1,x2,y2)) centred at image y=cy, x=1000."""
    return ("person", 0.9, (990, int(cy) - 20, 1010, int(cy) + 20))


def test_fusion_gate_stage_matches_flat_fusion_path():
    # A pedestrian walks down in image space (cy rising toward the band) while the ego
    # closes on the crossing (ego_eta falling). Drive the flat path (tracker + ped ETA
    # + fusion gate) and the stage and assert the same brake frame.
    params = dict(
        ttc_thresh_s=3.0, collision_window_s=2.0, prediction_horizon_s=7.0,
        min_ped_speed_pxps=1.5, tracker_max_miss=5, tracker_hist_len=12,
    )
    y_target = 350.0  # infra ROI cy midpoint (100..600)
    min_ped_speed, horizon = params["min_ped_speed_pxps"], params["prediction_horizon_s"]

    # frame i: (sim_time, ped cy, ego_eta)
    frames = []
    for i in range(10):
        frames.append((float(i) * 0.1, 200.0 + i * 20.0, 6.0 - i * 0.5))

    flat_tracker = FlatTracker(max_miss=5, hist_len=12)
    flat_gate = FlatFusionGate(3.0, 2.0, 7.0)
    stage = V2IFusionGateStage("fuse", params)

    flat_brakes, stage_brakes = [], []
    for i, (t, cy, ego_eta) in enumerate(frames):
        dets = [_ped_box(cy)]
        flat_tracker.update(dets, t_now=t)
        ped_eta = (
            flat_ped_eta(list(flat_tracker.hist), y_target, min_ped_speed, horizon)
            if flat_tracker.has_target else None
        )
        flat_brakes.append(flat_gate.update(ped_eta, ego_eta))

        msg = V2IInfraObs(dets=dets, ego_eta=ego_eta, y_target=y_target, frame_id=i)
        msg.sim_time = t
        out = stage.process(msg)
        stage_brakes.append(isinstance(out, BrakeDecision))

    assert _first_brake_frame(flat_brakes) is not None
    assert _first_brake_frame(flat_brakes) == _first_brake_frame(stage_brakes)
    assert sum(stage_brakes) == 1


def test_fusion_gate_stage_hard_ttc_brakes_without_ped():
    # Ego ETA alone inside the hard TTC threshold brakes even with no pedestrian seen.
    stage = V2IFusionGateStage("fuse", {"ttc_thresh_s": 3.0})
    msg = V2IInfraObs(dets=[], ego_eta=2.5, y_target=350.0, frame_id=0)
    msg.sim_time = 0.0
    out = stage.process(msg)
    assert isinstance(out, BrakeDecision) and out.brake
    # And only once.
    msg2 = V2IInfraObs(dets=[], ego_eta=1.0, y_target=350.0, frame_id=1)
    msg2.sim_time = 0.1
    assert stage.process(msg2) is None
