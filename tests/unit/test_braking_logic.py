"""Tests for the pure braking logic (no CARLA, no torch).

Covers: speed parsing, target-class id resolution, the ROI/conf brake trigger,
the brake controller math, and the pre-brake PID throttle.
"""

import pytest

from examples.braking.flat.app import (
    BrakeControllerState as BrakeController,
)
from examples.braking.flat.app import (
    frame_triggers_brake,
    pid_throttle,
    resolve_target_class_ids,
    target_speed_mps,
)


def test_target_speed_mps():
    assert target_speed_mps("40mph") == pytest.approx(40 * 0.44704)
    assert target_speed_mps("20kph") == pytest.approx(20 / 3.6)
    with pytest.raises(ValueError):
        target_speed_mps("40")


def test_resolve_target_class_ids():
    names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 7: "truck"}
    ids = resolve_target_class_ids(names, ["car", "motorcycle", "bicycle", "person"])
    assert set(ids) == {0, 1, 2, 3}
    assert 7 not in ids


def test_frame_triggers_brake():
    tids = [2]
    # in-class, in-ROI, high-conf -> True
    assert frame_triggers_brake([(2, 0.5, 400)], tids, 300, 500, 0.3) is True
    # low confidence -> False
    assert frame_triggers_brake([(2, 0.2, 400)], tids, 300, 500, 0.3) is False
    # outside ROI -> False
    assert frame_triggers_brake([(2, 0.9, 100)], tids, 300, 500, 0.3) is False
    # wrong class -> False
    assert frame_triggers_brake([(5, 0.9, 400)], tids, 300, 500, 0.3) is False


def test_brake_controller_ramps_and_bounds():
    c = BrakeController(target_decel=7.0, max_decel=7.3, kp=0.2, max_ramp=0.05)
    # First tick establishes baseline; decel measured from a speed drop.
    c.update_decel(16.0, 0.0)
    c.update_decel(15.0, 0.1)  # ~10 m/s^2 raw, filtered up
    b1 = c.step()
    assert 0.0 <= b1 <= 1.0
    # Command only moves by at most max_ramp per step.
    b0 = c.brake_cmd
    c.update_decel(14.5, 0.2)
    b2 = c.step()
    assert abs(b2 - b0) <= c.max_ramp + 1e-9


def test_brake_controller_backs_off_above_max_decel():
    c = BrakeController(target_decel=3.0, max_decel=3.0, kp=0.45, max_ramp=0.15)
    c.brake_cmd = 0.5
    c.filtered_decel = 5.0  # above max_decel -> should ease off
    out = c.step()
    assert out == pytest.approx(0.35)


def test_pid_throttle():
    # Far below target -> throttle proportional, capped at 1.
    assert pid_throttle(0.0, 20.0, kp=0.6) == pytest.approx(1.0)
    # Within deadband -> maintenance throttle.
    assert pid_throttle(19.5, 20.0, deadband=1.0) == 0.4
