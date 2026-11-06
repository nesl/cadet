"""Offline check of the V2V leader drive + in-process intent shim (no CARLA).

Drives the V2V pipeline client extension with a fake source whose leader crosses the
brake threshold, and asserts: the intent is scheduled at v1_brake_time + v2v_delay_s,
poll_intent fires once (and only once) after that wall time, the lead vehicle is kept
(not destroyed), and the done/summary logic reports both cars. The flat V2VClientApp
shares the same _DecelController and scheduling, so this covers the mechanism both
forms use.
"""

from examples.v2v.app import V2VParams
from examples.v2v.pipeline_ext import V2VPipelineExtension


class _Loc:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class _FakeSource:
    """Minimal source: a leader that moves in -Y each drive() tick, and a follower."""

    def __init__(self, leader_y=0.0, follower_y=30.0, leader_speed=8.94):
        self._leader_y = leader_y
        self._follower_y = follower_y
        self._leader_speed = leader_speed
        self.cruise_calls = 0
        self.brake_calls = 0

    def obstacle_location(self):
        return _Loc(x=0.0, y=self._leader_y, z=0.0)

    def vehicle_location(self):
        return _Loc(x=0.0, y=self._follower_y, z=0.0)

    def obstacle_speed_mps(self):
        return self._leader_speed

    def set_obstacle_target_velocity(self, speed_mps):
        self.cruise_calls += 1

    def apply_obstacle_control(self, throttle, brake):
        self.brake_calls += 1
        # Once braking, the leader slows to a stop over a few ticks.
        self._leader_speed = max(0.0, self._leader_speed - 3.0)


class _Cfg:
    """Just enough of the resolved config shape for the extension's __init__."""

    class _App:
        params = {"extension_params": {}}

    app = _App()


def test_intent_scheduled_and_delivered_after_v2v_delay():
    ext = V2VPipelineExtension(_Cfg())
    assert ext.keep_obstacle() is True
    src = _FakeSource(leader_y=0.0)  # above the brake-y (-20): cruising
    ext.setup(src)

    # Leader cruising: no intent scheduled, no brake applied, cruise kick sent.
    ext.drive(src, sim_time=10.0, now=10.0)
    assert ext.poll_intent(10.0) is None
    assert ext._v2v_pending_time is None and src.cruise_calls == 1

    # Leader crosses the brake threshold at sim_time 12.0 -> schedule intent.
    src._leader_y = -25.0
    ext.drive(src, sim_time=12.0, now=12.0)
    assert ext._v1_brake_time == 12.0
    assert abs(ext._v2v_pending_time - (12.0 + V2VParams().v2v_delay_s)) < 1e-9

    # Before the delay elapses: no intent yet.
    assert ext.poll_intent(12.0 + 0.01) is None
    # After the delay: the intent fires exactly once.
    first = ext.poll_intent(12.0 + 0.05)
    assert first is not None and first.brake is True
    assert ext.poll_intent(12.0 + 0.10) is None  # not re-delivered


def test_perception_mode_schedules_no_intent():
    cfg = _Cfg()
    cfg.app.params = {"extension_params": {"braking_mode": "perception"}}
    ext = V2VPipelineExtension(cfg)
    ext.setup(_FakeSource(leader_y=-25.0))
    ext.drive(ext.source, sim_time=13.0, now=13.0)
    assert ext._v1_brake_time == 13.0           # the leader still brakes...
    assert ext._v2v_pending_time is None        # ...but no intent is scheduled
    assert ext.poll_intent(100.0) is None


def test_is_done_waits_for_both_cars():
    ext = V2VPipelineExtension(_Cfg())
    src = _FakeSource(leader_y=-25.0)
    ext.setup(src)
    ext.drive(src, sim_time=12.0, now=12.0)  # leader brakes at 12.0
    # follower not stopped yet -> not done
    assert ext.is_done(follower_done=False, sim_time=13.0) is False
    # follower stopped but leader not yet, within timeout -> not done
    assert ext.is_done(follower_done=True, sim_time=13.0) is False
    # drive until the leader stops
    for t in (12.5, 13.0, 13.5, 14.0):
        ext.drive(src, sim_time=t, now=t)
    assert ext._v1_stop_time is not None
    assert ext.is_done(follower_done=True, sim_time=14.5) is True
    # or: follower stopped and the post-brake timeout elapsed, leader never stopped
    ext2 = V2VPipelineExtension(_Cfg())
    src2 = _FakeSource(leader_y=-25.0, leader_speed=8.94)
    ext2.setup(src2)
    ext2.drive(src2, sim_time=12.0, now=12.0)
    src2._leader_speed = 8.94  # keep the leader rolling (never stops)
    assert ext2.is_done(follower_done=True, sim_time=12.0 + 7.0) is True
