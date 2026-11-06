"""V2V client-side extension for the pipeline driver (leader drive + intent shim).

What this is for:
    The pipeline driver (:class:`cadet.pipeline.app.PipelineClientApp`) runs the
    staged decision/control pipeline, but the V2V run also needs two things a staged
    pipeline cannot express as pure stage output: driving the scripted *leader*
    vehicle each tick, and the in-process leader->follower *intent* (the default V2V
    mechanism). The pipeline client loads this class through its generic
    ``client_extension`` hook (``app.params.client_extension``) and calls its
    duck-typed hooks; all the V2V specifics stay here in ``examples/``.

What it contains:
    - :class:`V2VPipelineExtension`: ``setup``/``poll_intent``/``drive``/
      ``keep_obstacle``/``is_done``/``summary`` - the same leader control and intent
      timing as the flat :class:`examples.v2v.app.V2VClientApp`, reusing its ported
      ``_DecelController`` so the two forms stay in parity.

How it fits into CADET:
    Configured by ``app.params.client_extension:
    examples.v2v.pipeline_ext:V2VPipelineExtension`` in the V2V pipeline config. The
    follower's own control runs through the shared ``brake_controller`` stage; this
    extension only adds the leader actuation and the intent delivery.
"""

from typing import Optional

from cadet.pipeline.stages.braking import BrakeDecision
from examples.v2v.app import V2VParams, _DecelController


class V2VPipelineExtension:
    """Drives the scripted leader and runs the in-process V2V intent for the pipeline."""

    def __init__(self, config):
        # V2V-specific params ride in app.params.extension_params (the pipeline
        # driver keeps its own app.params strict); empty = original defaults.
        self.params = V2VParams.model_validate(config.app.params.get("extension_params", {}))
        self._mode = self.params.braking_mode
        self._v2v_delay_s = self.params.v2v_delay_s
        self._leader_cruise_mps = self.params.leader_cruise_mps
        self._leader_brake_y = self.params.leader_brake_y
        self._leader_ctrl = _DecelController(**self.params.leader_controller.model_dump())
        self.source = None
        # Leader state + the intent schedule (wall-clock, as the original).
        self._v1_brake_time = None
        self._v1_stop_time = None
        self._v1_stop_loc = None
        self._v2v_emit_time = None
        self._v2v_pending_time = None
        self._v2v_sent = False
        self._v2v_recv_time = None
        self._brake_latched = False
        self._min_gap = None
        self._loop_ticks = 0
        self._last_sim_time = 0.0
        self._post_brake_timeout_s = 6.0

    def setup(self, source) -> None:
        self.source = source

    def keep_obstacle(self) -> bool:
        """V2V keeps the lead vehicle alive (the follower reacts to it, not past it)."""
        return True

    def poll_intent(self, sim_time: float) -> Optional[BrakeDecision]:
        """Deliver the leader's intent once v2v_delay_s has elapsed (v2v mode only)."""
        if (self._mode == "v2v" and self._v2v_pending_time is not None
                and not self._v2v_sent and not self._brake_latched
                and sim_time >= self._v2v_pending_time):
            self._v2v_sent = True
            self._v2v_recv_time = sim_time
            # frame_id None: an intent is not tied to a detected frame.
            return BrakeDecision(brake=True, server_processing_time=0.0, frame_id=None)
        return None

    def drive(self, source, sim_time: float, now: float) -> None:
        """Cruise the leader until it passes leader_brake_y, then brake it; schedule intent."""
        self._loop_ticks += 1
        self._last_sim_time = sim_time
        # Per-tick telemetry matching the original's _telemetry_fields (fresh leader
        # velocity + leader location + ego location, every loop), so the pipeline's
        # per-tick load - and the loop rate and min gap - match the original.
        v1loc = source.obstacle_location()
        if v1loc is not None:
            source.obstacle_speed_mps()  # leader get_velocity (fresh, as the original)
            gap = v1loc.y - source.vehicle_location().y
            if self._min_gap is None or gap < self._min_gap:
                self._min_gap = gap
        if v1loc is None:
            return
        if v1loc.y > self._leader_brake_y:
            source.obstacle_speed_mps()  # cruise: leader get_velocity (as _set_speed_along_heading)
            source.set_obstacle_target_velocity(self._leader_cruise_mps)
            return
        if self._v1_brake_time is None:
            self._v1_brake_time = sim_time
            if self._mode == "v2v" and self._v2v_pending_time is None:
                self._v2v_emit_time = self._v1_brake_time
                self._v2v_pending_time = self._v1_brake_time + self._v2v_delay_s
                self._v2v_sent = False
        v1_speed = source.obstacle_speed_mps()
        if self._v1_stop_time is None and v1_speed < 0.5:
            self._v1_stop_time = sim_time
            self._v1_stop_loc = (round(v1loc.x, 2), round(v1loc.y, 2), round(v1loc.z, 2))
        self._leader_ctrl.update_decel(v1_speed, now, alpha=0.35)
        source.apply_obstacle_control(0.0, self._leader_ctrl.step())

    def is_done(self, follower_done: bool, sim_time: float) -> bool:
        """Finish once both cars stop, or the follower stops and a post-brake timeout elapses."""
        if not follower_done:
            return False
        self._brake_latched = True
        if self._v1_stop_time is not None:
            return True
        if self._v1_brake_time is not None:
            return (sim_time - self._v1_brake_time) > self._post_brake_timeout_s
        return False

    def summary(self) -> dict:
        loop_rate = (self._loop_ticks / self._last_sim_time) if self._last_sim_time > 0 else None
        return {
            "leader_brake_t": self._v1_brake_time,
            "leader_stop_t": self._v1_stop_time,
            "leader_stop_loc": self._v1_stop_loc,
            "min_gap_m": (None if self._min_gap is None else round(self._min_gap, 2)),
            "loop_ticks": self._loop_ticks,
            "loop_rate_hz": (None if loop_rate is None else round(loop_rate, 2)),
            "v2v_emit_t": self._v2v_emit_time,
            "v2v_recv_t": self._v2v_recv_time,
            "braking_mode": self._mode,
        }
