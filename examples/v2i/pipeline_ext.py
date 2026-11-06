"""V2I client-side extension for the pipeline driver (pedestrian-walker drive).

What this is for:
    The pipeline driver (:class:`cadet.pipeline.app.PipelineClientApp`) runs the
    staged camera -> yolo_v2i -> gate -> brake_controller pipeline, but the V2I run
    also needs the client to drive the crossing pedestrian each tick, exactly as the
    flat :class:`examples.v2i.app` does (the original drove the walker from the
    client's ``_walker_loop``, not the scenario). The pipeline client loads this
    class through its generic ``client_extension`` hook and calls its duck-typed
    hooks; the V2I specifics stay here in ``examples/``.

    This covers the single-feed modes (``baseline_local``/``baseline_cloud``): the
    ego camera alone streams to the RSU and the ``v2i_ego_gate`` stage decides. The
    two-feed modes (``v2i_local``/``v2i_external``) additionally need a *second*
    fixed infra-camera feed, which the single-source pipeline driver does not
    provide; those stay on the flat form until the driver grows a second feed.

How it fits into CADET:
    Configured by ``app.params.client_extension:
    examples.v2i.pipeline_ext:V2IPipelineExtension`` in the V2I pipeline config.
    Reuses the flat app's :class:`V2IParams` so the two forms read the same walker
    timing, ROI and gate values.
"""

from examples.v2i.app import V2IParams


class V2IPipelineExtension:
    """Drives the crossing pedestrian for the V2I pipeline (ego-feed modes)."""

    def __init__(self, config):
        # V2I params ride in app.params.extension_params (empty = original defaults).
        self.params = V2IParams.model_validate(config.app.params.get("extension_params", {}))
        self._mode = self.params.mode
        self._ped_speed_mps = self.params.ped_speed_mps
        self._ped_goal_x = self.params.ped_goal_x
        # Idle until the ego is near, then jaywalk (same as the flat app).
        self._ped_cross_start_s = self.params.t_ped_start_s + self.params.ped_start_delay_s
        self.source = None
        self._ticks = 0
        self._last_sim_time = 0.0

    def setup(self, source) -> None:
        self.source = source

    def keep_obstacle(self) -> bool:
        """Keep the pedestrian alive: the ego brakes for it, it is not passed."""
        return True

    def drive(self, source, sim_time: float, now: float) -> None:
        """Drive the crossing pedestrian, ported from the flat app's ``_drive_walker``.

        Idle until the cross-start time, then jaywalk in -X at ``ped_speed_mps``
        until the ped passes ``ped_goal_x``, then hold - so the ped's position
        relative to the approaching ego matches the flat form and the original.
        """
        self._ticks += 1
        self._last_sim_time = sim_time
        ped = source.obstacle_location()
        if ped is None:
            return
        if sim_time < self._ped_cross_start_s:
            return  # spawned, not yet crossing
        if ped.x <= self._ped_goal_x + 0.25:
            source.apply_walker_control(-1.0, 0.0, 0.0, 0.0)  # reached goal: hold
            return
        source.apply_walker_control(-1.0, 0.0, 0.0, self._ped_speed_mps)

    def is_done(self, follower_done: bool, sim_time: float) -> bool:
        """Finish once the ego (the pipeline's own vehicle) has stopped."""
        return bool(follower_done)

    def summary(self) -> dict:
        rate = (self._ticks / self._last_sim_time) if self._last_sim_time > 0 else None
        return {
            "braking_mode": self._mode,
            "ped_cross_start_s": self._ped_cross_start_s,
            "loop_ticks": self._ticks,
            "loop_rate_hz": (None if rate is None else round(rate, 2)),
        }
