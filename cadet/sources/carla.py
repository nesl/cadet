"""CARLA input source for CADET.

What this is for:
    Implements the :class:`cadet.core.source.Source` interface against a running
    CARLA simulator. It attaches to the ego vehicle that ScenarioRunner has
    already spawned, mounts an RGB camera the same way the original open-loop
    braking client did, and yields frames. It also exposes the small set of
    CARLA actuations the braking client needs (apply control, set target
    velocity, locate/destroy the obstacle).

What it contains:
    - :class:`CarlaSource`: open/next_frame/close/metadata plus the ego handle and
      actuation helpers.

How it fits into CADET:
    Built from the resolved config's ``source`` section (host/port/timeout,
    camera w/h/fov) and ``source.scenario`` (ego_vehicle, target_object). The
    braking client runtime (:mod:`cadet.roles.client`) drives it. ``carla`` is
    imported lazily so this module imports fine where CARLA is absent (unit tests).

How to configure it:
    `simulator_host` is a placeholder in the example configs; pass the real
    address at run time with ``--set source.carla.host=<HOST_IP>``.
"""

import time
from queue import Empty, Queue
from typing import Any, Dict, Optional

from cadet.core.source import Frame, Source

# Ego camera mount, copied from the original open-loop braking client so frames
# match the baseline: 2.5 m forward, 1.8 m up, pitched 10 deg down.
_CAMERA_X = 2.5
_CAMERA_Z = 1.8
_CAMERA_PITCH = -10.0
# How often to poll for the ego while waiting for the sim hook to spawn it.
_EGO_POLL_S = 0.2

# Blueprint each obstacle class maps to (as in the original client.find_obstacle).
_OBSTACLE_BLUEPRINTS = {
    "car": "vehicle.tesla.model3",
    "person": "walker.pedestrian.0001",
    "bike": "vehicle.kawasaki.ninja",
}


class CarlaSource(Source):
    """Frame source backed by a running CARLA simulator."""

    def __init__(self, config, logger=None):
        src = config.source
        self._host = src.carla.host
        self._port = src.carla.port
        self._timeout = src.carla.timeout
        self._cam_w = src.camera.width
        self._cam_h = src.camera.height
        self._cam_fov = src.camera.fov
        # Ego-camera mount (defaults = the braking/V2V baseline; V2I sets z=1.6).
        self._cam_mount_x = getattr(src.camera, "mount_x", _CAMERA_X)
        self._cam_mount_z = getattr(src.camera, "mount_z", _CAMERA_Z)
        self._cam_mount_pitch = getattr(src.camera, "mount_pitch", _CAMERA_PITCH)
        self._cam_queue_size = int(getattr(src.camera, "queue_size", 10))
        self._fps = src.fps  # camera capture rate (used only when camera.fixed_rate)
        # Opt-in deviation (default = original free-running camera); see below.
        self._camera_fixed_rate = bool(src.camera.fixed_rate)
        self._ego_name = src.scenario.ego_vehicle
        self._ego_role_name = src.scenario.ego_role_name
        # The blueprint fallback says so once rather than per poll: the caller
        # polls for the ego until it appears, and a line per attempt would bury
        # the run's own output.
        self._warned_blueprint_match = False
        self._attach_timeout_s = float(src.scenario.ego_attach_timeout_s)
        self._object_type = src.scenario.target_object
        self._map = src.scenario.map
        self._log = logger

        self._client = None
        self._world = None
        self._camera = None
        self._rgb_queue: "Queue" = Queue(maxsize=self._cam_queue_size)
        self.vehicle = None
        self.obstacle = None

    # --- Source interface (two-phase: prepare / await_scene / open) -------
    def prepare(self) -> None:
        """Connect to the simulator. Does not need the ego to exist yet.

        Split out of :meth:`open` so a client can finish everything that does
        not depend on the ego - including connecting to its server and loading
        its own model - and announce itself ready *before* the scenario is
        started. The ego exists only for a short window once the scenario runs,
        and a client still setting up when it appeared could miss it entirely.
        """
        import carla  # lazy: only needed at run time

        if self._world is not None:
            return
        self._client = carla.Client(self._host, self._port)
        self._client.set_timeout(float(self._timeout))
        self._world = self._client.get_world()

    def await_scene(self, timeout_s: float) -> bool:
        """Poll until the ego the sim hook spawned exists; True if it appeared.

        This is the client waiting for the scenario rather than the other way
        round. Polling is the only option available: CARLA 0.9.10 offers no
        actor-spawn notification.
        """
        self.prepare()
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            self.vehicle = self._find_ego()
            if self.vehicle is not None:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(_EGO_POLL_S)

    def open(self) -> None:
        """Attach to the ego (if not already) and set up the camera + obstacle."""
        import carla

        self.prepare()
        if self.vehicle is None and not self.await_scene(self._attach_timeout_s):
            raise RuntimeError("no ego vehicle found; is the sim hook (ScenarioRunner) running?")
        if self._camera is None:
            self._camera = self._spawn_camera(carla)
        self.obstacle = self._find_obstacle()

    def snapshot(self):
        """Return (frame_id, elapsed_s, dt_s) of the current world tick, or None.

        Lets a client gate control on the simulation clock (one command per tick)
        and use the tick dt for rate-dependent logic, instead of the loop's wall rate.
        """
        if self._world is None:
            return None
        snap = self._world.get_snapshot()
        ts = snap.timestamp
        return (snap.frame, ts.elapsed_seconds, ts.delta_seconds)

    def next_frame(self) -> Optional[Frame]:
        # Blocking read (release the GIL while no frame is queued) matches the
        # original client's dedicated sender doing ``ego_q.get()``: a busy
        # ``get_nowait`` spin holds the GIL and starves the camera callback
        # thread that fills this queue, lowering the effective capture rate.
        # Server-delay-bound apps (braking/V2V) are never camera-limited, so
        # this does not change their cadence.
        try:
            image, capture_loc, capture_time = self._rgb_queue.get(timeout=0.2)
        except Empty:
            return None
        import numpy as np

        # BGRA -> RGB, matching the original client's reshape/channel order. One
        # ``ascontiguousarray`` materialises the reversed view into a fresh,
        # memory-safe contiguous buffer (the source ``carla.Image`` may be
        # freed before ``process_frame`` pickles this later) in a SINGLE copy -
        # the original pickled the view directly (one copy); an explicit
        # ``.copy()`` of the raw buffer here on top of pickle's own copy doubled
        # the per-frame memory traffic (~12 ms) and dropped the send rate.
        view = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            (image.height, image.width, 4)
        )[:, :, :3][:, :, ::-1]
        array = np.ascontiguousarray(view)
        return Frame(image=array, meta={"capture_loc": capture_loc, "capture_time": capture_time})

    def close(self) -> None:
        for actor in (self._camera,):
            try:
                if actor is not None:
                    actor.destroy()
            except Exception:
                pass
        self._camera = None

    def exhausted(self) -> bool:
        """Never: a running simulator keeps producing frames until it is closed.

        ``next_frame`` returns None whenever the camera queue is momentarily
        empty, which it always is on the first tick after the camera is spawned.
        Reading that as the end of the stream ends a run before it has seen a
        single frame.
        """
        return False

    def metadata(self) -> Dict[str, Any]:
        return {
            "width": self._cam_w,
            "height": self._cam_h,
            "fov": self._cam_fov,
            "map": self._map,
        }

    # --- CARLA-specific helpers the braking client needs ------------------
    def apply_control(self, throttle: float, steer: float, brake: float) -> None:
        """Apply a vehicle control to the ego."""
        import carla

        self.vehicle.apply_control(
            carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        )

    def set_target_velocity(self, speed_mps: float) -> None:
        """Set the ego's velocity along its heading (the baseline's speed kick)."""
        import carla

        yaw = self.vehicle.get_transform().rotation.yaw
        fwd = carla.Rotation(yaw=yaw).get_forward_vector()
        self.vehicle.set_target_velocity(
            carla.Vector3D(fwd.x * speed_mps, fwd.y * speed_mps, fwd.z * speed_mps)
        )

    def ego_is_alive(self) -> bool:
        """Whether the ego actor still exists in the simulator.

        A scenario destroys the actors it spawned when it ends, and an app that
        is waiting on the ego - for it to brake, or to come to a stop - is then
        waiting for something that cannot happen. Asked of the actor rather than
        inferred from a stale reading: a destroyed actor keeps answering
        ``get_location`` with the last position it had, so a client watching
        only the position cannot tell the difference between stopped and gone.

        True when the simulator cannot be asked, because an unanswerable
        question is not evidence that the ego has gone.

        The world is asked for the actor by id, rather than the actor handle
        being asked about itself. A handle whose actor the server has already
        destroyed goes on answering: ``is_alive`` stayed true and
        ``get_location`` kept returning the last position, so a client trusting
        either kept braking a vehicle that was no longer there. Measured on the
        distributed tier, where the scenario reported it could not destroy the
        ego because the ego was already gone.
        """
        if self.vehicle is None:
            return False
        if self._world is None:
            return True
        actor_id = getattr(self.vehicle, "id", None)
        if actor_id is None:
            return True
        try:
            # The world's actor list, not a lookup by id: in this version both
            # the handle and a per-id lookup keep answering for an actor the
            # server has destroyed, and the list is the snapshot the server
            # actually holds. What a dead handle looks like, measured on a
            # distributed run whose scenario had ended: the ego reporting the
            # exact origin with zero rotation, tick after tick, while the
            # camera on it reported a real pose.
            return any(actor.id == actor_id for actor in self._world.get_actors())
        except RuntimeError:
            # The simulator could not answer; that is not evidence either way.
            return True

    def vehicle_location(self):
        """Current ego world location."""
        return self.vehicle.get_location()

    def vehicle_speed_mps(self) -> float:
        """Current ego speed magnitude in m/s."""
        import numpy as np

        v = self.vehicle.get_velocity()
        return float(np.linalg.norm([v.x, v.y, v.z]))

    def original_client_loop_poll(self, log) -> None:
        """The original client's per-tick debug polling.

        Ported verbatim from ``open_loop_run_braking.py``'s loop tail: two
        ``get_transform`` RPCs (ego + camera), the relative offset, four debug prints
        and the ego bounding-box read - run every control tick. This is the remaining
        per-tick cost (RPC round-trips + console I/O) that set the original client's
        loop rate; it is the real work, not a sleep. ``log`` carries the prints to the
        node log, the same I/O the original sent to stdout.
        """
        vehicle_tf = self.vehicle.get_transform()
        camera_tf = self._camera.get_transform() if self._camera is not None else vehicle_tf
        relative_loc = camera_tf.location - vehicle_tf.location
        log("[VEHICLE] Location: {}, Rotation: {}".format(vehicle_tf.location, vehicle_tf.rotation))
        log("[CAMERA ] Location: {}, Rotation: {}".format(camera_tf.location, camera_tf.rotation))
        log("Camera is at relative offset: {}".format(relative_loc))
        log(str(self.vehicle.bounding_box.extent))

    def poll_control(self):
        """The ego's current control (``get_control`` RPC), as the original read per frame."""
        return self.vehicle.get_control()

    def apply_walker_control(self, dx, dy, dz, speed):
        """Drive the obstacle as a walker (V2I pedestrian): direction vector + speed.

        The V2I client drives the crossing pedestrian itself, as the original did; the
        obstacle here is a ``walker.pedestrian.*`` actor, so it takes a WalkerControl
        rather than a VehicleControl.
        """
        import carla

        if self.obstacle is None:
            return
        self.obstacle.apply_control(
            carla.WalkerControl(direction=carla.Vector3D(dx, dy, dz), speed=speed)
        )

    def spawn_fixed_camera(self, x, y, z, pitch, yaw, width, height, fov):
        """Spawn an RGB camera at a FIXED world pose (not ego-attached); return (cam, queue).

        For the V2I infrastructure (roadside) feed: the one client process owns this
        camera in addition to its ego-attached one, as the original V2IClient did. The
        queue yields ``(rgb_array, capture_time)`` with the same BGRA->RGB conversion the
        ego path uses; it drops frames when full (keep-latest), like the ego camera.
        """
        import carla
        import numpy as np

        bp = self._world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov))
        transform = carla.Transform(
            carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw)
        )
        camera = self._world.spawn_actor(bp, transform)
        q: "Queue" = Queue(maxsize=self._cam_queue_size)

        def _callback(image):
            try:
                if len(image.raw_data) != image.height * image.width * 4:
                    return
                array = np.frombuffer(image.raw_data, dtype=np.uint8).copy()
                array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
                q.put_nowait((array, time.time()))
            except Exception:
                pass  # queue full -> drop (keep latest), as the ego camera does

        camera.listen(_callback)
        return camera, q

    def obstacle_location(self):
        """Current obstacle world location, or None when there is no obstacle.

        For V2V the obstacle is the scripted lead vehicle; the client reads its
        location to drive it (cruise/brake) and to measure the headway gap.
        """
        if self.obstacle is None:
            return None
        return self.obstacle.get_location()

    def obstacle_speed_mps(self) -> float:
        """Current obstacle speed magnitude in m/s (0.0 when there is no obstacle)."""
        import numpy as np

        if self.obstacle is None:
            return 0.0
        v = self.obstacle.get_velocity()
        return float(np.linalg.norm([v.x, v.y, v.z]))

    def obstacle_velocity(self):
        """Current obstacle velocity vector ``(vx, vy, vz)`` m/s, or None if no obstacle.

        For the V2I ground-truth pedestrian TTE (evaluation only): the signed y-velocity
        gives the ped's time to the crossing line.
        """
        if self.obstacle is None:
            return None
        v = self.obstacle.get_velocity()
        return (float(v.x), float(v.y), float(v.z))

    def apply_obstacle_control(self, throttle: float, brake: float) -> None:
        """Apply a throttle/brake control to the obstacle (V2V leader actuation)."""
        import carla

        if self.obstacle is None:
            return
        self.obstacle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake))

    def set_obstacle_target_velocity(self, speed_mps: float) -> None:
        """Set the obstacle's velocity along its heading (the V2V leader cruise kick)."""
        import carla

        if self.obstacle is None:
            return
        yaw = self.obstacle.get_transform().rotation.yaw
        fwd = carla.Rotation(yaw=yaw).get_forward_vector()
        self.obstacle.set_target_velocity(
            carla.Vector3D(fwd.x * speed_mps, fwd.y * speed_mps, fwd.z * speed_mps)
        )

    def destroy_obstacle(self) -> None:
        """Destroy the obstacle actor (called when braking begins)."""
        try:
            if self.obstacle is not None:
                self.obstacle.destroy()
        except Exception:
            pass
        self.obstacle = None

    # --- internals --------------------------------------------------------
    def _find_ego(self):
        """Return the ego actor by its CARLA ``role_name``, or by blueprint.

        The role name is how the original found it - ``role_name == 'hero'``,
        which is what ScenarioRunner tags an ego with - and is the default.

        Matching the blueprint instead is a deviation, reached only by setting
        ``source.scenario.ego_role_name`` empty, and it warns once when used. A
        blueprint is not unique: which actor it returns depends on what else is
        in the world and on when the question is asked, so a run that attached
        to the wrong vehicle looks like a working run whose ego never moves.

        Returns ``None`` when the ego is not there yet (the caller polls).
        """
        if self._ego_role_name:
            for actor in self._world.get_actors().filter("vehicle.*"):
                if actor.attributes.get("role_name") == self._ego_role_name:
                    return actor
            return None
        if not self._warned_blueprint_match:
            self._warned_blueprint_match = True
            if self._log:
                self._log(
                    "[source] matching the ego by blueprint {!r} because "
                    "source.scenario.ego_role_name is empty. The original "
                    "matched role_name 'hero'; a blueprint is not unique, so "
                    "this can attach to another vehicle of the same model."
                    .format(self._ego_name)
                )
        matches = self._world.get_actors().filter("vehicle.{}".format(self._ego_name))
        return matches[0] if len(matches) > 0 else None

    def _find_obstacle(self):
        bp = _OBSTACLE_BLUEPRINTS.get(self._object_type)
        if bp is None:
            raise ValueError("unknown target_object {!r}".format(self._object_type))
        matches = self._world.get_actors().filter(bp)
        if len(matches) == 1:
            return matches[0]
        if self._log:
            self._log(
                "no unique obstacle for {!r} (found {})".format(self._object_type, len(matches))
            )
        return None

    def _spawn_camera(self, carla):
        bp = self._world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self._cam_w))
        bp.set_attribute("image_size_y", str(self._cam_h))
        bp.set_attribute("fov", str(self._cam_fov))
        # Opt-in deviation (default = original free-running camera). Only when
        # camera.fixed_rate is set does the sensor fire once per 1/fps of
        # *simulated* time. Without it (the original) the sensor fires once per
        # rendered frame at whatever rate the asynchronous server runs, so each run
        # samples the approaching obstacle at different ego positions and the braking
        # point varies run-to-run. Pinning the tick makes the sampled positions, and
        # so the braking point, repeatable.
        if self._camera_fixed_rate and self._fps and self._fps > 0:
            bp.set_attribute("sensor_tick", str(1.0 / float(self._fps)))
        transform = carla.Transform(
            carla.Location(x=self._cam_mount_x, y=0, z=self._cam_mount_z),
            carla.Rotation(pitch=self._cam_mount_pitch),
        )
        camera = self._world.spawn_actor(bp, transform, attach_to=self.vehicle)

        def _callback(image):
            try:
                if len(image.raw_data) != image.height * image.width * 4:
                    return
                self._rgb_queue.put_nowait((image, self.vehicle.get_location(), time.time()))
            except Exception:
                pass  # queue full -> drop frame (keep latest), as in the original

        camera.listen(_callback)
        return camera
