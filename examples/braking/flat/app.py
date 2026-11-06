"""Open-loop emergency-braking app (CADET example, not part of the framework).

What this is for:
    The reference AV example: ports the original open-loop braking client/server
    onto the generic CADET runtimes. All AV-specific logic (brake controller,
    detection gating, YOLO) lives here, under examples/, not in cadet/.

What it contains:
    - The app's typed parameters: :class:`BrakingParams` (+ :class:`BrakeController`,
      :class:`RoiBand`), declared here so the framework stays app-agnostic.
    - Pure helpers (:func:`target_speed_mps`, :func:`frame_triggers_brake`, …).
    - :class:`BrakingServerApp` (YOLO + consecutive-detection, with framework
      warm-up and explicit device) and :class:`BrakingClientApp`.
    - :class:`BrakingApp`: the plugin, registered on import.

How it fits into CADET:
    Registered via ``cadet.core.apps.register_app``. Launch with
    ``cadet serve/client ... --app-module examples.braking.flat.app``. The framework
    (``cadet/roles``) owns the loop, transport, warm-up and device; this module
    owns what to compute. Nothing in ``cadet/`` imports this module.
"""
import os
import time
from queue import Empty, Queue
from threading import Thread
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Literal

from cadet.core.apps import AppPlugin, register_app
from cadet.core.codec import decode as wire_decode
from cadet.core.codec import encode as wire_encode
from cadet.core.model_registry import ModelRegistry, device_for_models, torch_device_for
from cadet.core.source import ego_removed
from cadet.metrics import MetricsLog

MPS_PER_MPH = 0.44704
MPS_TO_MPH = 2.23694
STOP_SPEED_MPS = 0.1  # ego is "stopped" below this speed (original threshold)
SPEED_KICK_Y = 90.0   # ego location gate at which the one-time velocity kick fires


# --------------------------------------------------------------------------
# Typed app parameters (declared with the app, not in the framework)
# --------------------------------------------------------------------------
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BrakeController(_Strict):
    """Per-vehicle deceleration controller gains."""

    target_decel: float
    max_decel: float
    kp: float
    max_ramp: float


class RoiBand(_Strict):
    """Pixel region-of-interest band used to select the relevant obstacle."""

    cx_min: int
    cx_max: int
    cy_min: int = 0
    cy_max: int = 0


def _braking_controllers() -> Dict[str, BrakeController]:
    return {
        "audi.tt": BrakeController(target_decel=7.0, max_decel=7.3, kp=0.2, max_ramp=0.05),
        "carlamotors.carlacola": BrakeController(target_decel=3.0, max_decel=3.0, kp=0.45, max_ramp=0.15),
        "kawasaki.ninja": BrakeController(target_decel=6.0, max_decel=6.0, kp=0.2, max_ramp=0.05),
    }


class BrakingParams(_Strict):
    """Parameters for the single-vehicle open-loop emergency-braking app."""

    plugin: Literal["braking"] = "braking"
    controllers: Dict[str, BrakeController] = Field(default_factory=_braking_controllers)
    # Opt-in; the default is what the original did. ``per_loop`` issues a control command
    # on every client loop iteration using wall-clock time, exactly as the original
    # client did. ``per_tick`` issues one command per simulation tick (gated on the
    # world snapshot) and drives the brake ramp off the simulation clock.
    control_rate: Literal["per_loop", "per_tick"] = "per_loop"
    detection_threshold: int = 3  # consecutive detections before braking
    target_classes: List[str] = Field(
        default_factory=lambda: ["car", "motorcycle", "bicycle", "person"]
    )
    conf_thresh: float = 0.3
    roi: RoiBand = Field(default_factory=lambda: RoiBand(cx_min=300, cx_max=500))
    # From the original (default): the original server wrote an annotated JPEG
    # per processed frame. Written by default under the run's output tree; set false
    # to disable the extra I/O.
    save_annotated_frames: bool = True
    # From the original (default): the original client rendered a pygame
    # visualiser every control tick and saved a PNG per captured frame, which set its
    # loop rate (~10 fps). On by default so a CADET run reproduces that loop rate (and
    # so the per_loop brake dynamics); set false for a lean, faster loop.
    client_visualizer: bool = True
    # Also from the original (default): the rest of the original client's
    # per-tick work beyond the visualiser - two get_transform RPCs + four debug prints
    # every tick, get_control per sent frame, a per-tick braking print. Ported as the
    # real calls (not a sleep); this is the remaining per-tick cost that set the loop
    # rate. One switch for all of it; set false for a lean loop.
    original_client_loop: bool = True


# --------------------------------------------------------------------------
# Pure helpers (no CARLA / no torch) — unit tested
# --------------------------------------------------------------------------
def target_speed_mps(speed_str: str) -> float:
    """Convert a speed string like ``40mph`` / ``20kph`` to metres per second."""
    if speed_str.endswith("kph"):
        return float(speed_str[:-3]) / 3.6
    if speed_str.endswith("mph"):
        return float(speed_str[:-3]) * MPS_PER_MPH
    raise ValueError("speed must end with 'kph' or 'mph', got {!r}".format(speed_str))


def resolve_target_class_ids(class_names: Dict[int, str], target_classes: List[str]) -> List[int]:
    """Map target class names to detector class ids (substring match, as original)."""
    wanted = [t.lower() for t in target_classes]
    return [k for k, v in class_names.items() if any(t in v.lower() for t in wanted)]


def frame_triggers_brake(boxes, target_ids, roi_cx_min, roi_cx_max, conf_thresh) -> bool:
    """True if any ``(class_id, confidence, center_x)`` is a target in-ROI above thresh."""
    for cls_id, conf, cx in boxes:
        if cls_id in target_ids and roi_cx_min <= cx <= roi_cx_max and conf > conf_thresh:
            return True
    return False


class BrakeControllerState:
    """Deceleration controller (ported verbatim from the original client)."""

    def __init__(self, target_decel: float, max_decel: float, kp: float, max_ramp: float):
        self.target_decel = target_decel
        self.max_decel = max_decel
        self.kp = kp
        self.max_ramp = max_ramp
        self.filtered_decel = 0.0
        self.brake_cmd = 0.0
        self.last_speed = 0.0
        self.last_time: Optional[float] = None

    def update_decel(self, current_speed: float, current_time: float, alpha: float = 0.3) -> float:
        dt = (current_time - self.last_time) if self.last_time is not None else 0.0
        raw = (self.last_speed - current_speed) / dt if dt > 0 else 0.0
        raw = max(0.0, raw)
        self.filtered_decel = alpha * raw + (1 - alpha) * self.filtered_decel
        self.last_speed = current_speed
        self.last_time = current_time
        return self.filtered_decel

    def step(self) -> float:
        if self.filtered_decel >= self.max_decel:
            self.brake_cmd = max(0.0, self.brake_cmd - self.max_ramp)
            return self.brake_cmd
        error = self.target_decel - self.filtered_decel
        desired = max(0.0, min(self.kp * error, 1.0))
        if desired > self.brake_cmd:
            self.brake_cmd = min(self.brake_cmd + self.max_ramp, desired)
        else:
            self.brake_cmd = max(self.brake_cmd - self.max_ramp, desired)
        return self.brake_cmd


def pid_throttle(current_speed: float, target_speed: float, kp: float = 0.6, deadband: float = 1.0) -> float:
    """Pre-brake cruise throttle (ported from the original client)."""
    error = target_speed - current_speed
    if abs(error) < deadband:
        return 0.4
    return max(0.0, min(kp * error, 1.0))


def save_annotated_detection(rgb, result, class_names, out_dir, frame_id) -> None:
    """Write an annotated detection JPEG for one frame (ported from the original server).

    Draws every detected box with its class/confidence label and saves
    ``frame_<id>.jpg`` under ``out_dir``. From the original: the original server wrote one
    of these per processed frame; CADET does so by default (see BrakingParams).
    """
    import os

    import cv2

    img = rgb.copy()
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img, "{} {:.2f}".format(class_names[cls], conf), (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
    out_path = os.path.join(out_dir, "frame_{:05d}.jpg".format(frame_id))
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------------------
# Server app
# --------------------------------------------------------------------------
class BrakingServerApp:
    """Runs YOLO on received frames and brakes after N consecutive detections."""

    def __init__(self, config, model_search_dirs: List[str], logger=None):
        self.cfg = config
        self.params = BrakingParams.model_validate(config.app.params)
        self.search_dirs = model_search_dirs
        self._log = logger or (lambda m: None)
        self._model = None
        self._device = None
        self._target_ids: List[int] = []
        self._consecutive = 0
        # From the original: annotated detection JPEGs (default on). Written under the
        # experiment/app output tree, as the original wrote them to a fixed path.
        self._save_annotated = self.params.save_annotated_frames
        self._annotated_dir = os.path.join(
            config.outputs.root, config.experiment.name, config.app.plugin, "server_detections"
        )

    def setup(self, ctx=None) -> None:
        from ultralytics import YOLO

        device = device_for_models(self.cfg)
        registry = ModelRegistry.from_config(self.cfg.models, device)
        res = registry.resolve(self.cfg.models.model, self.search_dirs)
        if res.path is None:
            raise RuntimeError(
                "model artifact {!r} not found under {}".format(res.artifact, self.search_dirs)
            )
        # Explicit torch device from the node config (framework helper).
        self._device = torch_device_for(self.cfg)
        self._log("[server] loading model {} ({}) on {}".format(
            self.cfg.models.model, res.path, self._device))
        self._model = YOLO(res.path)
        self._target_ids = resolve_target_class_ids(self._model.names, self.params.target_classes)
        if not self._target_ids:
            raise RuntimeError("none of the target classes are known to the model")
        self._class_names = self._model.names
        if self._save_annotated:
            os.makedirs(self._annotated_dir, exist_ok=True)
            self._log("[server] annotated detections -> {}".format(self._annotated_dir))
        # Parity-L1 recorder (no-op unless $CADET_RECORD_DIR is set).
        from cadet.pipeline.record import open_recorder

        self._rec = open_recorder("detections")
        self._rec.record(
            meta=True,
            target_ids=[int(i) for i in self._target_ids],
            roi_cx_min=self.params.roi.cx_min,
            roi_cx_max=self.params.roi.cx_max,
            conf_thresh=self.params.conf_thresh,
            detection_threshold=self.params.detection_threshold,
        )

    def warmup(self) -> None:
        """Framework-invoked: run one dummy inference so the model JITs now."""
        import numpy as np

        cam = self.cfg.source.camera
        dummy = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
        t0 = time.time()
        self._model.predict(dummy, verbose=False, device=self._device)
        self._log("[server] warmup predict took {:.3f}s on {}".format(time.time() - t0, self._device))

    def handle(self, payload: bytes, ctx=None) -> Optional[bytes]:
        message = wire_decode(payload)
        frame_id = message.get("frame_id", -1)
        rgb = message["rgb_frame"]
        t_start = time.time()
        detections = self._model.predict(rgb, verbose=False, device=self._device)

        boxes = []
        for det in detections[0].boxes:
            cls_id = int(det.cls[0])
            conf = float(det.conf[0])
            bbox = det.xyxy[0].cpu().numpy()
            cx = (bbox[0] + bbox[2]) / 2
            boxes.append((cls_id, conf, cx))

        if self._save_annotated:
            save_annotated_detection(
                rgb, detections[0], self._class_names, self._annotated_dir, frame_id
            )

        found = frame_triggers_brake(
            boxes, self._target_ids, self.params.roi.cx_min, self.params.roi.cx_max, self.params.conf_thresh
        )
        self._consecutive = self._consecutive + 1 if found else 0

        self._rec.record(
            frame_id=frame_id,
            boxes=[[int(c), float(cf), float(cx)] for c, cf, cx in boxes],
            consecutive=self._consecutive,
            brake=bool(self._consecutive >= self.params.detection_threshold),
        )

        if self._consecutive >= self.params.detection_threshold:
            reply = {
                "brake": True,
                "server_processing_time": time.time() - t_start,
                "frame_id": frame_id,
            }
            self._log("[server] brake issued at frame {}".format(frame_id))
            return wire_encode(reply)
        return None

    def stop_after_reply(self) -> bool:
        return True

    def teardown(self, ctx=None) -> None:
        if getattr(self, "_rec", None) is not None:
            self._rec.close()


# --------------------------------------------------------------------------
# Client app
# --------------------------------------------------------------------------
class BrakingClientApp:
    """Client hooks: cruise to target speed, send frames, brake on command."""

    def __init__(self, config, run_id: str, rep: int, csv_path: str, logger=None):
        self.cfg = config
        self.params = BrakingParams.model_validate(config.app.params)
        self.source = None
        self.transport = None
        self._log = logger or (lambda m: None)
        ego = config.source.scenario.ego_vehicle
        self.controller = BrakeControllerState(**self.params.controllers[ego].model_dump())
        self.target_mps = target_speed_mps(config.source.scenario.ego_speed)
        self.metrics = MetricsLog(csv_path, run_id, rep)
        # From the original: per-tick visualiser + per-frame PNG, under the run's dir.
        self._viz_enabled = self.params.client_visualizer
        self._viz_dir = os.path.join(os.path.dirname(csv_path), "client_frames")
        self._viz = None
        # Also from the original: the rest of the original client's per-tick work.
        self._orig_loop = self.params.original_client_loop

        self._start = None
        self._frame_counter = 0
        self._cur_fid = None
        self._transmit = True
        self._brake_engaged = False
        self._brake_recorded = False
        self._done = False
        # Set only when the ego has gone from the world before the run reached
        # its own end, which is the one case this client cannot wait out.
        self._ego_gone = False
        self._ego_checked_at = 0.0
        self._set_speed_once = False
        self._speed_forced = False
        self._last_speed_mph = 0.0
        self._control_rate = self.params.control_rate  # per_loop (original) | per_tick
        self._last_ctrl_frame = None  # last simulation tick a command was issued on (per_tick)
        self._replies: "Queue" = Queue()

    def setup(self, ctx) -> None:
        self.source = ctx.source
        self.transport = ctx.transport
        # The time origin is NOT taken here. setup() runs before the client waits
        # for the ego, so an origin set now would include that wait in every
        # reported time; it is taken on the first control tick instead.
        from cadet.pipeline.record import open_recorder

        self._rec = open_recorder("control")  # parity-L1; no-op unless $CADET_RECORD_DIR set
        cam = getattr(self.cfg.source, "camera", None)
        if self._viz_enabled and cam is not None:
            from cadet.roles.client_visualizer import ClientVisualizer

            self._viz = ClientVisualizer(cam.width, cam.height, self._viz_dir, logger=self._log)
        Thread(target=self._reply_listener, daemon=True).start()

    def _reply_listener(self) -> None:
        while True:
            try:
                raw = self.transport.recv()
            except Exception:
                break
            if raw is None:
                break
            try:
                cmd = wire_decode(raw)
                if cmd.get("brake", False):
                    self._replies.put(cmd)
            except Exception:
                continue

    def poll_reply(self, ctx) -> Optional[dict]:
        try:
            return self._replies.get_nowait()
        except Empty:
            return None

    def translate_action(self, frame, reply, ctx) -> None:
        # Control rate (opt-in; the default, per_loop, is what the original did):
        #   per_loop  - a command on every loop iteration, driven by wall time, as
        #               the original client did.
        #   per_tick  - one command per simulation tick (gate on the world snapshot
        #               frame id) and drive the brake ramp off the snapshot clock.
        if self._start is None:
            # First control tick: the moment the run actually begins, whatever
            # order the setup happened in.
            self._start = time.time()
        now = time.time()
        if self._control_rate == "per_tick":
            snap = self.source.snapshot() if hasattr(self.source, "snapshot") else None
            if snap is not None:
                tick_frame, sim_elapsed, _dt = snap
                if tick_frame == self._last_ctrl_frame:
                    return  # same simulation tick -> no new command
                self._last_ctrl_frame = tick_frame
                ctrl_t = sim_elapsed
            else:
                ctrl_t = now  # non-simulation source: fall back to wall time
        else:
            ctrl_t = now  # per_loop (original): every loop iteration, wall time
        sim_time = now - self._start
        speed_mps = self.source.vehicle_speed_mps()
        self._last_speed_mph = speed_mps * MPS_TO_MPH
        throttle, steer, brake = 0.0, 0.0, 0.0

        if reply is not None and not self._brake_engaged:
            self._brake_engaged = True
            self._transmit = False
            self.source.destroy_obstacle()
            loc = self.source.vehicle_location()
            brake_loc = (round(loc.x, 2), round(loc.y, 2), round(loc.z, 2))
            if not self._brake_recorded:
                self._brake_recorded = True
                self.metrics.mark_brake(
                    reply.get("frame_id"), reply.get("server_processing_time"), sim_time, brake_loc,
                    recv_wall=now,
                )
                self._log("[client] BRAKE at t={:.2f} {}".format(sim_time, brake_loc))

        if self._brake_engaged:
            self.controller.update_decel(speed_mps, ctrl_t)
            brake = self.controller.step()
            throttle = 0.0
            if self._orig_loop:  # the original printed this every braking tick
                self._log(
                    "[client][BRAKING] Time: {:.2f}s | Speed: {:.2f} m/s | Brake: {:.2f}".format(
                        sim_time, speed_mps, brake
                    )
                )
            if not self._done and speed_mps < STOP_SPEED_MPS:
                loc = self.source.vehicle_location()
                stop_loc = (round(loc.x, 2), round(loc.y, 2), round(loc.z, 2))
                self.metrics.mark_stop(sim_time, stop_loc)
                self._log("[client] STOP at t={:.2f} {}".format(sim_time, stop_loc))
                self._done = True
        else:
            loc = self.source.vehicle_location()
            if loc.y < SPEED_KICK_Y and not self._speed_forced:
                self._set_speed_once = True
                self._speed_forced = True
            if self._set_speed_once:
                self.source.set_target_velocity(self.target_mps)
                self._set_speed_once = False
            else:
                throttle = pid_throttle(speed_mps, self.target_mps)
                brake = 0.0

        self._rec.record(
            t=ctrl_t,
            speed_mps=speed_mps,
            loc_y=self.source.vehicle_location().y,
            brake_pending=bool(self._brake_engaged),
            throttle=throttle,
            steer=steer,
            brake=brake,
        )
        self.source.apply_control(throttle, steer, brake)
        # From the original: render the visualiser every control tick (the per-tick work
        # that sets the original's loop rate).
        if self._viz is not None:
            self._viz.update_frame(frame.image if frame is not None else None)
            self._viz.render(throttle, steer, brake, sim_time)
        # Also from the original: the rest of the original's per-tick work.
        if self._orig_loop and hasattr(self.source, "original_client_loop_poll"):
            self.source.original_client_loop_poll(self._log)

    def should_transmit(self, ctx) -> bool:
        return self._transmit

    def process_frame(self, frame, ctx) -> bytes:
        fid = self._frame_counter
        self._cur_fid = fid
        capture_time = frame.meta["capture_time"]
        cloc = frame.meta["capture_loc"]
        self.metrics.new_frame(
            fid, capture_time, (round(cloc.x, 2), round(cloc.y, 2), round(cloc.z, 2)), self._last_speed_mph
        )
        if self._viz is not None:
            # A PNG per captured frame, as the original client wrote; off
            # unless a visualizer is configured.
            self._viz.save_frame(frame.image, fid)
        if self._orig_loop and hasattr(self.source, "poll_control"):
            self.source.poll_control()  # the original read get_control per sent frame
        msg = {"frame_id": fid, "timestamp": capture_time, "rgb_frame": frame.image, "frame_type": "I"}
        return wire_encode(msg)

    def on_frame_sent(self, frame, ctx) -> None:
        send_time = time.time() - self._start
        sloc = self.source.vehicle_location()
        self.metrics.mark_sent(self._cur_fid, send_time, (round(sloc.x, 2), round(sloc.y, 2), round(sloc.z, 2)))
        self._frame_counter += 1

    def collect_metrics(self, frame, ctx) -> None:
        pass

    def is_done(self, ctx) -> bool:
        """Done when the ego has stopped, or when there is no ego left to stop.

        The second case is not a normal ending and does not look like one: a
        scenario destroys the actors it spawned when it ends, and if that
        happens while the client is still braking, the stop it is waiting for
        can never arrive. The run then sat until the active limit ended it,
        which reads as a hang rather than as a scenario that finished first.
        Seen on the distributed tier, where the brake lands later: the server
        issued it, the scenario exited, and the client waited 120 s for an ego
        the simulator had already removed.

        Nothing is checked unless the client is still waiting, and nothing is
        recorded when the ego is there, so a run that ends the way it should
        behaves and reports exactly as it did before.
        """
        if self._done:
            return True
        # Rate-limited in ego_removed, because the answer is a round trip to the
        # simulator and this runs on the measured path. Not gated on the brake
        # having been engaged: the case this exists for is a run whose reply
        # never arrives at all, so the client is never in that state.
        gone, self._ego_checked_at = ego_removed(self.source, self._ego_checked_at)
        if gone:
            self._ego_gone = True
            self._done = True
            self._log(
                "[client] the ego is no longer in the world; ending the run "
                "(the scenario ended before the stop)"
            )
            return True
        return False

    def message_stats(self):
        """Return what this client observed, for the run's link report.

        Latencies are the per-frame round trips the app already records, in
        milliseconds. ``sent`` is frames transmitted and ``received`` is frames
        answered, so a link that drops messages shows up as unanswered frames
        rather than being invisible.
        """
        from cadet.metrics.link_report import REPLY_ON_EVENT, MessageStats

        sent = 0
        received = 0
        latencies = []
        for record in self.metrics.records.values():
            send_t = record.get("frame_send_time")
            recv_t = record.get("server_response_receive_time")
            if send_t is None:
                continue  # captured but never transmitted
            sent += 1
            if recv_t is None:
                continue  # transmitted and never answered
            received += 1
            latencies.append((recv_t - send_t) * 1000.0)
        # This server answers only when it issues a brake, so unanswered frames
        # are the normal case and not a link measurement. The latencies that do
        # exist are the brake-command round trips, which is the number that
        # matters for this app.
        return MessageStats(
            sent=sent,
            received=received,
            latencies_ms=latencies,
            reply_policy=REPLY_ON_EVENT,
        )

    def finish(self, ctx) -> None:
        if getattr(self, "_rec", None) is not None:
            self._rec.close()
        if self._viz is not None:
            self._viz.close()
        self.metrics.write()
        s = self.metrics.summary()
        self._log("[client] ==== FINAL REPORT ====")
        self._log("[client] frames_sent={} brake_t={} stop_t={}".format(
            s["frames_sent"], s["brake_time"], s["stop_time"]))


@register_app
class BrakingApp(AppPlugin):
    """Open-loop emergency braking app plugin."""

    name = "braking"
    params_model = BrakingParams

    @staticmethod
    def client_success(client_app):
        """Frames sent *and* a brake decision received.

        Stronger than the default. This experiment measures when the ego brakes,
        so a run that streamed frames and never got a decision back measured
        nothing - the number it exists to produce is absent, whatever the exit
        code says.
        """
        if getattr(client_app, "_ego_gone", False):
            # The measurement this experiment exists for is the brake and the
            # stop that follows it. An ego removed before the stop leaves the
            # second half absent, so the run is not reported as a success - and
            # the reason is named, because "no stop recorded" on its own reads
            # as a braking failure rather than as a scenario that ended first.
            return False, "the ego was removed from the world before it stopped"
        stats = client_app.message_stats()
        if stats is None or stats.sent < 1:
            return False, "client transmitted no frames"
        if stats.received < 1:
            return False, "client sent {} frames and received no brake decision".format(
                stats.sent
            )
        return True, "sent={} decisions={}".format(stats.sent, stats.received)

    @staticmethod
    def sim_hooks(config, model_search_dirs):
        """ScenarioRunner hook: it spawns the ego the CARLA client attaches to.

        By default the command is the original ``run_scenario.sh``'s, built from
        the config by :func:`cadet.sim.scenario_command.scenario_runner_command`
        - so a run reloads the world and reaches the configured simulator
        without an operator having to remember either.

        ``CADET_SCENARIO_RUNNER_CMD`` overrides it for a tree whose CLI differs,
        with ``{host}`` replaced by ``source.carla.host``; ``CADET_SR_READY``
        overrides the readiness marker. If neither the env var nor the config
        names a scenario, no hook is returned (e.g. for non-CARLA use) and the
        orchestrator just starts the server + client.
        """
        import os
        import shlex

        from cadet.sim.scenario_command import READY_MARKER, scenario_runner_command

        cmd = os.environ.get("CADET_SCENARIO_RUNNER_CMD")
        if cmd:
            # `{host}` only means anything for a CARLA source, and only a
            # CARLA source has the block: the guard is the kind, not whether an
            # attribute happens to exist, and the value is the schema's.
            if config.source.kind != "carla":
                raise RuntimeError(
                    "CADET_SCENARIO_RUNNER_CMD names a scenario, which needs "
                    "source.kind: carla (got {!r})".format(config.source.kind)
                )
            argv = shlex.split(cmd.replace("{host}", str(config.source.carla.host)))
        else:
            argv = scenario_runner_command(config)
        if not argv:
            return []
        from cadet.core.launcher import LaunchSpec

        marker = os.environ.get("CADET_SR_READY", READY_MARKER)
        scenario = getattr(config.source, "scenario", None)
        # The scenario file reads these instead of importing the original pylot
        # package, so the swept vehicle/object axes reach it without a second
        # source of truth.
        env = {}
        if scenario is not None:
            if scenario.ego_vehicle:
                env["CADET_SCENARIO_VEHICLE"] = str(scenario.ego_vehicle)
            if scenario.target_object:
                env["CADET_SCENARIO_OBJECT"] = str(scenario.target_object)
        return [
            LaunchSpec(
                name="scenario_runner",
                argv=argv,
                env=env or None,
                readiness_marker=marker,
                readiness_timeout_s=float(os.environ.get("CADET_SR_TIMEOUT", "120")),
            )
        ]

    @staticmethod
    def build_server(config, model_search_dirs, logger=None) -> BrakingServerApp:
        return BrakingServerApp(config, model_search_dirs, logger=logger)

    @staticmethod
    def build_client(
        config, run_id, rep, csv_path, logger=None, node_name=None
    ) -> BrakingClientApp:
        return BrakingClientApp(config, run_id, rep, csv_path, logger=logger)
