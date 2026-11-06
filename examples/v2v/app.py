"""V2V emergency-braking app — decision core (CADET example, not framework).

What this is for:
    The vehicle-to-vehicle braking example: a follower ego reacts to a scripted
    leader. The server gates braking on a monocular range estimate of the lead
    vehicle and its time-to-collision (TTC). This module ports the *pure* decision
    logic from the original ``carla_wrapper/v2v_braking_server.py`` so it can be
    unit-tested with no CARLA and reused by both the flat and pipeline forms.

What it contains:
    - :class:`V2VParams`: the app's typed parameters (TTC/distance gate knobs,
      per-vehicle brake controllers, braking mode), declared with the app.
    - Pure helpers :func:`focal_length_pixels`, :func:`estimate_range_m`,
      :func:`nearest_lead_range_m`; :class:`V2VBrakeGate`, the per-frame TTC +
      distance brake decision (a direct port of the original server's frame loop).
    - :class:`V2VServerApp`: YOLO + the TTC/distance gate on received frames.
    - :class:`V2VClientApp`: the follower ego (cruise, stream, brake on command).
    - :class:`V2VApp`: the plugin, registered on import.

Deviation policy:
    By policy CADET reproduces the original 1:1 by default. The two network delays
    (leader->follower intent, ego->server) are modelled as CADET **links** whose
    default reproduces the original timing exactly, not as ``time.sleep`` in the
    app. The ego->server link is in the example config; the server therefore does
    NOT re-add the original's per-frame ``time.sleep(nw_delay_s)``.

Not yet here: the ``leader`` node + leader->follower
intent link (needs a launched leader process and a second inbound transport on the
follower - roles and orchestrator wiring; the transport primitive
``MultiConnectionEndpoint`` already exists). Until it lands, the follower brakes on
the server's perception decision. The pipeline-stage form is also still to come.
"""

import os
import time
from queue import Empty, Queue
from threading import Thread
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Literal

from cadet.core.apps import AppPlugin, register_app
from cadet.core.codec import decode as wire_decode
from cadet.core.codec import encode as wire_encode
from cadet.core.model_registry import ModelRegistry, device_for_models, torch_device_for
from cadet.metrics import MetricsLog

# Lead-vehicle real width (m) used for the monocular range estimate; a physical
# constant of the scenario's lead car (Tesla Model 3-ish), overridable per config.
DEFAULT_CAR_WIDTH_M = 1.85


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BrakeController(_Strict):
    """Per-vehicle deceleration controller gains (same shape as the braking app)."""

    target_decel: float
    max_decel: float
    kp: float
    max_ramp: float


class RoiBand(_Strict):
    """Pixel region-of-interest band used to select the relevant lead vehicle."""

    cx_min: int
    cx_max: int


def _v2v_controllers() -> Dict[str, BrakeController]:
    # Ported from the original per-vehicle brake tuning (shared with the braking app).
    return {
        "audi.tt": BrakeController(target_decel=7.0, max_decel=7.3, kp=0.2, max_ramp=0.05),
        "carlamotors.carlacola": BrakeController(
            target_decel=3.0, max_decel=3.0, kp=0.45, max_ramp=0.15
        ),
        "kawasaki.ninja": BrakeController(target_decel=6.0, max_decel=6.0, kp=0.2, max_ramp=0.05),
    }


class V2VParams(_Strict):
    """Parameters for the V2V follower-braking app (defaults = original values)."""

    plugin: Literal["v2v"] = "v2v"
    controllers: Dict[str, BrakeController] = Field(default_factory=_v2v_controllers)
    target_classes: List[str] = Field(
        default_factory=lambda: ["car", "motorcycle", "bicycle", "person"]
    )
    conf_thresh: float = 0.30
    roi: RoiBand = Field(default_factory=lambda: RoiBand(cx_min=300, cx_max=500))
    # TTC gate. Defaults are the original's *active* values: the server reads
    # ttc_thresh_s from v2v_braking_params (2.0), so its getattr fallback (3.0) never
    # applies. The 1:1 rule tracks what actually ran, so the default here is 2.0.
    ttc_thresh_s: float = 2.0
    k_ttc_frames: int = 2
    alpha_ttc: float = 0.30
    # Distance gate (original defaults).
    car_width_m: float = DEFAULT_CAR_WIDTH_M
    d_thresh_m: float = 24.0
    # The original only let the distance gate fire after this much sim time, so the
    # follower does not brake on the lead while it is still far/at cruise.
    d_gate_after_s: float = 12.0
    camera_fov_deg: float = 60.0
    # Control rate, mirroring the braking app (default = original per_loop).
    control_rate: Literal["per_loop", "per_tick"] = "per_loop"
    # Which signal brakes the follower (original default "v2v"). In "v2v" the
    # follower brakes on the leader's *intent*, delivered in-process v2v_delay_s
    # after the leader starts braking (exactly as the original single-process
    # client did); in "perception" it brakes only on the server's TTC decision. The
    # server runs perception either way; the mode gates whether the follower also
    # honours the intent. (The leader-as-node + real network intent link is an
    # opt-in deviation, not implemented here.)
    braking_mode: Literal["v2v", "perception"] = "v2v"
    # In-process leader->follower intent delay (original v2v_delay_s = 0.02 s). This
    # is the DEFAULT mechanism: the one client that drives both vehicles schedules
    # the intent at v1_brake_time + v2v_delay_s on the *wall* clock, as the original
    # did (sim_time = time.time() - start). Not a network link by default.
    v2v_delay_s: float = 0.02
    # Leader (V1) scripted motion, ported from the original client: cruise at
    # leader_cruise_mps until the leader passes leader_brake_y, then brake.
    leader_cruise_mps: float = 8.94      # ~20 mph, the original V1 cruise speed
    leader_brake_y: float = -20.0        # world-Y past which the leader brakes
    leader_controller: BrakeController = Field(
        default_factory=lambda: BrakeController(
            target_decel=7.0, max_decel=7.3, kp=0.2, max_ramp=0.05
        )
    )
    # From the original (default): the original server wrote an annotated JPEG
    # per processed frame. Written by default under the run's output tree.
    save_annotated_frames: bool = True
    # From the original (default): the original V2V client rendered a pygame
    # visualiser every control tick and logged per-tick telemetry (both cars' speed/
    # location + the headway gap), which set its loop rate (~10 fps). On by default so
    # a CADET run reproduces that loop rate (and so the per_loop brake dynamics and
    # frame count); set false for a lean, faster loop. Mirrors the braking app.
    client_visualizer: bool = True
    original_client_loop: bool = True


# ---------------------------------------------------------------------------
# Pure helpers (no CARLA / no torch) — unit tested
# ---------------------------------------------------------------------------
def focal_length_pixels(img_width_px: int, fov_deg: float) -> float:
    """Pinhole focal length in pixels from the horizontal FOV and image width."""
    fov_rad = np.deg2rad(fov_deg)
    return (img_width_px / 2.0) / np.tan(fov_rad / 2.0)


def estimate_range_m(w_pixels: float, f_pixels: float, real_width_m: float) -> float:
    """Monocular range (m) from a lead vehicle's apparent pixel width."""
    w_pixels = max(1.0, float(w_pixels))
    return (f_pixels * real_width_m) / w_pixels


def nearest_lead_range_m(
    boxes: List[Tuple[int, float, float, float]],
    target_ids: List[int],
    roi_cx_min: float,
    roi_cx_max: float,
    conf_thresh: float,
    f_pixels: float,
    car_width_m: float,
) -> Optional[float]:
    """Range (m) to the nearest in-ROI target lead, or None if there is none.

    ``boxes`` are ``(class_id, confidence, center_x, width_px)``. A box counts when
    its class is a target, its centre_x is within the ROI band and its confidence
    clears the threshold; the smallest estimated range among those wins. Ported
    from the original server's detection loop.
    """
    best_z = None
    for cls_id, conf, cx, w_px in boxes:
        if cls_id in target_ids and roi_cx_min <= cx <= roi_cx_max and conf > conf_thresh:
            z = estimate_range_m(w_px, f_pixels, car_width_m)
            if best_z is None or z < best_z:
                best_z = z
    return best_z


class V2VBrakeGate:
    """Per-frame follower brake decision from the lead-vehicle range (pure).

    A direct port of ``v2v_braking_server.py``'s frame loop. Feed it the nearest
    in-ROI lead range each frame (metres, or None when no lead is seen), the frame
    time and the elapsed sim time; it returns True once braking should engage. Two
    gates, exactly as the original:

    - **distance**: the lead is within ``d_thresh_m`` and closing, once sim time has
      passed ``d_gate_after_s``;
    - **TTC**: an EMA-smoothed time-to-collision stays below ``ttc_thresh_s`` for
      ``k_ttc_frames`` consecutive frames.

    Closing speed is estimated from the range change between frames and clamped to
    be non-negative (as the original did). State (previous range/time, the EMA and
    the below-threshold counter) advances only on frames that have a lead estimate.
    """

    def __init__(
        self,
        ttc_thresh_s: float = 2.0,
        k_ttc_frames: int = 2,
        alpha_ttc: float = 0.30,
        d_thresh_m: float = 24.0,
        d_gate_after_s: float = 12.0,
    ):
        self.ttc_thresh_s = ttc_thresh_s
        self.k_ttc_frames = k_ttc_frames
        self.alpha_ttc = alpha_ttc
        self.d_thresh_m = d_thresh_m
        self.d_gate_after_s = d_gate_after_s
        self._prev_z: Optional[float] = None
        self._prev_t: Optional[float] = None
        self._ema_ttc: Optional[float] = None
        self._ttc_below = 0

    def update(self, best_z: Optional[float], now_t: float, sim_time: float) -> bool:
        """Advance the gate one frame; return True once braking should engage."""
        if best_z is None:
            # No lead estimate this frame: the original advanced no state and did
            # not brake on it.
            return False
        should_brake = False
        if self._prev_z is not None and self._prev_t is not None:
            dt = max(1e-3, now_t - self._prev_t)
            v_rel = max(0.0, (self._prev_z - best_z) / dt)  # closing speed only (>=0)
            if best_z < self.d_thresh_m and v_rel > 0.0 and sim_time > self.d_gate_after_s:
                should_brake = True
            if v_rel > 1e-3:
                ttc = best_z / v_rel
                self._ema_ttc = (
                    ttc if self._ema_ttc is None
                    else self.alpha_ttc * ttc + (1 - self.alpha_ttc) * self._ema_ttc
                )
                if self._ema_ttc < self.ttc_thresh_s:
                    self._ttc_below += 1
                else:
                    self._ttc_below = 0
                if self._ttc_below >= max(1, int(self.k_ttc_frames)):
                    should_brake = True
            else:
                self._ttc_below = 0  # not closing; reset the TTC confirmation
        self._prev_z, self._prev_t = best_z, now_t
        return should_brake


# ---------------------------------------------------------------------------
# Annotated-frame helper (ported from the original server's draw_detections)
# ---------------------------------------------------------------------------
def save_annotated_detection(rgb, result, class_names, out_dir, frame_id) -> None:
    """Write an annotated detection JPEG for one frame; on by default.

    Draws every detected box with its class/confidence label and saves
    ``frame_<id>.jpg`` under ``out_dir``, as the original ``v2v_braking_server``
    did for every processed frame.
    """
    import cv2

    img = rgb.copy()
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img, "{} {:.2f}".format(class_names[cls], conf), (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
    out_path = os.path.join(out_dir, "frame_{:05d}.jpg".format(frame_id))
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def resolve_target_class_ids(class_names: Dict[int, str], target_classes: List[str]) -> List[int]:
    """Map target class names to detector class ids (substring match, as the original)."""
    wanted = [t.lower() for t in target_classes]
    return [k for k, v in class_names.items() if any(t in v.lower() for t in wanted)]


# ---------------------------------------------------------------------------
# Server app (RSU-style detection + TTC/distance gate)
# ---------------------------------------------------------------------------
class V2VServerApp:
    """Runs YOLO on received follower frames and brakes on the TTC/distance gate.

    A port of ``v2v_braking_server.py``'s ``frame_receiver_server`` onto the generic
    server runtime, reusing the pure decision core (:class:`V2VBrakeGate`,
    :func:`nearest_lead_range_m`). One behavioural difference from the original, by
    policy: the original's per-frame ``time.sleep(nw_delay_s)`` ego->server latency is
    NOT applied here - it is modelled by the ``ego_to_server`` link in the config, so
    the delay lives in transit rather than in the server loop (see the module
    docstring). ``sim_time`` for the distance gate is
    measured from the first frame handled, as the original measured it from the
    connection's accept time.
    """

    def __init__(self, config, model_search_dirs: List[str], logger=None):
        self.cfg = config
        self.params = V2VParams.model_validate(config.app.params)
        self.search_dirs = model_search_dirs
        self._log = logger or (lambda m: None)
        self._model = None
        self._device = None
        self._target_ids: List[int] = []
        self._class_names: Dict[int, str] = {}
        self._conn_start: Optional[float] = None
        self._gate = V2VBrakeGate(
            ttc_thresh_s=self.params.ttc_thresh_s,
            k_ttc_frames=self.params.k_ttc_frames,
            alpha_ttc=self.params.alpha_ttc,
            d_thresh_m=self.params.d_thresh_m,
            d_gate_after_s=self.params.d_gate_after_s,
        )
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
        self._device = torch_device_for(self.cfg)
        self._log("[server] loading model {} ({}) on {}".format(
            self.cfg.models.model, res.path, self._device))
        self._model = YOLO(res.path)
        self._class_names = self._model.names
        self._target_ids = resolve_target_class_ids(self._class_names, self.params.target_classes)
        if not self._target_ids:
            raise RuntimeError("none of the target classes are known to the model")
        if self._save_annotated:
            os.makedirs(self._annotated_dir, exist_ok=True)
            self._log("[server] annotated detections -> {}".format(self._annotated_dir))
        from cadet.pipeline.record import open_recorder

        self._rec = open_recorder("detections")
        # The full gate parameter set, so an offline replay can reconstruct the
        # exact V2VBrakeGate (parity L1). best_z + sim_time are recorded per frame.
        self._rec.record(
            meta=True,
            target_ids=[int(i) for i in self._target_ids],
            roi_cx_min=self.params.roi.cx_min,
            roi_cx_max=self.params.roi.cx_max,
            conf_thresh=self.params.conf_thresh,
            ttc_thresh_s=self.params.ttc_thresh_s,
            k_ttc_frames=self.params.k_ttc_frames,
            alpha_ttc=self.params.alpha_ttc,
            d_thresh_m=self.params.d_thresh_m,
            d_gate_after_s=self.params.d_gate_after_s,
        )

    def warmup(self) -> None:
        """Framework-invoked: one dummy inference so the model JITs before frames."""
        cam = self.cfg.source.camera
        dummy = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
        t0 = time.time()
        self._model.predict(dummy, verbose=False, device=self._device)
        self._log("[server] warmup predict took {:.3f}s on {}".format(
            time.time() - t0, self._device))

    def handle(self, payload: bytes, ctx=None) -> Optional[bytes]:
        message = wire_decode(payload)
        if self._conn_start is None:
            self._conn_start = time.time()
        frame_id = message.get("frame_id", -1)
        rgb = message["rgb_frame"]
        image_size = message.get("image_size", (rgb.shape[1], rgb.shape[0]))
        camera_fov = message.get("camera_fov", self.params.camera_fov_deg)
        t_start = time.time()
        detections = self._model.predict(rgb, verbose=False, device=self._device)

        width_px = image_size[0]
        f_px = focal_length_pixels(width_px, camera_fov)
        boxes = []
        for det in detections[0].boxes:
            cls_id = int(det.cls[0])
            conf = float(det.conf[0])
            bbox = det.xyxy[0].cpu().numpy()
            cx = float((bbox[0] + bbox[2]) / 2.0)
            w_px = float(bbox[2] - bbox[0])
            boxes.append((cls_id, conf, cx, w_px))

        if self._save_annotated:
            save_annotated_detection(
                rgb, detections[0], self._class_names, self._annotated_dir, frame_id)

        best_z = nearest_lead_range_m(
            boxes, self._target_ids, self.params.roi.cx_min, self.params.roi.cx_max,
            self.params.conf_thresh, f_px, self.params.car_width_m,
        )
        now_t = time.time()
        sim_time = now_t - self._conn_start
        should_brake = self._gate.update(best_z, now_t, sim_time)

        # Full precision (no rounding): the offline replay reruns the gate on these
        # exact values, so any rounding here could flip a borderline TTC decision.
        self._rec.record(
            frame_id=frame_id,
            best_z=(None if best_z is None else float(best_z)),
            sim_time=float(sim_time),
            brake=bool(should_brake),
        )

        if should_brake:
            reply = {
                "brake": True,
                "server_processing_time": time.time() - t_start,
                "frame_id": frame_id,
            }
            self._log("[server] BRAKE issued at frame {} (Z~{})".format(
                frame_id, "NA" if best_z is None else round(best_z, 1)))
            return wire_encode(reply)
        return None

    def stop_after_reply(self) -> bool:
        return True

    def teardown(self, ctx=None) -> None:
        if getattr(self, "_rec", None) is not None:
            self._rec.close()


# ---------------------------------------------------------------------------
# Follower client app (ego: cruise, stream, brake on command)
# ---------------------------------------------------------------------------
MPS_PER_MPH = 0.44704
MPS_TO_MPH = 2.23694
STOP_SPEED_MPS = 0.1  # ego is "stopped" below this (original threshold)
SPEED_KICK_Y = 100.0  # ego location gate for the one-time velocity kick (original)


def target_speed_mps(speed_str: str) -> float:
    """Convert a speed string like ``20mph`` / ``36kph`` to metres per second."""
    if speed_str.endswith("kph"):
        return float(speed_str[:-3]) / 3.6
    if speed_str.endswith("mph"):
        return float(speed_str[:-3]) * MPS_PER_MPH
    raise ValueError("speed must end with 'kph' or 'mph', got {!r}".format(speed_str))


def pid_throttle(
    current_speed: float, target_speed: float, kp: float = 0.6, deadband: float = 1.0
) -> float:
    """Pre-brake cruise throttle (ported from the original V2V client)."""
    error = target_speed - current_speed
    if abs(error) < deadband:
        return 0.4
    return max(0.0, min(kp * error, 1.0))


class _DecelController:
    """Follower (V2) deceleration controller, ported verbatim from the original."""

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


class V2VClientApp:
    """One client that drives BOTH vehicles and runs the in-process intent shim.

    A faithful port of ``v2v_braking.py``: a single process drives the follower ego
    (V2) *and* the scripted leader (V1), exactly as the original did (the CADET
    scenario only spawns the two cars and idles). The leader cruises at
    ``leader_cruise_mps`` until it passes ``leader_brake_y``, then brakes; when it
    starts braking the client schedules the follower's *intent* for
    ``v1_brake_time + v2v_delay_s`` on the WALL clock (``sim_time = time.time() -
    start``, the original's clock) and delivers it at the top of a later loop as a
    brake command. The follower also streams frames to the server; in ``v2v`` mode
    whichever brake reason arrives first wins (normally the intent), in
    ``perception`` mode only the server's TTC decision brakes it.

    The default is this in-process mechanism. Modelling the leader as its own node
    with a real network intent link is an opt-in deviation, not implemented
    here.
    """

    def __init__(self, config, run_id: str, rep: int, csv_path: str, logger=None):
        self.cfg = config
        self.params = V2VParams.model_validate(config.app.params)
        self.source = None
        self.transport = None
        self._log = logger or (lambda m: None)
        ego = config.source.scenario.ego_vehicle
        self.controller = _DecelController(**self.params.controllers[ego].model_dump())
        self.target_mps = target_speed_mps(config.source.scenario.ego_speed)
        self.metrics = MetricsLog(csv_path, run_id, rep)
        self._csv_path = csv_path
        self._camera_fov = config.source.camera.fov
        self._image_size = (config.source.camera.width, config.source.camera.height)
        # Leader (V1) scripted control + the in-process intent shim.
        self._mode = self.params.braking_mode
        self._v2v_delay_s = self.params.v2v_delay_s
        self._leader_cruise_mps = self.params.leader_cruise_mps
        self._leader_brake_y = self.params.leader_brake_y
        self._leader_ctrl = _DecelController(**self.params.leader_controller.model_dump())

        self._start = None
        self._frame_counter = 0
        self._cur_fid = None
        self._transmit = True
        self._brake_engaged = False
        self._brake_recorded = False
        self._follower_stopped = False
        self._follower_stop_loc = None
        self._first_brake_sim = None
        self._set_speed_once = False
        self._speed_forced = False
        self._last_speed_mph = 0.0
        self._trigger_reason = None
        self._control_rate = self.params.control_rate
        self._last_ctrl_frame = None
        self._replies: "Queue" = Queue()
        # Leader state + the intent schedule (wall-clock, as the original).
        self._v1_brake_time = None
        self._v1_stop_time = None
        self._v1_stop_loc = None
        self._v2v_emit_time = None
        self._v2v_pending_time = None
        self._v2v_sent = False
        self._v2v_recv_time = None
        # Parity metrics the user asked for: closest headway + loop rate.
        self._min_gap = None
        self._loop_ticks = 0
        self._last_sim_time = 0.0
        self._post_brake_timeout_s = 6.0  # original safety net
        # From the original: the original client's visualiser + per-tick work (sets loop rate).
        self._viz_enabled = self.params.client_visualizer
        self._orig_loop = self.params.original_client_loop
        self._viz_dir = os.path.join(os.path.dirname(csv_path), "client_frames")
        self._viz = None

    def setup(self, ctx) -> None:
        self.source = ctx.source
        self.transport = ctx.transport
        from cadet.pipeline.record import open_recorder

        self._rec = open_recorder("control")  # parity-L1; no-op unless $CADET_RECORD_DIR set
        cam = getattr(self.cfg.source, "camera", None)
        if self._viz_enabled and cam is not None:
            from cadet.roles.client_visualizer import ClientVisualizer

            # save_frames=False: the original V2V client ran with save_frames=False, so
            # it rendered the visualiser but wrote NO per-frame PNG. (Braking's original
            # did save, which is why that port saves; V2V must not - the disk I/O is the
            # main reason an earlier V2V port looped slower than the original.)
            self._viz = ClientVisualizer(
                cam.width, cam.height, self._viz_dir, save_frames=False, logger=self._log)
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
        # In-process V2V intent (the default mechanism): once the leader has braked
        # and v2v_delay_s (wall clock) has elapsed, deliver the intent as a brake
        # command - exactly as the original single-process client did via its
        # command_queue, checked at the top of the loop with a schedule set on an
        # earlier tick. v2v_delay_s is smaller than a loop iteration, so (as in the
        # original) this fires on the loop after the leader brakes.
        if (self._mode == "v2v" and self._v2v_pending_time is not None
                and not self._v2v_sent and not self._brake_engaged and self._start is not None):
            if (time.time() - self._start) >= self._v2v_pending_time:
                self._v2v_sent = True
                self._v2v_recv_time = time.time() - self._start
                return {"brake": True, "reason": "v2v"}
        try:
            return self._replies.get_nowait()
        except Empty:
            return None

    def translate_action(self, frame, reply, ctx) -> None:
        if self._start is None:
            self._start = time.time()
        now = time.time()
        if self._control_rate == "per_tick":
            snap = self.source.snapshot() if hasattr(self.source, "snapshot") else None
            if snap is not None:
                tick_frame, sim_elapsed, _dt = snap
                if tick_frame == self._last_ctrl_frame:
                    return
                self._last_ctrl_frame = tick_frame
                ctrl_t = sim_elapsed
            else:
                ctrl_t = now
        else:
            ctrl_t = now
        sim_time = now - self._start
        self._last_sim_time = sim_time
        speed_mps = self.source.vehicle_speed_mps()
        self._last_speed_mph = speed_mps * MPS_TO_MPH
        # Read each actor's pose ONCE per tick and reuse it, as the original did (its
        # _telemetry_fields computed ego/leader pose once per loop). Re-reading them
        # several times a tick is extra RPC round-trips that slow the loop below the
        # original's rate.
        ego_loc = self.source.vehicle_location()
        v1_loc = self.source.obstacle_location()
        throttle, steer, brake = 0.0, 0.0, 0.0

        if reply is not None and not self._brake_engaged:
            self._brake_engaged = True
            self._transmit = False
            brake_loc = (round(ego_loc.x, 2), round(ego_loc.y, 2), round(ego_loc.z, 2))
            if not self._brake_recorded:
                self._brake_recorded = True
                self._first_brake_sim = sim_time
                self._trigger_reason = reply.get("reason", "perception")
                self.metrics.mark_brake(
                    reply.get("frame_id"), reply.get("server_processing_time"), sim_time, brake_loc,
                    recv_wall=now,
                )
                self._log("[client] FOLLOWER BRAKE ({}) at t={:.2f} {}".format(
                    self._trigger_reason, sim_time, brake_loc))

        if self._brake_engaged:
            self.controller.update_decel(speed_mps, ctrl_t)
            brake = self.controller.step()
            throttle = 0.0
            if not self._follower_stopped and speed_mps < STOP_SPEED_MPS:
                self._follower_stop_loc = (
                    round(ego_loc.x, 2), round(ego_loc.y, 2), round(ego_loc.z, 2))
                self.metrics.mark_stop(sim_time, self._follower_stop_loc)
                self._log("[client] FOLLOWER STOP at t={:.2f} {}".format(
                    sim_time, self._follower_stop_loc))
                self._follower_stopped = True
        else:
            if ego_loc.y < SPEED_KICK_Y and not self._speed_forced:
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
            loc_y=ego_loc.y,
            brake_pending=bool(self._brake_engaged),
            throttle=throttle,
            steer=steer,
            brake=brake,
        )
        self.source.apply_control(throttle, steer, brake)
        # Drive the leader (V1), reusing the pose read above.
        self._drive_leader(v1_loc, sim_time, now)
        self._loop_ticks += 1
        # The original called log_tick EVERY loop (fresh _telemetry_fields: leader
        # get_velocity + leader get_location + ego get_location + a gap print) and
        # rendered the visualiser every loop. Replicate both so the per-tick load - and
        # so the loop rate and the min-gap it drives - match the original.
        if self._orig_loop:
            self._telemetry_fields()
        if self._viz is not None:
            self._viz.update_frame(frame.image if frame is not None else None)
            self._viz.render(throttle, steer, brake, sim_time)

    def _drive_leader(self, v1loc, sim_time: float, now: float) -> None:
        """Cruise the leader until it passes leader_brake_y, then brake it (ported).

        ``v1loc`` is the leader pose already read this tick (reused, not re-read). On
        the leader's first brake tick, schedule the follower's intent at
        ``v1_brake_time + v2v_delay_s`` (wall clock), as the original did.
        """
        if v1loc is None:
            return
        if v1loc.y > self._leader_brake_y:
            # The original's _set_speed_along_heading read the leader velocity and
            # printed it before setting the cruise target; replicate that per-tick read.
            v1_speed = self.source.obstacle_speed_mps()
            self._log("V1 Current speed: {}".format(v1_speed))
            self.source.set_obstacle_target_velocity(self._leader_cruise_mps)
            return
        if self._v1_brake_time is None:
            self._v1_brake_time = sim_time
            if self._mode == "v2v" and self._v2v_pending_time is None:
                self._v2v_emit_time = self._v1_brake_time
                self._v2v_pending_time = self._v1_brake_time + self._v2v_delay_s
                self._v2v_sent = False
            self._log("[client] LEADER BRAKE at t={:.2f}".format(sim_time))
        v1_speed = self.source.obstacle_speed_mps()
        if self._v1_stop_time is None and v1_speed < 0.5:
            self._v1_stop_time = sim_time
            self._v1_stop_loc = (round(v1loc.x, 2), round(v1loc.y, 2), round(v1loc.z, 2))
            self._log("[client] LEADER STOP at t={:.2f} {}".format(sim_time, self._v1_stop_loc))
        self._leader_ctrl.update_decel(v1_speed, now, alpha=0.35)
        self.source.apply_obstacle_control(0.0, self._leader_ctrl.step())

    def _telemetry_fields(self):
        """Fresh ego + leader reads and the headway-gap print, ported from the original.

        The original's ``_telemetry_fields`` read the leader velocity + location and the
        ego location (3 fresh RPCs) and printed the headway gap; it ran it every loop
        (``log_tick``) and again on each transmit. Replicated here so CADET's per-tick
        RPC load - and the loop rate and min gap that follow from it - match the
        original. Also tracks the minimum gap (the key V2V metric).
        """
        v1loc = self.source.obstacle_location()
        if v1loc is not None:
            self.source.obstacle_speed_mps()  # leader get_velocity (fresh, as the original)
        egoloc = self.source.vehicle_location()
        if v1loc is None:
            self._log("Headway gap:  None")
            return
        gap = v1loc.y - egoloc.y
        if self._min_gap is None or gap < self._min_gap:
            self._min_gap = gap
        self._log("Headway gap:  {}".format(round(gap, 2)))

    def should_transmit(self, ctx) -> bool:
        return self._transmit

    def process_frame(self, frame, ctx) -> bytes:
        fid = self._frame_counter
        self._cur_fid = fid
        capture_time = frame.meta["capture_time"]
        cloc = frame.meta["capture_loc"]
        self.metrics.new_frame(
            fid, capture_time,
            (round(cloc.x, 2), round(cloc.y, 2), round(cloc.z, 2)), self._last_speed_mph,
        )
        # The original's transmit path called _telemetry_fields again (fresh ego+leader
        # reads + gap print) while building the frame row; replicate that per-tick load.
        if self._orig_loop:
            self._telemetry_fields()
        loc = self.source.vehicle_location()
        speed_mps = self.source.vehicle_speed_mps()
        msg = {
            "frame_id": fid,
            "timestamp": capture_time,
            "rgb_frame": frame.image,
            "frame_type": "I",
            "ego_pose": (round(loc.x, 2), round(loc.y, 2), round(loc.z, 2)),
            "ego_speed_mps": float(speed_mps),
            "image_size": self._image_size,
            "camera_fov": float(self._camera_fov),
        }
        return wire_encode(msg)

    def on_frame_sent(self, frame, ctx) -> None:
        send_time = time.time() - self._start
        sloc = self.source.vehicle_location()
        self.metrics.mark_sent(
            self._cur_fid, send_time, (round(sloc.x, 2), round(sloc.y, 2), round(sloc.z, 2)))
        self._frame_counter += 1

    def collect_metrics(self, frame, ctx) -> None:
        pass

    def is_done(self, ctx) -> bool:
        # The original exited once both cars had stopped, or once the follower had
        # stopped and a post-brake timeout elapsed (leader stop given up on).
        if self._first_brake_sim is None:
            return False
        if self._follower_stopped and self._v1_stop_time is not None:
            return True
        timed_out = (self._last_sim_time - self._first_brake_sim) > self._post_brake_timeout_s
        return bool(timed_out and self._follower_stopped)

    def message_stats(self):
        """What the follower observed, for the run's link report (as the braking app)."""
        from cadet.metrics.link_report import REPLY_ON_EVENT, MessageStats

        sent = 0
        received = 0
        latencies = []
        for record in self.metrics.records.values():
            send_t = record.get("frame_send_time")
            recv_t = record.get("server_response_receive_time")
            if send_t is None:
                continue
            sent += 1
            if recv_t is None:
                continue
            received += 1
            latencies.append((recv_t - send_t) * 1000.0)
        return MessageStats(
            sent=sent, received=received, latencies_ms=latencies, reply_policy=REPLY_ON_EVENT)

    def finish(self, ctx) -> None:
        if getattr(self, "_rec", None) is not None:
            self._rec.close()
        if self._viz is not None:
            self._viz.close()
        self.metrics.write()
        s = self.metrics.summary()
        # The V2V parity metrics the braking MetricsLog does not carry (leader brake/
        # stop, closest gap, loop rate), written beside the per-frame CSV so a parity
        # run can read both cars' numbers from one place.
        loop_rate = (self._loop_ticks / self._last_sim_time) if self._last_sim_time > 0 else None
        summary = {
            "leader_brake_t": self._v1_brake_time,
            "follower_brake_t": s.get("brake_time"),
            "leader_stop_t": self._v1_stop_time,
            "leader_stop_loc": self._v1_stop_loc,
            "follower_stop_t": s.get("stop_time"),
            "follower_stop_loc": self._follower_stop_loc,
            "min_gap_m": (None if self._min_gap is None else round(self._min_gap, 2)),
            "frames_sent": s.get("frames_sent"),
            "loop_ticks": self._loop_ticks,
            "loop_rate_hz": (None if loop_rate is None else round(loop_rate, 2)),
            "v2v_emit_t": self._v2v_emit_time,
            "v2v_recv_t": self._v2v_recv_time,
            "trigger_reason": self._trigger_reason,
            "braking_mode": self._mode,
        }
        import json

        try:
            with open(os.path.splitext(self._csv_path)[0] + "_v2v_summary.json", "w") as fh:
                json.dump(summary, fh, indent=2, sort_keys=True)
        except Exception as exc:
            self._log("[client] could not write v2v summary: {}".format(exc))
        self._log("[client] ==== FINAL REPORT ====")
        self._log(
            "[client] leader_brake_t={} follower_brake_t={} min_gap={} frames={} loop_hz={}".format(
                self._v1_brake_time, s.get("brake_time"), summary["min_gap_m"],
                s.get("frames_sent"), summary["loop_rate_hz"]))


@register_app
class V2VApp(AppPlugin):
    """Vehicle-to-vehicle follower-braking app plugin."""

    name = "v2v"
    params_model = V2VParams

    @staticmethod
    def client_success(client_app):
        stats = client_app.message_stats()
        if stats is None or stats.sent < 1:
            return False, "client transmitted no frames"
        # In the default v2v mode the follower brakes on the in-process intent, which
        # is not a server reply, so a recorded brake (intent OR perception) is the
        # success signal; stats.received only sees perception replies.
        braked = client_app.metrics.summary().get("brake_time") is not None
        if not braked and stats.received < 1:
            return False, "client sent {} frames and never braked".format(stats.sent)
        return True, "sent={} braked={} server_replies={}".format(
            stats.sent, braked, stats.received)

    @staticmethod
    def sim_hooks(config, model_search_dirs):
        """ScenarioRunner hook: spawns + drives the leader V1 and the ego the client attaches to."""
        import shlex

        from cadet.core.launcher import LaunchSpec
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
        marker = os.environ.get("CADET_SR_READY", READY_MARKER)
        scenario = getattr(config.source, "scenario", None)
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
    def build_server(config, model_search_dirs, logger=None) -> "V2VServerApp":
        return V2VServerApp(config, model_search_dirs, logger=logger)

    @staticmethod
    def build_client(
        config, run_id, rep, csv_path, logger=None, node_name=None
    ) -> "V2VClientApp":
        return V2VClientApp(config, run_id, rep, csv_path, logger=logger)
