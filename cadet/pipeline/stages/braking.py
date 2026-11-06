"""Braking stages: detector, ROI/consecutive decision, and brake control.

What this is for:
    The stage subset the open-loop emergency-braking app needs, ported verbatim
    from the flat braking app (``examples/braking/flat/app.py``) so the pipeline
    form produces bit-identical decisions and commands. Splitting the flat app into
    stages is what lets detection+decision sit on one node and control on another,
    purely by config.

What it contains:
    - helpers ported from the flat app: :func:`target_speed_mps`,
      :func:`resolve_target_class_ids`, :func:`frame_triggers_brake`,
      :func:`pid_throttle`, and :class:`BrakeControllerState`.
    - extension messages :class:`BrakeDecision`, :class:`ControlInput`,
      :class:`BrakeControlCommand`, registered as wire types at the bottom of
      this module because they cross a node boundary (see
      :mod:`cadet.core.codec`).
    - stages :class:`YoloDetectorStage` (``yolo``), :class:`RoiConsecutiveStage`
      (``roi_consecutive``), :class:`BrakeControlStage` (``brake_controller``).

How it fits into CADET:
    Registered in the stage registry on import. The braking pipeline config places
    ``yolo`` + ``roi_consecutive`` on the server and ``brake_controller`` on the
    client, matching the flat app's split. The detector reads perceived data; the
    control stage reads ego ground-truth telemetry (so it declares that).

How to configure it:
    ``yolo`` resolves its model from ``models.model`` on its node (via the stage
    context). ``roi_consecutive`` params: ``target_classes``, ``roi`` (cx_min/cx_max),
    ``conf_thresh``, ``detection_threshold``. ``brake_controller`` params:
    ``controllers`` (per-vehicle gains), ``ego_vehicle``, ``ego_speed``.

Derived from the flat open-loop braking app, itself derived from ERDOS Pylot
(Apache-2.0); see the project NOTICE.
"""

import time
from typing import Dict, List, Optional

from cadet.core.codec import register_wire_types
from cadet.pipeline.messages import ControlCommand, Detections, Message
from cadet.pipeline.registry import register_stage
from cadet.pipeline.stage import GROUND_TRUTH, PERCEIVED, Stage

# Unit conversions and gates, ported verbatim from the flat braking app.
MPS_PER_MPH = 0.44704
MPS_TO_MPH = 2.23694
STOP_SPEED_MPS = 0.1  # ego is "stopped" below this speed (original threshold)
SPEED_KICK_Y = 90.0  # ego location gate at which the one-time velocity kick fires


# --------------------------------------------------------------------------
# Pure helpers (ported verbatim from the flat app)
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


def pid_throttle(current_speed: float, target_speed: float, kp: float = 0.6, deadband: float = 1.0):
    """Pre-brake cruise throttle (ported from the original client)."""
    error = target_speed - current_speed
    if abs(error) < deadband:
        return 0.4
    return max(0.0, min(kp * error, 1.0))


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


# --------------------------------------------------------------------------
# Extension messages
# --------------------------------------------------------------------------
class BrakeDecision(Message):
    """Decision from the server: brake now (emitted only once brake is triggered)."""

    def __init__(self, brake: bool = True, server_processing_time: float = 0.0, **kw):
        super().__init__(**kw)
        self.brake = brake
        self.server_processing_time = server_processing_time


class ControlInput(Message):
    """Per-tick client input to the control stage: ego telemetry + brake flag.

    Assembled by the client runtime each tick: ``speed_mps`` and ``loc_y`` are ego
    ground truth; ``t`` is the control-loop wall time; ``brake_pending`` is whether a
    :class:`BrakeDecision` has arrived from the server.
    """

    def __init__(
        self,
        speed_mps: float = 0.0,
        loc_y: float = 0.0,
        t: float = 0.0,
        brake_pending: bool = False,
        **kw,
    ):
        super().__init__(**kw)
        self.speed_mps = speed_mps
        self.loc_y = loc_y
        self.t = t
        self.brake_pending = brake_pending


class BrakeControlCommand(ControlCommand):
    """A control command plus the actuation side-effects the runtime must apply.

    The control *decision* is pure and replayable; the flags tell the client
    runtime which CARLA side effects to perform (so they stay out of the stage):
    ``speed_kick`` (set the one-time target velocity), ``engage_brake`` (first brake
    tick: destroy the obstacle, record the brake location), ``stopped`` (ego has
    come to rest).
    """

    def __init__(
        self, speed_kick: bool = False, engage_brake: bool = False, stopped: bool = False, **kw
    ):
        super().__init__(**kw)
        self.speed_kick = speed_kick
        self.engage_brake = engage_brake
        self.stopped = stopped


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
@register_stage("camera")
class CameraSourceStage(Stage):
    """Pipeline head: the ego camera on the client.

    A passthrough that marks where capture is placed (so the boundary into the
    first compute node is explicit). The runtime feeds it the frame from the node's
    configured ``source``; in offline replay it is fed recorded frames. Declared
    ground-truth: it is the raw sensor, not perceived output.
    """

    data_source = GROUND_TRUTH

    def process(self, msg: Message) -> Optional[Message]:
        return msg


def save_annotated_detection(rgb, result, class_names, out_dir, frame_id) -> None:
    """Write an annotated detection JPEG for one frame.

    Draws every detected box with its class/confidence label and saves
    ``frame_<id>.jpg`` under ``out_dir``, as the original server did. Kept here (not
    imported from the flat example) so the framework does not depend on examples.
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


@register_stage("yolo")
class YoloDetectorStage(Stage):
    """Run YOLO on a frame and emit in-image detections ``(class_id, conf, cx)``."""

    data_source = PERCEIVED

    def __init__(self, name, params=None):
        super().__init__(name, params)
        self._model = None
        self._device = None
        self._warm_shape = None  # (h, w, 3) for the warm-up dummy; set in setup
        self._log = lambda m: None
        # From the original (default on): write an annotated JPEG per frame.
        self._save_annotated = bool(self.params.get("save_annotated", True))
        self._annotated_dir = None

    def setup(self, ctx=None) -> None:
        from ultralytics import YOLO

        from cadet.core.model_registry import (
            ModelRegistry,
            device_for_models,
            torch_device_for,
        )

        if ctx is None or ctx.config is None:
            raise RuntimeError("yolo stage needs a StageContext with the resolved config")
        cfg = ctx.config
        registry = ModelRegistry.from_config(cfg.models, device_for_models(cfg))
        res = registry.resolve(cfg.models.model, ctx.model_dirs)
        if res.path is None:
            raise RuntimeError(
                "model artifact {!r} not found under {}".format(res.artifact, ctx.model_dirs)
            )
        self._device = torch_device_for(cfg)
        self._model = YOLO(res.path)
        self._log = ctx.log
        cam = cfg.source.camera
        self._warm_shape = (cam.height, cam.width, 3)
        if self._save_annotated:
            import os

            self._annotated_dir = os.path.join(
                cfg.outputs.root, cfg.experiment.name, cfg.app.plugin, "server_detections"
            )
            os.makedirs(self._annotated_dir, exist_ok=True)
        ctx.log("[stage {}] loaded {} on {}".format(self.name, res.path, self._device))

    def warmup(self) -> None:
        """Run one dummy predict so the model JITs before the first real frame.

        Without this the first live predict pays the full CUDA/cuDNN cold-start
        cost (seconds), and at the camera frame rate a backlog builds on the
        server's inbound queue that never drains, so every later frame is
        processed stale and the brake fires late. Ported from the flat braking
        server's warm-up, which the generic pipeline server invokes per stage.
        """
        import numpy as np

        if self._model is None or self._warm_shape is None:
            return
        t0 = time.time()
        dummy = np.zeros(self._warm_shape, dtype=np.uint8)
        self._model.predict(dummy, verbose=False, device=self._device)
        self._log(
            "[stage {}] warmup predict took {:.3f}s on {}".format(
                self.name, time.time() - t0, self._device
            )
        )

    def class_names(self):
        """The detector's ``id -> name`` map (available after :meth:`setup`)."""
        return self._model.names if self._model is not None else {}

    def process(self, msg: Message) -> Optional[Message]:
        rgb = getattr(msg, "image", None)
        detections = self._model.predict(rgb, verbose=False, device=self._device)
        if self._save_annotated and self._annotated_dir is not None:
            save_annotated_detection(
                rgb, detections[0], self._model.names, self._annotated_dir, msg.frame_id
            )
        boxes = []
        for det in detections[0].boxes:
            cls_id = int(det.cls[0])
            conf = float(det.conf[0])
            bbox = det.xyxy[0].cpu().numpy()
            cx = (bbox[0] + bbox[2]) / 2
            boxes.append((cls_id, conf, cx))
        return Detections(detections=boxes)


@register_stage("roi_consecutive")
class RoiConsecutiveStage(Stage):
    """Brake after ``detection_threshold`` consecutive in-ROI target detections.

    Emits a :class:`BrakeDecision` only on the frame that reaches the threshold
    (returns ``None`` otherwise), matching the flat server's reply-on-brake. The
    detector supplies class ids via ``class_names``; without it (offline replay from
    recorded ``(class_id, conf, cx)`` boxes) ``target_ids`` is taken from params.
    """

    data_source = PERCEIVED

    def __init__(self, name, params=None):
        super().__init__(name, params)
        self._consecutive = 0
        self._target_ids = list(self.params.get("target_ids", []))
        self._roi_min = self.params.get("roi", {}).get("cx_min", 0)
        self._roi_max = self.params.get("roi", {}).get("cx_max", 0)
        self._conf = self.params.get("conf_thresh", 0.3)
        self._threshold = self.params.get("detection_threshold", 3)

    def set_target_ids(self, ids: List[int]) -> None:
        """Set the detector class ids to watch (called once the model is known)."""
        self._target_ids = list(ids)

    def set_class_names(self, class_names) -> None:
        """Resolve the target class ids from a detector's ``id -> name`` map.

        Lets the runtime wire a live detector's labels into this stage without the
        framework knowing braking specifics (it just calls this duck-typed hook).
        """
        self._target_ids = resolve_target_class_ids(
            class_names, self.params.get("target_classes", [])
        )

    def process(self, msg: Message) -> Optional[Message]:
        boxes = getattr(msg, "detections", [])
        found = frame_triggers_brake(
            boxes, self._target_ids, self._roi_min, self._roi_max, self._conf
        )
        self._consecutive = self._consecutive + 1 if found else 0
        if self._consecutive >= self._threshold:
            return BrakeDecision(brake=True, server_processing_time=0.0, frame_id=msg.frame_id)
        return None


@register_stage("brake_controller")
class BrakeControlStage(Stage):
    """Per-tick ego control: cruise to target speed, then decelerate on brake.

    Ported verbatim from the flat client's ``translate_action`` control branch. The
    pure control decision is returned as a :class:`BrakeControlCommand`; CARLA
    side-effects (speed kick, obstacle destroy, stop) are returned as flags for the
    client runtime to apply, so the stage stays replayable with no simulator.
    """

    data_source = GROUND_TRUTH  # reads ego speed/location telemetry

    def __init__(self, name, params=None):
        super().__init__(name, params)
        ego = self.params["ego_vehicle"]
        gains = self.params["controllers"][ego]
        self.controller = BrakeControllerState(
            gains["target_decel"], gains["max_decel"], gains["kp"], gains["max_ramp"]
        )
        self.target_mps = target_speed_mps(self.params["ego_speed"])
        self._brake_engaged = False
        self._stopped = False
        self._set_speed_once = False
        self._speed_forced = False

    def process(self, msg: Message) -> Optional[Message]:
        speed_mps = msg.speed_mps
        now = msg.t
        out = BrakeControlCommand()
        throttle, steer, brake = 0.0, 0.0, 0.0

        if msg.brake_pending and not self._brake_engaged:
            self._brake_engaged = True
            out.engage_brake = True

        if self._brake_engaged:
            self.controller.update_decel(speed_mps, now)
            brake = self.controller.step()
            throttle = 0.0
            if not self._stopped and speed_mps < STOP_SPEED_MPS:
                self._stopped = True
                out.stopped = True
        else:
            if msg.loc_y < SPEED_KICK_Y and not self._speed_forced:
                self._set_speed_once = True
                self._speed_forced = True
            if self._set_speed_once:
                out.speed_kick = True
                self._set_speed_once = False
            else:
                throttle = pid_throttle(speed_mps, self.target_mps)
                brake = 0.0

        out.throttle, out.steer, out.brake = throttle, steer, brake
        return out


# The wire names these stage messages travel under: a brake decision crosses the
# boundary back to the client, and the control inputs/commands cross one when
# control is placed away from the source.
register_wire_types(
    {
        "BrakeDecision": BrakeDecision,
        "ControlInput": ControlInput,
        "BrakeControlCommand": BrakeControlCommand,
    }
)
