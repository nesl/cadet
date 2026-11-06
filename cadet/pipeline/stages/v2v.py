"""V2V stages: lead-range detector and the TTC/distance brake gate.

What this is for:
    The stage subset the V2V follower-braking app needs, ported verbatim from the
    flat V2V app (``examples/v2v/app.py``) so the pipeline form produces identical
    decisions. The follower *control* stage is shared with the braking app
    (``brake_controller``); only perception + the brake gate are V2V-specific.

What it contains:
    - helpers ported verbatim from the flat V2V decision core:
      :func:`focal_length_pixels`, :func:`estimate_range_m`,
      :func:`nearest_lead_range_m`, and :class:`V2VBrakeGate`.
    - extension message :class:`V2VLeadRange` (nearest in-ROI lead range per frame).
    - stages :class:`YoloV2VDetectorStage` (``yolo_v2v``) and
      :class:`V2VTtcGateStage` (``v2v_ttc_gate``), which emits the shared
      :class:`~cadet.pipeline.stages.braking.BrakeDecision`.

How it fits into CADET:
    Registered in the stage registry on import. A V2V pipeline config places
    ``yolo_v2v`` + ``v2v_ttc_gate`` on the server and the shared ``brake_controller``
    on the client. The gate is time-dependent (closing speed and TTC), so each lead
    range carries the frame's ``sim_time``; offline replay feeds the recorded
    ``best_z`` + ``sim_time`` straight into the gate (as the braking L1 replay feeds
    recorded boxes to ``roi_consecutive``), which keeps the decision deterministic.

Derived from the flat V2V app, itself derived from ERDOS Pylot (Apache-2.0); see
the project NOTICE.
"""

from typing import Optional

import numpy as np

from cadet.core.codec import register_wire_types
from cadet.pipeline.messages import Message
from cadet.pipeline.registry import register_stage
from cadet.pipeline.stage import PERCEIVED, Stage
from cadet.pipeline.stages.braking import BrakeDecision, resolve_target_class_ids

# Lead-vehicle real width (m) for the monocular range estimate (as the flat app).
DEFAULT_CAR_WIDTH_M = 1.85


# --------------------------------------------------------------------------
# Pure helpers (ported verbatim from the flat V2V decision core)
# --------------------------------------------------------------------------
def focal_length_pixels(img_width_px: int, fov_deg: float) -> float:
    """Pinhole focal length in pixels from the horizontal FOV and image width."""
    fov_rad = np.deg2rad(fov_deg)
    return (img_width_px / 2.0) / np.tan(fov_rad / 2.0)


def estimate_range_m(w_pixels: float, f_pixels: float, real_width_m: float) -> float:
    """Monocular range (m) from a lead vehicle's apparent pixel width."""
    w_pixels = max(1.0, float(w_pixels))
    return (f_pixels * real_width_m) / w_pixels


def nearest_lead_range_m(
    boxes, target_ids, roi_cx_min, roi_cx_max, conf_thresh, f_pixels, car_width_m
) -> Optional[float]:
    """Range (m) to the nearest in-ROI target lead, or None if there is none.

    ``boxes`` are ``(class_id, confidence, center_x, width_px)``. A box counts when
    its class is a target, its centre_x is within the ROI band and its confidence
    clears the threshold; the smallest estimated range among those wins.
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

    A verbatim port of the flat V2V decision core. Feed it the nearest in-ROI lead
    range each frame (metres, or None), the frame time and the elapsed sim time; it
    returns True once braking should engage (distance gate OR TTC gate). State
    advances only on frames that have a lead estimate.
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
                self._ttc_below = 0
        self._prev_z, self._prev_t = best_z, now_t
        return should_brake


# --------------------------------------------------------------------------
# Extension message
# --------------------------------------------------------------------------
class V2VLeadRange(Message):
    """Nearest in-ROI lead-vehicle range (m) for one frame, or None if none seen.

    ``sim_time`` is the frame's elapsed time, which the gate needs for closing speed
    and the distance-gate arming time (carried on the message so the gate stays pure
    and offline-replayable).
    """

    def __init__(self, best_z: Optional[float] = None, **kw):
        super().__init__(**kw)
        self.best_z = best_z


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
@register_stage("yolo_v2v")
class YoloV2VDetectorStage(Stage):
    """Run YOLO and emit the nearest in-ROI lead range (with the frame's sim_time).

    Like the braking ``yolo`` stage but keeps each box's pixel width and reduces to
    a single lead range, since the V2V gate reasons about range/TTC. ``sim_time`` is
    measured from the first processed frame, mirroring the flat server's
    connection-relative clock.
    """

    data_source = PERCEIVED

    def __init__(self, name, params=None):
        super().__init__(name, params)
        self._model = None
        self._device = None
        self._warm_shape = None
        self._log = lambda m: None
        self._target_ids = list(self.params.get("target_ids", []))
        self._roi_min = self.params.get("roi", {}).get("cx_min", 0)
        self._roi_max = self.params.get("roi", {}).get("cx_max", 0)
        self._conf = self.params.get("conf_thresh", 0.3)
        self._car_width_m = self.params.get("car_width_m", DEFAULT_CAR_WIDTH_M)
        self._fov_deg = self.params.get("camera_fov_deg", 60.0)
        self._f_px = None
        self._save_annotated = bool(self.params.get("save_annotated", True))
        self._annotated_dir = None
        self._t0 = None

    def setup(self, ctx=None) -> None:
        from ultralytics import YOLO

        from cadet.core.model_registry import (
            ModelRegistry,
            device_for_models,
            torch_device_for,
        )

        if ctx is None or ctx.config is None:
            raise RuntimeError("yolo_v2v stage needs a StageContext with the resolved config")
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
        self._f_px = focal_length_pixels(cam.width, self._fov_deg or cam.fov)
        if self._save_annotated:
            import os

            self._annotated_dir = os.path.join(
                cfg.outputs.root, cfg.experiment.name, cfg.app.plugin, "server_detections"
            )
            os.makedirs(self._annotated_dir, exist_ok=True)
        ctx.log("[stage {}] loaded {} on {}".format(self.name, res.path, self._device))

    def warmup(self) -> None:
        import time

        if self._model is None or self._warm_shape is None:
            return
        t0 = time.time()
        dummy = np.zeros(self._warm_shape, dtype=np.uint8)
        self._model.predict(dummy, verbose=False, device=self._device)
        self._log("[stage {}] warmup predict took {:.3f}s on {}".format(
            self.name, time.time() - t0, self._device))

    def set_class_names(self, class_names) -> None:
        """Resolve target class ids from a detector's ``id -> name`` map (as roi_consecutive)."""
        self._target_ids = resolve_target_class_ids(
            class_names, self.params.get("target_classes", [])
        )

    def process(self, msg: Message) -> Optional[Message]:
        import time

        rgb = getattr(msg, "image", None)
        detections = self._model.predict(rgb, verbose=False, device=self._device)
        if self._save_annotated and self._annotated_dir is not None:
            from cadet.pipeline.stages.braking import save_annotated_detection

            save_annotated_detection(
                rgb, detections[0], self._model.names, self._annotated_dir, msg.frame_id
            )
        boxes = []
        for det in detections[0].boxes:
            cls_id = int(det.cls[0])
            conf = float(det.conf[0])
            bbox = det.xyxy[0].cpu().numpy()
            cx = float((bbox[0] + bbox[2]) / 2.0)
            w_px = float(bbox[2] - bbox[0])
            boxes.append((cls_id, conf, cx, w_px))
        best_z = nearest_lead_range_m(
            boxes, self._target_ids, self._roi_min, self._roi_max, self._conf,
            self._f_px, self._car_width_m,
        )
        if self._t0 is None:
            self._t0 = time.time()
        out = V2VLeadRange(best_z=best_z)
        out.sim_time = time.time() - self._t0
        return out


@register_stage("v2v_ttc_gate")
class V2VTtcGateStage(Stage):
    """Brake on the TTC/distance gate; emits a BrakeDecision only on the brake frame.

    Holds a :class:`V2VBrakeGate` built from params. Reads each :class:`V2VLeadRange`'s
    ``best_z`` and ``sim_time`` and returns a :class:`BrakeDecision` once the gate
    fires (None otherwise), matching the flat server's reply-on-brake. ``sim_time``
    is used both as the gate's time axis (for closing speed) and as the distance-gate
    arming clock, so offline replay of recorded ``(best_z, sim_time)`` is exact.
    """

    data_source = PERCEIVED

    def __init__(self, name, params=None):
        super().__init__(name, params)
        self._gate = V2VBrakeGate(
            ttc_thresh_s=self.params.get("ttc_thresh_s", 2.0),
            k_ttc_frames=self.params.get("k_ttc_frames", 2),
            alpha_ttc=self.params.get("alpha_ttc", 0.30),
            d_thresh_m=self.params.get("d_thresh_m", 24.0),
            d_gate_after_s=self.params.get("d_gate_after_s", 12.0),
        )

    def process(self, msg: Message) -> Optional[Message]:
        best_z = getattr(msg, "best_z", None)
        sim_time = msg.sim_time if msg.sim_time is not None else 0.0
        if self._gate.update(best_z, sim_time, sim_time):
            return BrakeDecision(brake=True, server_processing_time=0.0, frame_id=msg.frame_id)
        return None


# The wire name this stage message travels under: a lead range crosses the
# boundary whenever the gate is placed away from the detector.
register_wire_types({"V2VLeadRange": V2VLeadRange})
