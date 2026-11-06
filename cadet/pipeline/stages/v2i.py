"""V2I pipeline stages: infra-assisted emergency braking, as placeable stages.

The flat V2I app (``examples/v2i/app.py``) decides inside one ``RSUServerApp.handle``.
This expresses the same decision as pipeline stages so the builder can place the YOLO
detector and the brake gates on separate nodes and replay them offline:

  - :class:`YoloV2IDetectorStage` (``yolo_v2i``): run YOLO on a frame and reduce it to
    the per-feed observation the gates need - for an ego frame, whether a target person
    is in the ego ROI plus the ego's world Y; for an infra frame, the in-ROI person
    boxes, the image-space crossing band, and the ego's ETA to the crossing.
  - :class:`V2IEgoGateStage` (``v2i_ego_gate``): the ego-camera consecutive-detection
    gate (every mode's ego path).
  - :class:`V2IFusionGateStage` (``v2i_fusion_gate``): the infra fusion gate (v2i modes)
    - a centroid tracker + constant-velocity ped ETA fused with the ego ETA.

Both gates emit the shared :class:`BrakeDecision` only on the brake frame (None
otherwise), matching the flat server's reply-on-brake. The decision helpers and gate
classes are kept in sync with the flat app by ``tests/unit/test_v2i_pipeline.py``,
which asserts the stage gates fire on exactly the same frames as the flat gates (core
must not import examples, so the logic is duplicated here, as for V2V).
"""

from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

from cadet.pipeline.messages import Message
from cadet.pipeline.registry import register_stage
from cadet.pipeline.stage import PERCEIVED, Stage
from cadet.pipeline.stages.braking import BrakeDecision


# ---------------------------------------------------------------------------
# Pure decision helpers (duplicated from examples/v2i/app.py; kept in sync by
# tests/unit/test_v2i_pipeline.py). An ROI is (cx_min, cx_max, cy_min, cy_max);
# cy bounds may be None for an x-only band.
# ---------------------------------------------------------------------------
def center_in_roi(cx: float, cy: float, roi: Tuple) -> bool:
    """True if ``(cx, cy)`` is inside the ROI (rectangle if cy bounds set, else x-band)."""
    cx_min, cx_max, cy_min, cy_max = roi
    if cy_min is not None and cy_max is not None:
        return (cx_min <= cx <= cx_max) and (cy_min <= cy <= cy_max)
    return cx_min <= cx <= cx_max


def ttc_to_crossing(
    ego_x: float, ego_y: float, ego_speed_mps: float, crossing_x: float, crossing_y: float
) -> Optional[float]:
    """Ego ETA (s) to the crossing point = straight-line gap / speed (None if crawling)."""
    if ego_speed_mps <= 1e-3:
        return None
    gap = ((crossing_x - ego_x) ** 2 + (crossing_y - ego_y) ** 2) ** 0.5
    return gap / max(ego_speed_mps, 1e-3)


def ped_eta_from_history(
    hist: List[Tuple[float, float, float]],
    y_target: float,
    min_speed_pxps: float,
    horizon_s: float,
) -> Optional[float]:
    """Image-space ped ETA (s) to a crossing band, constant-velocity on centroid y."""
    if len(hist) < 3:
        return None
    t_vals = [p[0] for p in hist]
    y_vals = [p[2] for p in hist]
    dt = t_vals[-1] - t_vals[0]
    if dt <= 1e-3:
        return None
    vy = (y_vals[-1] - y_vals[0]) / dt  # px/s
    y_now = y_vals[-1]
    dist = y_target - y_now
    moving_toward = (vy > 0 and dist > 0) or (vy < 0 and dist < 0)
    speed_pxps = abs(vy)
    if (not moving_toward) or (speed_pxps < min_speed_pxps):
        return None
    eta = abs(dist) / speed_pxps
    return eta if eta <= horizon_s else None


class CentroidTracker:
    """Single-target centroid tracker for a person already filtered by ROI (verbatim)."""

    def __init__(self, max_miss: int = 5, hist_len: int = 12):
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None
        self.has_target = False
        self.misses = 0
        self.max_miss = max_miss
        self.hist: "Deque[Tuple[float, float, float]]" = deque(maxlen=hist_len)

    def update(self, detections, t_now: Optional[float] = None) -> bool:
        """Advance one frame. ``detections`` are ``(label, conf, (x1,y1,x2,y2))``."""
        if not detections:
            self.misses += 1
            if self.misses > self.max_miss:
                self.cx = self.cy = None
                self.has_target = False
                self.hist.clear()
            return False
        _, _, (x1, y1, x2, y2) = max(detections, key=lambda d: d[1])
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        if self.cx is None:
            self.cx, self.cy = cx, cy
        else:
            self.cx = 0.6 * self.cx + 0.4 * cx
            self.cy = 0.6 * self.cy + 0.4 * cy
        self.misses = 0
        self.has_target = True
        if t_now is not None:
            self.hist.append((float(t_now), float(self.cx), float(self.cy)))
        return True


class EgoConsecGate:
    """Ego-camera brake gate: N consecutive in-ROI person frames, past an ego-Y gate."""

    def __init__(self, consecutive_n: int = 1, ego_brake_gate_y: float = -30.0):
        self.consecutive_n = consecutive_n
        self.ego_brake_gate_y = ego_brake_gate_y
        self._consec = 0
        self._braked = False

    def update(self, detection_in_roi: bool, ego_y: Optional[float]) -> bool:
        """Advance one ego frame; return True on the frame braking should engage."""
        self._consec = self._consec + 1 if detection_in_roi else 0
        if self._braked:
            return False
        past_gate = ego_y is not None and ego_y < self.ego_brake_gate_y
        if self._consec >= self.consecutive_n and past_gate:
            self._braked = True
            return True
        return False


class V2IFusionGate:
    """Infra-assisted brake gate: ego hard-TTC, or ego/ped ETAs within a window."""

    def __init__(
        self,
        ttc_thresh_s: float = 3.0,
        collision_window_s: float = 2.0,
        prediction_horizon_s: float = 7.0,
    ):
        self.ttc_thresh_s = ttc_thresh_s
        self.collision_window_s = collision_window_s
        self.prediction_horizon_s = prediction_horizon_s
        self._braked = False

    def update(self, ped_eta_s: Optional[float], ego_eta_s: Optional[float]) -> bool:
        """Advance one infra frame; return True on the frame braking should engage."""
        if self._braked:
            return False
        if ego_eta_s is not None and ego_eta_s <= self.ttc_thresh_s:
            self._braked = True
            return True
        if ped_eta_s is None or ego_eta_s is None:
            return False
        if ped_eta_s > self.prediction_horizon_s or ego_eta_s > self.prediction_horizon_s:
            return False
        if min(ped_eta_s, ego_eta_s) <= self.ttc_thresh_s:
            self._braked = True
            return True
        if abs(ped_eta_s - ego_eta_s) <= self.collision_window_s:
            self._braked = True
            return True
        return False


# ---------------------------------------------------------------------------
# Messages: per-feed reduced observations (replayable; carry their own sim_time)
# ---------------------------------------------------------------------------
class V2IEgoObs(Message):
    """Ego-feed observation: is a target person in the ego ROI, and the ego's world Y."""

    def __init__(self, in_roi: bool = False, ego_y: Optional[float] = None, **kw):
        super().__init__(**kw)
        self.in_roi = in_roi
        self.ego_y = ego_y


class V2IInfraObs(Message):
    """Infra-feed observation: in-ROI person boxes, the crossing band, and the ego ETA."""

    def __init__(
        self,
        dets: Optional[List[Tuple]] = None,
        ego_eta: Optional[float] = None,
        y_target: float = 0.0,
        **kw,
    ):
        super().__init__(**kw)
        self.dets = dets or []
        self.ego_eta = ego_eta
        self.y_target = y_target


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def _roi_tuple(d) -> Tuple:
    return (d.get("cx_min", 0), d.get("cx_max", 0), d.get("cy_min"), d.get("cy_max"))


@register_stage("yolo_v2i")
class YoloV2IDetectorStage(Stage):
    """Run YOLO on a frame and emit the per-feed V2I observation for the gates.

    Reads the frame's ``device_id``; for an ego frame emits :class:`V2IEgoObs` (target
    in the ego ROI + ego Y), for an infra frame emits :class:`V2IInfraObs` (in-ROI
    person boxes, the crossing band from the image height, and the ego ETA to the
    world crossing from the last-known ego pose). The person filter (target classes,
    confidence, ROI) matches the flat RSU's ``_detect_persons``.
    """

    data_source = PERCEIVED

    def __init__(self, name, params=None):
        super().__init__(name, params)
        self._model = None
        self._device = None
        self._log = lambda m: None
        self._person_ids: List[int] = []
        self._class_names = {}
        self._conf = self.params.get("conf_thresh", 0.40)
        self._ego_roi = _roi_tuple(self.params.get("ego_roi", {}))
        self._infra_roi = _roi_tuple(self.params.get("infra_roi", {}))
        self._crossing_x = self.params.get("crossing_x", 0.0)
        self._crossing_y = self.params.get("crossing_y", 0.0)
        self._min_ego_speed = self.params.get("min_ego_speed_mps", 0.5)
        self._target_classes = list(self.params.get("target_classes", ["person"]))

    def setup(self, ctx=None) -> None:
        from ultralytics import YOLO

        from cadet.core.model_registry import (
            ModelRegistry,
            device_for_models,
            torch_device_for,
        )

        if ctx is None or ctx.config is None:
            raise RuntimeError("yolo_v2i stage needs a StageContext with the resolved config")
        cfg = ctx.config
        registry = ModelRegistry.from_config(cfg.models, device_for_models(cfg))
        res = registry.resolve(cfg.models.model, ctx.model_dirs)
        if res.path is None:
            raise RuntimeError(
                "model artifact {!r} not found under {}".format(res.artifact, ctx.model_dirs)
            )
        self._device = torch_device_for(cfg)
        self._model = YOLO(res.path)
        self._class_names = self._model.names
        self._person_ids = [
            i for i, n in self._class_names.items() if n in self._target_classes
        ]
        self._log = ctx.log
        ctx.log("[stage {}] loaded {} on {}".format(self.name, res.path, self._device))

    def warmup(self) -> None:
        if self._model is None:
            return
        self._model.predict(np.zeros((600, 800, 3), dtype=np.uint8),
                            verbose=False, device=self._device)

    def _detect_persons(self, rgb, roi) -> List[Tuple]:
        out = self._model.predict(rgb, verbose=False, device=self._device)
        dets = []
        for b in out[0].boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            if cls not in self._person_ids or conf < self._conf:
                continue
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            if center_in_roi(cx, cy, roi):
                dets.append((self._class_names[cls], conf, (x1, y1, x2, y2)))
        return dets

    def _ego_eta(self, msg) -> Optional[float]:
        """Ego ETA to the world crossing, gated on min ego speed (flat ``_ego_eta``)."""
        ego_x = getattr(msg, "ego_x", None)
        ego_speed = getattr(msg, "ego_speed_mps", 0.0)
        if ego_x is None or ego_speed < self._min_ego_speed:
            return None
        return ttc_to_crossing(
            ego_x, getattr(msg, "ego_y", 0.0), ego_speed,
            self._crossing_x, self._crossing_y,
        )

    def _infra_y_target(self, frame_shape) -> float:
        """Crossing band in image Y: the infra ROI cy midpoint, else half the height."""
        _, _, cy_min, cy_max = self._infra_roi
        if cy_min is not None and cy_max is not None:
            return 0.5 * (cy_min + cy_max)
        return 0.5 * frame_shape[0]

    def process(self, msg: Message) -> Optional[Message]:
        rgb = getattr(msg, "image", None)
        device_id = getattr(msg, "device_id", "ego")
        # The pipeline driver carries the ego pose at capture in the frame's
        # meta["capture_loc"] (a plain (x, y, z); a carla handle cannot cross a
        # node boundary). Read the world Y/X from there - the gate needs the ego's
        # own position, the same value the flat app reads from the client each tick.
        cloc = (getattr(msg, "meta", None) or {}).get("capture_loc")
        ego_y = cloc[1] if cloc else getattr(msg, "ego_y", None)
        if device_id == "ego":
            dets = self._detect_persons(rgb, self._ego_roi)
            out = V2IEgoObs(in_roi=len(dets) > 0, ego_y=ego_y)
        else:
            dets = self._detect_persons(rgb, self._infra_roi)
            out = V2IInfraObs(
                dets=dets,
                ego_eta=self._ego_eta(msg),
                y_target=self._infra_y_target(rgb.shape if rgb is not None else (0,)),
            )
        out.frame_id = msg.frame_id
        out.sim_time = msg.sim_time
        return out


@register_stage("v2i_ego_gate")
class V2IEgoGateStage(Stage):
    """Ego consecutive-detection gate; emits a BrakeDecision only on the brake frame."""

    data_source = PERCEIVED

    def __init__(self, name, params=None):
        super().__init__(name, params)
        self._gate = EgoConsecGate(
            consecutive_n=self.params.get("consecutive_n", 1),
            ego_brake_gate_y=self.params.get("ego_brake_gate_y", -30.0),
        )

    def process(self, msg: Message) -> Optional[Message]:
        if self._gate.update(getattr(msg, "in_roi", False), getattr(msg, "ego_y", None)):
            return BrakeDecision(brake=True, server_processing_time=0.0, frame_id=msg.frame_id)
        return None


@register_stage("v2i_fusion_gate")
class V2IFusionGateStage(Stage):
    """Infra fusion gate: centroid tracker + ped ETA fused with ego ETA; brake once."""

    data_source = PERCEIVED

    def __init__(self, name, params=None):
        super().__init__(name, params)
        self._tracker = CentroidTracker(
            max_miss=self.params.get("tracker_max_miss", 5),
            hist_len=self.params.get("tracker_hist_len", 12),
        )
        self._gate = V2IFusionGate(
            ttc_thresh_s=self.params.get("ttc_thresh_s", 3.0),
            collision_window_s=self.params.get("collision_window_s", 2.0),
            prediction_horizon_s=self.params.get("prediction_horizon_s", 7.0),
        )
        self._min_ped_speed = self.params.get("min_ped_speed_pxps", 1.5)
        self._horizon = self.params.get("prediction_horizon_s", 7.0)

    def process(self, msg: Message) -> Optional[Message]:
        sim_time = msg.sim_time if msg.sim_time is not None else 0.0
        self._tracker.update(getattr(msg, "dets", []), t_now=sim_time)
        ped_eta = (
            ped_eta_from_history(
                list(self._tracker.hist), getattr(msg, "y_target", 0.0),
                self._min_ped_speed, self._horizon,
            )
            if self._tracker.has_target
            else None
        )
        if self._gate.update(ped_eta, getattr(msg, "ego_eta", None)):
            return BrakeDecision(brake=True, server_processing_time=0.0, frame_id=msg.frame_id)
        return None
