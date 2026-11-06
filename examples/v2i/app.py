"""V2I (infrastructure-assisted) emergency-braking app — decision core.

What this is for:
    The vehicle-to-infrastructure braking example: an ego and a fixed roadside
    (infra) camera both stream to an RSU, which brakes the ego on a crossing
    pedestrian. The infra camera can see the pedestrian while an occluder blocks the
    ego's view, so V2I brakes earlier. This module ports the *pure* decision logic
    from the original ``carla_wrapper/v2i_braking_server.py`` so it can be unit-tested
    with no CARLA and reused by the server app and the pipeline form.

What it contains:
    - :class:`V2IParams`: the app's typed parameters, defaults = the original's
      *active* values (from ``v2i_braking_params``, not the server's getattr fallbacks).
    - Pure helpers :func:`center_in_roi`, :func:`ttc_to_crossing`,
      :func:`ped_eta_from_history`.
    - :class:`CentroidTracker`: the single-target centroid tracker (verbatim).
    - :class:`EgoConsecGate`: the ego-camera brake gate (consecutive in-ROI person
      detections past an ego-Y gate), used in every mode.
    - :class:`V2IFusionGate`: the infra-assisted brake gate (ego ETA hard-TTC, or
      ego/ped ETAs within a collision window), used in the v2i modes.

Deviation policy:
    By policy CADET reproduces the original 1:1 by default. The two network hops the
    original modelled as per-frame server sleeps (``EGO_NET_MS`` in baseline_cloud,
    ``INFRA_NET_MS`` in v2i_external, each 50 ms) become CADET **links** / blocking
    server delays whose default reproduces the original timing; the paper cites 30 ms.

Not yet here: the RSU server app (fan-in of ego+infra via
``MultiConnectionEndpoint.serve(expect=2, model_for=...)``), the ego + infra clients,
the four modes, and the pipeline-stage form. The decision core below is what those
will call, kept pure and tested first (port before writing).
"""

import json
import os
import pickle
import time
import zlib
from collections import deque
from queue import Empty, Queue
from threading import Thread
from typing import Deque, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Literal

from cadet.core.apps import AppPlugin, register_app
from cadet.core.model_registry import ModelRegistry, device_for_models, torch_device_for
from cadet.metrics import MetricsLog


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoiRect(_Strict):
    """Pixel region-of-interest rectangle (a detection's centre must fall inside)."""

    cx_min: int
    cx_max: int
    cy_min: Optional[int] = None
    cy_max: Optional[int] = None


class InfraCamera(_Strict):
    """Fixed roadside (infra) camera pose + optics (the original's active values).

    The camera sits at ``(base_x, base_y, base_z + height_m)`` looking down the pole at
    ``pitch_deg`` / ``yaw_deg``. The one client process spawns it at this fixed world
    pose (not attached to a vehicle), exactly as the original V2IClient did.
    """

    base_x: float = -141.0
    base_y: float = -76.0
    base_z: float = 0.0
    height_m: float = 15.8
    pitch_deg: float = -39.0
    yaw_deg: float = 0.0
    fov_deg: float = 70.0
    width: int = 1280
    height: int = 720


def _ego_roi() -> RoiRect:
    # Original active values (v2i_braking_params.server_defaults), not the server's
    # getattr fallbacks (300..500 x-band).
    return RoiRect(cx_min=150, cx_max=700, cy_min=50, cy_max=700)


def _infra_roi() -> RoiRect:
    return RoiRect(cx_min=850, cx_max=1150, cy_min=100, cy_max=600)


class V2IParams(_Strict):
    """Parameters for the V2I app (defaults = the original's active values)."""

    plugin: Literal["v2i"] = "v2i"
    # The RSU brakes only for pedestrians (original TARGET_CLASSES).
    target_classes: List[str] = Field(default_factory=lambda: ["person"])
    conf_thresh: float = 0.40
    ego_roi: RoiRect = Field(default_factory=_ego_roi)
    infra_roi: RoiRect = Field(default_factory=_infra_roi)
    infra_camera: InfraCamera = Field(default_factory=InfraCamera)
    # Ego consecutive-detection gate. use_consecutive False -> N=1 (the original's
    # active setting), and the ego must be past this world-Y before it may brake.
    consecutive_n: int = 1
    ego_brake_gate_y: float = -30.0
    # World crossing point the ego's ETA is measured to (server defaults).
    crossing_x: float = -124.96
    crossing_y: float = -65.0
    # Fusion / TTC knobs (active values).
    ttc_thresh_s: float = 3.0
    collision_window_s: float = 2.0
    prediction_horizon_s: float = 7.0
    min_ped_speed_pxps: float = 1.5
    min_ego_speed_mps: float = 0.5
    # Mode: baseline uses the ego camera only; v2i also uses the infra camera. The
    # two 50 ms hops are applied to the matching mode (baseline_cloud -> ego hop,
    # v2i_external -> infra hop); see the config links.
    mode: Literal[
        "baseline_local", "baseline_cloud", "v2i_local", "v2i_external"
    ] = "baseline_local"
    # Ego deceleration controller (the original's audi.tt gains, from params).
    target_decel: float = 7.0
    max_decel: float = 7.3
    kp: float = 0.2
    max_ramp: float = 0.05
    # The two network hops, applied in the RSU per device_id by mode, exactly as the
    # original (EGO_NET_MS in baseline_cloud, INFRA_NET_MS in v2i_external). Default is
    # a blocking server sleep (the original's mechanism, like braking's
    # server_delay_mode); set hop_mode: link to model them as config links instead (then
    # the RSU does not sleep). Default 50 ms; the paper cites 30 ms.
    ego_net_ms: float = 50.0
    infra_net_ms: float = 50.0
    hop_mode: Literal["blocking", "link"] = "blocking"
    # Pedestrian walker, driven by the CLIENT (the original drove it from the client's
    # _walker_loop, not the scenario): idle until t_ped_start_s + ped_start_delay_s, then
    # jaywalk in -X at ped_speed_mps until ped.x <= ped_goal_x, then hold. Active values.
    ped_speed_mps: float = 2.0
    ped_goal_x: float = -140.0
    t_ped_start_s: float = 15.3
    ped_start_delay_s: float = 4.0
    # From the original (default): annotated detection JPEGs per processed frame.
    save_annotated_frames: bool = True


# ---------------------------------------------------------------------------
# Pure helpers (no CARLA / no torch) — unit tested
# ---------------------------------------------------------------------------
def center_in_roi(cx: float, cy: float, roi: RoiRect) -> bool:
    """True if ``(cx, cy)`` is inside the ROI (rectangle if cy bounds set, else x-band)."""
    if roi.cy_min is not None and roi.cy_max is not None:
        return (roi.cx_min <= cx <= roi.cx_max) and (roi.cy_min <= cy <= roi.cy_max)
    return roi.cx_min <= cx <= roi.cx_max


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
    """Image-space ped ETA (s) to a crossing band, constant-velocity on centroid y.

    ``hist`` is ``(t, cx, cy)`` samples (oldest first). Returns None when there are
    fewer than 3 samples, the ped is not moving toward the band, it is slower than
    ``min_speed_pxps``, or the ETA exceeds ``horizon_s`` — exactly the original's gates.
    """
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
    """Single-target centroid tracker for a person already filtered by ROI (verbatim).

    Smooths the highest-confidence box's centre (EMA 0.6/0.4), drops the target after
    ``max_miss`` empty frames, and keeps a short ``(t, cx, cy)`` history for the ETA.
    """

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
    """Ego-camera brake gate: N consecutive in-ROI person frames, past an ego-Y gate.

    A direct port of the original's ego decision (every mode): count frames that have
    a target in the ego ROI, reset on a miss, and brake once the count reaches
    ``consecutive_n`` *and* the ego is past ``ego_brake_gate_y`` in world Y.
    """

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
    """Infra-assisted brake gate: ego hard-TTC, or ego/ped ETAs within a window.

    A direct port of the original's ``_predict_fuse_and_decide`` (v2i modes). Given the
    ego ETA and (optionally) the perceived ped ETA each infra frame, brakes once:
    the ego ETA is within the hard TTC threshold; or both ETAs are within the horizon
    and either their minimum is within the TTC threshold or they are within the
    collision window. Fires at most once.
    """

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
        # Ego ETA inside the hard TTC threshold brakes even without a ped ETA.
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
# Ego deceleration controller (ported verbatim from the original V2I client)
# ---------------------------------------------------------------------------
class _DecelController:
    def __init__(self, target_decel, max_decel, kp, max_ramp):
        self.target_decel = target_decel
        self.max_decel = max_decel
        self.kp = kp
        self.max_ramp = max_ramp
        self.filtered_decel = 0.0
        self.brake_cmd = 0.0
        self.last_speed = 0.0
        self.last_time: Optional[float] = None

    def update_decel(self, current_speed, now, alpha=0.3):
        dt = max(1e-3, (now - self.last_time)) if self.last_time is not None else 0.0
        raw = max(0.0, (self.last_speed - current_speed) / dt) if dt > 0 else 0.0
        self.filtered_decel = alpha * raw + (1 - alpha) * self.filtered_decel
        self.last_speed = current_speed
        self.last_time = now
        return self.filtered_decel

    def step(self):
        if self.filtered_decel >= self.max_decel:
            self.brake_cmd = max(0.0, self.brake_cmd - self.max_ramp)
            return self.brake_cmd
        err = self.target_decel - self.filtered_decel
        desired = max(0.0, min(self.kp * err, 1.0))
        if desired > self.brake_cmd:
            self.brake_cmd = min(self.brake_cmd + self.max_ramp, desired)
        else:
            self.brake_cmd = max(self.brake_cmd - self.max_ramp, desired)
        return self.brake_cmd


def _pid_throttle(current_speed, target_speed, kp=0.6, deadband=1.0):
    err = target_speed - current_speed
    if abs(err) < deadband:
        return 0.4
    return max(0.0, min(kp * err, 1.0))


def _speed_mps(speed_str):
    if speed_str.endswith("kph"):
        return float(speed_str[:-3]) / 3.6
    if speed_str.endswith("mph"):
        return float(speed_str[:-3]) * 0.44704
    raise ValueError("speed must end with 'kph' or 'mph', got {!r}".format(speed_str))


def _as_plain(val):
    """A scenario sub-block as a plain dict, whether it is a pydantic model or a dict."""
    if hasattr(val, "model_dump"):
        return val.model_dump()
    if hasattr(val, "__dict__"):
        return {k: v for k, v in vars(val).items() if not k.startswith("_")}
    return val


def _resolve_person_ids(class_names, target_classes):
    wanted = [t.lower() for t in target_classes]
    return [k for k, v in class_names.items() if any(t == v.lower() for t in wanted)]


def _save_annotated(rgb, result, class_names, out_dir, frame_id):
    import cv2

    img = rgb.copy()
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "{} {:.2f}".format(class_names[cls], conf), (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(os.path.join(out_dir, "frame_{:05d}.jpg".format(frame_id)),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# RSU server app (fan-in of ego + infra; brakes the ego)
# ---------------------------------------------------------------------------
class RSUServerApp:
    """Runs YOLO on ego + infra frames and brakes the ego (ported from RSUServer).

    The fan-in server (``expect>1``) delivers frames from both sources into this one
    ``handle``; ego and infra are told apart by the message's ``device_id``. The ego
    path runs :class:`EgoConsecGate` (every mode); the infra path (v2i modes) runs
    :class:`CentroidTracker` + :class:`V2IFusionGate`. Because the fan-in reply goes to
    the frame's source but the EGO is the actuator, an infra-triggered brake is latched
    and the reply is emitted on the next ego frame, within one ego-frame of
    latency. Ground-truth ped pose is cached for evaluation only and never
    enters a decision.
    """

    def __init__(self, config, model_search_dirs, logger=None):
        self.cfg = config
        self.params = V2IParams.model_validate(config.app.params)
        self.search_dirs = model_search_dirs
        self._log = logger or (lambda m: None)
        self._model = None
        self._device = None
        self._person_ids: List[int] = []
        self._is_v2i = self.params.mode.startswith("v2i")
        self._ego_gate = EgoConsecGate(self.params.consecutive_n, self.params.ego_brake_gate_y)
        self._fusion = V2IFusionGate(
            self.params.ttc_thresh_s, self.params.collision_window_s,
            self.params.prediction_horizon_s,
        )
        self._tracker = CentroidTracker()
        self._braked = False
        # Connection id -> device_id ('ego'/'infra'), cached from each connection's
        # first frame so the feeder thread (on_ingress) decompresses a payload at most
        # once per connection to pick that connection's network hop.
        self._cid_device = {}
        # Cached ego odometry (from ego frames) for the ego ETA.
        self._ego_x = self._ego_y = None
        self._ego_speed = 0.0
        # Ground-truth ped pose/velocity (eval only) + the TTC/TTE/RM eval rows.
        self._gt_ped = None
        self._gt_ped_v = None
        self._eval_rows = []
        self._eval_dir = os.path.join(
            config.outputs.root, config.experiment.name, config.app.plugin
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
            raise RuntimeError("model artifact {!r} not found under {}".format(
                res.artifact, self.search_dirs))
        self._device = torch_device_for(self.cfg)
        self._model = YOLO(res.path)
        self._class_names = self._model.names
        self._person_ids = _resolve_person_ids(self._class_names, self.params.target_classes)
        if not self._person_ids:
            raise RuntimeError("none of the target classes are known to the model")
        if self._save_annotated:
            os.makedirs(self._annotated_dir, exist_ok=True)
        self._log("[rsu] loaded {} on {} (mode={})".format(
            res.path, self._device, self.params.mode))

    def warmup(self) -> None:
        cam = self.cfg.source.camera
        dummy = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
        self._model.predict(dummy, verbose=False, device=self._device)

    def _detect_persons(self, rgb, roi):
        """Return (in_roi_any, detections[(label,conf,box)]) for persons in the ROI."""
        out = self._model.predict(rgb, verbose=False, device=self._device)
        dets = []
        for b in out[0].boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            if cls not in self._person_ids or conf < self.params.conf_thresh:
                continue
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            if center_in_roi(cx, cy, roi):
                dets.append((self._class_names[cls], conf, (x1, y1, x2, y2)))
        return (len(dets) > 0), dets, out[0]

    def handle(self, payload: bytes, ctx=None) -> Optional[bytes]:
        msg = pickle.loads(zlib.decompress(payload))
        device_id = msg.get("device_id", "ego")
        rgb = msg["rgb_frame"]
        frame_id = msg.get("frame_id", -1)
        sim_time = msg.get("sim_time", msg.get("timestamp", 0.0))
        # Ground truth for evaluation only (never a decision input).
        if msg.get("gt_ped_loc") is not None:
            self._gt_ped = msg.get("gt_ped_loc")
        if msg.get("gt_ped_vel_mps") is not None:
            self._gt_ped_v = msg.get("gt_ped_vel_mps")
        # The per-device network hop is applied in the feeder thread (on_ingress) so it
        # overlaps inference instead of serializing with it; not re-applied here.
        t_start = time.time()
        reply = None
        result = None
        ped_eta = None
        ego_eta = None

        # One brake per run; the brake reply goes to the source that triggered it (the
        # ego feed for ego-perception, the infra feed for infra-fusion). The single
        # client process listens on both feeds and actuates on either immediately, so
        # the ego brakes the moment the decision is made - no latch. (A separate infra
        # NODE, where the reply must be forwarded to the ego, is an opt-in
        # deviation, not this.)
        if device_id == "ego":
            pose = msg.get("ego_pose")
            if pose is not None:
                self._ego_x, self._ego_y = pose[0], pose[1]
            self._ego_speed = float(msg.get("ego_speed_mps", 0.0))
            in_roi, dets, result = self._detect_persons(rgb, self.params.ego_roi)
            ego_eta = self._ego_eta()
            if not self._braked and self._ego_gate.update(in_roi, self._ego_y):
                self._braked = True
                reply = self._brake_reply("ego", frame_id, t_start)
        elif device_id == "infra" and self._is_v2i:
            _in, dets, result = self._detect_persons(rgb, self.params.infra_roi)
            self._tracker.update(dets, t_now=time.time())
            y_target = self._infra_y_target(rgb.shape)
            ped_eta = ped_eta_from_history(
                list(self._tracker.hist), y_target,
                self.params.min_ped_speed_pxps, self.params.prediction_horizon_s,
            ) if self._tracker.has_target else None
            ego_eta = self._ego_eta()
            if not self._braked and self._fusion.update(ped_eta, ego_eta):
                self._braked = True
                reply = self._brake_reply("v2i/infra", frame_id, t_start)
        else:
            return None  # infra frame in a baseline mode: ignored

        # Evaluation-only logging (never a decision input): TTC (ego ETA), perceived ped
        # ETA, GT ped TTE, and the risk margins RM = TTC - TTE. Ported from the original
        # RSU's _log_tick; the safety margin is RM.
        self._log_eval(sim_time, frame_id, device_id, ego_eta, ped_eta,
                       reply.get("reason") if reply else "")
        if self._save_annotated and result is not None:
            _save_annotated(rgb, result, self._class_names, self._annotated_dir, frame_id)
        if reply is not None:
            return zlib.compress(pickle.dumps(reply))
        return None

    def _tte_gt(self):
        """GT pedestrian TTE to the crossing line (eval only), from its y-pose/velocity."""
        if self._gt_ped is None or self._gt_ped_v is None:
            return None
        py = float(self._gt_ped[1])
        vy = float(self._gt_ped_v[1])
        dist = self.params.crossing_y - py
        if vy == 0.0 or (dist * vy) <= 0.0:  # not moving toward the line
            return None
        tte = abs(dist) / abs(vy)
        return tte if tte <= self.params.prediction_horizon_s else None

    def _log_eval(self, sim_time, frame_id, source, ego_eta, ped_eta, reason):
        tte_gt = self._tte_gt()
        rm_perc = (ego_eta - ped_eta) if (ego_eta is not None and ped_eta is not None) else None
        rm_gt = (ego_eta - tte_gt) if (ego_eta is not None and tte_gt is not None) else None
        self._eval_rows.append({
            "sim_time": round(float(sim_time), 4),
            "frame_id": frame_id,
            "source": source,
            "mode": self.params.mode,
            "ego_y": (None if self._ego_y is None else round(self._ego_y, 3)),
            "ego_speed_mps": round(self._ego_speed, 3),
            "TTC_e_s": (None if ego_eta is None else round(ego_eta, 4)),
            "TTE_p_perc_s": (None if ped_eta is None else round(ped_eta, 4)),
            "RM_perc_s": (None if rm_perc is None else round(rm_perc, 4)),
            "TTE_p_gt_s": (None if tte_gt is None else round(tte_gt, 4)),
            "RM_gt_s": (None if rm_gt is None else round(rm_gt, 4)),
            "reason": reason,
        })

    def _hop_seconds(self, device_id):
        """Per-device network-hop duration (the original's EGO/INFRA_NET_MS)."""
        if self.params.hop_mode != "blocking":
            return 0.0  # link mode: the delay is on a config link, not a server sleep
        if device_id == "ego" and self.params.mode == "baseline_cloud":
            return self.params.ego_net_ms / 1000.0
        if device_id == "infra" and self.params.mode == "v2i_external":
            return self.params.infra_net_ms / 1000.0
        return 0.0

    def on_ingress(self, source_id, payload):
        """Hold a frame for its connection's network hop, in the server's feeder thread.

        Runs concurrently with the worker's inference (as the original RSU slept in its
        per-connection receiver thread before enqueue), so the hop no longer serializes
        with YOLO and the inbound queue stays shallow - the key to on-time decisions in
        the hop modes. The connection's device is read from its first frame and cached,
        so the payload is decompressed at most once per connection here.
        """
        device_id = self._cid_device.get(source_id)
        if device_id is None:
            try:
                device_id = pickle.loads(zlib.decompress(payload)).get("device_id", "ego")
            except Exception:
                return
            self._cid_device[source_id] = device_id
        delay = self._hop_seconds(device_id)
        if delay > 0:
            time.sleep(delay)

    def _brake_reply(self, reason, frame_id, t_start):
        self._log("[rsu] BRAKE ({}) at frame {}".format(reason, frame_id))
        return {
            "brake": True,
            "server_processing_time": time.time() - t_start,
            "frame_id": frame_id,
            "reason": reason,
        }

    def _ego_eta(self):
        if self._ego_x is None or self._ego_speed < self.params.min_ego_speed_mps:
            return None
        return ttc_to_crossing(
            self._ego_x, self._ego_y, self._ego_speed,
            self.params.crossing_x, self.params.crossing_y,
        )

    def _infra_y_target(self, frame_shape):
        roi = self.params.infra_roi
        if roi.cy_min is not None and roi.cy_max is not None:
            return 0.5 * (roi.cy_min + roi.cy_max)
        return 0.5 * frame_shape[0]

    def stop_after_reply(self) -> bool:
        return True

    def teardown(self, ctx=None) -> None:
        if not self._eval_rows:
            return
        import csv

        os.makedirs(self._eval_dir, exist_ok=True)
        path = os.path.join(self._eval_dir, "rsu_eval_{}.csv".format(self.params.mode))
        cols = ["sim_time", "frame_id", "source", "mode", "ego_y", "ego_speed_mps",
                "TTC_e_s", "TTE_p_perc_s", "RM_perc_s", "TTE_p_gt_s", "RM_gt_s", "reason"]
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for row in self._eval_rows:
                w.writerow(row)
        self._log("[rsu] eval log ({} rows) -> {}".format(len(self._eval_rows), path))


# ---------------------------------------------------------------------------
# Ego client app (cruise toward the crossing, stream + odometry, brake on command)
# ---------------------------------------------------------------------------
MPS_TO_MPH = 2.23694
STOP_SPEED_MPS = 0.1


class V2IClientApp:
    """One client owning BOTH feeds (ego + infra), exactly as the original V2IClient.

    A faithful port: a single process drives the ego (cruise, stream ego frames +
    odometry on the role's connection, actuate) and - in the v2i modes - also mounts a
    fixed roadside camera and streams infra frames on a SECOND connection to the same
    RSU. Both connections carry ``device_id`` tags. The client listens for a brake on
    BOTH connections and actuates immediately on whichever carries it (the RSU replies
    to whichever feed triggered), so there is no latch. The pedestrian is never
    destroyed. A separate infra NODE is an opt-in deviation, not this.
    """

    def __init__(self, config, run_id, rep, csv_path, logger=None):
        self.cfg = config
        self.params = V2IParams.model_validate(config.app.params)
        self.source = None
        self.transport = None
        self._log = logger or (lambda m: None)
        self.controller = _DecelController(
            self.params.target_decel, self.params.max_decel, self.params.kp, self.params.max_ramp
        )
        self.target_mps = _speed_mps(config.source.scenario.ego_speed)
        self.metrics = MetricsLog(csv_path, run_id, rep)
        self._camera_fov = config.source.camera.fov
        self._image_size = (config.source.camera.width, config.source.camera.height)
        self._is_v2i = self.params.mode.startswith("v2i")
        self._start = None
        self._frame_counter = 0
        self._cur_fid = None
        self._transmit = True
        self._brake_engaged = False
        self._brake_recorded = False
        self._done = False
        self._set_speed_once = True  # the original kicked the ego to speed on the first loop
        self._last_speed_mph = 0.0
        self._replies: "Queue" = Queue()
        # Infra feed (v2i modes only): a second connection + a fixed roadside camera.
        self._infra_transport = None
        self._infra_camera = None
        self._infra_queue = None
        self._infra_frame_id = 0
        # Client-driven pedestrian walker (the original's _walker_loop).
        self._ped_speed_mps = self.params.ped_speed_mps
        self._ped_goal_x = self.params.ped_goal_x
        self._ped_cross_start_s = self.params.t_ped_start_s + self.params.ped_start_delay_s

    def setup(self, ctx) -> None:
        self.source = ctx.source
        self.transport = ctx.transport
        Thread(target=lambda: self._reply_listener(self.transport), daemon=True).start()
        if self._is_v2i:
            self._setup_infra_feed()

    def _setup_infra_feed(self) -> None:
        """Open the second (infra) connection + fixed roadside camera, and stream it."""
        from cadet.core.transport import FramedSocketTransport

        server = next((n for n in self.cfg.nodes if n.role == "server"), None)
        if server is None:
            self._log("[infra] no server node; infra feed disabled")
            return
        ic = self.params.infra_camera
        try:
            self._infra_camera, self._infra_queue = self.source.spawn_fixed_camera(
                ic.base_x, ic.base_y, ic.base_z + ic.height_m,
                ic.pitch_deg, ic.yaw_deg, ic.width, ic.height, ic.fov_deg,
            )
            self._infra_transport = FramedSocketTransport.connect(
                server.host, server.port, retry_s=10.0
            )
        except Exception as exc:
            self._log("[infra] setup failed ({}); infra feed disabled".format(exc))
            self._infra_transport = None
            return
        self._log("[infra] feed up: fixed camera + second connection to {}:{}".format(
            server.host, server.port))
        # Pace the infra feed to its channel's stream_fps (skip frames arriving sooner),
        # same reasoning as the ego feed in the client role: a free-running roadside
        # camera must not outpace a rate-limited RSU feeder (the v2i_external infra hop).
        self._infra_min_interval = None
        for ch in (getattr(self.cfg, "channels", None) or []):
            if getattr(ch, "device_id", None) == "infra" and ch.stream_fps and ch.stream_fps > 0:
                self._infra_min_interval = 1.0 / float(ch.stream_fps)
                break
        self._infra_last_send = None
        Thread(target=lambda: self._reply_listener(self._infra_transport), daemon=True).start()
        Thread(target=self._infra_sender, daemon=True).start()

    def _infra_sender(self) -> None:
        """Stream infra frames (device_id='infra') on the second connection."""
        while self._transmit and self._infra_transport is not None:
            try:
                rgb, capture_time = self._infra_queue.get(timeout=0.5)
            except Empty:
                continue
            if self._infra_min_interval is not None:
                now = time.monotonic()
                if (self._infra_last_send is not None
                        and now - self._infra_last_send < self._infra_min_interval):
                    continue  # skip: arrived sooner than 1/fps, keep the freshest
                self._infra_last_send = now
            msg = {
                "device_id": "infra",
                "frame_id": self._infra_frame_id,
                "timestamp": capture_time,
                # Relative sim clock (same origin as ego frames) so RSU eval rows
                # share one timeline; falls back to 0.0 before the first control tick.
                "sim_time": (time.time() - self._start) if self._start else 0.0,
                "rgb_frame": rgb,
                "image_size": (self.params.infra_camera.width, self.params.infra_camera.height),
                "camera_fov": float(self.params.infra_camera.fov_deg),
            }
            try:
                self._infra_transport.send(zlib.compress(pickle.dumps(msg)))
            except Exception:
                break
            self._infra_frame_id += 1

    def _reply_listener(self, transport) -> None:
        while True:
            try:
                raw = transport.recv()
            except Exception:
                break
            if raw is None:
                break
            try:
                cmd = pickle.loads(zlib.decompress(raw))
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
        if self._start is None:
            self._start = time.time()
        now = time.time()
        sim_time = now - self._start
        speed_mps = self.source.vehicle_speed_mps()
        self._last_speed_mph = speed_mps * MPS_TO_MPH
        throttle, steer, brake = 0.0, 0.0, 0.0

        if reply is not None and not self._brake_engaged:
            self._brake_engaged = True
            self._transmit = False
            loc = self.source.vehicle_location()
            brake_loc = (round(loc.x, 2), round(loc.y, 2), round(loc.z, 2))
            if not self._brake_recorded:
                self._brake_recorded = True
                self.metrics.mark_brake(
                    reply.get("frame_id"), reply.get("server_processing_time"), sim_time, brake_loc,
                    recv_wall=now,
                )
                self._log("[ego] BRAKE ({}) at t={:.2f} {}".format(
                    reply.get("reason", "?"), sim_time, brake_loc))

        if self._brake_engaged:
            self.controller.update_decel(speed_mps, now)
            brake = self.controller.step()
            throttle = 0.0
            if not self._done and speed_mps < STOP_SPEED_MPS:
                loc = self.source.vehicle_location()
                self.metrics.mark_stop(
                    sim_time, (round(loc.x, 2), round(loc.y, 2), round(loc.z, 2)))
                self._log("[ego] STOP at t={:.2f}".format(sim_time))
                self._done = True
        else:
            if self._set_speed_once:
                self.source.set_target_velocity(self.target_mps)
                self._set_speed_once = False
            else:
                throttle = _pid_throttle(speed_mps, self.target_mps)

        self.source.apply_control(throttle, steer, brake)
        self._drive_walker(sim_time)

    def _drive_walker(self, sim_time: float) -> None:
        """Drive the crossing pedestrian, ported from the original client's _walker_loop.

        Idle until the cross-start time, then jaywalk in -X at ``ped_speed_mps`` until
        the ped passes ``ped_goal_x``, then hold. The original drove the walker from the
        client (not the scenario), so the ped's position relative to the approaching ego
        - and thus when the RSU can see it - matches the original.
        """
        ped = self.source.obstacle_location()
        if ped is None:
            return
        if sim_time < self._ped_cross_start_s:
            return  # idle (spawned, not yet crossing)
        if ped.x <= self._ped_goal_x + 0.25:
            self.source.apply_walker_control(-1.0, 0.0, 0.0, 0.0)  # reached goal: hold
            return
        self.source.apply_walker_control(-1.0, 0.0, 0.0, self._ped_speed_mps)

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
        loc = self.source.vehicle_location()
        speed_mps = self.source.vehicle_speed_mps()
        msg = {
            "device_id": "ego",
            "frame_id": fid,
            "timestamp": capture_time,
            "sim_time": (time.time() - self._start) if self._start else 0.0,
            "rgb_frame": frame.image,
            "ego_pose": (round(loc.x, 2), round(loc.y, 2), round(loc.z, 2)),
            "ego_speed_mps": float(speed_mps),
            "image_size": self._image_size,
            "camera_fov": float(self._camera_fov),
        }
        # Ground-truth pedestrian pose + velocity for the RSU's TTE/RM evaluation only
        # (never a decision input), exactly as the original client sent them.
        ped = self.source.obstacle_location()
        pv = self.source.obstacle_velocity()
        if ped is not None:
            msg["gt_ped_loc"] = (round(ped.x, 3), round(ped.y, 3), round(ped.z, 3))
        if pv is not None:
            msg["gt_ped_vel_mps"] = pv
        return zlib.compress(pickle.dumps(msg))

    def on_frame_sent(self, frame, ctx) -> None:
        send_time = time.time() - self._start
        sloc = self.source.vehicle_location()
        self.metrics.mark_sent(
            self._cur_fid, send_time, (round(sloc.x, 2), round(sloc.y, 2), round(sloc.z, 2)))
        self._frame_counter += 1

    def collect_metrics(self, frame, ctx) -> None:
        pass

    def is_done(self, ctx) -> bool:
        return self._done

    def message_stats(self):
        from cadet.metrics.link_report import REPLY_ON_EVENT, MessageStats

        sent = received = 0
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
        # Tear down the infra feed (second connection + fixed camera) first.
        try:
            if self._infra_camera is not None:
                self._infra_camera.destroy()
            if self._infra_transport is not None:
                self._infra_transport.close()
        except Exception:
            pass
        self.metrics.write()
        s = self.metrics.summary()
        # In V2I the brake reply carries the *infra* frame_id, which never matches
        # an ego frame record, so the per-frame CSV's server_response_loc stays
        # empty. MetricsLog.{brake,stop}_loc are set unconditionally, so emit them
        # to a side summary the parser can read for brake_y/stop_y.
        try:
            summary_path = os.path.join(
                os.path.dirname(self.metrics.csv_path) or ".", "v2i_summary.json")
            with open(summary_path, "w") as fh:
                json.dump(s, fh)
        except Exception:
            pass
        self._log("[ego] ==== FINAL REPORT ====")
        self._log("[ego] frames_sent={} brake_t={} stop_t={}".format(
            s["frames_sent"], s["brake_time"], s["stop_time"]))


@register_app
class V2IApp(AppPlugin):
    """Vehicle-to-infrastructure emergency-braking app plugin."""

    name = "v2i"
    params_model = V2IParams

    @staticmethod
    def client_success(client_app):
        stats = client_app.message_stats()
        if stats is None or stats.sent < 1:
            return False, "client transmitted no frames"
        braked = client_app.metrics.summary().get("brake_time") is not None
        if not braked and stats.received < 1:
            return False, "client sent {} frames and never braked".format(stats.sent)
        return True, "sent={} braked={}".format(stats.sent, braked)

    @staticmethod
    def sim_hooks(config, model_search_dirs):
        """ScenarioRunner hook: spawn the ego + occluder + crossing pedestrian.

        Same mechanism as the braking/v2v apps: the command is built from the config by
        :func:`cadet.sim.scenario_command.scenario_runner_command` unless
        ``CADET_SCENARIO_RUNNER_CMD`` overrides it. The scenario reads its spawn geometry
        and pedestrian timing from the CADET env set here (no pylot dependency).
        """
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
        scenario = getattr(config.source, "scenario", None)
        env = {}
        weather = getattr(config.source, "weather", None)
        if weather:
            env["CADET_V2I_WEATHER"] = str(weather)  # clear | light | high
        if scenario is not None:
            if getattr(scenario, "ego_vehicle", None):
                env["CADET_SCENARIO_VEHICLE"] = str(scenario.ego_vehicle)
            # Spawn geometry + pedestrian timing for the env-driven scenario file.
            for attr, key in (
                ("ego_spawn", "CADET_V2I_EGO_SPAWN"),
                ("occluder", "CADET_V2I_OCCLUDER"),
                ("pedestrian", "CADET_V2I_PED"),
            ):
                val = getattr(scenario, attr, None)
                if val is not None:
                    import json

                    env[key] = json.dumps(val if isinstance(val, dict) else _as_plain(val))
        marker = os.environ.get("CADET_SR_READY", READY_MARKER)
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
    def build_server(config, model_search_dirs, logger=None) -> "RSUServerApp":
        return RSUServerApp(config, model_search_dirs, logger=logger)

    @staticmethod
    def build_client(
        config, run_id, rep, csv_path, logger=None, node_name=None
    ) -> "V2IClientApp":
        return V2IClientApp(config, run_id, rep, csv_path, logger=logger)
