"""Typed, per-app parameter models for CADET experiment configs.

What this is for:
    Each CADET application (open-loop braking, V2V, V2I) has its own set of
    parameters. Rather than a free-form dict, every app declares a pydantic
    model here. ``Config.app`` is a discriminated union over these models keyed
    by the ``plugin`` field, so a config is validated against exactly the model
    for the app it declares.

What it contains:
    - shared leaf models: ``BrakeController``, ``RoiBand``.
    - per-app models: ``BrakingParams``, ``V2VParams``, ``V2IParams`` (each with
      a ``plugin`` ``Literal`` discriminator) and their nested option models.
    - ``APP_PARAM_MODELS``: plugin name -> model class (used by the core app
      registry), and ``AppParams``: the discriminated-union annotation used by
      the schema.

How it fits into CADET:
    ``cadet.config.schema.Config.app`` is typed as :data:`AppParams`.
    ``cadet.core.apps`` registers the same models so app logic (added later) and
    config validation agree on the parameter shape. Brake-controller defaults
    are kept per app, matching the original ``*_braking_params.py`` files.
"""
from typing import Dict, List, Union

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated, Literal


class _Strict(BaseModel):
    """Base model that rejects unknown keys, so typos surface as errors."""

    model_config = ConfigDict(extra="forbid")


class BrakeController(_Strict):
    """Per-vehicle deceleration controller gains (as in the original params)."""

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


# Per-app brake-controller defaults. Kept separate (not unified) because the
# open-loop/V2V and V2I experiments were tuned to different values.
def _braking_controllers() -> Dict[str, BrakeController]:
    return {
        "audi.tt": BrakeController(target_decel=7.0, max_decel=7.3, kp=0.2, max_ramp=0.05),
        "carlamotors.carlacola": BrakeController(target_decel=3.0, max_decel=3.0, kp=0.45, max_ramp=0.15),
        "kawasaki.ninja": BrakeController(target_decel=6.0, max_decel=6.0, kp=0.2, max_ramp=0.05),
    }


def _v2i_controllers() -> Dict[str, BrakeController]:
    return {
        "audi.tt": BrakeController(target_decel=6.0, max_decel=7.0, kp=0.2, max_ramp=0.1),
        "carlamotors.carlacola": BrakeController(target_decel=3.0, max_decel=3.0, kp=0.45, max_ramp=0.15),
        "kawasaki.ninja": BrakeController(target_decel=6.0, max_decel=6.0, kp=0.2, max_ramp=0.05),
    }


# ---------------------------------------------------------------------------
# Open-loop braking
# ---------------------------------------------------------------------------
class BrakingParams(_Strict):
    """Parameters for the single-vehicle open-loop emergency-braking app."""

    plugin: Literal["braking"] = "braking"
    controllers: Dict[str, BrakeController] = Field(default_factory=_braking_controllers)
    # Consecutive positive detections required before issuing a brake command.
    detection_threshold: int = 3
    # Detector classes that count as a braking-relevant obstacle.
    target_classes: List[str] = Field(
        default_factory=lambda: ["car", "motorcycle", "bicycle", "person"]
    )
    # Decision thresholds (owned here, not in the model registry): a detection
    # counts only if confidence exceeds conf_thresh and its box centre-x falls in
    # the ROI band. Original open-loop values: conf 0.3, ROI cx in [300, 500].
    conf_thresh: float = 0.3
    roi: RoiBand = Field(default_factory=lambda: RoiBand(cx_min=300, cx_max=500))


# ---------------------------------------------------------------------------
# Vehicle-to-vehicle (follower reacts to a scripted leader)
# ---------------------------------------------------------------------------
class V2VTtc(_Strict):
    """Time-to-collision gate for the V2V follower."""

    thresh_s: float = 2.0
    k_frames: int = 2  # frames below threshold before braking (1 = instant)
    alpha: float = 0.30  # EMA smoothing factor


class V2VPerception(_Strict):
    """Lead-vehicle selection and detection gating on the server."""

    roi: RoiBand = Field(default_factory=lambda: RoiBand(cx_min=300, cx_max=500))
    conf_thresh: float = 0.30
    use_consecutive: bool = False
    consecutive_n: int = 3


class V2VParams(_Strict):
    """Parameters for the V2V braking app."""

    plugin: Literal["v2v"] = "v2v"
    controllers: Dict[str, BrakeController] = Field(default_factory=_braking_controllers)
    headway_m: float = 30.0
    t_brake_sched_s: float = 6.0
    decel_mps2: float = -5.5
    hold_cruise: bool = True
    braking_mode: str = "v2v"
    ttc: V2VTtc = Field(default_factory=V2VTtc)
    perception: V2VPerception = Field(default_factory=V2VPerception)


# ---------------------------------------------------------------------------
# Vehicle-to-infrastructure (roadside camera assists an occluded ego)
# ---------------------------------------------------------------------------
class InfraCamera(_Strict):
    """Roadside-unit (pole) camera pose and intrinsics for V2I."""

    enabled: bool = True
    fov_deg: float = 70.0
    height_m: float = 15.8
    pitch_deg: float = -39.0
    base_x: float = -141.0
    base_y: float = -76.0
    base_z: float = 0.0
    yaw_deg: float = 0.0
    image_width: int = 1280
    image_height: int = 720


class V2IPerception(_Strict):
    """Two-camera (ego + infra) risk gating on the RSU server."""

    ego_roi: RoiBand = Field(default_factory=lambda: RoiBand(cx_min=150, cx_max=700, cy_min=50, cy_max=700))
    infra_roi: RoiBand = Field(default_factory=lambda: RoiBand(cx_min=850, cx_max=1150, cy_min=100, cy_max=600))
    conf_thresh: float = 0.40
    use_consecutive: bool = False
    consecutive_n: int = 3
    ttc_thresh_s: float = 3.0
    collision_window_s: float = 2.0
    prediction_horizon_s: float = 7.0
    min_ped_speed_pxps: float = 1.5


class V2IParams(_Strict):
    """Parameters for the V2I braking app."""

    plugin: Literal["v2i"] = "v2i"
    controllers: Dict[str, BrakeController] = Field(default_factory=_v2i_controllers)
    infra_camera: InfraCamera = Field(default_factory=InfraCamera)
    perception: V2IPerception = Field(default_factory=V2IPerception)
    mode: Literal["baseline_local", "baseline_cloud", "v2i_local", "v2i_external"] = "baseline_local"


# plugin name -> model class. The schema and the core app registry both use this.
APP_PARAM_MODELS = {
    "braking": BrakingParams,
    "v2v": V2VParams,
    "v2i": V2IParams,
}

# Discriminated union used as the type of ``Config.app``: pydantic selects the
# model whose ``plugin`` literal matches the config's ``app.plugin`` value.
AppParams = Annotated[Union[BrakingParams, V2VParams, V2IParams], Field(discriminator="plugin")]
