"""CADET pipeline: placeable, measurable autonomy stages (core package).

The staged form of a CADET application. Where a flat app is one monolithic
client/server, a pipeline is an ordered list of small stages, each placeable on
a node by config. Consecutive stages on one node run in-process; a node change
becomes a transport boundary the network layer can emulate. Every message
carries timing stamped at each stage, giving the per-stage latency breakdown.
Subpackages: :mod:`~cadet.pipeline.messages` (typed messages),
:mod:`~cadet.pipeline.stage` (the Stage contract),
:mod:`~cadet.pipeline.registry` (impl-name registry),
:mod:`~cadet.pipeline.queue` (inter-stage queues),
:mod:`~cadet.pipeline.deadline` (time-to-decision budgets),
:mod:`~cadet.pipeline.builder` (placement + executor),
:mod:`~cadet.pipeline.validate` (registry check). Built from the config's
``pipeline.stages`` section; see ``docs/config-reference.md`` for the schema.
"""

from cadet.pipeline.builder import (
    Boundary,
    Pipeline,
    PlacedStage,
    Segment,
    build_pipeline,
    plan_placement,
)
from cadet.pipeline.messages import (
    ControlCommand,
    Detections,
    Frame,
    Message,
    Plan,
    Predictions,
    StageStamp,
    Tracks,
)
from cadet.pipeline.queue import FIFO, LATEST_ONLY, StageQueue
from cadet.pipeline.registry import (
    available_stages,
    get_stage,
    import_stage_modules,
    register_stage,
)
from cadet.pipeline.stage import GROUND_TRUTH, PERCEIVED, Stage, StageContext
from cadet.pipeline.validate import validate_pipeline

__all__ = [
    "Message",
    "StageStamp",
    "Frame",
    "Detections",
    "Tracks",
    "Predictions",
    "Plan",
    "ControlCommand",
    "Stage",
    "StageContext",
    "PERCEIVED",
    "GROUND_TRUTH",
    "register_stage",
    "get_stage",
    "available_stages",
    "import_stage_modules",
    "StageQueue",
    "LATEST_ONLY",
    "FIFO",
    "Pipeline",
    "PlacedStage",
    "Segment",
    "Boundary",
    "plan_placement",
    "build_pipeline",
    "validate_pipeline",
]
