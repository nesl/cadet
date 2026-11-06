"""Typed messages that flow between pipeline stages.

A pipeline is a chain of stages, each transforming one message into the next
(Frame -> Detections -> Tracks -> Predictions -> Plan -> ControlCommand). Every
message carries the identity and timing that make a run measurable: a frame id,
the simulator and wall capture times, and a timestamp at every stage entry and
exit (the per-stage latency breakdown the paper calls for). Provides
:class:`StageStamp` (one timestamp record), :class:`Message` (the base carrying
frame id, capture times, stamp trail, and per-stage deadline budget), and the
concrete types :class:`Frame`, :class:`Detections`, :class:`Tracks`,
:class:`Predictions`, :class:`Plan`, :class:`ControlCommand`. Stages consume and
produce these; the executor in :mod:`cadet.pipeline.builder` stamps entry/exit
and carries timing forward. These are data types: which stages run and where
comes from the ``pipeline.stages`` config section.

A message crosses a node boundary, so every type here is registered with
:func:`cadet.core.codec.register_wire_types` at the bottom of this module. That
registration is what lets the safe wire encoding rebuild one: it names the
classes a payload is allowed to become, so a boundary message travels without
the receiver unpickling whatever arrived. Registration lives here, beside the
classes, so a type added without a wire name is a visible omission.
"""

import time
from typing import Any, Dict, List, Optional

from cadet.core.codec import register_wire_types


class StageStamp:
    """One timestamp: a stage name, a phase (``enter``/``exit``) and the clock."""

    __slots__ = ("stage", "phase", "t")

    def __init__(self, stage: str, phase: str, t: float):
        self.stage = stage
        self.phase = phase
        self.t = t

    def __repr__(self) -> str:
        return "StageStamp({!r}, {!r}, {:.6f})".format(self.stage, self.phase, self.t)


class Message:
    """Base class for everything that flows through a pipeline.

    ``frame_id`` ties a message to the input frame it derives from; ``sim_time``
    and ``wall_time`` are the simulator and wall-clock capture times of that frame.
    ``stamps`` accumulates one :class:`StageStamp` per stage entry and exit, so the
    per-stage and end-to-end latencies can be reconstructed exactly.
    ``deadline_ttd_ms`` is the time-to-decision budget the machinery hands to the
    current stage (``None`` when no deadline applies); a stage reads it only if it
    adapts its work to the available time.
    """

    def __init__(
        self,
        frame_id: Optional[int] = None,
        sim_time: Optional[float] = None,
        wall_time: Optional[float] = None,
    ):
        self.frame_id = frame_id
        self.sim_time = sim_time
        self.wall_time = wall_time
        self.stamps: List[StageStamp] = []
        self.deadline_ttd_ms: Optional[float] = None

    def stamp(self, stage: str, phase: str) -> None:
        """Append a monotonic timestamp for ``stage`` at ``phase``."""
        self.stamps.append(StageStamp(stage, phase, time.monotonic()))

    def carry_from(self, src: "Message") -> None:
        """Inherit identity, capture times and stamp trail from ``src``.

        Called by the executor so a stage that returns a *new* message only has to
        set its own payload; the frame id, capture times, accumulated stamps and
        current deadline budget propagate automatically. A no-op when a stage
        mutates and returns the same object.
        """
        if src is self:
            return
        if self.frame_id is None:
            self.frame_id = src.frame_id
        if self.sim_time is None:
            self.sim_time = src.sim_time
        if self.wall_time is None:
            self.wall_time = src.wall_time
        # Prepend the upstream trail so stamps stay in chronological order.
        self.stamps = src.stamps + self.stamps
        self.deadline_ttd_ms = src.deadline_ttd_ms

    def stage_latencies_ms(self) -> Dict[str, float]:
        """Return ``{stage: exit - enter}`` in milliseconds for each stage.

        A stage with only an enter stamp (dropped mid-stage) is omitted.
        """
        enters: Dict[str, float] = {}
        out: Dict[str, float] = {}
        for s in self.stamps:
            if s.phase == "enter":
                enters[s.stage] = s.t
            elif s.phase == "exit" and s.stage in enters:
                out[s.stage] = (s.t - enters[s.stage]) * 1000.0
        return out

    def end_to_end_ms(self) -> Optional[float]:
        """First-enter to last-exit latency in milliseconds (``None`` if unstamped)."""
        if not self.stamps:
            return None
        return (self.stamps[-1].t - self.stamps[0].t) * 1000.0


class Frame(Message):
    """A captured sensor frame: an image plus arbitrary per-frame metadata."""

    def __init__(self, image: Any = None, meta: Optional[Dict[str, Any]] = None, **kw):
        super().__init__(**kw)
        self.image = image
        self.meta = meta or {}


class Detections(Message):
    """Per-frame detections: a list of ``(class_id, confidence, bbox)``-style items."""

    def __init__(self, detections: Optional[List[Any]] = None, **kw):
        super().__init__(**kw)
        self.detections = detections or []


class Tracks(Message):
    """Tracked obstacles with stable ids across frames."""

    def __init__(self, tracks: Optional[List[Any]] = None, **kw):
        super().__init__(**kw)
        self.tracks = tracks or []


class Predictions(Message):
    """Predicted future trajectories for tracked obstacles."""

    def __init__(self, predictions: Optional[List[Any]] = None, **kw):
        super().__init__(**kw)
        self.predictions = predictions or []


class Plan(Message):
    """A planned trajectory (waypoints) for the ego to follow."""

    def __init__(self, waypoints: Optional[List[Any]] = None, **kw):
        super().__init__(**kw)
        self.waypoints = waypoints or []


class ControlCommand(Message):
    """A low-level actuation command for the ego vehicle."""

    def __init__(self, throttle: float = 0.0, steer: float = 0.0, brake: float = 0.0, **kw):
        super().__init__(**kw)
        self.throttle = throttle
        self.steer = steer
        self.brake = brake


# The wire names these types travel under. Explicit and stable: the name is what
# a payload carries, so renaming a class must not change the wire, and a class
# that is not here cannot be rebuilt from a payload at all.
register_wire_types(
    {
        "StageStamp": StageStamp,
        "Message": Message,
        "Frame": Frame,
        "Detections": Detections,
        "Tracks": Tracks,
        "Predictions": Predictions,
        "Plan": Plan,
        "ControlCommand": ControlCommand,
    }
)
