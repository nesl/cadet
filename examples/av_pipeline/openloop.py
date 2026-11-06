"""Open-loop AV-pipeline stages: stub detector, per-frame decision log.

What this is for:
    The AV pipeline's closed-loop form (``cadet.pipeline.stages.braking``) ends
    in a control stage that actuates a simulator, so it is meaningful only on a
    CARLA source. These stages give the SAME pipeline shape -- detection -> ROI
    gate -> brake decision -- an *open-loop* terminus instead: every frame's
    decision is logged as a record, with no actuation sink. That lets the AV
    processing core run over any source (replay, live, synthetic, dataset), which
    is what the acceptance matrix's "every source x every processing core" asks.

What it contains:
    - :class:`StubDetectorStage` (``stub_detector``): a model-free detector that
      emits ``(class_id, conf, cx)`` boxes per frame. With ground truth forwarded
      in the frame meta (``gt_boxes``; see PipelineAppParams.forward_meta_keys) it
      detects the GT targets, so the pipeline has something real to decide on with
      no weights and no GPU; otherwise it emits a deterministic pattern so the
      decision still varies frame to frame. A real detector uses the ``yolo``
      stage instead -- this one exists so the open-loop path runs anywhere.
    - :class:`DecisionLogStage` (``decision_log``): the open-loop terminus. It
      runs the ROI + consecutive-detection gate (the same rule as
      ``roi_consecutive``) but logs EVERY frame -- detections, in-ROI flag,
      consecutive count, brake decision -- to ``decisions_<node>.csv``, and, when
      ground truth is present, whether the decision was correct (a target is in
      the ROI per GT iff the gate fired). It replies with a :class:`DecisionReport`
      every frame so the client records per-frame latency; it never actuates.

Derived from the flat/stage braking app (ROI gate rule) and ERDOS Pylot
(Apache-2.0); see the project NOTICE.
"""
import csv
import os
from typing import List, Optional

from cadet.core.codec import register_wire_types
from cadet.pipeline.messages import Detections, Message
from cadet.pipeline.registry import register_stage
from cadet.pipeline.stage import PERCEIVED, Stage
from cadet.pipeline.stages.braking import frame_triggers_brake, resolve_target_class_ids

# A tiny fixed class map, so the stub and the gate agree on ids without a model.
STUB_CLASS_NAMES = {0: "car", 1: "person", 2: "bicycle"}


class DecisionReport(Message):
    """Per-frame open-loop decision sent back to the client (keeps latency per frame)."""

    def __init__(self, brake: bool = False, correct: Optional[bool] = None, **kw):
        super().__init__(**kw)
        self.brake = brake
        self.correct = correct


def _gt_target_in_roi(gt_boxes, target_names, roi_min, roi_max) -> Optional[bool]:
    """True/False if GT present: is a target-class box centred in the ROI? None if no GT."""
    if not gt_boxes:
        return None
    wanted = [t.lower() for t in target_names]
    for box in gt_boxes:
        label = str(box.get("label", "")).lower()
        bbox = box.get("bbox") or []
        if len(bbox) >= 3 and any(w in label for w in wanted):
            cx = bbox[0] + bbox[2] / 2.0
            if roi_min <= cx <= roi_max:
                return True
    return False


@register_stage("stub_detector")
class StubDetectorStage(Stage):
    """Model-free detector: emit ``(class_id, conf, cx)`` boxes for the gate.

    Reads ``gt_boxes`` from the frame meta when the app forwarded it, so the
    decision has real targets to act on with no weights; otherwise emits a
    deterministic in-ROI target on a fixed cadence so the decision still varies.
    Carries the GT forward on the emitted message so the decision stage can score.
    """

    data_source = PERCEIVED

    def __init__(self, name, params=None):
        super().__init__(name, params)
        self._period = int(self.params.get("stub_period", 3))  # emit a target every Nth frame
        self._cx = float(self.params.get("stub_cx", 0.5))  # centre x of the stub box (fraction)

    def process(self, msg: Message) -> Optional[Message]:
        gt = (getattr(msg, "meta", None) or {}).get("gt_boxes")
        boxes = []
        if gt:
            # Detect each GT box as its labelled class, centred where GT says.
            name_to_id = {v: k for k, v in STUB_CLASS_NAMES.items()}
            for b in gt:
                label = str(b.get("label", "")).lower()
                cls_id = next((i for n, i in name_to_id.items() if n in label), 0)
                bb = b.get("bbox") or [0, 0, 0, 0]
                cx = bb[0] + (bb[2] / 2.0 if len(bb) > 2 else 0.0)
                boxes.append((cls_id, 0.9, cx))
        elif (msg.frame_id or 0) % self._period == 0:
            # No GT: a deterministic in-ROI "car" so decisions still fire and log.
            boxes.append((0, 0.9, self._cx))
        out = Detections(detections=boxes, frame_id=msg.frame_id)
        out.gt_boxes = gt  # carried to the decision stage (same node, no wire hop)
        return out


@register_stage("decision_log")
class DecisionLogStage(Stage):
    """Open-loop terminus: ROI + consecutive gate, logged per frame, scored vs GT.

    Params: ``target_classes`` (default [car, person]), ``roi`` (cx_min/cx_max),
    ``conf_thresh``, ``detection_threshold``. Writes ``decisions_<node>.csv`` and
    replies with a :class:`DecisionReport` each frame.
    """

    data_source = PERCEIVED
    # Every frame is answered, so an unanswered one is a measurement of the link
    # rather than the normal case; the run's link report reads this.
    answers_every_message = True

    def __init__(self, name, params=None):
        super().__init__(name, params)
        self._targets = list(self.params.get("target_classes", ["car", "person"]))
        self._roi_min = self.params.get("roi", {}).get("cx_min", 0.0)
        self._roi_max = self.params.get("roi", {}).get("cx_max", 1.0)
        self._conf = self.params.get("conf_thresh", 0.3)
        self._threshold = self.params.get("detection_threshold", 1)
        self._target_ids = resolve_target_class_ids(STUB_CLASS_NAMES, self._targets)
        self._consecutive = 0
        self._writer = None
        self._fh = None
        self._n = 0
        self._n_correct = 0
        self._n_scored = 0

    def setup(self, ctx=None) -> None:
        if ctx is None or ctx.config is None:
            return
        cfg = ctx.config
        out_dir = os.path.join(cfg.outputs.root, cfg.experiment.name, cfg.app.plugin)
        os.makedirs(out_dir, exist_ok=True)
        node = ctx.node or "server"
        self._fh = open(os.path.join(out_dir, "decisions_{}.csv".format(node)), "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(
            ["frame_id", "n_detections", "in_roi", "consecutive", "brake", "gt_present", "correct"]
        )

    def process(self, msg: Message) -> Optional[Message]:
        boxes = getattr(msg, "detections", []) or []
        in_roi = frame_triggers_brake(
            boxes, self._target_ids, self._roi_min, self._roi_max, self._conf
        )
        self._consecutive = self._consecutive + 1 if in_roi else 0
        brake = self._consecutive >= self._threshold
        gt = getattr(msg, "gt_boxes", None)
        gt_truth = _gt_target_in_roi(gt, self._targets, self._roi_min, self._roi_max)
        correct = None if gt_truth is None else (brake == gt_truth)
        if correct is not None:
            self._n_scored += 1
            self._n_correct += int(correct)
        self._n += 1
        if self._writer is not None:
            self._writer.writerow([
                msg.frame_id, len(boxes), int(in_roi), self._consecutive,
                int(brake), int(gt_truth is not None),
                "" if correct is None else int(correct),
            ])
        return DecisionReport(brake=brake, correct=correct, frame_id=msg.frame_id)

    def teardown(self) -> None:
        if self._fh is not None:
            if self._n_scored:
                self._fh.write("# decision_accuracy={:.3f} over {} scored frames\n".format(
                    self._n_correct / self._n_scored, self._n_scored))
            self._fh.close()


register_wire_types({"DecisionReport": DecisionReport})
