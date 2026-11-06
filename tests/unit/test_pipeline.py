"""Unit tests for the pipeline machinery (P1).

Covers, with synthetic stages and no CARLA: message timing/stamps, the stage
registry + builder, placement into same-node segments with auto-inserted
boundaries, queue freshness policies, and per-stage deadline budgets.
"""

import pytest

from cadet.config.schema import Config, DeadlineSpec, StageSpec
from cadet.pipeline import (
    FIFO,
    GROUND_TRUTH,
    LATEST_ONLY,
    ControlCommand,
    Detections,
    Frame,
    Plan,
    Stage,
    StageQueue,
    build_pipeline,
    plan_placement,
    register_stage,
    validate_pipeline,
)
from cadet.pipeline.builder import PlacedStage
from cadet.pipeline.deadline import ttd_ms


# --------------------------------------------------------------------------
# synthetic stages (registered once at import)
# --------------------------------------------------------------------------
@register_stage("t_detect")
class _Detect(Stage):
    def process(self, msg):
        return Detections(detections=[("car", 0.9, 400)])


@register_stage("t_plan")
class _Plan(Stage):
    def process(self, msg):
        return Plan(waypoints=[(0, 0), (1, 1)])


@register_stage("t_control")
class _Control(Stage):
    def process(self, msg):
        return ControlCommand(brake=1.0)


@register_stage("t_drop")
class _Drop(Stage):
    def process(self, msg):
        return None  # drops the message


@register_stage("t_record_ttd")
class _RecordTtd(Stage):
    def __init__(self, name, params=None):
        super().__init__(name, params)
        self.seen_ttd = "unset"

    def process(self, msg):
        self.seen_ttd = msg.deadline_ttd_ms
        return msg


@register_stage("t_groundtruth")
class _GroundTruth(Stage):
    data_source = GROUND_TRUTH

    def process(self, msg):
        return msg


def _spec(name, impl, node, **kw):
    return StageSpec(name=name, impl=impl, node=node, **kw)


# --------------------------------------------------------------------------
# messages / timestamps
# --------------------------------------------------------------------------
def test_timing_carries_and_stamps_each_stage():
    specs = [
        _spec("det", "t_detect", "server"),
        _spec("plan", "t_plan", "server"),
        _spec("ctl", "t_control", "server"),
    ]
    out = build_pipeline(specs).run_once(Frame(image=0, frame_id=42, sim_time=5.0, wall_time=1.0))
    assert isinstance(out, ControlCommand)
    # identity + capture times propagate through the new message types
    assert out.frame_id == 42 and out.sim_time == 5.0 and out.wall_time == 1.0
    lat = out.stage_latencies_ms()
    assert set(lat) == {"det", "plan", "ctl"}  # one latency per stage
    assert all(v >= 0 for v in lat.values())
    assert out.end_to_end_ms() is not None
    # enter/exit ordering preserved across the chain
    phases = [(s.stage, s.phase) for s in out.stamps]
    assert phases == [
        ("det", "enter"),
        ("det", "exit"),
        ("plan", "enter"),
        ("plan", "exit"),
        ("ctl", "enter"),
        ("ctl", "exit"),
    ]


def test_dropped_message_stops_pipeline():
    specs = [_spec("det", "t_detect", "server"), _spec("drop", "t_drop", "server")]
    assert build_pipeline(specs).run_once(Frame(frame_id=1)) is None


def test_ground_truth_declaration():
    assert _GroundTruth("g").data_source == GROUND_TRUTH
    assert _Detect("d").data_source == "perceived"


# --------------------------------------------------------------------------
# placement: segments + boundaries
# --------------------------------------------------------------------------
def _placed(pairs):
    # pairs: list of (name, node); impl is irrelevant to placement
    return [PlacedStage(None, _spec(n, "t_detect", node)) for n, node in pairs]


def test_single_node_has_no_boundary():
    segs, bounds = plan_placement(_placed([("a", "s"), ("b", "s"), ("c", "s")]))
    assert len(segs) == 1 and segs[0].node == "s" and segs[0].names == ["a", "b", "c"]
    assert bounds == []


def test_node_change_inserts_boundary():
    placed = [
        PlacedStage(None, _spec("a", "t_detect", "client")),
        PlacedStage(None, _spec("b", "t_detect", "client")),
        PlacedStage(None, _spec("c", "t_detect", "server", queue_policy="fifo")),
        PlacedStage(None, _spec("d", "t_detect", "server")),
    ]
    segs, bounds = plan_placement(placed)
    assert [s.node for s in segs] == ["client", "server"]
    assert segs[0].names == ["a", "b"] and segs[1].names == ["c", "d"]
    assert len(bounds) == 1
    assert (bounds[0].from_node, bounds[0].to_node) == ("client", "server")
    assert bounds[0].queue_policy == "fifo"  # the receiving stage's policy


def test_alternating_nodes_make_two_boundaries():
    segs, bounds = plan_placement(_placed([("a", "s"), ("b", "c"), ("x", "s")]))
    assert len(segs) == 3 and len(bounds) == 2
    assert [(b.from_node, b.to_node) for b in bounds] == [("s", "c"), ("c", "s")]


# --------------------------------------------------------------------------
# queue freshness policies
# --------------------------------------------------------------------------
def test_latest_only_keeps_newest_and_counts_drops():
    q = StageQueue(LATEST_ONLY)
    q.put("a")
    q.put("b")
    q.put("c")
    assert len(q) == 1
    assert q.get() == "c"
    assert q.get() is None
    assert q.drops == 2


def test_fifo_preserves_order_and_drops_oldest_when_full():
    q = StageQueue(FIFO, maxsize=2)
    q.put(1)
    q.put(2)
    q.put(3)  # 1 is dropped
    assert q.drops == 1
    assert q.get() == 2 and q.get() == 3 and q.get() is None


def test_unknown_queue_policy_rejected():
    with pytest.raises(ValueError):
        StageQueue("sometimes")


# --------------------------------------------------------------------------
# deadlines
# --------------------------------------------------------------------------
def test_ttd_none_static_dynamic():
    f = Frame(frame_id=1, wall_time=1000.0)
    assert ttd_ms("none", None, f) is None
    assert ttd_ms(None, None, f) is None
    assert ttd_ms("static", 50.0, f) == 50.0
    # dynamic: 100ms budget, 30ms already elapsed -> 70ms remaining
    assert ttd_ms("dynamic", 100.0, f, now=1000.03) == pytest.approx(70.0, abs=1e-6)
    # dynamic can go negative once the budget is blown
    assert ttd_ms("dynamic", 10.0, f, now=1000.5) == pytest.approx(-490.0, abs=1e-6)


def test_executor_hands_stage_its_deadline_budget():
    # static budget is handed through verbatim
    specs = [
        _spec(
            "rec", "t_record_ttd", "server", deadline=DeadlineSpec(policy="static", budget_ms=25.0)
        )
    ]
    pipe = build_pipeline(specs)
    pipe.run_once(Frame(frame_id=1, wall_time=1.0))
    assert pipe.placed[0].stage.seen_ttd == 25.0
    # no deadline -> None
    specs2 = [_spec("rec", "t_record_ttd", "server")]
    pipe2 = build_pipeline(specs2)
    pipe2.run_once(Frame(frame_id=1, wall_time=1.0))
    assert pipe2.placed[0].stage.seen_ttd is None


# --------------------------------------------------------------------------
# registry validation + schema
# --------------------------------------------------------------------------
def test_validate_pipeline_flags_unknown_impl():
    cfg = _min_config(stages=[{"name": "x", "impl": "no_such_stage", "node": "server"}])
    problems = validate_pipeline(cfg)
    assert len(problems) == 1 and "no_such_stage" in problems[0]
    cfg_ok = _min_config(stages=[{"name": "x", "impl": "t_detect", "node": "server"}])
    assert validate_pipeline(cfg_ok) == []


def test_schema_rejects_stage_on_unknown_node():
    with pytest.raises(Exception):
        _min_config(stages=[{"name": "x", "impl": "t_detect", "node": "ghost"}])


def test_schema_rejects_duplicate_stage_names():
    with pytest.raises(Exception):
        _min_config(
            stages=[
                {"name": "x", "impl": "t_detect", "node": "server"},
                {"name": "x", "impl": "t_plan", "node": "server"},
            ]
        )


def test_schema_static_deadline_requires_budget():
    with pytest.raises(Exception):
        DeadlineSpec(policy="static")


def _min_config(stages):
    return Config.model_validate(
        {
            "experiment": {"name": "t", "app": "ref"},
            "source": {"kind": "synthetic"},
            "nodes": [
                {"name": "client", "host": "localhost", "role": "client"},
                {"name": "server", "host": "127.0.0.1", "port": 9000, "role": "server"},
            ],
            "app": {"plugin": "ref"},
            "pipeline": {"stages": stages},
        }
    )
