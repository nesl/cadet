"""Unit tests for the boundary-driven chain planner (cadet.pipeline.runtime).

These are pure (no sockets): they check that plan_chain derives the right origin,
away hops, head/tail split, downstream links and inbound queue policy from the
placement — for the two-node braking shape and a three-node placement chain.
"""

import cadet.pipeline.stages.braking  # noqa: F401  (registers camera/yolo/... stages)
from cadet.config.schema import StageSpec
from cadet.pipeline.builder import build_pipeline
from cadet.pipeline.runtime import plan_chain


def _spec(name, impl, node, **kw):
    return StageSpec(name=name, impl=impl, node=node, **kw)


def _decide_params():
    return {
        "target_ids": [2],
        "roi": {"cx_min": 300, "cx_max": 500},
        "conf_thresh": 0.3,
        "detection_threshold": 3,
    }


def _control_params():
    return {
        "ego_vehicle": "audi.tt",
        "ego_speed": "40mph",
        "controllers": {
            "audi.tt": {"target_decel": 7.0, "max_decel": 7.3, "kp": 0.2, "max_ramp": 0.05}
        },
    }


def test_two_node_braking_chain():
    specs = [
        _spec("camera", "camera", "client"),
        _spec("detect", "yolo", "server", queue_policy="latest_only"),
        _spec("decide", "roi_consecutive", "server", params=_decide_params()),
        _spec("control", "brake_controller", "client", params=_control_params()),
    ]
    plan = plan_chain(build_pipeline(specs))
    assert plan.origin == "client"
    assert plan.node_chain == ["client", "server", "client"]  # 2 distinct nodes, round trip
    assert plan.origin_has_tail is True
    assert [ps.name for ps in plan.head_stages()] == ["camera"]
    assert [ps.name for ps in plan.tail_stages()] == ["control"]
    assert plan.first_away_node() == "server"
    assert [ps.name for ps in plan.local_stages("server")] == ["detect", "decide"]
    assert plan.downstream_of("server") is None  # last away hop -> replies up
    assert plan.inbound_policy("server") == "latest_only"


def test_three_node_placement_chain():
    # P6-shape: detection on an edge node between client and server.
    specs = [
        _spec("camera", "camera", "client"),
        _spec("detect", "yolo", "edge", queue_policy="fifo"),
        _spec("decide", "roi_consecutive", "server", params=_decide_params()),
        _spec("control", "brake_controller", "client", params=_control_params()),
    ]
    plan = plan_chain(build_pipeline(specs))
    assert plan.origin == "client" and plan.origin_has_tail is True
    assert plan.node_chain == ["client", "edge", "server", "client"]
    assert plan.first_away_node() == "edge"
    assert plan.downstream_of("edge") == "server"  # edge forwards to server
    assert plan.downstream_of("server") is None  # server is the last away hop
    assert [ps.name for ps in plan.local_stages("edge")] == ["detect"]
    assert [ps.name for ps in plan.local_stages("server")] == ["decide"]
    assert plan.inbound_policy("edge") == "fifo"


def test_forward_only_chain_without_return():
    # If the last segment is NOT on the origin, there is no tail / return.
    specs = [
        _spec("camera", "camera", "client"),
        _spec("detect", "yolo", "server"),
    ]
    plan = plan_chain(build_pipeline(specs))
    assert plan.origin == "client" and plan.origin_has_tail is False
    assert plan.tail_stages() == []
    assert plan.first_away_node() == "server"
    assert plan.downstream_of("server") is None
