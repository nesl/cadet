"""Tests for multi-hop routes: the path checks, and the composed behaviour.

A route is an ordering over existing links, so most of what can go wrong is a
config that does not describe a path at all (hops that do not chain, a waypoint
that is not a node, a route beside a direct link). Those are checked here
against the schema, and the composed per-message model is checked against hops
whose numbers are known exactly.
"""

import random

import pytest
from pydantic import ValidationError

from cadet.config.implemented import tier, unimplemented_features
from cadet.config.routes import hops_in_direction, is_route, path_into_node, route_hops
from cadet.config.schema import Config
from cadet.core.network import (
    CompositeNetworkModel,
    NullNetworkModel,
    build_path_network_model,
)
from cadet.metrics.link_report import MessageStats, write_link_report

NODES = [
    {"name": "ego", "host": "127.0.0.1", "role": "client"},
    {"name": "rsu", "host": "127.0.0.1", "role": "server", "port": 10101},
    {"name": "follower", "host": "127.0.0.1", "role": "server", "port": 10102},
]


def _delay(ms, loss=0.0, bandwidth=None):
    net = {"mode": "delay", "distribution": {"kind": "constant", "value_ms": ms}, "loss": loss}
    if bandwidth is not None:
        net["bandwidth_mbps"] = bandwidth
    return net


def _config(links, routes, **extra):
    data = {
        "experiment": {"name": "routes", "app": "reference"},
        "source": {"kind": "synthetic", "count": 1},
        "nodes": NODES,
        "links": links,
        "routes": routes,
        "app": {"plugin": "reference"},
    }
    data.update(extra)
    return Config(**data)


TWO_HOPS = [
    {"name": "access", "from": "ego", "to": "rsu", "network": _delay(20.0)},
    {"name": "backhaul", "from": "rsu", "to": "follower", "network": _delay(40.0)},
]
ROUTE = [{"name": "ego_to_follower", "from": "ego", "to": "follower",
          "hops": ["access", "backhaul"]}]


# --- the path checks ------------------------------------------------------
def test_a_route_over_chaining_hops_validates():
    config = _config(TWO_HOPS, ROUTE)
    route = config.routes[0]
    assert [hop.name for hop in route_hops(config, route)] == ["access", "backhaul"]
    assert route.realise == "compose"


def test_hops_must_chain():
    links = [
        {"name": "access", "from": "ego", "to": "rsu", "network": _delay(20.0)},
        {"name": "other", "from": "ego", "to": "follower", "network": _delay(40.0)},
    ]
    with pytest.raises(ValidationError) as err:
        _config(links, [{"name": "r", "from": "ego", "to": "follower",
                         "hops": ["access", "other"]}])
    assert "does not chain" in str(err.value)


def test_the_first_hop_must_start_at_the_route():
    with pytest.raises(ValidationError) as err:
        _config(TWO_HOPS, [{"name": "r", "from": "rsu", "to": "follower",
                            "hops": ["access", "backhaul"]}])
    assert "starts at" in str(err.value)


def test_a_hop_must_be_a_declared_link():
    with pytest.raises(ValidationError) as err:
        _config(TWO_HOPS, [{"name": "r", "from": "ego", "to": "follower",
                            "hops": ["access", "nonexistent"]}])
    assert "unknown link" in str(err.value)


def test_one_hop_is_a_link_not_a_route():
    with pytest.raises(ValidationError) as err:
        _config(TWO_HOPS, [{"name": "r", "from": "ego", "to": "rsu", "hops": ["access"]}])
    assert "declare the link alone" in str(err.value)


def test_a_route_beside_a_direct_link_is_refused():
    # Two answers for "what path leads into follower?" is the ambiguity the
    # explicit-hops decision exists to avoid.
    links = TWO_HOPS + [
        {"name": "direct", "from": "ego", "to": "follower", "network": _delay(5.0)}
    ]
    with pytest.raises(ValidationError) as err:
        _config(links, ROUTE)
    assert "two answers" in str(err.value)


def test_a_link_belongs_to_one_route():
    # `access` is the first hop of both paths, so which route a measurement on
    # it belongs to has no answer.
    nodes = NODES + [{"name": "cloud", "host": "127.0.0.1", "role": "server", "port": 10103}]
    links = TWO_HOPS + [
        {"name": "uplink", "from": "rsu", "to": "cloud", "network": _delay(5.0)}
    ]
    routes = ROUTE + [
        {"name": "ego_to_cloud", "from": "ego", "to": "cloud", "hops": ["access", "uplink"]}
    ]
    with pytest.raises(ValidationError) as err:
        Config(
            experiment={"name": "routes", "app": "reference"},
            source={"kind": "synthetic", "count": 1},
            nodes=nodes,
            links=links,
            routes=routes,
            app={"plugin": "reference"},
        )
    assert "belongs to one path" in str(err.value)


def test_a_reverse_link_between_the_same_pair_is_not_an_ambiguity():
    # It is a path into the other endpoint, so nothing is shadowed.
    links = TWO_HOPS + [
        {"name": "back", "from": "follower", "to": "ego", "network": _delay(5.0)}
    ]
    config = _config(links, ROUTE)
    assert is_route(path_into_node(config, "follower"))
    assert path_into_node(config, "ego").name == "back"


def test_via_is_checked_against_the_hops():
    with pytest.raises(ValidationError) as err:
        _config(TWO_HOPS, [{"name": "r", "from": "ego", "to": "follower",
                            "hops": ["access", "backhaul"], "via": ["ego"]}])
    assert "pass through" in str(err.value)

    config = _config(TWO_HOPS, [{"name": "r", "from": "ego", "to": "follower",
                                 "hops": ["access", "backhaul"], "via": ["rsu"]}])
    assert config.routes[0].via == ["rsu"]


def test_one_path_is_emulated_by_one_backend():
    links = [
        {"name": "access", "from": "ego", "to": "rsu", "network": _delay(20.0)},
        {
            "name": "backhaul",
            "from": "rsu",
            "to": "follower",
            "network": {"mode": "netem",
                        "distribution": {"kind": "constant", "value_ms": 40.0}},
        },
    ]
    with pytest.raises(ValidationError) as err:
        _config(links, ROUTE)
    assert "mixes link modes" in str(err.value)


def test_a_relayed_waypoint_must_listen():
    nodes = [dict(n) for n in NODES]
    nodes[1].pop("port")
    nodes[1]["role"] = "client"
    with pytest.raises(ValidationError) as err:
        Config(
            experiment={"name": "routes", "app": "reference"},
            source={"kind": "synthetic", "count": 1},
            nodes=nodes,
            links=TWO_HOPS,
            routes=[dict(ROUTE[0], realise="relay")],
            app={"plugin": "reference"},
        )
    assert "a relay listens" in str(err.value)


# --- the lookup -----------------------------------------------------------
def test_the_route_is_the_path_into_its_destination():
    config = _config(TWO_HOPS, ROUTE)
    path = path_into_node(config, "follower")
    assert is_route(path) and path.name == "ego_to_follower"
    # A hop still answers for its own destination, which is what a relayed
    # route's waypoint needs.
    hop = path_into_node(config, "rsu")
    assert not is_route(hop) and hop.name == "access"


def test_no_routes_means_the_old_lookup():
    config = _config(TWO_HOPS, [])
    assert path_into_node(config, "follower").name == "backhaul"
    assert path_into_node(config, "ego") is None


def test_direction_selects_the_hops_that_act():
    links = [
        dict(TWO_HOPS[0], direction="forward"),
        dict(TWO_HOPS[1], direction="reverse"),
    ]
    config = _config(links, ROUTE)
    hops = route_hops(config, config.routes[0])
    assert [h.name for h in hops_in_direction(hops, "forward")] == ["access"]
    assert [h.name for h in hops_in_direction(hops, "reverse")] == ["backhaul"]


# --- the composed model --------------------------------------------------
def test_delays_add_across_hops():
    config = _config(TWO_HOPS, ROUTE)
    nets = [hop.network for hop in route_hops(config, config.routes[0])]
    model = build_path_network_model(nets, rng=random.Random(1))
    assert isinstance(model, CompositeNetworkModel)
    assert model.sample_delay_s() == pytest.approx(0.060)


def test_one_hop_behaves_exactly_like_the_link():
    config = _config(TWO_HOPS, ROUTE)
    net = route_hops(config, config.routes[0])[0].network
    assert build_path_network_model([net]).sample_delay_s() == pytest.approx(0.020)


def test_a_message_must_survive_every_hop():
    links = [
        {"name": "access", "from": "ego", "to": "rsu", "network": _delay(1.0, loss=1.0)},
        {"name": "backhaul", "from": "rsu", "to": "follower", "network": _delay(1.0)},
    ]
    config = _config(links, ROUTE)
    nets = [hop.network for hop in route_hops(config, config.routes[0])]
    model = build_path_network_model(nets, rng=random.Random(2))
    assert model.drops() is True


def test_the_bottleneck_sets_the_rate():
    links = [
        {"name": "access", "from": "ego", "to": "rsu",
         "network": _delay(1.0, bandwidth=10.0)},
        {"name": "backhaul", "from": "rsu", "to": "follower",
         "network": _delay(1.0, bandwidth=100.0)},
    ]
    config = _config(links, ROUTE)
    nets = [hop.network for hop in route_hops(config, config.routes[0])]
    model = build_path_network_model(nets, rng=random.Random(3))
    # 1 Mbit at 10 Mbps is 0.1 s; the faster hop does not add to it.
    assert model.serialization_s(125000) == pytest.approx(0.1)


def test_hops_that_shape_in_the_kernel_are_not_delayed_in_process():
    links = [
        {"name": "access", "from": "ego", "to": "rsu",
         "network": {"mode": "netem", "distribution": {"kind": "constant", "value_ms": 20.0}}},
        {"name": "backhaul", "from": "rsu", "to": "follower",
         "network": {"mode": "netem", "distribution": {"kind": "constant", "value_ms": 40.0}}},
    ]
    config = _config(links, ROUTE)
    nets = [hop.network for hop in route_hops(config, config.routes[0])]
    assert isinstance(build_path_network_model(nets), NullNetworkModel)


# --- the record and the gate ---------------------------------------------
def test_the_report_records_every_hop_and_the_composition(tmp_path):
    config = _config(TWO_HOPS, ROUTE)
    report = write_link_report(config, str(tmp_path), MessageStats(sent=2, received=2))
    # A hop is reported inside its route, not twice.
    assert report["configured_links"] == []
    route = report["configured_routes"][0]
    assert route["via"] == ["rsu"]
    assert [hop["name"] for hop in route["hops"]] == ["access", "backhaul"]
    end = route["end_to_end"]["forward"]
    assert end["p50_ms"] == pytest.approx(60.0)
    assert end["p99_ms_upper_bound"] == pytest.approx(60.0)


def test_loss_composes_as_surviving_every_hop(tmp_path):
    links = [
        {"name": "access", "from": "ego", "to": "rsu", "network": _delay(1.0, loss=0.1)},
        {"name": "backhaul", "from": "rsu", "to": "follower", "network": _delay(1.0, loss=0.2)},
    ]
    config = _config(links, ROUTE)
    report = write_link_report(config, str(tmp_path), MessageStats(sent=1, received=1))
    end = report["configured_routes"][0]["end_to_end"]["forward"]
    assert end["loss"] == pytest.approx(1.0 - 0.9 * 0.8)


def test_compose_is_implemented_and_relay_is_planned():
    assert tier("routes.realise", "compose") == "implemented"
    assert tier("routes.realise", "relay") == "planned"
    config = _config(TWO_HOPS, ROUTE)
    assert unimplemented_features(config) == []


def test_a_relayed_route_stays_refused_however_loudly_it_asks():
    config = _config(TWO_HOPS, [dict(ROUTE[0], realise="relay")])
    refused = unimplemented_features(config, allow_experimental=True)
    assert any("realise=relay" in f for f in refused), refused


# --- composing for the kernel backend ------------------------------------
def test_the_kernel_shaping_is_composed_not_per_hop():
    from cadet.network.shaping import shaping_from_path

    nets = [
        _netem(20.0),
        _netem(40.0),
    ]
    assert shaping_from_path(nets).netem_args() == ["delay", "60.0ms"]


def test_one_hop_shapes_exactly_as_the_link_would():
    from cadet.network.shaping import shaping_from_network, shaping_from_path

    net = _netem(30.0)
    assert shaping_from_path([net]).netem_args() == shaping_from_network(net).netem_args()


def test_a_rate_on_any_hop_serialises_the_whole_path():
    # A jittered hop with no bandwidth of its own is refused as a link, because
    # netem would reorder its flow. In a composed path there is one qdisc, so a
    # bandwidth on any hop serialises the output and the reordering cannot
    # happen - the rule is deliberately weaker on a path than on a hop.
    from cadet.config.schema import LognormalDist, NetworkModel
    from cadet.network.shaping import (
        UnrepresentableDelay,
        shaping_from_network,
        shaping_from_path,
    )

    jittered = NetworkModel(
        mode="netem", distribution=LognormalDist(p50_ms=30.0, p99_ms=50.0)
    )
    with pytest.raises(UnrepresentableDelay):
        shaping_from_network(jittered)

    rated = NetworkModel(
        mode="netem",
        distribution={"kind": "constant", "value_ms": 10.0},
        bandwidth_mbps=50.0,
    )
    shaping = shaping_from_path([jittered, rated])
    assert shaping.preserves_order
    assert "rate" in shaping.netem_args()
    assert shaping.bandwidth_mbps == 50.0


def test_a_trace_hop_has_no_kernel_equivalent():
    from cadet.config.schema import NetworkModel, TraceDist
    from cadet.network.shaping import UnrepresentableDelay, shaping_from_path

    trace = NetworkModel(mode="netem", distribution=TraceDist(path="/nonexistent.txt"))
    with pytest.raises(UnrepresentableDelay) as err:
        shaping_from_path([trace, _netem(10.0)])
    assert "mode: delay" in str(err.value)


def _netem(ms, **extra):
    from cadet.config.schema import NetworkModel

    return NetworkModel(
        mode="netem", distribution={"kind": "constant", "value_ms": ms}, **extra
    )


# --- a relayed route's legs ----------------------------------------------
def test_a_relayed_route_is_shaped_hop_by_hop():
    # Each hop of a relayed route is a real connection between two processes,
    # so netem shapes each on its own sender rather than composing them.
    from cadet.config import build_config
    from cadet.network.netem import rules_for_node

    hops = [
        {"name": "access", "from": "ego", "to": "rsu", "direction": "forward",
         "network": {"mode": "netem", "distribution": {"kind": "constant", "value_ms": 20.0}}},
        {"name": "backhaul", "from": "rsu", "to": "follower", "direction": "forward",
         "network": {"mode": "netem", "distribution": {"kind": "constant", "value_ms": 40.0}}},
    ]
    config = build_config({
        "experiment": {"name": "routes", "app": "reference", "allow_experimental": True},
        "source": {"kind": "synthetic", "count": 1},
        "nodes": NODES,
        "links": hops,
        "routes": [dict(ROUTE[0], realise="relay")],
        "app": {"plugin": "reference"},
    })
    assert [r.link_name for r in rules_for_node(config, "ego")] == ["access"]
    assert [r.link_name for r in rules_for_node(config, "rsu")] == ["backhaul"]
    assert rules_for_node(config, "follower") == []
