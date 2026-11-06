"""Each inbound connection is shaped by its own peer's path.

A node can be the far end of several paths at once, and before this the server
resolved one path and handed the same hops to every connection: a cell whose
uplinks differed had that difference configured and then ignored.
"""

import pytest

from cadet.config import build_config
from cadet.config.routes import paths_into_node
from cadet.roles.server import PeerModels

BASE = {
    "experiment": {"name": "peers", "app": "reference", "repetitions": 1, "seed": 7},
    "source": {"kind": "synthetic", "count": 2},
    "app": {"plugin": "reference", "params": {"model_kind": "stub"}},
}


def _delay(ms):
    return {"mode": "delay", "distribution": {"kind": "constant", "value_ms": ms}}


# Two sensors on different hosts, so their connections can be told apart, plus
# one gateway they both reach.
NODES = [
    {"name": "near", "host": "127.0.0.1", "role": "client"},
    {"name": "far", "host": "127.0.0.2", "role": "client"},
    {"name": "gateway", "host": "127.0.0.1", "port": 15210, "role": "server"},
]
LINKS = [
    {"name": "near_up", "from": "near", "to": "gateway", "direction": "forward",
     "network": _delay(10.0)},
    {"name": "far_up", "from": "far", "to": "gateway", "direction": "forward",
     "network": _delay(80.0)},
]


def _config(nodes=None, links=None, routes=None):
    data = dict(BASE)
    data["nodes"] = nodes or NODES
    data["links"] = links or LINKS
    data["routes"] = routes or []
    data["channels"] = []
    return build_config(data)


def test_every_path_into_a_node_is_found():
    config = _config()
    assert sorted(peer for peer, _p in paths_into_node(config, "gateway")) == ["far", "near"]
    assert paths_into_node(config, "near") == []


def test_each_connection_gets_its_own_peer_s_delay():
    models = PeerModels(_config(), "gateway", seed=7)
    near_recv, _ = models(0, "127.0.0.1")
    far_recv, _ = models(1, "127.0.0.2")
    assert near_recv.sample_delay_s() == pytest.approx(0.010)
    assert far_recv.sample_delay_s() == pytest.approx(0.080)


def test_an_unknown_address_falls_back_to_the_first_path():
    models = PeerModels(_config(), "gateway", seed=7)
    recv, _ = models(0, "10.9.9.9")
    # The first path declared into the node, which is what a single-path server
    # has always used.
    assert recv.sample_delay_s() == pytest.approx(0.010)


def test_peers_sharing_an_address_are_reported_not_guessed():
    nodes = [dict(n) for n in NODES]
    nodes[1] = {"name": "far", "host": "127.0.0.1", "role": "client"}
    lines = []
    models = PeerModels(_config(nodes=nodes), "gateway", seed=7, log=lines.append)
    assert any("share an address" in line for line in lines), lines
    # Both connections then use one path, rather than being told apart wrongly.
    first, _ = models(0, "127.0.0.1")
    second, _ = models(1, "127.0.0.1")
    assert first.sample_delay_s() == second.sample_delay_s()


def test_the_fan_in_adapter_takes_a_source_id():
    models = PeerModels(_config(), "gateway", seed=7)
    recv, _ = models.for_source("127.0.0.2:54321")
    assert recv.sample_delay_s() == pytest.approx(0.080)


def test_a_server_with_one_path_is_unchanged():
    config = _config(nodes=NODES[:1] + NODES[2:], links=LINKS[:1])
    models = PeerModels(config, "gateway", seed=7)
    for peer_ip in ("127.0.0.1", None, "10.0.0.1"):
        recv, _ = models(0, peer_ip)
        assert recv.sample_delay_s() == pytest.approx(0.010)


def test_a_composed_route_is_the_peer_s_path():
    nodes = NODES + [{"name": "cloud", "host": "127.0.0.3", "port": 15211, "role": "server"}]
    links = LINKS + [
        {"name": "backhaul", "from": "gateway", "to": "cloud", "direction": "forward",
         "network": _delay(5.0)}
    ]
    routes = [
        {"name": "near_to_cloud", "from": "near", "to": "cloud",
         "hops": ["near_up", "backhaul"], "realise": "compose"}
    ]
    config = _config(nodes=nodes, links=links, routes=routes)
    models = PeerModels(config, "cloud", seed=7)
    recv, _ = models(0, "127.0.0.1")
    # The route composes: 10 ms access plus 5 ms backhaul.
    assert recv.sample_delay_s() == pytest.approx(0.015)
