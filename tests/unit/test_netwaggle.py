"""Unit tests for the netwaggle backend's config-to-topology generator.

What this covers:
    The translation rules that decide what crosses the emulated network: which
    nodes become emulated hosts, how a link's direction places the qdiscs, which
    distributions netem can express, the wireless validation, and the addressing
    that keeps shaped and unshaped traffic apart.

These tests never import Mininet. The generator and topology modules are pure
data, so the whole mapping is testable without root or the emulator installed —
which is the reason they are separated from :mod:`cadet.network.netwaggle.emulator`.
"""

import pytest

from cadet.config import build_config
from cadet.network.netwaggle import (
    UnsupportedLink,
    emulated_node_names,
    generate,
    shaping_for,
)

BASE = {
    "experiment": {"name": "nw", "app": "reference", "repetitions": 1},
    "source": {"kind": "synthetic", "count": 2},
    "app": {"plugin": "reference", "params": {"model_kind": "stub"}},
}


def make_config(nodes, links, channels=None):
    """Build a validated Config from node/link/channel fragments."""
    data = dict(BASE)
    data["nodes"] = nodes
    data["links"] = links
    data["channels"] = channels or []
    return build_config(data)


def netwaggle_link(name="l0", frm="client", to="server", direction="forward", **network):
    net = {"mode": "netwaggle"}
    net.update(network)
    return {"name": name, "from": frm, "to": to, "direction": direction, "network": net}


TWO_NODES = [
    {"name": "client", "host": "localhost", "role": "client"},
    {"name": "server", "host": "127.0.0.1", "port": 15010, "role": "server"},
]


# --- which nodes enter the topology --------------------------------------
def test_link_endpoints_become_hosts():
    cfg = make_config(TWO_NODES, [netwaggle_link()])
    topo = generate(cfg)
    assert sorted(h.name for h in topo.hosts) == ["client", "server"]


def test_channel_only_node_stays_outside_the_topology():
    """A node reachable solely by a channel must have no emulated path at all."""
    nodes = TWO_NODES + [
        {"name": "sim", "host": "127.0.0.1", "port": 2000, "role": "sim"},
    ]
    channels = [{"name": "frames", "from": "sim", "to": "client"}]
    cfg = make_config(nodes, [netwaggle_link()], channels)
    topo = generate(cfg)
    assert "sim" not in [h.name for h in topo.hosts]
    assert emulated_node_names(cfg) == ["client", "server"]


def test_links_in_other_modes_are_ignored():
    links = [
        netwaggle_link(name="emulated"),
        {
            "name": "inproc",
            "from": "client",
            "to": "server",
            "network": {"mode": "delay", "distribution": {"kind": "constant", "value_ms": 5}},
        },
    ]
    cfg = make_config(TWO_NODES, links)
    topo = generate(cfg)
    assert [lk.name for lk in topo.links] == ["emulated"]


def test_no_netwaggle_links_yields_an_empty_topology():
    links = [{"name": "inproc", "from": "client", "to": "server", "network": {"mode": "delay"}}]
    cfg = make_config(TWO_NODES, links)
    topo = generate(cfg)
    assert topo.hosts == [] and topo.links == []


# --- direction places the qdiscs -----------------------------------------
def test_forward_direction_shapes_only_the_request_path():
    cfg = make_config(
        TWO_NODES,
        [netwaggle_link(direction="forward", distribution={"kind": "constant", "value_ms": 60.0})],
    )
    link = generate(cfg).links[0]
    assert link.forward.delay_ms == 60.0
    assert link.reverse.is_empty


def test_reverse_direction_shapes_only_the_reply_path():
    cfg = make_config(
        TWO_NODES,
        [netwaggle_link(direction="reverse", distribution={"kind": "constant", "value_ms": 60.0})],
    )
    link = generate(cfg).links[0]
    assert link.forward.is_empty
    assert link.reverse.delay_ms == 60.0


def test_both_directions_get_independent_shaping_objects():
    cfg = make_config(
        TWO_NODES,
        [netwaggle_link(direction="both", distribution={"kind": "constant", "value_ms": 30.0})],
    )
    link = generate(cfg).links[0]
    assert link.forward.delay_ms == 30.0
    assert link.reverse.delay_ms == 30.0
    assert link.forward is not link.reverse


# --- distributions netem can and cannot express --------------------------
def test_normal_distribution_is_matched_on_p50_and_p99():
    # netem's jitter is a uniform half-width, so the std is converted through
    # the distribution's p99 rather than copied across.
    cfg = make_config(
        TWO_NODES,
        [
            netwaggle_link(
                distribution={"kind": "normal", "mean_ms": 50.0, "std_ms": 8.0},
                bandwidth_mbps=50.0,
            )
        ],
    )
    link = generate(cfg).links[0]
    assert link.forward.delay_ms == 50.0
    p50, p99 = link.forward.achieved_percentiles_ms
    assert p50 == pytest.approx(50.0)
    assert p99 == pytest.approx(50.0 + 2.3263478740408408 * 8.0, rel=1e-6)


def test_jitter_without_bandwidth_is_refused_as_reordering():
    # netem reorders a jittered flow unless its output is rate-serialised.
    cfg = make_config(
        TWO_NODES, [netwaggle_link(distribution={"kind": "normal", "mean_ms": 50.0, "std_ms": 8.0})]
    )
    with pytest.raises(UnsupportedLink) as err:
        generate(cfg)
    assert "bandwidth_mbps" in str(err.value)


def test_spread_wider_than_the_delay_is_refused():
    # p99 > 2*p50 would need a negative low side, which netem clamps to zero.
    cfg = make_config(
        TWO_NODES,
        [
            netwaggle_link(
                distribution={"kind": "lognormal", "p50_ms": 30.0, "p99_ms": 100.0},
                bandwidth_mbps=50.0,
            )
        ],
    )
    with pytest.raises(UnsupportedLink) as err:
        generate(cfg)
    assert "wider than the delay" in str(err.value)


def test_jittered_link_emits_a_rate_so_it_keeps_order():
    cfg = make_config(
        TWO_NODES,
        [
            netwaggle_link(
                distribution={"kind": "normal", "mean_ms": 50.0, "std_ms": 8.0},
                bandwidth_mbps=50.0,
            )
        ],
    )
    forward = generate(cfg).links[0].forward
    assert forward.preserves_order
    assert "rate" in forward.netem_args()


@pytest.mark.parametrize(
    "dist",
    [
        {"kind": "trace", "path": "/nonexistent/trace.txt"},
    ],
)
def test_unexpressible_distributions_are_refused_not_approximated(dist):
    cfg = make_config(TWO_NODES, [netwaggle_link(distribution=dist)])
    with pytest.raises(UnsupportedLink) as err:
        generate(cfg)
    assert "mode: delay" in str(err.value)


def test_loss_fraction_becomes_a_percentage():
    cfg = make_config(TWO_NODES, [netwaggle_link(loss=0.01)])
    assert generate(cfg).links[0].forward.loss_pct == pytest.approx(1.0)


def test_unset_parameters_install_no_qdisc():
    """Zero is not the same as unset: netem with an explicit 0 ms still queues."""
    cfg = make_config(TWO_NODES, [netwaggle_link()])
    shaping = generate(cfg).links[0].forward
    assert shaping.is_empty
    assert shaping.as_mininet_params() == {}


def test_mininet_params_carry_units():
    cfg = make_config(
        TWO_NODES,
        [
            netwaggle_link(
                distribution={"kind": "normal", "mean_ms": 50.0, "std_ms": 8.0},
                loss=0.01,
                bandwidth_mbps=50.0,
            )
        ],
    )
    params = generate(cfg).links[0].forward.as_mininet_params()
    assert params["delay"] == "50.0ms"
    assert params["jitter"].endswith("ms")
    assert params["loss"] == 1.0
    assert params["bw"] == 50.0


# --- addressing keeps shaped and unshaped traffic apart ------------------
def test_shaped_and_bypass_subnets_are_distinct():
    cfg = make_config(TWO_NODES, [netwaggle_link()])
    topo = generate(cfg)
    for host in topo.hosts:
        assert host.link_ip.startswith("10.60.0.")
        assert host.bypass_ip.startswith("10.61.0.")
    assert topo.address_map()["server"]["link_ip"] == "10.60.0.2"


def test_aliases_are_short_enough_for_mininet():
    cfg = make_config(TWO_NODES, [netwaggle_link()])
    for host in generate(cfg).hosts:
        assert len(host.alias) <= 5


# --- wireless -------------------------------------------------------------
def wifi_network(positions=("45,50,0", "50,50,0"), **extra):
    wifi = {
        "nodes": {
            "client": {"role": "station", "position": positions[0]},
            "server": {"role": "ap", "position": positions[1]},
        },
    }
    wifi.update(extra)
    return {"medium": "wifi", "wifi": wifi}


def test_wireless_link_carries_roles_and_propagation_model():
    cfg = make_config(
        TWO_NODES,
        [netwaggle_link(**wifi_network(propagation_model="logDistance", propagation_exp=3.5))],
    )
    topo = generate(cfg)
    link = topo.links[0]
    assert link.medium == "wifi"
    assert link.wifi.ap_name() == "server"
    assert link.wifi.role_of("client") == "station"
    assert link.wifi.propagation_exp == 3.5
    assert topo.has_wireless()
    assert topo.host("server").wifi_role == "ap"


def test_wireless_requires_a_role_for_every_endpoint():
    net = {"medium": "wifi", "wifi": {"nodes": {"server": {"role": "ap"}}}}
    cfg = make_config(TWO_NODES, [netwaggle_link(**net)])
    with pytest.raises(UnsupportedLink) as err:
        generate(cfg)
    assert "both endpoints" in str(err.value)


def test_wifi_block_without_wifi_medium_is_a_config_error():
    """Schema-level: a wifi block only makes sense with medium: wifi."""
    bad = {
        "name": "l0",
        "from": "client",
        "to": "server",
        "network": {"mode": "netwaggle", "wifi": {"nodes": {"server": {"role": "ap"}}}},
    }
    with pytest.raises(Exception) as err:
        make_config(TWO_NODES, [bad])
    assert "medium" in str(err.value)


def test_wifi_medium_needs_exactly_one_access_point():
    net = {
        "medium": "wifi",
        "wifi": {"nodes": {"client": {"role": "station"}, "server": {"role": "station"}}},
    }
    with pytest.raises(Exception) as err:
        make_config(TWO_NODES, [netwaggle_link(**net)])
    assert "exactly one" in str(err.value)


def test_wifi_medium_requires_the_netwaggle_backend():
    bad = {
        "name": "l0",
        "from": "client",
        "to": "server",
        "network": {
            "mode": "delay",
            "medium": "wifi",
            "wifi": {"nodes": {"server": {"role": "ap"}, "client": {"role": "station"}}},
        },
    }
    with pytest.raises(Exception) as err:
        make_config(TWO_NODES, [bad])
    assert "netwaggle" in str(err.value)


# --- a composed path across emulated hops --------------------------------
# netwaggle does not collapse a composed route into one qdisc the way netem
# must: each hop is a real interface, so the packet is carried across both and
# the kernel composes them. What the topology has to carry is the routing that
# makes the flow cross them at all.
PATH_NODES = [
    {"name": "client", "host": "10.0.0.5", "role": "client"},
    {"name": "edge", "host": "10.0.0.7", "role": "waypoint"},
    {"name": "server", "host": "10.0.0.9", "port": 15010, "role": "server"},
]


def _hop(name, src, dst, ms):
    return {
        "name": name,
        "from": src,
        "to": dst,
        "direction": "forward",
        "network": {
            "mode": "netwaggle",
            "distribution": {"kind": "constant", "value_ms": ms},
        },
    }


PATH_HOPS = [_hop("access", "client", "edge", 20.0), _hop("backhaul", "edge", "server", 40.0)]
PATH_ROUTE = {
    "name": "path",
    "from": "client",
    "to": "server",
    "hops": ["access", "backhaul"],
    "realise": "compose",
}


def _path_topology(realise="compose"):
    from cadet.config import build_config
    from cadet.network.netwaggle.generator import generate

    nodes = PATH_NODES
    if realise == "relay":
        # A relay listens, so the waypoint is a server with a port of its own.
        nodes = [dict(n) for n in PATH_NODES]
        nodes[1] = {"name": "edge", "host": "10.0.0.7", "port": 15020, "role": "server"}
    config = build_config(
        {
            "experiment": {
                "name": "nw_route",
                "app": "reference",
                "repetitions": 1,
                "allow_experimental": True,
            },
            "source": {"kind": "synthetic", "count": 2},
            "app": {"plugin": "reference", "params": {"model_kind": "stub"}},
            "nodes": nodes,
            "links": PATH_HOPS,
            "routes": [dict(PATH_ROUTE, realise=realise)],
            "channels": [],
        }
    )
    return generate(config)


def test_a_composed_route_becomes_a_path_with_waypoints():
    topo = _path_topology()
    assert [h.name for h in topo.hosts] == ["client", "edge", "server"]
    assert len(topo.paths) == 1
    path = topo.paths[0]
    assert path.hop_links == ["access", "backhaul"]
    assert path.waypoints == ["edge"]
    # The waypoint is the one host that has to forward.
    assert topo.forwarders() == ["edge"]


def test_each_hop_keeps_its_own_shaping():
    # The hops are not collapsed: 20 ms and 40 ms stay on their own interfaces.
    topo = _path_topology()
    delays = {link.name: link.forward.delay_ms for link in topo.links}
    assert delays == {"access": 20.0, "backhaul": 40.0}


def test_a_relayed_route_needs_no_transit_routing():
    # Every packet of a relayed hop is addressed to the next process, not
    # through it, so there is nothing to forward.
    topo = _path_topology(realise="relay")
    assert topo.paths == []
    assert topo.forwarders() == []


# --- which interface faces what, wired and wireless -----------------------
class _FakeNode:
    def __init__(self, name):
        self.name = name


class _WiredIntf:
    """A veth end: its link's endpoints are interface OBJECTS."""

    def __init__(self, name, peer_node_name):
        self.name = name
        peer = _WiredIntf.__new__(_WiredIntf)
        peer.name = "{}-peer".format(name)
        peer.node = _FakeNode(peer_node_name)
        peer.link = None
        self.node = _FakeNode("self")

        class _Link:
            intf1 = self
            intf2 = peer

        self.link = _Link()


class _WirelessIntf:
    """A station's radio: its link's endpoints are interface NAMES (strings).

    This is the shape that broke the Wi-Fi path - a station associating with an
    access point is not a veth pair, so there is no peer interface object and
    reading `.node` off a string raised AttributeError.
    """

    def __init__(self, name):
        self.name = name
        self.node = _FakeNode("self")

        class _Link:
            intf1 = name
            intf2 = "ap-wlan0"

        self.link = _Link()


class _LoopbackIntf:
    name = "lo"
    link = None


def _emulator_for_intf_tests():
    """An emulator instance with nothing built, for the two pure lookups."""
    from cadet.network.netwaggle.emulator import Emulator

    return Emulator.__new__(Emulator)


def test_a_wireless_interface_does_not_crash_the_bypass_lookup():
    """The bug: `'str' object has no attribute 'node'`, which took every Wi-Fi
    run down before addressing while wired runs passed."""
    from cadet.network.netwaggle.emulator import BYPASS_SWITCH

    emulator = _emulator_for_intf_tests()
    node = type("N", (), {"intfList": lambda self: [_WirelessIntf("sta1-wlan0")]})()
    assert emulator._bypass_intf(node) is None


def test_the_bypass_interface_is_still_found_beside_a_wireless_one():
    """A station has both: a radio, and a wired veth to the bypass switch."""
    from cadet.network.netwaggle.emulator import BYPASS_SWITCH

    emulator = _emulator_for_intf_tests()
    intfs = [
        _LoopbackIntf(),
        _WirelessIntf("sta1-wlan0"),
        _WiredIntf("sta1-eth1", BYPASS_SWITCH),
    ]
    node = type("N", (), {"intfList": lambda self: intfs})()
    assert emulator._bypass_intf(node) == "sta1-eth1"


def test_the_peer_lookup_tolerates_a_string_endpoint():
    emulator = _emulator_for_intf_tests()
    assert emulator._link_peer_node(_WirelessIntf("sta1-wlan0")) is None
    assert emulator._link_peer_node(_WiredIntf("n1-eth1", "bps0")) == "bps0"
    assert emulator._link_peer_node(_LoopbackIntf()) is None


def _wifi_topology():
    """The example's own cell, generated from its committed config.

    The real file rather than a hand-built fixture: what these tests are about
    is how a cell is built, and the config that expresses one is the one in the
    repository.
    """
    import os

    from conftest import REPO_ROOT

    from cadet.config.loader import load_run_configs
    from cadet.network.netwaggle.generator import generate

    runs, _base = load_run_configs(
        os.path.join(str(REPO_ROOT), "examples/distributed_sensing/wifi_5.yaml"), []
    )
    return generate(runs[0])


class _RecordingNet:
    """A stand-in for Mininet that records what it was asked to add."""

    def __init__(self):
        self.calls = {"stations": [], "hosts": [], "switches": []}

    def addAccessPoint(self, alias, **kw):
        self.calls["ap"] = (alias, kw)
        return "ap-node"

    def addSwitch(self, alias, **kw):
        self.calls["switches"].append((alias, kw))
        return "switch-node"

    def addStation(self, alias, **kw):
        self.calls["stations"].append((alias, kw))
        return "sta-node"

    def addHost(self, alias, **kw):
        self.calls["hosts"].append(alias)
        return "host-node"


def test_the_access_point_is_infrastructure_and_holds_no_address():
    """This replaces the opposite pin, and the reason is worth recording.

    An `ip=` used to be passed to `addAccessPoint` because a wireless run died
    the moment the server tried to bind - `OSError: [Errno 99] Cannot assign
    requested address` - when the cell's gateway was made to *be* the access
    point. That could not be fixed by handing it an address: an access point is
    a switch in the root namespace, and a role process cannot live in one.

    So the cell's access point is a connector, which runs nothing and holds no
    address, and the node that accepts connections is an ordinary station that
    gets its address like every other station. The parameter is dead and gone.
    """
    from cadet.network.netwaggle.emulator import Emulator

    net = _RecordingNet()
    emulator = Emulator.__new__(Emulator)
    emulator._nodes = {}
    emulator._connectors = {}
    emulator.topology = _wifi_topology()
    emulator._add_connectors(net)
    emulator._add_hosts(net, wireless=True)

    ap_alias, ap_kw = net.calls["ap"]
    assert ap_alias == emulator.topology.connector("cell_ap").alias
    assert "ip" not in ap_kw, "an access point has no address to hold"
    assert ap_kw["ssid"] == "cadet"
    assert ap_kw["position"] == "0,0,0"

    # Every member of the cell is a station, and each gets the shared subnet's
    # own prefix, because they are on-link for one another.
    assert net.calls["hosts"] == []
    assert len(net.calls["stations"]) == len(emulator.topology.hosts)
    for alias, kw in net.calls["stations"]:
        host = next(h for h in emulator.topology.hosts if h.alias == alias)
        assert kw["ip"] == emulator.topology.cell_cidr(host.name)
        assert kw["position"] == host.position


class _CommandNode:
    """A Mininet host stand-in that records the commands run inside it."""

    def __init__(self, alias):
        from cadet.network.netwaggle.emulator import BYPASS_SWITCH

        self.name = alias
        self.commands = []
        self.silently_drop_adds = False
        self._addresses = {}
        self._intfs = [
            _LoopbackIntf(),
            _WirelessIntf("{}-wlan0".format(alias)),
            _WiredIntf("{}-eth0".format(alias), BYPASS_SWITCH),
        ]

    def intfList(self):
        return self._intfs

    def cmd(self, command):
        """Answer as the kernel does, so a read-back in the emulator means something.

        Only the three address commands are modelled, because they are the ones
        the emulator reads back. ``silently_drop_adds`` is the failure this
        stand-in exists to be able to stage: an ``ip addr add`` that reports
        nothing and leaves the interface without an address.
        """
        self.commands.append(command)
        if command.startswith("ip addr add "):
            fields = command.split()
            if not self.silently_drop_adds:
                self._addresses.setdefault(fields[-1], []).append(fields[3])
            return ""
        if command.startswith("ip addr flush dev "):
            self._addresses.pop(command.split()[-1], None)
            return ""
        if command.startswith("ip -4 -o addr show dev "):
            intf = command.split()[-1]
            return "".join(
                "2: {} inet {} scope global {}\n".format(intf, address, intf)
                for address in self._addresses.get(intf, [])
            )
        return ""


def test_a_cell_member_is_addressed_on_its_radio_and_given_no_route():
    """The shared-subnet property, as the commands the emulator actually runs.

    Each station gets the shaped subnet's own prefix on its radio, so the other
    stations are on-link; and nothing installs a route between them, because
    there is no hop to route over. The only route written is the bypass default,
    which is the plumbing path and not the cell.
    """
    from cadet.network.netwaggle.emulator import Emulator

    topology = _wifi_topology()
    emulator = Emulator.__new__(Emulator)
    emulator.topology = topology
    emulator._connectors = {"cell_ap": "ap-node"}
    emulator._nodes = {h.name: _CommandNode(h.alias) for h in topology.hosts}
    emulator._configure_addressing()

    for host in topology.hosts:
        node = emulator._nodes[host.name]
        radio = "{}-wlan0".format(host.alias)
        assert "ip addr add {} dev {}".format(topology.cell_cidr(host.name), radio) in (
            node.commands
        )
        routes = [c for c in node.commands if c.startswith("ip route")]
        assert routes == [
            "ip route replace default via {} dev {}-eth0".format(
                topology.bypass_gateway, host.alias
            )
        ]


# --- connectors: a shared medium nodes attach to --------------------------
# A connector is the other shape a medium can have. These tests fix what makes
# it different from two point-to-point links in series - one subnet, no route
# between its members, and nothing running on the connector itself - and every
# way the config can confuse a connector with a node.
AP_CONNECTOR = {"name": "cell", "type": "ap", "position": "0,0,0"}
SWITCH_CONNECTOR = {"name": "seg", "type": "switch"}


def cell_node(name, connector="cell", position="5,0,0", **extra):
    node = {"name": name, "host": "localhost", "role": "client"}
    if connector is not None:
        node["connector"] = connector
    if position is not None:
        node["position"] = position
    node.update(extra)
    return node


def make_cell_config(nodes, connectors, links=None):
    """Build a validated Config with a ``connectors`` section."""
    data = dict(BASE)
    data["nodes"] = nodes
    data["connectors"] = connectors
    data["links"] = links or []
    data["channels"] = []
    return build_config(data)


CELL_NODES = [
    cell_node("client"),
    cell_node("server", position="0,1,0", host="127.0.0.1", port=15010, role="server"),
]


def test_a_cell_needs_no_links_at_all():
    """Attaching to a connector is itself the statement that a node is emulated."""
    cfg = make_cell_config(CELL_NODES, [AP_CONNECTOR])
    topo = generate(cfg)
    assert topo.links == []
    assert sorted(h.name for h in topo.hosts) == ["client", "server"]
    assert topo.has_wireless()


def test_a_connector_is_not_a_host_and_runs_nothing():
    cfg = make_cell_config(CELL_NODES, [AP_CONNECTOR])
    topo = generate(cfg)
    assert "cell" not in [h.name for h in topo.hosts]
    assert [c.name for c in topo.connectors] == ["cell"]
    assert topo.connector("cell").kind == "ap"


def test_every_node_on_an_access_point_is_a_station():
    """Nothing is the access point: that is the connector, which is not a node."""
    cfg = make_cell_config(CELL_NODES, [AP_CONNECTOR])
    topo = generate(cfg)
    assert {h.wifi_role for h in topo.hosts} == {"station"}
    assert [h.position for h in topo.hosts] == ["5,0,0", "0,1,0"]


def test_cell_members_share_a_subnet_and_get_no_route_to_each_other():
    """The property that makes a cell a cell rather than two links in series."""
    cfg = make_cell_config(CELL_NODES, [AP_CONNECTOR])
    topo = generate(cfg)
    assert topo.shares_subnet("client", "server")
    # The shaped subnet's own prefix, so each member is on-link for the others.
    assert topo.cell_cidr("client") == "10.60.0.1/24"
    assert topo.cell_cidr("server") == "10.60.0.2/24"
    # And no link between them, so there is no hop and nothing to route.
    assert topo.links == []


def test_a_switch_connector_is_the_same_code_without_radios():
    cfg = make_cell_config(
        [cell_node("client", connector="seg", position=None), cell_node(
            "server", connector="seg", position=None, host="127.0.0.1", port=15010, role="server"
        )],
        [SWITCH_CONNECTOR],
    )
    topo = generate(cfg)
    assert topo.connector("seg").kind == "switch"
    assert topo.connector("seg").medium == "wired"
    assert topo.connector("seg").wifi is None
    assert not topo.has_wireless()
    # Still one shared subnet with no route between its members.
    assert {h.wifi_role for h in topo.hosts} == {None}
    assert topo.shares_subnet("client", "server")
    assert topo.cell_cidr("client").endswith("/24")


def test_a_switch_connector_attaches_its_members_with_a_plain_veth():
    """The build path: a wired member gets a link to the switch, a station none.

    A station needs no link because its association is made by its radio, which
    is what the original script does too; a switch member gets a veth with no TC
    parameters, because a connector is shaped by its medium and not by a
    per-member qdisc.
    """
    from cadet.network.netwaggle.emulator import Emulator

    cfg = make_cell_config(
        [
            cell_node("client", connector="seg", position=None),
            cell_node(
                "server", connector="seg", position=None, host="127.0.0.1",
                port=15010, role="server",
            ),
        ],
        [SWITCH_CONNECTOR],
    )
    topo = generate(cfg)

    added = []

    class _Net(_RecordingNet):
        def addLink(self, a, b, **kw):
            added.append((a, b, kw))

    net = _Net()
    emulator = Emulator.__new__(Emulator)
    emulator.topology = topo
    emulator._nodes = {}
    emulator._connectors = {}
    emulator._add_connectors(net)
    emulator._add_hosts(net, wireless=False)
    emulator._add_links(net, tclink=None, wireless=False, wired="wired-cls")

    assert net.calls["switches"] == [(topo.connector("seg").alias, {"failMode": "standalone"})]
    assert net.calls["stations"] == []
    assert sorted(net.calls["hosts"]) == [h.alias for h in topo.hosts]
    # One veth per member, straight to the switch, with no shaping parameters.
    assert len(added) == len(topo.hosts)
    # The class is named rather than defaulted: a wireless emulation's default
    # link class is the radio medium, which would make a member's wired side a
    # second radio.
    assert all(b == "switch-node" and kw == {"cls": "wired-cls"} for _a, b, kw in added)


def test_connector_aliases_continue_the_host_numbering():
    """So the recovery sweep, which removes objects by that shape, covers them."""
    cfg = make_cell_config(CELL_NODES, [AP_CONNECTOR])
    topo = generate(cfg)
    assert [h.alias for h in topo.hosts] == ["n0", "n1"]
    assert [c.alias for c in topo.connectors] == ["n2"]
    assert all(len(c.alias) <= 5 for c in topo.connectors)


# --- validate-time refusals ----------------------------------------------
def test_attaching_to_an_unknown_connector_is_refused():
    with pytest.raises(Exception) as err:
        make_cell_config([cell_node("client", connector="nope")] + CELL_NODES[1:], [AP_CONNECTOR])
    assert "unknown connector 'nope'" in str(err.value)


def test_a_connector_named_like_a_node_is_refused():
    with pytest.raises(Exception) as err:
        make_cell_config(CELL_NODES, [dict(AP_CONNECTOR, name="client")])
    assert "same name as a node" in str(err.value)


def test_a_duplicate_connector_name_is_refused():
    with pytest.raises(Exception) as err:
        make_cell_config(CELL_NODES, [AP_CONNECTOR, dict(AP_CONNECTOR)])
    assert "duplicate connector name 'cell'" in str(err.value)


def test_two_access_points_in_one_cell_are_refused():
    """A station that is also a link's access point would give the cell two."""
    link = netwaggle_link(
        frm="client",
        to="server",
        medium="wifi",
        wifi={
            "nodes": {
                "client": {"role": "ap", "position": "5,0,0"},
                "server": {"role": "station", "position": "0,1,0"},
            }
        },
    )
    with pytest.raises(Exception) as err:
        make_cell_config(CELL_NODES, [AP_CONNECTOR], [link])
    assert "a cell has one access point" in str(err.value)


def test_a_station_declaring_a_second_radio_is_refused():
    link = netwaggle_link(
        frm="client",
        to="server",
        medium="wifi",
        wifi={
            "nodes": {
                "server": {"role": "ap", "position": "0,1,0"},
                "client": {"role": "station", "position": "5,0,0"},
            }
        },
    )
    with pytest.raises(Exception) as err:
        make_cell_config(CELL_NODES, [AP_CONNECTOR], [link])
    assert "a node has one radio" in str(err.value)


def test_a_station_with_no_connector_is_refused():
    """A position is radio-only state, so it cannot stand on its own."""
    with pytest.raises(Exception) as err:
        make_cell_config(
            [cell_node("client", connector=None)] + CELL_NODES[1:], [AP_CONNECTOR]
        )
    assert "attaches to no connector" in str(err.value)


def test_a_station_with_no_position_is_refused():
    with pytest.raises(Exception) as err:
        make_cell_config([cell_node("client", position=None)] + CELL_NODES[1:], [AP_CONNECTOR])
    assert "must declare a position" in str(err.value)


def test_a_position_on_a_wired_attachment_is_refused():
    with pytest.raises(Exception) as err:
        make_cell_config(
            [cell_node("client", connector="seg", position="5,0,0")]
            + [cell_node("server", connector="seg", position=None, host="127.0.0.1",
                         port=15010, role="server")],
            [SWITCH_CONNECTOR],
        )
    assert "has no propagation model to read it" in str(err.value)


def test_a_switch_with_radios_or_a_position_is_refused():
    with pytest.raises(Exception) as err:
        make_cell_config(CELL_NODES, [dict(SWITCH_CONNECTOR, wifi={"ssid": "x"})])
    assert "has no radios" in str(err.value)
    with pytest.raises(Exception) as err:
        make_cell_config(CELL_NODES, [dict(SWITCH_CONNECTOR, position="0,0,0")])
    assert "has no radios" in str(err.value)


def test_an_access_point_without_a_position_is_refused():
    with pytest.raises(Exception) as err:
        make_cell_config(CELL_NODES, [{"name": "cell", "type": "ap"}])
    assert "must declare a position" in str(err.value)


def test_a_config_with_no_connector_generates_what_it_always_did():
    """Pinned: the point-to-point path must be untouched by the connector model."""
    cfg = make_cell_config(TWO_NODES, [], [netwaggle_link()])
    topo = generate(cfg)
    assert topo.connectors == []
    assert [h.name for h in topo.hosts] == ["client", "server"]
    assert [h.alias for h in topo.hosts] == ["n0", "n1"]
    assert [h.connector for h in topo.hosts] == [None, None]
    assert [h.position for h in topo.hosts] == [None, None]
    assert [h.link_ip for h in topo.hosts] == ["10.60.0.1", "10.60.0.2"]
    assert not topo.has_wireless()
    assert not topo.shares_subnet("client", "server")
    # A point-to-point endpoint has no cell address: it gets a /32 and a route.
    for host in topo.hosts:
        with pytest.raises(KeyError):
            topo.cell_cidr(host.name)


# --- the plumbing path and its interface ------------------------------------
# Every host holds one interface onto an unshaped bridge carrying the default
# route. These tests fix the two ways a cell can come up looking healthy with
# that interface missing or unaddressed.


def test_a_bypass_address_that_does_not_land_is_an_error():
    """The plumbing failure that otherwise surfaces a long way from its cause.

    A host whose bypass interface carries no address still looks healthy: its
    radio is addressed and a server on it binds. What fails is a later fetch
    over the plumbing, so the emulator reads the address back and says so here.
    """
    from cadet.network.netwaggle.emulator import Emulator, EmulatorError

    topology = _wifi_topology()
    emulator = Emulator.__new__(Emulator)
    emulator.topology = topology
    emulator._connectors = {"cell_ap": "ap-node"}
    emulator._nodes = {h.name: _CommandNode(h.alias) for h in topology.hosts}
    for node in emulator._nodes.values():
        node.silently_drop_adds = True

    with pytest.raises(EmulatorError) as excinfo:
        emulator._configure_addressing()
    assert "did not land" in str(excinfo.value)


def test_a_host_with_no_interface_to_the_bypass_is_an_error():
    """Every host is attached to the bridge, so a missing one is a lookup bug."""
    from cadet.network.netwaggle.emulator import Emulator, EmulatorError

    topology = _wifi_topology()
    emulator = Emulator.__new__(Emulator)
    emulator.topology = topology
    emulator._connectors = {"cell_ap": "ap-node"}
    emulator._nodes = {h.name: _CommandNode(h.alias) for h in topology.hosts}
    for node in emulator._nodes.values():
        node._intfs = [i for i in node._intfs if not i.name.endswith("-eth0")]

    with pytest.raises(EmulatorError) as excinfo:
        emulator._configure_addressing()
    assert "no interface facing" in str(excinfo.value)


def test_the_plumbing_link_names_a_wired_class():
    """The bug that left a station with two radios and no plumbing interface.

    A wireless emulation is constructed with the radio medium as its default
    link class, so ``addLink`` with no class between a station and the plumbing
    bridge built another radio instead of a veth. The host came up holding
    ``n0-wlan0`` and ``n0-wlan1``, no interface facing the bridge, and a radio
    the topology had never asked for.
    """
    from cadet.network.netwaggle.emulator import BYPASS_SWITCH, Emulator

    topo = _wifi_topology()
    added = []

    class _Net(_RecordingNet):
        def addLink(self, a, b, **kw):
            added.append((a, b, kw))

        def addNAT(self, **kw):
            return "nat-node"

    net = _Net()
    emulator = Emulator.__new__(Emulator)
    emulator.topology = topo
    emulator._nodes = {h.name: "node-" + h.name for h in topo.hosts}
    emulator._connectors = {}
    emulator._add_bypass(net, "wired-cls")

    assert (BYPASS_SWITCH, {"failMode": "standalone"}) in net.calls["switches"]
    assert len(added) == len(topo.hosts)
    assert all(kw["cls"] == "wired-cls" for _a, _b, kw in added)
    # Named out of the radio namespace, or the wireless configuration assigns a
    # radio over it.
    assert sorted(kw["intfName1"] for _a, _b, kw in added) == sorted(
        "{}-bps".format(h.alias) for h in topo.hosts
    )
    assert not any("wlan" in kw["intfName1"] for _a, _b, kw in added)


# --- leftover kernel state --------------------------------------------------


def test_both_cleanups_drop_the_hwsim_module(monkeypatch):
    """The module has to go even from a scoped sweep, or it breaks the next run.

    Its radios survive a hung stop, Mininet-WiFi then does not reload it with
    the radio count the next topology needs, and that run dies inside wmediumd's
    txpower update with an error that says nothing about radios. Measured once
    for real: a crashed run left it loaded, `cadet down` reported everything
    cleaned, and the next wireless run failed that way.
    """
    from cadet.network.netwaggle import emulator as emulator_mod

    for scoped in (True, False):
        calls = []

        def _run(argv, check=False, _calls=calls):
            _calls.append(list(argv))
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        monkeypatch.setattr(emulator_mod, "_run", _run)
        monkeypatch.setattr(emulator_mod.shutil, "which", lambda name: None)

        removed = emulator_mod.force_cleanup(scoped=scoped)
        assert ["rmmod", "mac80211_hwsim"] in calls
        assert "module:mac80211_hwsim" in removed


def test_a_scoped_cleanup_still_removes_nothing_by_name(monkeypatch):
    """What scoping is for: the named objects are the recovery sweep's to remove."""
    from cadet.network.netwaggle import emulator as emulator_mod

    calls = []

    def _run(argv, check=False):
        calls.append(list(argv))
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(emulator_mod, "_run", _run)
    monkeypatch.setattr(emulator_mod.shutil, "which", lambda name: "/usr/bin/mn")

    emulator_mod.force_cleanup(scoped=True)
    assert not [argv for argv in calls if argv[0] == "mn"]
    assert not [argv for argv in calls if argv[0] == "ovs-vsctl"]
    assert not [argv for argv in calls if argv[:2] == ["ip", "netns"]]


def test_a_module_in_use_is_not_reported_as_removed(monkeypatch):
    """`rmmod` refuses a module a live emulation beside this one is using."""
    from cadet.network.netwaggle import emulator as emulator_mod

    def _run(argv, check=False):
        busy = argv[0] == "rmmod"
        return type(
            "R",
            (),
            {"returncode": 1 if busy else 0, "stdout": "", "stderr": "Module is in use"},
        )()

    monkeypatch.setattr(emulator_mod, "_run", _run)
    monkeypatch.setattr(emulator_mod.shutil, "which", lambda name: None)

    assert "module:mac80211_hwsim" not in emulator_mod.force_cleanup(scoped=True)


def test_a_wireless_teardown_removes_only_the_objects_it_named(monkeypatch):
    """The host's namespace holds the machine's own bridges, so names are the limit.

    A wireless topology has to be built in the host's network namespace, because
    that is where hwsim creates its radios. The unscoped sweep deletes every
    bridge and namespace it can see, which there are the machine's own, so a
    wireless teardown passes the names the generator assigned and this removes
    those and nothing else.
    """
    from cadet.network.netwaggle import emulator as emulator_mod

    calls = []

    def _run(argv, check=False):
        calls.append(list(argv))
        stdout = ""
        if argv[:2] == ["ovs-vsctl", "list-br"]:
            # The machine's own bridge is listed beside two of the topology's.
            stdout = "bps0\ncell0\nbr-machine\n"
        elif argv[:3] == ["ip", "netns", "list"]:
            stdout = "n0\nsomebody-elses-netns\n"
        return type("R", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

    monkeypatch.setattr(emulator_mod, "_run", _run)
    monkeypatch.setattr(emulator_mod.shutil, "which", lambda name: "/usr/bin/mn")

    removed = emulator_mod.force_cleanup(scoped=True, names=["bps0", "cell0", "n0"])

    assert not [argv for argv in calls if argv[0] == "mn"]
    deleted = [argv[-1] for argv in calls if argv[:2] == ["ovs-vsctl", "--if-exists"]]
    assert deleted == ["bps0", "cell0"]
    assert [argv[-1] for argv in calls if argv[:3] == ["ip", "netns", "delete"]] == ["n0"]
    assert "br-machine" not in " ".join(removed)
    assert "somebody-elses-netns" not in " ".join(removed)


def test_a_wireless_emulator_never_runs_the_unscoped_sweep(monkeypatch):
    """Which path a fallback cleanup takes follows from the topology, not the caller."""
    from cadet.network.netwaggle import emulator as emulator_mod

    seen = {}

    def _record(scoped=False, names=None):
        seen["scoped"] = scoped
        seen["names"] = names
        return []

    monkeypatch.setattr(emulator_mod, "force_cleanup", _record)

    wireless = generate(
        make_config(TWO_NODES, [netwaggle_link(**wifi_network())])
    )
    emulator_mod.Emulator(wireless)._force_cleanup()
    assert seen["scoped"] is True
    assert emulator_mod.BYPASS_SWITCH in seen["names"]
    for host in wireless.hosts:
        assert host.alias in seen["names"]

    wired = generate(make_config(TWO_NODES, [netwaggle_link()]))
    emulator_mod.Emulator(wired)._force_cleanup()
    assert seen["scoped"] is False


def test_an_emulated_runs_record_carries_the_same_provenance(monkeypatch, tmp_path):
    """An emulated run is a run: a reader asks it the same questions.

    `cadet emulate` writes its own record, and that record carried neither the
    commit nor the image it was built from, so a reader of an emulated run
    directory could not tell which code produced it. The acceptance matrix's
    metrics check wanted it of every cell.
    """
    from cadet.network.netwaggle import session
    from cadet.orchestrator import placement as placement_mod

    monkeypatch.setattr(placement_mod, "repo_revision", lambda *_a: ("a" * 40, True))
    assert session._provenance() == {"cadet": {"commit": "a" * 40, "dirty": True}}

    # Built in the emulation image, where there is no git: the commit comes from
    # the image, and the record says so.
    monkeypatch.setattr(placement_mod, "repo_revision", lambda *_a: (None, False))
    monkeypatch.setenv("CADET_IMAGE_REVISION", "b" * 40)
    assert session._provenance() == {
        "cadet": {"commit": "b" * 40, "dirty": False, "from": "image"}
    }

    # Neither available: no field at all, rather than one holding a guess.
    monkeypatch.delenv("CADET_IMAGE_REVISION", raising=False)
    assert session._provenance() == {}


def test_an_emulated_run_that_collected_nothing_writes_no_summary(tmp_path):
    """The roll-up is of collector CSVs, and a run without them has none."""
    from cadet.network.netwaggle import session

    lines = []
    session._write_device_summary(str(tmp_path), lines.append)
    assert not (tmp_path / "device_summary.json").exists()
    assert lines == []


def test_an_emulated_role_is_told_which_stage_modules_to_import(monkeypatch):
    """A role in an emulated host inherits nothing from the process that starts it.

    The stage modules travelled only through the environment, which a local
    child inherits. A role started inside an emulated host is spawned through
    the emulator, so a staged pipeline role reached an empty registry and died
    on the first stage its config named - the same failure a containerised role
    had, for the same reason.
    """
    from cadet.network.netwaggle import session

    monkeypatch.setenv("CADET_STAGE_MODULES", "cadet.pipeline.stages.braking")

    spawned = {}

    class _Emulator(object):
        def popen(self, node_name, argv, stdout=None, stderr=None):
            spawned["node"] = node_name
            spawned["argv"] = list(argv)
            return "handle"

    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "role.log")
        session._spawn(
            _Emulator(), "server", "serve", "cfg.yaml", [], 0, ["examples.app"], log_path
        )

    argv = spawned["argv"]
    assert "--stage-module" in argv
    assert argv[argv.index("--stage-module") + 1] == "cadet.pipeline.stages.braking"

    # Nothing published, nothing added.
    monkeypatch.delenv("CADET_STAGE_MODULES", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        session._spawn(
            _Emulator(), "server", "serve", "cfg.yaml", [], 0, [],
            os.path.join(tmp, "role.log"),
        )
    assert "--stage-module" not in spawned["argv"]
