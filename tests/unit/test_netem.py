"""Unit tests for the netem link backend.

What this covers:
    Which node installs a rule for which direction, which port the filter
    matches (and why the two directions differ), the shape of the generated tc
    tree, that unmatched traffic keeps an unshaped default class, that teardown
    is a single idempotent delete, and that a failed apply rolls back.

No tc is executed: :class:`~cadet.network.netem.NetemBackend` takes a runner, so
these tests pass a fake one and assert on the command lines.
"""

import pytest

from cadet.config import build_config
from cadet.network.netem import (
    DEFAULT_CLASSID,
    NetemBackend,
    ShapeRule,
    build_commands,
    rules_for_node,
    teardown_commands,
)
from cadet.network.shaping import Shaping

BASE = {
    "experiment": {"name": "ne", "app": "reference", "repetitions": 1},
    "source": {"kind": "synthetic", "count": 2},
    "app": {"plugin": "reference", "params": {"model_kind": "stub"}},
}

NODES = [
    {"name": "client", "host": "10.0.0.5", "role": "client"},
    {"name": "server", "host": "10.0.0.9", "port": 15010, "role": "server"},
]


def make_config(links, channels=None, nodes=None):
    data = dict(BASE)
    data["nodes"] = nodes or NODES
    data["links"] = links
    data["channels"] = channels or []
    return build_config(data)


def netem_link(direction="forward", **network):
    net = {"mode": "netem", "distribution": {"kind": "constant", "value_ms": 30.0}}
    net.update(network)
    return {"name": "l0", "from": "client", "to": "server", "direction": direction, "network": net}


# --- which side installs the rule ----------------------------------------
def test_forward_rule_sits_on_the_sender_and_matches_the_peer_port():
    cfg = make_config([netem_link("forward")])
    rules = rules_for_node(cfg, "client")
    assert len(rules) == 1
    assert rules[0].peer_ip == "10.0.0.9"
    assert rules[0].port == 15010
    assert rules[0].match == "dport"
    assert rules_for_node(cfg, "server") == []


def test_reverse_rule_sits_on_the_listener_and_matches_its_source_port():
    """Replies leave from server:port to an ephemeral port, so dport cannot match."""
    cfg = make_config([netem_link("reverse")])
    rules = rules_for_node(cfg, "server")
    assert len(rules) == 1
    assert rules[0].peer_ip == "10.0.0.5"
    assert rules[0].port == 15010
    assert rules[0].match == "sport"
    assert rules_for_node(cfg, "client") == []


def test_both_directions_install_one_rule_on_each_side():
    cfg = make_config([netem_link("both")])
    assert len(rules_for_node(cfg, "client")) == 1
    assert len(rules_for_node(cfg, "server")) == 1


def test_channel_only_traffic_gets_no_rule():
    """Nothing matches a channel, so it stays in the unshaped default class."""
    channels = [{"name": "frames", "from": "client", "to": "server"}]
    cfg = make_config([netem_link()], channels=channels)
    assert len(rules_for_node(cfg, "client")) == 1  # the link, not the channel


def test_links_in_other_modes_are_ignored():
    links = [
        netem_link(),
        {"name": "other", "from": "client", "to": "server", "network": {"mode": "netwaggle"}},
    ]
    cfg = make_config(links)
    assert [r.link_name for r in rules_for_node(cfg, "client")] == ["l0"]


def test_localhost_becomes_a_literal_address():
    nodes = [
        {"name": "client", "host": "localhost", "role": "client"},
        {"name": "server", "host": "localhost", "port": 15010, "role": "server"},
    ]
    cfg = make_config([netem_link()], nodes=nodes)
    assert rules_for_node(cfg, "client")[0].peer_ip == "127.0.0.1"


def test_unexpressible_distribution_is_refused():
    link = netem_link(distribution={"kind": "lognormal", "p50_ms": 10.0, "p99_ms": 16.0})
    cfg = make_config([link])
    with pytest.raises(ValueError) as err:
        rules_for_node(cfg, "client")
    assert "mode: delay" in str(err.value)


def test_fully_unshaped_link_produces_no_rule():
    cfg = make_config([netem_link(distribution={"kind": "constant", "value_ms": 0.0})])
    assert rules_for_node(cfg, "client") == []


# --- the generated tc tree ------------------------------------------------
def one_rule(**kwargs):
    shaping = kwargs.pop("shaping", Shaping(delay_ms=30.0))
    params = {"link_name": "l0", "peer_ip": "10.0.0.9", "port": 15010, "shaping": shaping}
    params.update(kwargs)
    return [ShapeRule(**params)]


def flatten(commands):
    return [" ".join(c) for c in commands]


def test_tree_has_an_unshaped_default_class():
    cmds = flatten(build_commands("eth0", one_rule()))
    assert any("htb default 999" in c for c in cmds)
    assert any("classid {} htb rate 10gbit".format(DEFAULT_CLASSID) in c for c in cmds)


def test_netem_leaf_carries_delay_jitter_and_loss():
    shaping = Shaping(delay_ms=50.0, jitter_ms=8.0, loss_pct=1.0)
    cmds = flatten(build_commands("eth0", one_rule(shaping=shaping)))
    leaf = [c for c in cmds if "netem" in c]
    assert len(leaf) == 1
    assert "delay 50.0ms 8.0ms" in leaf[0]
    assert "loss 1.0%" in leaf[0]


def test_bandwidth_becomes_the_shaped_class_rate():
    shaping = Shaping(delay_ms=10.0, bandwidth_mbps=50.0)
    cmds = flatten(build_commands("eth0", one_rule(shaping=shaping)))
    assert any("classid 1:10 htb rate 50.0mbit" in c for c in cmds)


def test_filter_matches_destination_and_port():
    cmds = flatten(build_commands("eth0", one_rule()))
    filt = [c for c in cmds if "filter add" in c]
    assert len(filt) == 1
    assert "match ip dst 10.0.0.9/32" in filt[0]
    assert "match ip dport 15010 0xffff" in filt[0]
    assert "flowid 1:10" in filt[0]


def test_reverse_rule_filters_on_source_port():
    rules = one_rule(match="sport")
    filt = [c for c in flatten(build_commands("eth0", rules)) if "filter add" in c]
    assert "match ip sport 15010 0xffff" in filt[0]


def test_each_rule_gets_its_own_class_and_priority():
    rules = [
        ShapeRule("a", "10.0.0.9", 1000, Shaping(delay_ms=10.0)),
        ShapeRule("b", "10.0.0.8", 2000, Shaping(delay_ms=20.0)),
    ]
    cmds = flatten(build_commands("eth0", rules))
    assert any("classid 1:10 " in c for c in cmds)
    assert any("classid 1:11 " in c for c in cmds)
    assert any("prio 1 " in c for c in cmds)
    assert any("prio 2 " in c for c in cmds)


def test_no_rules_means_no_commands():
    assert build_commands("eth0", []) == []


def test_teardown_is_a_single_root_delete():
    cmds = teardown_commands("eth0")
    assert flatten(cmds) == ["tc qdisc del dev eth0 root"]


# --- apply / remove through a runner -------------------------------------
class FakeRunner:
    """Records argv and returns a scripted return code per call."""

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on  # substring of the command that should fail

    def __call__(self, argv):
        self.calls.append(" ".join(argv))
        if self.fail_on and self.fail_on in " ".join(argv):
            return 1, "simulated failure"
        return 0, ""


def test_apply_runs_every_command_and_marks_applied():
    runner = FakeRunner()
    backend = NetemBackend("eth0", one_rule(), runner)
    log = backend.apply()
    assert backend.applied
    assert len(log) == len(runner.calls)
    assert any("qdisc add" in c for c in runner.calls)


def test_apply_rolls_back_on_failure():
    """A half-built tree must not be left shaping traffic the config didn't ask for."""
    runner = FakeRunner(fail_on="filter add")
    backend = NetemBackend("eth0", one_rule(), runner)
    with pytest.raises(RuntimeError) as err:
        backend.apply()
    assert "netem setup failed" in str(err.value)
    assert not backend.applied
    assert runner.calls[-1] == "tc qdisc del dev eth0 root"


def test_context_manager_removes_on_exit():
    runner = FakeRunner()
    with NetemBackend("eth0", one_rule(), runner) as backend:
        assert backend.applied
    assert runner.calls[-1] == "tc qdisc del dev eth0 root"


def test_no_rules_applies_nothing():
    runner = FakeRunner()
    assert NetemBackend("eth0", [], runner).apply() == []
    assert runner.calls == []


# --- a composed route ----------------------------------------------------
# One connection from end to end means one sender and one qdisc, so the hops
# are collapsed and the rule lands on the node the path starts at, matching the
# far end. No packet of this flow ever leaves a waypoint.
ROUTE_NODES = NODES + [{"name": "edge", "host": "10.0.0.7", "role": "waypoint"}]
ROUTE_HOPS = [
    {
        "name": "access",
        "from": "client",
        "to": "edge",
        "direction": "forward",
        "network": {"mode": "netem", "distribution": {"kind": "constant", "value_ms": 20.0}},
    },
    {
        "name": "backhaul",
        "from": "edge",
        "to": "server",
        "direction": "forward",
        "network": {"mode": "netem", "distribution": {"kind": "constant", "value_ms": 40.0}},
    },
]
ROUTE = [
    {
        "name": "path",
        "from": "client",
        "to": "server",
        "hops": ["access", "backhaul"],
        "realise": "compose",
    }
]


def _route_config(hops=None, routes=None, nodes=None):
    data = dict(BASE)
    data["nodes"] = nodes or ROUTE_NODES
    data["links"] = hops or ROUTE_HOPS
    data["routes"] = routes or ROUTE
    data["channels"] = []
    data["experiment"] = dict(BASE["experiment"], allow_experimental=True)
    return build_config(data)


def test_a_composed_route_is_one_rule_on_the_node_it_starts_at():
    cfg = _route_config()
    rules = rules_for_node(cfg, "client")
    assert len(rules) == 1
    rule = rules[0]
    assert rule.link_name == "path"
    # The far end, not the waypoint: that is where the packets go.
    assert rule.peer_ip == "10.0.0.9"
    assert rule.port == 15010
    assert rule.match == "dport"
    # The hops' delays are composed into the one qdisc.
    assert rule.shaping.netem_args() == ["delay", "60.0ms"]


def test_a_waypoint_of_a_composed_route_shapes_nothing():
    cfg = _route_config()
    assert rules_for_node(cfg, "edge") == []
    assert rules_for_node(cfg, "server") == []


def test_a_hop_is_not_also_shaped_as_a_link():
    # Without this the 20 ms hop would install its own rule on the client and
    # the path would be delayed twice.
    cfg = _route_config()
    assert [r.link_name for r in rules_for_node(cfg, "client")] == ["path"]


def test_a_reverse_hop_shapes_the_replies_at_the_far_end():
    hops = [
        dict(ROUTE_HOPS[0], direction="reverse"),
        dict(ROUTE_HOPS[1], direction="reverse"),
    ]
    cfg = _route_config(hops=hops)
    assert rules_for_node(cfg, "client") == []
    rules = rules_for_node(cfg, "server")
    assert len(rules) == 1
    assert rules[0].match == "sport"
    assert rules[0].port == 15010
    assert rules[0].peer_ip == "10.0.0.5"
    assert rules[0].shaping.netem_args() == ["delay", "60.0ms"]


# --- shaping needs a namespace the orchestrator can enter -----------------
def _shaping_config(launcher, direction="forward", near_launcher=None):
    """Reference client/server with one netem link, the launchers varied."""
    from cadet.config import build_config

    return build_config(
        {
            "experiment": {"name": "shape", "app": "reference", "repetitions": 1},
            "source": {"kind": "synthetic", "count": 2},
            "app": {"plugin": "reference", "params": {"model_kind": "stub"}},
            "nodes": [
                dict(
                    {"name": "client", "host": "localhost", "role": "client"},
                    **({"launcher": near_launcher} if near_launcher else {})
                ),
                {
                    "name": "server",
                    "host": "203.0.113.9",
                    "port": 15010,
                    "role": "server",
                    "launcher": launcher,
                },
            ],
            "links": [
                {
                    "name": "frames_link",
                    "from": "client",
                    "to": "server",
                    "direction": direction,
                    "network": {
                        "mode": "netem",
                        "distribution": {"kind": "constant", "value_ms": 10.0},
                    },
                }
            ],
        }
    )


def test_shaping_nodes_follows_the_sender():
    """A forward link shapes its source; a reverse direction also shapes its sink."""
    from cadet.orchestrator.placement import shaping_nodes

    assert shaping_nodes(_shaping_config("docker")) == ["client"]
    assert shaping_nodes(_shaping_config("docker", direction="both")) == [
        "client",
        "server",
    ]


def test_preflight_refuses_a_remote_sender_arriving_at_an_unplaced_node():
    """What a node on another machine sends is shaped where it arrives.

    Its own interface is that machine's, so a qdisc there would shape everything
    else on it. The arriving end therefore does the shaping, and it has to be a
    node in a container on this machine for the qdisc to sit on an interface
    this orchestrator owns. A client running as a bare process here has only
    this machine's own interface facing the other machine, which is the thing
    being avoided, so the link is refused and names what to set.
    """
    from cadet.orchestrator.runner import PreflightError, preflight

    config = _shaping_config("ssh_docker", direction="both")
    with pytest.raises(PreflightError) as err:
        preflight(config, [], allow_experimental=True, check_ports=False)
    message = str(err.value)
    assert "'frames_link'" in message
    assert "'client'" in message
    assert "launcher: docker" in message


def test_preflight_accepts_a_remote_sender_arriving_at_a_placed_node(experimental_launcher):
    """With the arriving node containerised, the qdisc has a namespace to go in.

    The fixture puts ``ssh_docker`` in the experimental tier for this test, so
    the gate accepts it with the opt-in and the netem check is what decides.
    """
    from cadet.orchestrator.runner import preflight

    config = _shaping_config("ssh_docker", direction="both", near_launcher="docker")
    preflight(config, [], allow_experimental=True, check_ports=False)


def test_preflight_allows_netem_on_a_link_shaped_from_a_local_sender(experimental_launcher):
    """A remote node that sends nothing shaped is not the orchestrator's problem.

    The fixture puts ``ssh_docker`` in the experimental tier for this test, so
    the gate accepts it with the opt-in and the netem check is what decides -
    which is the thing under test. Its real tier is planned.
    """
    from cadet.orchestrator.runner import preflight

    preflight(
        _shaping_config("ssh_docker"), [], allow_experimental=True, check_ports=False
    )


# --- a sender on another machine is shaped where it arrives ----------------


def test_a_remote_senders_direction_is_shaped_on_the_receivers_ingress():
    """Its own interface is that machine's, so the receiver shapes what arrives.

    A node reached over ssh has no namespace this orchestrator created: its
    interface is the other machine's, carrying everything else on it. The
    direction it sends is therefore shaped where it arrives - an ingress rule on
    the node at the other end, matching the sender's address as the source
    rather than the destination.
    """
    from cadet.network.netem import EGRESS, INGRESS, rules_for_node

    config = _shaping_config("ssh_docker", direction="both", near_launcher="docker")

    client = rules_for_node(config, "client", offmachine=["server"])
    # The client still shapes what it sends, and now also what arrives from the
    # server, which the server cannot shape for itself.
    assert sorted((r.hook, r.match, r.peer_ip) for r in client) == [
        (EGRESS, "dport", "203.0.113.9"),
        (INGRESS, "sport", "203.0.113.9"),
    ]
    # And the server is given nothing to install.
    assert rules_for_node(config, "server", offmachine=["server"]) == []


def test_an_ingress_rule_queues_on_an_intermediate_device():
    """An interface's ingress classifies and redirects; it cannot queue."""
    from cadet.network.netem import (
        INGRESS,
        ShapeRule,
        build_commands,
        ifb_name,
        ingress_commands,
    )
    from cadet.network.shaping import Shaping

    rule = ShapeRule(
        link_name="frames_link",
        peer_ip="203.0.113.9",
        port=15010,
        shaping=Shaping(delay_ms=8.0, jitter_ms=1.0, loss_pct=0.0, bandwidth_mbps=24.0),
        match="sport",
        hook=INGRESS,
    )
    commands = [" ".join(argv) for argv in ingress_commands("eth0", [rule])]
    device = ifb_name("eth0")
    assert commands[0] == "ip link add name {} type ifb".format(device)
    assert "tc qdisc add dev eth0 handle ffff: ingress" in commands
    assert any("mirred egress redirect dev {}".format(device) in c for c in commands)
    # The tree is on the device, and the rule matches the peer as the source.
    assert any(c.startswith("tc qdisc add dev {} root".format(device)) for c in commands)
    assert any("match ip src 203.0.113.9/32" in c and "sport 15010" in c for c in commands)
    # An egress build ignores it: the two hooks do not share a tree.
    assert build_commands("eth0", [rule]) == []


def test_teardown_removes_the_device_only_when_one_was_made():
    """A node that shaped only what it sends tears down exactly what it built."""
    from cadet.network.netem import ifb_name, teardown_commands

    egress_only = [" ".join(argv) for argv in teardown_commands("eth0")]
    assert egress_only == ["tc qdisc del dev eth0 root"]

    with_ingress = [" ".join(argv) for argv in teardown_commands("eth0", ingress=True)]
    # The redirect goes before the device it points at.
    assert with_ingress == [
        "tc qdisc del dev eth0 root",
        "tc qdisc del dev eth0 ingress",
        "tc qdisc del dev {} root".format(ifb_name("eth0")),
        "ip link del {}".format(ifb_name("eth0")),
    ]
