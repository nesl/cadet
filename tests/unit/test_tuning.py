"""Unit tests for per-node transport tuning and the per-run link report.

What this covers:
    Which tunable is realised by which mechanism (socket option, namespaced
    sysctl, route attribute), that an unset field leaves the kernel default
    alone, that nothing is ever emitted which would change a setting host-wide,
    and that the run record carries the configured parameters beside the
    measured ones.

No sysctl is written and no socket is tuned for real: the functions return
commands and option triples, which is what makes the policy reviewable.
"""

import socket

import pytest

from cadet.config import build_config
from cadet.metrics.link_report import MessageStats, link_summary
from cadet.network.tuning import (
    describe,
    docker_sysctl_args,
    namespace_commands,
    requires_namespace,
    socket_options,
    sysctl_settings,
)

BASE = {
    "experiment": {"name": "tun", "app": "reference", "repetitions": 1},
    "source": {"kind": "synthetic", "count": 2},
    "app": {"plugin": "reference", "params": {"model_kind": "stub"}},
}


def make_config(tuning=None, launcher="local", links=None):
    node = {"name": "client", "host": "localhost", "role": "client", "launcher": launcher}
    if tuning is not None:
        node["transport_tuning"] = tuning
    data = dict(BASE)
    data["nodes"] = [
        node,
        {"name": "server", "host": "127.0.0.1", "port": 15010, "role": "server"},
    ]
    data["links"] = links or []
    return build_config(data)


def client_of(config):
    return config.nodes[0]


# --- which mechanism realises which knob ---------------------------------
def test_nodelay_is_a_socket_option_and_needs_no_namespace():
    node = client_of(make_config({"tcp_nodelay": True}))
    assert socket_options(node) == [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]
    assert sysctl_settings(node) == {}
    assert requires_namespace(node) is False


def test_nodelay_false_is_distinct_from_unset():
    """Explicitly enabling Nagle is a choice; leaving it unset is not."""
    explicit = client_of(make_config({"tcp_nodelay": False}))
    assert socket_options(explicit) == [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 0)]
    unset = client_of(make_config({}))
    assert socket_options(unset) == []


def test_slow_start_after_idle_is_a_namespaced_sysctl():
    node = client_of(make_config({"tcp_slow_start_after_idle": False}))
    assert sysctl_settings(node) == {"net.ipv4.tcp_slow_start_after_idle": "0"}
    assert socket_options(node) == []
    assert requires_namespace(node) is True


def test_initcwnd_is_a_route_attribute_and_needs_a_route():
    node = client_of(make_config({"initcwnd": 20}))
    assert requires_namespace(node) is True
    # Without a route named, the setting is skipped rather than applied to a
    # route the config never mentioned.
    assert namespace_commands(node) == []
    cmds = namespace_commands(node, routes=[("10.60.0.2", "n0-eth1")])
    assert cmds == [
        ["ip", "route", "change", "10.60.0.2", "dev", "n0-eth1", "initcwnd", "20"]
    ]


def test_no_tuning_emits_nothing_at_all():
    node = client_of(make_config())
    assert socket_options(node) == []
    assert namespace_commands(node) == []
    assert docker_sysctl_args(node) == []
    assert requires_namespace(node) is False
    assert describe(node) == {}


# --- nothing may escape the node's namespace ------------------------------
def test_sysctls_are_passed_as_docker_flags_not_run_in_the_container():
    """An unprivileged container has a read-only /proc/sys, so creation-time is
    the only moment these can be set."""
    node = client_of(make_config({"tcp_slow_start_after_idle": False}))
    assert docker_sysctl_args(node) == [
        "--sysctl", "net.ipv4.tcp_slow_start_after_idle=0"
    ]


def test_namespace_commands_never_touch_a_global_setting():
    node = client_of(make_config({"tcp_slow_start_after_idle": True, "initcwnd": 10}))
    cmds = namespace_commands(node, routes=[("10.60.0.2", "eth1")])
    flat = [" ".join(c) for c in cmds]
    # sysctl -w on a net.* key is per network namespace; nothing here writes to
    # /proc/sys on the host or uses --system.
    assert flat[0] == "sysctl -qw net.ipv4.tcp_slow_start_after_idle=1"
    assert all("--system" not in f for f in flat)
    assert all("/proc/sys" not in f for f in flat)


def test_describe_reports_unset_fields_as_none():
    """CADET does not know the kernel's default and must not claim one."""
    node = client_of(make_config({"tcp_nodelay": True}))
    assert describe(node) == {
        "tcp_nodelay": True,
        "tcp_slow_start_after_idle": None,
        "initcwnd": None,
    }


# --- preflight refuses what would change the host ------------------------
@pytest.mark.parametrize("tuning", [
    {"tcp_slow_start_after_idle": False},
    {"initcwnd": 20},
])
def test_local_launcher_refuses_namespace_only_tuning(tuning):
    from cadet.orchestrator.runner import PreflightError, preflight

    config = make_config(tuning, launcher="local")
    with pytest.raises(PreflightError) as err:
        preflight(config, [])
    assert "namespace" in str(err.value)


def test_local_launcher_allows_nodelay():
    from cadet.orchestrator.runner import preflight

    preflight(make_config({"tcp_nodelay": True}, launcher="local"), [])


# --- the run record -------------------------------------------------------
def test_link_summary_states_the_backend_semantics():
    links = [{
        "name": "l0", "from": "client", "to": "server", "direction": "forward",
        "network": {"mode": "netem", "distribution": {"kind": "constant", "value_ms": 60}},
    }]
    config = make_config(links=links)
    summary = link_summary(config.links[0])
    assert summary["mode"] == "netem"
    assert summary["semantics"] == "packet-level"
    assert summary["direction"] == "forward"
    assert summary["distribution"]["value_ms"] == 60


def test_delay_backend_is_labelled_message_level():
    links = [{"name": "l0", "from": "client", "to": "server",
              "network": {"mode": "delay"}}]
    config = make_config(links=links)
    assert link_summary(config.links[0])["semantics"] == "message-level"


def test_message_stats_counts_unanswered_and_percentiles():
    stats = MessageStats(sent=100, received=98, latencies_ms=list(range(1, 99)))
    assert stats.unanswered == 2
    assert stats.unanswered_pct == pytest.approx(2.0)
    # Nearest rank over 98 samples of 1..98: index = round(pct/100 * 97).
    assert stats.percentile(50) == 49
    assert stats.percentile(99) == 97
    assert stats.percentile(0) == 1
    assert stats.percentile(100) == 98
    body = stats.as_dict()
    assert body["messages_sent"] == 100
    assert body["latency_ms"]["samples"] == 98


def test_message_stats_with_no_replies_reports_none_not_zero():
    """A missing measurement must not look like a measured zero."""
    stats = MessageStats(sent=10, received=0)
    body = stats.as_dict()
    assert body["unanswered"] == 10
    assert body["latency_ms"]["p50"] is None
    assert body["latency_ms"]["mean"] is None
