"""Tests for runtime wiring that doesn't need CARLA.

Covers: the client->server link network model (none vs constant delay), run-config
selection by sweep index, output-dir preparation (config saved alongside), and
node/link lookup helpers. The braking app plugin exposes client/server builders.
"""

import os

from conftest import BRAKING_CONFIG

import examples.braking.flat.app  # noqa: F401  (registers the braking plugin)
from cadet.core.apps import get_app
from cadet.core.network import NullNetworkModel, SampledNetworkModel, build_network_model
from cadet.roles.common import (
    link_into_node,
    node_by_role,
    prepare_output_dir,
    select_run_config,
)

BRAKING = str(BRAKING_CONFIG)
HOST_OVERRIDE = ["source.carla.host=127.0.0.1"]


def test_network_model_from_link():
    """Braking's link is a network path: a normal delay, p50 30 ms, p99 near 100 ms."""
    cfg, _ = select_run_config(BRAKING, HOST_OVERRIDE, 0)
    link = link_into_node(cfg, "server")
    assert link is not None
    assert link.network.distribution.kind == "normal"
    assert link.network.distribution.mean_ms == 30.0
    model = build_network_model(link.network)
    assert isinstance(model, SampledNetworkModel)
    # Sampled, so assert the distribution rather than one value.
    samples = [model.sample_delay_s() for _ in range(2000)]
    assert 0.025 < sum(samples) / len(samples) < 0.035
    assert len(set(samples)) > 1


def test_server_processing_cost_is_not_a_link():
    """The original fixed 13 ms is compute on the server, not a network delay.

    Keeping it on the link made a server cost emulatable as distance, so every
    network backend would have had to pretend the server was 13 ms away.
    """
    cfg, _ = select_run_config(BRAKING, HOST_OVERRIDE, 0)
    server = node_by_role(cfg, "server")
    assert server.extra_processing_ms == 13.0


def test_network_model_none():
    assert isinstance(build_network_model(_FakeNet("none")), NullNetworkModel)


class _FakeNet:
    def __init__(self, mode):
        self.mode = mode
        self.distribution = None


def test_select_run_config_indexes_sweep():
    cfg0, hash0 = select_run_config(BRAKING, HOST_OVERRIDE, 0)
    cfg8, hash8 = select_run_config(BRAKING, HOST_OVERRIDE, 8)
    assert cfg0.source.scenario.ego_vehicle == "audi.tt"
    assert cfg0.source.scenario.target_object == "car"
    # Different sweep cells -> different configs -> different hashes.
    assert hash0 != hash8


def test_node_lookup():
    cfg, _ = select_run_config(BRAKING, HOST_OVERRIDE, 0)
    assert node_by_role(cfg, "server").port == 10000
    assert node_by_role(cfg, "client").port is None


def test_prepare_output_dir_saves_config(tmp_path):
    cfg, h = select_run_config(BRAKING, HOST_OVERRIDE + ["outputs.root={}".format(tmp_path)], 0)
    out = prepare_output_dir(cfg, h + "-r0")
    assert os.path.isdir(out)
    assert os.path.exists(os.path.join(out, "config.yaml"))


def test_braking_plugin_has_builders():
    plugin = get_app("braking")
    assert hasattr(plugin, "build_client") and hasattr(plugin, "build_server")
