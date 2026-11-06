"""Tests for config schema validation.

Covers: the braking example validates; unknown keys, enum/type errors, cross-field
rules (app/experiment agreement, channel/link endpoints, active model in registry)
are rejected; and app params are validated dynamically against the registered
plugin (the schema itself is app-agnostic).
"""

import pytest
from conftest import EXAMPLE_CONFIGS
from pydantic import ValidationError

import examples.braking.flat.app  # noqa: F401  (registers the braking plugin)
from cadet.config import ConfigError, build_config, load_config_data, resolve_app_params


@pytest.mark.parametrize("path", EXAMPLE_CONFIGS, ids=lambda p: p.stem)
def test_example_configs_validate(path):
    cfg = build_config(load_config_data(str(path)))
    assert cfg.experiment.app == cfg.app.plugin


def _braking_dict():
    return load_config_data(str(EXAMPLE_CONFIGS[0]))


def test_unknown_key_rejected():
    data = _braking_dict()
    data["experiment"]["bogus"] = 1
    with pytest.raises(ValidationError):
        build_config(data)


def test_app_experiment_mismatch_rejected():
    data = _braking_dict()
    data["app"]["plugin"] = "v2v"  # experiment.app stays "braking"
    with pytest.raises(ValidationError):
        build_config(data)


def test_channel_unknown_node_rejected():
    data = _braking_dict()
    data["channels"][0]["to"] = "ghost"
    with pytest.raises(ValidationError):
        build_config(data)


def test_active_model_must_be_in_registry():
    data = _braking_dict()
    data["models"]["model"] = "not_a_model"
    with pytest.raises(ValidationError):
        build_config(data)


def test_bad_source_kind_rejected():
    data = _braking_dict()
    data["source"]["kind"] = "teleport"
    with pytest.raises(ValidationError):
        build_config(data)


def test_app_params_resolve_against_plugin():
    # The schema's app is generic; params validate against the registered plugin.
    cfg = build_config(_braking_dict())
    params = resolve_app_params(cfg)
    assert params.plugin == "braking"
    assert params.conf_thresh == 0.3  # braking default
    assert "audi.tt" in params.controllers


def test_unknown_app_plugin_rejected():
    data = _braking_dict()
    data["experiment"]["app"] = "nope"
    data["app"]["plugin"] = "nope"
    cfg = build_config(data)
    with pytest.raises(ConfigError):
        resolve_app_params(cfg)


def test_bad_app_params_rejected():
    data = _braking_dict()
    data["app"]["params"] = {"detection_threshold": "not-an-int"}
    cfg = build_config(data)
    with pytest.raises(ValidationError):
        resolve_app_params(cfg)


def test_invalid_type_rejected():
    data = _braking_dict()
    data["source"]["carla"]["port"] = "not-an-int"
    with pytest.raises(ValidationError):
        build_config(data)


def test_client_node_port_optional():
    # The client node in the example omits `port`; the config must still validate.
    cfg = build_config(_braking_dict())
    client = next(n for n in cfg.nodes if n.role == "client")
    assert client.port is None


def test_server_node_requires_port():
    data = _braking_dict()
    for node in data["nodes"]:
        if node["role"] == "server":
            node.pop("port", None)
    with pytest.raises(ValidationError):
        build_config(data)


def test_registry_entry_rejects_score_threshold():
    # score_threshold was removed from registry entries (now an app concern).
    data = _braking_dict()
    data["models"]["registry"]["yolo11x"]["score_threshold"] = 0.5
    with pytest.raises(ValidationError):
        build_config(data)


def _live_source(**fields):
    data = _braking_dict()
    data["source"] = dict(kind="live", **fields)
    return data


def test_live_source_accepts_a_device_index():
    cfg = build_config(_live_source(device=0))
    assert cfg.source.kind == "live" and cfg.source.device == 0 and cfg.source.url is None


def test_live_source_accepts_a_stream_url():
    cfg = build_config(_live_source(url="rtsp://camera:554/stream"))
    assert cfg.source.url == "rtsp://camera:554/stream" and cfg.source.device is None


def test_live_source_rejects_both_targets():
    with pytest.raises(ValidationError) as err:
        build_config(_live_source(device=0, url="rtsp://camera:554/stream"))
    assert "exactly one of" in str(err.value) and "both" in str(err.value)


def test_live_source_rejects_neither_target():
    with pytest.raises(ValidationError) as err:
        build_config(_live_source())
    assert "exactly one of" in str(err.value) and "neither" in str(err.value)


def test_live_source_has_no_loop():
    # loop is a replay concept: a stream is not seekable, so the key is refused
    # rather than silently ignored.
    with pytest.raises(ValidationError):
        build_config(_live_source(device=0, loop=True))
