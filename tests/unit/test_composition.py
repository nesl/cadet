"""Tests for config composition (extends:) and the not-implemented gate.

Covers deep-merge inheritance, discriminated-variant replacement, cycle
detection, and that declared-but-unimplemented options are reported (never
silently accepted).
"""

import pytest

from cadet.config.implemented import unimplemented_features
from cadet.config.loader import ConfigError, load_config_data, load_run_configs


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


BASE = """
source: {kind: synthetic, count: 8, width: 64, height: 48}
app: {plugin: reference, params: {model_kind: stub}}
outputs: {root: outputs, naming: "{experiment}/{app}/{run_id}"}
"""

CHILD = """
extends: base.yaml
experiment: {name: child, app: reference, repetitions: 1, seed: 0}
nodes:
  - {name: client, host: localhost, role: client, device: cpu}
  - {name: server, host: 127.0.0.1, port: 16000, role: server, device: cpu}
channels: [{name: c, from: client, to: server}]
"""


def test_extends_merges_base(tmp_path):
    _write(tmp_path, "base.yaml", BASE)
    child = _write(tmp_path, "child.yaml", CHILD)
    runs, _ = load_run_configs(child, [])
    cfg = runs[0]
    # Inherited from base:
    assert cfg.source.kind == "synthetic" and cfg.source.count == 8
    assert cfg.app.plugin == "reference"
    # Defined in child:
    assert cfg.experiment.name == "child"
    assert [n.name for n in cfg.nodes] == ["client", "server"]


def test_child_overrides_base_scalar(tmp_path):
    _write(tmp_path, "base.yaml", BASE)
    child = _write(
        tmp_path,
        "child.yaml",
        CHILD + "source: {kind: synthetic, count: 99}\n",
    )
    runs, _ = load_run_configs(child, [])
    # count overridden; width/height still inherited (same variant -> deep-merge).
    assert runs[0].source.count == 99
    assert runs[0].source.width == 64


def test_variant_switch_replaces_subtree(tmp_path):
    _write(tmp_path, "base.yaml", BASE)
    # Switching source.kind must drop synthetic-only keys (count/width/height),
    # otherwise the replay model would reject them.
    child = _write(
        tmp_path,
        "child.yaml",
        CHILD + "source: {kind: replay, path: /tmp/imgs}\n",
    )
    runs, _ = load_run_configs(child, [])
    assert runs[0].source.kind == "replay"
    assert runs[0].source.path == "/tmp/imgs"


def test_extends_cycle_detected(tmp_path):
    _write(tmp_path, "a.yaml", "extends: b.yaml\n")
    _write(tmp_path, "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ConfigError, match="circular extends"):
        load_config_data(str(tmp_path / "a.yaml"))


def test_gate_flags_unimplemented(tmp_path):
    _write(tmp_path, "base.yaml", BASE)
    # A planned option, reached through composition: no launcher class exists
    # for ssh_venv, and the gate has to see it even though the child inherited
    # the rest of its config from the base.
    child = _write(
        tmp_path,
        "child.yaml",
        CHILD
        + """nodes:
  - {name: client, host: localhost, role: client, device: cpu}
  - {name: server, host: 127.0.0.1, port: 16000, role: server, device: cpu}
  - {name: far, host: 127.0.0.1, port: 16001, role: server, device: cpu}
links:
  - name: access
    from: client
    to: server
    network: {mode: delay, distribution: {kind: constant, value_ms: 20.0}}
  - name: backhaul
    from: server
    to: far
    network: {mode: delay, distribution: {kind: constant, value_ms: 40.0}}
routes:
  - {name: c_to_far, from: client, to: far, hops: [access, backhaul],
     realise: relay}
""",
    )
    runs, _ = load_run_configs(child, [])
    missing = unimplemented_features(runs[0])
    # A relayed route is the refused value. It used to be a launcher, but every
    # launcher the schema names is implemented now, so a launcher value no
    # longer tests this at all - and asserting the absence of findings would
    # have inverted the test while keeping its name. What this is about is that
    # a gated value survives `extends:` merging and is still seen by the gate,
    # whichever option carries it.
    assert any("realise=relay" in m and "(planned)" in m for m in missing), missing


def test_gate_passes_for_supported(tmp_path):
    _write(tmp_path, "base.yaml", BASE)
    child = _write(tmp_path, "child.yaml", CHILD)
    runs, _ = load_run_configs(child, [])
    assert unimplemented_features(runs[0]) == []
