"""Tests for the feature gate's three tiers and the experimental opt-in.

The distinction under test is the one the opt-in exists for: an option with an
implementation that is merely unproven can be accepted deliberately, while an
option with nothing behind it stays refused however loudly the caller asks.
"""

import json
import os
import subprocess
import sys

import pytest
from conftest import BRAKING_CONFIG

from cadet.config.implemented import (
    EXPERIMENTAL,
    IMPLEMENTED,
    _requested,
    allow_experimental,
    experimental_features,
    tier,
    unimplemented_features,
)
from cadet.config.loader import load_run_configs

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG = os.path.join(REPO, "examples", "reference", "synthetic.yaml")
APP = "examples.reference.app"


def _config(*overrides):
    runs, _base = load_run_configs(CONFIG, list(overrides))
    return runs[0]


# The planned value these tests stand on. Every launcher is now at least
# experimental, so `nodes.launcher` no longer offers one - and a planned option
# has to come from somewhere, because what these tests check is the *tier*, not
# the feature. `routes.realise: relay` is the choice for the reason the old
# comment gave for ssh_venv: it cannot be promoted without someone first
# writing the forwarding stage a relayed hop needs, so it will not drift out
# from under this file. It cannot be expressed with a --set, so it comes as a
# whole config.
PLANNED_CONFIG = {
    "experiment": {"name": "gate_planned", "app": "reference"},
    "source": {"kind": "synthetic", "count": 1},
    "nodes": [
        {"name": "ego", "host": "127.0.0.1", "role": "client"},
        {"name": "rsu", "host": "127.0.0.1", "role": "server", "port": 10101},
        {"name": "follower", "host": "127.0.0.1", "role": "server", "port": 10102},
    ],
    "links": [
        {"name": "access", "from": "ego", "to": "rsu",
         "network": {"mode": "delay",
                     "distribution": {"kind": "constant", "value_ms": 20.0}}},
        {"name": "backhaul", "from": "rsu", "to": "follower",
         "network": {"mode": "delay",
                     "distribution": {"kind": "constant", "value_ms": 40.0}}},
    ],
    "routes": [{"name": "ego_to_follower", "from": "ego", "to": "follower",
                "hops": ["access", "backhaul"], "realise": "relay"}],
    "app": {"plugin": "reference"},
}


def _planned_config():
    from cadet.config import build_config

    return build_config(PLANNED_CONFIG)


def _planned_config_file(tmp_path):
    import yaml

    path = tmp_path / "planned.yaml"
    path.write_text(yaml.safe_dump(PLANNED_CONFIG, sort_keys=False))
    return str(path)


# --- the tiers themselves -------------------------------------------------
def test_tiers_are_disjoint():
    # A value in both maps would make its tier depend on lookup order.
    for path, trial in EXPERIMENTAL.items():
        assert not (trial & IMPLEMENTED.get(path, set())), path


def test_tier_classifies_each_launcher():
    # docker was promoted out of experimental once it passed the lifecycle
    # matrix, and ssh_docker once it passed the same four cases with a node on a
    # second machine; implemented.py records what each run was.
    assert tier("nodes.launcher", "local") == "implemented"
    assert tier("nodes.launcher", "docker") == "implemented"
    assert tier("nodes.launcher", "ssh_docker") == "implemented"
    # ssh_venv is implemented too now: a role as a process in a virtualenv on
    # a second machine, verified by the same four lifecycle cases ssh_docker
    # was held to. So *every* launcher the schema names is implemented, and
    # this assertion is what makes each tier move a deliberate edit rather
    # than a drift.
    assert tier("nodes.launcher", "ssh_venv") == "implemented"


def test_an_unknown_path_is_planned():
    assert tier("nothing.like.this", "whatever") == "planned"


# --- the gate -------------------------------------------------------------
def test_a_clean_config_passes_either_way():
    config = _config()
    assert unimplemented_features(config) == []
    assert unimplemented_features(config, allow_experimental=True) == []


def test_experimental_is_refused_by_default_and_says_how_to_accept(experimental_launcher):
    config = _config("nodes.1.launcher={}".format(experimental_launcher))
    missing = unimplemented_features(config)
    assert len(missing) == 1
    assert "experimental" in missing[0]
    assert "--allow-experimental" in missing[0]


def test_experimental_is_accepted_with_the_opt_in(experimental_launcher):
    config = _config("nodes.1.launcher={}".format(experimental_launcher))
    assert unimplemented_features(config, allow_experimental=True) == []


def test_planned_is_refused_even_with_the_opt_in():
    # The point of the tier split: there is nothing behind a planned option, so
    # accepting it could only replace a refusal with a crash.
    # A relayed route stands in for the planned tier. It is named rather than
    # taken from whichever option happens to be planned, because this test was
    # written against transport.protocol, then the clock_offset collector, then
    # ssh_venv, and every one of them was promoted under it. What the split
    # guarantees is checked here either way: a planned value is refused *with*
    # the opt-in, and whether code exists behind it is not the gate's business.
    config = _planned_config()
    missing = unimplemented_features(config, allow_experimental=True)
    assert len(missing) == 1, missing
    assert "realise=relay" in missing[0], missing
    assert "(planned)" in missing[0], missing


def test_async_source_is_planned_and_stays_refused():
    # sync_mode is a CARLA-source field, so this uses the braking example rather
    # than overriding the synthetic one into a shape it does not have.
    runs, _base = load_run_configs(str(BRAKING_CONFIG), ["source.sync_mode=false"])
    missing = unimplemented_features(runs[0], allow_experimental=True)
    assert any("sync_mode" in m and "(planned)" in m for m in missing)
    assert tier("source.sync_mode", False) == "planned"
    assert tier("source.sync_mode", True) == "implemented"


def test_a_collector_with_no_implementation_stays_refused():
    # Every collector the schema accepts is now implemented, so this tier is
    # checked against the gate directly rather than through a config: a name
    # the schema would reject cannot be put in one, and the rule under test is
    # "absent from the map means refused", not "clock_offset is refused".
    assert tier("metrics.collectors", "clock_offset_v2") == "planned"
    assert "planned" in unimplemented_features.__doc__


def test_every_collector_the_schema_accepts_is_implemented():
    # The gate and the schema have to agree in this direction: a collector a
    # config can ask for and the gate has never heard of is refused with no
    # way to opt in, which reads as a bug rather than as a plan.
    from cadet.config.schema import MetricsConfig

    accepted = MetricsConfig.model_fields["collectors"].annotation.__args__[0].__args__
    for name in accepted:
        assert tier("metrics.collectors", name) == "implemented", name


def test_a_verified_collector_is_accepted():
    # The five built collectors left the planned gate on live evidence; the
    # comment beside them in implemented.py carries it. clock_offset measures
    # nothing unless a node is on another machine, and records a zero offset
    # with its reason when none is - so it is accepted here too.
    config = _config("metrics.collectors=[gpu,cpu,energy,bandwidth,queue,clock_offset]")
    assert unimplemented_features(config) == []


def test_experimental_features_lists_what_was_asked_for(experimental_launcher):
    config = _config("nodes.1.launcher={}".format(experimental_launcher))
    assert experimental_features(config) == [
        "nodes['server'].launcher={}".format(experimental_launcher)
    ]
    # Reported whether or not it was opted into: the list is "what this config
    # asks for", which is what a run records.
    assert experimental_features(_config()) == []


# --- resolving the opt-in -------------------------------------------------
def test_the_flag_wins_over_a_config_that_does_not_ask():
    assert allow_experimental(_config(), True) is True


def test_the_config_field_is_used_when_no_flag_is_given():
    assert allow_experimental(_config("experiment.allow_experimental=true"), None) is True
    assert allow_experimental(_config(), None) is False


def test_an_absent_flag_does_not_override_the_config_field():
    config = _config("experiment.allow_experimental=true")
    assert allow_experimental(config, False) is True
    assert allow_experimental(config, None) is True


# --- end to end through the CLI ------------------------------------------
def _cli(*argv):
    return subprocess.run(
        [sys.executable, "-m", "cadet"] + list(argv),
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def test_validate_refuses_a_planned_option_through_the_cli(tmp_path):
    # The CLI runs in a subprocess, so this cannot stand a fixture in for the
    # tier maps: it exercises whatever the build really declares. A relayed
    # route is the planned value, written to a file because a route cannot be
    # expressed with a --set.
    planned = _planned_config_file(tmp_path)
    result = _cli("validate", planned, "--app-module", APP)
    assert result.returncode == 1
    assert "planned" in result.stderr
    # And the opt-in does not rescue it, which is the tier split's whole point.
    opted_in = _cli(
        "validate",
        planned,
        "--app-module",
        APP,
        "--set",
        "experiment.allow_experimental=true",
    )
    assert opted_in.returncode == 1
    assert "planned" in opted_in.stderr


def _ask_for(path, value):
    """Return ``(config, app module, --set arguments)`` that request one gated value.

    Per option rather than one dotted path, because asking for a value is not
    always one setting, and not always the same config. A config that names an
    analysis backend and no endpoint is refused by the schema before the gate is
    reached, so that option needs its company; and a cell is a shape rather than
    a value - an access point, its stations and their positions - so the option
    that asks for one comes as the example that has it. Returns ``None`` for an
    option this test has not been taught.
    """
    if path == "nodes.launcher":
        return CONFIG, APP, ["--set", "nodes.1.launcher={}".format(value)]
    if path == "analysis.backend":
        return (
            CONFIG,
            APP,
            [
                "--set",
                "analysis.backend={}".format(value),
                "--set",
                "analysis.endpoint=http://127.0.0.1:9/v1/chat/completions",
                "--set",
                "analysis.model=a-model",
            ],
        )
    if path == "connectors.type" and value == "ap":
        return (
            os.path.join(REPO, "examples", "distributed_sensing", "wifi_5.yaml"),
            "examples.distributed_sensing.app",
            [],
        )
    return None


@pytest.mark.skipif(
    not EXPERIMENTAL,
    reason="nothing is experimental in this build; the CLI path needs a real value",
)
def test_validate_accepts_an_experimental_option_with_the_config_opt_in():
    """The accepted-by-opt-in path, through the CLI, when a build has one.

    Skipped rather than faked: the CLI is a subprocess and cannot see a
    fixture's tier maps, and pointing this at a planned value would assert the
    opposite of what it is for. In-process coverage of the same path is above.
    """
    # One real experimental value is enough to exercise the CLI path, so this
    # takes the first the helper has been taught rather than insisting every
    # gated option be expressible as a `--set`: a cell is a shape, not a value.
    # The obligation stays - at least one has to be taught - which is what the
    # assertion reports when a build's experimental tier is all new.
    taught = [
        asked
        for path, values in sorted(EXPERIMENTAL.items())
        for value in sorted(values)
        for asked in [_ask_for(path, value)]
        if asked
    ]
    assert taught, "update _ask_for for one of {}".format(sorted(EXPERIMENTAL))
    config, app_module, overrides = taught[0]
    result = _cli(
        "validate",
        config,
        "--app-module",
        app_module,
        *overrides,
        "--set",
        "experiment.allow_experimental=true"
    )
    assert result.returncode == 0
    assert "EXPERIMENTAL, accepted by opt-in" in result.stdout


def test_run_gates_before_starting_anything(tmp_path):
    # The gate used to run only in `cadet validate`, which nothing obliges a
    # caller to run first. A relayed route is the refused value: it needs a
    # forwarding stage that does not exist, so a refusal that arrives is the
    # gate's - nothing further in could have produced one without first trying
    # to launch something, and there is nothing to launch.
    result = _cli(
        "run",
        _planned_config_file(tmp_path),
        "--app-module",
        APP,
        "--set",
        "outputs.root={}".format(tmp_path),
    )
    assert result.returncode == 2
    assert "will not run" in result.stderr


@pytest.mark.parametrize("opt_in", [True, False])
def test_run_records_the_opt_in(tmp_path, opt_in):
    argv = [
        "run",
        CONFIG,
        "--app-module",
        APP,
        "--set",
        "outputs.root={}".format(tmp_path),
    ]
    if opt_in:
        argv.insert(1, "--allow-experimental")
    result = _cli(*argv)
    assert result.returncode == 0, result.stdout + result.stderr
    markers = []
    for root, _dirs, files in os.walk(str(tmp_path)):
        if "run.json" in files:
            markers.append(os.path.join(root, "run.json"))
    assert markers, "no run.json written"
    with open(markers[0]) as handle:
        assert json.load(handle)["allow_experimental"] is opt_in


# --- the wireless medium ----------------------------------------------------
# Both spellings of it carry traffic and are implemented. The gate is still
# asserted on each, because the cell spelling is a connector and not a link:
# a gate on the link's medium alone would let the shape that has no links past.


def test_both_spellings_of_a_wireless_medium_are_implemented():
    """Verified by the five-sensor cell: see IMPLEMENTED for the measurement."""
    assert tier("links.network.medium", "wifi") == "implemented"
    assert tier("connectors.type", "ap") == "implemented"


def test_the_wired_shapes_of_the_same_two_options_are_implemented():
    """The wired medium is verified and runs, as it was before the cell did."""
    assert tier("links.network.medium", "wired") == "implemented"
    assert tier("connectors.type", "switch") == "implemented"


def test_a_cell_needs_no_opt_in_and_the_example_asks_for_none():
    """The connector spelling, which has no links for a link gate to catch."""
    config_path = os.path.join(REPO, "examples", "distributed_sensing", "wifi_5.yaml")
    runs, _base = load_run_configs(config_path, [])
    config = runs[0]

    assert unimplemented_features(config) == []
    assert experimental_features(config) == []
    assert config.experiment.allow_experimental is False
    # The connector is still reported as asked for, so a later tier change
    # reaches it rather than passing it by.
    assert any("type=ap" in name for name, _value, _path in _requested(config))
