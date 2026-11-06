"""Tests for the moved simulator keys: the refusal, and the migration.

The refusal has to name the replacement for *every* key that moved, because a
config carrying four of them should learn four new names rather than be told
once that a section is gone. The migration has to preserve comments, because
these files are read by people, and it has to refuse a config whose own
settings disagreed rather than pick one.
"""

import io
import os
import sys

import pytest

from cadet.config.loader import MOVED_KEYS, ConfigError, build_config

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import migrate_simulator_keys as migrate  # noqa: E402


def _every_old_key():
    """A config carrying every key that moved, in its old shape."""
    return {
        "sim": {
            "carla": {
                "image": "carlasim/carla:0.9.10.1",
                "rpc_port": 3000,
                "gpu_index": 1,
                "quality": "Low",
                "offscreen": True,
            },
            "client_egg": "carla-0.9.10-py3.7-linux-x86_64.egg",
            "carla_client_egg_host_path": "/eggs",
            "scenario_runner": {
                "path": "/opt/scenario_runner",
                "source_digest": "abc",
                "commit": "def",
                "image": "cadet:dev",
            },
        },
        "source": {"simulator_host": "h", "simulator_port": 1, "simulator_timeout": 2},
    }


def test_every_moved_key_is_refused_and_names_its_replacement():
    with pytest.raises(ConfigError) as err:
        build_config(_every_old_key())
    message = str(err.value)
    for old, new in MOVED_KEYS.items():
        assert old in message, old
        assert new in message, new


def test_an_unlisted_key_in_a_moved_section_is_refused_too():
    # Otherwise a config could carry a key nothing reads and validate.
    with pytest.raises(ConfigError) as err:
        build_config({"sim": {"something_nobody_listed": 1}})
    assert "that section moved to" in str(err.value)


def test_the_mapping_and_the_migration_cannot_drift():
    # The script imports the loader's table rather than keeping its own.
    assert migrate.KEY_MOVES is MOVED_KEYS


def test_migration_moves_the_flat_keys_under_the_source_block():
    text = (
        "source:\n"
        "  kind: carla\n"
        "  simulator_host: CARLA_HOST   # placeholder\n"
        "  simulator_port: 2000\n"
        "  simulator_timeout: 10\n"
        "models:\n"
        "  warmup: false\n"
    )
    new_text, changed, unhandled = migrate.migrate_yaml_text(text)
    assert not unhandled, unhandled
    assert "  carla:\n" in new_text
    assert "    host: CARLA_HOST   # placeholder" in new_text  # the comment survives
    assert "    port: 2000" in new_text
    assert "    timeout: 10" in new_text
    assert "simulator_host" not in new_text
    # Nothing outside the source block moved.
    assert new_text.rstrip().endswith("warmup: false")
    assert len(changed) == 3


def test_migration_nests_the_runner_under_an_existing_scenario_block():
    text = (
        "source:\n"
        "  kind: carla\n"
        "  scenario:\n"
        "    map: Town05\n"
        "sim:\n"
        "  carla:\n"
        "    image: img\n"
        "  scenario_runner:\n"
        "    path: /opt/scenario_runner\n"
    )
    new_text, _changed, unhandled = migrate.migrate_yaml_text(text)
    assert not unhandled, unhandled
    assert "    runner:\n      path: /opt/scenario_runner" in new_text
    assert "  carla:\n    image: img" in new_text
    assert "sim:" not in new_text


def test_migration_refuses_a_config_whose_two_ports_disagreed():
    # Five committed configs had exactly this: a launch on 3000 and a client on
    # 2000, with nothing tying them. Which one a run meant is not the script's
    # to guess.
    text = (
        "source:\n"
        "  kind: carla\n"
        "  simulator_port: 2000\n"
        "sim:\n"
        "  carla:\n"
        "    rpc_port: 3000\n"
    )
    new_text, _changed, unhandled = migrate.migrate_yaml_text(text)
    assert new_text == text, "a refused file must be left alone"
    assert any("two values for source.carla.port" in u for u in unhandled), unhandled


def test_migration_rewrites_a_command_line_and_prose():
    text = (
        "#!/bin/bash\n"
        "python -m cadet run cfg.yaml --set source.simulator_port=2200 \\\n"
        "  --set sim.carla.rpc_port=2200 --set sim.scenario_runner.path=/opt/sr\n"
    )
    new_text, changed = migrate.migrate_script_text(text)
    assert "source.carla.port=2200" in new_text
    assert "source.scenario.runner.path=/opt/sr" in new_text
    assert "simulator_port" not in new_text and "sim.carla" not in new_text
    assert changed


def test_the_migrator_refuses_to_rewrite_the_files_that_define_the_mapping(tmp_path):
    # It rewrote its own table the first time it was run over `scripts/`, which
    # turns every entry into new-to-new and silently stops the migration.
    source = io.open(
        os.path.join(REPO, "scripts", "migrate_simulator_keys.py"), encoding="utf-8"
    ).read()
    assert '"sim.carla.rpc_port": "source.carla.port"' in source
