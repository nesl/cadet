"""Tests for how the acceptance matrix finds the machine a cell runs on.

What this is for
    A distributed cell places a node on a second machine. Its address belongs
    to whoever runs the matrix, not to CADET, so the spec carries the
    placeholder ``edge-host`` and the address is supplied by the environment or
    by an untracked local file. These tests pin that substitution, because when
    it goes wrong the cell does not say so: it fails on whatever the resolved
    config happened to become.

What it contains
    Cases for the override paths that are emitted, and for the cells that must
    not be touched at all.

How it fits into CADET
    Unit tests over ``scripts/acceptance/run_matrix.py``, which is a script
    rather than a package module and so is loaded by path.
"""

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _harness():
    name = "run_matrix_under_test"
    if name in sys.modules:
        return sys.modules[name]
    path = REPO / "scripts" / "acceptance" / "run_matrix.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _cell(config):
    return {"id": "under.test", "config": config}


def test_a_remote_node_is_addressed_by_index_not_by_name(tmp_path):
    """An override path indexes a list by number only.

    A quoted name is read as a mapping key, so `nodes['server'].host=...`
    replaces the whole node with just the fields named and drops its name and
    role. The config then fails pydantic validation for extra inputs, which
    says nothing about the real mistake.
    """
    rm = _harness()
    config = tmp_path / "cell.yaml"
    config.write_text(
        "nodes:\n"
        "  - {name: client, host: localhost, role: client}\n"
        "  - {name: server, host: %s, role: server}\n" % rm.PLACEHOLDER_SECOND_MACHINE
    )
    cell = _cell(str(config))

    overrides = rm.second_machine_overrides(cell, {"host": "198.51.100.7"})

    assert overrides == ["nodes[1].host=198.51.100.7"]
    assert not any("'server'" in o for o in overrides)


def test_only_the_fields_that_were_supplied_are_set(tmp_path):
    """A field nobody named keeps the spec's value.

    The account and the key are only needed where they differ from the
    defaults, so supplying an address alone must not blank them.
    """
    rm = _harness()
    config = tmp_path / "cell.yaml"
    config.write_text(
        "nodes:\n  - {name: server, host: %s, role: server}\n" % rm.PLACEHOLDER_SECOND_MACHINE
    )
    cell = _cell(str(config))

    overrides = rm.second_machine_overrides(cell, {"host": "198.51.100.7", "user": "cadet"})

    assert overrides == ["nodes[0].host=198.51.100.7", "nodes[0].user=cadet"]


def test_a_single_host_cell_is_left_alone(tmp_path):
    """Nothing is overridden for a cell that places nothing remotely.

    The setting being present in the environment must not change a cell that
    does not need a second machine, or every single-host cell's run id would
    move with it.
    """
    rm = _harness()
    config = tmp_path / "cell.yaml"
    config.write_text(
        "nodes:\n"
        "  - {name: client, host: localhost, role: client}\n"
        "  - {name: server, host: 127.0.0.1, role: server}\n"
    )
    cell = _cell(str(config))

    assert rm.second_machine_overrides(cell, {"host": "198.51.100.7"}) == []
    assert rm.missing_second_machine(cell) is False


def test_a_cell_needing_a_machine_is_recognised(tmp_path):
    """So that it can be reported blocked rather than failed."""
    rm = _harness()
    config = tmp_path / "cell.yaml"
    config.write_text(
        "nodes:\n  - {name: server, host: %s, role: server}\n" % rm.PLACEHOLDER_SECOND_MACHINE
    )
    cell = _cell(str(config))

    assert rm.missing_second_machine(cell) is True
    assert rm.second_machine_overrides(cell, {}) == []


def test_the_shipped_distributed_cells_resolve_with_a_supplied_address():
    """The real spec, substituted, still validates.

    This is the case the unit above cannot catch: the override has to produce a
    config the schema accepts, with the node's own identity intact.
    """
    rm = _harness()
    cell = _cell("tests/acceptance/cells/synth_std_delay_dist.yaml")

    overrides = rm.second_machine_overrides(cell, {"host": "198.51.100.7", "user": "cadet"})
    cfg, _run_dir, _rid = rm.resolve_cell(cell, "/tmp/acceptance-under-test", overrides)

    server = [n for n in cfg.nodes if n.name == "server"][0]
    assert server.host == "198.51.100.7"
    assert server.user == "cadet"
    assert server.role == "server"
    assert server.launcher == "ssh_docker"
