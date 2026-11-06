"""Tests for the command-line entry point's own behaviour.

Covers what ``cadet``'s argument handling and error reporting do, as distinct
from the configuration, orchestration and metrics code the subcommands call:
each test here drives :func:`cadet.cli.main` with an argument list and reads
what it printed and what it returned. Nothing here needs a run directory, a
container or a simulator.
"""

from cadet import cli


def test_a_missing_base_is_named_rather_than_the_config_that_extends_it(tmp_path, capsys):
    """The file that cannot be opened is often the base, not the named config.

    A config naming a base that is not there used to be reported as the config
    itself being not found, which sends the reader to a file that is there.
    """
    config = tmp_path / "child.yaml"
    config.write_text("extends: missing_base.yaml\nexperiment: {name: x}\n")

    assert cli.main(["validate", str(config)]) == 2
    message = capsys.readouterr().err
    assert "missing_base.yaml" in message
    assert "child.yaml" in message


def test_a_config_that_is_itself_absent_is_named_once(tmp_path, capsys):
    """With nothing extending anything, the named file is the missing one."""
    missing = tmp_path / "nowhere.yaml"

    assert cli.main(["validate", str(missing)]) == 2
    message = capsys.readouterr().err
    assert str(missing) in message
    assert "extended by" not in message


def test_emulate_loads_the_stage_modules_it_is_given(tmp_path, monkeypatch, capsys):
    """`cadet emulate --stage-module X` accepted the flag and did nothing with it.

    The emulated path has to resolve a `pipeline` block's stages in this process
    and tell the roles it starts which modules to import. Neither happened, so a
    staged pipeline in an emulated network died on the first stage its config
    named.
    """
    import os

    monkeypatch.delenv("CADET_STAGE_MODULES", raising=False)
    config = tmp_path / "c.yaml"
    config.write_text("experiment: {name: x}\n")

    # The command fails on the config, which is not what this is about: what
    # matters is that by then the modules have been published and imported.
    cli.main(["emulate", str(config), "--stage-module", "cadet.pipeline.stages.braking"])
    capsys.readouterr()

    assert os.environ.get("CADET_STAGE_MODULES") == "cadet.pipeline.stages.braking"
    from cadet.pipeline.registry import get_stage

    assert get_stage("camera") is not None
