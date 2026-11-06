"""Shared pytest fixtures/paths for the CADET test suite.

Exposes the braking example config (the framework's reference AV example) and
ensures its plugin is registered for tests that resolve app params. Framework
tests use it as a sample config; nothing in cadet/ imports examples/.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BRAKING_CONFIG = REPO_ROOT / "examples" / "braking" / "flat" / "openloop_braking.yaml"

# The braking example is the sample config for framework-level tests.
EXAMPLE_CONFIGS = [BRAKING_CONFIG]

# Registering the braking plugin once makes its params_model available to tests
# that call resolve_app_params / get_app("braking").
import examples.braking.flat.app  # noqa: E402,F401


@pytest.fixture
def braking_config_path():
    """Path to the braking example config."""
    return str(BRAKING_CONFIG)


@pytest.fixture
def experimental_launcher(monkeypatch):
    """Make one launcher value experimental for one test, whatever its real tier.

    The opt-in machinery needs *something* in the experimental tier to
    exercise. This fixture has now been rewritten three times, because each
    stand-in it named was promoted out from under it: ``ssh_docker``, then
    ``ssh_venv``, each time breaking tests for a reason unrelated to what they
    test. The pattern is clear enough to design against rather than repeat.

    The failure mode was always the same. :func:`tier` reads
    :data:`IMPLEMENTED` first, so a value promoted into it is classified from
    there and patching the experimental map alone stops having any effect. So
    this patches **both**: the value is removed from implemented and added to
    experimental, for the duration of one test.

    That makes the fixture independent of every real tier, which is the
    property it should always have had - the tests that use it are about the
    gate's behaviour given an experimental value, not about which option
    happens to be unverified this week. ``ssh_venv`` stays the value only
    because it is a real value of a real option, so the messages under test
    read as they would in anger.
    """
    from cadet.config import implemented

    value = "ssh_venv"
    implemented_without = {
        path: (values - {value}) if path == "nodes.launcher" else values
        for path, values in implemented.IMPLEMENTED.items()
    }
    monkeypatch.setattr(implemented, "IMPLEMENTED", implemented_without)
    monkeypatch.setattr(implemented, "EXPERIMENTAL", {"nodes.launcher": {value}})
    return value


@pytest.fixture(autouse=True)
def isolated_state_store(tmp_path_factory, monkeypatch):
    """Give every test its own run-state store.

    The store defaults to the working directory, so without this every test in
    a suite run shares one - and a test that sweeps leftovers can reap a
    process another test is still using. That is not hypothetical: a canary
    failed about one run in six with its server SIGTERMed, and the same shape
    turned out to be reachable from inside one suite as well as from another
    checkout.

    Role subprocesses inherit the variable, so a run records into its own
    store rather than the repository's.
    """
    root = tmp_path_factory.mktemp("state")
    monkeypatch.setenv("CADET_STATE_ROOT", str(root))
    return root
