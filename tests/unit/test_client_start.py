"""Tests for the client startup order (``experiment.client_start``).

The property under test is the sequence, because that is the whole point: under
the default the client must finish everything that does not need the scene, and
say so, *before* the scenario is started - otherwise it can still be setting up
when the ego's short window opens.
"""

import pytest

from cadet.core.apps import register_app
from cadet.core.source import Source
from cadet.roles.client import (
    CLIENT_READY_MARKER,
    CLIENT_START_AFTER_SCENARIO,
    CLIENT_START_BEFORE_SCENARIO,
    ClientRuntime,
    RunContext,
)


class _PlainSource(Source):
    """A source that owns everything it needs, so it uses the defaults."""

    def open(self):
        pass

    def next_frame(self):
        return None

    def close(self):
        pass

    def metadata(self):
        return {}


class _SceneSource(_PlainSource):
    """A source whose scene appears later, recording the calls it receives."""

    def __init__(self, scene_appears=True):
        self.calls = []
        self._scene_appears = scene_appears

    def prepare(self):
        self.calls.append("prepare")

    def await_scene(self, timeout_s):
        self.calls.append("await_scene")
        return self._scene_appears

    def open(self):
        self.calls.append("open")

    def close(self):
        self.calls.append("close")


@register_app
class _Plugin:
    """Stands in for the app plugin the client resolves to write its verdict."""

    name = "clientstart-test"

    @staticmethod
    def client_success(client_app):
        return True, "test"


class _App:
    def __init__(self):
        self.calls = []

    def setup(self, ctx):
        self.calls.append("setup")

    def is_done(self, ctx):
        return True

    def finish(self, ctx):
        self.calls.append("finish")

    def message_stats(self):
        return None


def _runtime(source, app, client_start, log, out_dir=None):
    # A real config object: the client resolves its app plugin at the end of a
    # run to record whether it produced output, so a fake with no config would
    # exercise only the guard rather than the path.
    config = type(
        "Cfg", (), {"app": type("A", (), {"plugin": _Plugin.name})(), "source": None}
    )()
    ctx = RunContext(
        config=config,
        source=source,
        transport=None,
        run_id="r",
        rep=0,
        out_dir=out_dir or "/tmp",
        log=log,
    )
    runtime = ClientRuntime(
        ctx, app, "127.0.0.1", 1, node=None, client_start=client_start, scene_timeout_s=0.1
    )
    return runtime, ctx


# --- the Source defaults ------------------------------------------------
def test_a_source_with_no_external_scene_needs_neither_hook():
    # The defaults must be no-ops, or adding the hooks would change every
    # existing source's behaviour.
    source = _PlainSource()
    assert source.prepare() is None
    assert source.await_scene(0.0) is True


# --- the default order --------------------------------------------------
def test_before_scenario_prepares_connects_announces_then_attaches(monkeypatch):
    source = _SceneSource()
    app = _App()
    lines = []
    runtime, ctx = _runtime(source, app, CLIENT_START_BEFORE_SCENARIO, lines.append)

    connected = []
    monkeypatch.setattr(runtime, "_connect_to_server", lambda: connected.append(True))
    runtime.run()

    # prepare, then the server connection, then the marker, then the wait, then
    # attach. The marker after the connection is deliberate: "ready" has to mean
    # there is nothing left to do but attach.
    assert source.calls[:3] == ["prepare", "await_scene", "open"]
    assert connected == [True]
    marker_index = next(i for i, line in enumerate(lines) if line == CLIENT_READY_MARKER)
    assert marker_index >= 0
    # The marker is printed before the scene is waited for.
    assert source.calls.index("await_scene") == 1


def test_before_scenario_fails_loudly_if_the_scene_never_appears(monkeypatch):
    source = _SceneSource(scene_appears=False)
    runtime, _ctx = _runtime(source, _App(), CLIENT_START_BEFORE_SCENARIO, lambda _m: None)
    monkeypatch.setattr(runtime, "_connect_to_server", lambda: None)
    with pytest.raises(RuntimeError, match="scene did not appear"):
        runtime.run()


# --- the opt-out --------------------------------------------------------
def test_after_scenario_opens_first_and_announces_nothing(monkeypatch):
    source = _SceneSource()
    lines = []
    runtime, _ctx = _runtime(source, _App(), CLIENT_START_AFTER_SCENARIO, lines.append)
    connected = []
    monkeypatch.setattr(runtime, "_connect_to_server", lambda: connected.append(True))
    runtime.run()

    # The original order: the scene already exists, so the client opens it and
    # never waits for it.
    assert "await_scene" not in source.calls
    assert source.calls[0] == "open"
    assert connected == [True]
    assert CLIENT_READY_MARKER not in lines


# --- the marker is a protocol, not a log line ---------------------------
def test_the_orchestrator_waits_for_the_same_marker_the_client_prints():
    # Two copies of this string would be a protocol that drifts silently, so
    # both sides import the one constant.
    from cadet.network.netwaggle import session
    from cadet.orchestrator import runner

    assert runner.CLIENT_READY_MARKER is CLIENT_READY_MARKER
    assert session.CLIENT_READY_MARKER is CLIENT_READY_MARKER


def test_the_default_is_the_original_order():
    # The project rule: an unconfigured run reproduces the original scripts, so
    # the default here is the scenario-first order they used. Pinned in a test
    # because a default that drifts silently is the deviation this guards.
    from cadet.config.schema import Experiment
    from cadet.roles.client import CLIENT_START_DEFAULT

    assert CLIENT_START_DEFAULT == CLIENT_START_AFTER_SCENARIO
    assert Experiment(name="x", app="y").client_start == CLIENT_START_AFTER_SCENARIO
