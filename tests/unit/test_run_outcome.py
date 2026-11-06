"""Tests for the application's own verdict reaching the run's record.

What this is for
    The orchestrator knows whether the processes it started exited and whether
    it cleaned up after them. That is not the same as the application having
    done what it was for: a braking client whose ego is removed before it stops
    exits 0 and tears down cleanly. Each client role writes its own verdict
    beside its outputs; these tests pin that the run's record carries it, so a
    reader of run.json alone cannot mistake a clean teardown for a successful
    run.

What it contains
    Cases for the orchestrator's collection of client outcome files, including
    the ones that must not invent a verdict.

How it fits into CADET
    Unit tests over :mod:`cadet.orchestrator.runner`.
"""

import json

from cadet.orchestrator.runner import Orchestrator


def _write(tmp_path, node, payload):
    path = tmp_path / "client_outcome_{}.json".format(node)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return path


def test_each_clients_verdict_is_collected_by_node(tmp_path):
    """Several clients in one run each keep their own verdict.

    The file is per node precisely because a fixed name meant N clients
    overwrote each other, so the collection has to be keyed the same way.
    """
    _write(tmp_path, "client", {"ok": True, "detail": "stopped in time"})
    _write(tmp_path, "sensor_2", {"ok": False, "detail": "no frames arrived"})

    outcomes = Orchestrator._client_outcomes(str(tmp_path))

    assert outcomes == {
        "client": {"ok": True, "detail": "stopped in time"},
        "sensor_2": {"ok": False, "detail": "no frames arrived"},
    }


def test_a_verdict_that_cannot_be_read_is_not_invented(tmp_path):
    """Absent, truncated and malformed files leave the node out.

    A default would say something about the run that nobody recorded. Leaving
    the node out says "nothing was recorded", which is what happened.
    """
    _write(tmp_path, "truncated", "{not json")
    _write(tmp_path, "wrongshape", {"detail": "no ok field"})
    _write(tmp_path, "notadict", [1, 2, 3])
    _write(tmp_path, "good", {"ok": True, "detail": ""})

    outcomes = Orchestrator._client_outcomes(str(tmp_path))

    assert outcomes == {"good": {"ok": True, "detail": ""}}


def test_a_directory_with_no_outcomes_yields_none(tmp_path):
    """A run with no client role has no verdicts, not empty ones."""
    (tmp_path / "run.json").write_text("{}")
    assert Orchestrator._client_outcomes(str(tmp_path)) == {}
    assert Orchestrator._client_outcomes(str(tmp_path / "missing")) == {}


def test_unrelated_files_are_not_read_as_verdicts(tmp_path):
    """The prefix is matched exactly, not as a substring."""
    (tmp_path / "client_outcomes.json").write_text(json.dumps({"ok": False}))
    (tmp_path / "not_client_outcome_x.json").write_text(json.dumps({"ok": False}))
    (tmp_path / "client_outcome_x.txt").write_text(json.dumps({"ok": False}))
    assert Orchestrator._client_outcomes(str(tmp_path)) == {}
