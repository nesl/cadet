"""Tests for the rule that a run producing nothing fails.

The case this exists for: a client that set up, sent nothing and exited cleanly
used to be indistinguishable from one that ran the experiment, because success
was decided by exit codes and exiting zero having done nothing is still exiting
zero.
"""

import json
import os

from cadet.core.apps import AppPlugin
from cadet.metrics.link_report import MessageStats
from cadet.orchestrator import lifecycle
from cadet.roles.client import client_outcome_file


class _Client:
    """An app client reporting whatever statistics the test wants."""

    def __init__(self, sent=0, received=0):
        self._stats = MessageStats(sent=sent, received=received)

    def message_stats(self):
        return self._stats


class _Silent:
    """An app client that reports no statistics at all."""


class _NoneStats:
    def message_stats(self):
        return None


# --- the default condition ----------------------------------------------
def test_the_default_is_one_message_sent():
    ok, detail = AppPlugin.client_success(_Client(sent=1))
    assert ok
    assert "sent=1" in detail


def test_the_default_fails_a_client_that_sent_nothing():
    ok, detail = AppPlugin.client_success(_Client(sent=0))
    assert not ok
    assert "no messages" in detail


def test_the_default_does_not_require_a_reply():
    # The weakest claim that still means something happened. An app needing a
    # reply says so itself.
    ok, _detail = AppPlugin.client_success(_Client(sent=5, received=0))
    assert ok


def test_an_uninstrumented_app_is_not_failed_for_it():
    # Failing a run because the app reports no statistics would punish the lack
    # of instrumentation, not the lack of output.
    assert AppPlugin.client_success(_Silent())[0] is True
    assert AppPlugin.client_success(_NoneStats())[0] is True


# --- the app-declared conditions ----------------------------------------
def test_braking_requires_a_brake_decision():
    from examples.braking.flat.app import BrakingApp

    ok, detail = BrakingApp.client_success(_Client(sent=30, received=0))
    assert not ok
    assert "no brake decision" in detail
    assert BrakingApp.client_success(_Client(sent=30, received=1))[0] is True


def test_braking_fails_with_no_frames_at_all():
    from examples.braking.flat.app import BrakingApp

    ok, detail = BrakingApp.client_success(_Client(sent=0))
    assert not ok
    assert "no frames" in detail


def test_reference_requires_a_reply():
    from examples.reference.app import ReferenceApp

    ok, detail = ReferenceApp.client_success(_Client(sent=10, received=0))
    assert not ok
    assert "no reply" in detail
    assert ReferenceApp.client_success(_Client(sent=10, received=10))[0] is True


# --- the orchestrator's side -------------------------------------------
def test_a_failing_verdict_is_reported(tmp_path):
    with open(os.path.join(str(tmp_path), client_outcome_file("client")), "w") as handle:
        json.dump({"ok": False, "detail": "client transmitted no frames"}, handle)
    failures = lifecycle.client_output_failures(str(tmp_path), log=lambda *_a: None)
    assert failures == ["client transmitted no frames"]


def test_a_passing_verdict_reports_nothing(tmp_path):
    with open(os.path.join(str(tmp_path), client_outcome_file("client")), "w") as handle:
        json.dump({"ok": True, "detail": "sent=30"}, handle)
    assert lifecycle.client_output_failures(str(tmp_path), log=lambda *_a: None) == []


def test_a_missing_verdict_is_not_a_failure(tmp_path):
    # A client that never got far enough to write one has already failed the run
    # for a reason that is recorded; inventing a second one would obscure it.
    assert lifecycle.client_output_failures(str(tmp_path), log=lambda *_a: None) == []


def test_an_unreadable_verdict_is_not_a_failure(tmp_path):
    path = os.path.join(str(tmp_path), client_outcome_file("client"))
    with open(path, "w") as handle:
        handle.write("{not json")
    lines = []
    assert lifecycle.client_output_failures(str(tmp_path), log=lines.append) == []
    assert any("could not read" in line for line in lines)


def test_the_reason_exists_and_is_distinct():
    assert lifecycle.REASON_NO_OUTPUT == "no output"
    assert lifecycle.REASON_NO_OUTPUT != lifecycle.REASON_OK
