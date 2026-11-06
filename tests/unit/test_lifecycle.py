"""Tests for the run-lifecycle steps ported from the original orchestrators.

Covers the readiness barrier (including the case the original handled badly, a
process that dies before printing its marker), the config mapping for the
cleanup targets, the run limit, and the reasons a run can end.
"""

import os
import socket
import subprocess
import sys
import time

import pytest

from cadet.orchestrator import lifecycle


class _Node:
    def __init__(self, name, role, port=None):
        self.name = name
        self.role = role
        self.port = port


class _Config:
    def __init__(self, nodes, max_active_s=None, scenario_timeout_s=None, max_startup_s=None):
        self.nodes = nodes
        self.experiment = type(
            "E", (), {"max_active_s": max_active_s, "max_startup_s": max_startup_s}
        )()
        scenario = type("S", (), {"timeout_s": scenario_timeout_s})()
        self.source = type("Src", (), {"scenario": scenario})()


class _Spec:
    def __init__(self, argv):
        self.argv = argv


class _FakeProc:
    """A process whose exit is controlled by the test."""

    def __init__(self, code=None):
        self._code = code

    def poll(self):
        return self._code

    def exit(self, code=0):
        self._code = code


# --- the readiness barrier ----------------------------------------------
def test_wait_for_line_finds_a_marker_already_written(tmp_path):
    path = tmp_path / "sr.log"
    path.write_text("starting\nScenarioManager: Running scenario X\ndone\n")
    with open(path) as fh:
        assert lifecycle.wait_for_line(fh, "Running scenario", timeout=1.0)


def test_wait_for_line_times_out_without_the_marker(tmp_path):
    path = tmp_path / "sr.log"
    path.write_text("starting\n")
    with open(path) as fh:
        started = time.time()
        assert not lifecycle.wait_for_line(fh, "never appears", timeout=0.3)
        assert time.time() - started >= 0.3


def test_wait_for_marker_returns_early_when_the_process_dies(tmp_path):
    # The original sat in readline for the full timeout after ScenarioRunner had
    # already exited; this is the part that is not a straight port.
    path = tmp_path / "sr.log"
    path.write_text("booting\n")
    proc = _FakeProc(code=1)
    started = time.time()
    assert not lifecycle.wait_for_marker(str(path), "Running scenario", proc=proc, timeout=10.0)
    assert time.time() - started < 2.0


def test_wait_for_marker_sees_a_marker_written_just_before_exit(tmp_path):
    path = tmp_path / "sr.log"
    path.write_text("booting\nRunning scenario now\n")
    assert lifecycle.wait_for_marker(
        str(path), "Running scenario", proc=_FakeProc(code=0), timeout=1.0
    )


def test_wait_for_marker_gives_up_on_a_missing_log(tmp_path):
    missing = str(tmp_path / "absent.log")
    assert not lifecycle.wait_for_marker(missing, "anything", timeout=0.3)


def test_wait_for_marker_waits_for_a_late_marker(tmp_path):
    path = tmp_path / "late.log"
    path.write_text("booting\n")
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import time,sys; time.sleep(0.4); open(sys.argv[1],'a').write('ready now\\n'); "
         "time.sleep(2)",
         str(path)],
    )
    try:
        assert lifecycle.wait_for_marker(str(path), "ready now", proc=proc, timeout=5.0)
    finally:
        proc.kill()
        proc.wait()


# --- cleanup targets come from the config -------------------------------
def test_orphan_patterns_cover_roles_and_the_sim_program():
    config = _Config([_Node("c", "client"), _Node("s", "server", port=10100)])
    patterns = lifecycle.orphan_patterns(
        config, [_Spec(["python3", "/opt/scenario_runner/scenario_runner.py", "--host", "x"])]
    )
    assert "cadet serve" in patterns
    assert "cadet client" in patterns
    assert "scenario_runner.py" in patterns


def test_orphan_patterns_without_hooks_are_just_the_roles():
    config = _Config([_Node("c", "client")])
    assert lifecycle.orphan_patterns(config, []) == ["cadet serve", "cadet client", "cadet load"]


def test_listen_ports_are_the_listening_nodes():
    config = _Config(
        [
            _Node("c", "client", port=1),
            _Node("s", "server", port=10100),
            _Node("x", "sim", port=2000),
        ]
    )
    assert sorted(lifecycle.listen_ports(config)) == [2000, 10100]


def test_kill_orphans_never_raises_on_missing_tools():
    # "nothing to kill" is the normal case and must not fail a run.
    lifecycle.kill_orphans(["definitely-not-running-xyz"], [], log=lambda *_: None)


# --- the port-release wait ----------------------------------------------
def test_port_free_sees_a_bound_port():
    held = socket.socket()
    held.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    held.bind(("127.0.0.1", 0))
    held.listen(1)
    port = held.getsockname()[1]
    try:
        assert not lifecycle.port_free(port)
    finally:
        held.close()
    assert lifecycle.wait_for_ports_free([port], timeout_s=2.0, log=lambda *_: None)


def test_wait_for_ports_free_gives_up_on_a_port_nothing_released():
    # Something CADET did not start is holding it; say so rather than hang.
    held = socket.socket()
    held.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    held.bind(("127.0.0.1", 0))
    held.listen(1)
    port = held.getsockname()[1]
    messages = []
    try:
        started = time.time()
        assert not lifecycle.wait_for_ports_free([port], timeout_s=0.5, log=messages.append)
        assert time.time() - started >= 0.5
    finally:
        held.close()
    assert any("still held" in m for m in messages)


def test_wait_for_ports_free_is_trivially_true_with_no_ports():
    assert lifecycle.wait_for_ports_free([], log=lambda *_: None)


# --- the run limit ------------------------------------------------------
def test_run_limit_prefers_the_explicit_setting():
    assert lifecycle.active_limit_s(_Config([], max_active_s=12.0)) == 12.0


def test_run_limit_derives_from_the_scenario_timeout():
    config = _Config([], scenario_timeout_s=100.0)
    assert lifecycle.active_limit_s(config) == 100.0 + lifecycle.ACTIVE_LIMIT_MARGIN_S


def test_run_limit_falls_back_when_nothing_is_set():
    assert lifecycle.active_limit_s(_Config([])) == lifecycle.DEFAULT_ACTIVE_LIMIT_S


# --- why a run ended ----------------------------------------------------
def test_workers_exiting_cleanly_is_the_only_ok_outcome():
    workers = [("client:c", _FakeProc(code=0))]
    outcome = lifecycle.wait_for_workers(workers, 5.0, [], log=lambda *_: None)
    assert outcome.ok
    assert outcome.reason == lifecycle.REASON_OK


def test_a_nonzero_worker_exit_is_not_ok():
    outcome = lifecycle.wait_for_workers(
        [("client:c", _FakeProc(code=1))], 5.0, [], log=lambda *_: None
    )
    assert not outcome.ok
    assert outcome.reason == lifecycle.REASON_OK


def test_active_limit_stops_a_worker_that_never_finishes():
    outcome = lifecycle.wait_for_workers(
        [("client:c", _FakeProc(code=None))], 0.5, [], log=lambda *_: None
    )
    assert not outcome.ok
    assert outcome.reason == lifecycle.REASON_ACTIVE_LIMIT
    assert "0s" in outcome.detail or "within" in outcome.detail


def test_a_dead_guard_ends_the_run_with_its_name():
    guard = _FakeProc(code=3)
    outcome = lifecycle.wait_for_workers(
        [("client:c", _FakeProc(code=None))], 10.0, [("scenario_runner", guard)],
        log=lambda *_: None,
    )
    assert not outcome.ok
    assert outcome.reason == lifecycle.REASON_NODE_EXITED
    assert "scenario_runner" in outcome.detail


def test_a_failed_guard_beats_workers_that_exited_cleanly():
    # A client usually exits 0 when its server disappears, so a run whose
    # server died mid-stream would otherwise be reported as a success.
    outcome = lifecycle.wait_for_workers(
        [("client:c", _FakeProc(code=0))],
        10.0,
        [("serve:s", _FakeProc(code=-9))],
        log=lambda *_: None,
    )
    assert not outcome.ok
    assert outcome.reason == lifecycle.REASON_NODE_EXITED
    assert "serve:s" in outcome.detail


def test_a_guard_exiting_zero_is_a_normal_end():
    # The braking server is single-shot: it serves one client and returns, so
    # its clean exit must not be reported as a fault.
    outcome = lifecycle.wait_for_workers(
        [("client:c", _FakeProc(code=0))],
        10.0,
        [("serve:s", _FakeProc(code=0))],
        log=lambda *_: None,
    )
    assert outcome.ok
    assert outcome.reason == lifecycle.REASON_OK


def test_a_guard_that_finishes_mid_run_is_recorded():
    # ScenarioRunner can exit once its scenario is running: the client drives
    # the ego directly from then on, so the run completes. That it happened
    # still has to show up in the record.
    worker = _FakeProc(code=None)
    guard = _FakeProc(code=0)
    import threading

    def finish():
        time.sleep(0.3)
        worker.exit(0)

    threading.Thread(target=finish, daemon=True).start()
    outcome = lifecycle.wait_for_workers(
        [("client:c", worker)], 10.0, [("scenario_runner", guard)], log=lambda *_: None
    )
    assert outcome.ok
    assert "scenario_runner" in outcome.detail


def test_a_guard_exiting_zero_early_falls_through_to_the_run_limit():
    outcome = lifecycle.wait_for_workers(
        [("client:c", _FakeProc(code=None))],
        0.5,
        [("serve:s", _FakeProc(code=0))],
        log=lambda *_: None,
    )
    assert outcome.reason == lifecycle.REASON_ACTIVE_LIMIT


def test_outcome_serialises_its_reason():
    outcome = lifecycle.RunOutcome([0], lifecycle.REASON_OK, "all good")
    assert outcome.as_dict()["reason"] == lifecycle.REASON_OK
    assert outcome.as_dict()["exit_codes"] == [0]


# --- the client report echo ---------------------------------------------
def test_final_client_report_is_echoed_from_the_marker_onwards(tmp_path):
    path = tmp_path / "client.log"
    path.write_text("noise\n[client] ==== FINAL REPORT ====\n[client] frames_sent=32\n")
    lines = []
    lifecycle.print_final_client_report(str(path), lines.append)
    assert any("frames_sent=32" in line for line in lines)
    assert not any("noise" in line for line in lines)


def test_final_client_report_tolerates_a_missing_log(tmp_path):
    lifecycle.print_final_client_report(str(tmp_path / "gone.log"), lambda *_: None)

# --- the run-end rule (experiment.end_on) -------------------------------
def _end_config(end_on, grace_s=0.0):
    config = _Config([])
    config.experiment = type(
        "E", (), {"max_active_s": 5.0, "end_on": end_on, "end_grace_s": grace_s}
    )()
    return config


def test_end_rule_defaults_to_the_client_report():
    # An experiment block with neither field set behaves as the original did.
    config = _Config([])
    assert lifecycle.end_rule(config) == (lifecycle.END_ON_CLIENT_REPORT, 0.0)


def test_end_rule_reads_both_fields():
    assert lifecycle.end_rule(_end_config("first", 2.5)) == ("first", 2.5)


def test_client_report_ignores_a_sim_hook_that_finished_cleanly():
    # This is the original's rule: the scenario ending is not the run ending.
    hook = _FakeProc(code=0)
    worker = _FakeProc()
    workers = [("client:c", worker)]
    hooks = [("scenario_runner", hook)]

    def _finish_worker():
        worker.exit(0)

    import threading

    threading.Timer(0.4, _finish_worker).start()
    outcome = lifecycle.wait_for_workers(
        workers,
        5.0,
        guards=hooks,
        log=lambda *_: None,
        sim_hooks=hooks,
        end_on=lifecycle.END_ON_CLIENT_REPORT,
    )
    assert outcome.reason == lifecycle.REASON_OK
    assert outcome.ok
    # The departure is still recorded, even though it did not end the run.
    assert "scenario_runner" in outcome.detail


def test_sim_hook_exit_ends_the_run_after_the_grace_period():
    hook = _FakeProc(code=0)
    worker = _FakeProc()  # never finishes on its own
    started = time.monotonic()
    outcome = lifecycle.wait_for_workers(
        [("client:c", worker)],
        10.0,
        guards=[("scenario_runner", hook)],
        log=lambda *_: None,
        sim_hooks=[("scenario_runner", hook)],
        end_on=lifecycle.END_ON_SIM_HOOK_EXIT,
        grace_s=0.5,
    )
    elapsed = time.monotonic() - started
    assert outcome.reason == lifecycle.REASON_SIM_HOOK_EXIT
    # Ended on the hook, not on the 10 s limit, and not before the grace.
    assert 0.5 <= elapsed < 5.0
    # The workers were stopped on purpose, so no exit code is a failure here.
    assert outcome.ok


def test_sim_hook_exit_waits_for_the_hook_even_when_the_client_finishes_first():
    # Under this rule the scenario defines the run, so a client that returns
    # early does not end it; the limit is what stops this one.
    hook = _FakeProc()  # still running
    outcome = lifecycle.wait_for_workers(
        [("client:c", _FakeProc(code=0))],
        0.5,
        guards=[("scenario_runner", hook)],
        log=lambda *_: None,
        sim_hooks=[("scenario_runner", hook)],
        end_on=lifecycle.END_ON_SIM_HOOK_EXIT,
    )
    assert outcome.reason == lifecycle.REASON_ACTIVE_LIMIT


def test_first_ends_on_whichever_comes_first():
    hook = _FakeProc(code=0)
    outcome = lifecycle.wait_for_workers(
        [("client:c", _FakeProc())],
        10.0,
        guards=[("scenario_runner", hook)],
        log=lambda *_: None,
        sim_hooks=[("scenario_runner", hook)],
        end_on=lifecycle.END_ON_FIRST,
        grace_s=0.0,
    )
    assert outcome.reason == lifecycle.REASON_SIM_HOOK_EXIT


def test_a_failed_sim_hook_still_fails_the_run_under_every_rule():
    # A killed ScenarioRunner is not an end-of-run, it is a broken run, and the
    # end_on rule does not soften that.
    for end_on in lifecycle.END_ON_VALUES:
        hook = _FakeProc(code=-9)
        outcome = lifecycle.wait_for_workers(
            [("client:c", _FakeProc())],
            5.0,
            guards=[("scenario_runner", hook)],
            log=lambda *_: None,
            sim_hooks=[("scenario_runner", hook)],
            end_on=end_on,
        )
        assert outcome.reason == lifecycle.REASON_NODE_EXITED, end_on
        assert not outcome.ok


# --- the signal guard ----------------------------------------------------
def test_sigterm_raises_instead_of_ending_the_process():
    import signal as signal_module

    restore = lifecycle.install_interrupt_guard()
    try:
        with pytest.raises(lifecycle.RunInterrupted):
            os.kill(os.getpid(), signal_module.SIGTERM)
    finally:
        restore()


def test_the_guard_restores_the_previous_handler():
    import signal as signal_module

    before = signal_module.getsignal(signal_module.SIGTERM)
    lifecycle.install_interrupt_guard()()
    assert signal_module.getsignal(signal_module.SIGTERM) is before


# --- cleanup scoped to CADET's own processes ----------------------------
def test_port_holders_identifies_this_process():
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    probe.bind(("127.0.0.1", 0))
    probe.listen(1)
    port = probe.getsockname()[1]
    try:
        holders = lifecycle.port_holders(port)
        assert os.getpid() in [pid for pid, _cmd in holders]
        assert "pid {}".format(os.getpid()) in lifecycle.describe_port_holders(port)
    finally:
        probe.close()


def test_describe_port_holders_on_a_free_port():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    assert "no listening process" in lifecycle.describe_port_holders(port)


def test_ownership_comes_from_the_record_and_nothing_else():
    # A recorded pid is owned whatever its command line says.
    assert lifecycle.is_cadet_owned(os.getpid(), owned=[os.getpid()])
    # An unrecorded pid is not, however much its command line looks like ours.
    # This is the whole point: two runs on one machine have identical command
    # lines, so a cleanup that trusted them killed the other run's server.
    assert not lifecycle.is_cadet_owned(os.getpid())
    assert lifecycle.looks_like_a_role(os.getpid(), ("python",))
    assert not lifecycle.looks_like_a_role(os.getpid(), ("cadet serve",))
    # A pid that does not exist is nobody's.
    assert not lifecycle.looks_like_a_role(999999, ("python",))


def test_cleanup_does_not_kill_a_role_it_did_not_record():
    # A live process whose command line matches a role pattern, started by
    # someone else: it must survive the cleanup and be named in the log.
    import subprocess

    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        lines = []
        lifecycle.kill_orphans(["sleep 30"], [], log=lines.append)
        assert proc.poll() is None, "the cleanup killed a process it did not start"
        assert any("left alone" in line and str(proc.pid) in line for line in lines), lines
    finally:
        proc.kill()
        proc.wait()


def test_cleanup_leaves_a_foreign_port_holder_alone_and_names_it():
    # The decision this encodes: the original cleared the port with fuser -k,
    # which would have closed this socket. Here it is reported, not killed.
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    probe.bind(("127.0.0.1", 0))
    probe.listen(1)
    port = probe.getsockname()[1]
    try:
        foreign = lifecycle.kill_orphans(
            ["a-pattern-that-matches-nothing-at-all"], [port], log=lambda *_: None
        )
        assert [pid for _port, pid, _cmd in foreign] == [os.getpid()]
        # Still listening: nothing killed it.
        assert not lifecycle.port_free(port)
    finally:
        probe.close()

# --- the startup limit and the phase record ------------------------------
def test_startup_limit_defaults_to_the_sum_of_the_readiness_timeouts():
    # The sum is the phase's own bound taken one step at a time, so the default
    # can never refuse a startup the individual waits would have allowed.
    assert lifecycle.startup_limit_s(_Config([]), [60.0, 180.0]) == 240.0


def test_startup_limit_is_explicit_when_the_config_says_so():
    config = _Config([], max_startup_s=30.0)
    assert lifecycle.startup_limit_s(config, [60.0, 180.0]) == 30.0


def test_startup_limit_falls_back_when_there_is_nothing_to_wait_on():
    assert lifecycle.startup_limit_s(_Config([]), []) == lifecycle.DEFAULT_STARTUP_LIMIT_S


def test_a_readiness_wait_is_cut_to_what_is_left_of_the_budget():
    deadline = time.monotonic() + 2.0
    # The step would wait 60 s; only 2 s of the phase's budget remain.
    assert lifecycle.remaining_startup_s(deadline, 60.0, "serve:server") <= 2.0
    # A step that wants less than the remainder keeps its own timeout.
    assert lifecycle.remaining_startup_s(deadline, 0.5, "serve:server") == 0.5


def test_a_spent_budget_raises_rather_than_waiting_again():
    with pytest.raises(lifecycle.StartupLimitExceeded):
        lifecycle.remaining_startup_s(time.monotonic() - 1.0, 60.0, "serve:server")


def test_timing_records_each_phase_and_the_limit():
    timing = lifecycle.RunTiming()
    assert timing.as_dict() == {
        "startup_s": None,
        "active_s": None,
        "teardown_s": None,
        "limit": None,
    }
    timing.startup_s = 10.238
    timing.active_s = 20.0
    timing.teardown_s = 0.512
    timing.limit = lifecycle.LIMIT_ACTIVE
    assert timing.as_dict() == {
        "startup_s": 10.24,
        "active_s": 20.0,
        "teardown_s": 0.51,
        "limit": "active",
    }
    assert "limit=active" in str(timing)


def test_a_reused_pid_is_not_killed_just_because_it_was_recorded(monkeypatch):
    """Pids are reused, so a record names a number and not a process.

    Found by a flake rather than by reasoning: the route equivalence test failed
    about one run in five with the server SIGTERMed, while other streams ran
    their suites on the same machine. A stale record pointed at a number the
    kernel had given to somebody else's role.
    """
    from cadet.orchestrator import lifecycle

    patterns = ["cadet serve", "cadet client"]
    monkeypatch.setattr(lifecycle, "process_cmdline", lambda pid: {
        4242: "/usr/bin/python -m cadet serve config.yaml",
        4243: "/usr/bin/vim notes.txt",
    }.get(pid, ""))

    # Recorded and still a role: ours.
    assert lifecycle.is_cadet_owned(4242, owned=[4242], patterns=patterns) is True
    # Recorded, but the pid now belongs to something else entirely: not ours.
    assert lifecycle.is_cadet_owned(4243, owned=[4243], patterns=patterns) is False
    # Not recorded at all: never ours, whatever it looks like.
    assert lifecycle.is_cadet_owned(4242, owned=[], patterns=patterns) is False
    # With no patterns the record alone decides, which is what `cadet down`
    # wants: sweeping everything recorded is the point there.
    assert lifecycle.is_cadet_owned(4243, owned=[4243]) is True
