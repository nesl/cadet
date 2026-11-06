"""Tests for the recovery sweep behind ``cadet down``.

The property under test is scope: the sweep must recognise the names CADET
assigns and ignore everything else, because the whole point of replacing the
original's ``fuser -k`` was to stop removing things CADET did not create.
"""

import os

from cadet.orchestrator import recovery
from cadet.orchestrator.state import StateStore


def test_owns_the_names_the_topology_generator_assigns():
    for name in ("n0", "n1", "n42", "bps0"):
        assert recovery.OWNED_NODE_RE.match(name), name


def test_does_not_own_names_it_did_not_assign():
    for name in ("eth0", "docker0", "br-abc123", "mn0", "nat0", "n", "bps1"):
        assert not recovery.OWNED_NODE_RE.match(name), name


def test_owns_its_own_veths_only():
    assert recovery.OWNED_INTF_RE.match("n0-eth0")
    assert recovery.OWNED_INTF_RE.match("bps0-eth1")
    assert not recovery.OWNED_INTF_RE.match("eth0")
    assert not recovery.OWNED_INTF_RE.match("veth1234@if5")


def test_kills_a_recorded_pid_and_clears_the_record(tmp_path, monkeypatch):
    import subprocess
    import sys
    import time

    monkeypatch.chdir(tmp_path)
    store = StateStore()
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    store.add_proc("run-x", str(tmp_path), "client:client", proc.pid)
    assert store.recorded_pids() == [proc.pid]

    found = recovery.sweep(patterns=(), log=lambda *_a: None)

    deadline = time.time() + 10
    while proc.poll() is None and time.time() < deadline:
        time.sleep(0.1)
    assert proc.poll() is not None
    assert any(item.kind == "process" and str(proc.pid) in item.name for item in found)
    assert store.recorded_pids() == []


def test_a_sweep_does_not_kill_a_role_it_has_no_record_of(tmp_path, monkeypatch):
    """A sweep in one checkout must leave a live run in another one alone.

    The store it reads is the working directory's, so a run started elsewhere is
    not in it - and a command line is no substitute, because every CADET role's
    is ``cadet <sub>``. Measured before this: the process step, run from a
    directory whose store had recorded nothing at all, SIGTERMed the server and
    the client of a live run in another checkout. That was the end-to-end
    suite's "server killed by SIGTERM" flake, with another session's matrix
    running ``cadet down`` between its cells.
    """
    import subprocess
    import time

    monkeypatch.chdir(tmp_path)
    store = StateStore()
    assert store.recorded_pids() == []
    # A live process whose command line matches the pattern the sweep is given,
    # started by somebody else: it stands in for the other checkout's role.
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        lines = []
        found = recovery._processes(store, ("sleep 30",), lines.append)
        time.sleep(0.5)  # a kill would land well inside this
        assert proc.poll() is None, "the sweep killed a process it had no record of"
        assert [item.removed for item in found] == [False], [str(i) for i in found]
        assert str(proc.pid) in found[0].name
        assert any("left alone" in line and str(proc.pid) in line for line in lines), lines
    finally:
        proc.kill()
        proc.wait()


def test_removes_only_the_qdisc_interfaces_a_run_recorded(tmp_path, monkeypatch):
    # Scoped by record, not by inspection: an interface nothing recorded is not
    # touched, however its qdisc looks.
    monkeypatch.chdir(tmp_path)
    store = StateStore()
    store.record_netem("run-y", str(tmp_path), ["eth7"])

    calls = []
    monkeypatch.setattr(recovery, "_run", lambda argv: calls.append(argv) or None)

    recovery._qdiscs(store, log=lambda *_a: None)

    assert calls, "the recorded interface was not cleared"
    assert all("eth7" in " ".join(argv) for argv in calls)


def test_a_clean_host_reports_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(recovery, "_processes", lambda *_a: [])
    monkeypatch.setattr(recovery, "_containers", lambda *_a: [])
    monkeypatch.setattr(recovery, "_networks", lambda *_a: [])
    monkeypatch.setattr(recovery, "_emulated_network", lambda *_a: [])
    monkeypatch.setattr(recovery, "_qdiscs", lambda *_a: [])
    assert recovery.sweep(log=lambda *_a: None) == []


def test_a_failed_write_leaves_the_previous_record_readable(tmp_path, monkeypatch):
    """The record file is replaced, never rewritten in place.

    ``owner`` is the only thing that stops a pre-run cleanup killing a live run
    another orchestrator started in the same directory, and a reader that caught
    a half-written file read no runs at all - which makes every recorded pid
    look unowned. A write that fails part-way stands in for that window: the
    previous contents must still be there to read.
    """
    import json

    store = StateStore(root=str(tmp_path))
    store.add_proc("run-a", str(tmp_path), "serve:server", 4242)

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(json, "dump", boom)
    try:
        store.add_proc("run-b", str(tmp_path), "serve:server", 4343)
    except OSError:
        pass
    monkeypatch.undo()

    data = store.load()
    assert "run-a" in data, data
    assert data["run-a"]["owner"] == os.getpid()


def test_state_store_writes_one_pid_file_per_process(tmp_path):
    store = StateStore(root=str(tmp_path))
    store.add_proc("run-z", str(tmp_path), "serve:server", 4242)
    store.add_proc("run-z", str(tmp_path), "client:client", 4243)
    files = sorted(os.listdir(store.pid_dir("run-z")))
    assert files == ["client_client.pid", "serve_server.pid"]
    assert sorted(store.pids_for("run-z")) == [4242, 4243]
    store.clear("run-z")
    assert store.pids_for("run-z") == []


def test_the_container_listing_includes_containers_that_are_not_running(monkeypatch):
    """Without -a the sweep could not see the leftovers it exists to remove.

    `docker container ls` lists only running containers, so a crashed or
    interrupted run - which leaves containers stopped, or created and never
    started - was invisible to `cadet down`. It reported "nothing left to clean
    up" with a stale container sitting there, which is worse than reporting
    nothing at all.
    """
    calls = []

    class _Result:
        returncode = 0
        stdout = "cadet-0123456789-r0-server\n"

    monkeypatch.setattr(recovery, "_run", lambda argv: calls.append(argv) or _Result())
    assert recovery._docker_names("container") == ["cadet-0123456789-r0-server"]
    assert calls[0][:4] == ["docker", "container", "ls", "-a"]


def test_the_network_listing_takes_no_such_flag(monkeypatch):
    """`docker network ls` has no running/stopped distinction to opt into."""
    calls = []

    class _Result:
        returncode = 0
        stdout = ""

    monkeypatch.setattr(recovery, "_run", lambda argv: calls.append(argv) or _Result())
    recovery._docker_names("network")
    assert "-a" not in calls[0]


# --- ownership without the record, and the record without ownership ---------
# The sweep used to treat its own record as the only evidence of ownership there
# is. These are the two ways that was wrong: a recorded pid that is no longer
# the process it recorded, and a process of this checkout that the record lost.


def _tree_of(store):
    """The directory the sweep will attribute a process to."""
    return recovery._owning_tree(store)


def test_a_recorded_pid_whose_process_was_replaced_is_left_alone(tmp_path, monkeypatch):
    """A pid is reused, so a record naming one is not proof it is ours.

    Staged by recording a live process and then rewriting its start time, which
    is exactly what the kernel shows after the number has been handed to
    something else: the pid is in our record and the process behind it is not
    the one we started.
    """
    import subprocess
    import sys
    import time

    monkeypatch.chdir(tmp_path)
    store = StateStore()
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        store.add_proc("run-x", str(tmp_path), "client:client", proc.pid)
        # The start time of a process that is not this one. 1 is the clock tick
        # a process started at boot would carry, and nothing here did.
        pid_file = os.path.join(store.pid_dir("run-x"), "client_client.pid")
        with open(pid_file, "w") as handle:
            handle.write("{} 1".format(proc.pid))

        lines = []
        found = recovery._processes(store, (), lines.append)
        time.sleep(0.5)  # a kill would land well inside this

        assert proc.poll() is None, "the sweep killed a process it only shared a pid with"
        assert found == []
        assert any("pid reuse" in line and str(proc.pid) in line for line in lines), lines
    finally:
        proc.kill()
        proc.wait()


def test_a_recorded_pid_with_no_start_time_keeps_the_weaker_check(tmp_path, monkeypatch):
    """A store written before start times existed is not invalidated by them.

    It gets the check it has always had - the pid alone - rather than being
    read as unverifiable and skipped, which would leave every pre-existing
    record unreapable.
    """
    import subprocess
    import sys
    import time

    monkeypatch.chdir(tmp_path)
    store = StateStore()
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    store.add_proc("run-x", str(tmp_path), "client:client", proc.pid)
    pid_file = os.path.join(store.pid_dir("run-x"), "client_client.pid")
    with open(pid_file, "w") as handle:
        handle.write(str(proc.pid))  # the old format, a bare pid
    assert store.records_for("run-x") == [(proc.pid, None)]

    found = recovery._processes(store, (), log=lambda *_a: None)

    deadline = time.time() + 10
    while proc.poll() is None and time.time() < deadline:
        time.sleep(0.1)
    assert proc.poll() is not None
    assert any(str(proc.pid) in item.name for item in found)


def test_a_role_of_this_tree_is_reaped_after_its_pid_file_is_lost(tmp_path, monkeypatch):
    """The hole that removing the pattern kill opened, and the demonstration owed.

    A role started from this checkout whose record is gone could not be reaped
    at all. Its working directory is the checkout it was started in, which is
    evidence of ownership that needs no record, so it is attributed and killed.
    """
    import subprocess
    import time

    monkeypatch.chdir(tmp_path)
    store = StateStore()
    tree = _tree_of(store)
    proc = subprocess.Popen(["sleep", "30"], cwd=tree, start_new_session=True)
    try:
        store.add_proc("run-x", str(tmp_path), "client:client", proc.pid)
        # The record is lost - a crash before it was written, or a cleared
        # store - and the process keeps running.
        os.remove(os.path.join(store.pid_dir("run-x"), "client_client.pid"))
        assert store.recorded_pids() == []

        found = recovery._processes(store, ("sleep 30",), log=lambda *_a: None)

        deadline = time.time() + 10
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        assert proc.poll() is not None, "a role of this tree was left running"
        assert [item.removed for item in found] == [True], [str(i) for i in found]
        assert "attributed to this tree" in found[0].detail
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def test_a_role_looking_process_in_another_tree_is_still_left_alone(tmp_path, monkeypatch):
    """The attribution must not widen back into the pattern kill it replaced."""
    import subprocess
    import time

    monkeypatch.chdir(tmp_path)
    store = StateStore()
    elsewhere = tmp_path / "another-checkout"
    elsewhere.mkdir()
    proc = subprocess.Popen(["sleep", "30"], cwd=str(elsewhere), start_new_session=True)
    try:
        lines = []
        found = recovery._processes(store, ("sleep 30",), lines.append)
        time.sleep(0.5)

        assert proc.poll() is None, "the sweep killed a process from another checkout"
        assert [item.removed for item in found] == [False], [str(i) for i in found]
        assert any("left alone" in line for line in lines), lines
    finally:
        proc.kill()
        proc.wait()


def test_a_process_whose_directory_cannot_be_read_is_left_alone(tmp_path, monkeypatch):
    """What another account's process looks like from here.

    ``/proc/<pid>/cwd`` is unreadable for a process this account does not own,
    so the attribution comes back empty - and empty must mean "not ours", not
    "no reason to spare it". Patched rather than staged with a second account,
    because the unreadable answer is the whole input.
    """
    import subprocess
    import time

    from cadet.orchestrator import lifecycle

    monkeypatch.chdir(tmp_path)
    store = StateStore()
    proc = subprocess.Popen(["sleep", "30"], cwd=_tree_of(store), start_new_session=True)
    monkeypatch.setattr(lifecycle, "process_cwd", lambda pid: "")
    try:
        lines = []
        found = recovery._processes(store, ("sleep 30",), lines.append)
        time.sleep(0.5)

        assert proc.poll() is None, "a process this account cannot inspect was killed"
        assert [item.removed for item in found] == [False], [str(i) for i in found]
    finally:
        proc.kill()
        proc.wait()


def test_the_pid_file_carries_the_start_time_it_can_be_checked_by(tmp_path, monkeypatch):
    """Read at record time, because after the process exits there is nothing to read."""
    import subprocess
    import sys

    from cadet.orchestrator import lifecycle

    monkeypatch.chdir(tmp_path)
    store = StateStore()
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        store.add_proc("run-x", str(tmp_path), "serve:server", proc.pid)
        assert store.records_for("run-x") == [
            (proc.pid, lifecycle.process_start_ticks(proc.pid))
        ]
        assert store.pids_for("run-x") == [proc.pid]
    finally:
        proc.kill()
        proc.wait()


def test_a_dead_pids_working_directory_is_unknown_not_the_probe_path(tmp_path):
    """The bug `readlink` fixes: `realpath` does not raise for a path that is
    not there, it returns what it was given, so a dead pid came back as
    ``/proc/<pid>/cwd`` - a truthy answer meaning "unknown"."""
    import subprocess
    import sys

    from cadet.orchestrator import lifecycle

    proc = subprocess.Popen([sys.executable, "-c", ""], start_new_session=True)
    proc.wait()

    assert lifecycle.process_cwd(proc.pid) == ""
    assert lifecycle.process_start_ticks(proc.pid) is None
