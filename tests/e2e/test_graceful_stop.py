"""Tests for a role that is stopped rather than finishing by itself.

The orchestrator stops a role with SIGTERM and kills it after a grace, and
Python's default action for SIGTERM ends the process at once - so a server that
waits for work until it is stopped never reached its own shutdown path and never
wrote its per-request metrics. These run a real `cadet serve` subprocess and
signal it, because the defect lived in the signal path and an in-process test
would not have seen it.

The exit status is part of the contract and is asserted here too: the server
writes its records and then hands the process back to the signal, because the
orchestrator reads a worker's code during the active phase to tell a guard that
finished its job from one that died under it.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG = os.path.join(REPO, "examples", "reference", "synthetic.yaml")
APP = "examples.reference.app"


def _free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _serve(tmp_path, port):
    """Start a real server role and return (process, output root)."""
    root = str(tmp_path / "out")
    argv = [
        # The interpreter running the tests, not whatever "python" resolves to:
        # these four cases failed as "server exited early: No module named
        # 'yaml'" on a machine whose PATH python was not the one CADET is
        # installed in, which is a question about the host and not about a
        # graceful stop.
        sys.executable,
        "-m",
        "cadet",
        "serve",
        CONFIG,
        "--app-module",
        APP,
        "--set",
        "outputs.root={}".format(root),
        "--set",
        "nodes[1].port={}".format(port),
    ]
    proc = subprocess.Popen(
        argv, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    # Wait for it to be listening, which is also the readiness marker the
    # orchestrator waits for.
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return proc, root
        except OSError:
            if proc.poll() is not None:
                raise AssertionError("server exited early: {}".format(proc.stdout.read()))
            time.sleep(0.1)
    proc.kill()
    raise AssertionError("server never listened")


def _metrics_csv(root):
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.startswith("server_metrics") and name.endswith(".csv"):
                return os.path.join(dirpath, name)
    return None


def test_a_stopped_server_writes_its_records(tmp_path):
    proc, root = _serve(tmp_path, _free_port())
    proc.send_signal(signal.SIGTERM)
    # The drain is bounded at STOP_DRAIN_S, so this is generous but finite; the
    # launcher's own grace is five seconds.
    proc.wait(timeout=10)
    out = proc.stdout.read()
    # Signalled, not a clean exit: -15 is how a SIGTERMed process reports.
    assert proc.returncode == -signal.SIGTERM, (proc.returncode, out)
    assert "finishing" in out, out
    csv_path = _metrics_csv(root)
    assert csv_path is not None, os.listdir(root) if os.path.exists(root) else out
    with open(csv_path) as handle:
        header = handle.readline()
    assert "client_id" in header, header


def test_a_stopped_server_exits_well_inside_the_grace(tmp_path):
    proc, _root = _serve(tmp_path, _free_port())
    start = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=10)
    elapsed = time.monotonic() - start
    proc.stdout.read()
    # The launcher escalates to SIGKILL after five seconds; a server that needed
    # longer than that would lose exactly what this change exists to keep.
    assert elapsed < 3.0, elapsed


def test_an_interrupted_server_also_writes_and_reports_the_signal(tmp_path):
    """SIGINT takes the same path, so Ctrl-C does not lose a server's records."""
    proc, root = _serve(tmp_path, _free_port())
    proc.send_signal(signal.SIGINT)
    proc.wait(timeout=10)
    proc.stdout.read()
    assert proc.returncode == -signal.SIGINT, proc.returncode
    assert _metrics_csv(root) is not None


def test_the_run_record_of_a_server_only_config_is_not_claimed(tmp_path):
    """A stopped server writes metrics, not a run record: that is the orchestrator's."""
    proc, root = _serve(tmp_path, _free_port())
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=10)
    proc.stdout.read()
    for dirpath, _dirs, files in os.walk(root):
        assert "run.json" not in files, dirpath
    # And nothing claims a client outcome either.
    assert not any(
        name.startswith("client_outcome")
        for _d, _s, names in os.walk(root)
        for name in names
    )
    json.dumps({})  # keep the import honest
