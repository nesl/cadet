"""End-to-end orchestrator tests: run, sweep, resume, teardown, preflight.

Uses the reference app over a synthetic source (no CARLA). The orchestrator spawns
real `cadet serve` / `cadet client` subprocesses; these tests assert outputs are
produced, sweeps resume, nothing is left running, and preflight fails loudly.
"""

import os
import socket
import time

import pytest

from cadet.orchestrator import Orchestrator, PreflightError, preflight
from cadet.orchestrator.state import pid_alive
from cadet.roles.common import select_run_config

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG = os.path.join(REPO, "examples", "reference", "synthetic.yaml")
APP = "examples.reference.app"


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _sets(tmp_path, port):
    return ["outputs.root={}".format(tmp_path / "out"), "nodes[1].port={}".format(port)]


def _orch():
    # Run the orchestrator from REPO so `python -m cadet` and examples import.
    os.chdir(REPO)
    return Orchestrator(log=lambda m: None)


def test_run_produces_outputs(tmp_path):
    orch = _orch()
    res = orch.run_one(CONFIG, _sets(tmp_path, _free_port()), 0, 0, [APP], [])
    assert res.ok, res.detail
    assert os.path.exists(os.path.join(res.out_dir, "run.json"))
    csvs = [f for f in os.listdir(res.out_dir) if f.endswith("latency_location_log.csv")]
    assert csvs, os.listdir(res.out_dir)


def test_teardown_leaves_no_orphans(tmp_path):
    orch = _orch()
    res = orch.run_one(CONFIG, _sets(tmp_path, _free_port()), 0, 0, [APP], [])
    assert res.ok
    # This run is gone from the state and every process it started is dead.
    # Asserted per run, not over the whole store: the store lives in the working
    # directory and any other orchestrator started there shares it, so an empty
    # store would be a claim about the machine rather than about the teardown.
    recorded = orch.status()
    assert res.run_id not in recorded, recorded
    assert not [pid for pid in orch.state.pids_for(res.run_id) if pid_alive(pid)]


# No pause between cells in these tests: the five-second default is there to
# give a shared simulator time to settle, and there is no simulator here.
_NO_PAUSE = "experiment.inter_run_pause_s=0"


def test_sweep_and_resume(tmp_path):
    orch = _orch()
    sets = _sets(tmp_path, _free_port()) + [_NO_PAUSE]
    results = orch.sweep(CONFIG, sets, [APP], [], reps=2, resume=True)
    assert len(results) == 2 and all(r.ok for r in results)

    # Record marker mtimes, then re-run: completed cells must be skipped (resume),
    # so the markers are not rewritten.
    markers = {}
    for r in results:
        m = os.path.join(r.out_dir, "run.json")
        markers[m] = os.path.getmtime(m)
    time.sleep(0.05)
    again = orch.sweep(CONFIG, sets, [APP], [], reps=2, resume=True)
    assert all(r.ok for r in again)
    assert all(r.detail == "resumed" for r in again)
    for m, mtime in markers.items():
        assert os.path.getmtime(m) == mtime  # untouched -> was skipped


def test_a_sweep_re_runs_completed_cells_by_default(tmp_path):
    # The original re-ran every experiment, so resume is the deviation and has
    # to be asked for. Pinned here because a default that drifts is the failure
    # the fidelity rule exists to prevent.
    orch = _orch()
    sets = _sets(tmp_path, _free_port()) + [_NO_PAUSE]
    first = orch.sweep(CONFIG, sets, [APP], [], reps=1)
    assert len(first) == 1 and first[0].ok
    marker = os.path.join(first[0].out_dir, "run.json")
    before = os.path.getmtime(marker)
    time.sleep(0.05)
    again = orch.sweep(CONFIG, sets, [APP], [], reps=1)
    assert again[0].ok and again[0].detail != "resumed"
    assert os.path.getmtime(marker) > before  # re-run -> rewritten


def test_preflight_port_in_use(tmp_path):
    # Occupy a port, then point the server node at it: preflight must refuse.
    port = _free_port()
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", port))
    s.listen(1)
    try:
        cfg, _ = select_run_config(
            CONFIG, ["outputs.root={}".format(tmp_path / "o"), "nodes[1].port={}".format(port)], 0
        )
        # The refusal must name who holds the port: that is the whole
        # difference from the original, which cleared it with fuser -k and never
        # said whose socket it had closed.
        with pytest.raises(PreflightError, match="held by pid {}".format(os.getpid())):
            preflight(cfg, [])
    finally:
        s.close()


def test_preflight_missing_model(tmp_path):
    # Ask for a model with no registry entry/artifact: preflight must refuse.
    cfg, _ = select_run_config(
        CONFIG,
        [
            "outputs.root={}".format(tmp_path / "o"),
            "models.model=nope",
            "models.registry={nope: {per_device: {cpu: does_not_exist.pt}}}",
        ],
        0,
    )
    with pytest.raises(PreflightError, match="not found"):
        preflight(cfg, [str(tmp_path)])
