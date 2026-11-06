"""Each role's records say which node wrote them.

A run with several clients used to write one `client_outcome.json` and one
`link_report.json` into a shared run directory, so only the last client's
survived; `cadet client --node` existed and was ignored, so no client knew which
node it was. The distributed-sensing example runs two sensors, which makes it
the natural end-to-end check.
"""

import glob
import json
import os

import pytest

from cadet.orchestrator import Orchestrator

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SENSING = os.path.join(REPO, "examples", "distributed_sensing", "quick.yaml")
SENSING_APP = "examples.distributed_sensing.app"


def test_each_client_writes_its_own_records(tmp_path):
    os.chdir(REPO)
    orch = Orchestrator(log=lambda m: None)
    result = orch.run_one(
        SENSING,
        ["outputs.root={}".format(tmp_path / "out")],
        0,
        0,
        [SENSING_APP],
        [],
    )
    assert result.ok, result.detail

    outcomes = sorted(os.path.basename(p) for p in glob.glob(
        os.path.join(result.out_dir, "client_outcome_*.json")))
    reports = sorted(os.path.basename(p) for p in glob.glob(
        os.path.join(result.out_dir, "link_report_*.json")))
    # Two sensors, two of each: the names carry the node, so neither overwrites
    # the other.
    assert len(outcomes) == 2, outcomes
    assert len(reports) == 2, reports
    assert outcomes == sorted(set(outcomes))

    # And each report says whose measurement it is, matching its file name.
    for name in reports:
        with open(os.path.join(result.out_dir, name)) as handle:
            record = json.load(handle)
        assert record["node"] in name, (record["node"], name)

    # The server's per-request metrics are in the run directory now, not one
    # level up in the experiment directory.
    assert glob.glob(os.path.join(result.out_dir, "server_metrics_*.csv")), os.listdir(
        result.out_dir
    )


def test_a_client_can_be_asked_for_by_node_name():
    from cadet.roles.client import build_client

    runtime = build_client(SENSING, ["outputs.root=/tmp/cadet-node-check"], 0, 0,
                           logger=lambda m: None, node_name="sensor_1")
    assert runtime.node_name == "sensor_1"


def test_an_unknown_client_node_is_refused():
    from cadet.roles.client import build_client

    with pytest.raises(RuntimeError) as err:
        build_client(SENSING, ["outputs.root=/tmp/cadet-node-check"], 0, 0,
                     logger=lambda m: None, node_name="not_a_node")
    assert "no client node named" in str(err.value)


def test_asking_for_a_server_node_as_a_client_is_refused():
    from cadet.roles.client import build_client

    with pytest.raises(RuntimeError):
        build_client(SENSING, ["outputs.root=/tmp/cadet-node-check"], 0, 0,
                     logger=lambda m: None, node_name="gateway")
