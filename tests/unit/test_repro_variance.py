"""Unit tests for the reproducibility reduction (:mod:`cadet.repro`).

What this is for:
    Pins the reduction's behaviour on synthetic run directories, including the
    awkward cases the harness exists for: a metric missing from some
    repetitions (a finding, never averaged over), a single repetition (no
    spread to report), and a bimodal set (whose median describes neither mode).

What it contains:
    - helpers that write a synthetic repetition directory holding the files a
      real run writes (run marker, per-frame CSV, app summary sidecar, link
      report, client outcome, server per-request CSV, a collector CSV, and the
      resolved config);
    - tests for discovery and grouping, the statistics, the missing-metric
      finding, the single-repetition case, bimodality, the measurement-context
      caveats (warm-up, Nagle versus the send interval, context that differs
      between repetitions), and the report renderings.

How it fits into CADET:
    Part of the unit suite (``python -m pytest -q tests/``). No simulator, no
    network and no run: every input is a directory this file writes.

How to configure it:
    Nothing to configure; ``tmp_path`` provides the trees.
"""

import json
import os

from cadet.repro import discover_repetitions, read_repetition, reduce_tree, render_markdown
from cadet.repro.extract import parse_run_id

FRAME_HEADER = (
    "frame_id,frame_capture_time,frame_capture_loc,frame_send_time,frame_send_loc,"
    "server_processing_time,server_response_receive_time,server_response_loc,brake_issued,"
    "stopping_time,stopping_loc,vehicle_speed_mph,run_id,rep,brake_reaction_latency_s\n"
)

CONFIG = {
    "experiment": {"name": "v2v_braking", "app": "v2v", "repetitions": 20, "seed": 0},
    "app": {"plugin": "v2v"},
    "models": {"warmup": False},
    "source": {"kind": "carla", "fps": 10, "scenario": {"name": "S_1", "ego_speed": "20mph"}},
    "channels": [{"name": "frames", "stream_fps": 15}],
    "nodes": [
        {"name": "client", "role": "client"},
        {"name": "server", "role": "server", "extra_processing_ms": 20.0},
    ],
}


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)


def write_rep(
    root,
    rep,
    base="abc123",
    status="complete",
    brake_t=2.1,
    stop_y=40.0,
    frames=3,
    latency_p50=61.0,
    summary=None,
    with_link=True,
    with_server=True,
    config=None,
    collector=None,
):
    """Write one synthetic repetition directory and return its path."""
    run_id = "{}-r{}".format(base, rep)
    path = os.path.join(str(root), run_id)
    os.makedirs(path, exist_ok=True)
    _write(
        os.path.join(path, "run.json"),
        json.dumps(
            {
                "run_id": run_id,
                "status": status,
                "detail": "ok" if status == "complete" else "no_output",
                "reason": "" if status == "complete" else "no_output",
                "startup_s": 4.0 + rep * 0.1,
                "active_s": 20.0,
                "teardown_s": 0.3,
            }
        ),
    )
    _write(os.path.join(path, "config.yaml"), json.dumps(config or CONFIG))
    rows = [FRAME_HEADER]
    for i in range(frames):
        braked = i == frames - 1
        rows.append(
            "{fid},100.{fid},\"(-1.0, 85.0, 0.0)\",0.{fid},\"(-1.0, 84.0, 0.0)\","
            "{proc},{resp},\"(-1.0, {by}, 0.0)\",{braked},{stop_t},\"(-1.0, {sy}, 0.0)\","
            "{speed},{rid},{rep},{lat}\n".format(
                fid=i,
                proc="0.013" if braked else "",
                resp=brake_t if braked else "",
                by=50.0 if braked else 0.0,
                braked="True" if braked else "False",
                stop_t=3.4 if braked else "",
                sy=stop_y if braked else 0.0,
                speed=35.0 + i,
                rid=run_id,
                rep=rep,
                lat="1.4" if braked else "",
            )
        )
    _write(os.path.join(path, "{}_latency_location_log.csv".format(run_id)), "".join(rows))
    if summary is not None:
        _write(
            os.path.join(path, "{}_latency_location_log_v2v_summary.json".format(run_id)),
            json.dumps(summary),
        )
    if with_link:
        _write(
            os.path.join(path, "link_report_client.json"),
            json.dumps(
                {
                    "node": "client",
                    "configured_links": [{"name": "l", "p50_ms": 60.0}],
                    "configured_routes": [],
                    "measured": {
                        "reply_policy": "on_brake",
                        "messages_sent": 100,
                        "messages_received": 1,
                        "unanswered": 99,
                        "latency_ms": {"samples": 1, "p50": latency_p50, "p99": latency_p50},
                    },
                }
            ),
        )
    _write(
        os.path.join(path, "client_outcome_client.json"),
        json.dumps({"ok": status == "complete", "detail": "braked"}),
    )
    if with_server:
        _write(
            os.path.join(path, "server_metrics_server.csv"),
            "client_id,queue_wait_s,inference_s,queue_depth_at_enqueue,stale_drops_so_far,"
            "replied,inference_start_wall,avg_power_w,energy_j,power_samples\n"
            "c,0.001,1.2,0,0,False,,,,\n"
            "c,0.002,0.013,1,0,True,,,,\n"
            "c,0.002,0.014,0,0,False,,,,\n",
        )
    if collector is not None:
        _write(
            os.path.join(path, "gpu_server.csv"),
            "t_wall,t_monotonic,node,source,power_w\n1,1,server,nvml,{}\n2,2,server,nvml,{}\n".format(
                collector, collector + 2.0
            ),
        )
    return path


def _metric(cell, name):
    for metric in cell["metrics"]:
        if metric["metric"] == name:
            return metric
    raise AssertionError("no metric {} in {}".format(name, [m["metric"] for m in cell["metrics"]]))


def test_parse_run_id_splits_cell_and_repetition():
    assert parse_run_id("abc123-r7") == ("abc123", 7)
    # A hand-started run has no repetition suffix and is its own cell.
    assert parse_run_id("abc123") == ("abc123", None)


def test_discovery_finds_only_repetition_directories(tmp_path):
    write_rep(tmp_path, 0)
    write_rep(tmp_path, 1)
    os.makedirs(os.path.join(str(tmp_path), "abc123-r0", "logs"), exist_ok=True)
    _write(os.path.join(str(tmp_path), "notes.txt"), "not a run")
    found = discover_repetitions(str(tmp_path))
    assert [os.path.basename(p) for p in found] == ["abc123-r0", "abc123-r1"]


def test_reduces_twenty_repetitions_with_per_repetition_values(tmp_path):
    for rep in range(20):
        write_rep(tmp_path, rep, brake_t=2.0 + rep * 0.01, stop_y=40.0 + rep)
    report = reduce_tree(str(tmp_path))
    assert report["repetition_dirs"] == 20
    assert len(report["cells"]) == 1
    cell = report["cells"][0]
    assert cell["repetitions"] == 20
    assert cell["statuses"] == {"complete": 20}
    stop_y = _metric(cell, "frames.stop_y")
    assert stop_y["count"] == 20
    assert stop_y["missing_from"] == []
    assert stop_y["median"] == 49.5
    assert stop_y["iqr"] == 9.5
    assert stop_y["min"] == 40.0 and stop_y["max"] == 59.0
    # Per-repetition values travel with the summary, one per repetition.
    assert len(stop_y["per_repetition"]) == 20
    assert stop_y["per_repetition"]["r3"] == 43.0


def test_two_cells_are_grouped_separately(tmp_path):
    write_rep(tmp_path, 0, base="aaa")
    write_rep(tmp_path, 1, base="aaa")
    write_rep(tmp_path, 0, base="bbb")
    report = reduce_tree(str(tmp_path))
    assert [cell["cell"] for cell in report["cells"]] == ["aaa", "bbb"]
    assert [cell["repetitions"] for cell in report["cells"]] == [2, 1]


def test_missing_metric_is_a_finding_not_an_average(tmp_path):
    # Three of five repetitions wrote no link report at all.
    for rep in range(5):
        write_rep(tmp_path, rep, with_link=rep < 2)
    cell = reduce_tree(str(tmp_path))["cells"][0]
    p50 = _metric(cell, "link.client.latency_ms.p50")
    assert p50["count"] == 2
    assert sorted(p50["missing_from"]) == ["r2", "r3", "r4"]
    assert any(
        "link.client.latency_ms.p50 present in 2 of 5" in finding for finding in cell["findings"]
    )
    assert any("no link report" in finding for finding in cell["findings"])


def test_single_repetition_reports_no_spread(tmp_path):
    write_rep(tmp_path, 0)
    cell = reduce_tree(str(tmp_path))["cells"][0]
    stop_y = _metric(cell, "frames.stop_y")
    assert cell["repetitions"] == 1
    assert stop_y["count"] == 1
    assert stop_y["median"] == 40.0
    # One value has no quartiles and no deviation; reporting zero would claim
    # a tightness that was never measured.
    assert stop_y["iqr"] is None
    assert stop_y["q1"] is None
    assert stop_y["stdev"] is None
    assert stop_y["bimodal"] is False


def test_bimodal_set_is_flagged_and_both_modes_described(tmp_path):
    # Ten runs near 40 m and ten near 60 m: the median sits in the empty middle.
    for rep in range(20):
        write_rep(tmp_path, rep, stop_y=40.0 if rep < 10 else 60.0)
    cell = reduce_tree(str(tmp_path))["cells"][0]
    stop_y = _metric(cell, "frames.stop_y")
    assert stop_y["bimodal"] is True
    assert stop_y["modes"]["low"] == {"n": 10, "median": 40.0}
    assert stop_y["modes"]["high"] == {"n": 10, "median": 60.0}
    assert stop_y["median"] == 50.0
    assert any("is bimodal" in finding for finding in cell["findings"])


def test_tight_set_is_not_called_bimodal(tmp_path):
    for rep in range(20):
        write_rep(tmp_path, rep, stop_y=40.0 + rep * 0.1)
    cell = reduce_tree(str(tmp_path))["cells"][0]
    assert _metric(cell, "frames.stop_y")["bimodal"] is False


def test_failed_repetition_is_named_not_hidden(tmp_path):
    write_rep(tmp_path, 0)
    write_rep(tmp_path, 1, status="failed")
    cell = reduce_tree(str(tmp_path))["cells"][0]
    assert cell["statuses"] == {"complete": 1, "failed": 1}
    assert any("1 of 2 repetitions did not complete" in f for f in cell["findings"])
    assert any("r1: failed (no_output)" == f for f in cell["findings"])


def test_server_inference_keeps_median_and_warmup_max(tmp_path):
    for rep in range(4):
        write_rep(tmp_path, rep)
    cell = reduce_tree(str(tmp_path))["cells"][0]
    # The run's first inference (1.2 s, model start-up) is the max; the steady
    # state is the median. Both survive the reduction.
    assert _metric(cell, "server.server.inference_s.median")["median"] == 0.014
    assert _metric(cell, "server.server.inference_s.max")["median"] == 1.2
    assert _metric(cell, "server.server.requests")["median"] == 3.0
    assert any("models.warmup is false" in finding for finding in cell["findings"])


def test_send_interval_and_nagle_caveat_are_stated(tmp_path):
    for rep in range(3):
        write_rep(tmp_path, rep)
    cell = reduce_tree(str(tmp_path))["cells"][0]
    # 15 fps = 66.7 ms, inside twice the configured 60 ms one-way p50, with
    # tcp_nodelay unset (the original's default) on both nodes.
    assert cell["context"]["send_interval_ms"] == {"frames": 66.667}
    assert cell["context"]["configured_p50_ms"] == {"client": [60.0]}
    assert any("plus an extra round trip" in finding for finding in cell["findings"])
    assert any("send interval beside every latency" in finding for finding in cell["findings"])


def test_nagle_caveat_absent_when_paced_clear_of_the_boundary(tmp_path):
    config = json.loads(json.dumps(CONFIG))
    config["channels"] = [{"name": "frames", "stream_fps": 5}]  # 200 ms
    for rep in range(3):
        write_rep(tmp_path, rep, config=config)
    cell = reduce_tree(str(tmp_path))["cells"][0]
    assert cell["context"]["send_interval_ms"] == {"frames": 200.0}
    assert not any("plus an extra round trip" in finding for finding in cell["findings"])


def test_context_that_differs_between_repetitions_is_a_finding(tmp_path):
    other = json.loads(json.dumps(CONFIG))
    other["source"]["scenario"]["ego_speed"] = "40mph"
    write_rep(tmp_path, 0)
    write_rep(tmp_path, 1, config=other)
    cell = reduce_tree(str(tmp_path))["cells"][0]
    assert "varies" in cell["context"]["ego_speed"]
    assert any("context ego_speed differs between repetitions" in f for f in cell["findings"])


def test_app_summary_sidecar_metrics_are_reduced(tmp_path):
    for rep in range(4):
        write_rep(
            tmp_path,
            rep,
            summary={
                "leader_brake_t": 13.5 + rep * 0.01,
                "min_gap_m": -57.0 - rep,
                "loop_rate_hz": 80.0,
                "braking_mode": "v2v",
                "trigger_reason": None,
            },
        )
    cell = reduce_tree(str(tmp_path))["cells"][0]
    assert _metric(cell, "summary.min_gap_m")["median"] == -58.5
    assert _metric(cell, "summary.loop_rate_hz")["iqr"] == 0.0
    # A string stays context, not a metric; a null is simply absent.
    assert cell["context"]["summary"] == {"braking_mode": "v2v"}
    assert not any(m["metric"] == "summary.trigger_reason" for m in cell["metrics"])


def test_collector_csv_present_in_some_repetitions_only(tmp_path):
    for rep in range(4):
        write_rep(tmp_path, rep, collector=70.0 if rep < 3 else None)
    cell = reduce_tree(str(tmp_path))["cells"][0]
    samples = _metric(cell, "collector.gpu.server.samples")
    assert samples["count"] == 3 and samples["missing_from"] == ["r3"]
    assert _metric(cell, "collector.gpu.server.power_w.median")["median"] == 71.0


def test_repetition_without_a_marker_is_not_a_repetition(tmp_path):
    path = write_rep(tmp_path, 0)
    os.remove(os.path.join(path, "run.json"))
    assert discover_repetitions(str(tmp_path)) == []


def test_unreadable_files_are_recorded_not_raised(tmp_path):
    path = write_rep(tmp_path, 0)
    _write(os.path.join(path, "link_report_client.json"), "{not json")
    _write(os.path.join(path, "config.yaml"), "[1, 2]")
    item = read_repetition(path)
    assert "link_report_client unreadable" in item.notes
    assert any("config.yaml is not a mapping" in note for note in item.notes)


def test_note_common_to_every_repetition_is_said_once(tmp_path):
    # No server CSV anywhere: one line, not twenty. The uneven case keeps its
    # labels, so the three repetitions that differ stay findable.
    for rep in range(5):
        write_rep(tmp_path, rep, with_server=False, with_link=rep < 2)
    cell = reduce_tree(str(tmp_path))["cells"][0]
    assert ["all 5 repetitions: no server per-request CSV"] == [
        f for f in cell["findings"] if "server per-request CSV" in f
    ]
    assert ["r2, r3, r4: no link report"] == [
        f for f in cell["findings"] if "no link report" in f
    ]


def test_markdown_report_carries_every_repetition_column(tmp_path):
    for rep in range(3):
        write_rep(tmp_path, rep, stop_y=40.0 + rep)
    text = render_markdown(reduce_tree(str(tmp_path)))
    assert "# Variance over repetitions" in text
    assert "| metric | r0 | r1 | r2 |" in text
    assert "`frames.stop_y`" in text
    assert "Per-repetition values" in text
    assert "Reduction environment" in text
