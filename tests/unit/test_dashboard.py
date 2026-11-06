"""Tests for the dashboard's readers, the grouping rule and the plot emitter.

Covers: run discovery (including a run in progress and the non-run directories
that sit beside runs), the per-node file names and the legacy unsuffixed ones,
every degradation path a missing or malformed file takes, the placement x
network grouping and its crossover series, the device-series shape the metric
collectors write, and that the emitted pages are self-contained - no CDN, no
remotely fetched script, font or stylesheet.

The fixture tree is written here rather than copied from a real output tree, so
the tests pin the file names and column shapes the readers depend on instead of
whatever one machine happened to produce.
"""

import json
import os

import pytest
import yaml

from cadet.dashboard import device, render, runs, svg
from cadet.dashboard.sweep import group_runs, network_of, placement_of, swept_value

# Anything that would make the page need the network. A page containing one of
# these is the failure the whole brief is arranged around, so it is asserted on
# rather than reviewed by eye.
REMOTE_MARKERS = ("http://", "https://", "//cdn", "<script", "@import", "url(")


def _config(placement=None, delay_ms=None, stages=None, links=True):
    """A resolved config of the shape the orchestrator saves into a run directory."""
    network = None
    if delay_ms is not None:
        network = {
            "mode": "delay",
            "medium": "wired",
            "distribution": {"kind": "normal", "mean_ms": delay_ms, "std_ms": 1.0},
            "loss": 0.0,
            "bandwidth_mbps": None,
        }
    data = {
        "experiment": {"name": "fixture", "seed": 0, "repetitions": 2},
        "source": {"kind": "synthetic", "fps": 10},
        "app": {"plugin": "fixture", "params": {}},
        "nodes": [
            {"name": "edge", "role": "client", "device": "cpu", "host": "localhost"},
            {"name": "hub", "role": "server", "device": "gpu", "host": "localhost", "port": 9000},
        ],
        "links": (
            [{"name": "uplink", "from": "edge", "to": "hub", "network": network}]
            if links and network
            else []
        ),
        "routes": [],
        "channels": [{"name": "frames", "from": "edge", "to": "hub", "stream_fps": 10}],
        "transport": {"protocol": "tcp", "mode": "frame", "encoding": "safe"},
        "metrics": {"collectors": []},
        "pipeline": {"stages": stages} if stages else None,
        "sweep": {"axes": {}},
    }
    if placement is not None:
        data["app"]["params"]["placement"] = placement
    return data


def _write_run(root, run_id, status="complete", config=None, **files):
    """Write one run directory. ``status=None`` leaves out run.json (in progress)."""
    path = os.path.join(root, run_id)
    os.makedirs(path)
    if status is not None:
        with open(os.path.join(path, "run.json"), "w") as handle:
            json.dump(
                {
                    "run_id": run_id,
                    "status": status,
                    "reason": "workers exited",
                    "detail": "exit codes [0]",
                    "startup_s": 1.5,
                    "active_s": 30.0,
                    "teardown_s": 2.0,
                },
                handle,
            )
    with open(os.path.join(path, "config.yaml"), "w") as handle:
        yaml.safe_dump(config if config is not None else _config(delay_ms=10.0), handle)
    for name, body in files.items():
        name = name.replace("__", ".")
        with open(os.path.join(path, name), "w") as handle:
            handle.write(body)
    return path


FRAME_HEADER = (
    "frame_id,frame_capture_time,frame_send_time,server_processing_time,"
    "server_response_receive_time,brake_issued,stopping_time,stopping_loc,"
    "vehicle_speed_mph,brake_reaction_latency_s\n"
)


def _frame_csv(latency=0.25, triggered=True):
    rows = [FRAME_HEADER]
    for index in range(3):
        rows.append(
            "{i},100.{i},100.{i},0.03,101.{i},{t},102.0,\"(1.0, 2.5, 0.0)\",20.0,{lat}\n".format(
                i=index, t="True" if triggered and index == 1 else "False", lat=latency
            )
        )
    return "".join(rows)


def _link_report(p50=12.5, sent=100, received=100):
    return json.dumps(
        {
            "experiment": "fixture",
            "app": "fixture",
            "node": "edge",
            "configured_links": [{"name": "uplink", "mode": "delay"}],
            "configured_routes": [],
            "measured": {
                "reply_policy": "on-event",
                "messages_sent": sent,
                "messages_received": received,
                "unanswered": sent - received,
                "unanswered_pct": 0.0,
                "latency_ms": {
                    "samples": received,
                    "p50": p50,
                    "p99": p50 * 2,
                    "mean": p50,
                    "min": p50 / 2,
                    "max": p50 * 3,
                },
            },
        }
    )


SERVER_HEADER = (
    "client_id,queue_wait_s,inference_s,queue_depth_at_enqueue,"
    "stale_drops_so_far,replied,inference_start_wall\n"
)


def _server_csv(n=4):
    rows = [SERVER_HEADER]
    for index in range(n):
        rows.append("0,0.000{},0.02{},0,0,True,100.{}\n".format(index, index, index))
    return "".join(rows)


# The three clock columns every collector writes, then that collector's own.
# Pinned here because the readers key on them.
GPU_HEADER = (
    "t_wall,t_monotonic,node,source,gpu_index,util_gpu_pct,util_mem_pct,"
    "mem_used_mb,mem_total_mb,power_w\n"
)
CPU_HEADER = (
    "t_wall,t_monotonic,node,source,cpu_pct_host,cpu_count,mem_pct_host,"
    "mem_used_mb_host,cpu_pct_process,rss_mb_process,threads_process,children_process\n"
)
ENERGY_HEADER = (
    "t_wall,t_monotonic,node,source,power_w,interval_s,energy_j,energy_j_total,"
    "rail_gpu_mw,rail_cpu_mw,rail_soc_mw,activity_index\n"
)


def _gpu_csv(samples=5, devices=2):
    rows = [GPU_HEADER]
    for index in range(samples):
        for gpu in range(devices):
            rows.append(
                "1000.{i},{mono}.0,hub,nvml,{g},{util},10,2048,8192,{power}\n".format(
                    i=index, mono=500 + index, g=gpu, util=20 + index * 5 + gpu, power=60 + gpu
                )
            )
    return "".join(rows)


def _cpu_csv(samples=4):
    rows = [CPU_HEADER]
    for index in range(samples):
        rows.append(
            "1000.{i},{mono}.0,hub,psutil,{host},8,40.0,3000,{proc},512,12,0\n".format(
                i=index, mono=500 + index, host=10 + index, proc=5 + index
            )
        )
    return "".join(rows)


def _energy_csv(samples=3):
    rows = [ENERGY_HEADER]
    total = 0.0
    for index in range(samples):
        total += 1.5
        rows.append(
            "1000.{i},{mono}.0,hub,smi,75.0,0.02,1.5,{total},,,,\n".format(
                i=index, mono=500 + index, total=total
            )
        )
    return "".join(rows)


@pytest.fixture
def tree(tmp_path):
    """A two-placement, two-condition sweep plus the awkward cases beside it.

    Deliberately not uniform: one cell is missing (a hole in the matrix), one run
    is in progress, one run has no per-node file at all, one uses the legacy
    unsuffixed names, and two non-run directories sit among the runs.
    """
    root = str(tmp_path / "outputs")
    os.makedirs(root)
    for rep in (0, 1):
        _write_run(
            root,
            "aaaa1111-r{}".format(rep),
            config=_config(placement="edge", delay_ms=10.0),
            **{
                "fixture_latency_location_log__csv": _frame_csv(latency=0.25 + rep * 0.01),
                "link_report_edge__json": _link_report(p50=12.0 + rep),
                "client_outcome_edge__json": json.dumps({"ok": True, "detail": "sent=100"}),
                "server_metrics_hub__csv": _server_csv(),
                "gpu_hub__csv": _gpu_csv(),
                "cpu_hub__csv": _cpu_csv(),
                "energy_hub__csv": _energy_csv(),
            }
        )
    _write_run(
        root,
        "bbbb2222-r0",
        config=_config(placement="edge", delay_ms=50.0),
        **{
            "fixture_latency_location_log__csv": _frame_csv(latency=0.40),
            "link_report_edge__json": _link_report(p50=52.0),
        }
    )
    _write_run(
        root,
        "cccc3333-r0",
        config=_config(placement="hub", delay_ms=10.0),
        **{
            "fixture_latency_location_log__csv": _frame_csv(latency=0.31),
            "link_report_edge__json": _link_report(p50=11.0),
        }
    )
    # The hole: placement hub at 50 ms was never run.
    # A run in progress: no run.json yet, which is what the live view polls for.
    _write_run(root, "dddd4444-r0", status=None, config=_config(placement="hub", delay_ms=99.0))
    # Legacy unsuffixed per-node names, as a tree written before the rename has.
    _write_run(
        root,
        "eeee5555-r0",
        config=_config(placement="edge", delay_ms=10.0),
        **{"link_report__json": _link_report(p50=9.0), "client_outcome__json": '{"ok": true}'}
    )
    # Not runs, and they sit right beside the runs: a frame sink and a log dir.
    os.makedirs(os.path.join(root, "server_detections"))
    with open(os.path.join(root, "server_detections", "frame_00000.jpg"), "w") as handle:
        handle.write("not a run")
    os.makedirs(os.path.join(root, "aaaa1111-r0", "logs"))
    with open(os.path.join(root, "aaaa1111-r0", "logs", "server_hub.log"), "w") as handle:
        handle.write("log line")
    return root


# ---------------------------------------------------------------------------
# discovery


def test_discovery_finds_runs_and_skips_what_is_not_a_run(tree):
    found = [os.path.basename(p) for p in runs.discover_runs(tree)]
    assert "aaaa1111-r0" in found
    assert "server_detections" not in found
    assert "logs" not in found
    assert len(found) == 6


def test_discovery_includes_a_run_with_no_marker(tree):
    found = [os.path.basename(p) for p in runs.discover_runs(tree)]
    assert "dddd4444-r0" in found


def test_discovery_of_a_single_run_directory_is_that_run(tree):
    one = os.path.join(tree, "aaaa1111-r0")
    assert runs.discover_runs(one) == [one]


def test_discovery_of_a_missing_root_is_empty_not_an_error(tmp_path):
    assert runs.discover_runs(str(tmp_path / "nope")) == []


def test_parse_run_id_splits_the_repetition():
    assert runs.parse_run_id("aaaa1111-r3") == ("aaaa1111", 3)
    assert runs.parse_run_id("handstarted") == ("handstarted", None)


# ---------------------------------------------------------------------------
# reading one run


def test_read_run_reads_every_file(tree):
    record = runs.read_run(os.path.join(tree, "aaaa1111-r0"))
    assert record.status == "complete"
    assert not record.in_progress
    assert record.experiment == "fixture"
    assert [node["name"] for node in record.nodes] == ["edge", "hub"]
    assert "edge" in record.link_reports
    assert record.outcomes["edge"]["ok"] is True
    assert record.server["hub"]["requests"] == 4
    assert record.frames["rows"] == 3
    assert record.frames["triggered"] is True
    assert record.task_metric()[1] == pytest.approx(0.25)
    assert record.end_to_end_ms() == pytest.approx(12.0)


def test_a_run_with_no_marker_reads_as_in_progress(tree):
    record = runs.read_run(os.path.join(tree, "dddd4444-r0"))
    assert record.in_progress
    assert any("teardown" in note for note in record.notes)


def test_legacy_unsuffixed_per_node_files_are_still_read(tree):
    record = runs.read_run(os.path.join(tree, "eeee5555-r0"))
    assert list(record.link_reports) == [runs.LEGACY_NODE]
    assert record.end_to_end_ms() == pytest.approx(9.0)


def test_a_run_missing_every_per_node_file_degrades_to_notes(tmp_path):
    root = str(tmp_path / "out")
    os.makedirs(root)
    _write_run(root, "ffff6666-r0")
    record = runs.read_run(os.path.join(root, "ffff6666-r0"))
    assert record.link_reports == {}
    assert record.outcomes == {}
    assert record.server == {}
    assert record.task_metric() == ("task metric", None, "")
    assert record.end_to_end_ms() is None
    for expected in ("no link report", "no client outcome", "no server", "no per-frame"):
        assert any(expected in note for note in record.notes), expected


def test_malformed_files_degrade_rather_than_raise(tmp_path):
    root = str(tmp_path / "out")
    os.makedirs(root)
    path = _write_run(
        root,
        "gggg7777-r0",
        **{
            "link_report_edge__json": "{not json",
            "client_outcome_edge__json": "[]",
            "server_metrics_hub__csv": "",
            "fixture_latency_location_log__csv": "garbage,header\nonly,row\n",
        }
    )
    record = runs.read_run(path)
    assert record.link_reports == {}
    assert any("unreadable" in note for note in record.notes)
    assert record.task_metric()[1] is None


def test_a_run_with_no_config_degrades(tmp_path):
    root = str(tmp_path / "out")
    os.makedirs(root)
    path = os.path.join(root, "hhhh8888-r0")
    os.makedirs(path)
    with open(os.path.join(path, "run.json"), "w") as handle:
        handle.write('{"run_id": "hhhh8888-r0", "status": "failed"}')
    record = runs.read_run(path)
    assert record.nodes == []
    assert any("config.yaml" in note for note in record.notes)


def test_an_empty_collector_file_is_no_samples_not_zero(tmp_path):
    root = str(tmp_path / "out")
    os.makedirs(root)
    path = _write_run(root, "iiii9999-r0", **{"gpu_hub__csv": GPU_HEADER})
    assert device.read_device_series(path) == {}


# ---------------------------------------------------------------------------
# the grouping rule


def test_placement_comes_from_the_declared_parameter():
    record = runs.RunRecord("/x", "r", "r", None)
    record.app_params = {"placement": "gateway"}
    assert placement_of(record) == "gateway"


def test_placement_falls_back_to_the_pipeline_stage_nodes():
    record = runs.RunRecord("/x", "r", "r", None)
    record.stages = [
        {"name": "camera", "node": "edge"},
        {"name": "detect", "node": "hub"},
    ]
    assert placement_of(record) == "detect@hub"


def test_placement_falls_back_to_the_compute_node():
    record = runs.RunRecord("/x", "r", "r", None)
    record.nodes = [
        {"name": "edge", "role": "client", "device": "cpu"},
        {"name": "hub", "role": "server", "device": "gpu"},
    ]
    assert placement_of(record) == "hub/gpu"


def test_network_label_distinguishes_conditions_and_names_the_unshaped_one():
    shaped = runs.RunRecord("/x", "r", "r", None)
    shaped.links = _config(delay_ms=30.0)["links"]
    other = runs.RunRecord("/y", "r", "r", None)
    other.links = _config(delay_ms=60.0)["links"]
    bare = runs.RunRecord("/z", "r", "r", None)
    assert network_of(shaped) != network_of(other)
    assert network_of(bare) == "unshaped"


def test_swept_value_reads_the_central_delay():
    record = runs.RunRecord("/x", "r", "r", None)
    record.links = _config(delay_ms=30.0)["links"]
    assert swept_value(record) == pytest.approx(30.0)
    assert swept_value(runs.RunRecord("/y", "r", "r", None)) is None


def test_grouping_builds_the_matrix_and_keeps_the_hole(tree):
    table = group_runs(tree, [runs.read_run(p) for p in runs.discover_runs(tree)])
    assert table.placements == ["edge", "hub"]
    assert len(table.networks) == 3
    low = table.networks[0]
    assert table.cell("edge", low) is not None
    assert table.cell("hub", table.networks[1]) is None  # the hole, not invented


def test_repetitions_of_one_cell_group_together(tree):
    table = group_runs(tree, [runs.read_run(p) for p in runs.discover_runs(tree)])
    cell = table.cell("edge", table.networks[0])
    # The two repetitions of aaaa1111 plus the legacy-named run at the same
    # coordinate: a cell is every run at that coordinate, not one config hash.
    assert len(cell.runs) == 3
    # Median across the repetitions that wrote a metric. The legacy-named run
    # wrote no per-frame record, so it contributes a run to the cell but no
    # value to the median, which is the point of reducing only what is there.
    assert cell.metric()[1] == pytest.approx(0.255)


def test_networks_order_by_the_swept_value(tree):
    table = group_runs(tree, [runs.read_run(p) for p in runs.discover_runs(tree)])
    values = [
        table.cell(p, n).axis
        for n in table.networks
        for p in table.placements
        if table.cell(p, n) is not None and table.cell(p, n).axis is not None
    ]
    assert values == sorted(values)


def test_a_sweep_is_in_progress_while_any_run_is(tree):
    table = group_runs(tree, [runs.read_run(p) for p in runs.discover_runs(tree)])
    assert table.in_progress
    finished = [r for r in table.runs if not r.in_progress]
    assert not group_runs(tree, finished).in_progress


def test_crossover_series_skips_cells_with_no_metric(tree):
    table = group_runs(tree, [runs.read_run(p) for p in runs.discover_runs(tree)])
    series = table.series()
    names = {item["name"] for item in series}
    assert names == {"edge", "hub"}
    # The in-progress hub run wrote no metric, so it contributes no point rather
    # than a point at zero.
    hub = next(item for item in series if item["name"] == "hub")
    assert len(hub["points"]) == 1


def test_a_run_with_no_readable_config_is_named_not_placed(tmp_path):
    record = runs.RunRecord("/x", "zzzz-r0", "zzzz", 0)
    table = group_runs(str(tmp_path), [record])
    assert table.cells == {}
    assert any("zzzz-r0" in note for note in table.notes)


# ---------------------------------------------------------------------------
# the device series


def test_device_series_split_by_gpu_index(tree):
    found = device.read_device_series(os.path.join(tree, "aaaa1111-r0"))
    assert list(found) == ["hub"]
    gpu = [s for s in found["hub"] if s.collector == "gpu"]
    # Three quantities per card, two cards: a two-GPU node is two series per
    # quantity, not one merged series.
    assert len(gpu) == 6
    assert {s.key for s in gpu} == {"0", "1"}
    utilisation = [s for s in gpu if s.label == "GPU utilisation" and s.key == "0"]
    assert utilisation[0].samples == 5


def test_device_series_time_axis_starts_at_zero(tree):
    found = device.read_device_series(os.path.join(tree, "aaaa1111-r0"))
    for series in found["hub"]:
        assert series.t[0] == pytest.approx(0.0)
        assert series.t == sorted(series.t)


def test_device_series_reads_cpu_and_energy(tree):
    found = device.read_device_series(os.path.join(tree, "aaaa1111-r0"))
    labels = {s.label for s in found["hub"]}
    assert "CPU (host)" in labels
    assert "power" in labels


def test_energy_total_is_the_last_row_not_the_sum(tree):
    totals = device.cumulative_totals(os.path.join(tree, "aaaa1111-r0"))
    energy = [t for t in totals["hub"] if t["unit"] == "J"]
    assert energy[0]["value"] == pytest.approx(4.5)


def test_empty_rail_columns_are_absent_not_zero(tree):
    found = device.read_device_series(os.path.join(tree, "aaaa1111-r0"))
    # The energy rows leave the rail columns blank, which is normal for a source
    # that does not produce them; no series is built from an all-blank column.
    assert all("rail" not in s.label for s in found["hub"])


def test_a_run_with_no_collector_file_has_no_series(tree):
    assert device.read_device_series(os.path.join(tree, "bbbb2222-r0")) == {}


def test_a_missing_device_summary_is_an_empty_mapping(tree):
    assert device.read_device_summary(os.path.join(tree, "aaaa1111-r0")) == {}


def test_a_partial_device_summary_is_read_as_far_as_it_goes(tree):
    path = os.path.join(tree, "aaaa1111-r0", device.DEVICE_SUMMARY)
    with open(path, "w") as handle:
        json.dump({"nodes": {"hub": {"samples": 5, "cpu": {"cpu_pct_host_mean": 11.5}}}}, handle)
    summary = device.read_device_summary(os.path.join(tree, "aaaa1111-r0"))
    assert summary["hub"]["samples"] == 5
    # And the series still come from the CSVs, which stay the source of truth.
    assert device.read_device_series(os.path.join(tree, "aaaa1111-r0"))["hub"]


# ---------------------------------------------------------------------------
# the plots


def test_nice_ticks_are_round_and_cover_the_range():
    ticks = svg.nice_ticks(0.0, 0.47)
    assert ticks[0] <= 0.0
    assert ticks[-1] >= 0.4
    assert all(abs(t / 0.1 - round(t / 0.1)) < 1e-9 for t in ticks)


def test_nice_ticks_on_a_degenerate_range_does_not_spin():
    assert svg.nice_ticks(3.0, 3.0) == [3.0]


def test_scale_of_a_constant_series_does_not_divide_by_zero():
    scale = svg.Scale(5.0, 5.0, 0.0, 100.0)
    assert 0.0 <= scale(5.0) <= 100.0


def test_crossover_plot_draws_a_line_per_series():
    markup = svg.crossover_plot(
        [
            {"name": "edge", "points": [(10.0, 0.25, "a"), (50.0, 0.40, "b")]},
            {"name": "hub", "points": [(10.0, 0.31, "a"), (50.0, 0.30, "b")]},
        ],
        "delay (ms)",
        "metric (s)",
        "metric by placement",
    )
    assert markup.count("viz-line") == 2
    assert markup.count("viz-marker") == 4
    assert "var(--series-1)" in markup and "var(--series-2)" in markup
    assert "viz-legend" in markup  # two series, so identity is never colour alone


def test_crossover_plot_with_no_points_is_a_message_not_a_failure():
    markup = svg.crossover_plot([], "x", "y", "title")
    assert "viz-empty" in markup
    assert "viz-line" not in markup


def test_crossover_plot_never_invents_a_ninth_colour():
    series = [
        {"name": "p{}".format(i), "points": [(float(i), float(i), "")]} for i in range(12)
    ]
    markup = svg.crossover_plot(series, "x", "y", "t")
    assert "var(--series-9)" not in markup
    assert "first 8 of 12" in markup


def test_bar_panel_prints_a_dash_for_an_absent_value():
    markup = svg.bar_panel([("queue wait", 0.001), ("inference", None)], "t", "seconds")
    assert "not measured" in markup
    assert markup.count("viz-bar") == 1


def test_series_panel_thins_a_long_series_without_dropping_its_ends():
    long_series = [{"name": "power", "t": [i * 0.01 for i in range(50000)], "y": [1.0] * 50000}]
    markup = svg.series_panel(long_series, "t", "W")
    assert len(markup) < 200000
    assert "viz-line" in markup


def test_series_panel_with_nothing_says_so():
    assert "no device samples in this run" in svg.series_panel([], "t", "W")


def test_escape_closes_markup_injection():
    assert svg.escape('<a href="x">&') == "&lt;a href=&quot;x&quot;&gt;&amp;"


# ---------------------------------------------------------------------------
# the pages


def _assert_offline(markup):
    """Nothing on this page is fetched at page load."""
    for marker in REMOTE_MARKERS:
        assert marker not in markup, "page would fetch {!r} at load".format(marker)


def test_sweep_view_is_self_contained_and_shows_the_matrix(tree):
    table = group_runs(tree, [runs.read_run(p) for p in runs.discover_runs(tree)])
    markup = render.sweep_view(table, tree)
    _assert_offline(markup)
    assert "placement \\ network" in markup
    assert "not run" in markup  # the hole is shown as a hole
    assert "viz-line" in markup


def test_run_view_is_self_contained_and_shows_each_panel(tree):
    record = runs.read_run(os.path.join(tree, "aaaa1111-r0"))
    markup = render.run_view(record, tree)
    _assert_offline(markup)
    for expected in ("topology", "per-stage latency", "task metric", "link report"):
        assert expected in markup


def test_run_view_of_a_run_missing_everything_still_renders(tmp_path):
    root = str(tmp_path / "out")
    os.makedirs(root)
    path = _write_run(root, "jjjj1010-r0")
    markup = render.run_view(runs.read_run(path), root)
    _assert_offline(markup)
    assert "no link report in this run" in markup
    assert "no server per-request record in this run" in markup


def test_device_view_of_a_run_with_no_collector_says_so(tree):
    record = runs.read_run(os.path.join(tree, "bbbb2222-r0"))
    markup = render.device_view(record, tree)
    _assert_offline(markup)
    assert "no device samples in this run" in markup
    assert "enabled no metric collector" in markup


def test_device_view_distinguishes_an_asked_for_collector_that_wrote_nothing(tmp_path):
    root = str(tmp_path / "out")
    os.makedirs(root)
    config = _config(delay_ms=10.0)
    config["metrics"]["collectors"] = ["gpu"]
    path = _write_run(root, "kkkk1111-r0", config=config)
    markup = render.device_view(runs.read_run(path), root)
    assert "asked for the collector(s) gpu" in markup


def test_device_view_plots_each_quantity_in_its_own_panel(tree):
    record = runs.read_run(os.path.join(tree, "aaaa1111-r0"))
    markup = render.device_view(record, tree)
    _assert_offline(markup)
    assert "GPU utilisation [0]" in markup
    assert "GPU utilisation [1]" in markup
    assert markup.count("viz-line") >= 6


def test_live_view_refreshes_only_while_something_is_running(tree):
    table = group_runs(tree, [runs.read_run(p) for p in runs.discover_runs(tree)])
    running = render.live_view(table, tree, 5)
    _assert_offline(running)
    assert 'http-equiv="refresh"' in running
    finished = group_runs(tree, [r for r in table.runs if not r.in_progress])
    assert 'http-equiv="refresh"' not in render.live_view(finished, tree, 5)


def test_a_finished_run_view_does_not_refresh(tree):
    record = runs.read_run(os.path.join(tree, "aaaa1111-r0"))
    assert 'http-equiv="refresh"' not in render.run_view(record, tree)


def test_an_in_progress_run_view_refreshes(tree):
    record = runs.read_run(os.path.join(tree, "dddd4444-r0"))
    assert 'http-equiv="refresh"' in render.run_view(record, tree)


def test_an_empty_tree_renders_a_page_not_an_error(tmp_path):
    root = str(tmp_path / "empty")
    os.makedirs(root)
    table = group_runs(root, [])
    markup = render.sweep_view(table, root)
    _assert_offline(markup)
    assert "no runs" in markup


def test_every_page_declares_both_themes(tree):
    table = group_runs(tree, [runs.read_run(p) for p in runs.discover_runs(tree)])
    markup = render.sweep_view(table, tree)
    assert "prefers-color-scheme: dark" in markup
    assert "--series-1" in markup


def test_a_node_name_from_a_run_directory_cannot_inject_markup(tmp_path):
    root = str(tmp_path / "out")
    os.makedirs(root)
    config = _config(delay_ms=10.0)
    config["nodes"][0]["name"] = "<script>alert(1)</script>"
    path = _write_run(root, "llll1212-r0", config=config)
    markup = render.run_view(runs.read_run(path), root)
    assert "<script>" not in markup
    assert "&lt;script&gt;" in markup


# ---------------------------------------------------------------------------
# a config in a tree the dashboard did not produce


# Each case is a config.yaml that is syntactically fine and structurally wrong,
# which is what a run directory from an older or a hand-edited tree can hold.
# The run view has to render every one of them, because the alternative during a
# demonstration is a traceback.
MALFORMED_CONFIGS = [
    ("a list, not a mapping", b"- a\n- b\n"),
    ("a scalar", b"42\n"),
    ("an empty file", b""),
    ("sections that are scalars", b"app: 5\nexperiment: 7\ntransport: [1]\nmetrics: x\n"),
    ("sections that are null", b"nodes: null\nlinks: null\npipeline: null\napp: null\n"),
    ("a list holding non-mappings", b"nodes: [1, 2, {name: ok}]\nlinks: [3]\n"),
    ("a pipeline that is a list", b"pipeline: []\ncollectors: 1\n"),
    ("collectors that are not a list", b"metrics: {collectors: 5}\n"),
    ("a tag safe_load refuses", b"a: !!python/object/apply:os.system [echo no]\n"),
    ("bytes that are not text", b"\x00\x01\xff\xfe not utf-8\n"),
]


@pytest.mark.parametrize("label,body", MALFORMED_CONFIGS, ids=[c[0] for c in MALFORMED_CONFIGS])
def test_a_malformed_config_renders_a_page_and_a_note(tmp_path, label, body):
    root = str(tmp_path / "out")
    path = os.path.join(root, "mmmm-r0")
    os.makedirs(path)
    with open(os.path.join(path, "config.yaml"), "wb") as handle:
        handle.write(body)
    record = runs.read_run(path)
    assert record.nodes == [] or all(isinstance(n, dict) for n in record.nodes)
    assert record.notes, "a config that yielded no topology must say so"
    markup = render.run_view(record, root)
    _assert_offline(markup)
    # And it still groups, into no cell, rather than stopping the sweep view.
    table = group_runs(root, [record])
    render.sweep_view(table, root)


def test_a_malformed_per_run_file_set_still_groups_and_renders(tmp_path):
    """Every per-run file wrong at once: the whole page is notes, and it renders."""
    root = str(tmp_path / "out")
    path = os.path.join(root, "nnnn-r0")
    os.makedirs(path)
    files = {
        "run.json": b"[]",
        "config.yaml": b"nodes: [{name: n, role: server}]\n",
        "link_report_n.json": b"[1, 2]",
        "client_outcome_n.json": b"3",
        "server_metrics_n.csv": b"a,b\nnot,numbers\n",
        "gpu_n.csv": b"t_wall,t_monotonic,node,gpu_index,util_gpu_pct\nx,y,n,0,nope\n",
        "n_latency_location_log.csv": b"brake_issued\nperhaps\n",
        "n_summary.json": b"{\"a\": \"not a number\", \"b\": 2}",
        device.DEVICE_SUMMARY: b"{\"nodes\": 5}",
    }
    for name, body in files.items():
        with open(os.path.join(path, name), "wb") as handle:
            handle.write(body)
    record = runs.read_run(path)
    assert record.in_progress  # run.json was not a mapping, so there is no status
    assert record.sidecars == {"b": 2.0}  # the numeric leaf only
    assert device.read_device_series(path) == {}  # no parseable sample
    assert device.read_device_summary(path) == {}  # "nodes" was not a mapping
    for markup in (render.run_view(record, root), render.device_view(record, root)):
        _assert_offline(markup)
    table = group_runs(root, [record])
    _assert_offline(render.sweep_view(table, root))
    _assert_offline(render.live_view(table, root, 5))


# ---------------------------------------------------------------------------
# the server's path handling


def test_the_server_resolves_only_runs_it_discovered(tree):
    from cadet.dashboard.server import DashboardState

    state = DashboardState(tree)
    assert state.resolve("aaaa1111-r0").run_id == "aaaa1111-r0"
    assert state.resolve("../../etc") is None
    assert state.resolve("server_detections") is None
    assert state.resolve("") is None


# --- an application's own headline metric ---------------------------------
def test_a_summary_sidecars_first_number_is_the_headline_metric(tmp_path):
    """The dashboard does not know the application, so the application leads.

    An app with no per-frame record of the braking shape still has a headline
    number, and the sidecar is how it reaches the sweep matrix. The rule is "the
    first numeric entry", not a name this module knows, so the key order in the
    file is the application's own statement of which number leads.
    """
    run = tmp_path / "exp" / "app" / "0123456789-r0"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"run_id": "0123456789-r0", "status": "complete"}))
    (run / "sensing_summary.json").write_text(
        json.dumps({"e2e_p50_ms": 166.86, "aoi_p50_ms": 83.0, "uplink_kb": 22.0})
    )
    record = runs.read_run(str(run))
    name, value, unit = record.task_metric()
    assert (name, value, unit) == ("e2e p50", 166.86, "ms")


def test_the_unit_comes_off_the_key_and_ms_is_not_read_as_s(tmp_path):
    run = tmp_path / "exp" / "app" / "0123456789-r0"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"run_id": "0123456789-r0", "status": "complete"}))
    (run / "x_summary.json").write_text(json.dumps({"settle_ms": 12.0, "settle_s": 0.012}))
    record = runs.read_run(str(run))
    # "_ms" must win over "_s", or every millisecond is reported as a second.
    assert record.task_metric() == ("settle", 12.0, "ms")


def test_a_run_with_neither_a_frame_record_nor_a_sidecar_has_no_metric(tmp_path):
    run = tmp_path / "exp" / "app" / "0123456789-r0"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"run_id": "0123456789-r0", "status": "complete"}))
    name, value, unit = runs.read_run(str(run)).task_metric()
    assert value is None


# --- the condition axis, and the numbers on it -----------------------------
def test_the_condition_is_the_access_network_not_every_link(tmp_path):
    """Two topologies at one access quality belong in one column.

    A cloud placement declares a backhaul hop as well as an access hop. Labelling
    a cell by every link gave it a different coordinate from an edge run at the
    same access quality, so a matrix meant to be three columns wide came out nine
    and mostly "not run".
    """
    edge = runs.RunRecord("/edge", "aaaaaaaaaa-r0", "aaaaaaaaaa", 0)
    edge.nodes = [
        {"name": "sensor_1", "role": "client"},
        {"name": "gateway", "role": "server"},
    ]
    edge.links = [
        {
            "name": "uplink",
            "from": "sensor_1",
            "to": "gateway",
            "network": {"distribution": {"kind": "lognormal", "p50_ms": 8.0, "p99_ms": 25.0}},
        }
    ]
    cloud = runs.RunRecord("/cloud", "bbbbbbbbbb-r0", "bbbbbbbbbb", 0)
    cloud.nodes = [
        {"name": "sensor_1", "role": "client"},
        {"name": "gateway", "role": "waypoint"},
        {"name": "cloud", "role": "server"},
    ]
    cloud.links = list(edge.links) + [
        {
            "name": "backhaul",
            "from": "gateway",
            "to": "cloud",
            "network": {"distribution": {"kind": "lognormal", "p50_ms": 25.0, "p99_ms": 90.0}},
        }
    ]
    from cadet.dashboard.sweep import network_of, swept_value

    assert network_of(edge) == network_of(cloud)
    # And on the x axis too, or the two would not be above each other on the
    # plot that compares them.
    assert swept_value(edge) == swept_value(cloud) == 8.0


def test_a_record_with_no_client_falls_back_to_every_link():
    """Grouping by something it stated beats grouping by nothing."""
    from cadet.dashboard.sweep import access_links

    record = runs.RunRecord("/x", "cccccccccc-r0", "cccccccccc", 0)
    record.nodes = [{"name": "a", "role": "server"}]
    record.links = [{"name": "l", "from": "a", "to": "b", "network": {}}]
    assert access_links(record) == record.links


def test_four_digit_milliseconds_are_not_an_exponent():
    """A page of millisecond figures is where four digits are ordinary."""
    assert svg.fmt(1155.6) == "1156"
    assert svg.fmt(999.5) == "999.5"
    assert svg.fmt(12345.0) == "12345"
    # Genuinely large and genuinely tiny stay exponents, where that reads better.
    assert "e" in svg.fmt(2.5e7)
    assert "e" in svg.fmt(0.0000123)


# --- several sweeps in one tree -------------------------------------------
def _sweep_dir(root, name, cells):
    """Write a directory of run dirs: cells is [(run_id, experiment, placement)]."""
    import json

    for run_id, experiment, placement in cells:
        run = root / name / run_id
        run.mkdir(parents=True)
        (run / "run.json").write_text(json.dumps({"run_id": run_id, "status": "complete"}))
        (run / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "experiment": {"name": experiment},
                    "app": {"plugin": "x", "params": {"placement": placement}},
                    "nodes": [
                        {"name": "s1", "role": "client"},
                        {"name": "g", "role": "server"},
                    ],
                    "links": [
                        {
                            "name": "u",
                            "from": "s1",
                            "to": "g",
                            "network": {"distribution": {"kind": "constant", "value_ms": 8.0}},
                        }
                    ],
                }
            )
        )


def test_two_sweeps_in_one_tree_get_a_matrix_each(tmp_path):
    """Merged, a cell averaged runs from different experiments and said nothing."""
    from cadet.dashboard.sweep import group_sweeps

    _sweep_dir(tmp_path, "small", [("1111111111-r0", "small", "gateway")])
    _sweep_dir(tmp_path, "large", [("2222222222-r0", "large", "gateway")])
    records = [runs.read_run(p) for p in runs.discover_runs(str(tmp_path))]
    tables = group_sweeps(str(tmp_path), records)
    assert [t.name for t in tables] == ["large", "small"]
    assert all(len(t.runs) == 1 for t in tables)


def test_runs_gathered_into_one_directory_are_one_sweep(tmp_path):
    """A wrapper gathers an edge and a cloud sweep precisely to compare them.

    They carry different experiment names, so keying the grouping on the name
    split them back apart and undid the wrapper's purpose. The directory is the
    key, which follows whoever laid the tree out.
    """
    from cadet.dashboard.sweep import group_sweeps

    _sweep_dir(
        tmp_path,
        "gathered",
        [
            ("1111111111-r0", "edge_5", "gateway"),
            ("2222222222-r0", "edge_5", "sensor"),
            ("3333333333-r0", "cloud_5", "cloud"),
        ],
    )
    records = [runs.read_run(p) for p in runs.discover_runs(str(tmp_path))]
    tables = group_sweeps(str(tmp_path), records)
    assert len(tables) == 1
    assert sorted(tables[0].placements) == ["cloud", "gateway", "sensor"]


def test_the_sweep_page_shows_every_sweep_and_fetches_nothing(tmp_path):
    from cadet.dashboard.sweep import group_sweeps

    _sweep_dir(tmp_path, "small", [("1111111111-r0", "small", "gateway")])
    _sweep_dir(tmp_path, "large", [("2222222222-r0", "large", "sensor")])
    records = [runs.read_run(p) for p in runs.discover_runs(str(tmp_path))]
    html = render.sweep_view(group_sweeps(str(tmp_path), records), str(tmp_path))
    assert "<h2>small</h2>" in html and "<h2>large</h2>" in html
    assert "2 sweeps under" in html
    for forbidden in ("http://", "https://", "//cdn", "<script", "@import", "url("):
        assert forbidden not in html, forbidden


def test_one_sweep_is_still_titled_plainly(tmp_path):
    """A tree holding one sweep is the common case and must not gain furniture."""
    from cadet.dashboard.sweep import group_sweeps

    _sweep_dir(tmp_path, "only", [("1111111111-r0", "only", "gateway")])
    records = [runs.read_run(p) for p in runs.discover_runs(str(tmp_path))]
    html = render.sweep_view(group_sweeps(str(tmp_path), records), str(tmp_path))
    assert "<h2>sweep</h2>" in html
    assert "sweeps under" not in html


# --- a sweep between cells -------------------------------------------------
def _marker(root, name, state, done, cells):
    import json

    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / "sweep.json").write_text(
        json.dumps({"state": state, "completed": done, "cells": cells})
    )


def test_a_sweep_between_cells_is_in_progress(tmp_path):
    """The gap the run directories cannot see.

    Between two cells every run the sweep has produced is complete and more are
    coming, which from the runs alone is indistinguishable from a finished
    sweep - so the live view said "nothing is in progress" and stopped
    refreshing in the middle of one.
    """
    from cadet.dashboard.sweep import group_sweeps

    _sweep_dir(tmp_path, "s", [("1111111111-r0", "s", "gateway")])
    _marker(tmp_path, "s", "running", 1, 6)
    records = [runs.read_run(p) for p in runs.discover_runs(str(tmp_path))]
    table = group_sweeps(str(tmp_path), records)[0]
    assert all(not r.in_progress for r in table.runs)
    assert table.in_progress is True
    assert table.progress_detail == "1 of 6 cells"


def test_a_finished_sweep_is_not_in_progress(tmp_path):
    from cadet.dashboard.sweep import group_sweeps

    _sweep_dir(tmp_path, "s", [("1111111111-r0", "s", "gateway")])
    _marker(tmp_path, "s", "complete", 6, 6)
    records = [runs.read_run(p) for p in runs.discover_runs(str(tmp_path))]
    assert group_sweeps(str(tmp_path), records)[0].in_progress is False


def test_a_tree_with_no_marker_falls_back_to_the_runs(tmp_path):
    """Runs not produced by `cadet sweep`, or produced before the marker."""
    from cadet.dashboard.sweep import group_sweeps

    _sweep_dir(tmp_path, "s", [("1111111111-r0", "s", "gateway")])
    records = [runs.read_run(p) for p in runs.discover_runs(str(tmp_path))]
    table = group_sweeps(str(tmp_path), records)[0]
    assert table.progress == {}
    assert table.in_progress is False
    assert table.progress_detail == ""


def test_the_live_page_keeps_refreshing_between_cells(tmp_path):
    from cadet.dashboard.sweep import group_sweeps

    _sweep_dir(tmp_path, "s", [("1111111111-r0", "s", "gateway")])
    _marker(tmp_path, "s", "running", 1, 6)
    records = [runs.read_run(p) for p in runs.discover_runs(str(tmp_path))]
    html = render.live_view(group_sweeps(str(tmp_path), records), str(tmp_path), 5)
    assert 'http-equiv="refresh"' in html
    assert "1 of 6 cells" in html
    assert "Nothing is in progress" not in html


def test_the_live_page_stops_when_the_sweep_is_done(tmp_path):
    from cadet.dashboard.sweep import group_sweeps

    _sweep_dir(tmp_path, "s", [("1111111111-r0", "s", "gateway")])
    _marker(tmp_path, "s", "complete", 6, 6)
    records = [runs.read_run(p) for p in runs.discover_runs(str(tmp_path))]
    html = render.live_view(group_sweeps(str(tmp_path), records), str(tmp_path), 5)
    assert 'http-equiv="refresh"' not in html
    assert "Nothing is in progress" in html
