"""Tests for the per-node roll-up of the device collectors' samples.

The reduction is small and the ways it can be wrong are specific, so each one
gets a test rather than a shared "it summarises" assertion:

* a **cumulative** column must be read from the last row and never summed. The
  energy collector already integrates, so summing ``energy_j_total`` would
  report the run's energy multiplied by its sample count - a mistake that
  produces a plausible-looking number, which is why it is pinned.
* a **grouped** collector must be grouped. The GPU collector writes one row per
  device per sample and the bandwidth collector one per link and direction, so
  a reduction that ignores the grouping averages two devices into one figure
  that describes neither.
* an **empty or absent** file must read as "nothing measured" rather than as
  zero. Both happen for ordinary reasons: a config that did not ask for a
  collector writes no file at all, and a role killed before its first flush
  leaves a header with no rows. A summary that reported 0 W for either would be
  indistinguishable from a device that drew nothing.
* the **span column** is the collectors' own, so a rename there must fail here
  rather than silently produce summaries with no durations.
"""

import json
import os

from cadet.metrics.collectors import CLOCK_COLUMNS
from cadet.metrics.device_summary import (
    _SPAN_COLUMN,
    SPECS,
    SUMMARY_FILE,
    summarise_file,
    summarise_run,
    summary_lines,
    write_summary,
)


def _write(path, header, rows):
    """Write one collector CSV: the clock columns, then the collector's own."""
    with open(path, "w") as fh:
        fh.write(",".join(list(CLOCK_COLUMNS) + list(header)) + "\n")
        for row in rows:
            fh.write(",".join(str(cell) for cell in row) + "\n")


def _energy(path, node="server", samples=((1.0, 10.0), (2.0, 20.0), (3.0, 30.0))):
    """An energy CSV whose energy_j_total accumulates, as the collector's does."""
    header = (
        "source",
        "power_w",
        "interval_s",
        "energy_j",
        "energy_j_total",
        "rail_gpu_mw",
        "rail_cpu_mw",
        "rail_soc_mw",
        "activity_index",
    )
    rows = []
    total = 0.0
    for t, watts in samples:
        total += watts
        rows.append([1000.0 + t, t, node, "nvidia-smi", watts, 1.0, watts, total, "", "", "", ""])
    _write(path, header, rows)


def test_cumulative_column_is_the_last_row_not_the_sum(tmp_path):
    path = str(tmp_path / "energy_server.csv")
    _energy(path)
    reduced = summarise_file(path, "energy")
    # 10 + 20 + 30 accumulated to 60 in the collector; the sum of the column
    # would be 10 + 30 + 60 = 100, which is the error this pins.
    assert reduced["energy_j_total"] == 60.0
    assert reduced["power_w_mean"] == 20.0
    assert reduced["power_w_peak"] == 30.0
    assert reduced["samples"] == 3
    assert reduced["duration_s"] == 2.0


def test_gpu_rows_are_grouped_by_device(tmp_path):
    path = str(tmp_path / "gpu_server.csv")
    header = (
        "source",
        "gpu_index",
        "util_gpu_pct",
        "util_mem_pct",
        "mem_used_mb",
        "mem_total_mb",
        "power_w",
    )
    _write(
        path,
        header,
        [
            [1000.0, 1.0, "server", "nvml", 0, 10, 1, 100, 24564, 50],
            [1000.0, 1.0, "server", "nvml", 1, 90, 9, 900, 49140, 250],
            [1000.1, 2.0, "server", "nvml", 0, 20, 2, 200, 24564, 60],
            [1000.1, 2.0, "server", "nvml", 1, 100, 10, 1000, 49140, 300],
        ],
    )
    reduced = summarise_file(path, "gpu")
    assert reduced["samples"] == 4
    idle, busy = reduced["gpus"]["0"], reduced["gpus"]["1"]
    # Ungrouped, the mean utilisation would be 52.5% - a figure describing
    # neither device.
    assert idle["util_gpu_pct_mean"] == 15.0
    assert busy["util_gpu_pct_mean"] == 95.0
    assert idle["mem_used_mb_peak"] == 200.0
    assert busy["mem_total_mb"] == 49140.0


def test_bandwidth_rows_are_grouped_by_link_and_direction(tmp_path):
    path = str(tmp_path / "bandwidth_server.csv")
    header = (
        "link",
        "direction",
        "bytes_total",
        "bytes_delta",
        "messages_total",
        "messages_delta",
        "interval_s",
        "mbps",
    )
    _write(
        path,
        header,
        [
            [1000.0, 1.0, "server", "access", "recv", 100, 100, 1, 1, 0.1, 8.0],
            [1000.0, 1.0, "server", "access", "sent", 10, 10, 1, 1, 0.1, 0.8],
            [1000.1, 2.0, "server", "access", "recv", 300, 200, 2, 1, 0.1, 16.0],
            [1000.1, 2.0, "server", "access", "sent", 30, 20, 2, 1, 0.1, 1.6],
        ],
    )
    reduced = summarise_file(path, "bandwidth")
    assert sorted(reduced["links"]) == ["access/recv", "access/sent"]
    recv = reduced["links"]["access/recv"]
    assert recv["mbps_mean"] == 12.0
    assert recv["mbps_peak"] == 16.0
    assert recv["bytes_total"] == 300.0
    assert recv["messages_total"] == 2.0


def test_a_header_with_no_rows_is_no_samples_and_not_zero(tmp_path):
    """A role killed before its first flush leaves exactly this file."""
    path = str(tmp_path / "energy_server.csv")
    _energy(path, samples=())
    assert summarise_file(path, "energy") == {}
    assert summarise_run(str(tmp_path)) == {}
    assert write_summary(str(tmp_path)) is None
    assert not os.path.exists(str(tmp_path / SUMMARY_FILE))


def test_an_absent_file_is_not_an_error(tmp_path):
    assert summarise_file(str(tmp_path / "energy_nobody.csv"), "energy") == {}
    assert summarise_run(str(tmp_path / "no-such-run")) == {}


def test_unreadable_cells_stay_out_of_the_statistics(tmp_path):
    """Empty cells are normal: one file per collector spans sources that differ."""
    path = str(tmp_path / "energy_server.csv")
    header = (
        "source",
        "power_w",
        "interval_s",
        "energy_j",
        "energy_j_total",
        "rail_gpu_mw",
        "rail_cpu_mw",
        "rail_soc_mw",
        "activity_index",
    )
    _write(
        path,
        header,
        [
            [1000.0, 1.0, "server", "tegrastats", 10.0, 1.0, 10.0, 10.0, 500, "", "", ""],
            [1000.1, 2.0, "server", "tegrastats", "", 1.0, "", 10.0, "", "", "", ""],
            [1000.2, 3.0, "server", "tegrastats", 20.0, 1.0, 20.0, 30.0, 700, "", "", ""],
        ],
    )
    reduced = summarise_file(path, "energy")
    # The mean is over the two readable samples, not over three with a zero.
    assert reduced["power_w_mean"] == 15.0
    assert reduced["samples"] == 3


def test_the_run_is_summarised_per_node(tmp_path):
    _energy(str(tmp_path / "energy_gateway.csv"))
    _energy(str(tmp_path / "energy_sensor_1.csv"), node="sensor_1")
    summary = summarise_run(str(tmp_path))
    assert sorted(summary["nodes"]) == ["gateway", "sensor_1"]
    path = write_summary(str(tmp_path))
    assert path == os.path.join(str(tmp_path), SUMMARY_FILE)
    with open(path) as fh:
        assert json.load(fh) == summary
    assert any("energy 60.0J" in line for line in summary_lines(summary))


def test_the_span_column_is_one_the_collectors_write():
    """A rename in the collectors must fail here, not silently drop durations."""
    assert _SPAN_COLUMN in CLOCK_COLUMNS


def test_every_spec_names_a_collector_that_exists():
    from cadet.metrics.collectors import _factory

    for name in SPECS:
        assert _factory(name) is not None, name
