"""Tests for the per-frame metrics log.

Covers: original column order preserved (new columns appended), record lifecycle
(new/sent/brake/stop), frames_sent, median latency, and summary.
"""

import csv

from cadet.metrics import FIELDNAMES, NEW_COLUMNS, ORIGINAL_COLUMNS, MetricsLog


def test_column_order_preserved():
    # Original columns come first, in their original order; new ones appended.
    assert FIELDNAMES[: len(ORIGINAL_COLUMNS)] == ORIGINAL_COLUMNS
    assert FIELDNAMES[len(ORIGINAL_COLUMNS) :] == NEW_COLUMNS
    assert ORIGINAL_COLUMNS[0] == "frame_id"
    assert ORIGINAL_COLUMNS[-1] == "vehicle_speed_mph"


def test_record_lifecycle_and_summary(tmp_path):
    path = str(tmp_path / "log.csv")
    m = MetricsLog(path, run_id="abc123-r0", rep=0)
    m.new_frame(0, capture_time=1.0, capture_loc=(1.0, 2.0, 0.0), speed_mph=40.0)
    m.mark_sent(0, send_time=1.01, send_loc=(1.0, 2.1, 0.0))
    m.new_frame(1, capture_time=1.1, capture_loc=(1.0, 3.0, 0.0), speed_mph=39.0)
    m.mark_sent(1, send_time=1.11, send_loc=(1.0, 3.1, 0.0))
    # Brake on frame 1 at sim-time 1.5; reply processed at wall-clock 1.5.
    m.mark_brake(
        1, server_proc_time=0.02, response_time=1.5, response_loc=(1.0, 4.0, 0.0), recv_wall=1.5
    )
    m.mark_stop(3.0, (1.0, 9.0, 0.0))

    assert m.frames_sent() == 2
    # brake-reaction latency = recv_wall - frame_capture_time = 1.5 - 1.1 = 0.4
    assert m.median_brake_reaction_s() == 0.4
    s = m.summary()
    assert s["brake_time"] == 1.5 and s["stop_time"] == 3.0
    assert s["brake_loc"] == (1.0, 4.0, 0.0) and s["stop_loc"] == (1.0, 9.0, 0.0)

    m.write()
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == FIELDNAMES
    assert rows[1]["brake_issued"] == "True"
    assert rows[1]["run_id"] == "abc123-r0"
