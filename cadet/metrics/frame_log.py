"""Per-frame metric collection for the braking experiment.

What this is for:
    Collects one record per transmitted frame and writes a CSV compatible with the
    original ``*_latency_location_log.csv`` so existing analysis scripts keep
    working. New columns are appended at the end only.

What it contains:
    - :data:`ORIGINAL_COLUMNS` / :data:`NEW_COLUMNS` and the combined field order.
    - :class:`MetricsLog`: accumulate records, mark brake/stop, write CSV, and
      compute a summary (brake-engage/stop time+location, frames sent, median
      brake-reaction latency).

How it fits into CADET:
    The braking client app (:mod:`cadet.apps.braking`) owns a :class:`MetricsLog`,
    adds a record per frame sent, and finalises it on stop. The runtime never
    writes results into curated dirs; paths come from the resolved config's
    ``outputs`` section.

How to configure it:
    Construct with the output CSV path and the ``run_id``/``rep``; those populate
    the appended columns.
"""

import csv
import statistics
from typing import Any, Dict, Optional

# Original columns — order and names must not change (old analysis depends on them).
ORIGINAL_COLUMNS = [
    "frame_id",
    "frame_capture_time",
    "frame_capture_loc",
    "frame_send_time",
    "frame_send_loc",
    "server_processing_time",
    "server_response_receive_time",
    "server_response_loc",
    "brake_issued",
    "stopping_time",
    "stopping_loc",
    "vehicle_speed_mph",
]

# New columns appended at the end only.
#   brake_reaction_latency_s: wall-clock time from the brake-TRIGGERING frame's
#   capture to the client receiving the brake command. The server replies only
#   on brake, so this is a single sample per run (not a per-frame stream metric),
#   and it is inflated by the server's first-inference warmup (~1 s) stalling the
#   queue — see docs/e2e_latency.md.
NEW_COLUMNS = ["run_id", "rep", "brake_reaction_latency_s"]

FIELDNAMES = ORIGINAL_COLUMNS + NEW_COLUMNS


class MetricsLog:
    """Accumulates per-frame records and derives braking summary metrics."""

    def __init__(self, csv_path: str, run_id: str, rep: int):
        self.csv_path = csv_path
        self.run_id = run_id
        self.rep = rep
        self.records: Dict[int, Dict[str, Any]] = {}
        # Summary state.
        self.brake_time: Optional[float] = None
        self.brake_loc = None
        self.stop_time: Optional[float] = None
        self.stop_loc = None

    def new_frame(self, frame_id: int, capture_time: float, capture_loc, speed_mph: float) -> None:
        """Start a record for a captured/sent frame."""
        self.records[frame_id] = {
            "frame_id": frame_id,
            "frame_capture_time": round(capture_time, 6),
            "frame_capture_loc": capture_loc,
            "frame_send_time": None,
            "frame_send_loc": None,
            "server_processing_time": None,
            "server_response_receive_time": None,
            "server_response_loc": None,
            "brake_issued": False,
            "stopping_time": None,
            "stopping_loc": None,
            "vehicle_speed_mph": round(speed_mph, 2),
            "run_id": self.run_id,
            "rep": self.rep,
            "brake_reaction_latency_s": None,
        }

    def mark_sent(self, frame_id: int, send_time: float, send_loc) -> None:
        """Record when/where a frame was sent."""
        rec = self.records.get(frame_id)
        if rec is not None:
            rec["frame_send_time"] = round(send_time, 6)
            rec["frame_send_loc"] = send_loc

    def mark_brake(
        self, frame_id: int, server_proc_time, response_time: float, response_loc, recv_wall=None
    ) -> None:
        """Record the brake-triggering frame's response fields and summary.

        ``response_time`` is sim-time (seconds since start) and populates the
        original ``server_response_receive_time`` column. ``recv_wall`` is the
        wall-clock time the reply was processed; ``brake_reaction_latency_s`` is
        the wall-clock round trip ``recv_wall - frame_capture_time`` for the
        brake-triggering frame (both wall-clock, unlike the original which mixed
        clocks). See docs/e2e_latency.md for why it is ~1.4 s.
        """
        self.brake_time = response_time
        self.brake_loc = response_loc
        rec = self.records.get(frame_id)
        if rec is not None:
            rec["brake_issued"] = True
            rec["server_processing_time"] = (
                round(server_proc_time, 6) if server_proc_time is not None else None
            )
            rec["server_response_receive_time"] = round(response_time, 6)
            rec["server_response_loc"] = response_loc
            cap = rec.get("frame_capture_time")
            if recv_wall is not None and cap is not None:
                rec["brake_reaction_latency_s"] = round(recv_wall - cap, 6)

    def mark_stop(self, stop_time: float, stop_loc) -> None:
        """Record the full-stop time/location against braked frames."""
        self.stop_time = stop_time
        self.stop_loc = stop_loc
        for rec in self.records.values():
            if rec["brake_issued"] and rec["stopping_time"] is None:
                rec["stopping_time"] = round(stop_time, 6)
                rec["stopping_loc"] = stop_loc

    def write(self) -> None:
        """Write all records to the CSV at :attr:`csv_path`."""
        with open(self.csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
            writer.writeheader()
            for fid in sorted(self.records):
                writer.writerow(self.records[fid])

    def frames_sent(self) -> int:
        """Number of frames that were actually transmitted."""
        return sum(1 for r in self.records.values() if r["frame_send_time"] is not None)

    def median_brake_reaction_s(self) -> Optional[float]:
        """Median brake-reaction latency (one sample here), or ``None``."""
        lat = [
            r["brake_reaction_latency_s"]
            for r in self.records.values()
            if r["brake_reaction_latency_s"] is not None
        ]
        return statistics.median(lat) if lat else None

    def summary(self) -> Dict[str, Any]:
        """Return the headline metrics for reporting/parity."""
        return {
            "brake_time": self.brake_time,
            "brake_loc": self.brake_loc,
            "stop_time": self.stop_time,
            "stop_loc": self.stop_loc,
            "frames_sent": self.frames_sent(),
            "median_brake_reaction_s": self.median_brake_reaction_s(),
        }
