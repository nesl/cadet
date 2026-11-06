"""Record a flat run's per-frame inputs and decisions, for offline parity replay.

What this is for:
    Parity level 1 needs to prove the pipeline stages reproduce the flat app's
    decisions and commands exactly on *real* data. This records, from a flat CARLA
    run, the inputs each decision/control stage would see (detections per frame; ego
    telemetry per tick) and the flat outputs (brake frame; per-tick command), so the
    pipeline stages can be replayed offline (no CARLA) and compared byte for byte.

What it contains:
    - :class:`JsonlRecorder`: append-only one-JSON-object-per-line writer, a no-op
      unless a record directory is given (so a normal run is unaffected).
    - :func:`open_recorder`: build one from ``$CADET_RECORD_DIR`` (+ a name).
    - :func:`read_jsonl`: read a recording back.

How it fits into CADET:
    The flat braking app writes a ``*_detections.jsonl`` (server) and a
    ``*_control.jsonl`` (client) when ``$CADET_RECORD_DIR`` is set;
    ``scripts/braking_l1_parity.py`` replays them through the stages.

How to configure it:
    Set ``CADET_RECORD_DIR`` to enable recording; unset = no recording, no overhead.
"""

import json
import os
from typing import Dict, Iterator, List, Optional


class JsonlRecorder:
    """Append-only JSON-lines recorder; a no-op when ``path`` is ``None``."""

    def __init__(self, path: Optional[str]):
        self.path = path
        self._fh = None
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._fh = open(path, "w")

    def record(self, **fields) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(fields, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def open_recorder(name: str) -> JsonlRecorder:
    """Open a recorder at ``$CADET_RECORD_DIR/<name>.jsonl`` (no-op if unset)."""
    root = os.environ.get("CADET_RECORD_DIR")
    path = os.path.join(root, "{}.jsonl".format(name)) if root else None
    return JsonlRecorder(path)


def read_jsonl(path: str) -> List[Dict]:
    """Read a recording back into a list of dicts."""
    out: List[Dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _iter(path: str) -> Iterator[Dict]:
    for rec in read_jsonl(path):
        yield rec
