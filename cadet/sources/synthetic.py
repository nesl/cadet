"""Synthetic frame source (generated frames, for tests).

What this is for:
    A :class:`cadet.core.source.Source` that generates frames in-process — no
    files, no hardware, no CARLA. Used by framework tests and as the simplest
    way to exercise an app end-to-end deterministically.

What it contains:
    - :class:`SyntheticSource`: emits ``count`` frames (0 = unbounded) of a
      configured size, each stamped with its index, at an optional fps.

How it fits into CADET:
    Built from a ``source`` of ``kind: synthetic`` (see
    :class:`cadet.config.schema.SyntheticSource`). The same app runs over this or
    :class:`cadet.sources.replay.ReplaySource` unchanged.
"""

import time
from typing import Any, Dict, Optional

from cadet.core.source import Frame, Source


class SyntheticSource(Source):
    """Generates ``count`` frames of size ``width`` x ``height``."""

    def __init__(self, config, logger=None):
        src = config.source
        self._count = src.count
        self._w = src.width
        self._h = src.height
        self._fps = src.fps
        self._i = 0
        self._last = None

    def open(self) -> None:
        self._i = 0

    def next_frame(self) -> Optional[Frame]:
        if self._count and self._i >= self._count:
            return None
        if self._fps and self._fps > 0 and self._last is not None:
            dt = 1.0 / self._fps - (time.time() - self._last)
            if dt > 0:
                time.sleep(dt)
        import numpy as np

        # Deterministic content: a constant image whose value encodes the index.
        img = np.full((self._h, self._w, 3), self._i % 256, dtype=np.uint8)
        frame = Frame(image=img, meta={"index": self._i, "capture_time": time.time()})
        self._i += 1
        self._last = time.time()
        return frame

    def close(self) -> None:
        pass

    def metadata(self) -> Dict[str, Any]:
        return {"kind": "synthetic", "width": self._w, "height": self._h, "count": self._count}
