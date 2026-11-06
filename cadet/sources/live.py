"""Live frame source (a capture device or a network stream).

What this is for:
    A :class:`cadet.core.source.Source` for frames arriving in real time from a
    camera or a network stream rather than from recorded files. It is the replay
    path pointed somewhere else: the per-frame pacing, the frame counting, the
    RGB conversion and the stop condition all come from
    :class:`cadet.sources.replay.ReplaySource` unchanged, and the only thing
    that differs is where the next frame comes from - an OpenCV capture on a
    device index or a stream URL instead of on a directory of files.

What it contains:
    - :class:`LiveSource`: the replay source with a capture reader, plus the one
      behaviour a live stream genuinely does not share with a file - an empty
      read means "nothing has arrived yet", not "finished".

How it fits into CADET:
    Built from a ``source`` of ``kind: live`` (see
    :class:`cadet.config.schema.LiveSource`), which names exactly one of
    ``device`` (an integer capture index, e.g. ``0``) and ``url`` (a stream URL
    OpenCV can open, e.g. ``rtsp://host:554/path`` or an HTTP MJPEG stream).
    ``fps`` caps the rate frames are pulled at (``0``, the default, takes them
    as fast as the stream delivers them, so the stream is the clock), and
    ``limit`` caps how many are emitted - which is what lets a live run finish
    by itself. Without a ``limit`` nothing in the stream ends the run and it
    stops on ``experiment.max_active_s``.
"""

from typing import Any, Dict

from cadet.sources.replay import CaptureReader, FrameReader, ReplaySource


class LiveSource(ReplaySource):
    """Live frames from a capture device index or a stream URL."""

    def _build_reader(self) -> FrameReader:
        """Open a capture on the configured device or URL.

        ``cv2.VideoCapture`` takes an ``int`` index and a URL string directly,
        so this is the whole of the live backend. It never rewinds: a camera has
        no beginning to seek to and a stream is not seekable, so ``loop`` is a
        replay concept with no meaning here and the schema does not offer it.
        """
        src = self._src
        target = src.device if src.device is not None else src.url
        return CaptureReader(target, rewind_on_end=False)

    def exhausted(self) -> bool:
        """True only once the frame budget is spent.

        A live capture that hands back nothing has handed back nothing *now* - a
        dropped frame, a stalled stream, a camera that has not delivered its
        first frame yet - and that is not the end of the stream; see
        :meth:`cadet.core.source.Source.next_frame`. The one thing that does end
        a live run from the source's side is reaching ``limit``.
        """
        return self._limit is not None and self._emitted >= self._limit

    def metadata(self) -> Dict[str, Any]:
        src = self._src
        return {
            "kind": "live",
            "device": src.device,
            "url": src.url,
            "fps": src.fps,
            "limit": src.limit,
        }
