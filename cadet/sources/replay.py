"""Replay frame source (images folder or video file).

What this is for:
    A :class:`cadet.core.source.Source` that replays recorded frames (a folder
    of images or a single video file) at a configured fps, optionally looping.
    This is the "Dataset / replay" source mode: deterministic, no simulator.

What it contains:
    - :class:`FrameReader`: the seam that decides *where the next frame comes
      from*, and the two readers behind it - :class:`ImageFolderReader` over a
      sorted list of image files, and :class:`CaptureReader` over an OpenCV
      ``VideoCapture``, which takes a video path, a camera index or a stream URL
      alike.
    - :class:`ReplaySource`: open/next_frame/close/metadata over the frames.
      Everything that is not the reader lives here: the per-frame pacing, the
      frame counting and the stop condition.

How it fits into CADET:
    Built from a ``source`` of ``kind: replay`` (see
    :class:`cadet.config.schema.ReplaySource`). Images are read with OpenCV
    (imported lazily). :class:`cadet.sources.live.LiveSource` is this source
    with a different reader - a capture on a device or a stream instead of on a
    directory - so it inherits the pacing and the counting unchanged. The same
    app runs over any of them, or over
    :class:`cadet.sources.synthetic.SyntheticSource`, unchanged.
"""

import glob
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from cadet.core.source import Frame, Source

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


class FrameReader(ABC):
    """Where the next frame comes from, and nothing else.

    A reader hands back raw BGR images in OpenCV's own order; the pacing, the
    RGB conversion, the frame counting and the stop condition all belong to the
    source above it. ``read`` returning ``None`` means "no image this call" -
    what that means for the run is the source's decision, not the reader's.
    """

    @abstractmethod
    def open(self) -> None:
        """Acquire the underlying frames, raising if there are none to acquire."""

    @abstractmethod
    def read(self) -> Optional[Any]:
        """Return the next BGR image, or ``None`` if there is none to return."""

    def close(self) -> None:
        """Release whatever :meth:`open` acquired. Nothing to do by default."""

    def describe(self) -> Dict[str, Any]:
        """Reader-specific fields to fold into the source's metadata."""
        return {}


class ImageFolderReader(FrameReader):
    """Frames from a directory of image files, read in sorted order."""

    def __init__(self, path: str, loop: bool = False):
        self._path = path
        self._loop = loop
        self._images: List[str] = []
        self._i = 0

    def open(self) -> None:
        files: List[str] = []
        for ext in _IMAGE_EXTS:
            files += glob.glob(os.path.join(self._path, "*" + ext))
        self._images = sorted(files)
        if not self._images:
            raise RuntimeError("no images ({}) under {}".format(_IMAGE_EXTS, self._path))
        self._i = 0

    def read(self) -> Optional[Any]:
        import cv2

        if self._i >= len(self._images):
            if not self._loop:
                return None
            self._i = 0
        bgr = cv2.imread(self._images[self._i])
        self._i += 1
        return bgr

    def describe(self) -> Dict[str, Any]:
        return {"frames": len(self._images)}


class CaptureReader(FrameReader):
    """Frames from an OpenCV capture: a video file, a camera index or a stream URL.

    ``target`` is handed to ``cv2.VideoCapture`` as given, which is what makes
    the three cases one reader: an ``int`` is a capture device index, a ``str``
    is a path or a URL. ``rewind_on_end`` seeks back to the first frame when a
    read comes up empty, which is how a finite recording loops; a device or a
    stream has nothing to seek back to, so it leaves it off.
    """

    def __init__(self, target, rewind_on_end: bool = False):
        self._target = target
        self._rewind = rewind_on_end
        self._cap = None

    def open(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self._target)
        if not cap.isOpened():
            cap.release()
            # Said out loud rather than left to produce an empty run: a device
            # index with no camera behind it and an unreachable stream URL both
            # land here, and both are configuration mistakes.
            raise RuntimeError("could not open capture: {!r}".format(self._target))
        self._cap = cap

    def read(self) -> Optional[Any]:
        import cv2

        if self._cap is None:
            return None
        ok, bgr = self._cap.read()
        if not ok and self._rewind:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, bgr = self._cap.read()
        if not ok:
            return None
        return bgr

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def describe(self) -> Dict[str, Any]:
        # A capture reports no reliable frame count for a stream, and the replay
        # case never needed one, so neither claims one.
        return {"frames": None}


class ReplaySource(Source):
    """Replay frames from a folder of images or a video file."""

    def __init__(self, config, logger=None):
        src = config.source
        self._src = src
        self._fps = src.fps
        self._limit = src.limit
        self._log = logger or (lambda m: None)
        self._reader: Optional[FrameReader] = None
        self._emitted = 0
        self._last = None

    def _build_reader(self) -> FrameReader:
        """Pick the reader for this config: the seam a live source replaces.

        A directory is a sorted image list; a file with a video extension is an
        OpenCV capture. Anything else is a configuration error, because replay
        reads what is on disk and nothing else.
        """
        path = self._src.path
        if os.path.isdir(path):
            return ImageFolderReader(path, loop=self._src.loop)
        if os.path.isfile(path) and path.lower().endswith(_VIDEO_EXTS):
            return CaptureReader(path, rewind_on_end=self._src.loop)
        raise RuntimeError("replay path is not an image folder or video file: {}".format(path))

    def open(self) -> None:
        self._reader = self._build_reader()
        self._reader.open()
        self._emitted = 0

    def next_frame(self) -> Optional[Frame]:
        if self._limit is not None and self._emitted >= self._limit:
            return None
        if self._fps and self._fps > 0 and self._last is not None:
            dt = 1.0 / self._fps - (time.time() - self._last)
            if dt > 0:
                time.sleep(dt)
        bgr = self._reader.read() if self._reader is not None else None
        if bgr is None:
            return None
        meta = {"index": self._emitted, "capture_time": time.time()}
        self._emitted += 1
        self._last = time.time()
        return Frame(image=bgr[:, :, ::-1].copy(), meta=meta)  # BGR -> RGB

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def metadata(self) -> Dict[str, Any]:
        meta = {"kind": "replay", "path": self._src.path, "loop": self._src.loop, "frames": None}
        if self._reader is not None:
            meta.update(self._reader.describe())
        return meta
