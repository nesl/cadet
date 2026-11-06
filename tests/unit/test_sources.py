"""Unit tests for the non-CARLA sources (synthetic, replay and live).

The live tests generate their own media in a tmpdir and serve it over HTTP from
a thread in the test process, because the live source is exactly the replay
source with a capture reader and the only part that needs proving separately is
that reader: a stream URL frames really arrive over, and a device index that
reaches ``cv2.VideoCapture`` as an int. There is no camera here, so the device
path is checked as far as the capture call and no further.
"""

import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest

from cadet.sources import build_source
from cadet.sources.live import LiveSource
from cadet.sources.replay import CaptureReader, ImageFolderReader, ReplaySource
from cadet.sources.synthetic import SyntheticSource


def _cfg(source):
    return SimpleNamespace(source=source)


def _live(device=None, url=None, fps=0.0, limit=None):
    return _cfg(SimpleNamespace(kind="live", device=device, url=url, fps=fps, limit=limit))


def _write_clip(path, frames=12, size=(64, 48)):
    """Write a short generated clip, returning how many frames it holds."""
    import cv2

    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    assert writer.isOpened()
    for i in range(frames):
        writer.write(np.full((h, w, 3), (i * 11) % 256, np.uint8))
    writer.release()
    return frames


def _serve(directory):
    """Serve ``directory`` over HTTP on a free port; returns (base_url, stop)."""
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    def stop():
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    return "http://127.0.0.1:{}".format(httpd.server_address[1]), stop


def test_synthetic_emits_count_then_none():
    src = SyntheticSource(_cfg(SimpleNamespace(count=5, width=16, height=8, fps=0.0)))
    src.open()
    frames = []
    while True:
        f = src.next_frame()
        if f is None:
            break
        frames.append(f)
    assert len(frames) == 5
    assert frames[0].image.shape == (8, 16, 3)
    # Deterministic content encodes the index.
    assert frames[3].image.mean() == 3.0
    src.close()


def test_synthetic_unbounded_keeps_emitting():
    src = SyntheticSource(_cfg(SimpleNamespace(count=0, width=4, height=4, fps=0.0)))
    src.open()
    assert all(src.next_frame() is not None for _ in range(20))


def test_replay_reads_image_folder(tmp_path):
    import cv2

    for i in range(4):
        cv2.imwrite(str(tmp_path / "f{:02d}.png".format(i)), np.full((8, 8, 3), i * 10, np.uint8))
    src = ReplaySource(_cfg(SimpleNamespace(path=str(tmp_path), fps=0.0, loop=False, limit=None)))
    src.open()
    frames = []
    while True:
        f = src.next_frame()
        if f is None:
            break
        frames.append(f)
    assert len(frames) == 4  # stops at end
    assert frames[0].image.shape == (8, 8, 3)


def test_replay_limit_and_loop(tmp_path):
    import cv2

    for i in range(3):
        cv2.imwrite(str(tmp_path / "f{}.png".format(i)), np.zeros((4, 4, 3), np.uint8))
    src = ReplaySource(_cfg(SimpleNamespace(path=str(tmp_path), fps=0.0, loop=True, limit=7)))
    src.open()
    n = sum(1 for _ in iter(src.next_frame, None))
    assert n == 7  # loops but stops at the limit


def test_replay_missing_path_errors():
    src = ReplaySource(_cfg(SimpleNamespace(path="/no/such/dir", fps=0.0, loop=False, limit=None)))
    with pytest.raises(RuntimeError):
        src.open()


# --- the reader seam ------------------------------------------------------
def test_replay_picks_a_reader_per_path_shape(tmp_path):
    import cv2

    folder = tmp_path / "images"
    folder.mkdir()
    cv2.imwrite(str(folder / "a.png"), np.zeros((4, 4, 3), np.uint8))
    clip = tmp_path / "clip.mp4"
    _write_clip(clip)

    folder_src = ReplaySource(_cfg(SimpleNamespace(path=str(folder), fps=0.0, loop=False,
                                                  limit=None)))
    assert isinstance(folder_src._build_reader(), ImageFolderReader)
    clip_src = ReplaySource(_cfg(SimpleNamespace(path=str(clip), fps=0.0, loop=False, limit=None)))
    assert isinstance(clip_src._build_reader(), CaptureReader)


def test_replay_of_a_video_file_is_unchanged(tmp_path):
    # The default for an existing config must not move: a replay config still
    # reads its file through the same source class, frame for frame.
    clip = tmp_path / "clip.mp4"
    n = _write_clip(clip, frames=9)
    src = ReplaySource(_cfg(SimpleNamespace(path=str(clip), fps=0.0, loop=False, limit=None)))
    src.open()
    try:
        assert sum(1 for _ in iter(src.next_frame, None)) == n
        assert src.metadata()["kind"] == "replay"
        assert src.exhausted() is True  # a file that ran out really has ended
    finally:
        src.close()


def test_build_source_still_maps_the_old_kinds(tmp_path):
    # A config that does not say `kind: live` gets exactly what it got before.
    import cv2

    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((4, 4, 3), np.uint8))
    replay = build_source(
        _cfg(SimpleNamespace(kind="replay", path=str(tmp_path), fps=0.0, loop=False, limit=None))
    )
    assert type(replay) is ReplaySource
    synthetic = build_source(
        _cfg(SimpleNamespace(kind="synthetic", count=1, width=4, height=4, fps=0.0))
    )
    assert type(synthetic) is SyntheticSource


# --- the live source ------------------------------------------------------
def test_build_source_maps_live():
    assert isinstance(build_source(_live(url="http://127.0.0.1:1/none")), LiveSource)


def test_live_reads_frames_from_a_url(tmp_path):
    # The real thing the live reader has to do: frames arriving over a network
    # URL rather than off a path. The clip is generated here and served from
    # this process; nothing of it is kept.
    n = _write_clip(tmp_path / "clip.mp4", frames=12)
    base, stop = _serve(tmp_path)
    try:
        src = LiveSource(_live(url=base + "/clip.mp4", limit=n))
        src.open()
        try:
            frames = list(iter(src.next_frame, None))
            assert len(frames) == n
            assert frames[0].image.shape == (48, 64, 3)
            # Pacing and counting come from the replay source, unchanged.
            assert [f.meta["index"] for f in frames] == list(range(n))
            assert all(f.meta["capture_time"] > 0 for f in frames)
            assert src.metadata() == {
                "kind": "live",
                "device": None,
                "url": base + "/clip.mp4",
                "fps": 0.0,
                "limit": n,
            }
        finally:
            src.close()
    finally:
        stop()


def test_live_limit_stops_the_run(tmp_path):
    n = _write_clip(tmp_path / "clip.mp4", frames=12)
    base, stop = _serve(tmp_path)
    try:
        src = LiveSource(_live(url=base + "/clip.mp4", limit=4))
        src.open()
        try:
            assert sum(1 for _ in iter(src.next_frame, None)) == 4
            # The limit is the one thing that ends a live run by itself.
            assert src.exhausted() is True
        finally:
            src.close()
    finally:
        stop()
    assert n == 12


def test_live_without_a_limit_is_never_exhausted(tmp_path):
    # An empty read is "nothing yet", not "finished": a stalled stream must not
    # be mistaken for the end of one.
    _write_clip(tmp_path / "clip.mp4", frames=2)
    base, stop = _serve(tmp_path)
    try:
        src = LiveSource(_live(url=base + "/clip.mp4"))
        src.open()
        try:
            while src.next_frame() is not None:
                pass
            assert src.exhausted() is False
        finally:
            src.close()
    finally:
        stop()


def test_live_url_that_cannot_be_opened_is_named(tmp_path):
    src = LiveSource(_live(url="http://127.0.0.1:9/nothing.mp4"))
    with pytest.raises(RuntimeError) as err:
        src.open()
    assert "could not open capture" in str(err.value)


def test_live_device_index_reaches_the_capture_as_an_int(monkeypatch):
    # There is no camera on the machine this runs on, so what is checked is that
    # the device form reaches cv2.VideoCapture as a bare int (which is the API
    # it wants) and never rewinds. Whether a real camera then opens is not
    # provable here.
    import cv2

    seen = {}

    class _FakeCapture:
        def __init__(self, target, *args):
            seen["target"] = target

        def isOpened(self):
            return True

        def release(self):
            seen["released"] = True

    monkeypatch.setattr(cv2, "VideoCapture", _FakeCapture)
    src = LiveSource(_live(device=0))
    reader = src._build_reader()
    assert isinstance(reader, CaptureReader)
    assert reader._rewind is False  # a camera has no beginning to seek back to
    reader.open()
    assert seen["target"] == 0 and isinstance(seen["target"], int)
    reader.close()
    assert seen["released"] is True
