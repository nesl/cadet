"""Tests for the offered-rate-versus-link-capacity preflight warning.

The case that matters is the one that caused a nine-hour run: braking's
800x600 frames at 10 fps against a 100 Mbit/s link. The warning has to fire
there, stay silent where the link has no rate to exceed, and never be fatal.
"""

import pytest

from cadet.network import throughput


class _Camera:
    def __init__(self, width, height):
        self.width = width
        self.height = height


class _Source:
    def __init__(self, kind, fps, width=None, height=None, camera=None):
        self.kind = kind
        self.fps = fps
        self.width = width
        self.height = height
        self.camera = camera


class _Network:
    def __init__(self, mode, bandwidth_mbps=None):
        self.mode = mode
        self.bandwidth_mbps = bandwidth_mbps


class _Link:
    def __init__(self, name, network):
        self.name = name
        self.network = network


class _Channel:
    def __init__(self, stream_fps=None):
        self.stream_fps = stream_fps


class _Config:
    def __init__(self, source, links=(), channels=()):
        self.source = source
        self.links = list(links)
        self.channels = list(channels)


def _braking(mode="netem", bandwidth_mbps=100.0, fps=10):
    return _Config(
        _Source("carla", fps, camera=_Camera(800, 600)),
        [_Link("frames_link", _Network(mode, bandwidth_mbps))],
    )


def test_frame_bytes_from_camera_geometry():
    assert throughput.frame_bytes(_braking()) == 800 * 600 * 3


def test_frame_bytes_from_synthetic_geometry():
    config = _Config(_Source("synthetic", 20.0, width=64, height=48))
    assert throughput.frame_bytes(config) == 64 * 48 * 3


def test_frame_bytes_unknown_for_a_source_preflight_cannot_read():
    # replay reads geometry from a file and live from a device; neither can be
    # known before the run, so no estimate is attempted.
    assert throughput.frame_bytes(_Config(_Source("replay", 10.0))) is None


def test_offered_rate_is_the_braking_example_rate():
    # 800x600x3 B at 10 fps is 115.2 Mbit/s: the rate that overran the link.
    assert throughput.offered_mbps(_braking()) == pytest.approx(115.2)


def test_a_channel_fps_overrides_the_source_rate():
    config = _braking(fps=10)
    config.channels = [_Channel(stream_fps=2)]
    assert throughput.source_fps(config) == 2.0


def test_an_unpaced_source_has_no_rate_to_compare():
    config = _Config(_Source("synthetic", 0.0, width=64, height=48))
    assert throughput.source_fps(config) is None
    assert throughput.offered_mbps(config) is None
    assert throughput.throughput_warnings(config) == []


def test_warns_when_the_app_offers_more_than_the_link_carries():
    warnings = throughput.throughput_warnings(_braking(bandwidth_mbps=100.0))
    assert len(warnings) == 1
    assert "115.2 Mbit/s" in warnings[0]
    assert "100.0 Mbit/s" in warnings[0]


def test_silent_when_the_link_can_carry_the_offered_rate():
    assert throughput.throughput_warnings(_braking(bandwidth_mbps=200.0)) == []


def test_silent_for_a_message_level_backend_which_has_no_rate():
    # The delay backend holds whole messages in-process; there is no rate to
    # exceed, so an offered rate cannot overrun it.
    assert throughput.link_capacity_mbps(_Link("l", _Network("delay", 1.0))) is None
    assert throughput.throughput_warnings(_braking(mode="delay")) == []


def test_silent_for_a_kernel_link_with_no_bandwidth_configured():
    assert throughput.throughput_warnings(_braking(bandwidth_mbps=None)) == []
