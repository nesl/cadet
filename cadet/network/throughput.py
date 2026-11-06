"""Compare what an app will offer a link against what the link can carry.

What this is for:
    The nine-hour stall that started this work was not a bug in the shaping. It
    was a braking run offering about 115 Mbit/s of camera frames to a link
    configured at a fraction of that, so the sender's queue grew without bound
    and the run never ended. Nothing in the config was invalid and nothing said
    anything; the only symptom was a run that did not finish. This module is the
    warning that was missing.

What it contains:
    - :func:`frame_bytes` and :func:`source_fps`: the two numbers the estimate
      is built from, read from the config.
    - :func:`offered_mbps`: what the app will put on the wire.
    - :func:`link_capacity_mbps`: what a link can carry, per backend.
    - :func:`throughput_warnings`: one readable line per link that cannot keep
      up.

How it fits into CADET:
    :func:`cadet.orchestrator.runner.preflight` and the emulated session call
    :func:`throughput_warnings` before a run starts and log each line. They are
    warnings and never fatal: a deliberately saturated link is a legitimate
    experiment, and the choice belongs to the experimenter. What is not
    legitimate is finding out by watching a run fail to finish.

    **The estimate is an upper bound, and says so.** It is the raw frame size -
    width x height x 3 bytes - times the frame rate. What actually goes on the
    wire is whatever the app's encoder produces, and the two example apps
    compress their payload, so the true offered rate is lower by that app's
    compression ratio. An upper bound is the useful direction for a warning:
    it cannot miss a link that is genuinely too slow, and the line reports the
    basis so the margin can be read as the uncertainty it is.

How to configure it:
    Nothing of its own. It reads ``source`` (frame geometry and rate),
    ``channels[*].stream_fps`` where a channel paces its own stream, and each
    link's ``network.bandwidth_mbps``.
"""

from typing import List, Optional

# Frames are carried as 8-bit RGB (see cadet.sources.*), three bytes per pixel.
BYTES_PER_PIXEL = 3
# Report a link only when the offered rate exceeds its capacity by at least
# this factor. The estimate is an upper bound, so a margin this side of 1.0
# would warn about links that are in fact adequate.
WARN_RATIO = 1.0


def frame_bytes(config) -> Optional[int]:
    """Return the raw size of one frame, or None if the source does not say.

    ``carla`` takes the camera geometry, ``synthetic`` its own; ``replay`` and
    ``live`` get their geometry from a file or a device that preflight cannot
    read, so they return None and no warning is attempted.
    """
    source = getattr(config, "source", None)
    kind = getattr(source, "kind", None)
    if kind == "carla":
        camera = source.camera
        return int(camera.width) * int(camera.height) * BYTES_PER_PIXEL
    if kind == "synthetic":
        return int(source.width) * int(source.height) * BYTES_PER_PIXEL
    return None


def source_fps(config) -> Optional[float]:
    """Return the frame rate the client will send at, or None if unpaced.

    A channel's ``stream_fps`` overrides the source's own rate, because that is
    what paces the send loop. An unpaced source (``fps: 0``) has no rate to
    compare: it sends as fast as the pipeline allows, which is a different
    question from whether a configured rate fits.
    """
    for channel in getattr(config, "channels", None) or []:
        if getattr(channel, "stream_fps", None):
            return float(channel.stream_fps)
    fps = getattr(getattr(config, "source", None), "fps", None)
    if fps is None or float(fps) <= 0.0:
        return None
    return float(fps)


def offered_mbps(config) -> Optional[float]:
    """Return the app's offered send rate in Mbit/s, or None if not derivable."""
    size = frame_bytes(config)
    fps = source_fps(config)
    if size is None or fps is None:
        return None
    return size * fps * 8.0 / 1e6


def link_capacity_mbps(link) -> Optional[float]:
    """Return a link's achievable rate in Mbit/s, or None where it has no rate.

    Only the kernel backends have one. ``delay`` is an in-process delay line
    holding whole messages, so it imposes no rate at all, and ``none`` shapes
    nothing; neither can be overrun by an offered rate.
    """
    network = getattr(link, "network", None)
    if network is None or network.mode not in ("netem", "netwaggle"):
        return None
    return network.bandwidth_mbps


def throughput_warnings(config) -> List[str]:
    """Return one line per link whose capacity the app's send rate exceeds."""
    offered = offered_mbps(config)
    if offered is None:
        return []
    warnings = []
    for link in getattr(config, "links", None) or []:
        capacity = link_capacity_mbps(link)
        if capacity is None or capacity <= 0:
            continue
        if offered <= capacity * WARN_RATIO:
            continue
        warnings.append(
            "link {!r} is configured at {:.1f} Mbit/s but the app offers up to "
            "{:.1f} Mbit/s ({} B frames at {:.1f} fps, uncompressed - the real "
            "rate is lower by whatever the app's encoder achieves). The sender's "
            "queue grows for as long as that holds and the run may not finish; "
            "lower source.fps or the frame size, or raise "
            "links.network.bandwidth_mbps.".format(
                link.name, capacity, offered, frame_bytes(config), source_fps(config)
            )
        )
    return warnings
