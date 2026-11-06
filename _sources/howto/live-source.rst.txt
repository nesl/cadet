Live source
===========

``source.kind: live`` captures frames from a camera device or a network stream
using OpenCV. Use it to run a CADET experiment against real sensor input without
a simulator.

Config
------

Exactly one of ``device`` or ``url`` must be set.

**Camera device** (by index)::

    source:
      kind: live
      device: 0        # first attached camera; 1, 2, … for additional cameras
      fps: 30.0        # cap the pull rate; 0.0 = as fast as the stream delivers

**Network stream** (RTSP, HTTP MJPEG, or any URL OpenCV accepts)::

    source:
      kind: live
      url: "rtsp://203.0.113.10:554/stream"
      fps: 0.0         # let the stream set the pace

``fps``
    A positive value caps how fast CADET reads frames from the source; ``0.0``
    pulls as fast as OpenCV delivers them, so the stream controls the rate.

``limit``
    Optional cap on the number of frames emitted. Without it, the run stops at
    ``experiment.max_active_s``.

Minimal experiment::

    experiment:
      name: live_braking
      max_active_s: 60.0

    source:
      kind: live
      device: 0
      fps: 10.0

    nodes:
      - {name: sensor, host: localhost, role: client, device: cpu}
      - {name: server, host: localhost, role: server, device: gpu}

    links:
      - {name: uplink, from: sensor, to: server, direction: both}

    app:
      plugin: braking
      params: {conf_thresh: 0.5, roi_pct: 0.3}

Validate and run::

    cadet validate my_live.yaml --app-module examples.braking.flat.app
    cadet run      my_live.yaml --app-module examples.braking.flat.app

Notes
-----

- OpenCV must be able to open the device or URL at preflight time; ``cadet
  validate`` does not open it, but the run fails at the camera-open step if the
  device is absent or the URL is unreachable.
- A live source has no beginning to seek back to, so ``loop`` is absent.
- For network streams that require authentication, embed credentials in the URL
  as ``rtsp://user:pass@host/path`` — the URL is never logged.

See also
--------

- :doc:`../howto/replay-source` — replay a recorded session offline
- :doc:`../howto/synthetic-source` — generated frames, no hardware
- :doc:`../tutorials/01-first-experiment` — the quickest path to a first run
