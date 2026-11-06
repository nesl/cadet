Design Goals
============

CADET's design follows the goals stated in §III of the paper. Each goal
maps to a specific code feature.

Reproducibility
---------------

Every run is addressed by a stable, content-derived ``run_id``:

    ``{10-char SHA-256 of resolved config}-r{repetition}``

The hash covers the merged config after ``extends:`` resolution and
all ``--set`` overrides. The same config always maps to the same ``run_id``
regardless of when or where it runs.

**Code:** ``cadet/orchestrator/runner.py`` ``select_run_config()``.

The resolved config is saved as ``config.yaml`` in the run's output
directory. Reproducing an old run is ``cadet run
outputs/<exp>/<app>/<run_id>/config.yaml``.

**Code:** orchestrator writes ``config.yaml`` alongside ``run.json``.

Realistic network emulation
----------------------------

Three backends with different trade-offs let the researcher choose what
to measure:

- ``delay`` applies distribution per message, without TCP interaction,
  so the configured distribution is what the application sees.
- ``netem`` applies distribution per packet via ``tc`` qdisc, so TCP
  Nagle and retransmission are visible in the measurement.
- ``netwaggle`` adds a real Mininet-WiFi topology and a propagation model,
  enabling wireless and multi-hop scenarios.

**Code:** ``cadet/network/shaping.py``, ``cadet/network/netem.py``,
``cadet/network/netwaggle/emulator.py``.

Modular sources
---------------

The source layer is a single interface. Swapping ``source.kind`` from
``carla`` to ``replay`` to ``synthetic`` changes nothing else in the
experiment, and ``live`` reads a camera or a video device through the same
interface.

**Code:** ``cadet/sources/carla.py``, ``cadet/sources/replay.py``,
``cadet/sources/synthetic.py``, ``cadet/sources/live.py``.

Distributed deployment
-----------------------

Each node has a ``launcher`` field: ``local``, ``docker``, ``ssh_docker`` and
``ssh_venv``. The ``device`` field
combined with ``models.registry`` lets the same config select a different
model artifact on each node type. This separation lets the researcher ask:
"what happens when I move inference from the vehicle to an edge server?"
without rewriting the pipeline.

**Code:** ``cadet/core/launcher.py``, ``cadet/core/docker_launcher.py``,
``cadet/config/schema.py`` ``Models``.
