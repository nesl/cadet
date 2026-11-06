Frequently Asked Questions
==========================

.. contents:: Questions
   :local:
   :depth: 1

----

Why does the first inference take so long?
------------------------------------------

The first ``model.predict()`` call initialises the GPU kernel, loads weights
onto device memory, fuses model layers, and triggers cuDNN autotuner setup.
This takes 0.5--2 s depending on model size and GPU. Subsequent calls run at
steady-state speed (typically under 20 ms for YOLO11x on an A6000).

``models.warmup: true`` (the default) runs one inference before the experiment
starts so the first real frame does not pay this cost.

If ``brake_reaction_latency_s`` is around 1--2 s, warmup was disabled or the
model path was not found by the warmup pass. Check the server log for a
``model warmup`` line.

Why is ``netem`` p50 about 150 ms when I configured 60 ms?
-----------------------------------------------------------

The ``netem`` backend applies delay at the TCP packet level via a kernel qdisc.
With small messages paced at camera rate, TCP Nagle buffers each write until
the previous segment is acknowledged, and that acknowledgement itself waits
behind the configured qdisc delay. The resulting per-message latency is roughly
twice the configured one-way delay.

Use the ``delay`` backend when you need the application to observe the
configured distribution at the message level. Use ``netem`` when TCP
interaction (Nagle, retransmit) is part of what you want to measure. See
:doc:`explanation/network-backends` for a measured comparison.

What does ``cadet down`` do?
-----------------------------

It cleans up everything CADET owns on the host: kills CADET-managed processes,
removes ``cadet-*`` Docker containers, removes ``cadet-*`` network namespaces
and bridges, removes netem qdiscs from the state store, and calls
``store.clear_all()``. It does not affect non-CADET Docker containers or
system network config.

Run it after a crash or after ``kill -9`` on the orchestrator, before
starting the next experiment. See :doc:`howto/cadet-down`.

Can I run without Docker?
--------------------------

Yes. Install with ``pip install -U pip && pip install -e .`` (Python 3.8 to
3.13; 3.8's bundled pip cannot do an editable install) and run::

    PYTHONPATH=. cadet run examples/reference/synthetic.yaml \
        --app-module examples.reference.app

The ``netem`` backend requires ``iproute2`` (``tc`` must be on PATH). The
``netwaggle`` backend requires a Mininet-WiFi container. The ``docker`` and
``ssh_docker`` launchers require Docker. Everything else runs natively.

Does ``cadet validate`` accept ``launcher: ssh_docker``?
---------------------------------------------------------

Yes. Both ``docker`` and ``ssh_docker`` are **Supported** and need no opt-in.

``docker`` passed lifecycle verification on a GPU host (both link backends,
normal completion, the active limit, a killed server container, and a two-rep
sweep with nothing left behind). ``ssh_docker`` was promoted to Supported after
the distributed-sensing example ran across two real machines — sensors local, gateway in a
container on a second host — measuring end-to-end latency and clock offset
across the physical link.

How do I reproduce an old run?
--------------------------------

Each run saves its resolved config as ``config.yaml``. Pass it directly::

    cadet run outputs/{experiment}/{app}/{run_id}/config.yaml \
        --app-module examples.reference.app

The run ID is content-derived from the config, so the same config always
maps to the same ``run_id`` and the same output directory.

How are sweep cells identified?
---------------------------------

Each sweep cell's ``run_id`` is ``{10-char SHA-256 of the resolved config
for that cell}-r{repetition}``. Changing any swept parameter changes the
hash and therefore the output directory. Two cells that differ in any field
never collide.

``cadet sweep`` skips cells whose output directory already contains a
``run.json`` with ``"status": "complete"``. Interrupt and restart the same
command to resume.

How do I select a GPU for a node?
-----------------------------------

Set ``device: gpu`` and optionally ``gpu_index: N`` on the node spec. The
model registry resolves the artifact keyed by ``gpu``. With the ``docker``
launcher, CADET passes ``--gpus device=N`` and sets ``CUDA_VISIBLE_DEVICES=N``
inside the container.

With the local launcher, ``gpu_index`` is passed to the app as a config value;
the app selects the device when loading its model. See
:doc:`howto/model-registry`.

What CARLA version does CADET support?
---------------------------------------

CARLA 0.9.10. The Python client is the PythonAPI egg
(``carla-0.9.10-py3.7-linux-x86_64.egg``), mounted at the path given by
``sim.carla_python_api`` in your config. The egg is not on PyPI: do not
run ``pip install carla``. The server image is ``carlasim/carla:0.9.10.1``.
A ``0.9.10`` image is also compatible; preflight compares the first three
version components.

What signals does the orchestrator handle?
-------------------------------------------

SIGTERM and SIGHUP. Both are converted to a ``RunInterrupted`` exception
that triggers a clean teardown. The ``run.json`` for the interrupted run
will have ``"reason": "interrupted"``. SIGKILL bypasses this path; run
``cadet down`` afterward to clean up.
