Tutorial 1: First Experiment (Replay + Reference App + Delay Link)
=====================================================================

This tutorial runs a complete CADET experiment without a simulator: the
**replay source** feeds pre-recorded frames, the **reference app** measures
round-trip latency, and a **delay link** adds 60 ms of emulated latency.

By the end you will have:

- a working local CADET setup,
- a completed run with output files,
- an understanding of how the config, orchestrator, and metrics fit together.

Prerequisites
-------------

- CADET installed (see :doc:`../quickstart`).
- A directory of images to replay (JPEG or PNG, sorted lexicographically).
  If you do not have one, the synthetic-source note in Step 1 below lets you
  skip it entirely.

----

Step 1: Write the config
------------------------

Create ``my_first_experiment.yaml``:

.. code-block:: yaml

   extends: configs/base.yaml

   experiment:
     name: first_experiment
     app: reference
     repetitions: 1

   source:
     kind: replay
     path: /path/to/your/images    # directory of .jpg/.png files
     fps: 10.0

   nodes:
     - name: client
       host: localhost
       role: client
       device: cpu

     - name: server
       host: localhost
       port: 5100
       role: server
       device: cpu

   links:
     - name: inference_link
       from: client
       to: server
       network:
         mode: delay
         distribution:
           kind: constant
           value_ms: 60.0

   app:
     plugin: reference

.. note::

   **Channels vs links:** Frames between client and server travel on the
   *link* when you want network emulation. A channel is unshaped plumbing
   (simulator → client, for example). Using both a channel and a link
   between the same two nodes would route frames on the channel, bypassing
   emulation entirely. Use a link whenever you want latency, jitter, or
   loss on the data path.

.. _synthetic-source:

.. note::

   **No images?** Replace the ``source`` block with:

   .. code-block:: yaml

      source:
        kind: synthetic
        count: 50
        width: 224
        height: 224
        fps: 10.0

   The synthetic source generates solid-colour frames in memory and needs no
   external files.

----

Step 2: Validate
----------------

.. run-tag:: non-gpu

.. code-block:: bash

   cadet validate my_first_experiment.yaml

A clean config prints no errors. If you see ``PreflightError``, check that
port 5100 is not already in use.

----

Step 3: Run
-----------

.. run-tag:: non-gpu

.. code-block:: bash

   cadet run my_first_experiment.yaml

You will see orchestrator log lines:

.. code-block:: text

   [orch] starting serve:server
   [orch] sim hook: ...
   [orch] starting client:client
   [orch] run first_experiment-abc1234567-r0 ok (all workers exited cleanly)

The run completes when all frames have been sent and the client exits.

----

Step 4: Examine the output
--------------------------

Results land in ``outputs/first_experiment/reference/<run_id>/``:

.. code-block:: bash

   ls outputs/first_experiment/reference/

.. code-block:: text

   abc1234567-r0/
   ├── run.json
   ├── config.yaml
   ├── link_report.json
   └── *_latency_location_log.csv

Open ``link_report.json`` to see the configured vs achieved latency:

.. code-block:: json

   {
     "inference_link": {
       "configured": {"mode": "delay", "distribution": {"kind": "constant", "value_ms": 60.0}},
       "achieved":   {"p50_ms": 60.0, "p99_ms": 60.0},
       "messages":   {"sent": 50, "received": 50, "unanswered": 0},
       "latency_ms": {"p50": 60.4, "p99": 61.2, "min": 59.9, "max": 62.1}
     }
   }

The measured p50 is ≈ 60 ms — the ``delay`` backend honours the configured
constant at the message level. Compare this to the netem tutorial
where the same 60 ms config produces ≈ 149 ms due to TCP Nagle.

----

Step 5: Sweep delay values
--------------------------

Add a sweep to the config to run three latency conditions in one command:

.. code-block:: yaml

   sweep:
     axes:
       links[0].network.distribution.value_ms: [30, 60, 120]

.. code-block:: bash

   cadet sweep my_first_experiment.yaml

Three output directories appear under ``outputs/first_experiment/reference/``,
one per delay value. Each has a unique ``run_id`` derived from its resolved
config. Re-running the sweep skips any cell that already has a ``run.json``
with ``"status": "complete"``.

----

Next steps
----------

- **Add jitter:** change the distribution to ``kind: lognormal`` with
  ``p50_ms: 60`` and ``p99_ms: 200`` to model a realistic tail.
- **Try netem:** change ``network.mode`` to ``netem`` and add
  ``bandwidth_mbps: 10`` to see how TCP reacts. See
  :doc:`../explanation/network-backends` for the full comparison.
- **Use CARLA:** replace the source with ``kind: carla`` and follow
  the CARLA tutorial for the full braking scenario.
