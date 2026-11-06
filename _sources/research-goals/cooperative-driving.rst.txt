Cooperative Driving
====================

The emergency-braking scenario runs end to end:
:doc:`../tutorials/02-carla-braking`. V2V and V2I scenarios are
documented in :doc:`../tutorials/07-v2v-v2i`.

Research question
-----------------

How much does vehicle-to-vehicle or vehicle-to-infrastructure communication
improve safety outcomes under varying network conditions?

Layers involved
---------------

- **Data Source** — CARLA 0.9.10 simulator with ScenarioRunner
- **Processing Core** — braking pipeline (detect → ROI-gate → brake decision)
- **NetWaggle** — link between the client and server nodes
- **Analysis** — per-frame latency, braking reaction time, collision outcome

What to configure
-----------------

The V2V experiment uses a constant 20 ms delay for the DSRC 802.11p baseline:

.. literalinclude:: ../../snippets/v2v-constant-link.yaml
   :language: yaml
   :caption: DSRC baseline link — ``docs/snippets/v2v-constant-link.yaml``

For the lognormal 5G model (p50 = 30 ms, p99 = 100 ms), see
``docs/snippets/v2v-lognormal-link.yaml``:

.. literalinclude:: ../../snippets/v2v-lognormal-link.yaml
   :language: yaml
   :caption: Lognormal 5G link — ``docs/snippets/v2v-lognormal-link.yaml``

The full V2V experiment config is at ``examples/v2v/v2v_braking.yaml``.
Key fields:

- ``source.scenario.ego_speed`` — follower starting speed
- ``links[0].network.distribution.value_ms`` — link delay in ms
- ``source.weather`` — weather preset: ``clear``, ``light``, or ``high``

How to run
----------

Emergency-braking baseline:

.. code-block:: bash

   cadet run examples/braking/flat/atspeed_braking.yaml \
       --set source.carla.host=$(hostname -I | awk '{print $1}') \
       --app-module examples.braking.flat.app

V2V scenario:

.. code-block:: bash

   cadet run examples/v2v/v2v_braking.yaml \
       --set source.carla.host=$(hostname -I | awk '{print $1}') \
       --app-module examples.v2v.app

Where results land
------------------

``outputs/{experiment}/{app}/{run_id}/``:

- ``{run_id}_latency_location_log.csv`` — per-frame brake reaction latency
- ``run.json`` — run metadata and outcome
- ``config.yaml`` — resolved config for the run

The column ``brake_reaction_latency_s`` is the primary metric.

How to extend
-------------

- Change the link model: replace ``mode: delay`` with ``mode: netem``
  (packet-level, ``docs/snippets/netem-link.yaml``) or ``mode: netwaggle``
  (Mininet-WiFi, ``docs/snippets/netwaggle-wired.yaml``).
- Sweep over link delay and ego speed using ``links[0].network.distribution.value_ms``
  and ``source.scenario.ego_speed`` as sweep axes.
- Write a new app: see :doc:`../tutorials/06-write-an-app`.

See also
--------

- :doc:`../tutorials/02-carla-braking` — step-by-step braking setup
- :doc:`../tutorials/07-v2v-v2i` — V2V and V2I configs and results
- :doc:`../explanation/data-source-layer` — CARLA source details
- :doc:`../explanation/netwaggle-layer` — network backend details
