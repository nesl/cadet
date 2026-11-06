Model Benchmarking
===================

Research question
-----------------

How do different perception models trade off latency, energy, and safety
outcomes across network conditions and deployment placements?

Layers involved
---------------

- **Data Source** — CARLA, replay, or synthetic
- **Processing Core** — model registry; per-device artifact selection via
  ``nodes[*].device``
- **NetWaggle** — link conditions under which models are evaluated
- **Analysis** — per-frame latency, collision outcome, queue depth

What to configure
-----------------

The model registry maps a logical model name to per-device artifact paths.
The example below shows the registry structure from the braking experiment:

.. literalinclude:: ../../../examples/braking/flat/atspeed_braking.yaml
   :language: yaml
   :caption: Model registry and device config — ``examples/braking/flat/atspeed_braking.yaml``

Key fields:

- ``models.model`` — active logical model name (must be a key in
  ``models.registry``)
- ``models.registry.<key>.per_device.<device>`` — path to the model artifact
  for each device type
- ``nodes[*].device`` — which device type this node uses to look up the artifact

To sweep over link delay conditions:

.. literalinclude:: ../../snippets/sweep-link-delay.yaml
   :language: yaml
   :caption: Sweep over delay values — ``docs/snippets/sweep-link-delay.yaml``

To sweep over model variants, register each under a distinct key in
``models.registry`` and add ``models.model`` to the sweep axes.

How to run
----------

.. code-block:: bash

   cadet run examples/braking/flat/atspeed_braking.yaml \
       --set source.carla.host=$(hostname -I | awk '{print $1}') \
       --app-module examples.braking.flat.app

For a sweep over delay conditions:

.. code-block:: bash

   cadet sweep examples/braking/flat/atspeed_braking.yaml \
       --set source.carla.host=$(hostname -I | awk '{print $1}') \
       --app-module examples.braking.flat.app

Where results land
------------------

``outputs/{experiment}/{app}/{run_id}/``:

- ``{run_id}_latency_location_log.csv`` — per-frame latency and brake
  reaction time
- ``run.json`` — resolved model and node config for the cell

The ``brake_reaction_latency_s`` column is the primary metric. The paper's
Table II sweeps YOLO11n through YOLO11x against link delay percentiles.

How to extend
-------------

- Register additional YOLO variants under separate registry keys and sweep
  ``models.model`` over them.
- Evaluate on replayed data: set ``source.kind: replay`` with
  ``source.path`` pointing to a recorded frames directory.
- Enable the queue collector: add ``metrics.collectors: [queue]``.

See also
--------

- :doc:`../howto/model-registry` — artifact registration
- :doc:`../tutorials/05-sweeps` — sweep configuration
- :doc:`../tutorials/04-multi-device` — multi-device deployment
- :doc:`../explanation/dataset-mode` — VLM/LLM dataset mode
