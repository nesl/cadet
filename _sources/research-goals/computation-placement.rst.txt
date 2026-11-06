Computation Placement
======================

Research question
-----------------

Where should inference run — on the vehicle (edge), in a roadside unit, or
in the cloud — and how does placement affect latency and energy under
different network conditions?

Layers involved
---------------

- **Data Source** — any source (CARLA, replay, synthetic)
- **Processing Core** — model registry; ``device`` assignment per node
- **NetWaggle** — link between client and server
- **Analysis** — frame latency, queue depth (``metrics.collectors: [queue]``)

What to configure
-----------------

Assign a device to each node. The model registry maps logical model names to
per-device artifact paths. The braking experiment shows the full structure:

.. literalinclude:: ../../../examples/braking/flat/atspeed_braking.yaml
   :language: yaml
   :caption: Device and registry config — ``examples/braking/flat/atspeed_braking.yaml``

Key fields:

- ``nodes[*].device`` — device type (``cpu``, ``edge``, ``gpu``)
- ``nodes[*].gpu_index`` — explicit CUDA device index
- ``models.registry.<key>.per_device.<device>`` — path to the artifact for
  each device type (e.g. ``.pt`` for gpu, ``.engine`` for edge)

To sweep over link delay values:

.. literalinclude:: ../../snippets/sweep-link-delay.yaml
   :language: yaml
   :caption: Sweep over link delay — ``docs/snippets/sweep-link-delay.yaml``

How to run
----------

.. code-block:: bash

   cadet run examples/braking/flat/atspeed_braking.yaml \
       --set source.carla.host=$(hostname -I | awk '{print $1}') \
       --app-module examples.braking.flat.app

For a sweep:

.. code-block:: bash

   cadet sweep examples/braking/flat/atspeed_braking.yaml \
       --set source.carla.host=$(hostname -I | awk '{print $1}') \
       --app-module examples.braking.flat.app

Docker node isolation: set ``launcher: docker`` on a node to run it in its
own container on a per-run network. The ``ssh_docker`` launcher extends this
to a remote host.

Where results land
------------------

``outputs/{experiment}/{app}/{run_id}/``:

- ``{run_id}_latency_location_log.csv`` — per-frame latency; the
  ``location`` column records which node executed inference
- ``run.json`` — includes the resolved node and model config
- Queue collector output (if ``metrics.collectors: [queue]`` is set)

How to extend
-------------

- Add a second processing tier: add a node, wire a link from the first server
  to the second, and route frames in your app plugin.
- Enable the queue collector: add ``metrics.collectors: [queue]`` to your
  experiment config.

See also
--------

- :doc:`../tutorials/04-multi-device` — multi-device setup
- :doc:`../howto/model-registry` — wiring model artifacts
- :doc:`../explanation/processing-core-layer` — pipeline architecture
- :doc:`../explanation/analysis-layer` — orchestrator and metrics
