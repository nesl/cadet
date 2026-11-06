Distributed Sensing
====================

Research question
-----------------

Where should object detection run when N sensors stream into a contested-access
network? Running the detector on each sensor avoids shipping frames but pays a
higher per-frame compute cost on constrained hardware. Running it on a gateway
or cloud node pays the bandwidth cost of shipping frames but shares one compute
budget across N sensors. The trade-off is measured across three placements, three
sensor counts, and three access qualities.

Layers involved
---------------

- **Data Source** — synthetic or replay source per sensor
- **Processing Core** — fan-in at the server; one detector queue shared by all
  sensors
- **NetWaggle** — per-sensor access link; quality swept over good, contested,
  and severe profiles
- **Analysis** — end-to-end latency, age of information, detection rate, bytes
  per link

What to configure
-----------------

Declare one client node per sensor and one server node as the processing point.
Each sensor gets its own link to the server with ``direction: both`` (the result
travels back on the same path). The quick config shows the minimal two-sensor
topology:

.. literalinclude:: ../../../examples/distributed_sensing/quick.yaml
   :language: yaml
   :caption: Two-sensor quick config — ``examples/distributed_sensing/quick.yaml``

The edge sweep shows how the placement and access quality axes are declared:

.. literalinclude:: ../../../examples/distributed_sensing/edge_5.yaml
   :language: yaml
   :caption: Five-sensor placement sweep — ``examples/distributed_sensing/edge_5.yaml``

Key fields:

- ``nodes[*].role: client`` on each sensor node (multiple clients, one server)
- ``app.params.placement`` — ``sensor``, ``gateway``, or ``cloud``; the sweep
  axis in the edge configs
- ``app.params.detect_ms`` — per-node-class detector cost in milliseconds
  (config input, not a measurement)
- ``links[*].direction: both`` — the reply path uses the same link

How to run
----------

Validate and run the quick config::

    ./scripts/cadet validate examples/distributed_sensing/quick.yaml --app-module examples.distributed_sensing.app
    ./scripts/cadet run examples/distributed_sensing/quick.yaml --app-module examples.distributed_sensing.app

Run the placement sweep over 5 sensors::

    ./scripts/cadet sweep examples/distributed_sensing/edge_5.yaml --app-module examples.distributed_sensing.app

Results headline
----------------

From the sweep over five sensors and three access qualities (source:
``examples/distributed_sensing/README.md``):

- The sensor placement is flat across N: each sensor runs its own detector and
  its own source loop, so adding sensors adds independent workers rather than
  queue depth.
- The gateway saturates at N > 2: five sensors at 10 fps offer 50 frames per
  second; the gateway detector at 40 ms per frame handles 25. Median end-to-end
  latency jumps from 69.0 ms (N=2) to 1097.8 ms (N=5) at good access quality.
- The cloud wins at N=5 at good access quality (96.5 ms) and degrades more
  slowly than the gateway as access quality worsens.

Full results tables are in :doc:`../examples/distributed-sensing`.

How to extend
-------------

- Add more sensors: add ``client`` nodes and links in ``edge_5.yaml`` and
  adjust ``experiment.max_active_s`` if the run is long.
- Sweep access quality independently: the swept ``links`` list in
  ``edge_5.yaml`` can be extended with additional quality profiles.
- Try the emulated network: run ``wifi_5.yaml`` or ``emulated_5.yaml``
  via ``cadet emulate`` to add topology-level link shaping.

See also
--------

- :doc:`../examples/distributed-sensing` — full results, config listing, output format
- :doc:`../tutorials/06-write-an-app` — writing an app plugin
- :doc:`../explanation/netwaggle-layer` — network backend reference
- :doc:`../howto/link-distributions` — loss and delay parameters
