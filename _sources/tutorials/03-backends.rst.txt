Tutorial 3: Comparing Network Backends
========================================

.. note::

   The netem and netwaggle steps require a GPU host. The delay step runs
   on any machine.

Run the same experiment under all three backends to see how each one
changes the measured latency.

Base config
-----------

Save this as ``backends_compare.yaml``:

.. code-block:: yaml

   extends: configs/base.yaml

   experiment: {name: backends_compare, app: reference, repetitions: 1}
   source:
     kind: synthetic
     count: 200
     width: 224
     height: 224
     fps: 20.0
   nodes:
     - {name: client, host: localhost, role: client, device: cpu}
     - {name: server, host: 127.0.0.1, port: 15010, role: server, device: cpu}
   links:
     - name: inference_link
       from: client
       to: server
       network:
         mode: delay
         distribution: {kind: constant, value_ms: 60.0}
   app: {plugin: reference, params: {model_kind: stub}}

Step 1: delay backend
-----------------------

Run as-is::

    PYTHONPATH=. cadet run backends_compare.yaml \
        --app-module examples.reference.app

Expected ``link_report.json`` result: p50 ≈ 60 ms. The ``delay`` backend
honours the constant distribution per message without TCP interaction.

Step 2: netem backend (GPU host)
---------------------------------

.. note::

   Requires a GPU host (NET_ADMIN capability needed for tc qdisc).

Change the ``network`` block to:

.. code-block:: yaml

         mode: netem
         distribution: {kind: constant, value_ms: 60.0}
         bandwidth_mbps: 100.0

Run::

    PYTHONPATH=. cadet run backends_compare.yaml \
        --app-module examples.reference.app \
        --set links[0].network.mode=netem \
        --set links[0].network.bandwidth_mbps=100.0

Expected result: p50 ≈ 149 ms. TCP Nagle holds the write until the
previous ACK arrives; the ACK itself is behind the 60 ms qdisc, so the
effective round-trip is roughly 120 ms plus processing.

Add ``tcp_nodelay`` to reduce the Nagle overhead::

    PYTHONPATH=. cadet run backends_compare.yaml \
        --app-module examples.reference.app \
        --set links[0].network.mode=netem \
        --set links[0].network.bandwidth_mbps=100.0 \
        --set nodes[0].transport_tuning.tcp_nodelay=true

Expected result: p50 ≈ 110 ms.

Step 3: netwaggle backend (GPU host, emulation container)
----------------------------------------------------------

.. note::

   Requires the netwaggle emulation container and Mininet-WiFi.

Change the network block to:

.. code-block:: yaml

         mode: netwaggle
         medium: wired
         distribution: {kind: constant, value_ms: 60.0}
         bandwidth_mbps: 100.0

Run::

    cadet emulate backends_compare.yaml \
        --app-module examples.reference.app \
        --set links[0].network.mode=netwaggle \
        --set links[0].network.medium=wired \
        --set links[0].network.bandwidth_mbps=100.0

Expected result: same TCP Nagle effect as netem (p50 ≈ 149 ms for wired
without ``tcp_nodelay``). The netwaggle backend adds a real network
namespace but for a wired constant link the measured latency matches
netem.

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Backend
     - p50 (60 ms config)
     - Why
   * - ``delay``
     - ≈ 60 ms
     - Message-level; no TCP interaction
   * - ``netem``
     - ≈ 149 ms
     - Nagle doubles the round trip
   * - ``netem`` + tcp_nodelay
     - ≈ 110 ms
     - One-way delay plus processing
   * - ``netwaggle`` (wired)
     - ≈ 149 ms
     - Same Nagle effect as netem
