Use the synthetic source
========================

The synthetic source generates solid-colour frames in memory. It requires
no files, no simulator, and no GPU. Use it to benchmark the transport and
network stack without any inference cost.

Config
------

.. code-block:: yaml

   source:
     kind: synthetic
     count: 50
     width: 224
     height: 224
     fps: 10.0

- ``count``: number of frames to generate per run. The run ends when all
  frames have been sent and acknowledged.
- ``width``, ``height``: frame dimensions in pixels. Larger frames stress
  the transport layer.
- ``fps`` — frame injection rate. Combined with ``width``/``height``, this
  sets the offered bandwidth to the server.

Full config example
-------------------

.. code-block:: yaml

   extends: configs/base.yaml

   experiment: {name: synth_bench, app: reference, repetitions: 3}
   source:
     kind: synthetic
     count: 200
     width: 640
     height: 480
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
         distribution: {kind: constant, value_ms: 30.0}
   app: {plugin: reference, params: {model_kind: stub}}

Run::

    PYTHONPATH=. cadet run synth_bench.yaml --app-module examples.reference.app

The stub model returns immediately, so all measured latency comes from
the transport and network layers.

Throughput warning
------------------

If the offered frame rate times the frame size exceeds a link's
``bandwidth_mbps``, the orchestrator emits a preflight warning. The run
still starts, but frames will queue and the measured latency will grow.
To suppress the warning, either reduce ``count``/``fps``, increase
``bandwidth_mbps``, or use a smaller frame size.
