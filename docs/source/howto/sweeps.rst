Run and resume sweeps
=====================

A sweep defines a Cartesian product of parameter values and runs every
combination. Each cell gets a unique ``run_id`` derived from its resolved
config, so results never overwrite each other and a partial sweep can be
resumed.

Define a sweep
--------------

Add ``sweep.axes`` to the config:

.. code-block:: yaml

   extends: configs/base.yaml

   experiment: {name: latency_sweep, app: reference, repetitions: 3}
   source:
     kind: synthetic
     count: 100
     width: 224
     height: 224
   nodes:
     - {name: client, host: localhost, role: client, device: cpu}
     - {name: server, host: 127.0.0.1, port: 15010, role: server, device: cpu}
   links:
     - name: inference_link
       from: client
       to: server
       network:
         mode: delay
         distribution: {kind: constant, value_ms: 0}
   app: {plugin: reference, params: {model_kind: stub}}

   sweep:
     axes:
       links[0].network.distribution.value_ms: [10, 30, 60, 120]

This produces 4 cells × 3 repetitions = 12 runs.

Run the sweep
-------------

::

    PYTHONPATH=. cadet sweep latency_sweep.yaml --app-module examples.reference.app

Each cell writes to ``outputs/latency_sweep/reference/{run_id}/``. Run IDs
are of the form ``{10-char-hash}-r{rep}``.

Resume a partial sweep
----------------------

Re-run the same command. Cells whose ``run.json`` already has
``"status": "complete"`` are skipped. A cell with ``"status": "failed"``
is retried.

Check progress
--------------

::

    cadet status

Prints the state of the most recent run from the state store.

Override axes at the command line
----------------------------------

``--set`` overrides apply after sweep expansion, so they affect all cells::

    PYTHONPATH=. cadet sweep latency_sweep.yaml \
        --app-module examples.reference.app \
        --set experiment.repetitions=1

Dotted path with list-index notation
--------------------------------------

``links[0]`` addresses the first element of the ``links`` list. Any
schema field can be swept:

.. code-block:: yaml

   sweep:
     axes:
       links[0].network.distribution.value_ms: [30, 60]
       nodes[1].device: [cpu, gpu]

This produces a 2 × 2 = 4 cell sweep.
