Tutorial 5: Sweeps and Reproducibility
========================================

This tutorial runs on any machine — no GPU or CARLA required.

Save this config as ``sweep_tutorial.yaml``:

.. code-block:: yaml

   extends: configs/base.yaml

   experiment:
     name: sweep_tutorial
     app: reference
     repetitions: 2

   source:
     kind: synthetic
     count: 30
     width: 64
     height: 48

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
       links[0].network.distribution.value_ms: [10, 30, 60]

Step 1: Validate
-----------------

::

    PYTHONPATH=. cadet validate sweep_tutorial.yaml \
        --app-module examples.reference.app

Step 2: Run the sweep
----------------------

::

    PYTHONPATH=. cadet sweep sweep_tutorial.yaml \
        --app-module examples.reference.app

This produces 3 delay values × 2 repetitions = 6 runs. Watch the output
for lines like::

    [orch] run a1b2c3d4e5-r0 ok (workers exited ...) [startup=... active=...]
    [orch] run a1b2c3d4e5-r1 ok (workers exited ...)
    [orch] run b2c3d4e5f6-r0 ok (workers exited ...)
    ...

Step 3: Examine output directories
------------------------------------

::

    find outputs/sweep_tutorial -name "run.json" | sort

Each ``run_id`` is ``{10-char-hash}-r{rep}``. The hash comes from the
resolved config for that cell, so changing any swept parameter changes
the hash.

.. code-block:: text

   outputs/sweep_tutorial/reference/
     a1b2c3d4e5-r0/   (value_ms=10, rep 0)
     a1b2c3d4e5-r1/   (value_ms=10, rep 1)
     b2c3d4e5f6-r0/   (value_ms=30, rep 0)
     b2c3d4e5f6-r1/   (value_ms=30, rep 1)
     c3d4e5f6a7-r0/   (value_ms=60, rep 0)
     c3d4e5f6a7-r1/   (value_ms=60, rep 1)

Step 4: Resume a partial sweep
--------------------------------

Interrupt the sweep with Ctrl-C after the first two runs, then re-run the
same command::

    PYTHONPATH=. cadet sweep sweep_tutorial.yaml \
        --app-module examples.reference.app

Completed cells (``run.json`` status ``"complete"``) are skipped. Only the
remaining cells run. A cell with ``"status": "failed"`` is retried.

Step 5: Reproduce a run
-------------------------

Each run saves its resolved config as ``config.yaml``. Re-run any cell::

    PYTHONPATH=. cadet run \
        outputs/sweep_tutorial/reference/a1b2c3d4e5-r0/config.yaml \
        --app-module examples.reference.app

The run_id and output are identical to the original because the config
content (and therefore the hash) has not changed.
