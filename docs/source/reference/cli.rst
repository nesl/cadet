CLI Reference
=============

All commands accept ``--set KEY=VALUE`` to override config fields by dotted
path and ``--app-module MODULE`` to load an app plugin. Multiple ``--set``
and ``--app-module`` flags are allowed. The environment variable
``CADET_APP_MODULES`` (colon-separated) is equivalent to ``--app-module``.

----

cadet validate
--------------

Check a config file against the schema and the implemented-feature gate.

.. code-block:: text

   cadet validate <config> [--set KEY=VALUE ...] [--app-module MODULE ...]

Exits 0 if the config is valid and all requested features are implemented.
Exits non-zero and prints a message for each problem found.

.. code-block:: bash

   cadet validate examples/reference/synthetic.yaml

----

cadet show
----------

Print the resolved config (after ``extends:`` merging and ``--set``
overrides).

.. code-block:: text

   cadet show <config> [--set ...] [--app-module ...] [--run N] [--resolved]

``--run N`` selects a specific sweep cell. ``--resolved`` also prints the
computed run IDs.

.. code-block:: bash

   cadet show examples/reference/synthetic.yaml --resolved

----

cadet run
---------

Run a single experiment (one sweep cell, one repetition by default).

.. code-block:: text

   cadet run <config> [--set ...] [--app-module ...] [--run N] [--rep N]
             [--model-dir DIR ...]

``--run N`` selects a sweep cell (0-indexed). ``--rep N`` selects a
repetition. ``--model-dir`` adds a directory to the model artifact search
path.

.. code-block:: bash

   PYTHONPATH=. cadet run examples/reference/synthetic.yaml \
       --app-module examples.reference.app

----

cadet sweep
-----------

Run all cells in the ``sweep.axes`` Cartesian product. Skips cells whose
``run.json`` already has ``"status": "complete"``.

.. code-block:: text

   cadet sweep <config> [--set ...] [--app-module ...] [--reps N]
               [--model-dir DIR ...]

``--reps N`` overrides ``experiment.repetitions``.

.. code-block:: bash

   PYTHONPATH=. cadet sweep examples/reference/synthetic.yaml \
       --app-module examples.reference.app

----

cadet status
------------

Print the status of the most recent run (reads the state store).

.. code-block:: text

   cadet status

.. code-block:: bash

   cadet status

----

cadet down
----------

Kill all CADET-owned processes, remove ``cadet-*`` containers, clean up
emulation namespaces and netem qdiscs, and clear the state store.

.. code-block:: text

   cadet down

Use after a crash or interrupted run to restore a clean state before the
next experiment.

.. code-block:: bash

   cadet down

----

cadet summarize / cadet ask
---------------------------

Read a run directory, or a tree of them, and put it to a model. Both accept a
single run directory and a sweep directory.

.. code-block:: text

   cadet summarize <path> [--config FILE] [--context-only]
   cadet ask <path> "<question>" [--config FILE] [--context-only]

The ``analysis`` section says where to send the reduced record: it is taken
from ``--config`` when given, otherwise from the run directory's own saved
``config.yaml``. The default is ``analysis.backend: none``, in which state both
commands exit non-zero and print the fields to set. The API key is read from
the environment variable named by ``analysis.api_key_env`` (default
``CADET_FM_API_KEY``); there is no config field for it.

``--context-only`` prints the record that would be sent and sends nothing. It
needs no backend, so it works in the default state.

.. code-block:: bash

   cadet summarize outputs/sensing_placement/ --context-only
   cadet ask outputs/sensing_placement/ "which placement wins above 40 ms?"

----

cadet serve
-----------

Start a single server node (used by the orchestrator internally, or for
manual multi-process setups).

.. code-block:: text

   cadet serve <config> [--set ...] [--app-module ...] [--run N]
               [--node NAME] [--model-dir DIR ...]

----

cadet client
------------

Start a single client node.

.. code-block:: text

   cadet client <config> [--set ...] [--app-module ...] [--run N] [--rep N]

----

cadet load
----------

Start a load-generator node.

.. code-block:: text

   cadet load <config> [--set ...] [--app-module ...] [--run N]

----

cadet emulate
-------------

Run one experiment with roles inside a Mininet-WiFi emulated network.

.. note::

   ``cadet emulate`` must be invoked inside the netwaggle emulation
   container. It is not available in the standard ``cadet:dev`` image.

.. code-block:: text

   cadet emulate <config> [--set ...] [--app-module ...] [--run N] [--rep N]
                 [--model-dir DIR ...] [--interactive]

``--interactive`` drops into the Mininet CLI instead of running the roles.
