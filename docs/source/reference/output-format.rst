Output Format
=============

Every run writes its artifacts to a directory under ``outputs.root``, named
according to ``outputs.naming``. The default naming template is::

    {experiment}/{app}/{run_id}

so a run named ``braking_sweep`` using the ``braking`` app with run ID
``abc1234567-r0`` writes to::

    outputs/braking_sweep/braking/abc1234567-r0/

See :doc:`../explanation/reproducibility` for how ``run_id`` is derived, and
:doc:`../reference/metrics` for the metric file formats.

----

Directory structure
-------------------

.. code-block:: text

    outputs/
    └── {experiment}/
        └── {app}/
            └── {run_id}/
                ├── run.json                           # completion marker
                ├── config.yaml                        # resolved config (reproduces this run)
                ├── client_outcome_{node}.json         # per-client outcome (one per client node)
                ├── link_report_{node}.json            # network backend report (one per client node)
                ├── {run_id}_latency_location_log.csv  # per-frame latency
                ├── server_metrics_{node}.csv          # server-side metrics (if collector enabled)
                ├── queue_log.csv                      # queue depth (if collector enabled)
                └── logs/
                    ├── server_{node}.log
                    ├── client_{node}.log
                    └── {sim_hook}.log

----

``run.json``
------------

Written when the run completes (whether successfully or not). Contains:

.. code-block:: json

   {
     "run_id": "86f44c5d7a-r0",
     "status": "complete",
     "detail": "workers exited: exit codes [0]",
     "reason": "workers exited",
     "startup_s": 16.69,
     "active_s": 0.41,
     "teardown_s": 0.01,
     "limit": null
   }

``status`` is ``"complete"`` on success, ``"failed"`` otherwise.

``reason`` is one of:

- ``"workers exited"``: all workers exited cleanly (``end_on`` condition met).
- ``"active time limit"``: ``experiment.max_active_s`` was reached.
- ``"startup time limit"``: ``experiment.max_startup_s`` was reached.
- ``"node exited unexpectedly"``: a guard process (server or sim hook) exited with a non-zero code.
- ``"sim hook exited"``: a sim hook exited cleanly and ``end_on`` is ``sim_hook_exit`` or ``first``.
- ``"interrupted"`` — the orchestrator received SIGTERM or SIGHUP (the only two
  signals it handles; both are converted to a clean teardown).

``limit`` is ``"startup"``, ``"active"``, or ``null``. It is non-null only when the run was cut off by a time limit.

``startup_s`` / ``active_s`` / ``teardown_s`` are the wall-clock durations of each phase.

The sweep orchestrator reads ``run.json`` to decide whether a cell can be
skipped on resume. A missing or malformed ``run.json`` is treated as incomplete.

----

``config.yaml``
---------------

The resolved config used for this run: ``extends:`` merged, ``--set``
overrides applied, sweep cell values substituted. Re-running with this file
as the config reproduces the run::

    cadet run outputs/braking_sweep/braking/abc1234567-r0/config.yaml

----

``logs/``
---------

One log file per process spawned by the orchestrator. Log files capture the
process's stdout and stderr. If a run fails, the relevant log is the first
place to look.

The server log always contains the line ``"Waiting for connection"`` at the
point where the orchestrator unblocks and starts the sim hooks and clients.
If that line is absent, the server failed to start.
