Read outputs and run.json
=========================

Directory layout
----------------

Each run writes to::

    outputs/{experiment}/{app}/{run_id}/

Example::

    outputs/ref_synth/reference/86f44c5d7a-r0/
      run.json
      config.yaml
      86f44c5d7a-r0_latency_location_log.csv
      client_outcome_client.json
      link_report_client.json
      logs/
        server_server.log
        client_client.log

run.json
--------

Written on completion (success or failure):

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

``reason`` values:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - reason
     - When it fires
   * - ``"workers exited"``
     - All client/load workers exited cleanly
   * - ``"active time limit"``
     - ``max_active_s`` was reached during the worker phase
   * - ``"startup time limit"``
     - ``max_startup_s`` was reached before workers started
   * - ``"node exited unexpectedly"``
     - A guard process (server or sim hook) exited with a non-zero code
   * - ``"sim hook exited"``
     - A sim hook exited cleanly when ``end_on`` is ``sim_hook_exit`` or ``first``
   * - ``"interrupted"``
     - The orchestrator received SIGTERM or SIGHUP

``limit`` is ``"startup"``, ``"active"``, or ``null``. Non-null means the
run was cut off by a time limit.

``startup_s``, ``active_s``, ``teardown_s`` are wall-clock durations of
each lifecycle phase.

Time limits
-----------

``experiment.max_startup_s``: bounds the cleanup, preflight, netem
install, and readiness-wait phases. Default: sum of all readiness timeouts,
or 120 s.

``experiment.max_active_s`` — bounds the worker phase (default:
``source.scenario.timeout_s + 60`` if set, otherwise 600 s.

``end_on`` controls when the active phase ends:

- ``client_report`` (default): wait for all client workers to finish.
- ``sim_hook_exit``: end when the sim hook exits cleanly; apply
  ``end_grace_s`` (default 5 s) before stopping clients.
- ``first``: end when either clients or the sim hook finishes first.

Latency log
-----------

``{run_id}_latency_location_log.csv`` contains one row per frame. Columns
depend on the app; the reference app writes: ``frame_id``, ``latency_s``,
``server_processing_time``, ``result``.

link_report_{node}.json
-----------------------

Written by each client node at the end of the run. The ``{node}`` suffix is
the node's name from the config, so a run with multiple client nodes produces
one file per client. Written by the client at the end of every run:

.. code-block:: json

   {
     "experiment": "ref_synth",
     "app": "reference",
     "configured_links": [],
     "measured": {
       "reply_policy": "per-message",
       "messages_sent": 8,
       "messages_received": 8,
       "unanswered": 0,
       "unanswered_pct": 0.0,
       "latency_ms": {
         "samples": 8,
         "p50": 19.39,
         "p99": 19.56,
         "mean": 18.63,
         "min": 13.38,
         "max": 19.56
       }
     }
   }

``unanswered`` is non-zero only with ``mode: delay`` and ``loss > 0``.
With ``netem`` or ``netwaggle``, dropped packets are retransmitted by TCP.

Queue collector
---------------

Enable with::

    metrics:
      collectors: [queue]

Writes ``queue_log.csv`` with columns ``t`` (timestamp) and
``queue_depth`` (requests waiting for inference).
