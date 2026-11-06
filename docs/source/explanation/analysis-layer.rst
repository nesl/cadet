Analysis Layer
==============

The analysis layer turns a config into a running experiment, collects
metrics, and writes results to disk.

Orchestrator
------------

The orchestrator starts all processes in a fixed order and tears them down
in reverse. Its startup sequence is: cleanup, preflight, netem install,
servers, sim hooks, clients and load workers, wait for workers, teardown.

Each phase has a time budget. ``experiment.max_startup_s`` bounds
everything up to and including the readiness waits; its default is the sum
of all configured readiness timeouts (or 120 s if none are set).
``experiment.max_active_s`` bounds the worker phase; its default is
``source.scenario.timeout_s + 60`` or 600 s.

A run ends when the ``end_on`` condition fires. The default,
``client_report``, waits for all client workers to exit. Setting
``end_on: sim_hook_exit`` ends the run when the sim hook exits cleanly
(after an ``end_grace_s`` grace period, default 5 s) even if clients are
still running.

The orchestrator handles SIGTERM and SIGHUP: both signals turn into a
``RunInterrupted`` exception so teardown always runs.

``cadet down`` kills all CADET-owned processes, containers (names starting
with ``cadet-``), emulation namespaces, and netem qdiscs, then clears the
state store. Use it to restore a clean state after a crash.

Metrics
-------

Two files are written on every run: ``{run_id}_latency_location_log.csv``
(per-frame latency, app-dependent columns) and ``link_report.json``
(configured vs measured network parameters). An optional queue-depth
collector is enabled with ``metrics.collectors: [queue]``.

Config and sweeps
-----------------

One YAML config drives everything. ``extends:`` composes configs from a
base file, deep-merging fields in order. ``--set KEY=VALUE`` overrides
apply last and are reflected in the run hash.

``sweep.axes`` defines a Cartesian product of parameter values. ``cadet
sweep`` runs each cell, skipping cells whose ``run.json`` has
``"status": "complete"``. A cell with ``"status": "failed"`` is retried.

Each run's resolved config is saved as ``config.yaml`` in its output
directory. Re-running with that saved config reproduces the run.

FM interface
------------

``cadet summarize <path>`` and ``cadet ask <path> "<question>"`` read what a
run already wrote and put it to a model. ``<path>`` is a single run directory
or a directory holding run directories, in which case it is grouped into one
matrix per sweep - a sweep is a directory of runs - exactly as the dashboard
groups it.

What is sent is a *reduction*, never the raw files: the resolved config's
topology and network models, the run marker's status and phase durations, the
measured side of each link report, the per-request CSV reduced to a median and
a p99 per stage, the per-frame record's task metric, every numeric leaf of the
application's own ``*_summary.json``, and the per-node device roll-up. One run
comes to a few kilobytes and a sweep of nine runs to a few tens of kilobytes;
``analysis.max_context_bytes`` bounds it, and a record that has to be reduced
to fit says so in its own first line. ``--context-only`` prints the record and
sends nothing, which is how to see what would be sent before sending it.

A node's address, account and SSH identity are not part of the record, and
neither is anything that looks like a credential: there is **no config field
for an API key**. ``analysis.api_key_env`` names an environment variable, and
the key is read from the environment at the moment of the request. A failed
request names the endpoint and the HTTP status and never a header.

The default is ``analysis.backend: none``, in which state both commands refuse
and print the three fields to set. ``openai_chat`` is any OpenAI-compatible
chat-completions endpoint, or a local model served behind the same route shape;
it is one HTTP POST and adds no dependency. Both subcommands have answered
from a run directory against a model served locally behind that route shape,
with the record for a single run coming to 2,905 bytes against a default budget
of 200,000.

The ``analysis`` section is excluded from a run's identity hash, as ``outputs``
already was: where a question is sent is not a fact about the run that answered
it.
