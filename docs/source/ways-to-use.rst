Ways to Use CADET
==================

Three modes of engagement, from least to most invasive:

Reproduce
---------

Run the published experiments from the paper as-is, verify the results, and
compare against your own setup.

**Where to start:** :doc:`quickstart`, then :doc:`tutorials/02-carla-braking`
and :doc:`tutorials/07-v2v-v2i`.

**What you touch:** only ``--set`` flags on the command line to supply your
host IP.

**What you get:** per-frame latency logs and collision outcomes in
``outputs/`` for the emergency-braking, V2V, and V2I scenarios (Figures
3–4, Tables I–II of the paper).

Configure
---------

Change experiment parameters — link model, object detector, scenario speed,
weather — without modifying CADET's core.

**Where to start:** :doc:`tutorials/05-sweeps`, :doc:`howto/link-distributions`,
:doc:`howto/model-registry`.

**What you touch:** YAML config files and sweep axes. Everything under
``links``, ``nodes``, ``models.registry``, and ``source.scenario`` is fair
game without touching Python.

**What you get:** sweep outputs across a Cartesian product of parameter
values; each cell's results land under a content-derived run ID so nothing
overwrites anything.

Extend
------

Write a new app plugin, add a data source, or build a new pipeline stage.

**Where to start:** :doc:`tutorials/06-write-an-app`, then
:doc:`explanation/processing-core-layer` for the pipeline API.

**What you touch:** a Python module that implements the app interface; optionally
new pipeline stages under ``cadet/pipeline/stages/``.

**What you get:** CADET's orchestration, sweep engine, metrics collection, and
network emulation applied to your own perception logic.

----

See :doc:`research-goals/index` to map a research question to one of these
modes.
