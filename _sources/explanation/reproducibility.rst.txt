Reproducibility
===============

CADET makes every run addressable, resumable, and verifiable by deriving a
stable identifier from the experiment's content rather than its wall-clock time.

See :doc:`architecture` for the orchestrator lifecycle, and
:doc:`../config-reference` for the ``sweep`` and ``outputs`` schema.

----

Run ID derivation
-----------------

Every run is identified by a ``run_id`` of the form::

    {10-char SHA-256 of resolved config}-r{repetition}

The hash covers the resolved config **after** ``extends:`` merging and
all ``--set`` overrides are applied. The same config file with the same overrides
always produces the same hash, and therefore the same ``run_id``, regardless
of when or where it is run.

This has two consequences:

1. **Sweep resumption.** ``cadet sweep`` checks whether each ``(run_index, rep)``
   cell's output directory already contains a ``run.json`` with
   ``"status": "complete"`` before starting it. If it does, the cell is skipped.
   A sweep interrupted partway through can be restarted with the same command.

2. **Conflict detection.** Two configs that differ in any field produce different
   hashes. If you change a parameter and re-run, the old results stay untouched
   under their original ``run_id``; the new run writes to a new directory.

Two sections are deliberately **excluded** from the hash, because they say where
a result goes or how it is read afterwards rather than what the experiment was:

``outputs``
    Where results are written is not part of a run's identity. The same
    experiment written to two directories is one experiment.

``analysis``
    The model backend used to summarise or question a finished run
    (:doc:`analysis-layer`). It is read *after* a run is over and changes
    nothing the run does. Including it would be actively harmful: naming an
    endpoint would give every cell of a sweep a new ``run_id``, so ``--resume``
    would re-run work that was already finished and existing output trees would
    stop matching their own saved configs.

The rule to take from that: a field belongs in the hash when changing it would
change the measurement. Both exclusions are pinned by tests, and
``cadet/config/loader.py`` holds the list.

A ``run_id`` is stable across machines and interpreters, and **not across
versions of CADET**
...............................................................................

The hash covers the resolved config *as the running version's schema reads it*.
Adding a field, removing one, or changing a default therefore moves every
``run_id`` - uniformly, not selectively. Recomputing an old run's id from its own
saved ``config.yaml`` with newer code gives the id that newer code would assign
to the same cell, which is a different number.

This is measurable rather than theoretical. On this repository, 53 of 74 recorded
run directories no longer recompute to their own id, and in each case the value
they recompute to is exactly the id a later sweep of the same cell was given.
Nothing is corrupt and no key was dropped - every schema model sets
``extra="forbid"``, so an unknown key is rejected rather than silently ignored,
and a dropped key would have made several cells collapse onto one hash, which is
not what happens.

So ``run.json`` records the commit that produced the run::

    "cadet": {"commit": "6f5c6e1d4d14077fce97430a97596abcecff917f", "dirty": false}

To check a recorded id, recompute it **against that commit**. ``dirty`` says the
working tree had uncommitted changes, in which case the commit alone does not
identify the code that ran and the id is not reproducible from the repository at
all. A run directory with no ``cadet`` block predates this field or was produced
from a copy that is not a git checkout.

What this does and does not promise: the same config and the same version give
the same id anywhere, which is what resumption and conflict detection rely on.
Comparing ids across versions is not meaningful, and the saved ``config.yaml``
remains the authority on what a run actually ran.

----

The saved resolved config
--------------------------

The orchestrator writes the resolved config to the run's output directory
as ``config.yaml`` alongside ``run.json``. This means:

- The parameters of a past run are always recoverable from its output dir,
  even if the source config has since changed.
- Reproducing an old run is ``cadet run outputs/{experiment}/{app}/{run_id}/config.yaml``.

----

Config hierarchy
----------------

Configs compose through ``extends:``, resolved at load time:

.. code-block:: yaml

   extends: configs/base.yaml   # deep-merged first, values here override
   experiment:
     name: my_experiment
     repetitions: 3

``configs/base.yaml`` in the repository holds the minimal defaults (transport,
metrics, output naming). Experiment configs only set what differs.

``--set`` overrides are applied last, after ``extends:`` resolution:

.. code-block:: bash

   cadet run config.yaml --set source.fps=20 --set experiment.repetitions=3

Dotted paths address arbitrarily nested fields. The override is reflected in the
hash, so ``--set source.fps=20`` produces a different ``run_id`` from
``--set source.fps=10``.

----

Sweeps
------

A sweep defines a Cartesian product of parameter values:

.. code-block:: yaml

   sweep:
     axes:
       links[0].network.distribution.p50_ms: [30, 60, 120]
       scenario.ego_speed: ["20mph", "30mph"]

This produces 3 × 2 = 6 run configs. ``cadet sweep config.yaml`` runs each cell.
Each cell has its own ``run_id`` derived from its resolved config, so all six
sets of results land in separate output directories and can be resumed
independently.

The sweep axes use dotted paths with list-index notation (``links[0]`` addresses
the first element of ``links``). Any field in the schema can be swept.

----

Repetitions
-----------

``experiment.repetitions`` controls how many times each sweep cell is repeated.
A cell with ``run_id`` base ``abc1234567`` and 3 repetitions produces::

    abc1234567-r0
    abc1234567-r1
    abc1234567-r2

each in its own subdirectory. ``experiment.seed`` is passed to the sources that
take one, and to the delay lines, so a repetition draws the same network
samples. A simulator's own stochastic choices stay the simulator's, which is
what makes repetitions of a scenario independent samples of it.
