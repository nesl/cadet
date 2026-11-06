Quickstart
==========

Install, then run the fastest smoke test — two synthetic sensors, no simulator,
no GPU, no download::

    pip install -U pip && pip install -e ".[examples]"
    cadet run examples/distributed_sensing/quick.yaml --app-module examples.distributed_sensing.app

That runs 40 frames through the distributed-sensing example and exits in
about 2.4 s with a median end-to-end latency of 88.9 ms.

Install
-------

**Python (editable install)**

Requires Python 3.8 or later. Clone the CADET repository, then from inside the
clone::

    pip install -U pip        # Python 3.8's bundled pip cannot do editable installs
    pip install -e ".[examples]"

The ``examples`` extra adds numpy and OpenCV so the no-simulator examples run
without a separate download. To run the test suite::

    pip install -e ".[test]"

**Docker (for CARLA experiments)**

Build the image for configs that launch the CARLA simulator::

    docker build -f docker/Dockerfile -t cadet:dev .

The image includes Python 3.8, the CARLA client egg, and a docker client (no
daemon). Use ``scripts/cadet`` to run CARLA configs; see `Run with CARLA`_ below.

Validate a config
-----------------

Check a config before running it::

    cadet validate examples/distributed_sensing/quick.yaml --app-module examples.distributed_sensing.app

If a config is invalid, ``cadet validate`` names the setting and explains why.

Run without a simulator
-----------------------

The quick config is the fastest smoke test — two synthetic sensors, one
gateway, no CARLA, no weights::

    cadet run examples/distributed_sensing/quick.yaml --app-module examples.distributed_sensing.app

For the reference round-trip latency test::

    cadet run examples/reference/synthetic.yaml \
        --app-module examples.reference.app

Run the placement sweep
-----------------------

``placement_sweep.py`` runs the full three-placement comparison in one command
(sensor vs gateway vs cloud, across three access qualities)::

    python examples/distributed_sensing/placement_sweep.py --sensors 5

That shells out to ``cadet sweep`` twice, reduces every run to one row in
``summary.csv``, prints the table, and writes the crossover plot. Outputs land
under ``outputs/sensing_placement_5/``. Takes about 75 s for five sensors.

While the sweep is running, open the dashboard live view in another terminal::

    cadet dashboard outputs/sensing_placement_5

The dashboard binds to ``127.0.0.1:8787`` by default and shows a live sweep
matrix that refreshes as runs complete. Stop it with ``Ctrl-C``.

Run with CARLA
--------------

Configs that launch their own simulator need three things the container wrapper
provides: a docker client, Python 3.8 with the CARLA client egg, and the host
network namespace. Use ``scripts/cadet`` for those configs::

    export CADET_SCENARIO_RUNNER_DIR=/path/to/scenario_runner
    export CADET_CARLA_PYTHONAPI_DIR=/path/to/carla/PythonAPI
    ./scripts/cadet run examples/braking/flat/atspeed_braking.yaml \
        --app-module examples.braking.flat.app \
        --model-dir /path/to/weights

No ``--set source.carla.host`` is needed. ``scripts/cadet`` runs the
orchestrator on the host network namespace, so ``127.0.0.1`` in the config
reaches the CARLA server on the same machine.

Run ``scripts/cadet`` from inside the repository clone. The repository is
mounted at its own host path, not at the image's ``/app``.

Run a sweep
-----------

``cadet sweep`` expands the ``sweep.axes`` matrix and runs each cell::

    cadet sweep examples/distributed_sensing/edge_5.yaml --app-module examples.distributed_sensing.app

Cells whose ``run.json`` already has ``"status": "complete"`` are skipped on
resume. Interrupt and restart the same command to continue.

Override config fields
----------------------

Any dotted config path can be overridden with ``--set``::

    cadet run examples/reference/synthetic.yaml \
        --app-module examples.reference.app \
        --set experiment.repetitions=5 \
        --set source.count=100

Inspect outputs
---------------

Each run writes to ``outputs/{experiment}/{app}/{run_id}/``:

- ``run.json`` — status, timing phases, end reason
- ``config.yaml`` — resolved config; pass it back to reproduce the run
- ``{run_id}_latency_location_log.csv`` — per-frame latency
- ``client_outcome_<node>.json`` — per-client end-of-run summary
- ``link_report_<node>.json`` — configured vs measured per-message latency
- ``logs/`` — per-process stdout and stderr

Output files carry the node name because a run with N clients would otherwise
have N files writing the same name into the same directory.

Check run status without re-running::

    cadet status

Browse sweep results
--------------------

After ``placement_sweep.py`` or a manual sweep, open the dashboard::

    cadet dashboard outputs/sensing_placement_5

The dashboard serves four views: the sweep matrix, a crossover plot, a per-run
view, and a per-node device view. Nothing is fetched from outside; all plots are
inline SVG. Stop it with ``Ctrl-C``.

Port conflicts
~~~~~~~~~~~~~~

Nothing in the example configs needs overriding to run. If preflight refuses a
port conflict, it names the port and the setting that moves it::

    --set source.carla.port=<N>
