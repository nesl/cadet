Dashboard
=========

``cadet dashboard RUNS_DIR`` serves a local web UI over the outputs of a
completed or in-progress sweep. It binds to ``127.0.0.1:8787`` by default and
stops with ``Ctrl-C``.

::

    cadet dashboard outputs/sensing_placement_5

Open ``http://localhost:8787`` in a browser. Nothing is fetched from the network;
all plots are inline SVG.

Views
-----

**Sweep matrix**
    One cell per run, coloured by completion status. Click a cell to open that
    run's view. Refreshes automatically while a sweep is running.

**Crossover plot**
    Per-placement latency–accuracy or latency–AoI curves, one series per sweep
    axis value. Useful for reading the crossover point between placements.

**Per-run view**
    Frame-latency CDF, age-of-information series, link report (configured vs
    measured delay), and per-collector time series for the run.

**Per-device view**
    GPU utilisation, CPU load, energy, and bandwidth over time, one panel per
    node that collected them.

Live view during a sweep
------------------------

Start the dashboard in a separate terminal before or after starting the sweep::

    # Terminal 1: run the sweep
    cadet sweep examples/distributed_sensing/edge_5.yaml --app-module examples.distributed_sensing.app

    # Terminal 2: watch it live
    cadet dashboard outputs/sensing_edge_5

The sweep matrix refreshes as each run's ``run.json`` lands. Completed-run views
are available immediately after each run finishes.

Port conflict
~~~~~~~~~~~~~

If port 8787 is in use, override it::

    cadet dashboard outputs/sensing_placement_5 --port 9090

See also
--------

- :doc:`../tutorials/05-sweeps` — running and resuming sweeps
- :doc:`../howto/outputs` — the output directory layout
- :doc:`../reference/metrics` — what each metric file contains
