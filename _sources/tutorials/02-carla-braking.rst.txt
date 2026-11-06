Tutorial 2: CARLA Braking Experiment
=====================================

.. note::

   This tutorial requires a GPU host with a running CARLA 0.9.10 server.

This tutorial runs the open-loop braking experiment: a CADET client
captures frames from a CARLA simulation and sends them to a server running
YOLO detection. When the server detects a pedestrian in the ROI, the
client issues a brake command and records the reaction latency.

Prerequisites
-------------

- Docker and Docker Compose installed on the GPU host.
- An NVIDIA GPU with driver 550 or later and the NVIDIA container toolkit.
- The CADET repository cloned on the GPU host.

Step 1: Start the compose stack
--------------------------------

From the repository root::

    docker compose -f docker/compose.single-host.yml up --build

This starts two services:

- ``carla-sim``: ``carlasim/carla:0.9.10`` running headless on ports
  2000 and 2001 (internal to the compose network only).
- ``cadet`` — the CADET image with ``CADET_CARLA_HOST=carla-sim`` set
  automatically. No ``--set source.carla.host`` flag is needed.

Wait until the CARLA server logs ``Server stopped``
(the first ready message) before proceeding.

Step 2: Run the braking experiment
------------------------------------

The compose stack is enough to check the config against the simulator::

    docker compose -f docker/compose.single-host.yml run cadet \
        validate examples/braking/flat/openloop_braking.yaml \
        --app-module examples.braking.flat.app

The run itself needs one thing the stack does not carry: a ScenarioRunner
checkout, which spawns the ego the client attaches to. Run it with
``scripts/cadet``, which mounts the working tree and the trees named in the
environment::

    export CADET_SCENARIO_RUNNER_DIR=/path/to/scenario_runner
    export CADET_CARLA_PYTHONAPI_DIR=/path/to/CARLA/PythonAPI
    export PYTHONPATH=$CADET_CARLA_PYTHONAPI_DIR/carla:$CADET_SCENARIO_RUNNER_DIR

    ./scripts/cadet run examples/braking/flat/openloop_braking.yaml \
        --app-module examples.braking.flat.app \
        --set source.carla.host=203.0.113.10 \
        --set source.scenario.runner.path=$CADET_SCENARIO_RUNNER_DIR \
        --set source.scenario.name=BrakingScenario_1 \
        --set source.scenario.definition=examples/braking/flat/scenarios/open_loop_braking_scenario.py \
        --set source.scenario.config_file=examples/braking/flat/scenarios/OLBraking.xml \
        --model-dir /path/to/weights

A successful run ends with the brake and stop it was measuring::

    [client] frames_sent=38 brake_t=2.74 stop_t=4.73
    [orch] run 7342a7b93c-r0 ok

The four ``source.scenario`` settings name the scenario the sim hook drives.
Without them no sim hook is started, and the run fails about a minute later
with ``no ego vehicle found`` - there is no ego, because nothing spawned one.
``runner.path`` is your ScenarioRunner checkout; the other three have to agree
with one another, and the braking example's README lists the three sets the
repository ships. ``PYTHONPATH`` carries the client distribution because
ScenarioRunner imports ``agents``, which ships there rather than in the client
egg.

The orchestrator starts in this order:

1. Kills orphan processes from any previous run.
2. Runs preflight (checks ports, model artifacts, throughput).
3. Installs netem qdiscs (if a netem link is configured).
4. Starts the server; waits for ``"Waiting for connection"``.
5. Starts ScenarioRunner as a sim hook; waits for its readiness marker.
   ScenarioRunner spawns the ego vehicle and begins driving the scenario
   clock via ``world.tick()``.
6. Starts the CADET client; it attaches to the ego vehicle and receives
   frames via camera callback.
7. Waits for the client to report completion (``end_on: client_report``).
8. Tears down in reverse order.

Step 3: Examine the output
----------------------------

Results land in ``outputs/openloop_braking/braking/{run_id}/``::

    run.json
    config.yaml
    {run_id}_latency_location_log.csv
    link_report.json
    logs/
        server_server.log
        client_client.log
        scenario_runner.log

Open ``{run_id}_latency_location_log.csv``. The braking app records
``brake_reaction_latency_s`` for the frame that triggered the brake
command. Model warmup runs before the experiment timer starts, so the
first real frame's latency is not inflated by cuDNN setup.

Native (non-compose) setup
---------------------------

If CARLA is running natively, pass its IP address via ``--set``::

    PYTHONPATH=. cadet run examples/braking/flat/openloop_braking.yaml \
        --app-module examples.braking.flat.app \
        --set source.carla.host=203.0.113.10 \
        --set source.scenario.runner.path=/path/to/scenario_runner \
        --set source.scenario.name=BrakingScenario_1 \
        --set source.scenario.definition=examples/braking/flat/scenarios/open_loop_braking_scenario.py \
        --set source.scenario.config_file=examples/braking/flat/scenarios/OLBraking.xml \
        --model-dir /models

``source.carla.host`` flows through to the ScenarioRunner sim hook so both
the CADET client and ScenarioRunner connect to the same CARLA instance.
