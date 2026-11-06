# Braking example (reference AV app)

The reference autonomous-vehicle application: **open-loop emergency braking**. An
ego vehicle drives at a target speed toward an obstacle in CARLA; camera frames
are streamed to a detection server (device / edge / cloud); after N consecutive
in-ROI detections the server issues a brake command and the client runs a
vehicle-specific deceleration controller. Per-frame latency and the
brake-engage / stop locations are logged.

All AV-specific logic (YOLO detection gating, the brake controllers, speed
conversion) lives in [`app.py`](app.py) (`--app-module examples.braking.flat.app`); the
framework owns the loop, transport, warm-up and device selection.

## Files

- [`app.py`](app.py) — the `braking` app plugin: `BrakingServerApp`,
  `BrakingClientApp`, `BrakingParams`, and the pure helpers (unit-tested).
- [`openloop_braking.yaml`](openloop_braking.yaml) — the experiment config; the
  `sweep` section enumerates ego-vehicle × target-object.
- [`scenarios/`](scenarios/) — CARLA ScenarioRunner definitions (`AtSpeed`,
  `OpenLoop`, `OLBraking`) used as the sim-side hook for a run.

## Run

Requires a running CARLA 0.9.10 server and ScenarioRunner (both external). The
orchestrator (`cadet run` / `cadet sweep`) starts the ScenarioRunner sim-hook, the
detection server, and the client, waits for readiness, then tears everything down —
replacing the old hand-written launch scripts.

**Never commit the simulator IP or local paths** — pass them at run time via `--set`
and environment variables:

```bash
# --app-module on every command, validate included: the app lives in
# examples/, and the framework does not import example code by itself.
cadet validate examples/braking/flat/openloop_braking.yaml \
  --app-module examples.braking.flat.app

# Where your ScenarioRunner checkout is. The sim hook runs the tree named here.
export CADET_SCENARIO_RUNNER_DIR=/path/to/scenario_runner

# Your CARLA client distribution. Named here so `scripts/cadet` mounts it, and
# put on PYTHONPATH because ScenarioRunner imports `agents`, which ships in the
# distribution (PythonAPI/carla) rather than in the client egg. Without it the
# sim hook exits with "ModuleNotFoundError: No module named 'agents'".
export CADET_CARLA_PYTHONAPI_DIR=/path/to/CARLA/PythonAPI
export PYTHONPATH=$CADET_CARLA_PYTHONAPI_DIR/carla:$CADET_SCENARIO_RUNNER_DIR

cadet run examples/braking/flat/openloop_braking.yaml \
  --app-module examples.braking.flat.app \
  --set source.carla.host=<HOST_IP> \
  --set source.scenario.runner.path=$CADET_SCENARIO_RUNNER_DIR \
  --set source.scenario.name=BrakingScenario_1 \
  --set source.scenario.definition=examples/braking/flat/scenarios/open_loop_braking_scenario.py \
  --set source.scenario.config_file=examples/braking/flat/scenarios/OLBraking.xml \
  --model-dir /path/to/weights

# Whole vehicle x object matrix, resumable (the same four scenario settings):
cadet sweep examples/braking/flat/openloop_braking.yaml \
  --app-module examples.braking.flat.app \
  --set source.carla.host=<HOST_IP> --model-dir /path/to/weights \
  --set source.scenario.runner.path=$CADET_SCENARIO_RUNNER_DIR \
  --set source.scenario.name=BrakingScenario_1 \
  --set source.scenario.definition=examples/braking/flat/scenarios/open_loop_braking_scenario.py \
  --set source.scenario.config_file=examples/braking/flat/scenarios/OLBraking.xml
```

Put the four scenario settings in a config of your own if you run this often;
they are config fields, not flags, and `--set` is only how this page keeps a
local path out of the repository.

### A scenario is three pieces that have to agree

`name` is an instance declared in `config_file`, and its `type` there is the
class that `definition` defines. The three sets under
[`scenarios/`](scenarios/) are:

| `definition` | class | `config_file` | `name` |
|---|---|---|---|
| `open_loop_braking_scenario.py` | `BrakingScenario` | `OLBraking.xml` | `BrakingScenario_1`, `BrakingScenario_2` |
| `at_speed_braking_scenario.py` | `AtSpeedBrakingScenario` | `AtSpeed.xml` | `AtSpeedBraking_1` |
| `open_loop_scenario.py` | `OpenLoopScenario` | `OpenLoop.xml` | `OpenLoop_1` |

Mixing rows gets `Configuration for scenario <name> cannot be found!` from
ScenarioRunner. The first row is the one the sweep's vehicle × object axes
reach: that scenario reads them, so the swept ego vehicle is the one spawned.

With no scenario named, no sim hook is started — useful for a dry run, but a
CARLA run then has no ego to attach to and fails with `no ego vehicle found`.
`CADET_SCENARIO_RUNNER_CMD` replaces the whole command for a tree whose CLI
differs (`{host}` is substituted). The detection server loads the model named by
`models.model` (e.g. `yolo11x`) from `--model-dir`; keep weights out of git.
Outputs go to `outputs/{experiment}/{app}/{run_id}/` with per-process logs under
`logs/`.

## Behaviour settings

These configs pin the settings below. The pipeline form (`../pipeline/`) honors
the same ones, so the two forms stay in parity.

- **Control rate** — `app.params.control_rate`: `per_loop` (the default: a
  command every client loop iteration, driven by wall-clock time) or `per_tick`
  (one command per world-snapshot tick, brake-ramp driven off the snapshot clock,
  so the deceleration rate is independent of host speed).
- **Model warm-up** — `models.warmup`: `false` in these configs, so the first
  live frame pays the model's cold-start cost; `true` runs one dummy inference at
  startup instead.
- **Server processing delay** — `nodes[*].server_delay_mode`: `blocking` (the
  default: `extra_processing_ms` is a real `time.sleep` on the server thread) or
  `link` (no server sleep; the delay is modelled on a configured reverse link).
- **Camera frame rate** — `source.camera.fixed_rate`: `false` (the default: the
  camera free-runs at the asynchronous server's rate) or `true` (the sensor fires
  once per `1/source.fps` of simulated time, so the sampled ego positions repeat).
- **Inbound queue policy** — `nodes[*].enforce_queue_policy`: `false` (the
  default: one unbounded FIFO, every frame processed) or `true` (the inbound queue
  honors the config's freshness policy, a stage's `queue_policy` or a channel's
  `max_queue`, dropping stale frames).
- **Two-phase client start** — `experiment.client_start` (owned by the
  orchestrator): `before_scenario` does scene-independent setup, prints
  `[client] ready for ego`, then attaches the ego; `after_scenario` attaches after
  the scenario is up.
