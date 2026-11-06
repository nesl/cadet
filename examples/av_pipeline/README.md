# AV pipeline (vendored component library)

The full autonomous-driving pipeline — perception → tracking → prediction →
planning → control — vendored from ERDOS Pylot and rewritten as plain Python (no
ERDOS dataflow graph). It is the **component library** the AV examples draw on, not
a CADET app plugin itself.

```
pipeline/   objects/ camera/ detection/ perception/ prediction/ planning/ control/
utils/      logging, simulation, detection utils (CARLA glue)
params.py   central pipeline configuration (legacy; see note below)
scenarios/  CARLA ScenarioRunner definition (Custom)
```

## Status

- **Not a registered app.** There is no `app.py` here; the reference AV app that is
  wired into the framework is [`examples/braking`](../braking/README.md), which
  reuses pieces of this pipeline.
- **`params.py` is the legacy central config.** It is consumed only by modules
  inside this directory (`pipeline/*`, `utils/logging.py`), never by `cadet/`. It
  will be retired as each consumer moves to the typed per-app params pattern used
  by the braking app.
- **Vendored code — excluded from auto-formatting** to keep its diff against
  upstream Pylot reviewable.

## Run

There is no standalone entry point here; use it via the AV examples. See the
[root README](../../README.md) and [`examples/braking`](../braking/README.md).

Derived and modified from ERDOS Pylot (Apache-2.0); see [`NOTICE`](../../NOTICE).
