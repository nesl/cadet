# V2I infrastructure-assisted braking

Infrastructure-assisted emergency braking. A roadside unit (RSU) with its own
fixed camera shares detections with an ego vehicle, extending the vehicle's
effective field of view, and the ingress latency of that help is configurable.
The experiment measures what the infrastructure's view buys, and at what delay
it stops buying anything.

The example needs CARLA. See
[Tutorial 7](https://nesl.github.io/cadet/tutorials/07-v2v-v2i.html) for the
walkthrough, the four modes and the per-mode numbers.

```bash
cadet validate examples/v2i/v2i_braking.yaml --app-module examples.v2i.app
scripts/cadet run examples/v2i/v2i_braking.yaml \
    --set source.carla.host=127.0.0.1 \
    --app-module examples.v2i.app
```

## Contents

- [`app.py`](app.py) — the `v2i` app plugin: the infrastructure-camera
  projection, the fusion of the ego and RSU views, and the TTC gate, with the
  client that listens on both feeds and actuates on whichever carries the brake.
- [`v2i_braking.yaml`](v2i_braking.yaml) — the experiment config
  (`plugin: v2i`), including the infrastructure camera's pose and field of view
  and the TTC settings. `mode` selects the comparison:
  `baseline_local` and `baseline_cloud` are the ego-only baselines,
  `v2i_local` and `v2i_external` add the RSU's view.
- [`pipeline/`](pipeline/) and [`pipeline_ext.py`](pipeline_ext.py) — the same
  experiment as a placed stage pipeline.
- [`scenarios/`](scenarios/) — the CARLA ScenarioRunner definition
  (`V2IBraking`) and its scenario driver.
- Parameter defaults shared with the V2V example live in
  [`examples/v2x_params.py`](../v2x_params.py).

See the root [README](../../README.md) for the overall architecture.
