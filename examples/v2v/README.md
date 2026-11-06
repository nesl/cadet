# V2V cooperative braking

Vehicle-to-vehicle cooperative emergency braking. A lead vehicle shares a
braking-intent packet so a following vehicle brakes earlier than its own
perception would let it, and the network delay between the two vehicles is the
variable under study. The experiment measures the follower's reaction advantage:
whether it stops in time for a hazard that camera-only perception would have
caught too late.

The example needs CARLA. See
[Tutorial 7](https://nesl.github.io/cadet/tutorials/07-v2v-v2i.html) for the
walkthrough, the link models the paper uses, and the sweep over delay and ego
speed.

```bash
cadet validate examples/v2v/v2v_braking.yaml --app-module examples.v2v.app
scripts/cadet run examples/v2v/v2v_braking.yaml \
    --set source.carla.host=127.0.0.1 \
    --app-module examples.v2v.app
```

## Contents

- [`app.py`](app.py) — the `v2v` app plugin: the monocular range estimate, the
  nearest-in-ROI lead selection, and the TTC-plus-distance brake gate, with the
  client and server runtimes that drive them. Its pure logic is pinned against
  the original in `tests/unit/test_v2v_logic.py`.
- [`v2v_braking.yaml`](v2v_braking.yaml) — the experiment config
  (`plugin: v2v`). `braking_mode: v2v` is the cooperative path, in which the
  follower honours the leader's intent; `braking_mode: perception` is the
  baseline, in which it brakes on the server's TTC decision alone.
- [`pipeline/`](pipeline/) and [`pipeline_ext.py`](pipeline_ext.py) — the same
  experiment as a placed stage pipeline, so detection, decision and control are
  individually placeable.
- [`scenarios/`](scenarios/) — the CARLA ScenarioRunner definition
  (`V2VBraking`) and its scenario driver.
- Parameter defaults shared with the V2I example live in
  [`examples/v2x_params.py`](../v2x_params.py).

See the root [README](../../README.md) for the overall architecture.
