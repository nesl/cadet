# Braking — pipeline form

The same open-loop emergency-braking experiment as
[`examples/braking/flat`](../flat/README.md), expressed as a placed stage pipeline
on `cadet/pipeline`. Identical decisions and commands to the flat app (the stages are
ported verbatim); the difference is that detection, decision and control are now
separate, individually placeable stages.

## Stage graph and placement

```
 client                         || boundary ||            server                      || boundary ||   client
 ─────────                                                ──────────────────                             ─────────
 source (camera)  ───frame───►   [client→server]   ───►   yolo ─► roi_consecutive   ───decision───►      brake_controller ─► ego
   (role source)                  latest_only                (detect)  (decide)        [server→client]      (control)
```

| stage | impl | node | data source | role |
|---|---|---|---|---|
| camera | `camera` | client | ground truth (camera) | capture frames (pipeline head) |
| detect | `yolo` | server | perceived | detections from the frame |
| decide | `roi_consecutive` | server | perceived | brake after N in-ROI detections |
| control | `brake_controller` | client | ground truth (ego telemetry) | cruise, then decelerate on brake |

- **Placement matches the flat split**: detection + decision on the `server`, control
  on the `client` (which owns the ego). The two `||` boundaries are inserted
  automatically where consecutive stages change node and are realised as transport
  links (the same client↔server link the flat app uses).
- **Queue policy**: `latest_only` across the client→server frame boundary (keep the
  newest frame), matching the flat channel's keep-latest behaviour.
- The **camera source is the client's configured `source`** (see the `source:`
  section); the first client→server boundary is the source→`detect` hop.

## Run

```bash
cadet run examples/braking/pipeline/openloop_braking.yaml \
  --stage-module cadet.pipeline.stages.braking \
  --set source.carla.host=<HOST_IP> \
  --model-dir /path/to/weights
```

(`--stage-module` imports the braking stage library so the registry can resolve the
`impl` names.)

## Parity with the flat app

- **Level 1 (offline, exact):** the decision (`roi_consecutive`) and control
  (`brake_controller`) stages are byte-for-byte ported from the flat app and are
  checked against the flat logic over identical input sequences in
  `tests/unit/test_braking_pipeline_parity.py` (same brake frame, same per-tick
  commands). Recording real CARLA runs and replaying their inputs is the
  higher-fidelity form of the same check.
- **Level 2 (live):** the same comparison in CARLA, three flat runs against three
  pipeline runs, read against the flat run-to-run spread.

## Behaviour

Decisions and commands match the flat app exactly (the stages are ported verbatim).
Both forms issue **one control command per simulation tick** and use a **two-phase
client start** — see "Deliberate changes from the original" in
[`../flat/README.md`](../flat/README.md).
