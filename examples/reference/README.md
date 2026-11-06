# Reference example (non-AV)

The smallest useful CADET app. It exercises the whole framework — source, client
and server runtimes, transport, model registry, metrics — **without any
autonomous-driving concepts**, so it is the fastest way to understand how an app
plugs in and the base for the end-to-end tests.

- The **client** pulls frames from any source and sends each one.
- The **server** runs a model (a cheap stub, or real YOLO) and replies to *every*
  frame with the result plus its processing time.
- The client logs each reply with its round-trip latency to a CSV.

The app and its parameters live in [`app.py`](app.py) (`--app-module
examples.reference.app`); nothing in `cadet/` imports it.

## Configs

| Config | Source | Model | Notes |
|---|---|---|---|
| [`synthetic.yaml`](synthetic.yaml) | `synthetic` (8 generated frames) | stub | No simulator, no GPU — used by the e2e tests. |
| [`load.yaml`](load.yaml) | n/a (load role generates frames) | stub (`process_ms: 3`) | A `load` node drives 4 concurrent clients at 100 Hz to measure server queueing. |
| [`demo_yolo.yaml`](demo_yolo.yaml) | `replay` (a folder of images) | `yolo11x` on GPU | Real inference over a 30 ms ± 8 ms emulated link. |

## Run

```bash
# Validate first.
cadet validate examples/reference/synthetic.yaml

# Run (no simulator required).
cadet run examples/reference/synthetic.yaml

# Replay a folder of images through YOLO (point the source at your images):
cadet run examples/reference/demo_yolo.yaml --set source.path=/path/to/images
```

Outputs are written under `outputs/{experiment}/{app}/{run_id}/`.
