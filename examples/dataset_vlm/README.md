# Dataset-mode VLM serving: evaluate one model over a dataset, across tiers

One dataset node streams an evaluation dataset to one inference server. The
server loads a **single** vision-language model and answers each sample; the
dataset node scores the answer against the ground truth the dataset carries. The
one thing the example varies is **where the model is served** - on the device, on
an edge box, or in the cloud - so the measured difference belongs to the tier.

This is the paper's **Dataset Mode** (Fig. 5): a model evaluated over a dataset
with model-, system- and task-level numbers recorded per run. It serves one model
only; choosing among models, switching models, or routing a query to a model is
out of scope and appears nowhere in this example.

The example needs no CARLA and no simulator, and its stub path needs no model
download and no GPU, so it runs under a plain Python install. From a fresh clone,
install CADET and the example's extras once, then drive it with the `cadet`
console script (equivalently `python -m cadet`):

```bash
pip install -e .[examples]        # once, in a fresh clone

# Validate, then run the quick config: one dataset node, the built-in VQA set,
# the stub model on the edge tier. No download, no GPU, seconds.
cadet validate examples/dataset_vlm/quick.yaml --app-module examples.dataset_vlm.app
cadet run      examples/dataset_vlm/quick.yaml --app-module examples.dataset_vlm.app

# The detection task on the stub model (built-in boxes, IoU scoring).
cadet run examples/dataset_vlm/detection.yaml --app-module examples.dataset_vlm.app

# The placement sweep: the same model served on each tier in turn.
cadet sweep examples/dataset_vlm/sweep.yaml --app-module examples.dataset_vlm.app
python examples/dataset_vlm/summarise.py outputs/dataset_vlm_sweep/dataset_vlm/*-r0

# Put a run's numbers into the dashboard (writes dataset_vlm_summary.json).
python examples/dataset_vlm/summarise.py --write outputs/dataset_vlm_quick/dataset_vlm/*-r0
cadet dashboard outputs/dataset_vlm_quick
```

## What it measures

Per run, from the per-sample log the dataset node writes:

- **Task score.** For VQA, the VQA accuracy of each answer against the human
  answers, averaged over the samples. For detection, the per-image F1 of the
  predicted boxes against the ground-truth boxes, averaged over the samples.
- **End-to-end latency**, from a sample's capture to its answer arriving back.
- **Server inference time**, reported by the server per sample.

The score is a *task-level* metric; the latencies are *model/system-level*. The
placement trades them: a nearer tier has less network but slower compute.

## The configs

| Config | Task | Model | Placement | Runs | Source |
|---|---|---|---|---|---|
| `quick.yaml` | vqa | stub | edge | 1 | built-in |
| `device.yaml` | vqa | stub | device | 1 | built-in |
| `edge.yaml` | vqa | stub | edge | 1 | built-in |
| `cloud.yaml` | vqa | stub | cloud | 1 | built-in |
| `detection.yaml` | detection | stub | edge | 1 | built-in |
| `sweep.yaml` | vqa | stub | device, edge, cloud | 3 | built-in |
| `real.yaml` | vqa | hf | cloud | 1 | fetched VQAv2 |

`device`/`edge`/`cloud` each carry their own access link (loopback / LAN / WAN)
*and* their own stub inference cost, so they show the full tier. `sweep.yaml`
holds the link fixed and varies only the placement, so it isolates the compute
cost the way the distributed sensing example's placement axis does. The three per-tier configs vary the
network too.

Stub inference cost is a config input, not a measurement: `app.params.infer_ms`
in `_base.yaml` gives the device 200 ms per sample, the edge 80 ms and the cloud
25 ms, standing for a constrained device, an edge box and a cloud accelerator.
The stub model loads no weights and its answer is a fixed keyword rule, so the
score is reproducible with nothing downloaded. Set `model: hf` with a `model_id`
to serve a real model instead (see "Source data" below).

## What the runs showed

Measured on this host with the `delay` backend, the eight-sample built-in VQA
set. The source runs free of a simulator clock, so exact latencies vary a few
percent run to run.

The quick config (`cadet run examples/dataset_vlm/quick.yaml`) answered 8 of 8
samples at **VQA accuracy 0.625**, median end-to-end latency 455.8 ms, 80 ms
server inference, in about 1.2 s of worker time (2 s wall with startup and
teardown).

The placement sweep (`cadet sweep examples/dataset_vlm/sweep.yaml`), summarised:

```
run            task  placement  samples  answered  score  e2e_p50_ms  e2e_p99_ms  infer_p50_ms
06cc2712a6-r0  vqa   cloud      8        8         0.625  187.2       260.2       25.1
d5012016ae-r0  vqa   device     8        8         0.625  1060.6      1654.2      200.2
f9828aeea2-r0  vqa   edge       8        8         0.625  452.2       690.7       80.1
```

Accuracy is identical across the three - it is the same model - while end-to-end
latency tracks the tier's inference cost: the cloud at 25 ms per sample is an
order of magnitude ahead of the device at 200 ms, and that gap, not the network,
dominates here because the payloads are small.

The detection config (`cadet run examples/dataset_vlm/detection.yaml`) answered
8 of 8 with a mean per-image F1 of **0.208** for the fixed stub boxes against the
built-in ground truth - a deterministic mix of matched and missed boxes, which is
the scorer being exercised, not a model's quality.

## Scoring

Both scorers live in `scoring.py`:

- **VQA.** `normalize_answer` ports the answer normalisation from the official VQA
  v2.0 evaluation code (`vqaEval.py`, https://github.com/GT-Vision-Lab/VQA):
  lower-case, strip periods, map number words to digits, drop articles, expand a
  fixed contraction table, and remove punctuation. `vqa_accuracy` is the official
  per-answer metric, `min(matching human answers / 3, 1)`.
- **Detection.** `match_detections` greedily matches predicted to ground-truth
  boxes by IoU at a single threshold (`iou_thresh`, default 0.5), same label
  required, each ground-truth box matched at most once, and returns precision,
  recall and F1. COCO's AP averages over several thresholds; this is the simplest
  documented single-threshold variant.

The ground truth never crosses the wire: the dataset node keeps it and scores the
reply, so the server only serves the model.

## Source data and its license

The stub configs use a **built-in, generated sample set** (`cadet.sources.dataset`
with no `path`): no file, no download, nothing committed. The built-in VQA
questions and the built-in detection boxes are generated in process.

`real.yaml` reads a fetched subset that `fetch_dataset.py` downloads at run time
into `datasets/dataset_vlm/vqa`, a path `.gitignore` already excludes. Nothing
downloaded is committed or redistributed.

```bash
python examples/dataset_vlm/fetch_dataset.py --task vqa --limit 50
cadet run examples/dataset_vlm/real.yaml --app-module examples.dataset_vlm.app
```

The source is **VQA v2.0** (the VQA Consortium). Its questions and annotations
are licensed under **Creative Commons Attribution 4.0 International**
(https://visualqa.org/terms.html), which permits commercial use with attribution,
so an open-source example can depend on them. The images those questions
reference come from the **COCO val2014** split, fetched one at a time from COCO's
own bucket; COCO's terms (https://cocodataset.org/#termsofuse) license the
annotations CC BY 4.0 and state the consortium does not own the images, whose use
must follow the Flickr Terms of Use. The example uses images only, caches them
locally and redistributes none of them - the same license-clean approach as
`examples/distributed_sensing/fetch_frames.py`.

**No authentication token is used anywhere.** The real model is loaded through
transformers with no token; `model_id` must name an openly available model.

## Features used

Everything the configs ask for is marked implemented in
`cadet/config/implemented.py`:

- `source.kind: dataset` (new; see "Core change" below), `task: vqa` and
  `detection`
- `nodes.launcher: local`
- `links.network.mode: delay`
- `transport.protocol: tcp`, `transport.mode: frame`, `transport.encoding: safe`

### Core change (flagged for review)

This example needed one minimal addition to `cadet/` core: a new input source,
`source.kind: dataset` (`cadet/sources/dataset.py`), registered in
`cadet/sources/__init__.py`, declared in `cadet/config/schema.py`
(`DatasetSource`) and gated as implemented in `cadet/config/implemented.py`. It
yields one sample per frame carrying the input image and the sample's ground
truth, which is what a dataset-evaluation source needs and no existing source
provided.

### A note on structure

The example follows the structure of `examples/distributed_sensing`: an `AppPlugin` with a
client runtime (the dataset node) and a server runtime (the model), and a model
runner (`StubVlm` / `HfVlm` / `build_vlm`) that mirrors the distributed sensing example's
`StubDetector` / `YoloDetector` / `build_detector`. The model is placed on a tier
exactly as the distributed sensing example places its detector - by the server node's `device` and the
`placement` param - rather than through the `cadet/pipeline/stages` registry,
because the distributed sensing template this example follows uses the plugin runtimes, not the
pipeline-stage builder.
