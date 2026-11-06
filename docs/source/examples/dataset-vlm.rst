Dataset-mode VLM serving
========================

One dataset node streams an evaluation dataset (VQAv2 questions and images, or
COCO-style detection boxes) to one inference server. The server loads a single
model and answers each sample; the dataset node scores the answer against the
ground truth carried in the frame. The only axis this example varies is where
the model is served — device, edge, or cloud — so the measured difference
belongs to placement alone.

This is the paper's Dataset Mode (Fig. 5): model-, system- and task-level
numbers per run. Choosing among models, switching models, or routing queries to
different models is out of scope.

No CARLA and no simulator required. The stub model paths run without a GPU or
any download, so the example works from a plain pip install.

Configs
-------

All configs are in ``examples/dataset_vlm/``.

.. list-table::
   :header-rows: 1
   :widths: 28 16 16 16 12

   * - Config
     - Task
     - Model
     - Placement
     - Requires GPU
   * - ``quick.yaml``
     - VQA
     - stub
     - edge
     - No
   * - ``device.yaml``
     - VQA
     - stub
     - device
     - No
   * - ``edge.yaml``
     - VQA
     - stub
     - edge
     - No
   * - ``cloud.yaml``
     - VQA
     - stub
     - cloud (40 ms WAN)
     - No
   * - ``detection.yaml``
     - Detection
     - stub
     - edge
     - No
   * - ``sweep.yaml``
     - VQA
     - stub
     - device / edge / cloud
     - No
   * - ``real.yaml``
     - VQA
     - real VLM endpoint
     - edge
     - Yes (or remote)

Commands
--------

Validate, then run the quick config (no download, no GPU, seconds)::

    cadet validate examples/dataset_vlm/quick.yaml --app-module examples.dataset_vlm.app
    cadet run      examples/dataset_vlm/quick.yaml --app-module examples.dataset_vlm.app

Run the detection task::

    cadet run examples/dataset_vlm/detection.yaml --app-module examples.dataset_vlm.app

Run the placement sweep (device, edge, cloud in turn)::

    cadet sweep examples/dataset_vlm/sweep.yaml --app-module examples.dataset_vlm.app
    python examples/dataset_vlm/summarise.py outputs/dataset_vlm_sweep/dataset_vlm/*-r0

Open the dashboard while the sweep is running::

    cadet dashboard outputs/dataset_vlm_sweep

Outputs
-------

Each run writes the standard CADET outputs under
``outputs/{experiment}/{app}/{run_id}/`` plus one file specific to this example:

``{run_id}_sample_log.csv``
    One row per sample: ``sample_id``, ``answer``, ``correct`` (VQA accuracy or
    detection F1), ``end_to_end_latency_s``, ``server_inference_s``.

The placement sweep produces one ``summary.csv`` with per-placement aggregates
(task score, p50/p99 latency, server inference time) so you can plot
accuracy–latency tradeoffs across tiers.

See also
--------

- :doc:`../explanation/dataset-mode` — how dataset mode works
- :doc:`../tutorials/05-sweeps` — Cartesian sweeps and ``cadet dashboard``
- :doc:`../reference/metrics` — per-run metric files
