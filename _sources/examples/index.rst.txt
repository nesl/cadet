Examples Gallery
================

.. toctree::
   :hidden:

   distributed-sensing
   dataset-vlm

.. list-table::
   :header-rows: 1
   :widths: 22 38 20 20

   * - Example
     - What it demonstrates
     - Source
     - GPU/CARLA
   * - Reference
     - Round-trip latency measurement, network backend comparison, stub
       and real YOLO models
     - Synthetic, Replay
     - No (stub model only)
   * - Braking (open-loop)
     - Emergency braking with YOLO detection, ScenarioRunner scenario,
       brake reaction latency
     - CARLA
     - Yes
   * - Braking (AV pipeline)
     - Full Pylot-derived pipeline: detection, tracking, prediction,
       control
     - CARLA
     - Yes
   * - Distributed Sensing
     - Detector placement study: N sensors, fan-in, access quality sweep,
       age of information
     - Synthetic, Replay
     - No
   * - V2V / V2I
     - Multi-vehicle cooperative braking and infrastructure-assisted
       perception; four operating modes
     - CARLA
     - Yes (multi-host)
   * - Dataset-mode VLM
     - One model served over VQAv2/COCO across device/edge/cloud; accuracy
       and latency per placement
     - Dataset
     - No (stub); Yes (real VLM)

Distributed sensing
-------------------

**Configs:** ``examples/distributed_sensing/``

N sensor nodes stream into one gateway; the detector runs on each sensor, on
the gateway, or in the cloud. Six sweep configs cover three placements, three
sensor counts, and three access qualities. See :doc:`distributed-sensing`.

Reference example
-----------------

**Configs:** ``examples/reference/``

The reference app sends frames from any source to a server running a model
(``model_kind: stub`` for tests, ``model_kind: yolo`` for real inference)
and records per-frame round-trip latency. It is the recommended starting
point for new apps and for measuring network emulation effects in
isolation.

Run the synthetic variant (no GPU)::

    PYTHONPATH=. cadet run examples/reference/synthetic.yaml \
        --app-module examples.reference.app

Braking example (open-loop)
-----------------------------

**Config:** ``examples/braking/flat/openloop_braking.yaml``

An ego vehicle drives toward an obstacle in CARLA. The CADET client
streams camera frames to a YOLO detection server. After three consecutive
in-ROI detections, the server issues a single brake command; the client
applies per-vehicle deceleration gains and records the reaction latency.

Requires a GPU host and a running CARLA 0.9.10 server. See
:doc:`../tutorials/02-carla-braking`.

Braking example (staged pipeline)
----------------------------------

**Config:** ``examples/av_pipeline/``

The full Pylot-derived pipeline: object detection (YOLO), multi-object
tracking (SORT), linear prediction, and an MPC controller. See
:doc:`../explanation/processing-core-layer`.

V2V / V2I
----------

See :doc:`../tutorials/07-v2v-v2i`.

Dataset-mode VLM
-----------------

**Configs:** ``examples/dataset_vlm/``

One dataset node streams VQAv2 questions and images (or COCO detection boxes)
to one inference server. The server loads a single model; the dataset node
scores each answer. The placement (device, edge, or cloud) is the only
experimental variable. See :doc:`dataset-vlm`.
