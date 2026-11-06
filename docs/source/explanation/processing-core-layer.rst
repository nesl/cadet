Processing Core Layer
=====================

The processing core hosts the model and returns inference results to the
client. A server node receives a frame, runs inference, and sends back
detections or predictions.

Model registry
--------------

``models.registry`` maps a logical model name to per-device artifact paths:

.. code-block:: yaml

   models:
     model: pedestrian_v2
     registry:
       pedestrian_v2:
         gpu:   /models/yolo11x.pt
         edge:  /models/yolo11n.pt
         hailo: /models/yolo11n.hef

At run time the orchestrator reads the server node's ``device`` field
(``gpu``, ``edge``, ``hailo``, ``cpu``) and passes the matching artifact
path to the server process. The same config runs on different hardware
without edits.

Warmup
------

``models.warmup: true`` (the default) runs one inference on a blank frame
before the experiment timer starts. The warmup covers weights transfer to
the GPU, model fusing, and cuDNN kernel selection. Without it, the first
real frame bears all of these costs, inflating its measured latency.

Pipeline machinery
------------------

The ``pipeline`` config key is in the schema. A placed-stage pipeline is
declared as ``pipeline.stages``: an ordered list where each entry names a
stage implementation, assigns it to a node, and sets its queue policy and
deadline budget.

The builder (``cadet/pipeline/builder.py``) groups consecutive same-node
stages into *segments* that run in-process. A change of node is a
*boundary* that the runtime realises as a transport link: the network layer
can emulate that link, so the same ``pipeline`` config works whether the
two nodes are on the same host or different ones.

Config schema for a two-stage braking pipeline:

.. code-block:: yaml

   pipeline:
     stages:
       - name: detect
         impl: yolo
         node: server
         params:
           conf_thresh: 0.5
         queue_policy: latest_only
         deadline:
           policy: static
           budget_ms: 30.0

       - name: decide
         impl: roi_consecutive
         node: client
         queue_policy: latest_only

The ``impl`` value is a registry key. The ``cadet/pipeline/registry.py``
module resolves it to a :class:`~cadet.pipeline.stage.Stage` subclass at
load time. Stage impls are registered with
:func:`~cadet.pipeline.registry.register_stage`.

The stage library
~~~~~~~~~~~~~~~~~

Nine implementations are registered. Braking is ``camera``, ``yolo``,
``roi_consecutive`` and ``brake_controller``; V2V adds ``yolo_v2v`` and
``v2v_ttc_gate``; V2I adds ``yolo_v2i``, ``v2i_ego_gate`` and
``v2i_fusion_gate``. A ``pipeline`` block naming any other ``impl`` is refused
at validate with the name it did not recognise.

The stage modules are imported by ``--stage-module`` (or
``$CADET_STAGE_MODULES``, which spawned roles read), so a braking pipeline is
validated and run with ``--stage-module cadet.pipeline.stages.braking``. The
registry resolves each ``impl`` once those modules are imported.

The component library those stages draw on (``examples/av_pipeline/``) is
ported from Pylot (Apache-2.0; see ``NOTICE``) and holds the detection,
tracking, prediction and control modules.

ERDOS is not used. The pipeline code was decoupled from the ERDOS dataflow
runtime as part of the port. See ``NOTICE`` for the full attribution.

What the schema supports today
-------------------------------

- ``models.warmup``: run one inference before the experiment
- ``models.model``: activate a named registry entry
- ``models.registry``: per-device artifact paths
- ``pipeline.stages``: a placed-stage pipeline, with each stage's ``impl``
  resolved from the registry

The ``app.params`` block in each app config carries any additional
per-model parameters (confidence threshold, ROI, etc.) outside the main
schema.
