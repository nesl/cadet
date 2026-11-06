Register per-device model artifacts
=====================================

``models.registry`` maps a logical model name to artifact paths for each
device type. At run time the orchestrator reads the server node's ``device``
field and picks the matching artifact path. The same config runs on
different hardware without edits.

Config
------

.. code-block:: yaml

   models:
     model: pedestrian_v2
     registry:
       pedestrian_v2:
         gpu:   /models/yolo11x.pt
         edge:  /models/yolo11n.pt
         hailo: /models/yolo11n.hef
     warmup: true

- ``model``: the logical name to activate. Must be a key in ``registry``.
- ``registry``: maps logical names to per-device artifact paths. Device
  keys match the ``device`` field on each node (``gpu``, ``edge``,
  ``hailo``, ``cpu``, etc.).
- ``warmup`` — when ``true`` (the default), runs one inference before the
  experiment starts. This loads weights onto the GPU, triggers cuDNN
  kernel selection, and avoids the cost on the first real frame.

Node device assignment
----------------------

Set ``device`` on the server node to select the artifact:

.. code-block:: yaml

   nodes:
     - name: server
       host: 127.0.0.1
       port: 15010
       role: server
       device: gpu

If ``device`` is ``cpu`` and the registry has no ``cpu`` entry, the
orchestrator raises a ``PreflightError`` naming the missing artifact.

Model search path
-----------------

Artifact paths are resolved relative to directories passed with
``--model-dir``. Pass multiple directories to search in order::

    cadet run config.yaml --model-dir /models/v2 --model-dir /models/v1

The orchestrator resolves a relative path against each directory in order
and uses the first match.

Warmup detail
-------------

The warmup inference runs on a blank frame of the configured source
dimensions before the experiment timer starts. It covers weight transfer
to the GPU, model fusing, and cuDNN setup. Without warmup the first real
frame's latency includes all of these costs.
