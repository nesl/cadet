App Plugin Guide
================

CADET is app-agnostic.  All AV-specific logic (model inference, brake
controllers, detection gating) lives under ``examples/``, not in ``cadet/``.
An app is a Python module that registers one or more :class:`~cadet.core.apps.AppPlugin`
subclasses; the framework calls its methods at the right points in the run
lifecycle.

The AppPlugin interface
-----------------------

Every app subclasses :class:`~cadet.core.apps.AppPlugin` and sets two class
attributes:

``name`` (str)
    The plugin key referenced in the config's ``app.plugin`` field.

``params_model`` (Type[BaseModel])
    A Pydantic model that validates the ``app.params`` block in the config.
    The framework calls ``params_model(**config.app.params)`` before the run
    starts; an invalid parameter set is reported as a validation error, not a
    runtime crash.

There is one optional override:

``sim_hooks(config, model_search_dirs) → list[LaunchSpec]``
    Return processes that must be running before servers or clients start
    (typically a simulator hook such as ScenarioRunner).  The orchestrator starts
    them first and tears them down last.  The default implementation returns an
    empty list.

Registration
------------

Decorate the plugin class with :func:`~cadet.core.apps.register_app`::

    from cadet.core.apps import AppPlugin, register_app
    from pydantic import BaseModel

    class MyParams(BaseModel):
        threshold: float = 0.5
        max_frames: int = 100

    @register_app
    class MyApp(AppPlugin):
        name = "my_app"
        params_model = MyParams

Registration happens as a side-effect of importing the module.  Pass the module
path to the CLI with ``--app-module``::

    cadet run config.yaml --app-module my_package.my_app

The ``app.plugin`` field in the config must match the plugin's ``name``.

Config side
-----------

In the YAML config, declare the plugin and any parameters::

    app:
      plugin: my_app
      params:
        threshold: 0.7
        max_frames: 200

The framework validates ``params`` against ``params_model`` before dispatch.

The reference example (minimal)
--------------------------------

``examples/reference/app.py`` is the smallest useful app.  It exercises the
full framework (source, client/server runtimes, transport, model registry,
metrics) without any AV concepts.

* The **client** pulls frames from any source and sends each one.
* The **server** runs a model (a cheap stub for tests, or real YOLO) and
  replies to *every* frame with the result and its processing time.
* The client logs each reply with its round-trip latency to a CSV.

``ReferenceParams`` has two fields: ``model_kind`` (``"stub"`` or ``"yolo"``)
and ``process_ms`` (optional artificial server work time for load experiments).

Because the server replies to every frame, the reference app produces a true
per-frame round-trip latency distribution — useful for measuring network
emulation effects in isolation.

The braking example (AV)
-------------------------

``examples/braking/flat/app.py`` is the reference AV application: open-loop
emergency braking.  An ego vehicle drives toward an obstacle in CARLA; camera
frames are streamed to a detection server; after a configurable number of
consecutive in-ROI detections the server issues a single brake command; the
client runs a per-vehicle deceleration controller.

Key design points:

* **Server replies once** (the brake command).  There is no per-frame
  round-trip latency; the ``brake_reaction_latency_s`` metric pairs the brake
  reply with the capture time of the decisive frame.
* **All AV logic stays in** ``examples/braking/flat/app.py``.  The framework owns
  the loop, transport, warm-up, and device selection.
* ``BrakingParams`` includes typed sub-models for brake controller gains
  (``BrakeController``) and the ROI pixel band (``RoiBand``).
* ``BrakingApp.sim_hooks`` returns a ``LaunchSpec`` for ScenarioRunner when
  ``CADET_SCENARIO_RUNNER_CMD`` is set in the environment.

``BrakingParams`` fields
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 15 35

   * - field
     - type
     - default
     - notes
   * - ``consecutive_detections``
     - int
     - ``3``
     - detections needed before issuing the brake command
   * - ``roi``
     - RoiBand
     - (required)
     - pixel row range selecting the relevant obstacle
   * - ``controllers``
     - map[str → BrakeController]
     - (required)
     - per-vehicle deceleration gains keyed by ``scenario.ego_vehicle``
   * - ``conf_threshold``
     - float
     - ``0.5``
     - YOLO confidence gate

Writing a new app
-----------------

1. Create a module under ``examples/`` (or your own package).
2. Define a Pydantic params model.
3. Subclass :class:`~cadet.core.apps.AppPlugin`, set ``name`` and
   ``params_model``, and decorate with :func:`~cadet.core.apps.register_app`.
4. Implement the server and client logic.  The framework provides the
   transport layer; the app decides what to send and what to do with replies.
5. Optionally override ``sim_hooks`` if your app needs a simulator hook.
6. Pass ``--app-module your.module`` to the CLI.

The reference app (``examples/reference/app.py``) is the recommended starting
point: it is short, well-documented, and covered by the end-to-end tests.
