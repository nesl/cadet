Tutorial 6: Write an App Plugin
================================

An app is a Python module that registers an :class:`~cadet.core.apps.AppPlugin`
subclass. The framework calls its methods at the right points in the run
lifecycle. All AV-specific logic lives in the app, not in ``cadet/``.

Step 1: Define a params model
------------------------------

Use a Pydantic model to validate ``app.params`` from the config:

.. code-block:: python

   from pydantic import BaseModel

   class MyParams(BaseModel):
       threshold: float = 0.5
       max_frames: int = 100

The framework validates ``params`` before the run starts. An invalid
parameter raises a validation error, not a runtime crash.

Step 2: Register the plugin
-----------------------------

Decorate an :class:`~cadet.core.apps.AppPlugin` subclass with
:func:`~cadet.core.apps.register_app`:

.. code-block:: python

   from cadet.core.apps import AppPlugin, register_app

   @register_app
   class MyApp(AppPlugin):
       name = "my_app"
       params_model = MyParams

``name`` must match the ``app.plugin`` field in the config. Registration
happens as a side-effect of importing the module.

Step 3: Implement the server handler
--------------------------------------

Override the server method to process incoming frames and send a reply:

.. code-block:: python

   @register_app
   class MyApp(AppPlugin):
       name = "my_app"
       params_model = MyParams

       def server(self, config, params, model_path):
           # Called once per server process startup.
           # Load your model here.
           pass

       def handle_frame(self, frame_id, image_bytes, params):
           # Called for each frame received from the client.
           # Return any serialisable result.
           return {"frame_id": frame_id, "detected": False}

Step 4: Implement the client loop
-----------------------------------

Override the client method to drive the source and send frames:

.. code-block:: python

       def client(self, config, params, source):
           for frame_id, image_bytes in source:
               reply = self.send(frame_id, image_bytes)
               # Log reply here.

``source`` is an iterator that yields ``(frame_id, image_bytes)`` tuples.
The framework handles transport, reconnection, and timing.

Step 5: Add a sim hook (optional)
-----------------------------------

Return a ``LaunchSpec`` from ``sim_hooks`` to start a subprocess before
the client connects:

.. code-block:: python

       def sim_hooks(self, config, model_search_dirs):
           import os
           cmd = os.environ.get("MY_SIMULATOR_CMD")
           if not cmd:
               return []
           from cadet.core.launcher import LaunchSpec
           return [LaunchSpec(name="sim", cmd=cmd.split())]

The orchestrator starts sim hooks after servers and before clients, and
tears them down after clients.

Step 6: Wire the config
-------------------------

.. code-block:: yaml

   app:
     plugin: my_app
     params:
       threshold: 0.7
       max_frames: 200

Run::

    PYTHONPATH=. cadet run my_config.yaml --app-module my_package.my_app

Reference app
-------------

``examples/reference/app.py`` is the canonical minimal implementation:
it exercises the full framework (source, transport, model registry,
metrics) without any AV-specific logic. Read it before writing your own.

The braking app (``examples/braking/flat/app.py``) shows how to wire
ScenarioRunner as a sim hook and implement the detection gating and brake
controller logic.
