Use the replay source
=====================

The replay source reads a directory of image files or a video file from
disk, feeding frames to the client at a fixed rate. No GPU or simulator
is required.

Images directory
----------------

Set ``source.kind: replay`` and point ``source.path`` at a directory
containing JPEG or PNG files:

.. code-block:: yaml

   source:
     kind: replay
     path: /data/drive_001
     fps: 10.0

Files are read in lexicographic (ASCII sort) order, so ``frame_0001.png``
comes before ``frame_0010.png``. Zero-pad frame numbers if order matters.

Video file
----------

Point ``source.path`` at a video file instead:

.. code-block:: yaml

   source:
     kind: replay
     path: /data/drive_001.mp4
     fps: 10.0

``fps`` controls the rate at which frames are injected into the pipeline.
Set it to match the original capture rate for realistic timing, or lower it
to slow down replay.

Full config example
-------------------

.. code-block:: yaml

   extends: configs/base.yaml

   experiment: {name: replay_test, app: reference, repetitions: 1}
   source:
     kind: replay
     path: /data/drive_001
     fps: 10.0
   nodes:
     - {name: client, host: localhost, role: client, device: cpu}
     - {name: server, host: 127.0.0.1, port: 15010, role: server, device: cpu}
   links:
     - name: inference_link
       from: client
       to: server
       network:
         mode: delay
         distribution: {kind: constant, value_ms: 60.0}
   app: {plugin: reference, params: {model_kind: stub}}

Validate and run::

    PYTHONPATH=. cadet validate replay_test.yaml --app-module examples.reference.app
    PYTHONPATH=. cadet run replay_test.yaml --app-module examples.reference.app
