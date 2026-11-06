Version Matrix
==============

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Component
     - Version
     - Notes
   * - Python
     - 3.8
     - Required. The CARLA 0.9.10 egg is cp37/cp38; it imports under 3.8
       but not under 3.10 or later.
   * - CARLA simulator
     - ``carlasim/carla:0.9.10``
     - Docker image; pinned in ``docker/compose.single-host.yml``.
   * - CARLA Python client
     - egg from PythonAPI (0.9.10)
     - Not a pip install. ``carla==0.9.10`` was never published to PyPI;
       use the PythonAPI egg (``carla-0.9.10-py3.7-linux-x86_64.egg`` or
       ``carla-0.9.10-py3.8-linux-x86_64.egg``) mounted at
       ``sim.carla_python_api``.
   * - Client/server version matching
     - First three components compared
     - A server image tagged ``0.9.10.1`` is a patch of the same release;
       the egg named ``carla-0.9.10-...`` matches it. Preflight checks the
       first three components and refuses a mismatch.
   * - ScenarioRunner
     - Pinned by content digest
     - ``source.scenario.runner.source_digest`` is a 16-character hex digest
       of the tree's file contents, not a commit id. The ``commit`` field
       is recorded for reference but is never compared.
   * - Mininet-WiFi
     - ``v2.4.3``
     - Pinned in ``docker/netwaggle.Dockerfile`` (``MININET_WIFI_TAG``).
       Required only for the ``netwaggle`` backend.
   * - Mininet core
     - ``2.3.1b4``
     - Pinned in ``docker/netwaggle.Dockerfile`` (``MININET_TAG``).
   * - PyTorch
     - 2.4.1
     - Installed when ``INSTALL_ML=1`` (the default).
   * - Torchvision
     - 0.19.1
     - Installed with PyTorch.
   * - Ultralytics (YOLO)
     - 8.3.28
     - Installed when ``INSTALL_ML=1``.
   * - CUDA driver
     - 550 or later (recommended)
     - Required for GPU inference inside the container.

All version pins are in ``docker/Dockerfile`` and
``docker/compose.single-host.yml``.

.. note::

   ``pip install carla==0.9.10`` does not work: version 0.9.10 was never
   published to PyPI. Use the PythonAPI egg distributed with the CARLA
   0.9.10 release and mount it at the path declared in ``sim.carla_python_api``.
