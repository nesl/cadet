Installation
============

Docker (recommended)
---------------------

Build the image from the repository root::

    docker build -f docker/Dockerfile -t cadet:dev .

This installs PyTorch 2.4.1, Ultralytics, and the CARLA 0.9.10 Python
client (``INSTALL_ML=1`` is the default). For a smaller image without the
ML stack::

    docker build --build-arg INSTALL_ML=0 -f docker/Dockerfile -t cadet-slim:dev .

The slim image still runs the reference example and the test suite using a
stub model.

Start a single-host CARLA experiment with Docker Compose::

    docker compose -f docker/compose.single-host.yml up --build

The compose file starts the ``carlasim/carla:0.9.10`` simulator as service
``carla-sim`` and sets ``CADET_CARLA_HOST=carla-sim`` in the CADET container
automatically. No ``--set source.carla.host`` flag is needed when using
compose.

Native (editable install)
--------------------------

Clone the CADET repository, then install in editable mode::

    cd cadet
    pip install -U pip   # Python 3.8's bundled pip cannot do editable installs
    pip install -e .

This requires Python 3.8 or later. Runtime dependencies are ``pydantic>=2``
and ``PyYAML>=5.4``. To run the test suite::

    pip install -e ".[test]"

When running examples with ``--app-module``::

    PYTHONPATH=. cadet run examples/reference/synthetic.yaml \
        --app-module examples.reference.app

``PYTHONPATH=.`` makes the ``examples`` package importable. Inside the Docker
container, ``PYTHONPATH=/app`` is set in the image so this flag is not needed.

GPU notes
---------

GPU inference requires the NVIDIA container toolkit and driver 550 or later.
Pass ``--gpus all`` to expose all GPUs::

    docker run --rm --gpus all -it cadet:dev /bin/bash

On a machine with multiple GPUs, set ``CUDA_VISIBLE_DEVICES`` to select one::

    docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=1 -it cadet:dev /bin/bash

The CARLA 0.9.10 Python client is installed as the ``carla==0.9.10`` PyPI
wheel (cp37/cp38 only). If the wheel is not available for your platform,
mount the PythonAPI egg from a CARLA installation onto ``PYTHONPATH``::

    docker run --rm \
        -v /path/to/CARLA_0.9.10.1/PythonAPI/carla/dist:/carla-egg:ro \
        -e PYTHONPATH=/carla-egg/carla-0.9.10-py3.8-linux-x86_64.egg \
        cadet:dev /bin/bash
