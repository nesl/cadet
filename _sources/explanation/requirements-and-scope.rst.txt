Requirements and scope
======================

Environmental requirements
--------------------------

**Privileged container for network emulation**
    The ``netwaggle`` and ``netem`` backends require ``--cap-add NET_ADMIN`` on
    the node container. This is set automatically when CADET manages the
    container lifecycle via the ``docker`` or ``ssh_docker`` launcher. With the
    local launcher, the host process must have the capability (run as root or
    grant it via ``setcap``). Without it, ``cadet validate`` rejects configs
    that use those backends.

**CARLA version**
    The CARLA server must be ``carlasim/carla:0.9.10`` (or the compatible patch
    ``0.9.10.1``). The Python client is the PythonAPI egg
    ``carla-0.9.10-py3.7-linux-x86_64.egg``, mounted at the path given by
    ``sim.carla_python_api``; do not use ``pip install carla``. A preflight
    check compares the first three version components and fails the run if they
    do not match.

**GPU and CUDA driver**
    CADET nodes with ``device: gpu`` require a CUDA-capable GPU. The CUDA
    driver version on the host must be at least the minimum supported by the
    CUDA runtime in the container. Mismatched drivers produce an error at
    container start rather than at inference time.

**Python version**
    CADET runs on Python 3.8 through 3.13 (``requires-python >=3.8,<3.14``).
    Configs that use the CARLA source require the CARLA PythonAPI egg
    (``carla-0.9.10-py3.7-linux-x86_64.egg``), which targets CPython 3.7/3.8
    ABI; use the CADET Docker image for those configs — it provides Python 3.8
    and the egg. All other configs (replay, synthetic, dataset, live) run on
    any supported Python.

**App plugins on PYTHONPATH**
    ScenarioRunner is bundled in the CADET container. The ``examples/``
    app plugins (braking, V2V, V2I) must be present on the Python path at
    run time. With the local launcher, run from the repo root with
    ``PYTHONPATH=.``; with the container launcher the repo is bind-mounted
    automatically.

Feature scope
-------------

The maturity table on the :doc:`../index` home page lists every feature and
its status. ``cadet validate`` accepts the options listed there and names
anything else a config asks for, so a run either has what it asked for or does
not start.

**docker launcher**
    Lifecycle-verified on a GPU host: both link backends, normal completion,
    the active limit, a killed server container and a two-rep sweep.

**ssh_docker launcher**
    ``SshDockerLauncher`` in ``cadet/core/docker_launcher.py``, verified by the
    distributed-sensing example across two machines, with the sensors local and the gateway
    remote.

**ssh_venv launcher**
    The same placement without a container: a role runs in a virtual
    environment on the second machine, which is what a device too small for a
    container image needs.

**netwaggle backend**
    Runs under ``cadet emulate`` in a privileged emulation container. Wired
    links put each host in its own namespace over shaped veth pairs. A Wi-Fi
    cell adds one requirement: the container shares the host's network
    namespace, because ``mac80211_hwsim`` creates its radios there. The
    five-sensor cell answered 20 of 20 messages per sensor over wmediumd's
    medium with the interference model. Mininet's own teardown can hang, so the
    emulator bounds it and then removes what it named itself; ``cadet down``
    clears anything a killed run left.

**Pipeline stages**
    The pipeline machinery is in ``cadet/pipeline/`` and the stage
    implementations are in ``cadet/pipeline/stages/``: ``camera``, ``yolo``,
    ``roi_consecutive`` and ``brake_controller`` for braking, ``yolo_v2v`` and
    ``v2v_ttc_gate`` for V2V, and ``yolo_v2i``, ``v2i_ego_gate`` and
    ``v2i_fusion_gate`` for V2I. A ``pipeline`` block names one per stage in
    ``impl``, and the component library it draws on is in
    ``examples/av_pipeline/``.

**WebSocket transport**
    ``transport.protocol: websocket`` puts the same payloads in RFC 6455 binary
    frames after one HTTP upgrade, so a link can cross infrastructure that
    forwards HTTP and nothing else. The reference example has run end to end
    over it with every frame answered, and the same run over a 60 ms ``delay``
    link measured the configured delay, so the delay lines still see whole
    payloads through the new carrier.

**Dataset mode**
    See :doc:`dataset-mode`: one model serving a replayed dataset, with
    model-, system- and task-level numbers per run.

**FM interface**
    ``cadet summarize`` and ``cadet ask`` read a run directory or a sweep,
    reduce it to a record and put that record to an OpenAI-compatible
    chat-completions endpoint. The endpoint and model are config; the key is
    read from the environment variable the config names, and never stored.
