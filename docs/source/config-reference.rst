Config Reference
================

This reference is generated from ``cadet/config/schema.py`` by
``scripts/gen_config_reference.py``.  To regenerate it after a schema change,
re-run that script and copy the output here.

Each enumerated field lists the values a run may ask for. ``cadet validate``
names anything else a config asks for and refuses the run.

Link backend semantics
----------------------

``links[*].network.mode`` selects *where* a link is emulated, and that changes
what the other fields mean.  The two families are not interchangeable and neither
is made to imitate the other.

.. list-table::
   :header-rows: 1
   :widths: 20 35 35

   * -
     - ``delay``
     - ``netem``, ``netwaggle``
   * - layer
     - **message-level**: the transport's delay line holds whole application messages
     - **packet-level**: a kernel qdisc holds packets, and TCP reacts
   * - ``distribution``
     - sampled per message, per the configured distribution
     - the mean and jitter of a netem delay, applied per packet
   * - ``direction: forward``
     - delays request messages only; replies are untouched
     - delays *every* packet leaving the sender, including the handshake and ACKs
   * - ``direction: reverse``
     - delays reply messages only
     - delays every packet leaving the listener
   * - ``loss``
     - **drops application messages**; a dropped message is never answered
     - **drops packets**; TCP retransmits, so the message still arrives
   * - ``bandwidth_mbps``
     - serialization delay computed per message
     - a real rate limit on the queue
   * - needs
     - nothing
     - ``netem``: NET_ADMIN.  ``netwaggle``: the emulation container

What this means in practice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A configured delay is reproduced at the message level, and approximately
at the packet level.  Measured on a 60 ms forward link with the reference app
paced at 20 fps: ``delay`` gave p50 60.4 ms, while ``netem`` and ``netwaggle``
both gave p50 ≈ 149 ms.  The extra is not emulator error: an isolated
request/response probe over the same qdisc measures 60.1 ms.  It is TCP: with
small paced messages, Nagle holds a write until the previous one is acknowledged,
and the acknowledgement is itself behind the configured delay.  Setting
``transport_tuning.tcp_nodelay: true`` on the sending node brings the same run
to p50 110.2 ms.

Configured loss means different things.  ``delay`` loses messages, so a 1% loss
produced 1.25% unanswered frames.  ``netem`` and ``netwaggle`` lose packets under
a reliable transport, so the same 1% produced 0% unanswered and showed up as
retransmission latency in p99 instead.  If an experiment needs observable message
loss, use ``delay`` or an unreliable transport.

Choose by question, not by fidelity.  Use ``delay`` to state a per-message
latency distribution and have it honoured.  Use ``netem`` or ``netwaggle`` to ask
what a real protocol stack does on a delayed, lossy, rate-limited path.
``netwaggle`` additionally gives a real topology, a wireless medium and
propagation model, and hosts in separate namespaces.

Every run writes ``link_report.json`` next to its metrics, holding the configured
parameters beside the measured per-message latency and unanswered counts, so the
gap is visible per run rather than assumed.

Schema reference
----------------

Config
~~~~~~

A complete CADET experiment configuration.

.. list-table::
   :header-rows: 1
   :widths: 25 35 15 25

   * - field
     - type
     - default
     - notes
   * - ``experiment``
     - Experiment
     - (required)
     -
   * - ``source``
     - CarlaSource | ReplaySource | SyntheticSource | LiveSource
     - (required)
     -
   * - ``models``
     - Models
     - (defaults)
     -
   * - ``nodes``
     - list[Node]
     - (required)
     -
   * - ``channels``
     - list[Channel]
     - ``[]``
     -
   * - ``links``
     - list[Link]
     - ``[]``
     -
   * - ``transport``
     - TransportConfig
     - (defaults)
     -
   * - ``metrics``
     - MetricsConfig
     - (defaults)
     -
   * - ``app``
     - App
     - (required)
     -
   * - ``sweep``
     - Sweep
     - (defaults)
     -
   * - ``outputs``
     - Outputs
     - (defaults)
     -

Experiment
~~~~~~~~~~

Identity and repetition policy for a run family.

.. list-table::
   :header-rows: 1
   :widths: 25 20 15 40

   * - field
     - type
     - default
     - notes
   * - ``name``
     - str
     - (required)
     -
   * - ``app``
     - str
     - (required)
     -
   * - ``repetitions``
     - int
     - ``1``
     -
   * - ``seed``
     - int (optional)
     - ``None``
     -
   * - ``max_run_s``
     - float (optional)
     - ``None``
     -

Source types
~~~~~~~~~~~~

``CarlaSource``: Input from a running CARLA simulator (the only live backend today).

.. list-table::
   :header-rows: 1
   :widths: 30 20 15 35

   * - field
     - type
     - default
     - notes
   * - ``kind``
     - ``'carla'``
     - ``carla``
     -
   * - ``simulator_host``
     - str
     - ``127.0.0.1``
     -
   * - ``simulator_port``
     - int
     - ``2000``
     -
   * - ``simulator_timeout``
     - int
     - ``10``
     -
   * - ``fps``
     - int
     - ``10``
     -
   * - ``sync_mode``
     - bool
     - ``True``
     -
   * - ``camera``
     - Camera
     - (defaults)
     -
   * - ``scenario``
     - Scenario
     - (defaults)
     -
   * - ``weather``
     - ``'clear'`` | ``'light'`` | ``'high'`` (optional)
     - ``None``
     -

``ReplaySource``: Deterministic replay from a folder of images or a video file.

.. list-table::
   :header-rows: 1
   :widths: 30 20 15 35

   * - field
     - type
     - default
     - notes
   * - ``kind``
     - ``'replay'``
     - ``replay``
     -
   * - ``path``
     - str
     - (required)
     -
   * - ``fps``
     - float
     - ``10.0``
     -
   * - ``loop``
     - bool
     - ``False``
     -
   * - ``limit``
     - int (optional)
     - ``None``
     -

``SyntheticSource``: Generated frames for tests (no files, no hardware).

.. list-table::
   :header-rows: 1
   :widths: 30 20 15 35

   * - field
     - type
     - default
     - notes
   * - ``kind``
     - ``'synthetic'``
     - ``synthetic``
     -
   * - ``count``
     - int
     - ``10``
     -
   * - ``width``
     - int
     - ``64``
     -
   * - ``height``
     - int
     - ``48``
     -
   * - ``fps``
     - float
     - ``0.0``
     -

``LiveSource``: Live capture: a camera device index or a network stream URL.

.. list-table::
   :header-rows: 1
   :widths: 30 20 15 35

   * - field
     - type
     - default
     - notes
   * - ``kind``
     - ``'live'``
     - ``live``
     -
   * - ``device``
     - int (optional)
     - ``None``
     -
   * - ``url``
     - str (optional)
     - ``None``
     -
   * - ``fps``
     - float
     - ``0.0``
     -
   * - ``limit``
     - int (optional)
     - ``None``
     -

Camera
~~~~~~

Ego camera sensor geometry.

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - field
     - type
     - default
     - notes
   * - ``width``
     - int
     - ``960``
     -
   * - ``height``
     - int
     - ``512``
     -
   * - ``fov``
     - float
     - ``90.0``
     -

Scenario
~~~~~~~~

Scenario geometry and lifetime (fields optional; apps use a subset).

.. list-table::
   :header-rows: 1
   :widths: 25 20 15 40

   * - field
     - type
     - default
     - notes
   * - ``map``
     - str (optional)
     - ``None``
     -
   * - ``ego_vehicle``
     - str (optional)
     - ``None``
     -
   * - ``target_object``
     - str (optional)
     - ``None``
     -
   * - ``ego_speed``
     - str (optional)
     - ``None``
     -
   * - ``ego_spawn``
     - Spawn (optional)
     - ``None``
     -
   * - ``occluder``
     - Occluder (optional)
     - ``None``
     -
   * - ``pedestrian``
     - Pedestrian (optional)
     - ``None``
     -
   * - ``timeout_s``
     - float (optional)
     - ``None``
     -

Spawn
~~~~~

A world pose (metres, degrees) for spawning an actor.

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - field
     - type
     - default
     - notes
   * - ``x``
     - float
     - (required)
     -
   * - ``y``
     - float
     - (required)
     -
   * - ``z``
     - float
     - ``0.0``
     -
   * - ``yaw``
     - float
     - ``0.0``
     -

Models
~~~~~~

Pipeline component choices plus the logical-name → artifact registry.

.. list-table::
   :header-rows: 1
   :widths: 25 25 15 35

   * - field
     - type
     - default
     - notes
   * - ``detector``
     - str
     - ``yolo``
     -
   * - ``tracker``
     - str
     - ``sort``
     -
   * - ``predictor``
     - str
     - ``linear``
     -
   * - ``model``
     - str (optional)
     - ``None``
     -
   * - ``warmup``
     - bool
     - ``True``
     -
   * - ``registry``
     - map[str → ModelEntry]
     - ``{}``
     -

Node
~~~~

A compute node that hosts a role (client / server / sim).

.. list-table::
   :header-rows: 1
   :widths: 30 30 15 25

   * - field
     - type
     - default
     - notes
   * - ``name``
     - str
     - (required)
     -
   * - ``host``
     - str
     - (required)
     -
   * - ``port``
     - int (optional)
     - ``None``
     -
   * - ``launcher``
     - ``'local'`` | ``'docker'`` | ``'ssh_docker'`` | ``'ssh_venv'``
     - ``local``
     -
   * - ``role``
     - ``'client'`` | ``'server'`` | ``'sim'`` | ``'leader'`` | ``'load'``
     - (required)
     -
   * - ``device``
     - str
     - ``cpu``
     -
   * - ``gpu_index``
     - int (optional)
     - ``None``
     -
   * - ``scheduler``
     - ``'fifo'`` | ``'round_robin'`` (optional)
     - ``None``
     -
   * - ``extra_processing_ms``
     - float (optional)
     - ``None``
     -
   * - ``clients``
     - int (optional)
     - ``None``
     -
   * - ``rate_hz``
     - float (optional)
     - ``None``
     -
   * - ``arrival``
     - ``'constant'`` | ``'poisson'`` (optional)
     - ``None``
     -
   * - ``frames``
     - int (optional)
     - ``None``
     -
   * - ``transport_tuning``
     - TransportTuning (optional)
     - ``None``
     -

TransportTuning
~~~~~~~~~~~~~~~

TCP stack settings for one node, applied only in that node's namespace.

.. list-table::
   :header-rows: 1
   :widths: 35 20 15 30

   * - field
     - type
     - default
     - notes
   * - ``tcp_nodelay``
     - bool (optional)
     - ``None``
     -
   * - ``tcp_slow_start_after_idle``
     - bool (optional)
     - ``None``
     -
   * - ``initcwnd``
     - int (optional)
     - ``None``
     -

Channel
~~~~~~~

A direct data path between two nodes (no network emulation).

.. list-table::
   :header-rows: 1
   :widths: 30 20 15 35

   * - field
     - type
     - default
     - notes
   * - ``name``
     - str
     - (required)
     -
   * - ``from``
     - str
     - (required)
     -
   * - ``to``
     - str
     - (required)
     -
   * - ``stream_fps``
     - int (optional)
     - ``None``
     -
   * - ``max_queue``
     - int
     - ``2``
     -
   * - ``send_meta``
     - bool
     - ``False``
     -
   * - ``stop_on_brake``
     - bool
     - ``False``
     -
   * - ``device_id``
     - str (optional)
     - ``None``
     -

Link
~~~~

A network-emulated data path between two nodes.

.. list-table::
   :header-rows: 1
   :widths: 25 30 15 30

   * - field
     - type
     - default
     - notes
   * - ``name``
     - str
     - (required)
     -
   * - ``from``
     - str
     - (required)
     -
   * - ``to``
     - str
     - (required)
     -
   * - ``direction``
     - ``'forward'`` | ``'reverse'`` | ``'both'``
     - ``forward``
     -
   * - ``network``
     - NetworkModel
     - (defaults)
     -

NetworkModel
~~~~~~~~~~~~

Emulated-network behaviour applied to a link's traffic.

.. list-table::
   :header-rows: 1
   :widths: 30 40 15 15

   * - field
     - type
     - default
     - notes
   * - ``mode``
     - ``'none'`` | ``'delay'`` | ``'netem'`` | ``'netwaggle'``
     - ``none``
     - implemented
   * - ``medium``
     - ``'wired'`` | ``'wifi'``
     - ``wired``
     -
   * - ``distribution``
     - ConstantDist | NormalDist | LognormalDist | TraceDist
     - (defaults)
     -
   * - ``loss``
     - float
     - ``0.0``
     -
   * - ``bandwidth_mbps``
     - float (optional)
     - ``None``
     -
   * - ``wifi``
     - WifiConfig (optional)
     - ``None``
     -

Delay distributions
~~~~~~~~~~~~~~~~~~~

``ConstantDist``: fixed per-packet delay.

.. list-table::
   :header-rows: 1
   :widths: 25 20 15 40

   * - field
     - type
     - default
     - notes
   * - ``kind``
     - ``'constant'``
     - ``constant``
     -
   * - ``value_ms``
     - float
     - ``0.0``
     -

``NormalDist``: Gaussian delay.

.. list-table::
   :header-rows: 1
   :widths: 25 20 15 40

   * - field
     - type
     - default
     - notes
   * - ``kind``
     - ``'normal'``
     - ``normal``
     -
   * - ``mean_ms``
     - float
     - (required)
     -
   * - ``std_ms``
     - float
     - (required)
     -

``LognormalDist``: log-normal delay (long tail).

.. list-table::
   :header-rows: 1
   :widths: 25 20 15 40

   * - field
     - type
     - default
     - notes
   * - ``kind``
     - ``'lognormal'``
     - ``lognormal``
     -
   * - ``p50_ms``
     - float
     - (required)
     - median of the distribution
   * - ``p99_ms``
     - float
     - (required)
     - 99th-percentile of the distribution

``TraceDist``: replay delays from a recorded trace file.

.. list-table::
   :header-rows: 1
   :widths: 25 20 15 40

   * - field
     - type
     - default
     - notes
   * - ``kind``
     - ``'trace'``
     - ``trace``
     -
   * - ``path``
     - str
     - (required)
     -

TransportConfig
~~~~~~~~~~~~~~~

Wire transport for node-to-node traffic.

.. list-table::
   :header-rows: 1
   :widths: 25 30 15 30

   * - field
     - type
     - default
     - notes
   * - ``protocol``
     - ``'tcp'`` | ``'websocket'``
     - ``tcp``
     -
   * - ``mode``
     - ``'frame'`` | ``'stream'``
     - ``frame``
     -

MetricsConfig
~~~~~~~~~~~~~

Optional measurement collectors to run during an experiment.

.. list-table::
   :header-rows: 1
   :widths: 25 45 15 15

   * - field
     - type
     - default
     - notes
   * - ``collectors``
     - list of ``'gpu'`` | ``'cpu'`` | ``'bandwidth'`` | ``'queue'`` | ``'energy'`` | ``'clock_offset'``
     - ``[]``
     -

App
~~~

App selector and its raw parameters.

.. list-table::
   :header-rows: 1
   :widths: 25 30 15 30

   * - field
     - type
     - default
     - notes
   * - ``plugin``
     - str
     - (required)
     -
   * - ``params``
     - map[str → Any]
     - ``{}``
     -

Sweep
~~~~~

Cartesian sweep over dotted config paths → list of values.

.. list-table::
   :header-rows: 1
   :widths: 25 30 15 30

   * - field
     - type
     - default
     - notes
   * - ``axes``
     - map[str → list[Any]]
     - ``{}``
     -

Outputs
~~~~~~~

Where run artifacts go.  By policy, results and weights are never committed.

.. list-table::
   :header-rows: 1
   :widths: 25 20 30 25

   * - field
     - type
     - default
     - notes
   * - ``root``
     - str
     - ``outputs``
     -
   * - ``naming``
     - str
     - ``{experiment}/{app}/{run_id}``
     -
   * - ``save``
     - list[str]
     - ``[]``
     -
