Tested configurations
=====================

The acceptance matrix verifies every combination of source, processing core,
network backend, and infrastructure tier listed here. Each cell runs its
example app end-to-end and must pass: clean teardown, well-formed metrics,
correct link report, and FM query answering within tolerance of the offline
measurement.

The table covers the full four-layer product space. Cells marked ``—`` are
structurally incompatible (the distributed HIL and Mininet emulation rows
explain why below the table).

.. list-table::
   :header-rows: 1
   :widths: 14 16 20 20 15 15

   * - Source
     - Processing
     - Network
     - Infrastructure
     - Launcher
     - Cell ID
   * - CARLA
     - AV pipeline
     - none
     - single host
     - local
     - carla.pipe.none.sh
   * - CARLA
     - AV pipeline
     - none
     - distributed HIL
     - ssh_docker
     - carla.pipe.none.dist
   * - CARLA
     - AV pipeline
     - delay
     - single host
     - local
     - carla.pipe.delay.sh
   * - CARLA
     - AV pipeline
     - delay
     - distributed HIL
     - ssh_docker
     - carla.pipe.delay.dist
   * - CARLA
     - AV pipeline
     - netem
     - single host
     - docker
     - carla.pipe.netem.sh
   * - CARLA
     - AV pipeline
     - netem
     - distributed HIL
     - ssh_docker
     - carla.pipe.netem.dist
   * - CARLA
     - AV pipeline
     - netwaggle (wired)
     - single host
     - local
     - carla.pipe.nw.sh
   * - CARLA
     - standalone
     - none
     - single host
     - local
     - carla.std.none.sh
   * - CARLA
     - standalone
     - none
     - distributed HIL
     - ssh_docker
     - carla.std.none.dist
   * - CARLA
     - standalone
     - delay
     - single host
     - local
     - carla.std.delay.sh
   * - CARLA
     - standalone
     - delay
     - distributed HIL
     - ssh_docker
     - carla.std.delay.dist
   * - CARLA
     - standalone
     - netem
     - single host
     - docker
     - carla.std.netem.sh
   * - CARLA
     - standalone
     - netem
     - distributed HIL
     - ssh_docker
     - carla.std.netem.dist
   * - CARLA
     - standalone
     - netwaggle (wired)
     - single host
     - local
     - carla.std.nw.sh
   * - CARLA
     - standalone
     - netwaggle (Wi-Fi cell)
     - single host
     - local
     - carla.std.nwwifi.sh
   * - Replay
     - AV pipeline
     - none
     - single host
     - local
     - replay.pipe.none.sh
   * - Replay
     - AV pipeline
     - none
     - distributed HIL
     - ssh_docker
     - replay.pipe.none.dist
   * - Replay
     - AV pipeline
     - delay
     - single host
     - local
     - replay.pipe.delay.sh
   * - Replay
     - AV pipeline
     - delay
     - distributed HIL
     - ssh_docker
     - replay.pipe.delay.dist
   * - Replay
     - AV pipeline
     - netem
     - single host
     - docker
     - replay.pipe.netem.sh
   * - Replay
     - AV pipeline
     - netem
     - distributed HIL
     - ssh_docker
     - replay.pipe.netem.dist
   * - Replay
     - AV pipeline
     - netwaggle (wired)
     - single host
     - local
     - replay.pipe.nw.sh
   * - Replay
     - standalone
     - none
     - single host
     - local
     - replay.std.none.sh
   * - Replay
     - standalone
     - none
     - distributed HIL
     - ssh_docker
     - replay.std.none.dist
   * - Replay
     - standalone
     - delay
     - single host
     - local
     - replay.std.delay.sh
   * - Replay
     - standalone
     - delay
     - distributed HIL
     - ssh_docker
     - replay.std.delay.dist
   * - Replay
     - standalone
     - netem
     - single host
     - docker
     - replay.std.netem.sh
   * - Replay
     - standalone
     - netem
     - distributed HIL
     - ssh_docker
     - replay.std.netem.dist
   * - Replay
     - standalone
     - netwaggle (wired)
     - single host
     - local
     - replay.std.nw.sh
   * - Replay
     - standalone
     - netwaggle (Wi-Fi cell)
     - single host
     - local
     - replay.std.nwwifi.sh
   * - Live
     - AV pipeline
     - none
     - single host
     - local
     - live.pipe.none.sh
   * - Live
     - AV pipeline
     - none
     - distributed HIL
     - ssh_docker
     - live.pipe.none.dist
   * - Live
     - AV pipeline
     - delay
     - single host
     - local
     - live.pipe.delay.sh
   * - Live
     - AV pipeline
     - delay
     - distributed HIL
     - ssh_docker
     - live.pipe.delay.dist
   * - Live
     - AV pipeline
     - netem
     - single host
     - docker
     - live.pipe.netem.sh
   * - Live
     - AV pipeline
     - netem
     - distributed HIL
     - ssh_docker
     - live.pipe.netem.dist
   * - Live
     - AV pipeline
     - netwaggle (wired)
     - single host
     - local
     - live.pipe.nw.sh
   * - Live
     - standalone
     - none
     - single host
     - local
     - live.std.none.sh
   * - Live
     - standalone
     - none
     - distributed HIL
     - ssh_docker
     - live.std.none.dist
   * - Live
     - standalone
     - delay
     - single host
     - local
     - live.std.delay.sh
   * - Live
     - standalone
     - delay
     - distributed HIL
     - ssh_docker
     - live.std.delay.dist
   * - Live
     - standalone
     - netem
     - single host
     - docker
     - live.std.netem.sh
   * - Live
     - standalone
     - netem
     - distributed HIL
     - ssh_docker
     - live.std.netem.dist
   * - Live
     - standalone
     - netwaggle (wired)
     - single host
     - local
     - live.std.nw.sh
   * - Synthetic
     - AV pipeline
     - none
     - single host
     - local
     - synth.pipe.none.sh
   * - Synthetic
     - AV pipeline
     - none
     - distributed HIL
     - ssh_docker
     - synth.pipe.none.dist
   * - Synthetic
     - AV pipeline
     - delay
     - single host
     - local
     - synth.pipe.delay.sh
   * - Synthetic
     - AV pipeline
     - delay
     - distributed HIL
     - ssh_docker
     - synth.pipe.delay.dist
   * - Synthetic
     - AV pipeline
     - netem
     - single host
     - docker
     - synth.pipe.netem.sh
   * - Synthetic
     - AV pipeline
     - netem
     - distributed HIL
     - ssh_docker
     - synth.pipe.netem.dist
   * - Synthetic
     - AV pipeline
     - netwaggle (wired)
     - single host
     - local
     - synth.pipe.nw.sh
   * - Synthetic
     - standalone
     - none
     - single host
     - local
     - synth.std.none.sh
   * - Synthetic
     - standalone
     - none
     - distributed HIL
     - ssh_venv
     - synth.std.none.dist
   * - Synthetic
     - standalone
     - delay
     - single host
     - local
     - synth.std.delay.sh
   * - Synthetic
     - standalone
     - delay
     - distributed HIL
     - ssh_docker
     - synth.std.delay.dist
   * - Synthetic
     - standalone
     - netem
     - single host
     - docker
     - synth.std.netem.sh
   * - Synthetic
     - standalone
     - netem
     - distributed HIL
     - ssh_docker
     - synth.std.netem.dist
   * - Synthetic
     - standalone
     - netwaggle (wired)
     - single host
     - local
     - synth.std.nw.sh
   * - Synthetic
     - standalone
     - netwaggle (Wi-Fi cell)
     - single host
     - local
     - synth.std.nwwifi.sh
   * - Dataset
     - AV pipeline
     - none
     - single host
     - local
     - dataset.pipe.none.sh
   * - Dataset
     - AV pipeline
     - none
     - distributed HIL
     - ssh_docker
     - dataset.pipe.none.dist
   * - Dataset
     - AV pipeline
     - delay
     - single host
     - local
     - dataset.pipe.delay.sh
   * - Dataset
     - AV pipeline
     - delay
     - distributed HIL
     - ssh_docker
     - dataset.pipe.delay.dist
   * - Dataset
     - AV pipeline
     - netem
     - single host
     - docker
     - dataset.pipe.netem.sh
   * - Dataset
     - AV pipeline
     - netem
     - distributed HIL
     - ssh_docker
     - dataset.pipe.netem.dist
   * - Dataset
     - AV pipeline
     - netwaggle (wired)
     - single host
     - local
     - dataset.pipe.nw.sh
   * - Dataset
     - standalone
     - none
     - single host
     - local
     - dataset.std.none.sh
   * - Dataset
     - standalone
     - none
     - distributed HIL
     - ssh_docker
     - dataset.std.none.dist
   * - Dataset
     - standalone
     - delay
     - single host
     - local
     - dataset.std.delay.sh
   * - Dataset
     - standalone
     - delay
     - distributed HIL
     - ssh_docker
     - dataset.std.delay.dist
   * - Dataset
     - standalone
     - netem
     - single host
     - docker
     - dataset.std.netem.sh
   * - Dataset
     - standalone
     - netem
     - distributed HIL
     - ssh_docker
     - dataset.std.netem.dist
   * - Dataset
     - standalone
     - netwaggle (wired)
     - single host
     - local
     - dataset.std.nw.sh
   * - Dataset
     - standalone
     - netwaggle (Wi-Fi cell)
     - single host
     - local
     - dataset.std.nwwifi.sh

Layers
------

**Source** — where frames come from

- *CARLA*: live CARLA 0.9.10 simulator with ScenarioRunner
- *Replay*: a directory of images or a video file
- *Synthetic*: generated frames, no hardware
- *Live*: camera device index or network stream URL
- *Dataset*: VQAv2/COCO-style dataset with ground truth labels

**Processing** — what runs on each frame

- *AV pipeline*: staged perception→decision pipeline (braking stages for CARLA;
  open-loop decision log for all other sources)
- *Standalone*: a single model served over the wire (detector for image sources,
  VLM for dataset)

**Network** — how the link between client and server is shaped

- *none*: plain transport, no shaping
- *delay*: message-level delay line (constant, normal, lognormal, or trace
  distribution; loss and bandwidth caps)
- *netem*: packet-level ``tc``/``netem`` inside each sender's container
- *netwaggle (wired)*: Mininet emulation over shaped veth pairs, one namespace
  per host; run with ``cadet emulate``
- *netwaggle (Wi-Fi cell)*: Mininet-WiFi shared-medium cell; each station's
  signal strength derived from propagation model and distance; run with
  ``cadet emulate --privileged --net=host``

**Infrastructure** — where the nodes run

- *single host*: every node is a process or container on one machine
- *distributed HIL*: a real second machine carries the server node, reached
  via ``ssh_docker`` or ``ssh_venv``

Compatibility notes
-------------------

**netwaggle + distributed HIL**
    Mininet emulates a single host: its emulated hosts are network namespaces
    of the emulation container. A role on a second machine would need to be a
    Mininet host, which it cannot be from outside the container. Distributed
    HIL shapes with ``netem`` inside each node's container or with ``delay``
    in the transport.

**Wi-Fi cell + distributed HIL**
    Same constraint: ``mac80211_hwsim`` radios exist only in the host's network
    namespace of the emulation container.

See also
--------

- :doc:`../explanation/network-backends` — comparison of all three backends
- :doc:`../howto/wifi-cell` — Wi-Fi cell config walkthrough
- :doc:`../tutorials/04-multi-device` — distributed HIL setup
