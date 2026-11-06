Wi-Fi cell
==========

A Wi-Fi cell places every node on a shared wireless medium. Each node's
distance to the access point determines its signal strength via the propagation
model; the medium shapes all traffic, so there is no per-link qdisc. Use this
for experiments where medium contention and distance-dependent loss matter.

Run Wi-Fi cell configs with ``cadet emulate``, not ``cadet run`` — the wireless
backend shapes in the kernel rather than in the transport::

    cadet emulate examples/distributed_sensing/wifi_5.yaml --app-module examples.distributed_sensing.app

The emulation container must share the host's network namespace because
``mac80211_hwsim`` creates its radios there::

    ./scripts/cadet emulate --privileged --net=host examples/distributed_sensing/wifi_5.yaml \
        --app-module examples.distributed_sensing.app

Config structure
----------------

A Wi-Fi cell adds two top-level sections to a standard config: ``connectors``
and per-node ``connector``/``position`` fields.

**Connector** (the access point)::

    connectors:
      - name: cell_ap
        type: ap
        position: "0,0,0"
        wifi:
          ssid: cadet
          mode: g
          channel: 1
          propagation_model: logDistance
          propagation_exp: 3.5

``type: ap`` gives the connector radios. ``position`` is the access point's
location in metres; each station's distance to it is what the propagation model
acts on. The ``wifi`` block is the whole cell's configuration, not any one
station's.

**Nodes** (stations)::

    nodes:
      - name: sensor_1
        host: localhost
        role: client
        device: cpu
        connector: cell_ap
        position: "5,0,0"
      - name: gateway
        host: 127.0.0.1
        port: 15110
        role: server
        device: cpu
        connector: cell_ap
        position: "0,1,0"

Each node names the connector it attaches to. ``position`` places the station
in the same coordinate space as the access point; the propagation model uses
the Euclidean distance between station and access point.

**Link** (the shared medium)::

    links:
      - name: uplink
        from: [sensor_1, sensor_2, sensor_3, sensor_4, sensor_5]
        to: gateway
        direction: both
        network:
          mode: netwaggle
          medium: wifi
          wifi:
            nodes:
              sensor_1: {role: station}
              sensor_2: {role: station}
              sensor_3: {role: station}
              sensor_4: {role: station}
              sensor_5: {role: station}
              gateway:  {role: station}

``medium: wifi`` selects the wireless path. The ``wifi.nodes`` block maps each
endpoint to its role (``station``). There is no distribution: the propagation
model and distances set the shaping.

Propagation models
------------------

``propagation_model: logDistance``
    Signal strength decays as ``10 * exp * log10(d / d0)``. ``propagation_exp``
    is the path-loss exponent (2 = free space; 3–4 = indoor/urban). The default
    is 3.5.

``propagation_model: twoRayGround``
    Two-ray ground reflection; accurate for longer outdoor links. No
    ``propagation_exp``; pass ``ht`` and ``hr`` (transmitter/receiver heights
    in metres) in ``propagation_params`` if the defaults are wrong.

Working example
---------------

``examples/distributed_sensing/wifi_5.yaml`` places five sensors and one gateway on a single
cell. The five sensors sit at increasing distances (5–55 m) from the access
point, so one run already crosses the propagation gradient::

    ./scripts/cadet emulate --privileged --net=host \
        examples/distributed_sensing/wifi_5.yaml --app-module examples.distributed_sensing.app

See also
--------

- :doc:`../explanation/netwaggle-layer` — network backend reference
- :doc:`../howto/link-distributions` — wired-medium loss and delay parameters
- :doc:`../explanation/network-backends` — comparison of all three backends
