"""Mininet-WiFi link backend, adapted from the NESL NetWaggle testbed.

What this is for:
    Implements ``links[*].network.mode: netwaggle``: a real emulated network,
    built with Mininet-WiFi, carrying the traffic of links while framework
    plumbing bypasses it on an unshaped interface.

    This is **packet-level** emulation, with the same consequences as
    :mod:`cadet.network.netem` — ``direction`` applies to every packet leaving
    that side, ``loss`` drops packets that TCP retransmits, and the transport's
    reaction is included in what the application measures. What this backend adds
    over netem is a real topology: each node in its own namespace, a wired or
    wireless medium with a propagation model, and a per-host bypass interface.
    For a configured per-message latency distribution reproduced exactly, use
    ``mode: delay``. See "Link backend semantics" in
    ``docs/config-reference.md``.

What it contains:
    - :mod:`.topology`: the dependency-free topology description, in which
      ``hosts`` run roles and ``connectors`` are shared media that run nothing.
    - :mod:`.generator`: CADET config -> topology.
    - :mod:`.emulator`: builds, starts, stops and reports on the network.

    The public names are re-exported here, so callers use
    ``from cadet.network.netwaggle import generate, Emulator``.

How it fits into CADET:
    The orchestrator calls :func:`generate` then drives an :class:`Emulator`,
    launching role processes inside emulated hosts with ``Emulator.popen``.
    Importing this package does not import Mininet: the emulator defers that to
    ``build()`` so configs can be generated and tested anywhere.

How to configure it:
    Through the config's ``links`` and ``channels``; see the generator for the
    mapping and ``docs/config-reference.md`` for the fields.

A wireless topology needs the host's network namespace:
    ``mac80211_hwsim`` creates its radios in the host's initial network
    namespace, and a container with a namespace of its own sees none of them.
    Measured on a six-radio topology: the host's ``/sys/class/ieee80211``
    listed every radio and the container's listing was empty.

    So the emulation container runs a wireless topology with the host's network
    namespace and keeps its own process namespace. It needs, besides that:
    privileges enough to load a module and create interfaces, the host's
    ``/lib/modules`` to load ``mac80211_hwsim`` from, the host's Open vSwitch
    socket directory so the access point's bridge is made by the switch daemon
    already running there, and ``/sys/kernel/debug`` mounted, because
    Mininet-WiFi enumerates hwsim radios under ``/sys/kernel/debug/ieee80211``.

    Keeping the process namespace is not incidental. The fallback cleanup kills
    ``hostapd``, ``wmediumd`` and ``wpa_supplicant`` by name, so a container
    sharing the host's process namespace would put the machine's own
    ``wpa_supplicant`` in range of it. Sharing the network namespace has the
    same shape of hazard for bridges and namespaces, and
    :func:`.emulator.force_cleanup` answers it by removing only the objects the
    topology generator named when the topology is wireless.

    Verified by the five-sensor cell in ``examples/distributed_sensing/wifi_5.yaml``: six
    stations on one access point, every station associated, 20 of 20 messages
    answered per sensor and 100 arrivals totalling 55,870 bytes at the gateway,
    over wmediumd's medium with the interference model, so the propagation
    model and the node positions governed the link. Mininet-WiFi's own
    ``net.stop()`` hung on that teardown, which is what the bounded stop and
    the fallback cleanup are for; the run recorded the forced path and left the
    machine as it found it.

Provenance:
    The approach — emulate link characteristics in Mininet, run application
    processes in namespaces, keep non-emulated traffic off the topology — comes
    from the NESL NetWaggle testbed (IoCT-Testbed-Simulation). See the module
    docstring of :mod:`.emulator` for what was changed and why, and NOTICE for
    attribution.
"""

from cadet.network.netwaggle.emulator import Emulator, EmulatorError, EmulatorStatus
from cadet.network.netwaggle.generator import (
    NETWAGGLE_MODE,
    UnsupportedLink,
    emulated_node_names,
    generate,
    shaping_for,
)
from cadet.network.netwaggle.topology import (
    EmulatedConnector,
    EmulatedHost,
    EmulatedLink,
    Shaping,
    Topology,
    WifiNode,
    WifiSpec,
)

__all__ = [
    "Emulator",
    "EmulatorError",
    "EmulatorStatus",
    "EmulatedConnector",
    "EmulatedHost",
    "EmulatedLink",
    "NETWAGGLE_MODE",
    "Shaping",
    "Topology",
    "UnsupportedLink",
    "WifiNode",
    "WifiSpec",
    "emulated_node_names",
    "generate",
    "shaping_for",
]
