Network Backend Semantics
=========================

CADET offers three network backends. Choosing the right one depends on what
question you are asking.

See :doc:`architecture` for the channels-vs-links rule, and
:doc:`../config-reference` for the full ``links[*].network`` schema.

----

The three backends
------------------

.. list-table::
   :header-rows: 1
   :widths: 14 16 16 18 16 20

   * - Mode
     - Granularity
     - Where it runs
     - Delay semantics
     - Loss semantics
     - Use when
   * - ``delay``
     - Message-level
     - In-process (transport)
     - Sampled per application message; distribution honoured
     - Drops application messages; 1% loss produces ~1% unanswered frames
     - You want to specify a latency budget and have it honoured
   * - ``netem``
     - Packet-level
     - ``tc`` qdisc in the sending node's container (requires NET\_ADMIN)
     - Applied per packet; TCP reacts (Nagle, retransmit, CWND)
     - Drops packets; TCP retransmits → appears as p99 latency spike, not missing frames
     - You want to see how a real protocol stack behaves on a degraded path
   * - ``netwaggle``
     - Packet-level
     - Mininet-WiFi emulated network in its own container
     - Same as netem, plus propagation model for wireless links
     - Same as netem; additionally models wireless association / SNR loss
     - You need a real topology, a wireless medium, mobility, or separate namespaces

----

Why netem and delay report different numbers
--------------------------------------------

A 60 ms forward-link configured identically under ``delay`` and ``netem``
produces very different measured latencies:

.. code-block:: text

   delay backend:  p50 ≈  60 ms   (message-level, exact)
   netem backend:  p50 ≈ 149 ms   (TCP Nagle doubles the round trip)

The reason is **Nagle's algorithm**: TCP holds a small write until the previous
ACK arrives. With ``mode: delay`` the delay line lives *inside* the transport,
so the ACK returns immediately and Nagle never fires. With ``mode: netem`` the
qdisc sits on the interface, so the ACK is itself behind the 60 ms delay.
The sender waits for an ACK that costs 60 ms, then the reply costs another
60 ms, totalling approximately 120 ms plus processing.

Setting ``transport_tuning.tcp_nodelay: true`` on the sending node disables
Nagle and brings the same ``netem`` run to p50 ≈ 110 ms (one-way delay + processing).

----

netem jitter and packet reordering
------------------------------------

netem's jitter is **uniform**, not Gaussian: ``delay 30ms 15ms`` spreads delays
evenly over 15–45 ms. Its p99 is therefore at ``delay + 0.98 × jitter``, not a
Gaussian tail.

netem reorders packets whenever it jitters them: each packet
gets an independent delay and is released when that delay expires, so a packet
behind can overtake one in front. Measured on a loopback pair with 200 paced
datagrams: ``delay 30ms`` gave 0 inversions; ``delay 30ms 15ms`` gave 65
inversions. Adding ``distribution normal`` made it worse (78 inversions).

The fix is a ``rate`` behind the netem leaf: it serialises the output and
preserves order. CADET therefore **refuses** a jittered link without
``bandwidth_mbps`` rather than silently installing one that measures reordering
instead of latency:

.. code-block:: text

   UnrepresentableDelay: a jittered kernel link needs network.bandwidth_mbps.
   netem gives each packet an independent delay ...

----

Loss semantics
--------------

``mode: delay``
    Loss is applied per application **message** before it enters the transport.
    A ``loss: 0.01`` setting (1%) dropped ≈ 1.25% of frames in the reference
    experiment (Bernoulli variation). The receiving side sees fewer frames.

``mode: netem`` / ``mode: netwaggle``
    Loss is applied per **packet**. TCP retransmits lost packets, so the frame
    eventually arrives (but late). A 1% packet-loss setting produced 0% missing
    frames in the same experiment and appeared as elevated p99 latency instead.

    This is the *intended* behaviour of packet-level emulation: it measures what
    a real TCP stack does on a lossy path. Use ``mode: delay`` if you want to
    count unanswered requests.

----

Where the qdisc goes, on one machine and across two
----------------------------------------------------

A ``netem`` qdisc lives in the network namespace of the node whose traffic it
shapes, on the interface that reaches the peer, filtered to that peer's address
and port. Everything else on the interface stays in the unshaped default class,
which is why a channel or a simulator connection sharing it is untouched.

On one machine each node shapes what it sends, on its own egress. A node with
``launcher: docker`` has a container of its own, and the qdisc goes there; the
container is given ``NET_ADMIN`` for it.

With a node on a second machine (``launcher: ssh_docker``), the two directions
of a link are placed differently. What the local node sends is shaped on its own
egress, as before. What arrives from the node on the other machine is shaped on
the local node's **ingress**: the arriving packets are mirrored to an
intermediate device inside that node's own container, and the same queueing
tree - the rate, the delay and the loss the link states - is applied there,
matching the far node's address as the source.

So both directions of a distributed link carry the latency, loss and rate the
configuration gives them, and every queueing discipline sits on an interface
inside a container CADET created and removes. A machine's own interface is never
shaped.

The arrangement needs one thing of the configuration: the node at the local end
of such a link runs with ``launcher: docker``, so that there is a container
namespace to hold the tree. ``cadet validate`` says so, and names the node, when
a config asks for shaping it cannot place.

----

How CADET maps a distribution to kernel parameters
---------------------------------------------------

Both packet-level backends (``netem`` and ``netwaggle``) share
:mod:`cadet.network.shaping` to convert a config distribution to ``tc``
arguments, so they cannot drift apart. The mapping:

- ``constant``: ``tc netem delay Xms``.
- ``normal`` / ``lognormal``: matched on p50 and p99 to netem's uniform jitter.
  The achieved shape is recorded in ``link_report.json`` so the approximation
  is visible.
- ``trace``: **not supported** by packet-level backends (netem takes a delay
  and a jitter, not a sequence). Use ``mode: delay`` for trace-driven emulation.

Two distributions are refused outright:

1. p99 > 2 × p50: netem's symmetric jitter would need negative delays.
2. Jitter without ``bandwidth_mbps`` — would reorder the flow silently.

----

netwaggle specifics
-------------------

``netwaggle`` builds a full Mininet-WiFi topology: each node gets a separate
Linux network namespace with its own interface, and links are realised as either
veth pairs (``medium: wired``) or 802.11 associations (``medium: wifi``).

A wireless link requires a ``network.wifi`` block declaring one AP and one or
more stations with positions. Propagation is modelled via a configurable model
(``logDistance`` is the default) so signal strength and loss
both follow distance. A ``mobility_trace`` file can move a node during the run.

.. note::

   ``netwaggle`` runs in the netwaggle emulation container, which carries
   Mininet-WiFi; see the installation guide for the Compose setup. A Wi-Fi
   cell needs that container to share the host's network namespace, because
   ``mac80211_hwsim`` creates its radios there.

----

Channels vs links: the design rule
-----------------------------------

- **Channels** are framework plumbing. They carry frames between nodes over a
  plain TCP socket with no qdisc and no emulation. The netem backend excludes
  channel traffic from its filter rules, so channels sharing an interface
  with a shaped link are never accidentally delayed.

- **Links** are the only place network emulation happens. Every link has a
  ``NetworkModel``; the default is ``mode: none`` (a direct socket, structurally
  identical to a channel but with the emulation hook available).

The rule is structural: channels cannot have a ``network`` block, and links
cannot be promoted to channels after the fact. This makes it impossible to
accidentally emulate the simulator connection or the log-shipping path.

.. code-block:: yaml

   channels:
     - name: frames          # unshaped; no emulation ever
       from: client
       to: server

   links:
     - name: inference_link  # emulated
       from: client
       to: server
       network:
         mode: netem
         distribution:
           kind: lognormal
           p50_ms: 30
           p99_ms: 51.6
         bandwidth_mbps: 10
