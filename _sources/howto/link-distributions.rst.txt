Configure link distributions
=============================

Each link has a ``network`` block that selects the backend and the latency
distribution. The three backends apply distributions differently; choose
based on what you want to measure.

Backend summary
---------------

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Mode
     - Granularity
     - Use when
   * - ``delay``
     - Message-level
     - You want the configured distribution honoured per message, with no
       TCP interaction
   * - ``netem``
     - Packet-level
     - You want to see how a real TCP stack behaves on a degraded path
   * - ``netwaggle``
     - Packet-level
     - You need a real network topology, wireless medium, or separate
       namespaces

delay backend
-------------

Supports all distribution kinds: ``constant``, ``normal``, ``lognormal``,
and ``trace``. No additional fields required.

.. code-block:: yaml

   links:
     - name: inference_link
       from: client
       to: server
       network:
         mode: delay
         distribution:
           kind: constant
           value_ms: 60.0

For a lognormal distribution:

.. code-block:: yaml

   links:
     - name: inference_link
       from: client
       to: server
       network:
         mode: delay
         distribution:
           kind: lognormal
           p50_ms: 30.0
           p99_ms: 100.0

netem backend
-------------

Supports ``constant``, ``normal``, and ``lognormal`` only. The ``trace``
kind is not supported by packet-level backends; use ``mode: delay`` for
trace-driven emulation.

**``bandwidth_mbps`` is required when jitter is non-zero.** Without a rate
stage, netem assigns each packet an independent delay and releases it when
that delay expires, so a later packet with a shorter delay overtakes an
earlier one. Providing ``bandwidth_mbps`` adds a serialiser that preserves
order.

.. code-block:: yaml

   links:
     - name: inference_link
       from: client
       to: server
       network:
         mode: netem
         distribution:
           kind: lognormal
           p50_ms: 30.0
           p99_ms: 51.6
         bandwidth_mbps: 10.0

Refusal rules
^^^^^^^^^^^^^

CADET refuses a netem or netwaggle link if:

1. ``p99 > 2 * p50`` — the symmetric jitter would need negative delays to
   achieve the low side.
2. Jitter is non-zero and ``bandwidth_mbps`` is not set.

For a lognormal distribution with ``p50_ms: 30`` and ``p99_ms: 51.6``, the
p99 is 51.6 ms, well within the 60 ms limit (2 × 30 ms).
Using ``p99_ms: 307`` with ``p50_ms: 30`` would exceed the limit and be refused.

netwaggle backend
-----------------

Uses the same distribution rules as netem. Requires the emulation container
(``cadet emulate``). For wired links:

.. code-block:: yaml

   links:
     - name: v2i_link
       from: client
       to: server
       network:
         mode: netwaggle
         medium: wired
         distribution:
           kind: lognormal
           p50_ms: 30.0
           p99_ms: 51.6
         bandwidth_mbps: 10.0

For a Wi-Fi link, add ``medium: wifi`` and a ``wifi`` block with AP and
station positions. See :doc:`../explanation/network-backends` for the
full netwaggle Wi-Fi configuration.

Loss
----

Add ``loss: 0.01`` (1%) to any backend:

.. code-block:: yaml

         mode: delay
         loss: 0.01

With ``delay``, messages are dropped at the application layer, so 1% loss
produces roughly 1% unanswered frames. With ``netem`` and ``netwaggle``,
packets are dropped at the kernel layer; TCP retransmits them, so 1%
packet loss appears as elevated p99 latency rather than missing frames.
