Transport tuning
================

The default TCP transport uses Nagle's algorithm, which holds a small
write until the previous ACK arrives. With the ``netem`` backend the ACK
is itself behind the configured delay, so the effective round-trip is
roughly double the configured one-way delay.

Setting ``tcp_nodelay: true`` on the sending node disables Nagle and
reduces the overhead. A 60 ms netem forward link measured at 20 fps:

- Without ``tcp_nodelay``: p50 ≈ 149 ms
- With ``tcp_nodelay``: p50 ≈ 110 ms (one-way delay plus processing)

Config
------

Add ``transport_tuning`` to the sending (client) node:

.. code-block:: yaml

   nodes:
     - name: client
       host: localhost
       role: client
       device: cpu
       transport_tuning:
         tcp_nodelay: true

     - name: server
       host: 127.0.0.1
       port: 15010
       role: server
       device: cpu

``tcp_nodelay`` has no effect on the ``delay`` backend because the delay
line lives inside the transport, not on the network interface. The ACK
returns without the configured delay and Nagle never holds a write.

When to use it
--------------

Enable ``tcp_nodelay`` when:

- You are using ``netem`` or ``netwaggle`` and want to separate one-way
  delay from round-trip TCP overhead.
- Nagle doubling is distorting the latency distribution you want to measure.

Leave it off when:

- You want to see how a real application stack behaves, including TCP
  overhead.
- You are using the ``delay`` backend (no effect either way).
