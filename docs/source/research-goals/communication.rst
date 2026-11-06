Communication
=============

Research question
-----------------

How does the link model — delay distribution, loss, bandwidth — affect
end-to-end latency and application outcomes in a cooperative autonomy stack?

Layers involved
---------------

- **Data Source** — any source
- **NetWaggle** — link backend and distribution
- **Analysis** — frame latency, link statistics report

What to configure
-----------------

``delay`` backend (message-level, exact distribution):

.. literalinclude:: ../../snippets/delay-link.yaml
   :language: yaml
   :caption: Constant delay link — ``docs/snippets/delay-link.yaml``

``netem`` backend (packet-level):

.. literalinclude:: ../../snippets/netem-link.yaml
   :language: yaml
   :caption: Netem lognormal link — ``docs/snippets/netem-link.yaml``

``netwaggle`` backend (Mininet-WiFi wired; requires ``cadet emulate``):

.. literalinclude:: ../../snippets/netwaggle-wired.yaml
   :language: yaml
   :caption: Netwaggle wired link — ``docs/snippets/netwaggle-wired.yaml``

See :doc:`../howto/link-distributions` for the full field reference and
when each backend applies. Key constraint: ``netem`` and ``netwaggle``
cannot render distributions where p99/p50 exceeds 2×; the ``delay`` backend
has no such restriction.

Sweeping over link delay:

.. literalinclude:: ../../snippets/sweep-link-delay.yaml
   :language: yaml
   :caption: Sweep over delay values — ``docs/snippets/sweep-link-delay.yaml``

How to run
----------

.. code-block:: bash

   cadet run examples/reference/backends.yaml \
       --set links.0.network.mode=delay \
       --app-module examples.reference.app

For netwaggle (requires the emulation container):

.. code-block:: bash

   cadet emulate examples/reference/backends.yaml \
       --set links.0.network.mode=netwaggle \
       --app-module examples.reference.app

Where results land
------------------

``outputs/{experiment}/{app}/{run_id}/``:

- ``{run_id}_latency_location_log.csv`` — per-frame end-to-end latency
- ``{run_id}_link_report.csv`` — per-link send/receive counts, drop rate,
  p50/p90/p99 latency
- ``run.json`` — resolved link config

How to extend
-------------

- Model a 5G lognormal tail: see ``docs/snippets/v2v-lognormal-link.yaml``.
- Add loss: set ``links[0].network.loss: 0.01`` (fraction, not percent).
- Multi-hop topology: add a second ``links`` entry between server nodes.

See also
--------

- :doc:`../explanation/netwaggle-layer` — backend architecture
- :doc:`../explanation/network-backends` — backend comparison
- :doc:`../howto/link-distributions` — distribution field reference
- :doc:`../tutorials/03-backends` — backend swap tutorial
