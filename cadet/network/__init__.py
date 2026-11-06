"""Network-emulation backends for CADET links.

What this is for:
    Holds the implementations behind ``links[*].network.mode``. A link either
    gets no emulation, an in-process delay line, or a real emulated network.

What it contains:
    - :mod:`cadet.network.netwaggle`: a Mininet-WiFi backend, adapted from the
      NESL NetWaggle testbed.

    The ``delay`` backend is not here: it is a property of the transport
    (``cadet/core/transport.py`` delay lines, fed by ``cadet/core/network.py``),
    because it perturbs messages inside the sending and receiving processes
    rather than building a network.

How it fits into CADET:
    The orchestrator inspects a run's links and brings up a backend when one is
    needed. Nothing in ``cadet/core`` imports this package, so a CADET install
    without the emulator still runs the ``none`` and ``delay`` backends.

How to configure it:
    Per link, via ``links[*].network`` in the experiment config.
"""
