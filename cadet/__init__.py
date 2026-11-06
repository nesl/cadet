"""CADET: a modular platform for evaluating distributed cooperative autonomy.

CADET is app-agnostic. The framework (this ``cadet`` package) owns configuration,
sources, role runtimes, transport/network emulation, metrics and orchestration;
concrete applications live under ``examples/`` and register themselves via the app
plugin registry. ``cadet`` never imports ``examples``.

Sub-packages:
    - :mod:`cadet.config` — the experiment config schema, loader and overrides.
    - :mod:`cadet.core` — core interfaces (source, transport, network, model
      registry, launcher) and the app plugin registry.
    - :mod:`cadet.roles` — generic client/server/load role runtimes.
    - :mod:`cadet.sources` — frame-source backends (carla, replay, synthetic).
    - :mod:`cadet.metrics` — per-frame measurement and run summaries.
    - :mod:`cadet.orchestrator` — run/sweep lifecycle management.
    - :mod:`cadet.cli` — the ``cadet`` command-line entry point.
"""
