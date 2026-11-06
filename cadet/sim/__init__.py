"""Simulator-side ownership: the versions that must agree, and the containers.

What this is for:
    The simulator is not one program but a set of pieces that have to match: a
    CARLA server, the Python client egg that talks to it, and a ScenarioRunner
    tree that drives it. When they disagree the failure is never "versions
    disagree" - it is an import error, a silent protocol mismatch, or a scenario
    that loads and then behaves differently. This package declares the set in
    config and checks it before a run.

What it contains:
    - :mod:`cadet.sim.versions`: the declared versions and the preflight check.

How it fits into CADET:
    ``cadet.orchestrator.runner.preflight`` calls
    :func:`cadet.sim.versions.check_versions`, so a mismatch stops a run before
    anything starts.

How to configure it:
    The ``source.carla`` and ``source.scenario`` blocks of the experiment
    config; see :class:`cadet.config.schema.CarlaSettings`,
    :class:`cadet.config.schema.Scenario` and
    :class:`cadet.config.schema.ScenarioRunnerPin`.
"""
