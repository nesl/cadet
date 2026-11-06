"""Validate a config's pipeline against the stage registry.

The schema checks the structure of ``pipeline.stages`` (node references exist,
names are unique, enums are valid) without importing stage code. This module
adds the semantic check that each ``impl`` is actually registered, which needs
the stage modules imported first, so it is separate from schema validation (the
same split the app registry uses). Provides :func:`validate_pipeline`, which
returns a list of human-readable problems (empty means the pipeline is
resolvable). Called after stage modules are imported and before a pipeline is
built or run. Nothing to configure; it reflects the registry and the config.
"""

from typing import List

from cadet.pipeline.registry import _REGISTRY, available_stages


def validate_pipeline(config) -> List[str]:
    """Return problems with ``config.pipeline`` against the registry (empty == ok)."""
    problems: List[str] = []
    pipeline = getattr(config, "pipeline", None)
    if pipeline is None or not getattr(pipeline, "stages", None):
        return problems
    for spec in pipeline.stages:
        if spec.impl not in _REGISTRY:
            problems.append(
                "stage {!r}: impl {!r} is not registered (known: {})".format(
                    spec.name, spec.impl, ", ".join(available_stages()) or "none"
                )
            )
    return problems
