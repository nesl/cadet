"""Stage registry: map an ``impl`` name to a :class:`Stage` class.

Config names a stage implementation by a short string (``impl: yolo``); the
builder needs to turn that string into a class. Stages register themselves here
on import (like the app plugin registry), so the framework never hard-codes the
stage library and example packages can add custom stages. Provides
:func:`register_stage` (class decorator), :func:`get_stage`,
:func:`available_stages`, and :func:`import_stage_modules`. The CLI imports
stage modules before building a pipeline; the config layer validates ``impl``
names against this registry (see :func:`cadet.pipeline.validate.validate_pipeline`).
Nothing to configure; it is a process-wide registry populated by imports.
"""

import importlib
from typing import Dict, List, Type

from cadet.pipeline.stage import Stage

_REGISTRY: Dict[str, Type[Stage]] = {}


def register_stage(impl: str):
    """Class decorator registering a :class:`Stage` subclass under ``impl``."""

    def _decorate(cls: Type[Stage]) -> Type[Stage]:
        if not impl:
            raise ValueError("stage impl name must be non-empty")
        if impl in _REGISTRY and _REGISTRY[impl] is not cls:
            raise ValueError("stage impl {!r} already registered".format(impl))
        cls.impl_name = impl
        _REGISTRY[impl] = cls
        return cls

    return _decorate


def get_stage(impl: str) -> Type[Stage]:
    """Return the stage class registered under ``impl`` (raises ``KeyError``)."""
    return _REGISTRY[impl]


def available_stages() -> List[str]:
    """Return the sorted impl names of all registered stages."""
    return sorted(_REGISTRY)


def import_stage_modules(modules: List[str]) -> None:
    """Import each module so its ``@register_stage`` classes register."""
    for mod in modules:
        importlib.import_module(mod)
