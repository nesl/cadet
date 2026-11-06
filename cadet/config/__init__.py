"""CADET configuration package.

What this is for:
    The typed configuration layer for CADET experiments: the schema, the typed
    per-app parameter models, the weather presets, and the loader that applies
    overrides, expands sweeps and derives run ids.

What it contains:
    Re-exports the most-used names so callers can do ``from cadet.config import
    Config, load_run_configs, run_id``.

How it fits into CADET:
    Everything that needs to read an experiment definition (the ``cadet
    validate`` CLI, and later the runner/orchestrator and roles) depends on this
    package rather than on the legacy ``cadet/params.py`` module.
"""

from cadet.config.implemented import unimplemented_features
from cadet.config.loader import (
    ConfigError,
    build_config,
    config_hash,
    expand_sweep,
    format_validation_error,
    load_config_data,
    load_run_configs,
    load_yaml,
    resolve_app_params,
    run_id,
    seed_for_rep,
)
from cadet.config.schema import Config

__all__ = [
    "Config",
    "ConfigError",
    "build_config",
    "config_hash",
    "expand_sweep",
    "format_validation_error",
    "load_config_data",
    "load_run_configs",
    "load_yaml",
    "resolve_app_params",
    "run_id",
    "seed_for_rep",
    "unimplemented_features",
]
