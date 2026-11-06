"""CADET core interfaces.

What this is for:
    The abstract contracts the rest of CADET is built on — sources, transports,
    network models, the model registry, launchers and the app plugin registry.
    Concrete role runtimes live in :mod:`cadet.roles`.

What it contains:
    Re-exports the key abstract bases and concrete backends so callers can do
    ``from cadet.core import Source, Transport, NetworkModel, ModelRegistry,
    Launcher`` and the app-registry helpers.

How it fits into CADET:
    The config layer (:mod:`cadet.config`) describes *what* to run; these
    interfaces describe *how* the pieces plug together. The runner/orchestrator
    (later) wires concrete implementations selected by the config.
"""

from cadet.core.apps import (
    AppPlugin,
    UnknownAppError,
    available_apps,
    get_app,
    import_app_modules,
    register_app,
)
from cadet.core.launcher import Launcher, LaunchStatus, LocalLauncher
from cadet.core.model_registry import (
    ModelRegistry,
    ModelResolution,
    PreflightError,
    device_for_models,
    torch_device_for,
)
from cadet.core.network import (
    NetworkModel,
    NullNetworkModel,
    SampledNetworkModel,
    build_network_model,
    build_path_network_model,
)
from cadet.core.source import Frame, Source
from cadet.core.transport import FramedSocketTransport, Transport

__all__ = [
    "AppPlugin",
    "available_apps",
    "get_app",
    "import_app_modules",
    "UnknownAppError",
    "register_app",
    "torch_device_for",
    "Launcher",
    "LaunchStatus",
    "LocalLauncher",
    "ModelRegistry",
    "ModelResolution",
    "PreflightError",
    "device_for_models",
    "NetworkModel",
    "NullNetworkModel",
    "SampledNetworkModel",
    "build_network_model",
    "build_path_network_model",
    "Frame",
    "Source",
    "FramedSocketTransport",
    "Transport",
]
