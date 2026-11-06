"""Model registry: logical model name -> concrete artifact per device.

What this is for:
    Resolves a logical model (e.g. ``yolo11x``) to the artifact file for a given
    device type (e.g. ``gpu`` -> ``yolo11x.pt``, ``edge`` -> ``yolo11x.engine``,
    ``hailo`` -> ``yolo11x.hef``), and preflights that the needed artifacts
    actually exist before a run starts.

What it contains:
    - :class:`ModelResolution`: a resolved (name, device, artifact, path) record.
    - :class:`ModelRegistry`: resolve + preflight, built from a schema ``Models``.
    - :class:`PreflightError`: raised when a required artifact is missing.

How it fits into CADET:
    Built from ``config.models`` (:mod:`cadet.config.schema`). Detection/tracking
    roles ask it for the artifact matching the node's device type; the runner
    calls :meth:`ModelRegistry.preflight` up front so missing weights fail fast
    rather than mid-experiment.

How to configure it:
    ``models.registry`` maps logical names to ``per_device`` artifacts;
    ``models.device`` picks the active device type. Artifacts are looked up under
    the search directories passed to :meth:`preflight`/:meth:`resolve`.
"""

import os
from typing import Dict, List, Optional


def device_for_models(config) -> str:
    """Return the device type of the node that hosts the model.

    Inference runs on the ``server`` node, so its ``device`` is the one the model
    registry resolves against. Falls back to a ``sim`` node, then to ``cpu``.
    """
    by_role = {node.role: node for node in config.nodes}
    host = by_role.get("server") or by_role.get("sim")
    return host.device if host is not None else "cpu"


def visible_gpu_index(gpu_index: int) -> int:
    """Map a host GPU index onto the index torch will actually see.

    ``CUDA_VISIBLE_DEVICES`` renumbers: whatever it lists becomes 0, 1, ... in
    that order, so a config's *host* index is not what torch indexes once the
    variable is set - and the container launchers set it, because
    ``--gpus device=N`` filters the devices through the device cgroup and the
    one that survives is renumbered to 0.

    Measured, with the server in a container pinned to the host's second GPU:
    the role was told ``cuda:1`` from its config and ultralytics refused with
    ``Invalid CUDA 'device=1' requested`` and ``torch.cuda.is_available():
    False``, while ``cuda:0`` is that same card. Unset, nothing is renumbered
    and the host index is the right answer, which is the case for a role that
    runs as a host process.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not raw or not raw.strip():
        return gpu_index
    visible = [part.strip() for part in raw.split(",") if part.strip()]
    if str(gpu_index) in visible:
        return visible.index(str(gpu_index))
    if len(visible) == 1:
        # One device, and it is not the host index the config names: something
        # upstream has already pinned it, so position 0 is the only device
        # there is. This is the ordinary containerised case.
        return 0
    return gpu_index


def torch_device_for(config) -> str:
    """Return the preferred torch device string for the hosting node.

    Maps the server/sim node to a torch device: an explicit ``gpu_index`` ->
    ``cuda:<i>``, where ``i`` is that device's index *as this process sees it*
    (see :func:`visible_gpu_index`); a gpu/edge ``device`` type -> ``cuda``;
    otherwise ``cpu``. This is torch-free (returns a string); callers fall back
    to cpu if cuda is absent.
    """
    by_role = {node.role: node for node in config.nodes}
    host = by_role.get("server") or by_role.get("sim")
    if host is None:
        return "cpu"
    if host.gpu_index is not None:
        return "cuda:{}".format(visible_gpu_index(host.gpu_index))
    if host.device in ("gpu", "edge"):
        return "cuda"
    return "cpu"


class PreflightError(Exception):
    """Raised when one or more required model artifacts are missing."""

    def __init__(self, missing: List["ModelResolution"]):
        self.missing = missing
        names = ", ".join("{}[{}]->{}".format(m.name, m.device, m.artifact) for m in missing)
        super().__init__("missing model artifacts: {}".format(names))


class ModelResolution:
    """A logical model resolved to a concrete artifact for one device type."""

    def __init__(self, name: str, device: str, artifact: str, path: Optional[str] = None):
        self.name = name
        self.device = device
        self.artifact = artifact
        self.path = path  # absolute path if found during preflight, else None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "ModelResolution(name={!r}, device={!r}, artifact={!r}, path={!r})".format(
            self.name, self.device, self.artifact, self.path
        )


class ModelRegistry:
    """Resolve logical models to per-device artifacts and preflight them."""

    def __init__(self, registry: Dict[str, Dict[str, str]], device: str):
        """``registry``: name -> {device_type: artifact}. ``device``: active type."""
        self._registry = registry
        self._device = device

    @classmethod
    def from_config(cls, models, device: str) -> "ModelRegistry":
        """Build from a schema ``Models`` object and the hosting node's device.

        ``device`` comes from the node that runs the model (see
        :func:`device_for_models`), not from the models section itself.
        """
        reg = {name: dict(entry.per_device) for name, entry in models.registry.items()}
        return cls(reg, device)

    def artifact(self, name: str, device: Optional[str] = None) -> str:
        """Return the artifact filename for ``name`` on the active/given device."""
        dev = device or self._device
        if name not in self._registry:
            raise KeyError("unknown logical model {!r}".format(name))
        per_device = self._registry[name]
        if dev not in per_device:
            raise KeyError("model {!r} has no artifact for device {!r}".format(name, dev))
        return per_device[dev]

    def resolve(
        self, name: str, search_dirs: List[str], device: Optional[str] = None
    ) -> ModelResolution:
        """Resolve ``name`` to an artifact and locate it under ``search_dirs``."""
        dev = device or self._device
        artifact = self.artifact(name, dev)
        path = self._locate(artifact, search_dirs)
        return ModelResolution(name, dev, artifact, path)

    def preflight(
        self, search_dirs: List[str], device: Optional[str] = None
    ) -> List[ModelResolution]:
        """Resolve every model for the active device; raise if any are missing.

        Returns the list of successful resolutions. Raises :class:`PreflightError`
        listing everything that could not be located.
        """
        dev = device or self._device
        found: List[ModelResolution] = []
        missing: List[ModelResolution] = []
        for name in self._registry:
            res = self.resolve(name, search_dirs, dev)
            (found if res.path else missing).append(res)
        if missing:
            raise PreflightError(missing)
        return found

    @staticmethod
    def _locate(artifact: str, search_dirs: List[str]) -> Optional[str]:
        """Return the first existing path for ``artifact`` across ``search_dirs``."""
        if os.path.isabs(artifact) and os.path.exists(artifact):
            return artifact
        for d in search_dirs:
            candidate = os.path.join(d, artifact)
            if os.path.exists(candidate):
                return candidate
        return None
