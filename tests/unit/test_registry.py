"""Tests for the model registry resolution and preflight.

Covers: per-device artifact resolution, unknown name/device errors, preflight
failure when an artifact is missing, and preflight success once it exists.
"""

import pytest
from conftest import EXAMPLE_CONFIGS

from cadet.config import build_config, load_config_data
from cadet.core import ModelRegistry, PreflightError, device_for_models


def _registry():
    cfg = build_config(load_config_data(str(EXAMPLE_CONFIGS[0])))
    # Device is derived from the server node hosting the model (gpu).
    return ModelRegistry.from_config(cfg.models, device_for_models(cfg))


def test_device_derived_from_server_node():
    cfg = build_config(load_config_data(str(EXAMPLE_CONFIGS[0])))
    assert device_for_models(cfg) == "gpu"


def test_artifact_resolution_per_device():
    reg = _registry()
    assert reg.artifact("yolo11x") == "yolo11x.pt"  # active device is gpu
    assert reg.artifact("yolo11x", device="edge") == "yolo11x.engine"
    assert reg.artifact("yolo11x", device="hailo") == "yolo11x.hef"


def test_unknown_name_and_device_raise():
    reg = _registry()
    with pytest.raises(KeyError):
        reg.artifact("does_not_exist")
    with pytest.raises(KeyError):
        reg.artifact("yolo11x", device="toaster")


def test_preflight_fails_when_missing(tmp_path):
    reg = _registry()
    with pytest.raises(PreflightError) as exc:
        reg.preflight([str(tmp_path)])
    # Both the model weights and the coco labels should be reported missing.
    missing_artifacts = {m.artifact for m in exc.value.missing}
    assert "yolo11x.pt" in missing_artifacts
    assert "coco.names" in missing_artifacts


def test_preflight_passes_when_present(tmp_path):
    reg = _registry()
    (tmp_path / "yolo11x.pt").write_text("")
    (tmp_path / "coco.names").write_text("")
    found = reg.preflight([str(tmp_path)])
    assert {r.name for r in found} == {"yolo11x", "coco_labels"}
    assert all(r.path is not None for r in found)


# --- a host GPU index is not what torch indexes in a container ------------
def test_an_unset_mask_leaves_the_host_index_alone(monkeypatch):
    """A role running as a host process indexes the host's own numbering."""
    from cadet.core.model_registry import visible_gpu_index

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert visible_gpu_index(1) == 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert visible_gpu_index(1) == 1


def test_a_single_masked_device_is_index_zero(monkeypatch):
    """The containerised case: `--gpus device=1` renumbers the survivor to 0,
    and a role told `cuda:1` there fails with "Invalid CUDA 'device=1'"."""
    from cadet.core.model_registry import visible_gpu_index

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert visible_gpu_index(1) == 0
    # The privileged case sets the host index instead, and masking to one
    # device still leaves it at position 0.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert visible_gpu_index(1) == 0


def test_a_mask_of_several_devices_uses_the_position_in_the_mask(monkeypatch):
    from cadet.core.model_registry import visible_gpu_index

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert visible_gpu_index(2) == 0
    assert visible_gpu_index(3) == 1
    # Not in the mask at all and more than one device: nothing better to say
    # than the configured index, and torch will report it as invalid.
    assert visible_gpu_index(7) == 7


def test_the_builtin_pipeline_app_needs_no_app_module():
    """A plugin that ships with the framework resolves without being asked for.

    It did not: `app.plugin: pipeline` raised a bare KeyError out of the
    orchestrator unless the caller also passed --app-module for a module inside
    cadet/, and the two example configs that use it documented only
    --stage-module. Both failed at the first launch for this reason.
    """
    from cadet.core.apps import BUILTIN_APP_MODULES, get_app

    assert "cadet.pipeline.app" in BUILTIN_APP_MODULES
    assert get_app("pipeline").name == "pipeline"


def test_an_unknown_app_says_what_is_registered():
    """The refusal names the registered plugins and how to add one."""
    from cadet.core.apps import UnknownAppError, get_app

    try:
        get_app("not_an_app")
    except UnknownAppError as exc:
        message = str(exc)
    else:
        raise AssertionError("an unknown plugin should not resolve")
    assert "not_an_app" in message
    assert "pipeline" in message
    assert "--app-module" in message
    # Still a KeyError, which is what callers catch.
    assert issubclass(UnknownAppError, KeyError)
