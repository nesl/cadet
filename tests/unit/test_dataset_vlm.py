"""Unit tests for the dataset-mode VLM example.

Covers the four things that could silently drift: every config in
``examples/dataset_vlm`` stays loadable and expands to the documented run count,
the ``dataset`` source yields samples carrying their ground truth, the stub model
and the client/server wiring put the right thing on the wire and score it, and
the VQA and detection scorers compute the documented metric. Nothing here starts
a process or opens a socket; the live path is exercised by running the configs.
"""

import os

import numpy as np
import pytest
import yaml

import examples.dataset_vlm.app as app
import examples.dataset_vlm.scoring as scoring
from cadet.config.loader import load_run_configs
from cadet.config.schema import Config, DatasetSource
from cadet.sources.dataset import DatasetSource as Source
from cadet.sources.dataset import built_in_samples

EXAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples",
    "dataset_vlm",
)

# Config name -> how many runs its sweep expands to (the README's contract).
RUN_COUNTS = {
    "quick.yaml": 1,
    "device.yaml": 1,
    "edge.yaml": 1,
    "cloud.yaml": 1,
    "detection.yaml": 1,
    "sweep.yaml": 3,
    "real.yaml": 1,
}


def _load(name):
    return load_run_configs(os.path.join(EXAMPLE_DIR, name), [])[0]


@pytest.mark.parametrize("name,expected", sorted(RUN_COUNTS.items()))
def test_config_expands_to_the_documented_run_count(name, expected):
    runs = _load(name)
    assert len(runs) == expected
    for config in runs:
        assert config.app.plugin == "dataset_vlm"
        app.DatasetVlmParams.model_validate(config.app.params)


def test_every_config_in_the_directory_is_covered():
    present = {
        name
        for name in os.listdir(EXAMPLE_DIR)
        if name.endswith(".yaml") and not name.startswith("_")
    }
    assert present == set(RUN_COUNTS)


def test_sweep_crosses_the_three_placements():
    seen = {cfg.app.params["placement"] for cfg in _load("sweep.yaml")}
    assert seen == {"device", "edge", "cloud"}


# -- the dataset source -------------------------------------------------------


class _Cfg:
    def __init__(self, source):
        self.source = source


def test_vqa_source_yields_question_and_answers():
    source = Source(_Cfg(DatasetSource(kind="dataset", task="vqa", count=4)))
    source.open()
    frame = source.next_frame()
    assert frame.image.shape[2] == 3
    assert frame.meta["task"] == "vqa"
    assert isinstance(frame.meta["question"], str) and frame.meta["question"]
    assert len(frame.meta["answers"]) >= 1


def test_detection_source_yields_ground_truth_boxes():
    source = Source(_Cfg(DatasetSource(kind="dataset", task="detection", count=4)))
    source.open()
    frame = source.next_frame()
    assert frame.meta["task"] == "detection"
    boxes = frame.meta["gt_boxes"]
    assert boxes and "bbox" in boxes[0] and len(boxes[0]["bbox"]) == 4


def test_source_respects_limit_and_then_stops():
    source = Source(_Cfg(DatasetSource(kind="dataset", task="vqa", count=8, limit=3)))
    source.open()
    emitted = 0
    while source.next_frame() is not None:
        emitted += 1
    assert emitted == 3


def test_built_in_samples_are_deterministic():
    assert built_in_samples("vqa", 4) == built_in_samples("vqa", 4)
    assert len(built_in_samples("detection", 6)) == 6


# -- scoring ------------------------------------------------------------------


def test_vqa_normalisation_matches_the_official_rules():
    assert scoring.normalize_answer("The Dog.") == "dog"
    assert scoring.normalize_answer("two") == "2"
    assert scoring.normalize_answer("  YES! ") == "yes"
    # Number words and articles are canonicalised, so these are one answer.
    assert scoring.normalize_answer("three") == scoring.normalize_answer("3")


def test_vqa_accuracy_is_the_min_of_matches_over_three():
    # Three or more annotators agree -> full credit.
    assert scoring.vqa_accuracy("yes", ["yes", "yes", "yes"]) == 1.0
    # One annotator agrees -> one third.
    assert scoring.vqa_accuracy("yes", ["yes", "no", "no"]) == pytest.approx(1 / 3)
    assert scoring.vqa_accuracy("maybe", ["yes", "no", "no"]) == 0.0


def test_iou_and_detection_matching():
    assert scoring.iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert scoring.iou([0, 0, 10, 10], [100, 100, 10, 10]) == 0.0
    result = scoring.match_detections(
        [{"bbox": [0, 0, 10, 10], "label": "car"}],
        [{"bbox": [0, 0, 10, 10], "label": "car"}],
        iou_thresh=0.5,
    )
    assert result["tp"] == 1 and result["precision"] == 1.0 and result["recall"] == 1.0


def test_detection_label_mismatch_is_not_a_match():
    result = scoring.match_detections(
        [{"bbox": [0, 0, 10, 10], "label": "car"}],
        [{"bbox": [0, 0, 10, 10], "label": "person"}],
        iou_thresh=0.5,
    )
    assert result["tp"] == 0 and result["fp"] == 1 and result["fn"] == 1


# -- the stub model -----------------------------------------------------------


def test_stub_vlm_is_deterministic_and_rule_based():
    model = app.StubVlm("vqa", 0.0)
    blank = np.zeros((16, 16, 3), dtype="uint8")
    assert model.answer(blank, "Is the light on?") == "yes"
    assert model.answer(blank, "How many cars are there?") == "2"
    assert model.answer(blank, "What color is it?") == "red"
    # Same question, same answer.
    assert model.answer(blank, "What is this?") == model.answer(blank, "What is this?")


def test_stub_vlm_detection_returns_fixed_boxes():
    model = app.StubVlm("detection", 0.0)
    boxes = model.detect(np.zeros((64, 96, 3), dtype="uint8"))
    assert len(boxes) == len(app.STUB_DET_BOXES)
    assert all("bbox" in b and "label" in b for b in boxes)


# -- params -------------------------------------------------------------------


def test_params_refuse_a_placement_with_no_declared_cost():
    with pytest.raises(ValueError):
        app.DatasetVlmParams(placement="cloud", infer_ms={"edge": 10.0})


def test_real_model_requires_a_model_id():
    with pytest.raises(ValueError):
        app.DatasetVlmParams(model="hf")
    app.DatasetVlmParams(model="hf", model_id="some/open-model")


# -- client/server wiring -----------------------------------------------------


class _Frame:
    def __init__(self, image, meta):
        self.image = image
        self.meta = meta


class _Ctx:
    def __init__(self):
        self.transport = None


def _config(tmp_path, task, name="dataset_vlm_test"):
    data = yaml.safe_load(
        """
experiment: {name: PLACEHOLDER, app: dataset_vlm, repetitions: 1, seed: 0}
source: {kind: dataset, task: TASK, count: 2}
models: {warmup: false}
nodes:
  - {name: dataset, host: localhost, role: client, device: cpu}
  - {name: server, host: 127.0.0.1, port: 15999, role: server, device: cpu}
app: {plugin: dataset_vlm, params: {task: TASK, model: stub}}
""".replace("PLACEHOLDER", name).replace("TASK", task)
    )
    data["app"]["params"]["placement"] = "edge"
    data["app"]["params"]["infer_ms"] = {"device": 0.0, "edge": 0.0, "cloud": 0.0}
    data["outputs"] = {"root": str(tmp_path), "naming": "{experiment}/{app}/{run_id}"}
    return Config.model_validate(data)


def test_vqa_round_trip_and_scoring(tmp_path):
    config = _config(tmp_path, "vqa")
    server = app.ServerApp(config, [])
    server.setup()
    client = app.ClientApp(config, "run", 0, str(tmp_path / "s.csv"))
    meta = {
        "sample_id": 0,
        "capture_time": 1.0,
        "task": "vqa",
        "question": "Is the light on?",
        "answers": ["yes", "yes", "yes"],
        "multiple_choice_answer": "yes",
    }
    payload = client.process_frame(_Frame(np.zeros((16, 16, 3), "uint8"), meta), _Ctx())
    reply = app.unpack(server.handle(payload))
    server.teardown()
    assert reply["answer"] == "yes"
    client.translate_action(None, (reply, 2.0), _Ctx())
    row = client._rows[0]
    assert row["prediction"] == "yes"
    assert row["vqa_score"] == 1.0
    assert row["e2e_latency_ms"] is not None


def test_detection_round_trip_and_scoring(tmp_path):
    config = _config(tmp_path, "detection")
    server = app.ServerApp(config, [])
    server.setup()
    client = app.ClientApp(config, "run", 0, str(tmp_path / "s.csv"))
    # Ground truth equal to the stub's fixed boxes -> perfect match.
    meta = {
        "sample_id": 0,
        "capture_time": 1.0,
        "task": "detection",
        "gt_boxes": [dict(b) for b in app.STUB_DET_BOXES],
    }
    payload = client.process_frame(_Frame(np.zeros((64, 96, 3), "uint8"), meta), _Ctx())
    reply = app.unpack(server.handle(payload))
    server.teardown()
    assert len(reply["boxes"]) == len(app.STUB_DET_BOXES)
    client.translate_action(None, (reply, 2.0), _Ctx())
    row = client._rows[0]
    assert row["det_precision"] == 1.0 and row["det_recall"] == 1.0 and row["det_f1"] == 1.0


def test_server_payload_carries_no_ground_truth(tmp_path):
    """The server must only ever see the image and the question, never the GT."""
    config = _config(tmp_path, "vqa")
    client = app.ClientApp(config, "run", 0, str(tmp_path / "s.csv"))
    meta = {
        "sample_id": 0,
        "capture_time": 1.0,
        "task": "vqa",
        "question": "What color is it?",
        "answers": ["red", "red", "red"],
        "multiple_choice_answer": "red",
    }
    payload = client.process_frame(_Frame(np.zeros((16, 16, 3), "uint8"), meta), _Ctx())
    message = app.unpack(payload)
    assert "answers" not in message and "multiple_choice_answer" not in message
    assert "question" in message and "frame" in message


def test_client_success_needs_an_answered_sample(tmp_path):
    config = _config(tmp_path, "vqa")
    client = app.ClientApp(config, "run", 0, str(tmp_path / "s.csv"))
    ok, detail = app.DatasetVlmApp.client_success(client)
    assert not ok and "no samples" in detail
    client._sent = 1
    ok, detail = app.DatasetVlmApp.client_success(client)
    assert not ok and "no answer" in detail
    client._answered = 1
    ok, _detail = app.DatasetVlmApp.client_success(client)
    assert ok
