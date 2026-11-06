"""Dataset frame source: deterministic sample replay for model evaluation.

What this is for:
    A :class:`cadet.core.source.Source` that replays an evaluation dataset, one
    sample per frame, carrying the input image AND the sample's ground truth in
    the frame metadata. This is the "Dataset Mode" source: a consumer runs a
    model on each frame and scores its answer against the ground truth the frame
    carries. Two tasks are supported:

    - ``vqa``: each frame carries ``question`` and the human ``answers`` (plus the
      ``multiple_choice_answer``), so an answer can be scored with the VQA metric.
    - ``detection``: each frame carries ``gt_boxes`` (``[x, y, w, h]`` plus a
      ``label``), so predicted boxes can be scored by IoU.

What it contains:
    - :class:`DatasetSource`: open/next_frame/close/metadata over the samples.
    - :func:`built_in_samples`: the deterministic, in-process sample set used when
      no ``path`` is configured, so the stub config and the tests need no file.

How it fits into CADET:
    Built from a ``source`` of ``kind: dataset`` (see
    :class:`cadet.config.schema.DatasetSource`). With ``path`` unset it generates
    its samples in process (no file, no download); with ``path`` set it reads a
    JSON manifest and the images it names. The same consumer app runs over either.

How to configure it:
    ``source.task`` picks ``vqa`` or ``detection``; ``source.path`` points at a
    folder with a manifest and images (the example's ``fetch_dataset.py`` writes
    one), or is left unset for the built-in set; ``source.count`` sizes the
    built-in set and ``source.limit`` caps either.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from cadet.core.source import Frame, Source

# Default manifest file name per task, inside ``source.path``.
_MANIFEST_DEFAULT = {"vqa": "vqa_manifest.json", "detection": "detection_manifest.json"}

# Geometry of a built-in generated image. Small, so a stub run stays in memory
# and the wire payload is tiny; three channels so it looks like an RGB frame.
_STUB_HW = (64, 96)

# The built-in VQA vocabulary. A question is written so that a simple keyword
# answerer (the example's stub model) has a definite answer, and the ground truth
# is set to match it for some samples and to differ for others, so a stub run
# produces a deterministic accuracy that is neither 0 nor 1 and both branches of
# the scorer are exercised. The dataset does not depend on the model: it only
# states a question and a human answer, exactly as a real VQA record does.
_STUB_VQA = [
    # (question, answers)  -- answers is the list a VQA record carries.
    ("What color is the sign? Answer in one word.", ["red", "red", "red"]),
    ("How many people are there? Answer in one word.", ["three", "3", "three"]),
    ("Is the light on? Answer in one word.", ["yes", "yes", "yes"]),
    ("What animal is shown? Answer in one word.", ["dog", "dog", "a dog"]),
    ("What color is the car? Answer in one word.", ["blue", "blue", "blue"]),
    ("How many cars are there? Answer in one word.", ["two", "2", "two"]),
    ("Are there any birds? Answer in one word.", ["no", "no", "no"]),
    ("What is the object? Answer in one word.", ["cat", "cat", "a cat"]),
]

# The built-in detection ground truth: a couple of boxes per frame, in
# ``[x, y, w, h]`` with a label, over the _STUB_HW canvas.
_STUB_DETECTION = [
    [{"bbox": [8, 8, 24, 24], "label": "car"}, {"bbox": [48, 30, 20, 20], "label": "person"}],
    [{"bbox": [10, 10, 30, 30], "label": "car"}],
    [{"bbox": [4, 20, 18, 18], "label": "person"}, {"bbox": [60, 10, 24, 24], "label": "car"}],
    [{"bbox": [20, 20, 40, 30], "label": "car"}],
    [{"bbox": [2, 2, 16, 16], "label": "person"}],
    [{"bbox": [30, 30, 28, 24], "label": "car"}, {"bbox": [70, 40, 18, 18], "label": "person"}],
    [{"bbox": [12, 12, 22, 22], "label": "car"}],
    [{"bbox": [40, 8, 30, 30], "label": "person"}],
]


def _stub_image(index: int):
    """Return a deterministic RGB image whose content encodes ``index``."""
    import numpy as np

    height, width = _STUB_HW
    image = np.full((height, width, 3), index % 256, dtype="uint8")
    # A deterministic bright block, so detection boxes have something to align to
    # and the image is not one flat colour.
    image[8:32, 8:32] = (index * 37) % 256
    return image


def built_in_samples(task: str, count: int) -> List[Dict[str, Any]]:
    """Return ``count`` deterministic in-process samples for ``task``.

    These stand in for a fetched dataset so the stub config and the tests run
    with no file and no download. Each record has the same shape a manifest entry
    has, minus the image file name (the image is generated here instead).
    """
    out: List[Dict[str, Any]] = []
    for index in range(count):
        if task == "vqa":
            question, answers = _STUB_VQA[index % len(_STUB_VQA)]
            out.append(
                {
                    "sample_id": index,
                    "question_id": 1000 + index,
                    "question": question,
                    "answers": list(answers),
                    "multiple_choice_answer": answers[0],
                }
            )
        else:
            boxes = _STUB_DETECTION[index % len(_STUB_DETECTION)]
            out.append({"sample_id": index, "gt_boxes": [dict(box) for box in boxes]})
    return out


class DatasetSource(Source):
    """Replay evaluation samples, each frame carrying its own ground truth."""

    def __init__(self, config, logger=None):
        src = config.source
        self._task = src.task
        self._path = src.path
        self._manifest = src.manifest
        self._fps = src.fps
        self._limit = src.limit
        self._count = src.count
        self._log = logger or (lambda m: None)
        self._samples: List[Dict[str, Any]] = []
        self._i = 0
        self._emitted = 0
        self._last = None

    def open(self) -> None:
        if self._path is None:
            self._samples = built_in_samples(self._task, self._count)
            self._log(
                "[dataset] built-in {} set: {} sample(s)".format(self._task, len(self._samples))
            )
        else:
            self._samples = self._load_manifest()
            self._log(
                "[dataset] {} manifest {}: {} sample(s)".format(
                    self._task, self._path, len(self._samples)
                )
            )
        if not self._samples:
            raise RuntimeError("dataset source has no samples (task={})".format(self._task))
        self._i = 0
        self._emitted = 0

    def _load_manifest(self) -> List[Dict[str, Any]]:
        name = self._manifest or _MANIFEST_DEFAULT[self._task]
        manifest_path = os.path.join(self._path, name)
        if not os.path.isfile(manifest_path):
            raise RuntimeError(
                "no manifest at {}; run the example's fetch_dataset.py first".format(manifest_path)
            )
        with open(manifest_path) as handle:
            data = json.load(handle)
        samples = data.get("samples") if isinstance(data, dict) else data
        if not isinstance(samples, list):
            raise RuntimeError("manifest {} is not a list of samples".format(manifest_path))
        return samples

    def _image_for(self, sample: Dict[str, Any]):
        """Return the RGB image for a sample: generated, or read from disk."""
        if self._path is None:
            return _stub_image(int(sample.get("sample_id", self._emitted)))
        image_name = sample.get("image")
        if not image_name:
            raise RuntimeError("manifest sample {} names no image".format(sample.get("sample_id")))
        import cv2

        image_path = os.path.join(self._path, image_name)
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise RuntimeError("could not read image {}".format(image_path))
        return bgr[:, :, ::-1].copy()  # BGR -> RGB

    def next_frame(self) -> Optional[Frame]:
        if self._limit is not None and self._emitted >= self._limit:
            return None
        if self._i >= len(self._samples):
            return None
        if self._fps and self._fps > 0 and self._last is not None:
            dt = 1.0 / self._fps - (time.time() - self._last)
            if dt > 0:
                time.sleep(dt)
        sample = self._samples[self._i]
        self._i += 1
        image = self._image_for(sample)
        meta: Dict[str, Any] = {
            "index": self._emitted,
            "capture_time": time.time(),
            "task": self._task,
            "sample_id": sample.get("sample_id", self._emitted),
        }
        if self._task == "vqa":
            meta["question"] = sample["question"]
            meta["question_id"] = sample.get("question_id")
            meta["answers"] = sample.get("answers") or []
            meta["multiple_choice_answer"] = sample.get("multiple_choice_answer")
        else:
            meta["gt_boxes"] = sample.get("gt_boxes") or []
        self._emitted += 1
        self._last = time.time()
        return Frame(image=image, meta=meta)

    def close(self) -> None:
        pass

    def metadata(self) -> Dict[str, Any]:
        return {
            "kind": "dataset",
            "task": self._task,
            "path": self._path,
            "samples": len(self._samples),
        }
