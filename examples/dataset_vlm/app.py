"""Dataset-mode VLM serving app: serve one model over a dataset and score it.

What this is for:
    One client streams an evaluation dataset (the :mod:`cadet.sources.dataset`
    source) to one inference server. The server loads a SINGLE vision-language
    model and answers each sample; the client scores the answer against the
    ground truth the dataset carries. The one thing the example varies is where
    the model is served - on the device, on an edge box, or in the cloud - which
    is the placement axis, exactly as the distributed sensing example varies its detector's
    placement. This is the paper's "Dataset Mode": a model evaluated over a
    dataset with model-, system- and task-level numbers recorded per run.

    Only single-model serving is implemented here: one model is loaded and run
    over the dataset. Choosing among models, switching models or routing a query
    to a model is deliberately out of scope and is not present anywhere in this
    example.

What it contains:
    - :class:`DatasetVlmParams`: the task, the model kind, the placement and the
      per-placement stub inference cost.
    - :class:`StubVlm`, :class:`HfVlm` and :func:`build_vlm`: the model runner,
      mirroring the distributed sensing example's ``StubDetector`` / ``YoloDetector`` / ``build_detector``.
    - :class:`ClientApp`: the dataset side. Stream samples, transmit, receive the
      answer, score it against the ground truth, and log per-sample.
    - :class:`ServerApp`: the serving side. Load one model, answer each sample.
    - :class:`DatasetVlmApp`: the plugin (``--app-module examples.dataset_vlm.app``).

How it fits into CADET:
    Runs on the generic runtimes: one ``cadet client`` for the dataset node and
    one ``cadet serve`` for the model node, the same shape as distributed sensing. The model is
    placed on the server node, whose ``device`` decides cpu vs cuda, so the
    placement axis is a config choice and needs no code change.

How to configure it:
    ``app.params.placement`` names the tier the model is served on and
    ``app.params.infer_ms`` gives that tier its stub inference cost; the configs
    in this directory set both. ``app.params.model: hf`` serves a real model
    named by ``app.params.model_id`` instead of the stub (no token is ever used).
"""

import csv
import os
import time
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Literal

from cadet.config.loader import config_hash
from cadet.core.apps import AppPlugin, register_app
from cadet.core.codec import decode as wire_decode
from cadet.core.codec import encode as wire_encode
from cadet.metrics.link_report import MessageStats

# Local imports work because the example dir is on sys.path when the app module
# is imported by name (see cadet.core.apps.import_app_modules), the way the distributed sensing
# sibling scripts import summarise.
try:
    from examples.dataset_vlm import scoring
except Exception:  # pragma: no cover - direct-dir import fallback
    import scoring

# How long the client keeps draining replies after the dataset is exhausted.
DRAIN_TIMEOUT_S = 5.0

# A frame goes on the wire JPEG-encoded, so the bytes-per-sample a run reports is
# what a camera or a dataset loader would actually send, not the array's dtype.
FRAME_FORMAT = ".jpg"
JPEG_QUALITY = 90

# The stub VLM's fixed detection guesses. The first is grounded in the generated
# image (its bright block sits at this box); the second is a fixed extra guess,
# so a stub detection run has both a true and a false positive to score.
STUB_DET_BOXES = [
    {"bbox": [8, 8, 24, 24], "label": "car"},
    {"bbox": [50, 30, 20, 20], "label": "person"},
]

CLIENT_COLUMNS = [
    "sample_id",
    "task",
    "capture_time",
    "send_time",
    "reply_time",
    "e2e_latency_ms",
    "server_infer_ms",
    "prediction",
    "ground_truth",
    "vqa_score",
    "det_precision",
    "det_recall",
    "det_f1",
]


def pack(message: Dict) -> bytes:
    """Encode one wire message (the encoding the flat examples all use)."""
    return wire_encode(message)


def unpack(payload: bytes) -> Dict:
    """Decode one wire message produced by :func:`pack`."""
    return wire_decode(payload)


def node_id(node_name=None) -> str:
    """Return this node's identifier: its node name, or a host+pid fallback."""
    if node_name:
        return str(node_name)
    import socket

    return "n{}-{:05d}".format(socket.gethostname(), os.getpid() % 100000)


def encode_frame(image, quality: int = JPEG_QUALITY) -> bytes:
    """JPEG-encode one RGB frame."""
    import cv2

    ok, buffer = cv2.imencode(
        FRAME_FORMAT, image[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("could not encode a frame of shape {}".format(image.shape))
    return buffer.tobytes()


def decode_frame(payload: bytes):
    """Decode a JPEG payload back into an RGB array."""
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(payload, dtype="uint8"), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("could not decode a {}-byte frame".format(len(payload)))
    return image[:, :, ::-1]


class DatasetVlmParams(BaseModel):
    """Parameters of the dataset-VLM app."""

    model_config = ConfigDict(extra="forbid")
    plugin: Literal["dataset_vlm"] = "dataset_vlm"
    # The evaluation task; it must match source.task.
    task: Literal["vqa", "detection"] = "vqa"
    # "stub" needs no weights and runs anywhere; "hf" serves a real model.
    model: Literal["stub", "hf"] = "stub"
    # The tier the model is served on. It does not change the topology (the model
    # always serves on the server node); it names the tier for the record and
    # selects the stub inference cost, mirroring the distributed sensing example's detector placement.
    placement: Literal["device", "edge", "cloud"] = "edge"
    # Stub inference cost per tier, in milliseconds. One mapping rather than one
    # value, so the placement axis can be swept on its own (a swept cost that had
    # to move with the placement would pair a cloud cost with a device run). The
    # order stands for a constrained device, an edge box and a cloud accelerator.
    infer_ms: Dict[str, float] = Field(
        default_factory=lambda: {"device": 200.0, "edge": 80.0, "cloud": 25.0}
    )
    # Real-model settings (ignored by the stub). model_id names an openly
    # available model; no authentication token is ever read or required.
    model_id: Optional[str] = None
    max_new_tokens: int = Field(default=8, ge=1, le=256)
    prompt_suffix: str = " Please answer in one word."
    # Detection match threshold for scoring (COCO-style single-threshold IoU).
    iou_thresh: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _cost_declared_for_placement(self) -> "DatasetVlmParams":
        if self.placement not in self.infer_ms:
            raise ValueError(
                "infer_ms declares no cost for placement {!r}; it has {}".format(
                    self.placement, sorted(self.infer_ms)
                )
            )
        return self

    @model_validator(mode="after")
    def _real_model_needs_an_id(self) -> "DatasetVlmParams":
        if self.model == "hf" and not self.model_id:
            raise ValueError("model: hf requires model_id (an openly available model name)")
        return self

    @property
    def infer_cost_s(self) -> float:
        """The stub model's inference cost on the tier it is placed on."""
        return float(self.infer_ms[self.placement]) / 1000.0


class StubVlm:
    """Deterministic fake VLM: no weights, no GPU, a fixed compute cost.

    For VQA it answers from a small keyword rule over the question, so a given
    question always yields the same one-word answer; this is a stand-in for a
    model, not a claim about accuracy. For detection it returns a fixed pair of
    boxes, the first grounded in the generated image. The point is a run that is
    reproducible and needs nothing downloaded, so the wiring and the scorer can
    be exercised on CPU.
    """

    def __init__(self, task: str, cost_s: float):
        self._task = task
        self._cost_s = cost_s

    def _pay(self) -> None:
        if self._cost_s > 0:
            time.sleep(self._cost_s)

    def answer(self, image, question: str) -> str:
        """Return a deterministic one-word answer to a VQA question."""
        self._pay()
        q = (question or "").strip().lower()
        first = q.split()[0] if q else ""
        if first in ("is", "are", "does", "do", "was", "were", "has", "have", "can"):
            return "yes"
        if "how many" in q:
            return "2"
        if "color" in q or "colour" in q:
            return "red"
        if "animal" in q:
            return "dog"
        return "cat"

    def detect(self, image) -> List[Dict]:
        """Return a deterministic list of boxes for one frame."""
        self._pay()
        return [dict(box) for box in STUB_DET_BOXES]


class HfVlm:
    """A single real vision-language model served through transformers.

    This is the plain "load one model, run it over a dataset" path extracted from
    the reference serving scripts, with everything about multiple models or
    choosing among them left out. It never reads an authentication token: only an
    openly available ``model_id`` is loadable. It serves the VQA task; a real
    model for the detection task is out of scope (the detection path is
    demonstrated with the stub and the IoU scorer).
    """

    def __init__(self, model_id: str, device: str, max_new_tokens: int):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self._torch = torch
        self._device = device
        self._max_new_tokens = max_new_tokens
        dtype = torch.float16 if "cuda" in device else torch.float32
        # No token, no auth: an openly available model only.
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        self._model.eval()

    def answer(self, image, question: str) -> str:
        """Run the model on one (image, question) and return its answer text."""
        from PIL import Image

        pil = Image.fromarray(image)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": question}],
            }
        ]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(images=pil, text=prompt, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens)
        trimmed = generated[0][inputs["input_ids"].shape[1]:]
        text = self._processor.decode(trimmed, skip_special_tokens=True).strip()
        return text.split()[0] if text else ""

    def detect(self, image) -> List[Dict]:
        raise NotImplementedError(
            "the real (hf) model serves the vqa task; the detection task uses the stub model"
        )


def build_vlm(config, params: DatasetVlmParams, logger=None):
    """Return the model runner for this run: the stub, or a real model.

    Mirrors the distributed sensing example's ``build_detector``. The stub needs no weights and pays the
    configured cost, which is what the placement axis varies. ``model: hf``
    resolves ``model_id`` through transformers on the device the config gives the
    server node (``torch_device_for``), with no token.
    """
    log = logger or (lambda m: None)
    if params.model == "stub":
        log(
            "[dataset_vlm] stub model, task={} placement={} ({:.0f} ms/sample)".format(
                params.task, params.placement, params.infer_cost_s * 1000.0
            )
        )
        return StubVlm(params.task, params.infer_cost_s)
    from cadet.core.model_registry import torch_device_for

    device = torch_device_for(config)
    try:
        import torch

        if "cuda" in device and not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"
    log("[dataset_vlm] model {} on {}".format(params.model_id, device))
    return HfVlm(params.model_id, device, params.max_new_tokens)


class ClientApp:
    """The dataset node: stream samples, receive answers, score, log per-sample."""

    def __init__(self, config, run_id, rep, csv_path, logger=None, node_name=None):
        self.cfg = config
        self.params = DatasetVlmParams.model_validate(config.app.params)
        self._log = logger or (lambda m: None)
        self.node = node_id(node_name)
        self.csv_path = os.path.join(
            os.path.dirname(csv_path), "samples_{}.csv".format(self.node)
        )
        self.transport = None
        self._rows: Dict[int, Dict] = {}
        self._gt: Dict[int, Dict] = {}
        self._sent = 0
        self._answered = 0
        self._current = None
        self._source_done = False
        self._server_closed = False
        self._drain_started = None
        from queue import Queue
        from threading import Thread

        self._Thread = Thread
        self._replies = Queue()

    def setup(self, ctx) -> None:
        self.transport = ctx.transport
        # Warm up the image codec so the first sample does not pay its first-call
        # cost as if it were latency (the client has no warm-up hook of its own).
        import numpy as np

        encode_frame(np.zeros((8, 8, 3), dtype="uint8"))
        self._Thread(target=self._listen, daemon=True).start()

    def _listen(self) -> None:
        while True:
            try:
                raw = self.transport.recv()
            except Exception:
                self._server_closed = True
                return
            if raw is None:
                self._server_closed = True
                return
            try:
                reply = unpack(raw)
            except Exception:
                continue
            self._replies.put((reply, time.time()))

    def poll_reply(self, ctx):
        from queue import Empty

        try:
            return self._replies.get_nowait()
        except Empty:
            return None

    def translate_action(self, frame, reply, ctx) -> None:
        if frame is None:
            self._source_done = True
        if reply is None:
            return
        reply, arrived_at = reply
        sample_id = reply.get("sample_id")
        row = self._rows.get(sample_id)
        if row is None:
            return
        self._answered += 1
        row["reply_time"] = round(arrived_at, 6)
        row["e2e_latency_ms"] = _rounded((arrived_at - row["capture_time"]) * 1000.0)
        row["server_infer_ms"] = _rounded(reply.get("infer_ms"))
        self._score(row, reply, self._gt.get(sample_id, {}))

    def _score(self, row, reply, gt) -> None:
        """Score one reply against the ground truth and fill the row's columns."""
        if self.params.task == "vqa":
            prediction = reply.get("answer", "")
            answers = gt.get("answers") or []
            row["prediction"] = prediction
            row["ground_truth"] = gt.get("multiple_choice_answer")
            row["vqa_score"] = round(scoring.vqa_accuracy(prediction, answers), 4)
        else:
            predicted = reply.get("boxes") or []
            gt_boxes = gt.get("gt_boxes") or []
            result = scoring.match_detections(predicted, gt_boxes, self.params.iou_thresh)
            row["prediction"] = len(predicted)
            row["ground_truth"] = len(gt_boxes)
            row["det_precision"] = round(result["precision"], 4)
            row["det_recall"] = round(result["recall"], 4)
            row["det_f1"] = round(result["f1"], 4)

    def should_transmit(self, ctx) -> bool:
        return not self._source_done

    def process_frame(self, frame, ctx) -> bytes:
        sample_id = frame.meta.get("sample_id", self._sent)
        capture_time = float(frame.meta.get("capture_time") or time.time())
        message = {
            "sample_id": sample_id,
            "task": self.params.task,
            "capture_t": capture_time,
            "frame": encode_frame(frame.image),
        }
        if self.params.task == "vqa":
            message["question"] = frame.meta.get("question", "")
        payload = pack(message)
        self._rows[sample_id] = {
            "sample_id": sample_id,
            "task": self.params.task,
            "capture_time": round(capture_time, 6),
            "send_time": None,
            "reply_time": None,
            "e2e_latency_ms": None,
            "server_infer_ms": None,
            "prediction": None,
            "ground_truth": None,
            "vqa_score": None,
            "det_precision": None,
            "det_recall": None,
            "det_f1": None,
        }
        # The ground truth stays on the client; only the image (and question)
        # cross the wire, so the server scores nothing and serves only the model.
        self._gt[sample_id] = dict(frame.meta)
        self._current = sample_id
        return payload

    def on_frame_sent(self, frame, ctx) -> None:
        row = self._rows.get(self._current)
        if row is not None:
            row["send_time"] = time.time()
            self._sent += 1

    def collect_metrics(self, frame, ctx) -> None:
        """Nothing extra per tick: the per-sample rows carry the measurements."""

    def is_done(self, ctx) -> bool:
        if self._server_closed:
            return True
        if not self._source_done or not self._sent:
            return False
        if self._answered >= self._sent:
            return True
        if self._drain_started is None:
            self._drain_started = time.time()
            return False
        return (time.time() - self._drain_started) >= DRAIN_TIMEOUT_S

    def message_stats(self):
        latencies = [
            row["e2e_latency_ms"]
            for row in self._rows.values()
            if row["e2e_latency_ms"] is not None
        ]
        return MessageStats(sent=self._sent, received=self._answered, latencies_ms=latencies)

    def finish(self, ctx) -> None:
        rows = [self._rows[sid] for sid in sorted(self._rows) if self._rows[sid]["send_time"]]
        for row in rows:
            row["send_time"] = round(row["send_time"], 6)
        with open(self.csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CLIENT_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        self._log(
            "[dataset_vlm] node {} task={} sent={} answered={} {} csv={}".format(
                self.node,
                self.params.task,
                self._sent,
                self._answered,
                self._headline(rows),
                self.csv_path,
            )
        )

    def _headline(self, rows) -> str:
        """A one-line score summary for the client log."""
        if self.params.task == "vqa":
            scores = [r["vqa_score"] for r in rows if r["vqa_score"] is not None]
            if not scores:
                return "accuracy=n/a"
            return "accuracy={:.3f}".format(sum(scores) / len(scores))
        f1s = [r["det_f1"] for r in rows if r["det_f1"] is not None]
        if not f1s:
            return "f1=n/a"
        return "f1={:.3f}".format(sum(f1s) / len(f1s))


class ServerApp:
    """The serving node: load ONE model and answer each sample.

    The server never sees the ground truth and never scores; it only serves the
    model. That is the whole of the single-model serving path this example ports.
    """

    def __init__(self, config, model_search_dirs, logger=None):
        self.cfg = config
        self.params = DatasetVlmParams.model_validate(config.app.params)
        self._log = logger or (lambda m: None)
        self._model = None
        self._served = 0
        out_dir = os.path.join(config.outputs.root, config.experiment.name, config.app.plugin)
        self.csv_path = os.path.join(out_dir, "served_{}.csv".format(config_hash(config)))
        self._handle = None
        self._writer = None

    def setup(self, ctx=None) -> None:
        directory = os.path.dirname(self.csv_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._handle = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(
            self._handle, fieldnames=["sample_id", "task", "infer_ms", "answer_len"]
        )
        self._writer.writeheader()
        self._handle.flush()
        self._model = build_vlm(self.cfg, self.params, self._log)

    def warmup(self) -> None:
        """Run one blank frame through the model before the first real sample."""
        if self._model is None:
            return
        import numpy as np

        blank = np.zeros((16, 16, 3), dtype="uint8")
        try:
            if self.params.task == "vqa":
                self._model.answer(blank, "warmup?")
            else:
                self._model.detect(blank)
        except Exception as exc:  # pragma: no cover - a warmup failure is not fatal
            self._log("[dataset_vlm] warmup skipped: {}".format(exc))

    def handle(self, payload: bytes, ctx=None) -> Optional[bytes]:
        message = unpack(payload)
        image = decode_frame(message["frame"])
        started = time.monotonic()
        reply = {"sample_id": message.get("sample_id")}
        answer_len = 0
        if message.get("task") == "vqa":
            answer = self._model.answer(image, message.get("question", ""))
            reply["answer"] = answer
            answer_len = len(answer)
        else:
            boxes = self._model.detect(image)
            reply["boxes"] = boxes
            answer_len = len(boxes)
        infer_ms = (time.monotonic() - started) * 1000.0
        reply["infer_ms"] = infer_ms
        self._served += 1
        self._writer.writerow(
            {
                "sample_id": message.get("sample_id"),
                "task": message.get("task"),
                "infer_ms": _rounded(infer_ms),
                "answer_len": answer_len,
            }
        )
        self._handle.flush()
        return pack(reply)

    def stop_after_reply(self) -> bool:
        return False

    def teardown(self, ctx=None) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._log(
            "[dataset_vlm] served {} sample(s), csv={}".format(self._served, self.csv_path)
        )


def _rounded(value, digits: int = 3):
    """Round a measurement for the CSV, tolerating ``None``."""
    return None if value is None else round(float(value), digits)


@register_app
class DatasetVlmApp(AppPlugin):
    """Serve one VLM over a dataset and score it: the paper's Dataset Mode."""

    name = "dataset_vlm"
    params_model = DatasetVlmParams

    @staticmethod
    def client_success(client_app):
        stats = client_app.message_stats()
        if stats is None or stats.sent < 1:
            return False, "client sent no samples"
        if stats.received < 1:
            return False, "client sent {} samples and got no answer back".format(stats.sent)
        return True, "sent={} answered={}".format(stats.sent, stats.received)

    @staticmethod
    def build_server(config, model_search_dirs, logger=None) -> ServerApp:
        return ServerApp(config, model_search_dirs, logger=logger)

    @staticmethod
    def build_client(config, run_id, rep, csv_path, logger=None, node_name=None) -> ClientApp:
        return ClientApp(config, run_id, rep, csv_path, logger=logger, node_name=node_name)
