"""Reference app (generic, non-AV) — a minimal CADET example.

What this is for:
    The smallest useful app that exercises the framework without any AV concepts:
    the client sends frames from any source, the server runs a model from the
    registry and replies to EVERY frame, and the client logs each reply with its
    round-trip latency. Used by the end-to-end framework tests and the non-AV
    demo (replay a folder of images through YOLO).

What it contains:
    - :class:`ReferenceParams`: ``model_kind`` (``stub`` for tests, ``yolo`` for
      real inference).
    - :class:`RefServerApp`: loads the model, replies per frame with a result +
      server processing time.
    - :class:`RefClientApp`: sends frames, logs replies + latency to a CSV.
    - :class:`ReferenceApp`: the plugin (``--app-module examples.reference.app``).

How it fits into CADET:
    Runs on the generic runtimes (``cadet serve``/``cadet client``) over any
    source (``synthetic`` / ``replay`` / ...). The server replies to every frame,
    so per-frame latency is real (unlike the braking app's single brake reply).
"""

import csv
import time
from queue import Empty, Queue
from threading import Thread
from typing import List, Optional

from pydantic import BaseModel, ConfigDict
from typing_extensions import Literal

from cadet.core.apps import AppPlugin, register_app
from cadet.core.codec import decode as wire_decode
from cadet.core.codec import encode as wire_encode
from cadet.core.model_registry import ModelRegistry, device_for_models, torch_device_for

# Tiny (height, width) warm-up image used when the source declares no camera.
_WARMUP_FALLBACK_HW = (48, 64)


class ReferenceParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plugin: Literal["reference"] = "reference"
    model_kind: Literal["stub", "yolo"] = "stub"
    # Optional work-time for the stub, so load experiments can create contention.
    process_ms: float = 0.0


class _StubModel:
    """Deterministic non-AV model: summarises a frame cheaply."""

    def __init__(self, process_ms: float = 0.0):
        self._work_s = process_ms / 1000.0

    def predict(self, image):
        import numpy as np

        if self._work_s > 0:
            time.sleep(self._work_s)
        arr = np.asarray(image)
        return {"mean": float(arr.mean()), "shape": list(arr.shape)}


class _YoloModel:
    """Optional real model: returns the number of detections."""

    def __init__(self, path, device):
        from ultralytics import YOLO

        self._m = YOLO(path)
        self._device = device

    def predict(self, image):
        det = self._m.predict(image, verbose=False, device=self._device)
        return {"detections": int(len(det[0].boxes))}


class RefServerApp:
    """Replies to every frame with the model result + processing time."""

    def __init__(self, config, model_search_dirs: List[str], logger=None):
        self.cfg = config
        self.params = ReferenceParams.model_validate(config.app.params)
        self.search_dirs = model_search_dirs
        self._log = logger or (lambda m: None)
        self._model = None

    def setup(self, ctx=None) -> None:
        if self.params.model_kind == "stub":
            self._model = _StubModel(self.params.process_ms)
            self._log("[server] using stub model (process_ms={})".format(self.params.process_ms))
            return
        device = torch_device_for(self.cfg)
        registry = ModelRegistry.from_config(self.cfg.models, device_for_models(self.cfg))
        res = registry.resolve(self.cfg.models.model, self.search_dirs)
        if res.path is None:
            raise RuntimeError("model artifact not found: {}".format(res.artifact))
        self._model = _YoloModel(res.path, device)
        self._log("[server] using yolo model {} on {}".format(res.path, device))

    def warmup(self) -> None:
        import numpy as np

        cam = getattr(self.cfg.source, "camera", None)
        h, w = (cam.height, cam.width) if cam is not None else _WARMUP_FALLBACK_HW
        self._model.predict(np.zeros((h, w, 3), dtype=np.uint8))

    def handle(self, payload: bytes, ctx=None) -> Optional[bytes]:
        msg = wire_decode(payload)
        t0 = time.time()
        result = self._model.predict(msg["image"])
        reply = {
            "frame_id": msg["frame_id"],
            "result": result,
            "server_processing_time": time.time() - t0,
        }
        return wire_encode(reply)

    def stop_after_reply(self) -> bool:
        return False  # keep serving every frame until the client disconnects

    def teardown(self, ctx=None) -> None:
        pass


class RefClientApp:
    """Sends every frame, logs each reply with its round-trip latency."""

    def __init__(self, config, run_id: str, rep: int, csv_path: str, logger=None):
        self.cfg = config
        self._log = logger or (lambda m: None)
        self.csv_path = csv_path
        self.transport = None
        self._fid = 0
        self._sent = {}  # frame_id -> wall send time
        self._rows = []  # per-reply records
        self._source_done = False
        self._server_closed = False
        self._replies: "Queue" = Queue()
        # When the source ran dry, so the drain window can be bounded.
        self._drain_started = None

    def setup(self, ctx) -> None:
        self.transport = ctx.transport
        Thread(target=self._listen, daemon=True).start()

    def _listen(self) -> None:
        while True:
            try:
                raw = self.transport.recv()
            except Exception:
                self._server_closed = True
                break
            if raw is None:
                self._server_closed = True  # server closed the connection
                break
            try:
                reply = wire_decode(raw)
            except Exception:
                continue
            # Stamp arrival here, in the listener thread. Stamping it where the
            # main loop polls would add that loop's tick interval to every
            # measurement, which is the difference between measuring the link and
            # measuring the polling rate.
            self._replies.put((reply, time.time()))

    def poll_reply(self, ctx):
        try:
            return self._replies.get_nowait()
        except Empty:
            return None

    def translate_action(self, frame, reply, ctx) -> None:
        if frame is None:
            self._source_done = True
        if reply is not None:
            reply, arrived_at = reply
            fid = reply["frame_id"]
            send_t = self._sent.get(fid)
            latency = (arrived_at - send_t) if send_t is not None else None
            self._rows.append(
                {
                    "frame_id": fid,
                    "latency_s": round(latency, 6) if latency is not None else None,
                    "server_processing_time": round(reply["server_processing_time"], 6),
                    "result": reply["result"],
                }
            )

    def should_transmit(self, ctx) -> bool:
        return not self._source_done

    def process_frame(self, frame, ctx) -> bytes:
        fid = self._fid
        self._fid += 1
        self._cur_fid = fid
        msg = {"frame_id": fid, "image": frame.image}
        return wire_encode(msg)

    def on_frame_sent(self, frame, ctx) -> None:
        self._sent[self._cur_fid] = time.time()

    def collect_metrics(self, frame, ctx) -> None:
        pass

    def is_done(self, ctx) -> bool:
        # Done when the server closed (clean exit even if incomplete), or once the
        # source is exhausted and every sent frame got a reply.
        if self._server_closed:
            return True
        if not self._source_done or not self._sent:
            return False
        if len(self._rows) >= len(self._sent):
            return True
        # Replies still outstanding. They may simply be in flight on a delayed
        # link, or they may have been dropped by a link with loss configured —
        # which is a legitimate experiment, so waiting for them forever would
        # turn a measurable outcome into a hang. Drain for a bounded window and
        # then finish, counting whatever did not arrive as lost.
        if self._drain_started is None:
            self._drain_started = time.time()
            return False
        return (time.time() - self._drain_started) >= DRAIN_TIMEOUT_S

    def message_stats(self):
        """Return what this client observed, for the run's link report.

        Latencies are per-message round trips in milliseconds, taken at reply
        arrival. The framework pairs them with the configured link parameters.
        """
        from cadet.metrics.link_report import MessageStats

        return MessageStats(
            sent=len(self._sent),
            received=len(self._rows),
            latencies_ms=[
                r["latency_s"] * 1000.0 for r in self._rows if r["latency_s"] is not None
            ],
        )

    def finish(self, ctx) -> None:
        with open(self.csv_path, "w", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=["frame_id", "latency_s", "server_processing_time", "result"],
            )
            w.writeheader()
            for r in sorted(self._rows, key=lambda x: x["frame_id"]):
                w.writerow(r)
        sent, got = len(self._sent), len(self._rows)
        if self._server_closed and got < sent:
            self._log(
                "[client] ERROR: server closed mid-run; {}/{} replies received; csv={}".format(
                    got, sent, self.csv_path
                )
            )
        elif got < sent:
            # Expected whenever the link has loss configured; report the rate so
            # it can be compared against what was asked for.
            self._log(
                "[client] replies={}/{} ({:.2f}% unanswered after drain) csv={}".format(
                    got, sent, 100.0 * (sent - got) / sent, self.csv_path
                )
            )
        else:
            self._log("[client] replies={} csv={}".format(got, self.csv_path))


# How long the client keeps draining replies after its source is exhausted.
# Long enough to cover a slow link's in-flight replies, short enough that a
# run with loss still terminates promptly.
DRAIN_TIMEOUT_S = 5.0


@register_app
class ReferenceApp(AppPlugin):
    """Generic non-AV reference app."""

    name = "reference"
    params_model = ReferenceParams

    @staticmethod
    def client_success(client_app):
        """At least one reply. This app answers every message, so a run with no
        reply measured no round trip - which is the only thing it measures.
        """
        stats = client_app.message_stats()
        if stats is None or stats.sent < 1:
            return False, "client transmitted no messages"
        if stats.received < 1:
            return False, "client sent {} messages and received no reply".format(stats.sent)
        return True, "sent={} received={}".format(stats.sent, stats.received)

    @staticmethod
    def build_server(config, model_search_dirs, logger=None) -> RefServerApp:
        return RefServerApp(config, model_search_dirs, logger=logger)

    @staticmethod
    def build_client(
        config, run_id, rep, csv_path, logger=None, node_name=None
    ) -> RefClientApp:
        return RefClientApp(config, run_id, rep, csv_path, logger=logger)
