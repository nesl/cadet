"""The distributed-sensing app: where object detection should run.

What this is for:
    N sensor nodes stream frames over a contested access network into one
    processing node. The processing node is either the edge gateway the sensors
    associate with, or a cloud node reached through that gateway over a composed
    two-hop path. The one thing the example varies is where the detector runs:
    on each sensor, on the gateway, or in the cloud. The fan-in, the link
    models and the source are held fixed, so the measured difference belongs to
    the placement.

What it contains:
    - :class:`SensingParams`: the placement, the detector kind, the per-node-class
      detector cost and the frame encoding quality.
    - :class:`StubDetector`, :class:`YoloDetector` and :func:`build_detector`.
    - :class:`SensorApp`: the sensor side. Capture, optionally detect locally,
      transmit, and log per-event latency and bytes.
    - :class:`InferenceApp`: the processing side. Decode, optionally detect,
      log age of information per arrival, and reply.
    - :class:`SensingApp`: the plugin (``--app-module examples.distributed_sensing.app``).

How it fits into CADET:
    Runs on the generic runtimes: one ``cadet client`` per sensor node and one
    ``cadet serve`` for the processing node. The server runtime accepts every
    sensor on its own connection with its own delay line, which is the fan-in
    this example needs. Nothing in ``cadet/`` imports this module.

How to configure it:
    ``app.params.placement`` names the node class the detector runs on and
    ``app.params.detect_ms`` gives that class its compute cost. The configs in
    this directory set both; the README states the matrix they cover.
"""

import csv
import os
import time
import zlib
from queue import Empty, Queue
from threading import Thread
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Literal

from cadet.config.loader import config_hash
from cadet.core.apps import AppPlugin, register_app
from cadet.core.codec import decode as wire_decode
from cadet.core.codec import encode as wire_encode
from cadet.metrics.link_report import MessageStats

# How long a sensor keeps draining replies after its source is exhausted. Long
# enough to cover a slow path's in-flight replies, short enough that a run over
# a lossy link still terminates.
DRAIN_TIMEOUT_S = 5.0

# Stub detector shape. The box count comes from the frame's bytes, so it is
# reproducible for a given frame and carries no claim about detector accuracy;
# the geometry is fixed so the detections payload has a stated size per box.
STUB_MAX_DETECTIONS = 6
STUB_BOX_PX = 32.0
STUB_SCORE = 0.5

# A frame goes on the wire JPEG-encoded. Sending the raw array would make the
# bytes-per-link axis a property of the array dtype instead of a property of
# what a camera sends.
FRAME_FORMAT = ".jpg"

# Frame geometry used to warm the detector up when the source declares no
# camera, matching the synthetic source's default aspect.
WARMUP_HW = (48, 64)

# Columns of the per-sensor event log and of the processing node's arrival log.
SENSOR_COLUMNS = [
    "sensor",
    "seq",
    "capture_time",
    "send_time",
    "bytes_sent",
    "sensor_detect_ms",
    "remote_detect_ms",
    "detections",
    "aoi_ms",
    "reply_time",
    "e2e_latency_ms",
    "round_trip_ms",
]
ARRIVAL_COLUMNS = [
    "sensor",
    "seq",
    "arrival_time",
    "aoi_ms",
    "bytes_in",
    "detections",
    "detect_ms",
    "detected_here",
]


def pack(message: Dict) -> bytes:
    """Encode one wire message (the encoding the flat examples all use)."""
    return wire_encode(message)


def unpack(payload: bytes) -> Dict:
    """Decode one wire message produced by :func:`pack`."""
    return wire_decode(payload)


def sensor_id(node_name=None) -> str:
    """Return this sensor's identifier: its node name, or a fallback.

    The node name is what this should always be, and it is what the runtime now
    passes. The fallback exists for a caller that passes none and is deliberately
    **not** the process id alone, which is what this used to be: a pid is
    distinct between host processes and *identical* between containers, because
    every node container's pid namespace starts at 1. Five containerised sensors
    therefore all called themselves ``s00001``, wrote to one file, and the run
    reported one sensor instead of five - silently, because each of the five
    wrote a well-formed log. The host name is in the fallback for that case: in a
    container it is the container, which is per node.
    """
    if node_name:
        return str(node_name)
    import socket

    return "s{}-{:05d}".format(socket.gethostname(), os.getpid() % 100000)


def warmup_payload(config, quality: int) -> bytes:
    """Return one encoded blank frame, sized as this config's source frames."""
    import numpy as np

    camera = getattr(config.source, "camera", None)
    if camera is not None:
        height, width = camera.height, camera.width
    else:
        height = getattr(config.source, "height", None) or WARMUP_HW[0]
        width = getattr(config.source, "width", None) or WARMUP_HW[1]
    return encode_frame(np.zeros((height, width, 3), dtype="uint8"), quality)


def encode_frame(image, quality: int) -> bytes:
    """JPEG-encode one RGB frame at ``quality``."""
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


class SensingParams(BaseModel):
    """Parameters of the distributed sensing app."""

    model_config = ConfigDict(extra="forbid")
    plugin: Literal["distributed_sensing"] = "distributed_sensing"
    # The node class the detector runs on. "sensor" detects before transmitting,
    # so only the boxes cross the access network. "gateway" and "cloud" both
    # detect on the receiving node and differ in the path the frames took.
    placement: Literal["sensor", "gateway", "cloud"] = "gateway"
    detector: Literal["stub", "yolo"] = "stub"
    # Detector compute cost per node class, in milliseconds. One mapping rather
    # than one value, so the placement axis can be swept on its own: a sweep
    # sets each dotted path independently, so a cost that had to move with the
    # placement would produce combinations pairing a cloud cost with a sensor.
    detect_ms: Dict[str, float] = Field(
        default_factory=lambda: {"sensor": 120.0, "gateway": 40.0, "cloud": 12.0}
    )
    jpeg_quality: int = Field(default=80, ge=1, le=100)

    @model_validator(mode="after")
    def _cost_declared_for_placement(self) -> "SensingParams":
        if self.placement not in self.detect_ms:
            raise ValueError(
                "detect_ms declares no cost for placement {!r}; it has {}".format(
                    self.placement, sorted(self.detect_ms)
                )
            )
        return self

    @property
    def detector_cost_s(self) -> float:
        """The detector's compute cost on the node class it is placed on."""
        return float(self.detect_ms[self.placement]) / 1000.0

    @property
    def detects_on_sensor(self) -> bool:
        """True when each sensor runs the detector before transmitting."""
        return self.placement == "sensor"


class StubDetector:
    """Stand-in detector with a configured compute cost.

    The box count is derived from the frame's bytes, so a given frame always
    yields the same count. That makes the detection rate reproducible; it says
    nothing about how a real detector would score the frame.
    """

    def __init__(self, cost_s: float):
        self._cost_s = cost_s

    def detect(self, image) -> List[Dict[str, float]]:
        """Return the boxes for one frame, after paying the configured cost."""
        if self._cost_s > 0:
            time.sleep(self._cost_s)
        import numpy as np

        digest = zlib.crc32(np.ascontiguousarray(image).tobytes())
        height, width = image.shape[0], image.shape[1]
        span_x = max(1.0, width - STUB_BOX_PX)
        span_y = max(1.0, height - STUB_BOX_PX)
        boxes = []
        for index in range(digest % (STUB_MAX_DETECTIONS + 1)):
            seed = digest >> (3 * index)
            boxes.append(
                {
                    "x": float(seed % int(span_x)),
                    "y": float((seed // 7) % int(span_y)),
                    "w": STUB_BOX_PX,
                    "h": STUB_BOX_PX,
                    "score": STUB_SCORE,
                }
            )
        return boxes


class YoloDetector:
    """Real detector resolved through the model registry."""

    def __init__(self, path, device):
        from ultralytics import YOLO

        self._model = YOLO(path)
        self._device = device

    def detect(self, image) -> List[Dict[str, float]]:
        """Return the boxes the model found in one frame."""
        result = self._model.predict(image, verbose=False, device=self._device)[0]
        boxes = []
        for box, score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            boxes.append(
                {
                    "x": float(box[0]),
                    "y": float(box[1]),
                    "w": float(box[2] - box[0]),
                    "h": float(box[3] - box[1]),
                    "score": float(score),
                }
            )
        return boxes


def build_detector(config, params: SensingParams, model_search_dirs, log=None):
    """Return the detector for this run, from the stub or the model registry.

    The stub is what every config in this directory uses: it needs no weights
    and its cost is the configured one, which is the quantity the placement axis
    varies. ``detector: yolo`` resolves ``models.model`` through
    :class:`cadet.core.model_registry.ModelRegistry` for the device the config
    gives the model's node.
    """
    log = log or (lambda m: None)
    if params.detector == "stub":
        log(
            "[sensing] stub detector on the {} ({:.1f} ms per frame)".format(
                params.placement, params.detector_cost_s * 1000.0
            )
        )
        return StubDetector(params.detector_cost_s)
    from cadet.core.model_registry import ModelRegistry, device_for_models, torch_device_for

    device = torch_device_for(config)
    registry = ModelRegistry.from_config(config.models, device_for_models(config))
    resolved = registry.resolve(config.models.model, list(model_search_dirs or []))
    if resolved.path is None:
        raise RuntimeError("model artifact not found: {}".format(resolved.artifact))
    log("[sensing] detector {} on {}".format(resolved.path, device))
    return YoloDetector(resolved.path, device)


class SensorApp:
    """One sensor node: capture, optionally detect, transmit, log the round trip."""

    def __init__(self, config, run_id: str, rep: int, csv_path: str, logger=None, node_name=None):
        self.cfg = config
        self.params = SensingParams.model_validate(config.app.params)
        self._log = logger or (lambda m: None)
        self.sensor = sensor_id(node_name)
        # One event log per sensor. Every sensor of a run is handed the same path
        # by the runtime, so a per-node name is what stops N sensors from
        # overwriting each other. `events_<node>.csv` follows the framework's own
        # naming for a per-node file (client_outcome_<node>.json,
        # link_report_<node>.json) rather than inventing a second rule.
        self.csv_path = os.path.join(
            os.path.dirname(csv_path), "events_{}.csv".format(self.sensor)
        )
        self.transport = None
        self._detector = None
        self._seq = 0
        self._current = None
        self._rows: Dict[int, Dict] = {}
        self._sent = 0
        self._answered = 0
        self._source_done = False
        self._server_closed = False
        self._replies: "Queue" = Queue()
        self._drain_started = None

    # -- runtime hooks -------------------------------------------------------
    def setup(self, ctx) -> None:
        """Attach the transport, build the local detector, start the listener."""
        self.transport = ctx.transport
        if self.params.detects_on_sensor:
            self._detector = build_detector(self.cfg, self.params, [], self._log)
            # The client runtime has no warm-up hook, and the first call into the
            # codec and the detector costs more than the ones after it, so the
            # first frame would otherwise carry that cost as if it were latency.
            self._detector.detect(
                decode_frame(warmup_payload(self.cfg, self.params.jpeg_quality))
            )
        else:
            encode_frame(
                decode_frame(warmup_payload(self.cfg, self.params.jpeg_quality)),
                self.params.jpeg_quality,
            )
        Thread(target=self._listen, daemon=True).start()

    def _listen(self) -> None:
        """Read replies off the socket and stamp their arrival here.

        Stamping arrival in this thread rather than where the main loop polls
        keeps the loop's tick interval out of every measurement.
        """
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
        """Return one arrived reply, or ``None``."""
        try:
            return self._replies.get_nowait()
        except Empty:
            return None

    def translate_action(self, frame, reply, ctx) -> None:
        """Note the source running dry, and fold a reply into its event row."""
        if frame is None:
            self._source_done = True
        if reply is None:
            return
        reply, arrived_at = reply
        row = self._rows.get(reply.get("seq"))
        if row is None:
            return
        self._answered += 1
        row["remote_detect_ms"] = _rounded(reply.get("detect_ms"))
        row["detections"] = reply.get("detections")
        row["aoi_ms"] = _rounded(reply.get("aoi_ms"))
        row["reply_time"] = round(arrived_at, 6)
        row["e2e_latency_ms"] = _rounded((arrived_at - row["capture_time"]) * 1000.0)
        row["round_trip_ms"] = _rounded((arrived_at - row["send_time"]) * 1000.0)

    def should_transmit(self, ctx) -> bool:
        """Transmit while the source still has frames."""
        return not self._source_done

    def process_frame(self, frame, ctx) -> bytes:
        """Build this frame's payload: its boxes if detecting here, else the frame."""
        seq = self._seq
        self._seq += 1
        capture_time = float(frame.meta.get("capture_time") or time.time())
        message = {"sensor": self.sensor, "seq": seq, "capture_t": capture_time}
        detect_ms = None
        detections = None
        if self._detector is not None:
            started = time.monotonic()
            boxes = self._detector.detect(frame.image)
            detect_ms = (time.monotonic() - started) * 1000.0
            detections = len(boxes)
            message["detections"] = boxes
            message["detect_ms"] = detect_ms
        else:
            message["frame"] = encode_frame(frame.image, self.params.jpeg_quality)
        payload = pack(message)
        self._rows[seq] = {
            "sensor": self.sensor,
            "seq": seq,
            "capture_time": round(capture_time, 6),
            "send_time": None,
            "bytes_sent": len(payload),
            "sensor_detect_ms": _rounded(detect_ms),
            "remote_detect_ms": None,
            "detections": detections,
            "aoi_ms": None,
            "reply_time": None,
            "e2e_latency_ms": None,
            "round_trip_ms": None,
        }
        self._current = seq
        return payload

    def on_frame_sent(self, frame, ctx) -> None:
        """Stamp when the payload went to the transport."""
        row = self._rows.get(self._current)
        if row is not None:
            row["send_time"] = time.time()
            self._sent += 1

    def collect_metrics(self, frame, ctx) -> None:
        """Nothing extra per tick: the event rows carry the measurements."""

    def is_done(self, ctx) -> bool:
        """Finish once every sent frame is answered, or the drain window closes."""
        if self._server_closed:
            return True
        if not self._source_done or not self._sent:
            return False
        if self._answered >= self._sent:
            return True
        # Replies still outstanding. They are either in flight on a slow path or
        # were dropped by a link with loss configured, which is a legitimate
        # outcome, so drain for a bounded window and count the rest as lost.
        if self._drain_started is None:
            self._drain_started = time.time()
            return False
        return (time.time() - self._drain_started) >= DRAIN_TIMEOUT_S

    def message_stats(self):
        """Return what this sensor observed, for the run's link report."""
        latencies = [
            row["round_trip_ms"] for row in self._rows.values() if row["round_trip_ms"] is not None
        ]
        return MessageStats(sent=self._sent, received=self._answered, latencies_ms=latencies)

    def finish(self, ctx) -> None:
        """Write this sensor's event log and report what it carried."""
        rows = [self._rows[seq] for seq in sorted(self._rows) if self._rows[seq]["send_time"]]
        for row in rows:
            row["send_time"] = round(row["send_time"], 6)
        with open(self.csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SENSOR_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        uplink_bytes = sum(row["bytes_sent"] for row in rows)
        self._log(
            "[sensing] sensor {} placement={} sent={} answered={} uplink_bytes={} csv={}".format(
                self.sensor,
                self.params.placement,
                self._sent,
                self._answered,
                uplink_bytes,
                self.csv_path,
            )
        )


class InferenceApp:
    """The processing node: the gateway, or the cloud node behind it."""

    def __init__(self, config, model_search_dirs, logger=None):
        self.cfg = config
        self.params = SensingParams.model_validate(config.app.params)
        self._log = logger or (lambda m: None)
        self._search_dirs = model_search_dirs
        self._detector = None
        self._arrivals = 0
        self._bytes_in = 0
        self._sensors = set()
        self._handle = None
        self._writer = None
        # Named by the config hash and placed beside the server's own per-request
        # metrics, which is the convention the server runtime already follows for
        # a file it writes without knowing the run id.
        out_dir = os.path.join(
            config.outputs.root, config.experiment.name, config.app.plugin
        )
        self.csv_path = os.path.join(out_dir, "arrivals_{}.csv".format(config_hash(config)))

    def setup(self, ctx=None) -> None:
        """Open the arrival log and build the detector, if this node runs it."""
        self._open_log()
        if self.params.detects_on_sensor:
            self._log("[sensing] aggregating boxes detected on the sensors")
            return
        self._detector = build_detector(self.cfg, self.params, self._search_dirs, self._log)

    def _open_log(self) -> None:
        """Open the arrival log and write its header.

        Each row is written and flushed as its frame arrives, rather than
        accumulated and written at teardown. The orchestrator stops a server
        role as soon as the clients have exited, so a log written at teardown is
        the log of a process that was already signalled.
        """
        directory = os.path.dirname(self.csv_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._handle = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=ARRIVAL_COLUMNS)
        self._writer.writeheader()
        self._handle.flush()

    def warmup(self) -> None:
        """Run one frame through the whole receive path before the first arrival.

        The decode is part of it, not only the detector: the first call into the
        image codec costs more than the ones after it, and charging that to the
        first sensor's first frame would put it in the measured latency.
        """
        if self._detector is None:
            return
        self._detector.detect(decode_frame(warmup_payload(self.cfg, self.params.jpeg_quality)))

    def handle(self, payload: bytes, ctx=None) -> Optional[bytes]:
        """Record one arrival, detect if this is the detector's node, and reply."""
        arrival_time = time.time()
        message = unpack(payload)
        aoi_ms = (arrival_time - float(message["capture_t"])) * 1000.0
        if self._detector is not None:
            started = time.monotonic()
            boxes = self._detector.detect(decode_frame(message["frame"]))
            detect_ms = (time.monotonic() - started) * 1000.0
            detected_here = True
        else:
            boxes = message.get("detections") or []
            detect_ms = message.get("detect_ms")
            detected_here = False
        self._writer.writerow(
            {
                "sensor": message.get("sensor"),
                "seq": message.get("seq"),
                "arrival_time": round(arrival_time, 6),
                "aoi_ms": _rounded(aoi_ms),
                "bytes_in": len(payload),
                "detections": len(boxes),
                "detect_ms": _rounded(detect_ms),
                "detected_here": detected_here,
            }
        )
        self._handle.flush()
        self._arrivals += 1
        self._bytes_in += len(payload)
        self._sensors.add(message.get("sensor"))
        return pack(
            {
                "sensor": message.get("sensor"),
                "seq": message.get("seq"),
                "detections": len(boxes),
                "detect_ms": detect_ms,
                "aoi_ms": aoi_ms,
            }
        )

    def stop_after_reply(self) -> bool:
        """Keep serving: every sensor is answered per frame until it disconnects."""
        return False

    def teardown(self, ctx=None) -> None:
        """Close the arrival log, which is where age of information is measured."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._log(
            "[sensing] {} arrivals from {} sensor(s), {} bytes in, csv={}".format(
                self._arrivals, len(self._sensors), self._bytes_in, self.csv_path
            )
        )


def _rounded(value, digits: int = 3):
    """Round a measurement for the CSV, tolerating ``None``."""
    return None if value is None else round(float(value), digits)


@register_app
class SensingApp(AppPlugin):
    """Distributed sensing under contested networks: detector placement."""

    name = "distributed_sensing"
    params_model = SensingParams

    @staticmethod
    def client_success(client_app):
        """A sensor succeeded if it transmitted and got at least one result back."""
        stats = client_app.message_stats()
        if stats is None or stats.sent < 1:
            return False, "sensor transmitted no frames"
        if stats.received < 1:
            return False, "sensor sent {} frames and got no result back".format(stats.sent)
        return True, "sent={} answered={}".format(stats.sent, stats.received)

    @staticmethod
    def build_server(config, model_search_dirs, logger=None) -> InferenceApp:
        return InferenceApp(config, model_search_dirs, logger=logger)

    @staticmethod
    def build_client(config, run_id, rep, csv_path, logger=None, node_name=None) -> SensorApp:
        return SensorApp(config, run_id, rep, csv_path, logger=logger, node_name=node_name)
