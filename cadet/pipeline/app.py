"""The built-in ``pipeline`` app: run a stage pipeline on the generic roles.

What this is for:
    Makes a ``pipeline.stages`` config run live under the orchestrator without a
    bespoke app. It is framework infrastructure, not an AV app: the orchestrator
    launches ``cadet serve`` for each compute node and ``cadet client`` for the
    origin, and this plugin maps the placed stages onto those roles, driving the
    boundary connections from :func:`cadet.pipeline.runtime.plan_chain`.

What it contains:
    - :class:`PipelineServerApp`: a compute node — runs its local stages on each
      received message, forwards to the downstream node if the chain continues, and
      replies upstream (reply-on-event: no reply when a stage drops the message).
    - :class:`PipelineClientApp`: the origin — runs the head stages on each source
      frame and sends them; on a reply, latches the decision and runs the tail
      stages each tick over ego telemetry, applying the resulting command to the
      source (duck-typed, so it needs no app-specific imports).
    - :class:`PipelineApp`: the plugin, registered as ``pipeline``.

How it fits into CADET:
    ``app.plugin: pipeline`` in the config; load the stage library with
    ``--stage-module`` (e.g. ``cadet.pipeline.stages.braking``). Boundaries use the
    public :class:`cadet.core.transport` API only. Metrics use the core
    :class:`cadet.metrics.MetricsLog`, so a pipeline run's CSV matches the flat app's.

How to configure it:
    ``pipeline.stages`` (placement) and ``nodes`` host/port drive everything;
    ``app.params.stop_after_reply`` (default true) matches the braking server that
    stops after it issues the single brake.
"""

import os
import time
from queue import Empty, Queue
from threading import Thread
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Literal

from cadet.core.apps import AppPlugin, register_app
from cadet.core.source import ego_removed
from cadet.core.transport import FramedSocketTransport
from cadet.pipeline.builder import build_pipeline
from cadet.pipeline.messages import Frame as FrameMessage
from cadet.pipeline.messages import Message
from cadet.pipeline.runtime import (
    deserialize_message,
    plan_chain,
    run_segment,
    serialize_message,
)
from cadet.pipeline.stage import StageContext

# Stopped-speed gate for recording the ego's stop, matching the flat braking app.
STOP_SPEED_MPS = 0.1
MPS_TO_MPH = 2.23694
DOWNSTREAM_CONNECT_RETRY_S = 30.0


def _server_node(config, node_name):
    """Resolve which server node these stages run on (by name, else by role)."""
    if node_name is not None:
        node = next((n for n in config.nodes if n.name == node_name), None)
        if node is not None:
            return node
    return next((n for n in config.nodes if n.role == "server"), None)


def _speed_mps(speed_str):
    """Parse a speed string like ``40mph`` / ``20kph`` into m/s (unit helper)."""
    if speed_str is None:
        return None
    if speed_str.endswith("kph"):
        return float(speed_str[:-3]) / 3.6
    if speed_str.endswith("mph"):
        return float(speed_str[:-3]) * 0.44704
    return None


class PipelineServerApp:
    """A compute node in the pipeline (conforms to the server-role app contract)."""

    def __init__(self, config, model_dirs, logger=None, node_name=None):
        self.cfg = config
        self.model_dirs = model_dirs
        self.log = logger or (lambda m: None)
        self._node_name = node_name
        self._local: List = []
        self._downstream: Optional[FramedSocketTransport] = None
        self._stop_after_reply = bool(config.app.params.get("stop_after_reply", True))

    def setup(self, ctx=None) -> None:
        plan = plan_chain(build_pipeline(self.cfg.pipeline.stages))
        node = _server_node(self.cfg, self._node_name)
        if node is None:
            raise RuntimeError("pipeline server: no server node in config")
        self._local = plan.local_stages(node.name)
        sctx = StageContext(
            config=self.cfg, node=node.name, model_dirs=self.model_dirs, logger=self.log
        )
        for ps in self._local:
            ps.stage.setup(sctx)
        self._wire_detector_labels()

        downstream = plan.downstream_of(node.name)
        if downstream is not None:
            dnode = next((n for n in self.cfg.nodes if n.name == downstream), None)
            if dnode is None:
                raise RuntimeError(
                    "pipeline server: unknown downstream node {!r}".format(downstream)
                )
            self.log(
                "[pipeline-server {}] connecting downstream to {}".format(node.name, downstream)
            )
            self._downstream = FramedSocketTransport.connect(
                dnode.host,
                dnode.port,
                retry_s=DOWNSTREAM_CONNECT_RETRY_S,
                protocol=self.cfg.transport.protocol,
                mode=self.cfg.transport.mode,
            )

    def _wire_detector_labels(self) -> None:
        """Hand a detector stage's class-name map to a decision stage (duck-typed)."""
        names = None
        for ps in self._local:
            if hasattr(ps.stage, "class_names"):
                names = ps.stage.class_names()
        if not names:
            return
        for ps in self._local:
            if hasattr(ps.stage, "set_class_names"):
                ps.stage.set_class_names(names)

    def warmup(self) -> None:
        for ps in self._local:
            if hasattr(ps.stage, "warmup"):
                ps.stage.warmup()

    def handle(self, payload: bytes, ctx=None) -> Optional[bytes]:
        out = run_segment(self._local, deserialize_message(payload))
        if out is None:
            return None  # reply-on-event: nothing to send this frame
        if self._downstream is not None:
            self._downstream.send(serialize_message(out))
            back = self._downstream.recv()
            out = deserialize_message(back) if back is not None else None
            if out is None:
                return None
        return serialize_message(out)

    def stop_after_reply(self) -> bool:
        return self._stop_after_reply

    def teardown(self, ctx=None) -> None:
        if self._downstream is not None:
            self._downstream.close()
        for ps in self._local:
            if hasattr(ps.stage, "teardown"):
                ps.stage.teardown()


def _as_xyz(location):
    """Return ``(x, y, z)`` floats for a simulator location, or None.

    A simulator's location object is a native handle that cannot be serialised,
    so anything crossing a node boundary carries the coordinates instead of the
    object.
    """
    if location is None:
        return None
    try:
        return (
            round(float(location.x), 2),
            round(float(location.y), 2),
            round(float(location.z), 2),
        )
    except (AttributeError, TypeError, ValueError):
        return None


def _load_client_extension(dotted, config, log):
    """Resolve a ``module:attr`` client extension and instantiate it with the config.

    Returns None when no extension is configured. The class is constructed with the
    resolved config; its hooks (setup/poll_intent/drive/keep_obstacle/is_done/
    summary) are duck-typed and all optional, so the pipeline client calls only the
    ones present.
    """
    if not dotted:
        return None
    import importlib

    mod_name, _, attr = dotted.partition(":")
    if not mod_name or not attr:
        raise ValueError("client_extension must be 'module:attr', got {!r}".format(dotted))
    cls = getattr(importlib.import_module(mod_name), attr)
    ext = cls(config)
    log("[pipeline-client] client extension: {}".format(dotted))
    return ext


class PipelineClientApp:
    """The pipeline origin (conforms to the client-role app contract).

    Runs the head stages on each source frame and sends them; a listener thread
    latches the first reply (the decision); each tick it assembles ego telemetry
    into a message, runs the tail stages, and applies the resulting command to the
    source. Actuation is duck-typed: it calls ``source`` methods only if present, so
    the same client drives a CARLA braking run or a no-actuation pipeline.
    """

    def __init__(self, config, run_id, rep, csv_path, logger=None):
        from cadet.metrics import MetricsLog

        self.cfg = config
        self.log = logger or (lambda m: None)
        plan = plan_chain(build_pipeline(config.pipeline.stages))
        self._head = plan.head_stages()
        self._tail = plan.tail_stages()
        self._control = next((ps.stage for ps in self._tail if hasattr(ps.stage, "process")), None)
        scenario = getattr(config.source, "scenario", None)
        self._target_mps = _speed_mps(getattr(scenario, "ego_speed", None))
        self._metrics = MetricsLog(csv_path, run_id, rep)
        self._viz_enabled = bool(config.app.params.get("client_visualizer", True))
        self._viz_save_frames = bool(config.app.params.get("client_save_frames", True))
        self._viz_dir = os.path.join(os.path.dirname(csv_path), "client_frames")
        self._viz = None
        self._orig_loop = bool(config.app.params.get("original_client_loop", True))
        self._forward_meta_keys = list(config.app.params.get("forward_meta_keys", []))
        self._replies: "Queue" = Queue()
        self._latched = False
        self._brake_recorded = False
        self._transmit = True
        self._done = False
        self._source_done = False
        # Set when the simulator says the ego this client was driving is gone.
        # Kept so the run's verdict can say why it ended rather than reporting
        # a run that stopped early as one that finished.
        self._ego_gone = False
        self._ego_checked_at = 0.0
        self._control_rate = config.app.params.get("control_rate", "per_loop")
        self._last_ctrl_frame = None  # last simulation tick a command was issued on (per_tick)
        self._start = None
        self._fid = 0
        self._cur_fid = None
        self._last_speed_mph = 0.0
        self._last_sim_time = 0.0
        self._csv_path = csv_path
        # Optional client-side extension (e.g. V2V leader drive + intent); see
        # PipelineAppParams.client_extension. Duck-typed, all hooks optional.
        self._ext = _load_client_extension(
            config.app.params.get("client_extension"), config, self.log
        )

    def setup(self, ctx) -> None:
        self.source = ctx.source
        self.transport = ctx.transport
        # Not here: setup() runs before the client waits for the ego, so an
        # origin set now would fold that wait into every reported time. Taken on
        # the first control tick instead, as in the flat client.
        sctx = StageContext(config=self.cfg, node="client", logger=self.log)
        for ps in self._head + self._tail:
            ps.stage.setup(sctx)
        cam = getattr(self.cfg.source, "camera", None)
        if self._viz_enabled and cam is not None and hasattr(self.source, "apply_control"):
            from cadet.roles.client_visualizer import ClientVisualizer

            self._viz = ClientVisualizer(
                cam.width, cam.height, self._viz_dir,
                save_frames=self._viz_save_frames, logger=self.log)
        if self._ext is not None and hasattr(self._ext, "setup"):
            self._ext.setup(self.source)
        Thread(target=self._listen, daemon=True).start()

    def _listen(self) -> None:
        while True:
            try:
                raw = self.transport.recv()
            except Exception:
                break
            if raw is None:
                break
            msg = deserialize_message(raw)
            if msg is not None:
                self._replies.put(msg)

    def poll_reply(self, ctx):
        # Let a client extension deliver an in-process reply first (V2V intent),
        # exactly as the flat client's poll_reply does. Checked at the top of the
        # loop with the wall clock, as the original did.
        if self._ext is not None and hasattr(self._ext, "poll_intent") and self._start is not None:
            intent = self._ext.poll_intent(time.time() - self._start)
            if intent is not None:
                return intent
        try:
            return self._replies.get_nowait()
        except Empty:
            return None

    def process_frame(self, frame, ctx) -> bytes:
        self._cur_fid = self._fid
        cloc = frame.meta.get("capture_loc")
        capture_time = frame.meta.get("capture_time")
        self._metrics.new_frame(
            self._fid,
            capture_time,
            (round(cloc.x, 2), round(cloc.y, 2), round(cloc.z, 2)) if cloc is not None else None,
            self._last_speed_mph,
        )
        if self._viz is not None:
            # A PNG per frame, as the original client wrote; off unless a
            # visualizer is configured.
            self._viz.save_frame(frame.image, self._fid)
        if self._orig_loop and hasattr(self.source, "poll_control"):
            self.source.poll_control()  # the original read get_control per sent frame
        # The source hands back a cadet.core.source.Frame: an image and metadata
        # and nothing else. A pipeline stage needs a pipeline Message - the same
        # payload plus the frame id, the capture times and the stage-stamp trail
        # that the per-stage latency reconstruction is built from. Wrapping it
        # here is where that identity is assigned.
        #
        # The metadata is rebuilt rather than forwarded. The source's own holds
        # a simulator handle (a carla.Location), which cannot be serialised, and
        # this message crosses a node boundary; only plain values may travel.
        meta = {
            "capture_time": capture_time,
            "capture_loc": _as_xyz(cloc),
        }
        # Forward any ground-truth keys the app asked for (open-loop decision
        # scoring reads these on the server); default is none, so existing apps
        # send exactly what they did before.
        for key in self._forward_meta_keys:
            if key in frame.meta:
                meta[key] = frame.meta[key]
        head_in = FrameMessage(
            image=frame.image,
            meta=meta,
            frame_id=self._fid,
            sim_time=capture_time,
            wall_time=time.time(),
        )
        head_out = run_segment(self._head, head_in)  # camera passthrough
        return serialize_message(head_out)

    def on_frame_sent(self, frame, ctx) -> None:
        sloc = self.source.vehicle_location() if hasattr(self.source, "vehicle_location") else None
        self._metrics.mark_sent(
            self._cur_fid,
            time.time() - self._start,
            (round(sloc.x, 2), round(sloc.y, 2), round(sloc.z, 2)) if sloc is not None else None,
        )
        self._fid += 1

    def translate_action(self, frame, reply, ctx) -> None:
        # A None frame means "nothing available right now", which for a live
        # simulator source is the normal state between sensor callbacks and is
        # always true on the first tick. Only an exhausted source is done; see
        # Source.exhausted.
        if self._start is None:
            self._start = time.time()  # first control tick; see setup()
        if frame is None and ctx.source.exhausted():
            self._source_done = True
        # Control rate (opt-in; the default, per_loop, is what the original did); identical to
        # the flat client's rule so the two stay in parity. per_loop: a command every
        # loop iteration on wall time (original). per_tick: one per simulation tick,
        # gated on the world snapshot, driven off the snapshot clock.
        now = time.time()
        ctrl_t = None
        if self._control_rate == "per_tick":
            snap = self.source.snapshot() if hasattr(self.source, "snapshot") else None
            if snap is not None:
                tick_frame, sim_elapsed, _dt = snap
                if tick_frame == self._last_ctrl_frame:
                    return  # same simulation tick -> no new command
                self._last_ctrl_frame = tick_frame
                ctrl_t = sim_elapsed
        # per_loop, or a non-simulation source: fall back to wall time.
        ctrl_t = now if ctrl_t is None else ctrl_t
        sim_time = now - self._start
        self._last_sim_time = sim_time
        has_ego = hasattr(self.source, "vehicle_speed_mps")
        speed_mps = self.source.vehicle_speed_mps() if has_ego else 0.0
        self._last_speed_mph = speed_mps * MPS_TO_MPH

        if reply is not None and not self._latched:
            self._latched = True
            self._transmit = False
            # An extension may keep the obstacle (V2V leaves the lead vehicle alive).
            keep = self._ext is not None and getattr(self._ext, "keep_obstacle", lambda: False)()
            if hasattr(self.source, "destroy_obstacle") and not keep:
                self.source.destroy_obstacle()
            loc = self.source.vehicle_location() if has_ego else None
            brake_loc = (
                (round(loc.x, 2), round(loc.y, 2), round(loc.z, 2)) if loc is not None else None
            )
            if not self._brake_recorded:
                self._brake_recorded = True
                self._metrics.mark_brake(
                    getattr(reply, "frame_id", None),
                    getattr(reply, "server_processing_time", 0.0),
                    sim_time,
                    brake_loc,
                    recv_wall=now,
                )
                self.log("[pipeline-client] BRAKE at t={:.2f} {}".format(sim_time, brake_loc))

        # Client extension per-tick work (e.g. V2V: drive the scripted leader and
        # schedule the in-process intent). Runs every tick, independent of the tail.
        if self._ext is not None and hasattr(self._ext, "drive"):
            self._ext.drive(self.source, sim_time, now)

        if not self._tail:
            return  # no client-side control (e.g. a forward-only pipeline)

        # Assemble ego telemetry for the control stage (duck-typed attributes the
        # brake_controller stage reads); run the tail and apply the command.
        telem = Message()
        telem.speed_mps = speed_mps
        telem.loc_y = self.source.vehicle_location().y if has_ego else 0.0
        telem.t = ctrl_t
        telem.brake_pending = self._latched
        cmd = run_segment(self._tail, telem)
        if cmd is None:
            return
        if getattr(cmd, "speed_kick", False) and self._target_mps is not None and has_ego:
            self.source.set_target_velocity(self._target_mps)
        if hasattr(self.source, "apply_control"):
            self.source.apply_control(
                getattr(cmd, "throttle", 0.0),
                getattr(cmd, "steer", 0.0),
                getattr(cmd, "brake", 0.0),
            )
        # From the original: render the visualiser every control tick (sets the loop rate).
        if self._viz is not None:
            self._viz.update_frame(frame.image if frame is not None else None)
            self._viz.render(
                getattr(cmd, "throttle", 0.0),
                getattr(cmd, "steer", 0.0),
                getattr(cmd, "brake", 0.0),
                sim_time,
            )
        # Also from the original: the rest of the original's per-tick work.
        if self._orig_loop and hasattr(self.source, "original_client_loop_poll"):
            self.source.original_client_loop_poll(self.log)
            if self._latched:  # the original printed this every braking tick
                self.log(
                    "[pipeline-client][BRAKING] Time: {:.2f}s | Speed: {:.2f} m/s | "
                    "Brake: {:.2f}".format(sim_time, speed_mps, getattr(cmd, "brake", 0.0))
                )
        if getattr(cmd, "stopped", False) and not self._done and has_ego:
            loc = self.source.vehicle_location()
            stop_loc = (round(loc.x, 2), round(loc.y, 2), round(loc.z, 2))
            self._metrics.mark_stop(sim_time, stop_loc)
            # Same shape as the flat client's STOP line, and as this client's own
            # BRAKE line, so a stop is comparable straight from the logs instead
            # of only from the metrics CSV.
            self.log("[pipeline-client] STOP at t={:.2f} {}".format(sim_time, stop_loc))
            self._done = True

    def should_transmit(self, ctx) -> bool:
        return self._transmit

    def collect_metrics(self, frame, ctx) -> None:
        pass

    def is_done(self, ctx) -> bool:
        if self._source_done:
            return True
        # A scenario that ends before the vehicle stops destroys the ego, and
        # this client then waits for a stop that cannot arrive. Rate-limited in
        # ego_removed, and never consulted for a source that has no ego, so a
        # replay, dataset or synthetic run behaves exactly as it did.
        gone, self._ego_checked_at = ego_removed(self.source, self._ego_checked_at)
        if gone:
            self._ego_gone = True
            self._done = True
            self.log(
                "[pipeline-client] the ego is no longer in the world; ending "
                "the run (the scenario ended before the stop)"
            )
            return True
        # An extension may hold the run open past the follower's stop (V2V waits for
        # the leader to stop too, with a post-brake timeout), matching the flat app.
        if self._ext is not None and hasattr(self._ext, "is_done"):
            return bool(self._ext.is_done(self._done, self._last_sim_time))
        return self._done

    def message_stats(self):
        """Return what this client observed, for the run's link report.

        The same measurement the flat client reports, from the same records: the
        per-frame round trips in milliseconds, how many frames were transmitted
        and how many were answered, so a link that drops messages shows as
        unanswered frames rather than being invisible. This used to be absent,
        and a staged pipeline on a shaped link therefore wrote no link report at
        all - a gap in this client, not a property of staged pipelines.

        Whether an unanswered frame means anything depends on the terminus: the
        closed-loop brake controller answers on an event, so unanswered is the
        normal case; the open-loop decision log answers every frame, so an
        unanswered one is a measurement of the link. The terminal stage declares
        which it is (``Stage.answers_every_message``) rather than this inferring
        it from the data, where a lossy per-frame link and an event-driven one
        look alike.
        """
        from cadet.metrics.link_report import (
            REPLY_ON_EVENT,
            REPLY_PER_MESSAGE,
            MessageStats,
        )

        sent = 0
        received = 0
        latencies = []
        for record in self._metrics.records.values():
            send_t = record.get("frame_send_time")
            recv_t = record.get("server_response_receive_time")
            if send_t is None:
                continue  # captured but never transmitted
            sent += 1
            if recv_t is None:
                continue  # transmitted and never answered
            received += 1
            latencies.append((recv_t - send_t) * 1000.0)
        policy = (
            REPLY_PER_MESSAGE if self._terminus_answers_every_message() else REPLY_ON_EVENT
        )
        return MessageStats(
            sent=sent,
            received=received,
            latencies_ms=latencies,
            reply_policy=policy,
        )

    def _terminus_answers_every_message(self) -> bool:
        """Whether the last stage of the configured pipeline answers every frame.

        Read from the stage class the config names, so a pipeline whose terminus
        is not registered here (it runs on the server's node) still answers the
        question, and an unknown impl falls back to the conservative reading:
        unanswered frames are the normal case.
        """
        from cadet.pipeline.registry import get_stage

        stages = list(getattr(self.cfg.pipeline, "stages", ()) or ())
        if not stages:
            return False
        try:
            return bool(get_stage(stages[-1].impl).answers_every_message)
        except (KeyError, AttributeError):
            return False

    def finish(self, ctx) -> None:
        if self._viz is not None:
            self._viz.close()
        self._metrics.write()
        s = self._metrics.summary()
        self.log(
            "[pipeline-client] frames_sent={} brake_t={} stop_t={}".format(
                s["frames_sent"], s["brake_time"], s["stop_time"]
            )
        )
        if self._ext is not None and hasattr(self._ext, "summary"):
            import json

            extra = dict(self._ext.summary())
            extra.update(frames_sent=s.get("frames_sent"), follower_brake_t=s.get("brake_time"),
                         follower_stop_t=s.get("stop_time"))
            try:
                with open(os.path.splitext(self._csv_path)[0] + "_v2v_summary.json", "w") as fh:
                    json.dump(extra, fh, indent=2, sort_keys=True)
            except Exception as exc:
                self.log("[pipeline-client] could not write extension summary: {}".format(exc))


class PipelineAppParams(BaseModel):
    """Params for the ``pipeline`` driver; the pipeline itself is in ``pipeline.stages``."""

    model_config = ConfigDict(extra="forbid")
    # Match the braking server that stops after issuing its single brake; set false
    # for a pipeline whose server replies on every frame.
    stop_after_reply: bool = True
    # Opt-in; the default is what the original did. Same meaning as the flat app's
    # control_rate: ``per_loop`` issues a command every loop iteration on wall time
    # (original); ``per_tick`` issues one per simulation tick off the snapshot clock.
    control_rate: Literal["per_loop", "per_tick"] = "per_loop"
    # From the original (default): per-tick visualiser + per-frame PNG save on
    # the client, which sets the loop rate. Same as the flat app's client_visualizer.
    client_visualizer: bool = True
    # Whether the visualiser writes a PNG per captured frame. Default True matches the
    # braking original; the V2V original ran save_frames=False (render only, no PNG),
    # so the V2V pipeline sets this false to avoid disk I/O the original never did.
    client_save_frames: bool = True
    # Also from the original (default): the rest of the original client's
    # per-tick work (2 get_transform RPCs + 4 debug prints per tick, get_control per
    # frame, a per-tick braking print). Same as the flat app's original_client_loop.
    original_client_loop: bool = True
    # Optional client-side extension: a ``module:attr`` dotted path to a class the
    # pipeline client instantiates with the config. It lets an app add client-side
    # actuation the staged pipeline cannot express as pure stage output - e.g. the
    # V2V example drives the scripted leader and runs the in-process intent shim. All
    # of its hooks are duck-typed and optional (setup/poll_intent/drive/keep_obstacle/
    # is_done/summary); None (the default) means no extension, so other apps are
    # unaffected. App specifics stay in examples/; this hook is generic.
    client_extension: Optional[str] = None
    # Free-form params the client extension reads (the pipeline driver does not look
    # inside). Keeps PipelineAppParams strict while letting an extension carry its own
    # typed params, e.g. the V2V extension validates these as V2VParams.
    extension_params: Dict[str, Any] = Field(default_factory=dict)
    # Per-frame source metadata keys to forward across the node boundary alongside
    # the image, so a server-side stage can read ground truth the source carried
    # (e.g. the dataset source's ``gt_boxes``). Default empty = forward nothing
    # beyond capture_time/capture_loc, so existing pipelines are unaffected. Only
    # plain (serialisable) values should be named; a simulator handle must not be.
    # Used by the open-loop AV pipeline to score decisions against ground truth.
    forward_meta_keys: List[str] = Field(default_factory=list)


@register_app
class PipelineApp(AppPlugin):
    """Built-in stage-pipeline driver (not an AV app)."""

    name = "pipeline"
    params_model = PipelineAppParams

    @staticmethod
    def client_success(client_app):
        """Return ``(ok, detail)``: did this client's run mean anything?

        One case is added to the default claim (that the client transmitted
        something): a run the simulator ended by removing the ego did not reach
        the end it was measuring, and a client that had already sent frames
        would otherwise report those frames as a success. Every other run is
        judged exactly as before, by the base class.
        """
        if getattr(client_app, "_ego_gone", False):
            return False, "the ego was removed from the world before the run ended"
        return AppPlugin.client_success(client_app)

    @staticmethod
    def sim_hooks(config, model_search_dirs):
        """ScenarioRunner hook, so a CARLA pipeline can spawn its ego.

        Identical mechanism to the flat braking app: by default the command is the
        original ``run_scenario.sh``'s, built from the config by
        :func:`cadet.sim.scenario_command.scenario_runner_command`.
        ``CADET_SCENARIO_RUNNER_CMD`` overrides it (``{host}`` replaced with the
        simulator host); ``CADET_SR_READY`` overrides the readiness marker. If
        neither the env var nor the config names a scenario, no hook is returned
        (non-CARLA pipelines). Kept here rather than imported from examples, which
        the framework must not depend on.
        """
        import os
        import shlex

        from cadet.sim.scenario_command import READY_MARKER, scenario_runner_command

        cmd = os.environ.get("CADET_SCENARIO_RUNNER_CMD")
        if cmd and config.source.kind != "carla":
            # A scenario drives a simulator, and a source that is not the
            # simulator has none to drive: this pipeline is open-loop over
            # frames, so there is no hook to start. The variable is a property
            # of the environment rather than of the run - one shell runs many
            # configs - and refusing the run because it is set was wrong: it
            # made an unrelated variable break every non-simulator pipeline.
            # `{host}` is likewise meaningless here, which is why the command is
            # dropped rather than expanded against a block the source lacks.
            cmd = None
        if cmd:
            argv = shlex.split(cmd.replace("{host}", str(config.source.carla.host)))
        else:
            argv = scenario_runner_command(config)
        if not argv:
            return []
        from cadet.core.launcher import LaunchSpec

        marker = os.environ.get("CADET_SR_READY", READY_MARKER)
        scenario = getattr(config.source, "scenario", None)
        env = {}
        if scenario is not None:
            if scenario.ego_vehicle:
                env["CADET_SCENARIO_VEHICLE"] = str(scenario.ego_vehicle)
            if scenario.target_object:
                env["CADET_SCENARIO_OBJECT"] = str(scenario.target_object)
        return [
            LaunchSpec(
                name="scenario_runner",
                argv=argv,
                env=env or None,
                readiness_marker=marker,
                readiness_timeout_s=float(os.environ.get("CADET_SR_TIMEOUT", "120")),
            )
        ]

    @staticmethod
    def build_server(config, model_search_dirs, logger=None, node_name=None):
        return PipelineServerApp(config, model_search_dirs, logger=logger, node_name=node_name)

    @staticmethod
    def build_client(config, run_id, rep, csv_path, logger=None, node_name=None):
        return PipelineClientApp(config, run_id, rep, csv_path, logger=logger)
