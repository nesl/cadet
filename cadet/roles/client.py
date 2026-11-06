"""Generic client runtime (paper Fig. 1(b)).

What this is for:
    Drives one client run through the pipeline stages (source, frame processing,
    transport, action translation, metric collection), delegating the per-stage
    behaviour to an app plugin's client hooks.

What it contains:
    - :class:`RunContext`: the shared handles (config, source, transport, output).
    - :class:`ClientRuntime`: the staged loop.
    - :func:`build_client`: assemble context + app for a config/run from the CLI.

How it fits into CADET:
    `cadet client <config> --node client [--run N] [--rep R]` calls
    :func:`build_client` then :meth:`ClientRuntime.run`. The app (e.g.
    :mod:`examples.braking.flat.app`) supplies control/encoding/metrics; the source is
    :class:`cadet.sources.carla.CarlaSource`.

    **System metrics.** When ``metrics.collectors`` names them, the runtime also
    runs the per-node collectors (:mod:`cadet.metrics.collectors`) for this
    client's node, around the loop, writing ``<collector>_<node>.csv`` into the
    run directory. Default is no collectors, and then nothing here changes.
"""

import json
import os
import time
from typing import List

from cadet.core.apps import get_app
from cadet.core.codec import set_wire_encoding_from_config
from cadet.core.transport import FramedSocketTransport
from cadet.metrics.collectors import CollectorSet, build_collectors
from cadet.roles.common import node_by_role, prepare_output_dir, select_run_config

# How long to connect-retry while the server is still loading its model.
SERVER_CONNECT_RETRY_S = 60.0

# The two startup orders; see experiment.client_start.
#   after_scenario  - the default, and what the original scripts did: the
#       scenario runs first and the client finds a scene already in place.
#   before_scenario - opt in. The client does everything that does not need the
#       scene, says it is ready, and the orchestrator starts the scenario only
#       once every client has. The ego exists for a short window, and a client
#       still setting up when it opened could miss it; this removes that race,
#       and is a deviation from the original.
CLIENT_START_BEFORE_SCENARIO = "before_scenario"
CLIENT_START_AFTER_SCENARIO = "after_scenario"
# The default lives here, in one place, so the schema and both orchestrator
# paths cannot drift from each other about what an absent setting means.
CLIENT_START_DEFAULT = CLIENT_START_AFTER_SCENARIO

# Printed once the client can do nothing further without the scene. The
# orchestrator waits for this before starting the sim hook, so it must be
# stable: it is a protocol between the two, not a log message.
CLIENT_READY_MARKER = "[client] ready, waiting for the scene"

# Fallback for how long a client waits for the scene after announcing itself
# ready, when the config names no scenario. It has to cover the orchestrator
# starting the sim hook and the scenario reaching the point where it spawns the
# ego. A scenario that spawns its ego later says so with
# ``source.scenario.ego_attach_timeout_s``, which wins.
SCENE_WAIT_TIMEOUT_S = 120.0

# Where a client records whether it produced output. Read by the orchestrator,
# which fails a run that produced none; see AppPlugin.client_success.
# The client's own verdict, one file per client node. It used to be a single
# fixed name, which meant N clients in one run directory overwrote each other
# and only the last survived - the distributed-sensing example hit exactly that
# with N sensors.
CLIENT_OUTCOME_PREFIX = "client_outcome"


def client_outcome_file(node_name) -> str:
    """Return the outcome file name for a client node."""
    return "{}_{}.json".format(CLIENT_OUTCOME_PREFIX, node_name)


def scene_timeout_for(config) -> float:
    """Return how long this config's client should wait for the scene."""
    scenario = getattr(getattr(config, "source", None), "scenario", None)
    declared = getattr(scenario, "ego_attach_timeout_s", None) if scenario else None
    return float(declared) if declared else SCENE_WAIT_TIMEOUT_S


class RunContext:
    """Handles shared between the runtime and the app for one run."""

    def __init__(self, config, source, transport, run_id, rep, out_dir, log):
        self.config = config
        self.source = source
        self.transport = transport
        self.run_id = run_id
        self.rep = rep
        self.out_dir = out_dir
        self.log = log


class ClientRuntime:
    """Runs the client loop until the app reports it is done."""

    def __init__(
        self,
        ctx: RunContext,
        app,
        server_host,
        server_port,
        node=None,
        server_label=None,
        client_start: str = CLIENT_START_DEFAULT,
        scene_timeout_s: float = SCENE_WAIT_TIMEOUT_S,
        collectors=None,
    ):
        self.ctx = ctx
        self.app = app
        self._server_host = server_host
        self._server_port = server_port
        self._server_label = server_label or "{}:{}".format(server_host, server_port)
        # This client's own node, for its transport tuning.
        self._node = node
        self._client_start = client_start
        self._scene_timeout_s = scene_timeout_s
        # The per-node system-metric collectors for this node, if the config
        # asked for any. An empty set is the default and costs nothing.
        self._collectors = collectors if collectors is not None else CollectorSet()
        # Outbound send pacing. A channel from this node may cap its frame rate via
        # ``stream_fps``; when set, the client sends at most one frame per 1/fps and
        # SKIPS (drops) any frame that arrives sooner, keeping the freshest one. This
        # stops a free-running camera from outpacing a rate-limited server feeder (e.g.
        # the V2I network hop) and piling frames in the OS socket buffer. Unset =
        # free-running (the default, and braking/V2V's behaviour), so nothing changes
        # unless a config opts in. Only the ego feed is paced here; an app that drives
        # a second feed (V2I infra) paces that one itself.
        self._min_send_interval = None
        node_name = getattr(self._node, "name", None)
        for ch in (getattr(ctx.config, "channels", None) or []):
            if (ch.from_ == node_name and ch.stream_fps and ch.stream_fps > 0
                    and ch.device_id in (None, "ego")):
                self._min_send_interval = 1.0 / float(ch.stream_fps)
                break
        self._last_send_mono = None

    def run(self) -> None:
        if self._client_start == CLIENT_START_AFTER_SCENARIO:
            self._start_after_scenario()
            self.app.setup(self.ctx)
        else:
            # app.setup() happens *inside* the before-scenario start, ahead of
            # the ready marker: loading an app's stages or model needs no scene,
            # so doing it first is what makes "ready" mean ready.
            self._start_before_scenario()
        # Started after setup so loading the model or the stages is not charged
        # to the run, and before the loop so every frame it sends is covered.
        self._collectors.start()
        try:
            while not self.app.is_done(self.ctx):
                frame = self.ctx.source.next_frame()  # 1. source
                reply = self.app.poll_reply(self.ctx)  # 2. transport in
                self.app.translate_action(frame, reply, self.ctx)  # 3. action
                # Pace the outbound feed to the channel's stream_fps (when set): skip a
                # frame that arrives sooner than 1/fps since the last send, keeping the
                # freshest frame rather than flooding a rate-limited server feeder.
                paced_ok = True
                tick_now = None
                if self._min_send_interval is not None and frame is not None:
                    tick_now = time.monotonic()
                    if self._last_send_mono is not None and (
                            tick_now - self._last_send_mono < self._min_send_interval):
                        paced_ok = False
                if frame is not None and paced_ok and self.app.should_transmit(self.ctx):
                    # Schedule the next slot from this tick (before processing), so the
                    # per-frame encode cost does not inflate the interval: 1/fps between
                    # sends, not 1/fps + encode.
                    if tick_now is not None:
                        self._last_send_mono = tick_now
                    # The server closes the socket right after issuing a brake, so a
                    # send racing that close can raise BrokenPipe. Swallow transmit
                    # errors (as the original client did) — the brake reply is already
                    # queued and is handled on the next tick, which stops transmission.
                    try:
                        payload = self.app.process_frame(frame, self.ctx)  # 4a. process
                        self.ctx.transport.send(payload)  # 4b. transport out
                        self.app.on_frame_sent(frame, self.ctx)
                    except Exception as exc:
                        self.ctx.log("[client] transmit error (ignored): {}".format(exc))
                self.app.collect_metrics(frame, self.ctx)  # 5. metrics
        finally:
            # The app's own finish() can still send, and closing the transport
            # is the last thing that can move a byte, so the collectors' final
            # window is closed after both - otherwise this node's totals are
            # short by whatever its teardown transferred, while the node at the
            # other end counts it. Both in their own try, because a metrics
            # record must never be the reason a run that worked is reported as
            # failed.
            try:
                self.app.finish(self.ctx)
            finally:
                self.ctx.source.close()
                try:
                    self.ctx.transport.close()
                except Exception:
                    pass
                try:
                    self._collectors.stop()
                except Exception as exc:  # noqa: BLE001
                    self.ctx.log("[client] collectors did not stop cleanly: {}".format(exc))
            self._write_link_report()
            self._write_outcome()

    def _start_after_scenario(self) -> None:
        """The default, and the original order: the scene already exists.

        Attach to the ego and camera first, then connect to the server
        (retrying while it loads its model). This is what the original client
        did, so it is what an unconfigured run does; see
        ``experiment.client_start``.
        """
        self.ctx.source.open()
        self._connect_to_server()

    def _start_before_scenario(self) -> None:
        """Opt-in order: be ready *before* the scenario spawns the ego.

        Everything that does not need the scene happens first - connecting to
        the simulator and to the server - and only then does the client announce
        itself ready. The orchestrator starts ScenarioRunner once every client
        has, so by the time the ego appears the client has nothing left to do
        but attach to it.

        The ego exists for a short window. A client that was still connecting to
        its server, or still loading, when that window opened could miss it
        entirely, which is what this ordering exists to prevent.
        """
        self.ctx.source.prepare()
        self._connect_to_server()
        # The app's own setup - stages, model, listener thread - needs no scene,
        # so it belongs before the marker. Leaving it until after would mean the
        # scenario started while the client was still loading, which is the gap
        # this ordering exists to close.
        self.app.setup(self.ctx)
        # The marker the orchestrator waits on. Printed only once the client can
        # do nothing more without the scene, so "ready" means exactly that.
        self.ctx.log(CLIENT_READY_MARKER)
        if not self.ctx.source.await_scene(self._scene_timeout_s):
            raise RuntimeError(
                "the scene did not appear within {:.0f}s of this client becoming "
                "ready; the scenario may have failed to start or to spawn the "
                "ego".format(self._scene_timeout_s)
            )
        self.ctx.source.open()

    def _connect_to_server(self) -> None:
        self.ctx.transport = FramedSocketTransport.connect(
            self._server_host,
            self._server_port,
            retry_s=SERVER_CONNECT_RETRY_S,
            node=self._node,
            # The wire this run uses, from the one `transport` section both ends
            # read: a connection cannot negotiate it, so both are configured.
            protocol=self.ctx.config.transport.protocol,
            mode=self.ctx.config.transport.mode,
            # Names the link in bandwidth_<node>.csv by where it goes, rather
            # than by an ephemeral port pair.
            label=self._server_label,
        )
        self.ctx.log(
            "[client] connected to server {}:{}".format(self._server_host, self._server_port)
        )

    def _write_outcome(self) -> None:
        """Record whether this client produced output, for the orchestrator.

        Evaluated here because this is where the evidence is: the app object
        holds the counts, and the condition for what counts is the app's to
        declare. The orchestrator reads the file rather than parsing a log,
        which would make a run's verdict depend on a log line's wording.
        """
        from cadet.core.apps import get_app

        try:
            plugin = get_app(self.ctx.config.app.plugin)
            ok, detail = plugin.client_success(self.app)
        except Exception as exc:  # noqa: BLE001 - runs in the run's finally
            # Resolving the plugin or running its check must not raise here: an
            # exception would propagate out of the teardown and hide what the
            # run actually did. A verdict that cannot be reached is not a
            # failure, so it defaults to not contradicting the run.
            ok, detail = True, "success check unavailable: {}".format(exc)
        path = os.path.join(self.ctx.out_dir, client_outcome_file(self.node_name))
        try:
            with open(path, "w") as handle:
                json.dump({"ok": bool(ok), "detail": detail}, handle, indent=2)
        except OSError as exc:
            self.ctx.log("[client] could not write {}: {}".format(path, exc))

    @property
    def node_name(self) -> str:
        """This client's node name, which its records are keyed by."""
        return getattr(self._node, "name", None) or "client"

    def _write_link_report(self) -> None:
        """Record the configured link parameters beside what was measured.

        Skipped for an app that does not report message statistics: the record
        would otherwise claim a measurement it does not have.
        """
        stats_fn = getattr(self.app, "message_stats", None)
        if stats_fn is None:
            return
        from cadet.metrics.link_report import write_link_report

        try:
            report = write_link_report(
                self.ctx.config, self.ctx.out_dir, stats_fn(), node_name=self.node_name
            )
        except Exception as exc:
            # A metrics record must never fail a run that otherwise succeeded.
            self.ctx.log("[client] link report not written: {}".format(exc))
            return
        measured = report["measured"]
        self.ctx.log(
            "[client] effective message latency p50={p50} p99={p99} ms, "
            "unanswered={un}/{sent}".format(
                p50=_fmt_ms(measured["latency_ms"]["p50"]),
                p99=_fmt_ms(measured["latency_ms"]["p99"]),
                un=measured["unanswered"],
                sent=measured["messages_sent"],
            )
        )


def _fmt_ms(value):
    """Format a millisecond value for the log, tolerating None."""
    return "n/a" if value is None else "{:.2f}".format(value)


def build_client(
    config_path: str,
    overrides: List[str],
    run_index: int,
    rep: int,
    logger=None,
    node_name=None,
):
    """Build a :class:`ClientRuntime` for one run from a config file.

    ``node_name`` selects which client node this process is, the way
    :func:`cadet.roles.server.build_server` already did for servers. Without it
    every client process in a run believed it was the first client node: the
    flag existed on ``cadet client`` and was never passed through, so with N
    clients they shared one identity, one outcome file and one link report.
    """
    log = logger or (lambda m: print(m))
    config, run_id_base = select_run_config(config_path, overrides, run_index)
    # The wire encoding is a per-process choice the apps read through
    # cadet.core.codec, so it is selected once here, before any transport
    # exists. Selecting it later would leave a payload encoded one way and
    # decoded another.
    set_wire_encoding_from_config(config)
    run_id = "{}-r{}".format(run_id_base, rep)
    out_dir = prepare_output_dir(config, run_id)

    # Source is selected by config.source.kind (carla / replay / synthetic / live).
    from cadet.sources import build_source

    source = build_source(config, logger=log)

    server = node_by_role(config, "server")
    if server is None:
        raise RuntimeError("config has no node with role 'server'")

    csv_name = "{}_latency_location_log.csv".format(run_id)
    csv_path = os.path.join(out_dir, csv_name)

    # The node this process is, so an app can name its own per-node outputs the
    # way the framework names its own. Without it an app has only the process id
    # to tell itself apart from its siblings, which is distinct between host
    # processes and *identical* between containers - every node container's pid
    # namespace starts at 1 - so N containerised clients silently overwrote one
    # another's log and the run looked like it had one client.
    app = get_app(config.app.plugin).build_client(
        config, run_id, rep, csv_path, logger=log, node_name=node_name
    )
    ctx = RunContext(
        config, source, transport=None, run_id=run_id, rep=rep, out_dir=out_dir, log=log
    )
    if node_name is not None and node_name != "client":
        client_node = next((n for n in config.nodes if n.name == node_name), None)
        if client_node is None or client_node.role != "client":
            raise RuntimeError("no client node named {!r}".format(node_name))
    else:
        client_node = next((n for n in config.nodes if n.role == "client"), None)
    client_name = client_node.name if client_node is not None else "client"
    return ClientRuntime(
        ctx,
        app,
        server.host,
        server.port,
        node=client_node,
        server_label=server.name,
        collectors=build_collectors(config, client_name, out_dir, logger=log),
        client_start=getattr(
            config.experiment, "client_start", CLIENT_START_DEFAULT
        ),
        scene_timeout_s=scene_timeout_for(config),
    )
