"""Implemented-feature gate: which declared options actually work yet.

The schema (:mod:`cadet.config.schema`) deliberately declares the full planned
config surface, including options not implemented yet, so configs are
forward-compatible and self-documenting. This module is the single place that
says which of those options are wired up, so an experiment that asks for an
unbuilt feature fails loudly ("not implemented yet: X, planned") instead of
being silently ignored.

Contains :data:`IMPLEMENTED` (option path to set of built and verified values),
:data:`EXPERIMENTAL` (implemented but unverified; refused by default), and the
gate functions :func:`unimplemented_features`, :func:`experimental_features`,
and :func:`allow_experimental`.

``cadet validate`` and ``preflight`` (so ``cadet run`` / ``sweep`` /
``emulate``) call :func:`unimplemented_features` after schema validation and
refuse to proceed if it is non-empty. Three tiers exist because two were not
enough: a feature that is written but unproven should be refusable by default
yet exercisable with an explicit opt-in that is recorded in the run record. A
planned option stays refused regardless of the opt-in: there is nothing behind
it to run. The opt-in is ``experiment.allow_experimental`` in the config, or
``--allow-experimental`` on ``run`` / ``sweep`` / ``emulate``.
"""

from typing import List, Optional, Tuple

from cadet.config.schema import Config

# Enumerated options and the subset of their values that are built AND
# verified. Everything the schema allows that appears in neither this map nor
# EXPERIMENTAL counts as "planned" and is always refused.
IMPLEMENTED = {
    # `live` is the replay path with a different reader, not a fourth source: a
    # cv2.VideoCapture behind cadet.sources.replay.FrameReader, opened on an
    # integer `device` index or a stream `url` (exactly one, enforced in the
    # schema), with replay's own pacing, frame counting and stop condition above
    # it. It left this gate on a live check, not on its tests. There is no camera
    # on the machine it was run on and no kernel module was loaded for one, so
    # the stream was a generated clip served out of a process as an HTTP MJPEG
    # stream - unbounded, server-paced and not seekable, which is what a camera
    # looks like to a capture. Over that URL the reference example ran end to
    # end: 20 frames pulled, every one answered, p50 round trip 0.89 ms and no
    # unanswered message, the frame log, the link report and the client outcome
    # written as on any other source, status `complete`, and nothing left
    # running afterwards. The two rate cases both came out where they should:
    # with `fps: 0` (the default, which makes the stream the clock) the 20 frames
    # took 1.6 s of active time against the server's 15 fps, and with `fps: 5.0`
    # the same 20 took 4.2 s. `cadet validate` accepts that config and refuses a
    # live source naming both targets or neither, by name.
    #
    # What that check does *not* cover is the `device` index against hardware:
    # no camera is reachable from here, so the furthest it was taken is that
    # `device: 0` reaches cv2.VideoCapture as a bare int and that a run against
    # it fails loudly - `could not open capture: 0` - rather than producing an
    # empty run. **Verified on a camera: no.** That is a limit of the machine,
    # not of the reader, which is the same code either way.
    #
    # `dataset` is the deterministic Dataset-Mode source
    # (cadet.sources.dataset): one sample per frame, carrying the input image and
    # the sample's ground truth, for scoring a model's answer against it. Its
    # built-in generated sample set needs no file and no download, which is what
    # the dataset-VLM example's stub config and unit tests run on; a manifest
    # path reads public data the example fetches without committing it.
    "source.kind": {"carla", "replay", "synthetic", "live", "dataset"},
    "source.sync_mode": {True},  # async (free-running) execution not implemented
    # local runs a role as a host process; docker runs it in a per-node
    # container on a per-run network, verified by the lifecycle matrix (both
    # link backends, normal completion, the active limit, a killed server
    # container and a two-rep sweep, with nothing left behind).
    #
    # The model-artifact gap this comment used to record is closed: the
    # placement stages a node's artifacts onto the machine its container runs
    # on, from a per-machine cache keyed on a digest of the file's bytes, and
    # mounts them read-only. It applies to the local container case too, which
    # was never the special case it looked like - a container on this machine
    # cannot see this machine's search directories either.
    #
    # ssh_venv is deliberately absent, and the reason is no longer "there is
    # no code for it": a launcher exists (cadet/core/venv_launcher.py) and the
    # placement builds it, stages a venv node's outputs, config and weights on
    # its machine and tears them down. What is absent is the evidence. Every
    # value in this map carries the run that verified it, and nothing has run
    # this one: it needs a second machine and a key, and the four lifecycle
    # cases ssh_docker was held to cannot be performed without them. It is
    # unit-tested against fakes and against its own remote shell command
    # executed locally (tests/unit/test_venv_launcher.py), which proves the
    # argv, the pid-file teardown and the refusals, and proves nothing about
    # SSH or a second kernel. So it stays planned and refused, and the live
    # matrix that would move it is recorded in the hand-off report rather than
    # here, where only verified values belong.
    #
    # ssh_docker runs a role in a container on another machine, reached over
    # SSH. What has been run, with the gateway on a second machine and the
    # sensors on the orchestrator's own: the distributed-sensing example over
    # `delay`, and the four lifecycle cases - normal completion, the active
    # time limit, the remote container removed mid-run (recorded as `node
    # exited unexpectedly`, code 137) and SIGTERM to the orchestrator (recorded
    # as `interrupted: signal 15`). Each ended with nothing on the second
    # machine: no container, no outputs staging tree, no staged config, and no
    # root-owned path created there. Before that, on the reference example and
    # the yolo demo: both link backends, the same four cases, and the weights
    # shipped once and then reused from the per-machine cache.
    #
    # Two things remain true of it and are refusals rather than gaps. A netem
    # link whose *sending* node is remote is refused in preflight: a qdisc needs
    # that node's own namespace and the orchestrator has no handle on one on
    # another machine, so such a path is shaped from the local end instead. And
    # a role on another machine reads the run's config from a copy staged there
    # by the placement, not from the path the orchestrator was given, because
    # that path names nothing on a machine with no checkout.
    #
    # ssh_venv runs a role as a plain process in a virtualenv on another
    # machine - no container. What ran, with the gateway in a venv on a second
    # machine and the sensors here: the distributed-sensing example end to end
    # (20 of 20 frames answered per sensor), and the four lifecycle cases -
    # normal completion, the active time limit, the remote role killed mid-run
    # by the pid in its own pid file (recorded as `node exited unexpectedly`,
    # code 255, which is ssh reporting its peer), and SIGTERM to the
    # orchestrator (`interrupted: signal 15`). Each ended with no role running
    # on that machine and none of the three staging directories left: the
    # outputs tree, the staged config, or the run directory holding the pid
    # file. The cross-machine clock difference was *measured* over the venv's
    # own interpreter (round_trip, +0.37 ms with its uncertainty), not assumed.
    #
    # Two things about it are deliberate and worth knowing before changing
    # them. It requires CADET already installed in that venv and never copies
    # the working tree: a venv is an installed environment, and the heavy
    # dependencies could not travel anyway. The consequence is that the version
    # which runs is whatever is installed, so the venv is probed for its commit
    # and a mismatch is warned about with the reinstall command - a warning,
    # not a refusal, because running a deliberately older version is
    # legitimate. That warning fired in anger during this verification, with
    # the second machine one commit behind.
    "nodes.launcher": {"local", "docker", "ssh_docker", "ssh_venv"},
    # All four link backends are implemented: none, the in-process delay line,
    # tc inside the sending node, and the Mininet-WiFi emulated network.
    "links.network.mode": {"none", "delay", "netem", "netwaggle"},
    # The medium a netwaggle link is carried over. Wired is verified: six hosts
    # in their own namespaces over five shaped links, clean teardown. Wi-Fi is
    # verified by the five-sensor cell in examples/distributed_sensing/wifi_5.yaml: six
    # stations on one access point, each associated to it, 20 of 20 messages
    # answered per sensor and 100 arrivals totalling 55,870 bytes at the
    # gateway, with the round trip from 42.6 ms at its fastest. The medium is
    # wmediumd's with the interference model, so the propagation model and the
    # node positions governed the link.
    #
    # Held there by four acceptance cells, one per source family, each a real
    # cell of stations on an access-point connector: the CARLA one brakes on
    # detections served over the air. They are the `netwaggle_wifi` tier of
    # tests/acceptance/matrix.yaml.
    #
    # A wireless topology has one requirement a wired one does not: it must be
    # built in the host's network namespace, because mac80211_hwsim creates its
    # radios there and a container with a namespace of its own sees none of
    # them. The emulation container therefore shares the host's network
    # namespace and keeps its own process namespace. See the wireless note in
    # cadet/network/netwaggle/__init__.py for the arrangement in full.
    "links.network.medium": {"wired", "wifi"},
    # A shared medium nodes attach to: a switch connector is the wired shape,
    # an access-point connector is the cell. The cell is the shape the
    # five-sensor Wi-Fi run above was built from.
    "connectors.type": {"switch", "ap"},
    # A multi-hop path, composed: the hops' delays add into one per-message
    # model on `delay` and into one qdisc on `netem`. Verified live on both - a
    # two-hop 20 ms + 40 ms route measured p50 60.4 ms against a single 60 ms
    # link, with the rule on the node the path starts at and none on the
    # waypoint - and refused on `netwaggle`, which has no per-hop routing yet.
    # `relay` is absent: it needs a forwarding stage that does not exist.
    "routes.realise": {"compose"},
    # Both carriers are implemented. `tcp` is the socket every example has
    # always used. `websocket` puts the same payloads in RFC 6455 binary frames
    # after one HTTP upgrade (cadet.core.websocket), so a link can cross
    # infrastructure that forwards HTTP and nothing else. It left this gate on
    # its own live check, not on its tests: the reference example ran end to end
    # over it, every frame answered, the accept digest checked on the client
    # side, and the same run over a 60 ms `delay` link measured the configured
    # delay - so the delay lines still see whole payloads through the new
    # carrier. Measurements in phase-transport.md.
    "transport.protocol": {"tcp", "websocket"},
    # Both framings are implemented. `frame` is the 4-byte length prefix in
    # cadet.core.service. `stream` sends no prefix: the connection is one
    # continuous stream of wire records, and the receiver reads each record's
    # extent out of the record itself (cadet.core.codec.record_length), which is
    # why the schema refuses it with `encoding: pickle` - a pickle payload has no
    # length in it. Its own live check: the reference example end to end over
    # tcp/stream and over websocket/stream, every frame answered, and the
    # per-connection delay line still measuring the configured delay.
    "transport.mode": {"frame", "stream"},
    # The wire format of a payload. `safe` is a JSON header plus the raw image
    # bytes (cadet.core.codec), which executes nothing it decodes; `pickle` is
    # what the original scripts put on the wire, and it executes code chosen by
    # whoever can connect to the port, which is why it is an opt-in and not the
    # default. Both ran the reference example end to end; the size and timing
    # comparison that made `safe` the default found the two neutral for both.
    "transport.encoding": {"safe", "pickle"},
    # Four system collectors, each running on the node it measures and writing
    # <collector>_<node>.csv into the run directory on the frame and inference
    # logs' own clocks. Each left this gate on a live check, not on its tests:
    # `gpu` and `energy` read the reference and braking examples' GPU node and
    # agreed with a concurrent `nvidia-smi` capture of the same device within
    # its reporting resolution; `cpu` recorded the host and this role's own
    # share, cross-checked against the host's own load; `bandwidth` counted the
    # payload bytes the run's own frame log says it sent. `energy` additionally
    # attributes each inference its own energy, by the original's definition
    # (latency x average power over that inference's window), evaluated in the
    # process that timed the inference. Measurements in `phase-metrics.md`.
    #
    # Those checks cover the x86 NVIDIA path only. `energy`'s Jetson sources
    # (`tegrastats`, `tegra_sysfs`) are now ports of the project's own
    # tegrastats_logger.py and jetson_power_logger.py - rails, units, cadence
    # and failure behaviour taken from those scripts - and are unit-tested
    # against fake readers, but they remain **verified on hardware: no**: no
    # Jetson is reachable from here, so no reading has come off a board. They
    # also meter different rail sets and are not interchangeable.
    #
    # `rpi_proxy` ports the method in rpi_yolo.py and is **not a power
    # measurement**; it is recorded in its own activity_index column, never as
    # watts or joules, and `auto` will not choose it.
    #
    # These are limits of the sources, not of the collector, so they are
    # recorded in the schema docstring and the module docstrings rather than
    # by gating the collector.
    #
    # `clock_offset` is the fifth, and it is not a sampler: the orchestrator
    # measures each node's clock against its own once before the run and once
    # after the roles have stopped, and writes the offset, its uncertainty and
    # every measurement behind them into run.json. It left this gate on a
    # two-host run, not on its tests: with the server in a container on a
    # second machine and the client here, the offset came out at a fifth of a
    # millisecond with a half-width of six hundredths of a millisecond from
    # eleven round trips whose minimum was 0.12 ms, stable to a few
    # microseconds across the run and across runs, and the independent
    # time-daemon cross-check agreed with it - which is the whole of what
    # "agreed" can mean there, because that method's own half-width is wider
    # than the offset being measured. A node on this machine is recorded as
    # `same_clock` with a zero offset, which is a statement about the kernel
    # clock a container shares rather than a measurement. Measurements in
    # `phase-clock.md`.
    #
    # It corrects nothing. The frame and request records keep the clocks they
    # were written with, there is no setting that would change that, and the
    # record carries the sign convention so a reader can correct a cross-node
    # interval themselves and carry the uncertainty with it. That uncertainty
    # is the same order as the offset on a local network, which is the reason
    # the record exists rather than a flaw in it.
    "metrics.collectors": {"queue", "gpu", "cpu", "energy", "bandwidth", "clock_offset"},
    # The foundation-model interface (`cadet summarize` / `cadet ask`). Only
    # `none` is here, and `none` is a real implemented state rather than a
    # placeholder: it is the default, and what it does - refuse both
    # subcommands with a message naming the three fields to set and the
    # environment variable the key is read from - is the behaviour pinned in
    # tests/unit/test_analysis.py.
    #
    # `openai_chat` is any OpenAI-compatible /chat/completions endpoint. It has
    # now been run against a real model rather than a stub: a local
    # transformers server on this machine, model
    # HuggingFaceTB/SmolLM2-1.7B-Instruct on GPU 0, endpoint
    # http://127.0.0.1:18111/v1/chat/completions, over a real two-machine run
    # directory. Both subcommands answered:
    #
    #   `cadet ask` - "Did this run complete successfully, and how many sensor
    #   nodes were there?" - answered correctly and cited the record: the run
    #   completed, quoting the exit codes from run.json, two sensor nodes, and
    #   the right experiment, placement and network condition.
    #
    #   `cadet summarize` - produced an accurate opening paragraph and then
    #   repeated itself for several more.
    #
    # The repetition is the 1.7B model's, not the interface's, and it is
    # recorded rather than hidden because it is what a reader would otherwise
    # discover for themselves: a small model summarises a long structured
    # record badly. Which model answers is configuration - endpoint and model
    # name are config, the key is an environment variable - so the fix for a
    # poor summary is a better endpoint, not a change here. What this tier
    # claims is that the interface builds the record, makes the request and
    # returns the answer, and that is what was verified.
    "analysis.backend": {"none", "openai_chat"},
}

# Options that have an implementation but have not been verified end to end.
# Refused unless a run opts in; see the module docstring.
#
# Membership here is a statement about code that exists, and it is checked
# against the codebase rather than aspiration:
#   * nodes.launcher ssh_docker has been promoted to implemented above. The
#     gate it waited on was changed deliberately and the substitution is
#     recorded here rather than dropped. The old gate was the braking example
#     against a remote server, which was blocked on something outside the
#     launcher: a braking run's sim hook is ScenarioRunner, and the
#     orchestrator launches sim hooks as host processes by design, so the
#     example could not run from the orchestrator's host at all. The gate is
#     now the distributed-sensing example, which needs no simulator and
#     exercises more of the remote path than braking would: several clients
#     fanning in to one remote server, that server's device metrics measured on
#     its own machine, and the two machines' clock difference measured instead
#     of assumed. The braking-against-a-remote-server run is still worth doing
#     and is no longer a precondition for anything.
#   * source.kind live has been promoted to implemented above, on the live
#     check recorded there. Its one remaining gap - the device-index form
#     against a real camera - is recorded as a gap rather than as a tier,
#     because the code path is the same capture either way and there is no
#     camera here to point it at.
#   * routes.realise relay is NOT here: a relayed hop needs a forwarding stage
#     on the waypoint, which does not exist yet, so it is planned and stays
#     refused however loudly a config asks. `compose` is implemented above.
# Nothing is experimental in this build: every option the schema names is either
# implemented with the run that verified it recorded above, or planned and
# refused. The map stays, because the tier is a real one and the next unverified
# option belongs in it.
EXPERIMENTAL = {}


def tier(path: str, value) -> str:
    """Return ``"implemented"``, ``"experimental"`` or ``"planned"`` for a value."""
    if value in IMPLEMENTED.get(path, ()):
        return "implemented"
    if value in EXPERIMENTAL.get(path, ()):
        return "experimental"
    return "planned"


def _requested(config: Config) -> List[Tuple[str, object, str]]:
    """Return ``(label, value, path)`` for every gated option this config asks for."""
    asked: List[Tuple[str, object, str]] = [
        ("source.kind={}".format(config.source.kind), config.source.kind, "source.kind"),
    ]
    sync = getattr(config.source, "sync_mode", True)
    asked.append(("source.sync_mode=false (async execution)", sync, "source.sync_mode"))
    for node in config.nodes:
        asked.append(
            (
                "nodes[{!r}].launcher={}".format(node.name, node.launcher),
                node.launcher,
                "nodes.launcher",
            )
        )
    for link in config.links:
        asked.append(
            (
                "links[{!r}].network.mode={}".format(link.name, link.network.mode),
                link.network.mode,
                "links.network.mode",
            )
        )
        asked.append(
            (
                "links[{!r}].network.medium={}".format(link.name, link.network.medium),
                link.network.medium,
                "links.network.medium",
            )
        )
    for conn in getattr(config, "connectors", None) or []:
        # A cell is spelled on the connector, not on a link, so gating the
        # link's medium alone would let the shape that has no links through.
        asked.append(
            (
                "connectors[{!r}].type={}".format(conn.name, conn.type),
                conn.type,
                "connectors.type",
            )
        )
    for route in getattr(config, "routes", None) or []:
        asked.append(
            (
                "routes[{!r}].realise={}".format(route.name, route.realise),
                route.realise,
                "routes.realise",
            )
        )
    asked.append(
        (
            "transport.protocol={}".format(config.transport.protocol),
            config.transport.protocol,
            "transport.protocol",
        )
    )
    asked.append(
        ("transport.mode={}".format(config.transport.mode), config.transport.mode, "transport.mode")
    )
    asked.append(
        (
            "transport.encoding={}".format(config.transport.encoding),
            config.transport.encoding,
            "transport.encoding",
        )
    )
    for collector in config.metrics.collectors:
        asked.append(
            ("metrics.collectors[{}]".format(collector), collector, "metrics.collectors")
        )
    backend = getattr(config, "analysis", None)
    if backend is not None:
        asked.append(
            (
                "analysis.backend={}".format(backend.backend),
                backend.backend,
                "analysis.backend",
            )
        )
    return asked


def allow_experimental(config: Config, flag: Optional[bool] = None) -> bool:
    """Resolve the experimental opt-in from the command line and the config.

    The flag wins when it is given, so ``--allow-experimental`` can enable a
    config that does not ask for it; otherwise
    ``experiment.allow_experimental`` decides, which is what lets ``cadet
    validate`` check a config that carries its own opt-in.
    """
    if flag:
        return True
    return bool(getattr(config.experiment, "allow_experimental", False))


def experimental_features(config: Config) -> List[str]:
    """Return the experimental options this config asks for, opted into or not."""
    return [
        label
        for label, value, path in _requested(config)
        if tier(path, value) == "experimental"
    ]


def unimplemented_features(config: Config, allow_experimental: bool = False) -> List[str]:
    """Return the requested features this codebase will not run.

    Each string is a human-readable "<path>=<value>" plus the tier that refused
    it, so the reader can tell "there is no code for this" from "there is code
    but you did not opt in".

    ``allow_experimental`` accepts the experimental tier. It never accepts the
    planned tier: there is nothing behind a planned option, so allowing it could
    only replace a clear refusal with a crash.
    """
    out: List[str] = []
    for label, value, path in _requested(config):
        level = tier(path, value)
        if level == "implemented":
            continue
        if level == "experimental":
            if allow_experimental:
                continue
            out.append(
                "{} (experimental: implemented but unverified; pass "
                "--allow-experimental or set experiment.allow_experimental: "
                "true to accept it)".format(label)
            )
            continue
        out.append("{} (planned)".format(label))
    return out
