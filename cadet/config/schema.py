"""CADET experiment configuration schema (pydantic, Python 3.8 compatible).

Defines the single, typed shape of a CADET experiment config. A YAML file is
parsed into :class:`Config`; validation errors are turned into readable
messages by the loader/CLI. Nothing here executes an experiment: this is the
contract every runner, role and launcher reads.

Top-level sections of :class:`Config`: experiment, source (discriminated union
on ``kind``), models, nodes, channels, links (with a typed network model),
transport, metrics, app (generic ``plugin`` + raw ``params``, validated against
the registered plugin's model by the loader), sweep, outputs, analysis. Plus the leaf
models those sections use and the cross-field validators (app/plugin agreement,
channel/link endpoints must name existing nodes).

Some options are declared here but not implemented yet (e.g.
``nodes.launcher: ssh_venv``). They
validate against the schema and are
rejected with a "planned" message by :mod:`cadet.config.implemented` at
``cadet validate``; they are never silently ignored.

``cadet.config.loader`` builds :class:`Config` from YAML + ``--set``
overrides, expands ``sweep`` into concrete per-run configs, and derives the
``run_id``. ``cadet validate`` reports validation errors against this schema.
Author a YAML with the sections above (see ``configs/*.yaml``). Weather is a
named preset (:mod:`cadet.config.weather`); model artifacts are per-device in
``models.registry``; swept fields are dotted paths under ``sweep.axes``.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Annotated, Literal

from cadet.config.weather import WeatherName
from cadet.metrics.clock_offset import (
    PROBE_COUNT_DEFAULT as CLOCK_PROBE_COUNT,
)
from cadet.metrics.clock_offset import (
    PROBE_TIMEOUT_S_DEFAULT as CLOCK_PROBE_TIMEOUT_S,
)
from cadet.metrics.collectors import (
    COUNTER_SAMPLE_INTERVAL_MS,
    ORIGINAL_SAMPLE_INTERVAL_MS,
)


class _Strict(BaseModel):
    """Base model that rejects unknown keys so typos become validation errors."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------
class Experiment(_Strict):
    """Identity and repetition policy for a run family."""

    name: str
    app: str  # plugin key; must equal ``Config.app.plugin``
    repetitions: int = 1
    seed: Optional[int] = None
    # A run has two phases and a limit on each. One limit over both would mean
    # different things on different hardware: an AV run spends about ten seconds
    # loading model weights before its client sends anything, so a single bound
    # is mostly a bound on that load.
    #
    # Everything before the workers start: the cleanup, preflight, the shaping,
    # and every readiness wait. None takes the sum of the readiness timeouts the
    # run will actually wait on, which is already the phase's bound one step at
    # a time (see cadet.orchestrator.lifecycle.startup_limit_s).
    max_startup_s: Optional[float] = None
    # The worker phase. None derives it from source.scenario.timeout_s (see
    # cadet.orchestrator.lifecycle.active_limit_s); it exists to stop a run
    # whose workers never finish, not to bound a healthy run.
    max_active_s: Optional[float] = None
    # What ends a run:
    #   client_report  - the clients (and load generators) finishing. This is
    #                    what the original orchestrators did: they ran one
    #                    client in the foreground and a sim hook that outlived
    #                    or predeceased it changed nothing.
    #   sim_hook_exit  - the sim hook finishing, i.e. the scenario deciding the
    #                    run is over, with the workers stopped after the grace
    #                    below.
    #   first          - whichever of the two happens first.
    # Left implicit, a scenario that ended while the client kept streaming ran
    # on until max_active_s; this makes the choice a stated one.
    end_on: Literal["client_report", "sim_hook_exit", "first"] = "client_report"
    # How long the workers may keep running after a sim hook has exited, under
    # the two rules that end on it. It is there so a client can finish writing
    # its own report instead of being stopped mid-metric.
    end_grace_s: float = 5.0
    # When the clients start, relative to the app's sim-side hooks.
    #   after_scenario  - the default: the original order, with the sim hook
    #       started before the clients, which then find a scene already in
    #       place. The default is the original's behaviour.
    #   before_scenario - opt in. Clients start after the servers are listening,
    #       do everything that does not need the scene, and announce themselves
    #       ready; the orchestrator starts the sim hook only once every client
    #       has. The ego a scenario spawns exists for a short window, and a
    #       client still setting up when it opened could miss it, so this
    #       removes a race the original had, and is therefore a deviation from it.
    client_start: Literal["before_scenario", "after_scenario"] = "after_scenario"
    # Whether a run that produced no output fails. The original judged a run by
    # the client'"'"'s exit code alone, so a client that connected, sent nothing and
    # exited cleanly counted as a successful run with an empty metrics file;
    # that is the default here. Turned on, the app'"'"'s own client_success decides
    # as well (see cadet.core.apps.AppPlugin.client_success), which fails runs
    # the original counted - a deviation, hence opt-in.
    require_output: bool = False
    # How long a sweep waits between cells. The original slept five seconds
    # between experiments, which on a shared simulator is what gave the previous
    # run'"'"'s actors time to be cleared before the next --reloadWorld.
    inter_run_pause_s: float = 5.0
    # Whether a sweep skips a cell that already has a run.json. The original
    # re-ran everything, so that is the default; resume is the deviation and is
    # asked for with --resume.
    resume: bool = False
    # Accept config options that are implemented but not yet verified end to end
    # (see cadet.config.implemented.EXPERIMENTAL). Off by default, because a run
    # should not depend on unproven code without saying so; the choice is
    # recorded in run.json so a result that leaned on it can be found later.
    # Options with no implementation at all stay refused regardless.
    allow_experimental: bool = False


# ---------------------------------------------------------------------------
# source (discriminated union on `kind`)
# ---------------------------------------------------------------------------
class Camera(_Strict):
    """Ego camera sensor geometry."""

    width: int = 960
    height: int = 512
    fov: float = 90.0
    # Ego-camera mount on the vehicle (forward X, up Z in metres, pitch in degrees).
    # Defaults reproduce the original braking/V2V client mount (x=2.5, z=1.8,
    # pitch=-10); the original V2I client mounted 0.2 m lower (z=1.6), set per-app
    # in that config so one camera height never silently changes another app.
    mount_x: float = 2.5
    mount_z: float = 1.8
    mount_pitch: float = -10.0
    # Camera->client frame buffer depth (drop-newest when full). Default 10; the
    # original V2I client used 5 (ego_q/infra_q, v2i_braking.py:77-78), so the V2I
    # config sets 5 to match its in-flight bound.
    queue_size: int = 10
    # Opt-in; the default is what the original did. The original client left the camera
    # free-running (no sensor_tick), so it fired once per rendered frame at the
    # asynchronous server's own rate. True pins the sensor to one frame per
    # ``1/source.fps`` of simulated time, which makes the sampled ego positions
    # repeat across runs; left False to reproduce the original exactly.
    fixed_rate: bool = False


class Spawn(_Strict):
    """A world pose (metres, degrees) for spawning an actor."""

    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0


class Occluder(_Strict):
    """A parked actor used to block the ego's line of sight."""

    model: str
    spawn: Spawn


class Pedestrian(_Strict):
    """A pedestrian that holds, then crosses at ``speed_mps`` after ``t_start_s``."""

    spawn: Spawn
    speed_mps: float
    t_start_s: float


class CarlaClientEgg(_Strict):
    """Which CARLA client egg a run must use, and where to find it.

    ``name`` is the egg's filename, and it is the pin: the Python and CARLA
    versions are read out of it and checked against the interpreter and the
    server image (see :mod:`cadet.orchestrator.versions`). ``host_path`` is a
    directory on the machine the role runs on; with it unset the image's own
    baked path is used, which is not configurable because it is the image's
    convention rather than a machine's.

    The two are separate because one earlier field had to mean both "which egg"
    and "where the egg is", and could express neither when they disagreed.
    """

    name: str  # e.g. "carla-0.9.10-py3.7-linux-x86_64.egg"
    host_path: Optional[str] = None


class ScenarioRunnerPin(_Strict):
    """Where the ScenarioRunner tree is, and which tree it must be.

    ``source_digest`` pins the tree **by file contents**, not by a commit id.
    The copies in play are the same files in different repositories - a working
    tree with its own upstream history and a local snapshot repository that
    tracks the same files - so their commit ids differ by design and a hash pin
    could never hold for both. See :mod:`cadet.sim.versions`.
    """

    path: str  # where the tree is mounted or checked out
    source_digest: Optional[str] = None  # the pin; None means "do not check"
    commit: Optional[str] = None  # informational only, never compared
    # The image the scenario driver's own container runs from. A CADET image
    # carries ScenarioRunner's pinned dependency set (see docker/Dockerfile,
    # INSTALL_SIM), which is what makes the scenario reproducible: the
    # alternative, a shared virtualenv kept in step by hand, fails at the
    # scenario's first import and reads as a startup timeout. None means no
    # scenario container is started, which is what happens when the simulator
    # is one somebody else runs.
    image: Optional[str] = None


class CarlaSettings(_Strict):
    """How a run reaches CARLA, and - when it launches one - how it starts it.

    ``host``, ``port`` and ``timeout`` are what every role uses to reach the
    simulator, and they are what ScenarioRunner is told as well, so the
    scenario and the client cannot end up on different servers. That was two
    fields once - the client's connect port and the launch's own port, with
    nothing tying them - and a config could launch a server on one port while
    the client dialled another, which fails as an ego-attach timeout rather than
    as a connection error.

    ``port`` defaults to CARLA's own 2000, which is the original's: its client
    opened ``carla.Client('localhost', 2000)`` and ``run_scenario.sh`` passed no
    ``--port``, taking ScenarioRunner's default of 2000. A run that must not
    collide with a hand-started server sets another port; preflight already
    refuses a port held by a process that is not ours, and names the holder.

    ``image`` is what makes this a launch rather than an attach: with it set the
    orchestrator starts the server, and without it the simulator is somebody
    else's to run.
    """

    # The original's literal was 'localhost'. This is the same endpoint with one
    # ambiguity removed: `localhost` can resolve to ::1 first, while the server
    # binds IPv4, which reads as a connect timeout rather than a wrong address.
    host: str = "127.0.0.1"
    port: int = 2000
    timeout: int = 10
    # Launch settings; `image` unset means no server is started.
    image: Optional[str] = None  # e.g. "carlasim/carla:<version>"
    gpu_index: Optional[int] = None  # None leaves the container's own default
    quality: Literal["Low", "Epic"] = "Low"
    # Offscreen rendering: there is no display in a container, and a server that
    # tries to open one fails in a way that looks like a timeout.
    offscreen: bool = True
    # Whether each run starts a server of its own. The original ran one
    # long-lived simulator for a whole study and cleared the previous run's
    # actors with `--reloadWorld` and a pause between experiments, which is what
    # `orchestrator.inter_run_pause_s` is, so reuse is the default here as well.
    # It is also what makes a sweep affordable: a CARLA start is tens of
    # seconds, and twenty cells pay it twenty times for a server that would have
    # been identical. A run asks for `per_run: true` when it must not inherit
    # any simulator state at all, and two sweeps sharing a machine need either
    # that or different ports, because what a reused server is found by is its
    # launch settings.
    per_run: bool = False
    client_egg: Optional[CarlaClientEgg] = None


class Scenario(_Strict):
    """Scenario geometry and lifetime (fields optional; apps use a subset)."""

    map: Optional[str] = None
    # Swept experiment axes: which ego to spawn, obstacle class, and target speed.
    ego_vehicle: Optional[str] = None  # blueprint tail, e.g. "audi.tt"
    target_object: Optional[str] = None  # obstacle class, e.g. "car" | "person" | "bike"
    ego_speed: Optional[str] = None  # e.g. "40mph" | "20kph" (units parsed downstream)
    ego_spawn: Optional[Spawn] = None
    occluder: Optional[Occluder] = None
    pedestrian: Optional[Pedestrian] = None
    timeout_s: Optional[float] = None
    # How the client finds the ego the sim hook spawned, matching a CARLA
    # ``role_name`` attribute. The default is the original's: it looked for
    # ``role_name == 'hero'`` and nothing else (``wait_for_ego_vehicle`` in the
    # reference client's simulation module), which is what ScenarioRunner tags
    # an ego with. Setting this empty falls back to matching the ego blueprint,
    # which is a deviation and says so when it happens: a blueprint is not
    # unique, so which actor it finds depends on what else is in the world and
    # when - under a distributed start it attached to the wrong vehicle, with
    # the camera somewhere else on the map and the ego never driven.
    # ``ego_attach_timeout_s`` is how long the client polls for the ego after it
    # signals ready (the two-phase start: the scenario spawns the ego only once
    # the client is set up and waiting).
    ego_role_name: Optional[str] = "hero"
    ego_attach_timeout_s: float = 60.0
    # Where the ScenarioRunner tree is and which tree it must be, plus the image
    # its container runs from. Absent means no scenario driver is started.
    runner: Optional[ScenarioRunnerPin] = None
    # What the scenario driver is told to run. These are what the original'"'"'s
    # run_scenario.sh took as its argument and had baked into the tree; naming
    # them here is what lets the framework build the scenario command itself
    # instead of taking it whole from the environment. See
    # cadet.sim.scenario_command.
    name: Optional[str] = None  # --scenario
    definition: Optional[str] = None  # --additionalScenario (a .py)
    config_file: Optional[str] = None  # --configFile (an .xml)


class CarlaSource(_Strict):
    """Input from a running CARLA simulator (the only live backend today)."""

    kind: Literal["carla"] = "carla"
    carla: CarlaSettings = Field(default_factory=CarlaSettings)
    fps: int = 10
    # Synchronous mode: CADET ticks the simulator and blocks each frame. False is
    # free-running (async) execution, planned but not implemented yet.
    sync_mode: bool = True
    camera: Camera = Field(default_factory=Camera)
    scenario: Scenario = Field(default_factory=Scenario)
    weather: Optional[WeatherName] = None


class ReplaySource(_Strict):
    """Deterministic replay from a folder of images or a video file."""

    kind: Literal["replay"] = "replay"
    path: str  # directory of images, or a video file
    fps: float = 10.0  # playback rate (frames per second)
    loop: bool = False  # loop forever, or stop at end
    limit: Optional[int] = None  # cap the number of frames emitted


class SyntheticSource(_Strict):
    """Generated frames for tests (no files, no hardware)."""

    kind: Literal["synthetic"] = "synthetic"
    count: int = 10  # number of frames to emit (0 = unbounded)
    width: int = 64
    height: int = 48
    fps: float = 0.0  # 0 = as fast as possible


class LiveSource(_Strict):
    """Live capture: a camera device index or a network stream URL.

    The replay path pointed at something arriving in real time rather than at a
    directory (see :class:`cadet.sources.live.LiveSource`). Exactly one of
    ``device`` and ``url`` names where frames come from; OpenCV takes either
    directly. ``loop`` is absent on purpose: a camera has no beginning to seek
    back to and a stream is not seekable.
    """

    kind: Literal["live"] = "live"
    device: Optional[int] = None  # capture device index, e.g. 0 for the first camera
    url: Optional[str] = None  # stream URL, e.g. "rtsp://host:554/path" or an HTTP stream
    # 0 pulls frames as fast as the stream delivers them, so the stream is the
    # clock; a positive value caps the pull rate the way replay paces a file.
    fps: float = 0.0
    # Cap on frames emitted, and the only thing in a live source that ends a run
    # by itself: with no limit the run stops on experiment.max_active_s.
    limit: Optional[int] = None

    @model_validator(mode="after")
    def _exactly_one_target(self):
        """Refuse a live source that names both targets or neither."""
        if (self.device is None) == (self.url is None):
            raise ValueError(
                "source.kind: live needs exactly one of `device` (a capture index, e.g. "
                "device: 0) and `url` (a stream URL, e.g. url: rtsp://host:554/path); "
                "this config names {}".format(
                    "both" if self.device is not None else "neither"
                )
            )
        return self


class DatasetSource(_Strict):
    """Deterministic dataset replay for model evaluation (the "Dataset Mode").

    Yields one sample per frame, carrying both the input image and the sample's
    ground truth in the frame metadata, so a consumer can score a model's answer
    against it. Two tasks are supported: ``vqa`` (image + question + answers) and
    ``detection`` (image + ground-truth boxes).

    With ``path`` unset the source generates a small, deterministic, built-in
    sample set in process: no file, no download and no network, which is what the
    stub/quick config and the unit tests run on. With ``path`` set it reads a
    manifest (a JSON list of samples) and the images it names from that folder;
    the example's ``fetch_dataset.py`` writes one from public data.
    """

    kind: Literal["dataset"] = "dataset"
    task: Literal["vqa", "detection"] = "vqa"
    # Directory holding the manifest and the images it names. None = the built-in
    # generated sample set (no files, no download).
    path: Optional[str] = None
    # Manifest file name inside ``path`` (a JSON list of samples). None takes the
    # per-task default (``vqa_manifest.json`` / ``detection_manifest.json``).
    manifest: Optional[str] = None
    fps: float = 0.0  # 0 = as fast as possible
    limit: Optional[int] = None  # cap the number of samples emitted
    count: int = 8  # built-in sample count when ``path`` is unset


Source = Annotated[
    Union[CarlaSource, ReplaySource, SyntheticSource, LiveSource, DatasetSource],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# sim (the simulator side the orchestrator owns)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
class ModelEntry(_Strict):
    """A logical model mapped to a concrete artifact per device type.

    Only the artifact mapping lives here; decision thresholds (conf_thresh, …)
    belong to the app params so each value has a single source of truth.
    """

    # device type (e.g. "gpu", "edge", "hailo") -> artifact filename/path.
    per_device: Dict[str, str]


class Models(_Strict):
    """Pipeline component choices plus the logical-name -> artifact registry.

    There is no ``device`` field: the device type a model runs on is derived from
    the node that hosts it (the server/sim node's ``device``), so device lives in
    one place.
    """

    detector: str = "yolo"
    tracker: str = "sort"
    predictor: str = "linear"
    model: Optional[str] = None  # active logical model name; must be a registry key
    # Warm up the model before serving the first real frame (framework feature),
    # so the first frame runs at steady-state latency instead of ~1 s first-call JIT.
    warmup: bool = True
    registry: Dict[str, ModelEntry] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_active_model_in_registry(self) -> "Models":
        if self.model is not None and self.model not in self.registry:
            raise ValueError(
                "models.model ({!r}) is not a key in models.registry".format(self.model)
            )
        return self


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------
class TransportTuning(_Strict):
    """TCP stack settings for one node, applied only in that node's namespace.

    Packet-level backends (``netem``, ``netwaggle``) let the transport react to
    the emulated link, and for small paced request/response messages that
    reaction is large: measured on a 60 ms forward link, the same traffic took
    141 ms with kernel defaults and 110 ms with Nagle off. These knobs make that
    policy explicit instead of inherited.

    Every field defaults to ``None``, meaning "leave the kernel's default
    alone". ``tcp_nodelay`` is a socket option and works anywhere;
    ``tcp_slow_start_after_idle`` is a per-namespace sysctl and ``initcwnd`` a
    route attribute, so both require the node to own a network namespace (a
    container or an emulated host) and are refused for a ``local`` node, where
    they would change the host.
    """

    # Disable Nagle, so a small write is not held waiting for an outstanding ACK.
    tcp_nodelay: Optional[bool] = None
    # Keep the congestion window across idle gaps rather than resetting it.
    tcp_slow_start_after_idle: Optional[bool] = None
    # Initial congestion window, in packets, on the route to the link peers.
    initcwnd: Optional[int] = None


class Node(_Strict):
    """A compute node that hosts a role (client/server/sim).

    ``launcher`` and the optional ``container``/``venv``/``user``/``ssh_port``
    fields capture how the node is reached; they generalise the constants in the
    external orchestrator (ssh-into-docker, ssh+docker, ssh+venv tiers).
    """

    name: str
    host: str
    # Optional: only roles that listen (server, sim) must declare a port.
    port: Optional[int] = None
    launcher: Literal["local", "docker", "ssh_docker", "ssh_venv"] = "local"
    # "leader" is an in-sim cooperating vehicle (e.g. the V2V leader): it is not a
    # network process, it only serves as an endpoint for an intent link.
    # "waypoint" is an endpoint a route passes through: with
    # `routes[*].realise: compose` nothing runs there, because a composed path
    # keeps one end-to-end connection and the hops only say where it goes. A
    # relayed route needs a process on its waypoints, so those declare the role
    # that runs it instead.
    role: Literal["client", "server", "sim", "leader", "load", "waypoint"]
    device: str = "cpu"  # device type; keys into Models.registry[*].per_device
    gpu_index: Optional[int] = None  # explicit CUDA index for this node's model (torch device)
    # server-only: request scheduling policy across concurrent clients.
    scheduler: Optional[Literal["fifo", "round_robin"]] = None
    # server-only: a fixed per-request processing cost, in milliseconds, on top
    # of the model's own time. This is compute, not network: it is applied once
    # per request by the server and is independent of any link backend.
    extra_processing_ms: Optional[float] = None
    # Server-only opt-in; the default is what the original did. How ``extra_processing_ms``
    # is applied: ``blocking`` (the original) holds the reply back with a real
    # ``time.sleep`` on the server thread, exactly as the original server did;
    # ``link`` instead leaves it to a configured reverse link (no server sleep), so
    # the delay is modelled on the wire. Default blocking reproduces the original.
    server_delay_mode: Literal["blocking", "link"] = "blocking"
    # Server-only opt-in; the default is what the original did. The original server read
    # every frame into one unbounded FIFO. True makes the inbound queue honor the
    # config's freshness policy (a pipeline stage's ``queue_policy`` or a flat
    # channel's ``max_queue``), dropping stale frames; left False to reproduce the
    # original unbounded behavior.
    enforce_queue_policy: bool = False
    # Server-only: a bounded BLOCKING inbound queue at this per-connection depth
    # (backpressure to the client), rather than dropping. None = off. This reproduces
    # the V2I RSU's original ``Queue(maxsize=N).put()`` behavior, where a full queue
    # blocks the reader and throttles the sender (which then drops newest at its camera).
    inbound_block_at: Optional[int] = None
    # load-only: concurrent synthetic clients for load experiments.
    clients: Optional[int] = None
    rate_hz: Optional[float] = None
    arrival: Optional[Literal["constant", "poisson"]] = None
    frames: Optional[int] = None
    # TCP settings for this node, applied inside its namespace only.
    transport_tuning: Optional[TransportTuning] = None
    # Shared medium this node attaches to, naming a ``connectors[*]`` entry.
    # Every node on one connector is in the same subnet and reaches the others
    # directly over it: no L3 hop, no route. Emulated backend only.
    connector: Optional[str] = None
    # ``"x,y,z"`` in metres: where this node sits in its cell. Required for a
    # node on an access point, because it is what the propagation model acts on,
    # and refused otherwise, because nothing else reads it.
    position: Optional[str] = None
    container: Optional[str] = None
    venv: Optional[str] = None
    user: Optional[str] = None
    ssh_port: Optional[int] = None
    # Private key the orchestrator authenticates with when it reaches this node
    # over SSH. A path on the machine the orchestrator runs on, and a config
    # field rather than a convention because the key for an experiment run is
    # not usually the operator's default identity, and relying on the default
    # means the choice lives in ~/.ssh/config where the run cannot record it.
    ssh_identity: Optional[str] = None

    @model_validator(mode="after")
    def _listening_roles_need_port(self) -> "Node":
        if self.role in ("server", "sim") and self.port is None:
            raise ValueError(
                "node {!r} has role {!r} and must declare a port".format(self.name, self.role)
            )
        return self


# ---------------------------------------------------------------------------
# channels (internal plumbing between nodes)
# ---------------------------------------------------------------------------
class Channel(_Strict):
    """A direct data path between two nodes (no network emulation)."""

    name: str
    from_: str = Field(alias="from")
    to: str
    stream_fps: Optional[int] = None
    max_queue: int = 2
    send_meta: bool = False
    stop_on_brake: bool = False
    device_id: Optional[str] = None


# ---------------------------------------------------------------------------
# links (emulated-network paths between nodes)
# ---------------------------------------------------------------------------
class ConstantDist(_Strict):
    """Fixed per-packet delay."""

    kind: Literal["constant"] = "constant"
    value_ms: float = 0.0


class NormalDist(_Strict):
    """Gaussian delay."""

    kind: Literal["normal"] = "normal"
    mean_ms: float
    std_ms: float


class LognormalDist(_Strict):
    """Log-normal delay (long tail), stated as two percentiles.

    A latency spec is normally given as percentiles, and these are them:
    ``p50_ms`` is the median (the distribution's scale, e^mu) and ``p99_ms``
    fixes the shape, sigma = ln(p99/p50) / 2.3263. Neither is the mean or the
    standard deviation of the random variable, which is why they are not called
    that: a log-normal's mean sits above its median, so a field named ``mean``
    holding the median is read wrong by everyone who has not read the code.
    """

    kind: Literal["lognormal"] = "lognormal"
    p50_ms: float
    p99_ms: float

    @model_validator(mode="after")
    def _check_tail_above_median(self) -> "LognormalDist":
        if self.p99_ms < self.p50_ms:
            raise ValueError(
                "lognormal p99_ms ({}) must be at least p50_ms ({})".format(
                    self.p99_ms, self.p50_ms
                )
            )
        return self


class TraceDist(_Strict):
    """Replay delays from a recorded trace file."""

    kind: Literal["trace"] = "trace"
    path: str


Distribution = Annotated[
    Union[ConstantDist, NormalDist, LognormalDist, TraceDist],
    Field(discriminator="kind"),
]


class WifiNodeSpec(_Strict):
    """One endpoint's radio role and position on a wireless link.

    ``position`` is ``"x,y,z"`` in metres. It is what the propagation model acts
    on, so leaving it unset on a link whose model depends on distance makes the
    link's behaviour arbitrary.
    """

    role: Literal["ap", "station"]
    position: Optional[str] = None


class WifiMedium(_Strict):
    """Radio parameters of one wireless medium, with no endpoints in them.

    The propagation model is the point of the wireless medium: it derives signal
    strength, and therefore association and loss, from the distance between
    positions. ``mobility_trace`` names a file of timed positions for a node
    that moves during the run.

    These are properties of the medium, not of whoever is on it, which is why
    they are separated from the endpoints: a point-to-point wireless link states
    them alongside its two radio roles (:class:`WifiConfig`), and an access-point
    connector states them for the whole cell (:class:`Connector`).
    """

    ssid: str = "cadet"
    mode: Literal["a", "b", "g", "n", "ac"] = "g"
    channel: int = 1
    propagation_model: Literal[
        "logDistance", "friis", "twoRayGround", "logNormalShadowing", "ITU"
    ] = "logDistance"
    propagation_exp: float = 3.5
    # Extra model-specific parameters (e.g. gRandom for logNormalShadowing).
    propagation_params: Dict[str, float] = Field(default_factory=dict)
    mobility_trace: Optional[str] = None


class WifiConfig(WifiMedium):
    """A point-to-point wireless link's medium plus the radio role of each end.

    This is the spelling for a link whose two endpoints are the whole cell. A
    cell with several stations on one access point is not point to point and is
    declared as a ``connectors`` entry instead; see :class:`Connector`.
    """

    # Radio role and position per endpoint, keyed by node name.
    nodes: Dict[str, WifiNodeSpec]

    @model_validator(mode="after")
    def _exactly_one_ap(self) -> "WifiConfig":
        aps = [name for name, spec in self.nodes.items() if spec.role == "ap"]
        if len(aps) != 1:
            raise ValueError(
                "wifi.nodes must declare exactly one node with role 'ap', found {}".format(
                    sorted(aps)
                )
            )
        return self


# ---------------------------------------------------------------------------
# connectors (shared media that nodes attach to)
# ---------------------------------------------------------------------------
class Connector(_Strict):
    """A shared medium that nodes attach to, and that runs nothing itself.

    A ``links`` entry is point to point: one ``from``, one ``to``, and its own
    qdiscs. A Wi-Fi cell is not that shape. Five stations associating with one
    access point share one medium, sit in **one subnet**, and reach each other
    over the air with no L3 hop and no route between them, which two
    point-to-point links in series cannot express.

    So a connector is infrastructure, deliberately *not* a ``nodes`` entry:
    there is no role to dispatch on it and nothing binds an address on it. In
    Mininet-WiFi an access point is a switch in the root namespace with no
    address of its own, so a role process placed on one has nothing to bind and
    dies with ``Cannot assign requested address``; keeping the two namespaces
    apart is what prevents that config from being expressible at all.

    ``type``
        ``ap`` is an access point: the nodes on it are stations, associating
        over a radio whose ``wifi`` parameters govern the whole cell, and each
        one's ``position`` is what the propagation model acts on. ``switch`` is
        the same idea without radios - a wired shared segment - and goes through
        the same code rather than a special case.

    Shaping is the medium's own, not a per-link qdisc: a cell's loss and delay
    come from the propagation model and the distances, which is why the
    distances are the axis such an experiment varies. A connector therefore has
    no ``distribution``, ``loss`` or ``bandwidth_mbps``; a path that needs those
    is a ``links`` entry.
    """

    name: str
    type: Literal["ap", "switch"]
    # ``"x,y,z"`` in metres. Required on an access point and refused on a
    # switch: it exists for the propagation model, and a switch has no radios.
    position: Optional[str] = None
    # Radio parameters shared by every station in this cell. Access points only.
    wifi: Optional[WifiMedium] = None

    @model_validator(mode="after")
    def _radios_belong_to_access_points(self) -> "Connector":
        if self.type == "ap":
            if self.position is None:
                raise ValueError(
                    "connector {!r} is an access point and must declare a position; "
                    "it is what the propagation model measures each station "
                    "against".format(self.name)
                )
            return self
        if self.wifi is not None:
            raise ValueError(
                "connector {!r} has type 'switch' but declares a wifi block; a "
                "switch is a wired shared segment and has no radios".format(self.name)
            )
        if self.position is not None:
            raise ValueError(
                "connector {!r} has type 'switch' but declares a position; a position "
                "is what a propagation model acts on, and a switch has no "
                "radios".format(self.name)
            )
        return self


class NetworkModel(_Strict):
    """Emulated-network behaviour applied to a link's traffic.

    ``mode`` selects the backend: ``none`` (no emulation), ``delay`` (an
    in-process delay line in the transport), ``netem`` (tc inside the sending
    node's own container, filtered to this link's flow) or ``netwaggle`` (a real
    emulated network built with Mininet-WiFi). ``medium`` applies to
    ``netwaggle`` only and chooses a wired or a wireless link.
    """

    mode: Literal["none", "delay", "netem", "netwaggle"] = "none"
    # netwaggle only: a wired link, or a wireless one governed by `wifi`.
    medium: Literal["wired", "wifi"] = "wired"
    distribution: Distribution = Field(default_factory=ConstantDist)
    loss: float = 0.0  # fraction in [0, 1]
    bandwidth_mbps: Optional[float] = None
    wifi: Optional[WifiConfig] = None

    @model_validator(mode="after")
    def _wifi_matches_medium(self) -> "NetworkModel":
        if self.medium == "wifi":
            if self.mode != "netwaggle":
                raise ValueError(
                    "network.medium: wifi requires network.mode: netwaggle "
                    "(got mode {!r}); the delay backend has no medium".format(self.mode)
                )
            if self.wifi is None:
                raise ValueError("network.medium: wifi requires a network.wifi block")
        elif self.wifi is not None:
            raise ValueError(
                "network.wifi is set but network.medium is {!r}; set medium: wifi".format(
                    self.medium
                )
            )
        return self


class Link(_Strict):
    """A network-emulated data path between two nodes.

    ``direction`` picks which path is delayed: ``forward`` (requests, from->to;
    the default, matching the originals which delayed only requests), ``reverse``
    (replies, to->from), or ``both``.
    """

    name: str
    from_: str = Field(alias="from")
    to: str
    direction: Literal["forward", "reverse", "both"] = "forward"
    network: NetworkModel = Field(default_factory=NetworkModel)


# ---------------------------------------------------------------------------
# routes (a path across several links)
# ---------------------------------------------------------------------------
class Route(_Strict):
    """A path from one node to another across several links, in order.

    ``hops`` names existing ``links`` entries, so a hop is a link and a route is
    an ordering over links: everything that already reads ``links`` - the netem
    rules, the emulated topology, the link report, the feature gate - keeps
    working on the hops without knowing a route exists.

    ``realise`` says what a multi-hop path *is*, which is not something that can
    be inferred:

    ``compose``
        One end-to-end connection whose behaviour is the composition of the
        hops. Nothing runs on the waypoints; they name where the path goes. A
        composed path cannot queue, cannot drop on a full buffer and cannot be
        where computation happens.
    ``relay``
        Each waypoint runs a forwarding stage that receives a message and sends
        it on, so a hop is a real store-and-forward step with its own queue and
        its own failure mode. This is what an experiment *about* the waypoint
        needs.

    ``via`` is optional and redundant: when given it must name exactly the
    waypoints the hops already imply, which makes a route readable without
    re-deriving it and is checked rather than trusted.
    """

    name: str
    from_: str = Field(alias="from")
    to: str
    hops: List[str]
    realise: Literal["compose", "relay"] = "compose"
    via: Optional[List[str]] = None

    @model_validator(mode="after")
    def _at_least_two_hops(self) -> "Route":
        if len(self.hops) < 2:
            raise ValueError(
                "route {!r} has {} hop(s); a route with one hop is a link, so declare "
                "the link alone".format(self.name, len(self.hops))
            )
        if len(set(self.hops)) != len(self.hops):
            raise ValueError("route {!r} names the same link twice".format(self.name))
        return self


# ---------------------------------------------------------------------------
# transport (how payloads are carried over a link)
# ---------------------------------------------------------------------------
class TransportConfig(_Strict):
    """Wire transport for node-to-node traffic.

    ``protocol`` selects the carrier: ``tcp`` is a plain socket, ``websocket``
    wraps the same framing in RFC 6455 binary frames over one HTTP upgrade.
    ``mode`` selects framing: ``frame`` sends one length-prefixed message per
    payload, ``stream`` drops the transport's prefix and lets the receiver find
    each record's end in the encoding itself (so it needs ``encoding: safe``).

    ``encoding`` selects the wire format of a payload. ``safe`` is a
    self-describing record - a JSON header plus the raw image bytes - that
    executes nothing it decodes. ``pickle`` is the original scripts' format
    (``zlib.compress(pickle.dumps(...))``); it is what the originals put on the
    wire, and it means the receiver runs code chosen by whoever can connect to
    the port, so it is an explicit opt-in rather than a default.
    """

    protocol: Literal["tcp", "websocket"] = "tcp"
    mode: Literal["frame", "stream"] = "frame"
    encoding: Literal["safe", "pickle"] = "safe"

    @model_validator(mode="after")
    def _stream_needs_a_delimited_encoding(self) -> "TransportConfig":
        """Refuse ``mode: stream`` with ``encoding: pickle``.

        Stream mode sends no length prefix and finds each record's end in the
        record itself. A pickle payload carries no length, so the combination
        cannot work at all - it is refused here rather than at the first byte of
        the first frame, where it would look like a corrupt stream.
        """
        if self.mode == "stream" and self.encoding == "pickle":
            raise ValueError(
                "transport.mode: stream needs transport.encoding: safe - a pickle "
                "payload carries no length, so a stream of them cannot be split "
                "back into messages"
            )
        return self


# ---------------------------------------------------------------------------
# metrics (which collectors run alongside a run)
# ---------------------------------------------------------------------------
class GpuMetrics(_Strict):
    """Options for the ``gpu`` collector (utilization, memory, power).

    ``interval_ms`` defaults to the original's: its power log came from
    ``nvidia-smi ... -lms 10``. ``source`` pins the reader - ``auto`` prefers
    NVML and falls back to the ``nvidia-smi`` stream the original used, and
    every row records which one produced it. ``devices`` selects device indices;
    unset means every visible device.
    """

    interval_ms: int = Field(default=ORIGINAL_SAMPLE_INTERVAL_MS, gt=0)
    source: Literal["auto", "nvml", "nvidia-smi"] = "auto"
    devices: Optional[List[int]] = None


class CpuMetrics(_Strict):
    """Options for the ``cpu`` collector (host and per-process CPU and memory).

    ``interval_ms`` is 100 ms rather than the GPU default: a CPU percentage is
    time divided by a window, so too short a window measures the sampling
    jitter. ``per_process`` records this role's own share beside the host's;
    set it false for the host scope only.
    """

    interval_ms: int = Field(default=COUNTER_SAMPLE_INTERVAL_MS, gt=0)
    per_process: bool = True


class EnergyMetrics(_Strict):
    """Options for the ``energy`` collector (power over the run, per inference).

    ``interval_ms`` unset takes **the chosen source's own original cadence**:
    10 ms for the GPU and Jetson readers, which is what every one of those
    originals sampled at, and 100 ms for ``rpi_proxy``, which is what its
    original used. Setting it pins one value for whatever source is chosen.

    ``source`` pins the power source; ``auto`` tries NVML, the ``nvidia-smi``
    stream, ``tegrastats``, the Jetson INA channels in sysfs, then x86 RAPL, and
    refuses if none can be read.

    The two Jetson sources are **ports** of the project's own
    ``tegrastats_logger.py`` and ``jetson_power_logger.py`` - rails, units,
    cadence and failure behaviour taken from those scripts - but they are
    **verified on hardware: no**: no Jetson is reachable from this environment.
    They also meter **different rail sets** (``tegrastats`` reads the GPU and
    CPU rails; sysfs reads GPU, CPU and SOC), so they are not interchangeable.

    ``rpi_proxy`` is a port of the method in ``rpi_yolo.py`` and **is not a
    power measurement**: it is a weighted sum of a CPU percentage, a memory
    percentage and an undifferenced cumulative disk-byte counter. It is recorded
    in its own ``activity_index`` column and never as watts or joules, and
    ``auto`` will not choose it.

    ``tegrastats_sudo`` keeps the ``sudo`` the original ran ``tegrastats``
    under; set it false where the role is already root or there is no ``sudo``.
    """

    interval_ms: Optional[int] = Field(default=None, gt=0)
    source: Literal[
        "auto", "nvml", "nvidia-smi", "tegrastats", "tegra_sysfs", "rapl", "rpi_proxy"
    ] = "auto"
    devices: Optional[List[int]] = None
    tegrastats_sudo: bool = True


class BandwidthMetrics(_Strict):
    """Options for the ``bandwidth`` collector (bytes per link and direction).

    ``interval_ms`` is 100 ms for the same reason as the CPU collector: a byte
    counter only becomes a rate over a window.
    """

    interval_ms: int = Field(default=COUNTER_SAMPLE_INTERVAL_MS, gt=0)


class ClockOffsetMetrics(_Strict):
    """Options for the ``clock_offset`` collector (per-node clock difference).

    Unlike the sampling collectors this one is measured by the orchestrator,
    once per run per node, and recorded in ``run.json`` rather than in a CSV:
    it describes the run's timestamps rather than being one of them.

    ``method`` is ``auto`` by default, which estimates the offset from round
    trips the orchestrator makes itself and records each machine's own time
    daemon beside it as an independent cross-check. ``round_trip`` records
    only the former; ``time_service`` uses the daemons as the estimate and
    refuses the run if either machine runs none, rather than falling back to a
    different method than the one that was asked for.
    """

    method: Literal["auto", "round_trip", "time_service"] = "auto"
    probes: int = Field(default=CLOCK_PROBE_COUNT, gt=0)
    timeout_s: float = Field(default=CLOCK_PROBE_TIMEOUT_S, gt=0)


class MetricsConfig(_Strict):
    """Optional measurement collectors to run during an experiment.

    Per-frame latency (model-level) and per-request server queue metrics are
    always recorded. The ``collectors`` list enables additional system/task
    collectors, each of which runs on the node it measures and writes
    ``<collector>_<node>.csv`` into the run directory.

    ``clock_offset`` is the one exception to that shape: a clock difference is
    a property of a pair of machines rather than of a node, so the orchestrator
    measures it and writes it, with its uncertainty, into the ``clock_offset``
    block of ``run.json``. It corrects nothing - see
    :mod:`cadet.metrics.clock_offset`.

    **The default is an empty list**, so an unconfigured run starts no sampling
    threads, opens no readers and writes no extra files - it behaves exactly as
    it did before the collectors existed. That default is pinned in
    ``tests/unit/test_collectors.py``.
    """

    collectors: List[Literal["gpu", "cpu", "bandwidth", "queue", "energy", "clock_offset"]] = Field(
        default_factory=list
    )
    gpu: GpuMetrics = Field(default_factory=GpuMetrics)
    cpu: CpuMetrics = Field(default_factory=CpuMetrics)
    energy: EnergyMetrics = Field(default_factory=EnergyMetrics)
    bandwidth: BandwidthMetrics = Field(default_factory=BandwidthMetrics)
    clock_offset: ClockOffsetMetrics = Field(default_factory=ClockOffsetMetrics)


# ---------------------------------------------------------------------------
# pipeline (staged form of an application)
# ---------------------------------------------------------------------------
class DeadlineSpec(_Strict):
    """A per-stage time-to-decision budget (ported deadline-enforcement semantics).

    ``none`` runs the stage at full fidelity; ``static`` hands it a fixed
    ``budget_ms`` every frame; ``dynamic`` hands it the time *remaining* until the
    end-to-end ``budget_ms``. A stage reads the budget only if it adapts its work.
    """

    policy: Literal["none", "static", "dynamic"] = "none"
    budget_ms: Optional[float] = None

    @model_validator(mode="after")
    def _budget_required_when_timed(self) -> "DeadlineSpec":
        if self.policy in ("static", "dynamic") and self.budget_ms is None:
            raise ValueError("deadline policy {!r} requires budget_ms".format(self.policy))
        return self


class StageSpec(_Strict):
    """One placed stage: an impl from the registry assigned to a node."""

    name: str  # unique instance name within the pipeline
    impl: str  # stage registry key (validated against the registry at load time)
    node: str  # must name a node in ``Config.nodes``
    params: Dict[str, Any] = Field(default_factory=dict)
    # Freshness policy for the queue feeding this stage (esp. across a node boundary).
    queue_policy: Literal["latest_only", "fifo"] = "latest_only"
    deadline: Optional[DeadlineSpec] = None


class PipelineConfig(_Strict):
    """The ordered list of stages that make up a staged application."""

    stages: List[StageSpec] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# sweep + outputs
# ---------------------------------------------------------------------------
class Sweep(_Strict):
    """Cartesian sweep over dotted config paths -> list of values."""

    axes: Dict[str, List[Any]] = Field(default_factory=dict)


class Outputs(_Strict):
    """Where run artifacts go. By policy, results/weights are never committed."""

    root: str = "outputs"
    naming: str = "{experiment}/{app}/{run_id}"
    save: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# analysis (the foundation-model interface over a run's own records)
# ---------------------------------------------------------------------------
class AnalysisConfig(_Strict):
    """Where ``cadet summarize`` / ``cadet ask`` send a run's reduced record.

    The analysis layer ends in files; this section says what may read them back
    and answer questions about them. It configures no part of a run: nothing in
    the orchestrator, the roles or the transport reads it, and a run never calls
    a model.

    ``backend: none`` is the default, so an unconfigured CADET has **no backend
    configured** and both subcommands refuse with a message naming what to set.
    ``openai_chat`` is any OpenAI-compatible ``/chat/completions`` endpoint, or a
    local model served behind the same route shape; the call is one HTTP POST,
    so it adds no dependency.

    **There is no field for an API key, and there will not be one.**
    ``api_key_env`` names an environment variable, and the key is read from the
    environment at the moment of a request. A key in a config would be copied
    into every run directory's resolved ``config.yaml``, which is the one place
    it must never be. Set it empty for an endpoint that takes no key, such as a
    model served on the machine the command runs on.
    """

    backend: Literal["none", "openai_chat"] = "none"
    # The chat-completions URL in full (e.g. ".../v1/chat/completions"), and the
    # model name that endpoint serves. ``openai_chat`` requires both.
    endpoint: Optional[str] = None
    model: Optional[str] = None
    # The environment variable the key is read from - the name, never the key.
    # Empty says the endpoint takes no key, which is the state a model served
    # on the machine running the command is in: then no Authorization header is
    # sent, rather than one carrying a placeholder.
    api_key_env: str = "CADET_FM_API_KEY"
    timeout_s: float = Field(default=60.0, gt=0)
    # A bound on the record sent, in bytes of UTF-8. Parts of the record are
    # dropped largest first to meet it, and the text says which (see
    # cadet.analysis.context.render_context). The default fits a sweep of a few
    # dozen runs at full detail.
    max_context_bytes: int = Field(default=200000, gt=0)
    # Zero by default: two summaries of one run should differ only if the run
    # does.
    temperature: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _backend_needs_an_endpoint(self) -> "AnalysisConfig":
        """A backend with nowhere to send is a misconfiguration, not a default.

        Checked here rather than at the request, because ``cadet validate`` is
        where a config is meant to be told it is wrong.
        """
        if self.backend != "none" and not (self.endpoint and self.model):
            raise ValueError(
                "analysis.backend {!r} requires analysis.endpoint and "
                "analysis.model".format(self.backend)
            )
        return self


# ---------------------------------------------------------------------------
# top-level config
# ---------------------------------------------------------------------------
class App(_Strict):
    """App selector + its raw parameters.

    The framework is app-agnostic: ``params`` is a free mapping here and is
    validated against the registered plugin's typed model separately (see
    :func:`cadet.config.loader.resolve_app_params`), so ``cadet/`` never imports
    app-specific param models.
    """

    plugin: str
    params: Dict[str, Any] = Field(default_factory=dict)


class Config(_Strict):
    """A complete CADET experiment configuration."""

    experiment: Experiment
    source: Source
    models: Models = Field(default_factory=Models)
    nodes: List[Node]
    channels: List[Channel] = Field(default_factory=list)
    # Shared media nodes attach to. Absent means every path is point to point,
    # which is how every run worked before this section existed.
    connectors: List[Connector] = Field(default_factory=list)
    links: List[Link] = Field(default_factory=list)
    # Multi-hop paths over those links. Absent means every link is a direct
    # path, which is how every run worked before this section existed.
    routes: List[Route] = Field(default_factory=list)
    transport: TransportConfig = Field(default_factory=TransportConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    app: App
    # Optional staged form: when present, the app runs as a placed stage pipeline.
    pipeline: Optional[PipelineConfig] = None
    sweep: Sweep = Field(default_factory=Sweep)
    outputs: Outputs = Field(default_factory=Outputs)
    # Reads a finished run tree and answers questions about it; configures no
    # part of a run. Default: no backend configured.
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)

    @model_validator(mode="after")
    def _check_app_matches_experiment(self) -> "Config":
        if self.experiment.app != self.app.plugin:
            raise ValueError(
                "experiment.app ({!r}) must match app.plugin ({!r})".format(
                    self.experiment.app, self.app.plugin
                )
            )
        return self

    @model_validator(mode="after")
    def _check_endpoints_exist(self) -> "Config":
        node_names = {n.name for n in self.nodes}
        for ch in self.channels:
            for endpoint in (ch.from_, ch.to):
                if endpoint not in node_names:
                    raise ValueError(
                        "channel {!r} references unknown node {!r}".format(ch.name, endpoint)
                    )
        for lk in self.links:
            for endpoint in (lk.from_, lk.to):
                if endpoint not in node_names:
                    raise ValueError(
                        "link {!r} references unknown node {!r}".format(lk.name, endpoint)
                    )
        return self

    @model_validator(mode="after")
    def _check_connectors(self) -> "Config":
        """Check the shared media: names, attachments, and one access point per cell.

        A connector runs nothing and a node runs something, so they are two
        namespaces rather than one list with a flag. Every refusal here is a way
        those two namespaces can be confused for each other, and each is refused
        by name at validate time rather than discovered when the emulator fails
        to bring a radio up.
        """
        node_names = {n.name for n in self.nodes}
        by_connector: Dict[str, Connector] = {}
        for conn in self.connectors:
            if conn.name in by_connector:
                raise ValueError("duplicate connector name {!r}".format(conn.name))
            if conn.name in node_names:
                raise ValueError(
                    "connector {!r} has the same name as a node; a connector is "
                    "infrastructure that runs nothing, so the two cannot share a "
                    "name".format(conn.name)
                )
            by_connector[conn.name] = conn

        # Radio roles declared the point-to-point way, so a node cannot be given
        # a second radio by the other spelling.
        link_roles: Dict[str, Tuple[str, str]] = {}
        for lk in self.links:
            wifi = getattr(lk.network, "wifi", None)
            if wifi is None:
                continue
            for name, spec in wifi.nodes.items():
                link_roles.setdefault(name, (lk.name, spec.role))

        for node in self.nodes:
            if node.connector is None:
                if node.position is not None:
                    raise ValueError(
                        "node {!r} declares a position but attaches to no connector; "
                        "a position is what a propagation model acts on, so only a "
                        "station on an access point has one".format(node.name)
                    )
                continue
            conn = by_connector.get(node.connector)
            if conn is None:
                raise ValueError(
                    "node {!r} attaches to unknown connector {!r}".format(
                        node.name, node.connector
                    )
                )
            if conn.type == "switch":
                if node.position is not None:
                    raise ValueError(
                        "node {!r} attaches to switch connector {!r} and declares a "
                        "position; a wired segment has no propagation model to read "
                        "it".format(node.name, conn.name)
                    )
                continue
            if node.position is None:
                raise ValueError(
                    "node {!r} is a station on access point {!r} and must declare a "
                    "position; without one the propagation model has no distance to "
                    "act on and the cell's behaviour is arbitrary".format(
                        node.name, conn.name
                    )
                )
            declared = link_roles.get(node.name)
            if declared is None:
                continue
            link_name, role = declared
            if role == "ap":
                raise ValueError(
                    "node {!r} is a station on access point {!r} and is also the "
                    "access point of link {!r}; a cell has one access point, and it "
                    "is the connector".format(node.name, conn.name, link_name)
                )
            raise ValueError(
                "node {!r} attaches to access point {!r} and also declares a wifi "
                "role on link {!r}; a node has one radio, so declare the cell or the "
                "point-to-point link, not both".format(node.name, conn.name, link_name)
            )
        return self

    @model_validator(mode="after")
    def _check_routes(self) -> "Config":
        """Check that each route is a path: its hops exist, chain and are unique.

        A route whose hops do not chain is a list of unrelated links that
        happens to validate, and a route beside a direct link between the same
        endpoints gives a role's path lookup two answers, so both are refused
        here rather than discovered as a wrong measurement.
        """
        if not self.routes:
            return self
        node_names = {n.name for n in self.nodes}
        by_link_name = {lk.name: lk for lk in self.links}
        seen_names = set()
        owner_of_hop = {}

        for route in self.routes:
            if route.name in seen_names:
                raise ValueError("duplicate route name {!r}".format(route.name))
            seen_names.add(route.name)
            for endpoint in (route.from_, route.to):
                if endpoint not in node_names:
                    raise ValueError(
                        "route {!r} references unknown node {!r}".format(route.name, endpoint)
                    )

            hops = []
            for hop_name in route.hops:
                hop = by_link_name.get(hop_name)
                if hop is None:
                    raise ValueError(
                        "route {!r} names unknown link {!r}".format(route.name, hop_name)
                    )
                previous = owner_of_hop.get(hop_name)
                if previous is not None:
                    raise ValueError(
                        "link {!r} is a hop of both route {!r} and route {!r}; a link "
                        "belongs to one path".format(hop_name, previous, route.name)
                    )
                owner_of_hop[hop_name] = route.name
                hops.append(hop)

            if hops[0].from_ != route.from_:
                raise ValueError(
                    "route {!r} starts at {!r} but its first hop {!r} starts at {!r}".format(
                        route.name, route.from_, hops[0].name, hops[0].from_
                    )
                )
            if hops[-1].to != route.to:
                raise ValueError(
                    "route {!r} ends at {!r} but its last hop {!r} ends at {!r}".format(
                        route.name, route.to, hops[-1].name, hops[-1].to
                    )
                )
            for first, second in zip(hops, hops[1:]):
                if first.to != second.from_:
                    raise ValueError(
                        "route {!r} does not chain: hop {!r} ends at {!r} but hop {!r} "
                        "starts at {!r}".format(
                            route.name, first.name, first.to, second.name, second.from_
                        )
                    )

            waypoints = [hop.to for hop in hops[:-1]]
            if route.via is not None and route.via != waypoints:
                raise ValueError(
                    "route {!r} declares via {} but its hops pass through {}".format(
                        route.name, route.via, waypoints
                    )
                )

            modes = {hop.network.mode for hop in hops}
            if len(modes) > 1:
                raise ValueError(
                    "route {!r} mixes link modes {}; one path is emulated by one "
                    "backend".format(route.name, sorted(modes))
                )

            if route.realise == "relay":
                by_node = {n.name: n for n in self.nodes}
                for waypoint in waypoints:
                    if by_node[waypoint].port is None:
                        raise ValueError(
                            "route {!r} is relayed through {!r}, which declares no port; "
                            "a relay listens".format(route.name, waypoint)
                        )

            for link in self.links:
                if link.name in route.hops:
                    continue
                if (link.from_, link.to) == (route.from_, route.to):
                    raise ValueError(
                        "route {!r} and link {!r} both connect {!r} and {!r}; a path "
                        "lookup would have two answers".format(
                            route.name, link.name, route.from_, route.to
                        )
                    )
        return self

    @model_validator(mode="after")
    def _check_pipeline_nodes(self) -> "Config":
        """Each pipeline stage must name an existing node and a unique stage name.

        The ``impl`` is checked separately against the stage registry once stage
        modules are imported (see :func:`cadet.pipeline.validate.validate_pipeline`),
        so the schema stays free of stage-implementation imports.
        """
        if self.pipeline is None:
            return self
        node_names = {n.name for n in self.nodes}
        seen = set()
        for st in self.pipeline.stages:
            if st.node not in node_names:
                raise ValueError(
                    "pipeline stage {!r} references unknown node {!r}".format(st.name, st.node)
                )
            if st.name in seen:
                raise ValueError("duplicate pipeline stage name {!r}".format(st.name))
            seen.add(st.name)
        return self
