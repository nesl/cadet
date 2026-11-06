"""Orchestrator: turn a config into running processes and back again, cleanly.

This module owns the lifecycle of a run and a sweep. For a run it clears its
own leftovers, runs preflight, starts the servers and waits for readiness,
starts the sim-side hooks and waits for their marker, starts clients/load,
waits for the run to end by its ``experiment.end_on`` rule, then tears
everything down in reverse order so no process is orphaned. For a sweep it
enumerates the matrix and runs each cell; with ``experiment.resume`` (or
``--resume``) it skips cells whose output is already complete, and by default
it reruns them. An interrupted run is part of that lifecycle, not an exception
to it: SIGTERM and SIGHUP are turned into an exception
(:func:`cadet.orchestrator.lifecycle.interrupt_guard`) so the teardown runs,
and whatever a harder kill leaves behind is removed by ``cadet down``
(:func:`cadet.orchestrator.recovery.sweep`). It provides
:class:`PreflightError`, :func:`preflight`, :class:`RunResult` and
:class:`Orchestrator`.

It is called by ``cadet run`` / ``cadet sweep`` in :mod:`cadet.cli`, and spawns
the role entry points (``cadet serve`` / ``cadet client`` / ``cadet load``)
through the launcher each node's config asks for: a host process, a container
on this machine, or (in dry-run) a container on another machine.
:mod:`cadet.orchestrator.placement` owns that dispatch, the per-run docker
network and the per-node containers; this module keeps the ordering and the
teardown, which do not change with where a role runs. Sim hooks come from the
app plugin's :meth:`AppPlugin.sim_hooks`.

It also records what a cross-node timestamp is worth: with
``metrics.collectors: clock_offset`` it measures each node's clock against its
own before the run and again after the roles have stopped, and writes the
offset and its uncertainty into ``run.json``
(:mod:`cadet.metrics.clock_offset`). Nothing is corrected by it.

It also owns the lifetime of each link's network backend, installing netem
qdiscs before any role starts and removing them on teardown so a failed run
cannot leave traffic shaped. For a containerised node the qdisc goes inside
that node's own namespace, through a runner the placement supplies, because the
interface to shape is the container's and not the host's. The ``delay`` backend
needs nothing here (it lives inside the transport), and ``netwaggle`` is driven
by ``cadet emulate`` inside the emulation container.

Driven entirely by the experiment config (``nodes``, ``nodes.launcher``,
``outputs``) plus ``--app-module`` / ``--model-dir`` passed to role processes.
"""

import json
import os
import sys
import time
import traceback
from typing import List, Optional

from cadet.config.loader import expand_sweep, load_config_data
from cadet.core.apps import get_app, import_app_modules
from cadet.core.launcher import (
    LaunchSpec,
    LocalLauncher,
    role_executable,
    role_python_version,
)
from cadet.core.model_registry import ModelRegistry, device_for_models
from cadet.core.venv_launcher import SshVenvLauncher
from cadet.metrics import clock_offset, device_summary
from cadet.network import netem, throughput
from cadet.orchestrator import lifecycle
from cadet.orchestrator import placement as placement_mod
from cadet.orchestrator.placement import Placement, PlacementError
from cadet.orchestrator.state import StateStore
from cadet.roles.client import (
    CLIENT_READY_MARKER,
    CLIENT_START_AFTER_SCENARIO,
    CLIENT_START_DEFAULT,
)
from cadet.roles.common import output_dir, select_run_config
from cadet.sim.containers import SimSide, carla_launch_settings
from cadet.sim.scenario_launcher import scenario_launcher_for

# Substring the server prints once it is listening (see cadet/roles/server.py).
SERVER_READY_MARKER = "Waiting for connection"
# How long to wait for a client to report ready under the default ordering. It
# covers the client connecting to the simulator and to its server.
CLIENT_READINESS_TIMEOUT_S = 120.0
# How long to wait for that line. Named here rather than left to the LaunchSpec
# default because the startup limit defaults to the sum of these.
SERVER_READINESS_TIMEOUT_S = 60.0
# Marker file written in a run's output dir once the run completed successfully.
RUN_MARKER = "run.json"
# Written beside a sweep's run directories while the sweep runs, and finalised
# when it ends. A run directory says whether *that run* is finished; nothing
# said whether a sweep was still going, so a reader watching one saw an idle
# tree during the pause between cells - every run complete, more to come, and no
# way to tell that from a finished sweep.
SWEEP_MARKER = "sweep.json"
# The collector name that asks for a per-node clock offset in run.json. It is
# in metrics.collectors with the samplers because it is a measurement a run
# opts into, but it is measured here rather than inside a role: the difference
# between two machines' clocks is not a property either role can see.
CLOCK_OFFSET_COLLECTOR = "clock_offset"


class PreflightError(Exception):
    """Raised when a run cannot safely start (bad model, port in use, ...)."""


def preflight(
    config,
    model_dirs: List[str],
    log=print,
    check_ports: bool = True,
    allow_experimental: Optional[bool] = None,
) -> None:
    """Check a run can start: model artifacts resolve and listening ports are free.

    Raises :class:`PreflightError` with a readable message on the first problem.
    Throughput is a warning rather than a problem: a saturated link is a valid
    experiment, and only the experimenter can say whether this one is intended.

    ``check_ports`` is False for the emulated backend, whose servers bind inside
    their own network namespaces: a port busy on the container's localhost says
    nothing about the port an emulated host will bind, and refusing on it would
    reject runs that are fine.

    ``allow_experimental`` accepts config options that are implemented but
    unverified; None defers to ``experiment.allow_experimental``.
    """
    # The implemented-feature gate. It used to run only in `cadet validate`,
    # which nothing obliges a caller to run first, so `cadet run` would start a
    # config asking for features that do not exist and fail later and less
    # clearly. Checking it here covers run, sweep and emulate at once.
    from cadet.config.implemented import (
        allow_experimental as _resolve_allow,
    )
    from cadet.config.implemented import (
        experimental_features,
        unimplemented_features,
    )

    allowed = _resolve_allow(config, allow_experimental)
    missing = unimplemented_features(config, allow_experimental=allowed)
    if missing:
        raise PreflightError(
            "config asks for features this build will not run:\n  - "
            + "\n  - ".join(missing)
        )
    if allowed:
        asked = experimental_features(config)
        if asked:
            log(
                "[orch] running with EXPERIMENTAL features accepted: {}".format(
                    ", ".join(asked)
                )
            )
    # Model artifact must resolve if the config names an active model.
    if config.models.model is not None:
        device = device_for_models(config)
        registry = ModelRegistry.from_config(config.models, device)
        res = registry.resolve(config.models.model, model_dirs)
        if res.path is None:
            raise PreflightError(
                "model artifact {!r} for {!r} not found under {}".format(
                    res.artifact, config.models.model, model_dirs
                )
            )

    # A sysctl or route tuning needs a namespace of its own. Under the local
    # launcher there is none, so applying it would change the host — refuse
    # rather than silently skip it or silently widen its scope.
    from cadet.network.tuning import requires_namespace

    # `ssh_venv` is refused for the same reason and is worse: the role is a
    # plain process on *another* machine, so there is no namespace to put the
    # setting in and applying it would change that machine for everything else
    # running on it.
    for node in config.nodes:
        if node.launcher in ("local", "ssh_venv") and requires_namespace(node):
            raise PreflightError(
                "node {!r} sets transport tuning that needs its own network "
                "namespace (tcp_slow_start_after_idle / initcwnd), but its "
                "launcher is {!r}, which gives it none. Use launcher: docker, "
                "or the netwaggle backend, or set only tcp_nodelay.".format(
                    node.name, node.launcher
                )
            )

    # A netem qdisc goes inside the sending node's own network namespace, and
    # the orchestrator has a handle only on a namespace it created: a container
    # on this machine, or an emulated host. A node on another machine has none -
    # its interface is that machine's - so a direction it sends is shaped where
    # it arrives, on the interface of the node at the other end of the link.
    #
    # That requires the other end to be a node whose namespace this orchestrator
    # does own. A link with a node on another machine at both ends, or one whose
    # near end runs as a bare process on this host (where the interface facing
    # the far machine is this machine's own), has nowhere safe to put the qdisc,
    # and is refused rather than shaping a machine's NIC.
    off_machine = ("ssh_docker", "ssh_venv")
    by_name = {node.name: node for node in config.nodes}
    nowhere_to_shape = []
    for link in getattr(config, "links", None) or []:
        if link.network.mode != "netem":
            continue
        direction = getattr(link, "direction", "forward")
        # Only a direction that is actually shaped matters, and only the end
        # that *sends* it: a remote node at the other end of an unshaped
        # direction sends nothing this has to place.
        for wanted, sender, receiver in (
            ("forward", link.from_, link.to),
            ("reverse", link.to, link.from_),
        ):
            if direction not in (wanted, "both"):
                continue
            sender_node, receiver_node = by_name.get(sender), by_name.get(receiver)
            if sender_node is None or receiver_node is None:
                continue
            if getattr(sender_node, "launcher", "local") not in off_machine:
                continue
            if getattr(receiver_node, "launcher", "local") != "docker":
                nowhere_to_shape.append((link.name, receiver, sender))
    if nowhere_to_shape:
        raise PreflightError(
            "link(s) {} shape traffic from a node on another machine, which is "
            "shaped where it arrives - and the node it arrives at must run in a "
            "container on this machine for the qdisc to have an interface this "
            "orchestrator owns. Set launcher: docker on {}, or use "
            "links[*].network.mode: delay, which needs no namespace.".format(
                ", ".join(repr(name) for name, _near, _far in nowhere_to_shape),
                ", ".join(sorted({repr(near) for _n, near, _f in nowhere_to_shape})),
            )
        )

    # Listening nodes (server/sim) on the local host must have a free port. The
    # cleanup above has already killed any CADET process holding one, so a port
    # still in use here belongs to something CADET did not start. Name it: the
    # alternative, clearing it anyway, is what the original did and it kills
    # whatever happens to be listening.
    for node in (config.nodes if check_ports else ()):
        if node.role in ("server", "sim") and node.port is not None:
            if node.host in ("localhost", "127.0.0.1", "0.0.0.0", ""):
                bind_host = "127.0.0.1" if node.host in ("localhost", "") else node.host
                if not lifecycle.port_free(node.port, bind_host):
                    raise PreflightError(
                        "port {} for node {!r} is held by {}. CADET did not start it, "
                        "so it was left alone; stop it or move the node to another "
                        "port.".format(
                            node.port, node.name, lifecycle.describe_port_holders(node.port)
                        )
                    )

    # The simulator's own ports, when this run is the one launching it. Docker
    # refuses the bind itself, but it does so from inside `docker run` and names
    # neither the node nor the holder - and the holder is the useful half, since
    # it is usually a hand-started server or another session's run. CARLA takes
    # the RPC port and the two above it, and all three are published, so all
    # three are checked.
    if check_ports:
        from cadet.sim.containers import CARLA_PORT_SPAN, carla_launch_settings

        launch = carla_launch_settings(config)
        if launch is not None:
            for offset in range(CARLA_PORT_SPAN):
                port = launch.port + offset
                if not lifecycle.port_free(port, "127.0.0.1"):
                    raise PreflightError(
                        "port {} is held by {}, and this run launches its own "
                        "simulator on {}-{}. CADET did not start the holder, so "
                        "it was left alone; stop it or set source.carla.port.".format(
                            port,
                            lifecycle.describe_port_holders(port),
                            launch.port,
                            launch.port + CARLA_PORT_SPAN - 1,
                        )
                    )

    # The simulator-side pieces must agree with each other. A refusal rather
    # than a warning: unlike an over-subscribed link, there is no experiment for
    # which running a mismatched client against a server is the intent.
    from cadet.sim import versions

    # The client egg is imported by the role that opens the simulator, so the
    # interpreter to ask about is that role's and not this process's. An
    # orchestrator running on a host interpreter while its client runs in the
    # image is a correct arrangement, and asking about itself refused it. The
    # checks that do not depend on the interpreter return the same line for
    # every placement, so a line is reported once however many placements agree
    # on it.
    client_pythons = sorted(
        {
            role_python_version(node.launcher)
            for node in config.nodes
            if node.role == "client"
        }
    ) or [None]
    mismatches = []
    for python in client_pythons:
        for problem in versions.check_versions(config, interpreter=python):
            if problem not in mismatches:
                mismatches.append(problem)
    if mismatches:
        raise PreflightError(
            "simulator version mismatch:\n  - " + "\n  - ".join(mismatches)
        )

    # What the app will offer each link, against what the link can carry. This
    # is the check whose absence turned an over-subscribed braking link into a
    # run that simply never finished.
    for warning in throughput.throughput_warnings(config):
        log("[orch] WARNING: {}".format(warning))


class RunResult:
    """Outcome of a single run."""

    def __init__(self, run_id: str, out_dir: str, ok: bool, detail: str = "", reason: str = ""):
        self.run_id = run_id
        self.out_dir = out_dir
        self.ok = ok
        self.detail = detail
        # Why the run ended; see cadet.orchestrator.lifecycle REASON_* values.
        self.reason = reason


def _stage_modules() -> List[str]:
    """The pipeline stage modules this process was told to import.

    Read from the environment, which is where ``run``/``sweep`` publish what
    ``--stage-module`` named, so a role's argv can carry it to a process that
    inherits no environment.
    """
    value = os.environ.get("CADET_STAGE_MODULES", "")
    return [part for part in value.split(os.pathsep) if part]


class Orchestrator:
    """Starts and supervises the processes of a run or sweep (local launcher)."""

    def __init__(self, launcher=None, log=print):
        # Kept for callers and tests that inject a launcher: it overrides the
        # dispatch for every node, which is what a test harness wants. Normal
        # runs leave it None and each node gets the launcher its config asks
        # for (see cadet.orchestrator.placement).
        self.launcher = launcher
        self._local_launcher = launcher or LocalLauncher()
        self.log = log
        self.state = StateStore()

    # -- helpers ---------------------------------------------------------------
    def _base_argv(
        self,
        sub: str,
        config_path: str,
        overrides: List[str],
        run_index: int,
        app_modules: List[str],
        executable: Optional[str] = None,
    ) -> List[str]:
        """Return the argv that runs one role subcommand.

        ``executable`` is the interpreter the role will run under, which is a
        property of where it is placed rather than of this process - see
        :func:`cadet.core.launcher.role_executable`. ``None`` keeps this
        process's own, which is what a local placement gets anyway.
        """
        argv = [executable or sys.executable, "-m", "cadet", sub, config_path]
        for mod in app_modules or []:
            argv += ["--app-module", mod]
        # The stage modules travel on the argv for the same reason the app
        # modules do. They used to travel only through CADET_STAGE_MODULES,
        # which a role inherits when it is a local child of this process and
        # does not when it runs in a container: a containerised pipeline role
        # then started with an empty registry and died on the first stage its
        # config named. Measured on an acceptance cell: `KeyError: 'camera'`
        # from a server whose config named four registered stages.
        for mod in _stage_modules():
            argv += ["--stage-module", mod]
        for ov in overrides or []:
            argv += ["--set", ov]
        argv += ["--run", str(run_index)]
        return argv

    def _role_config(self, placement, node, config_path: str, run_index: int):
        """Return the ``(config_path, run_index)`` for one node's role process.

        A node whose machine has no copy of the repository is handed the run's
        resolved config, staged there by the placement, and index 0 with it: the
        staged file is this one run with the sweep already expanded, so there is
        nothing left to index. Every other node keeps the path and the index the
        orchestrator was given.
        """
        staged = placement.role_config_path(node)
        if staged is None:
            return config_path, run_index
        return staged, 0

    def is_complete(self, config, run_id: str) -> bool:
        """True if the run's output dir already has a success marker (resume)."""
        marker = os.path.join(output_dir(config, run_id), RUN_MARKER)
        if not os.path.exists(marker):
            return False
        try:
            with open(marker) as fh:
                return json.load(fh).get("status") == "complete"
        except (OSError, ValueError):
            return False

    def _clock_sources(self, config, placement):
        """Return node -> clock source, or None for a node on this machine.

        A node reached with ``ssh_docker`` runs in a container that joins the
        other machine's network namespace and so its kernel, which is why the
        offset of that machine is the offset of the node. A node reached with
        ``ssh_venv`` is a process on that machine and so reads that kernel
        directly, which is the same answer by a shorter route. Everything else
        runs on this machine's clock and is recorded as such without being
        probed.
        """
        remote = {node.name for node in placement.offmachine_nodes()}
        sources = {}
        timeout_s = config.metrics.clock_offset.timeout_s
        for node in config.nodes:
            source = None
            if node.name in remote:
                try:
                    launcher = placement.launcher_for(node)
                    # The responder runs under an interpreter on that machine,
                    # and a venv node names one that is certain to be there -
                    # which a bare `python3` is not, on a machine whose Python
                    # lives in the venv and nowhere else. That is a plausible
                    # machine rather than a contrived one: it is part of why a
                    # node would be placed with `ssh_venv` at all.
                    source = clock_offset.SshClockSource(
                        launcher,
                        timeout_s=timeout_s,
                        **(
                            {"python": launcher.python()}
                            if isinstance(launcher, SshVenvLauncher)
                            else {}
                        ),
                    )
                except PlacementError as exc:
                    self.log("[clock] {}: {}".format(node.name, exc))
            sources[node.name] = source
        return sources

    def _measure_clock_offsets(self, config, placement, phase):
        """Measure every node's clock offset, or return None if unasked.

        Never raises: the offset describes a run's timestamps, and failing to
        measure it must not fail the run that produced them. A node that could
        not be measured is recorded as unmeasured instead.
        """
        if CLOCK_OFFSET_COLLECTOR not in (config.metrics.collectors or []):
            return None
        options = config.metrics.clock_offset
        try:
            measured = clock_offset.measure_nodes(
                self._clock_sources(config, placement),
                method=options.method,
                probes=options.probes,
                log=self.log,
            )
        except Exception as exc:  # defensive: see the docstring
            self.log("[clock] {} measurement failed: {}".format(phase, exc))
            return None
        for name, estimate in sorted(measured.items()):
            self.log(
                "[clock] {} {}: {:+.6f} s +/- {:.6f} s ({})".format(
                    phase, name, estimate.offset_s, estimate.uncertainty_s, estimate.method
                )
            )
        return measured

    # -- single run ------------------------------------------------------------
    def run_one(
        self,
        config_path: str,
        overrides: List[str],
        run_index: int,
        rep: int,
        app_modules: List[str],
        model_dirs: List[str],
        allow_experimental: Optional[bool] = None,
    ) -> RunResult:
        """Run one (run_index, rep) cell end-to-end, tearing everything down after."""
        # Register app plugins in THIS process too, so sim_hooks() can be read here
        # (the role subprocesses register them again via --app-module).
        import_app_modules(app_modules or [])
        # The startup phase is timed from here: the cleanup and preflight below
        # are part of what max_startup_s bounds, not free time before it.
        started_at = time.monotonic()
        timing = lifecycle.RunTiming()
        config, run_id_base = select_run_config(config_path, overrides, run_index)
        run_id = "{}-r{}".format(run_id_base, rep)
        out_dir = output_dir(config, run_id)
        log_dir = os.path.join(out_dir, "logs")
        # Not created here. Placing containerised nodes rewrites their addresses,
        # which are part of the config hash, so the run id this derived can still
        # change - and creating the directory first left an empty one behind
        # under the old id, holding a `logs` folder and nothing else, for every
        # run of every config that places a container. The paths are needed now
        # (the resume check reads the marker), the directories are made once the
        # id is final.

        # The app's sim-side hooks are read before anything starts: their
        # programs name the orphans to clear, and they are launched further down
        # once the servers are up.
        hook_specs = list(get_app(config.app.plugin).sim_hooks(config, model_dirs))

        # Pre-run cleanup first, exactly as the original did: kill leftover role
        # and sim processes and free the ports they held, so a previous crashed
        # run cannot fail this one in preflight.
        try:
            lifecycle.kill_orphans(
                lifecycle.orphan_patterns(config, hook_specs),
                lifecycle.listen_ports(config),
                self.log,
                owned=self.state.abandoned_pids(),
            )
        except Exception:  # noqa: BLE001 - cleanup must never fail a run
            # With the traceback: this is the one step of a run that sends
            # signals, and a bare message here left a flake - the server
            # SIGTERMed - with no way to tell which of the cleanup's steps had
            # raised, or whether it had already killed something before it did.
            self.log(
                "[orch] orphan cleanup skipped:\n{}".format(traceback.format_exc().rstrip())
            )

        preflight(config, model_dirs, self.log, allow_experimental=allow_experimental)
        from cadet.config.implemented import allow_experimental as _resolve_allow

        allowed = _resolve_allow(config, allow_experimental)

        servers = [n for n in config.nodes if n.role == "server"]
        clients = [n for n in config.nodes if n.role == "client"]
        loads = [n for n in config.nodes if n.role == "load"]

        # Constructed here (it creates nothing) and *started* inside the try
        # below, so a failure after the network exists still reaches a teardown.
        placement = Placement(
            config,
            run_id,
            out_dir,
            self.log,
            override=self.launcher,
            # The placement needs these because a role in a container cannot
            # see the orchestrator's search directories: it stages the
            # artifacts onto the machine the container runs on and mounts them.
            model_dirs=model_dirs,
        )
        # Two views of the config, deliberately: `effective` is the host's (what
        # the orchestrator resolves) and `role_argv_overrides` is the base for
        # what a role is told. The per-role part - outputs.root, which means a
        # different directory on each side of a bind mount - is added at each
        # launch from `placement.role_overrides_for(node)`, because it applies
        # only to the roles that run in a container: a run can mix placements,
        # and a local client told to write into a container's mount path would
        # write to a directory that does not exist here.
        effective = list(overrides)
        role_argv_overrides = list(overrides)
        placed = {}

        def _place():
            """Start the placement and re-resolve the config on its addresses."""
            nonlocal placed, effective, role_argv_overrides, config, run_id
            nonlocal out_dir, log_dir, servers, clients, loads
            placed = placement.start()
            if not placed:
                return
            # Containerised nodes bind and are reached by their container name on
            # the run's own network. Only this one field moves, exactly as the
            # emulated backend does for its hosts - and re-resolving afterwards
            # matters because the override changes the config hash, so the role
            # processes would otherwise derive a different run id and write their
            # metrics somewhere else.
            effective = list(overrides) + placement.address_overrides_for(placed)
            config, base = select_run_config(config_path, effective, run_index)
            run_id = "{}-r{}".format(base, rep)
            out_dir = output_dir(config, run_id)
            log_dir = os.path.join(out_dir, "logs")
            servers = [n for n in config.nodes if n.role == "server"]
            clients = [n for n in config.nodes if n.role == "client"]
            loads = [n for n in config.nodes if n.role == "load"]
            role_argv_overrides = list(effective)
            self.log("[orch] addresses: {}".format(json.dumps(placed)))

        # The run's simulator, when the config declares one to launch. Built
        # here and started inside the try below, so a failure after it starts
        # still reaches the teardown that removes it.
        sim_side = (
            SimSide(config, run_id, log=self.log)
            if carla_launch_settings(config) is not None
            else None
        )

        started = []  # (launcher, handle) in start order; torn down in reverse
        # Processes that must stay alive for the run to mean anything; their
        # exit ends the run with a recorded reason.
        guards = []
        # The subset of the guards whose *clean* exit can end a run; see
        # experiment.end_on.
        sim_hooks = []
        client_log_paths = []
        netem_sessions = []
        # (phase, {node: estimate}) for each time the clocks were compared.
        clock_phases = []
        startup_deadline = started_at + lifecycle.DEFAULT_STARTUP_LIMIT_S
        end_on, grace_s = lifecycle.end_rule(config)
        ok = False
        detail = ""
        reason = ""
        # A terminating signal must reach the teardown below rather than ending
        # the process on the spot; restored in the finally.
        restore_signals = lifecycle.install_interrupt_guard()
        try:
            # The simulator first: it is the slowest thing in a run to become
            # ready, every role needs it, and a scenario cannot be started at
            # all until it answers. Inside the try because it creates a network
            # and a container, which the teardown below has to remove.
            if sim_side is not None:
                sim_side.start()
                if not sim_side.wait_ready():
                    raise PreflightError(
                        "the simulator did not become ready: {}".format(
                            sim_side.status().get("detail") or "no detail"
                        )
                    )

            # Placing the nodes is the first thing inside the try: it creates a
            # docker network and containers, and a failure after that must still
            # reach the teardown. (This was the bug already fixed once in the
            # emulated session, where the topology was built outside the try.)
            _place()
            # The id is final from here, whether or not anything was placed, so
            # this is the one place the run's directories are created.
            os.makedirs(log_dir, exist_ok=True)

            # Before any shaping is installed, and so before a probe could
            # measure an emulated delay instead of a clock. The second
            # measurement is taken after the shaping has been removed again,
            # for the same reason.
            measured = self._measure_clock_offsets(config, placement, "pre_run")
            if measured is not None:
                clock_phases.append(("pre_run", measured))

            # The readiness waits this run will make, which is what the startup
            # limit defaults to the sum of. Computed after placement, because
            # placement can re-resolve the config and so the node lists.
            readiness_timeouts = [SERVER_READINESS_TIMEOUT_S] * len(servers)
            readiness_timeouts += [spec.readiness_timeout_s for spec in hook_specs]
            client_start = getattr(
                config.experiment, "client_start", CLIENT_START_DEFAULT
            )
            if client_start != CLIENT_START_AFTER_SCENARIO:
                # Each client's own ready marker is a readiness wait like any
                # other, so the startup budget has to allow for it.
                readiness_timeouts += [CLIENT_READINESS_TIMEOUT_S] * len(clients)
            startup_deadline = started_at + lifecycle.startup_limit_s(
                config, readiness_timeouts
            )

            # netem shaping is installed before anything sends, and removed in
            # the finally block below whatever happens, so a crashed run never
            # leaves a qdisc behind to corrupt the next one.
            #
            # One session set per node, each with the runner for the namespace
            # whose interface it has to shape: the host for a local node,
            # `docker exec` in the container for a placed one. This is also why
            # a node container is started before its role - the qdisc has to be
            # in place before the sender sends.
            # A node on another machine shapes nothing itself: its interface is
            # that machine's, and a qdisc there would shape everything else on
            # it. What it sends is shaped where it arrives instead, by the node
            # at the other end of the link, on that node's own interface.
            offmachine = [node.name for node in placement.offmachine_nodes()]
            for node in config.nodes:
                netem_sessions += netem.sessions_for_config(
                    config,
                    node_names=[node.name],
                    runner=placement.runner_for(node.name),
                    offmachine=offmachine,
                )
            # Recorded before they are applied: an interrupt between applying a
            # qdisc and the teardown below must still leave something that says
            # which interface to clear.
            self.state.record_netem(
                run_id, out_dir, [session.interface for session in netem_sessions]
            )
            if netem_sessions:
                for command in netem.apply_sessions(netem_sessions):
                    self.log("[orch] netem: {}".format(command))
            # 1) servers, each waited to "listening" first. The original
            # started the server before ScenarioRunner so that the scenario's
            # own clock does not run while the model is still loading.
            for node in servers:
                node_config, node_index = self._role_config(
                    placement, node, config_path, run_index
                )
                argv = self._base_argv(
                    "serve",
                    node_config,
                    role_argv_overrides + placement.role_overrides_for(node),
                    node_index,
                    app_modules,
                    executable=role_executable(node.launcher),
                )
                argv += ["--node", node.name, "--rep", str(rep)]
                argv += placement.role_args(node)
                for d in model_dirs or []:
                    argv += ["--model-dir", d]
                spec = LaunchSpec(
                    name="serve:{}".format(node.name),
                    argv=argv,
                    log_path=os.path.join(log_dir, "server_{}.log".format(node.name)),
                    readiness_marker=SERVER_READY_MARKER,
                    readiness_timeout_s=SERVER_READINESS_TIMEOUT_S,
                )
                launcher = placement.launcher_for(node)
                self.log(
                    "[orch] starting {} ({})".format(spec.name, node.launcher)
                )
                h = launcher.launch(spec)
                started.append((launcher, h))
                self.state.add_proc(run_id, out_dir, spec.name, h.pid)
                guards.append((spec.name, h.proc))
                # The wait is cut to whatever is left of the startup budget, so
                # the phase cannot overrun its limit by waiting out a step.
                budget = lifecycle.remaining_startup_s(
                    startup_deadline, spec.readiness_timeout_s, spec.name
                )
                if not launcher.wait_ready(h, timeout_s=budget):
                    if time.monotonic() >= startup_deadline:
                        raise lifecycle.StartupLimitExceeded(
                            "startup limit reached waiting for {}".format(spec.name)
                        )
                    raise PreflightError(
                        "server failed to start: {}".format(
                            lifecycle.readiness_failure(
                                node.name, proc=h.proc, log_path=spec.log_path
                            )
                        )
                    )

            def _start_hooks():
                """Start the app's sim hooks and block on their marker."""
                # A scenario with an image of its own runs in that container;
                # without one it stays on the host, as every run did before.
                # Either way it is a launcher, so everything below - readiness,
                # supervision, teardown - is the same code for both.
                # The repository root is the directory the run was started
                # from, which is what a config's relative scenario paths
                # already mean: the scenario container has to resolve
                # `--additionalScenario` the same way a host hook would.
                hook_launcher = (
                    scenario_launcher_for(config, sim_side, os.getcwd())
                    or self._local_launcher
                )
                for spec in hook_specs:
                    if spec.log_path is None:
                        spec.log_path = os.path.join(log_dir, "{}.log".format(spec.name))
                    self.log("[orch] sim hook: {}".format(spec.name))
                    # Sim hooks drive the simulator: they belong beside it, never
                    # behind a node's emulated link, which would shape the
                    # simulator's own control loop into the experiment.
                    h = hook_launcher.launch(spec)
                    started.append((hook_launcher, h))
                    self.state.add_proc(run_id, out_dir, spec.name, h.pid)
                    guards.append((spec.name, h.proc))
                    sim_hooks.append((spec.name, h.proc))
                    budget = lifecycle.remaining_startup_s(
                        startup_deadline, spec.readiness_timeout_s, spec.name
                    )
                    if not self._local_launcher.wait_ready(h, timeout_s=budget):
                        if time.monotonic() >= startup_deadline:
                            raise lifecycle.StartupLimitExceeded(
                                "startup limit reached waiting for {}".format(spec.name)
                            )
                        raise PreflightError(
                            "sim hook {!r} failed to become ready; see {}".format(
                                spec.name, spec.log_path
                            )
                        )

            def _start_workers(wait_ready):
                """Start the clients and load generators.

                ``wait_ready`` blocks on each client's own ready marker, which is
                what lets the sim hook be started only once every client can do
                nothing more without the scene.
                """
                specs = []
                for node in clients:
                    node_config, node_index = self._role_config(
                        placement, node, config_path, run_index
                    )
                    argv = self._base_argv(
                        "client",
                        node_config,
                        role_argv_overrides + placement.role_overrides_for(node),
                        node_index,
                        app_modules,
                        executable=role_executable(node.launcher),
                    )
                    argv += ["--node", node.name, "--rep", str(rep)]
                    specs.append(("client", node, argv))
                for node in loads:
                    node_config, node_index = self._role_config(
                        placement, node, config_path, run_index
                    )
                    argv = self._base_argv(
                        "load",
                        node_config,
                        role_argv_overrides + placement.role_overrides_for(node),
                        node_index,
                        app_modules,
                        executable=role_executable(node.launcher),
                    )
                    specs.append(("load", node, argv))

                handles = []
                for sub, node, argv in specs:
                    # A client announces itself ready before the scene exists;
                    # a load generator has no scene to wait for, so only the
                    # clients carry the marker.
                    marker = CLIENT_READY_MARKER if (wait_ready and sub == "client") else None
                    spec = LaunchSpec(
                        name="{}:{}".format(sub, node.name),
                        argv=argv,
                        log_path=os.path.join(log_dir, "{}_{}.log".format(sub, node.name)),
                        readiness_marker=marker,
                        readiness_timeout_s=CLIENT_READINESS_TIMEOUT_S,
                    )
                    launcher = placement.launcher_for(node)
                    self.log("[orch] starting {} ({})".format(spec.name, node.launcher))
                    h = launcher.launch(spec)
                    started.append((launcher, h))
                    self.state.add_proc(run_id, out_dir, spec.name, h.pid)
                    handles.append((spec.name, h.proc))
                    if sub == "client":
                        client_log_paths.append(spec.log_path)
                    if marker is not None:
                        budget = lifecycle.remaining_startup_s(
                            startup_deadline, CLIENT_READINESS_TIMEOUT_S, spec.name
                        )
                        if not launcher.wait_ready(h, timeout_s=budget):
                            if time.monotonic() >= startup_deadline:
                                raise lifecycle.StartupLimitExceeded(
                                    "startup limit reached waiting for {}".format(spec.name)
                                )
                            raise PreflightError(
                                "client did not report ready: {}".format(
                                    lifecycle.readiness_failure(
                                        node.name, proc=h.proc, log_path=spec.log_path
                                    )
                                )
                            )
                        self.log("[orch] {} ready".format(spec.name))
                return handles

            # 2) and 3), in the order experiment.client_start asks for.
            #
            # after_scenario (the default): the original order - the sim hook
            # goes first and the clients find a scene already in place.
            #
            # before_scenario: the clients go first and the sim hook follows, so
            # the scenario starts only once every client is ready for it. The ego
            # a scenario spawns exists for a short window and a client still
            # setting up would miss it; opting in removes that race.
            if client_start == CLIENT_START_AFTER_SCENARIO:
                _start_hooks()
                timing.startup_s = time.monotonic() - started_at
                worker_handles = _start_workers(wait_ready=False)
            else:
                worker_handles = _start_workers(wait_ready=True)
                _start_hooks()
                # Startup ends once the scenario is up: under this order the
                # clients are already running, and waiting for the ego is part
                # of the run rather than part of starting it.
                timing.startup_s = time.monotonic() - started_at

            # Block until the end_on rule fires, the active limit passes, or a
            # server or sim hook fails.
            active_at = time.monotonic()
            outcome = lifecycle.wait_for_workers(
                worker_handles,
                lifecycle.active_limit_s(config),
                guards,
                self.log,
                sim_hooks=sim_hooks,
                end_on=end_on,
                grace_s=grace_s,
            )
            timing.active_s = time.monotonic() - active_at
            ok = outcome.ok
            detail = str(outcome)
            reason = outcome.reason
            if reason == lifecycle.REASON_ACTIVE_LIMIT:
                timing.limit = lifecycle.LIMIT_ACTIVE
            # A run whose workers all exited cleanly but produced nothing is
            # not a success. Only the clients can say so, and only after they
            # have finished writing, so the check comes after the wait.
            if ok:
                failures = lifecycle.client_output_failures(out_dir, config, self.log)
                if failures:
                    ok = False
                    reason = lifecycle.REASON_NO_OUTPUT
                    detail = "{}: {}".format(
                        lifecycle.REASON_NO_OUTPUT, "; ".join(failures)
                    )

        except lifecycle.StartupLimitExceeded as exc:
            ok = False
            reason = lifecycle.REASON_STARTUP_LIMIT
            detail = "{}: {}".format(lifecycle.REASON_STARTUP_LIMIT, exc)
            timing.limit = lifecycle.LIMIT_STARTUP
            self.log("[orch] {}; tearing down".format(detail))
        except (KeyboardInterrupt, lifecycle.RunInterrupted) as exc:
            # The teardown in the finally below has not run yet; it runs as this
            # propagates out of the try, and the run is recorded as interrupted
            # rather than left with no record at all.
            ok = False
            reason = lifecycle.REASON_INTERRUPTED
            # KeyboardInterrupt carries no message, so a bare str(exc) records
            # "interrupted: " and says nothing about what arrived.
            detail = "{}: {}".format(
                lifecycle.REASON_INTERRUPTED, str(exc) or type(exc).__name__
            )
            self.log("[orch] {}; tearing down".format(detail))
        finally:
            # A run that never reached its workers still spent time in startup,
            # and that is the number worth recording for it.
            if timing.startup_s is None:
                timing.startup_s = time.monotonic() - started_at
            teardown_at = time.monotonic()
            # Tear down everything we started, in reverse order, each handle
            # through the launcher that started it. No orphans.
            for launcher, h in reversed(started):
                launcher.stop(h)
            netem.remove_sessions(netem_sessions)
            # Containers last: a role runs inside one, so removing the container
            # first would take the role with it and lose its exit code.
            for item in placement.stop():
                self.log("[orch] removed {}".format(item))
            # The simulator after them, for the same reason one step further
            # out: a role still shutting down may hold a connection to it, and
            # a server pulled out from under a client turns an ordinary
            # teardown into an error in the client's last log line.
            if sim_side is not None:
                outcome = sim_side.stop()
                for item in outcome.get("removed", ()):
                    self.log("[orch] removed {}".format(item))
            self.state.clear(run_id)
            restore_signals()
            timing.teardown_s = time.monotonic() - teardown_at

        for path in client_log_paths:
            if path:
                lifecycle.print_final_client_report(path, self.log)
        # After the teardown, so the qdiscs are gone and this probe sees the
        # same unshaped path the first one did. Measuring twice is what makes
        # the offset's constancy over the run an observation: the record
        # carries both, and widens the uncertainty if they moved apart.
        if clock_phases:
            measured = self._measure_clock_offsets(config, placement, "post_run")
            if measured is not None:
                clock_phases.append(("post_run", measured))
        # Also after the teardown, and for a reason of its own: a node placed on
        # another machine writes its samples into a staging tree there, and the
        # placement fetches that tree back as it stops. Reducing earlier would
        # summarise the local nodes and omit the remote ones without saying so.
        self._write_device_summary(out_dir)
        self._write_marker(
            out_dir,
            run_id,
            ok,
            detail,
            reason,
            timing,
            allowed,
            clock_phases=clock_phases,
            simulator=sim_side.status() if sim_side is not None else None,
            images=placement.image_facts,
        )
        status = "ok" if ok else "FAILED"
        self.log("[orch] run {} {} ({}) [{}]".format(run_id, status, detail, timing))
        return RunResult(run_id, out_dir, ok, detail, reason)

    def _write_device_summary(self, out_dir) -> None:
        """Reduce this run's collector CSVs to one per-node record, and log it.

        Writes nothing for a run that collected nothing, which is the collectors'
        own default. Failures are logged rather than raised: the samples are
        already on disk and a run that measured its devices must not be reported
        as failed because the convenience roll-up of those samples could not be
        written.
        """
        try:
            summary = device_summary.summarise_run(out_dir)
            if not summary:
                return
            path = device_summary.write_summary(out_dir)
            for line in device_summary.summary_lines(summary):
                self.log(line)
            if path is None:
                self.log(
                    "[metrics] WARNING: the per-node device summary could not be "
                    "written; the collector CSVs in the run directory are intact."
                )
        except Exception as exc:  # noqa: BLE001 - see the docstring
            self.log(
                "[metrics] WARNING: the per-node device summary was skipped "
                "({}: {}); the collector CSVs in the run directory are "
                "intact.".format(type(exc).__name__, exc)
            )

    @staticmethod
    def _client_outcomes(out_dir):
        """Return each client node's own verdict on the run, by node name.

        The orchestrator knows whether the processes it started exited and
        whether it cleaned up after them. It does not know whether the
        application did what it was for: a braking client whose ego was removed
        before it stopped exits 0 and tears down cleanly. Each client role
        already writes that verdict beside its outputs, so the run's record
        carries it rather than leaving a reader to find the file - a run whose
        application reported failure read as a successful run.
        """
        from cadet.roles.client import CLIENT_OUTCOME_PREFIX

        outcomes = {}
        try:
            names = sorted(os.listdir(out_dir))
        except OSError:
            return outcomes
        for name in names:
            if not name.startswith(CLIENT_OUTCOME_PREFIX + "_") or not name.endswith(".json"):
                continue
            node = name[len(CLIENT_OUTCOME_PREFIX) + 1 : -len(".json")]
            try:
                with open(os.path.join(out_dir, name)) as handle:
                    record = json.load(handle)
            except (OSError, ValueError):
                # A verdict that cannot be read is not a verdict. Leaving the
                # node out says "nothing was recorded", which is true; invented
                # defaults would say something that is not.
                continue
            if isinstance(record, dict) and "ok" in record:
                outcomes[node] = {
                    "ok": bool(record.get("ok")),
                    "detail": record.get("detail") or "",
                }
        return outcomes

    def _write_marker(
        self,
        out_dir,
        run_id,
        ok,
        detail,
        reason="",
        timing=None,
        allow_experimental=False,
        clock_phases=(),
        simulator=None,
        images=None,
    ):
        record = {
            "run_id": run_id,
            "status": "complete" if ok else "failed",
            "detail": detail,
            "reason": reason,
            # Whether this run accepted unverified features, so a result that
            # leaned on one can be identified without re-reading the config.
            "allow_experimental": bool(allow_experimental),
        }
        # Which code produced this run. A run id is a hash of the resolved
        # config *as this version's schema reads it*, so it is reproducible
        # against this version and not against a later one: adding a field or
        # changing a default moves every id, uniformly. Measured on this
        # repository - 53 of 74 recorded run directories no longer recompute to
        # their own id, and each recomputes to the id current code gives that
        # same cell. Without this field a reader cannot tell a drifted id from a
        # mismatched one, which is the difference between "re-run it with the
        # recorded version" and "something is wrong".
        commit, dirty = placement_mod.repo_revision()
        if commit:
            record["cadet"] = {"commit": commit, "dirty": bool(dirty)}
        else:
            # An orchestrator running in the CADET image has no git to ask, so
            # it falls back to the commit the image was built from and says
            # that is where the answer came from. It is the image's code, which
            # is what ran unless a working tree was mounted over it - and a
            # mounted tree with no git in reach cannot be identified at all,
            # which is why the source is recorded rather than implied.
            image_commit = os.environ.get("CADET_IMAGE_REVISION", "").strip()
            if image_commit and image_commit != "unknown":
                record["cadet"] = {
                    "commit": image_commit,
                    "dirty": False,
                    "from": "image",
                }
        # Phase durations and which limit, if any, ended the run: enough to tell
        # a run that hit a bound from one that was merely slow, and to see which
        # phase the time went to, without re-reading the logs.
        record.update((timing or lifecycle.RunTiming()).as_dict())
        # Only when the run asked for it, so a run that did not measure its
        # clocks has no field that could be read as a zero offset.
        if clock_phases:
            record["clock_offset"] = clock_offset.run_record(clock_phases)
        # Which simulator this run launched, when it launched one. Absent rather
        # than null for a run that attached to somebody else's, because those
        # are different claims: "this run started no server" is not "this run
        # started a server and recorded nothing about it".
        if simulator is not None:
            record["simulator"] = simulator
        # Which image each containerised node ran, by node: the tag it asked
        # for, the digest the daemon resolved that tag to, and the commit the
        # image records. A tag is rebuilt, so two runs naming the same one need
        # not have run the same code, and the commit of *this* repository says
        # nothing about the code inside a container. Absent for a run that
        # placed no node in a container, rather than empty.
        if images:
            record["images"] = {name: dict(facts) for name, facts in sorted(images.items())}
        # What the application itself reported, by client node. Absent for a run
        # with no client role rather than empty, because "no client reported"
        # and "a client reported nothing" are different claims.
        clients = self._client_outcomes(out_dir)
        if clients:
            record["clients"] = clients
        with open(os.path.join(out_dir, RUN_MARKER), "w") as fh:
            json.dump(record, fh, indent=2)

    # -- sweep -----------------------------------------------------------------
    def sweep(
        self,
        config_path: str,
        overrides: List[str],
        app_modules: List[str],
        model_dirs: List[str],
        reps: Optional[int] = None,
        allow_experimental: Optional[bool] = None,
        resume: Optional[bool] = None,
    ) -> List[RunResult]:
        """Run every (run_index, rep) cell.

        A completed cell is re-run unless resume is asked for, which is what the
        original did; ``resume`` (or ``experiment.resume``) skips cells that
        already have a run.json.

        Cells are separated by ``experiment.inter_run_pause_s``, five seconds as
        in the original. It is not idle time: on a shared simulator it is what
        gave the previous run's actors time to be cleared before the next
        scenario reloaded the world.

        Stops early if a run is interrupted: a terminating signal is a request
        to stop the sweep, and the cells it never reached are left absent from
        the output.
        """
        import_app_modules(app_modules or [])
        data = load_config_data(config_path)
        run_configs = expand_sweep(data)
        results = []
        first = True
        sweep_dir, total = self._sweep_scope(config_path, overrides, run_configs, reps)
        self._write_sweep_marker(sweep_dir, "running", 0, total)
        state = "interrupted"
        try:
            for run_index, cfg in enumerate(run_configs):
                n_reps = reps if reps is not None else cfg.experiment.repetitions
                resuming = (
                    cfg.experiment.resume if resume is None else resume
                )
                for rep in range(n_reps):
                    _c, base = select_run_config(config_path, overrides, run_index)
                    run_id = "{}-r{}".format(base, rep)
                    if resuming and self.is_complete(_c, run_id):
                        self.log("[orch] skip {} (complete)".format(run_id))
                        results.append(RunResult(run_id, output_dir(_c, run_id), True, "resumed"))
                        self._write_sweep_marker(sweep_dir, "running", len(results), total)
                        continue
                    if not first:
                        self._pause_between_cells(_c)
                    first = False
                    result = self.run_one(
                        config_path,
                        overrides,
                        run_index,
                        rep,
                        app_modules,
                        model_dirs,
                        allow_experimental=allow_experimental,
                    )
                    results.append(result)
                    self._write_sweep_marker(sweep_dir, "running", len(results), total)
                    # One Ctrl-C means stop the sweep, not just the cell it
                    # landed in. The run itself has already recorded why it
                    # ended and torn itself down; the cells not reached stay
                    # absent from the output, which is exactly what makes the
                    # next attempt resume them.
                    if result.reason == lifecycle.REASON_INTERRUPTED:
                        self.log("[orch] sweep interrupted after {}".format(result.run_id))
                        return results
            state = "complete"
            return results
        finally:
            # In a finally so that an interrupt, an exception and a clean finish
            # all leave a marker that is not "running". A hard kill of this
            # process still can, and then the sweep reads as in progress - the
            # same exposure a run directory with no run.json already has, and
            # recorded here rather than guarded against.
            self._write_sweep_marker(sweep_dir, state, len(results), total)

    def _sweep_scope(self, config_path, overrides, run_configs, reps):
        """Return ``(directory, total_cells)`` for the sweep about to run.

        The directory is the one that holds the run directories - the same one a
        reader points a dashboard at - and every cell of one sweep shares it,
        because the naming template varies only by run id. Derived from the
        first cell rather than assumed, and ``None`` if it cannot be derived,
        which only disables the marker.
        """
        if not run_configs:
            return None, 0
        total = 0
        for run_index, cfg in enumerate(run_configs):
            total += reps if reps is not None else cfg.experiment.repetitions
        try:
            first_config, base = select_run_config(config_path, overrides, 0)
            directory = os.path.dirname(output_dir(first_config, "{}-r0".format(base)))
        except (IndexError, OSError, ValueError):
            return None, total
        return directory, total

    def _write_sweep_marker(self, directory, state, done, total) -> None:
        """Write ``sweep.json``, or do nothing if it cannot be written.

        Never raises: a sweep that measured everything it was asked to must not
        be reported as failed because a progress file could not be written, and
        the run directories carry the results either way.
        """
        if not directory:
            return
        try:
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, SWEEP_MARKER), "w") as fh:
                json.dump(
                    {"state": state, "completed": done, "cells": total}, fh, indent=2
                )
        except OSError as exc:
            self.log("[orch] WARNING: sweep progress not recorded ({})".format(exc))

    def _pause_between_cells(self, config) -> None:
        """Wait ``experiment.inter_run_pause_s`` before the next cell starts."""
        pause = float(config.experiment.inter_run_pause_s)
        if pause <= 0:
            return
        self.log("[orch] pausing {:.1f}s before the next cell".format(pause))
        time.sleep(pause)

    # -- status / down ---------------------------------------------------------
    def status(self) -> dict:
        """Return the current run-state (recorded active processes)."""
        return self.state.load()

    def down(self):
        """Remove every CADET-owned leftover; return what was removed.

        This is the recovery path for a run that never reached its teardown - a
        SIGKILL, or a stop before the signal guard was installed. It covers more
        than the processes it used to: containers, docker networks, emulated
        namespaces, OVS bridges, veths and the qdiscs a run recorded, each
        matched by ownership rather than by kind. See
        :mod:`cadet.orchestrator.recovery`.
        """
        from cadet.orchestrator import recovery

        return recovery.sweep(log=self.log)
