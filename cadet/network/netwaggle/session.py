"""Run one CADET experiment with its roles inside an emulated network.

What this is for:
    The ``netwaggle`` backend puts role processes inside Mininet host
    namespaces, which only something running in the emulation container can do.
    This module is that something: it builds the topology, starts the emulator,
    launches the roles in their emulated hosts, waits for the workers, and tears
    everything down. The host-side orchestrator reaches it with a single
    ``docker exec ... cadet emulate``.

What it contains:
    - :func:`address_overrides`: the ``--set`` strings that move each node onto
      its shaped address.
    - :class:`EmulatedRunResult`
    - :func:`run_emulated`: the whole lifecycle for one run.

How it fits into CADET:
    ``cadet emulate <config>`` (see :mod:`cadet.cli`) calls :func:`run_emulated`.
    It is the emulated counterpart of
    :meth:`cadet.orchestrator.runner.Orchestrator.run_one`: same ordering
    (servers to readiness, then workers, then teardown in reverse), but the
    processes live in namespaces rather than on the host.

    Only the node addresses are rewritten. Anything else the config points at —
    the simulator host, a model store — keeps the address the user configured
    and is reached over each host's unshaped bypass interface, which is what
    keeps framework plumbing off the emulated network.

How to configure it:
    Nothing extra. It reads the same experiment config; the emulated topology
    comes from ``links`` with ``network.mode: netwaggle``.
"""

import json
import os
import subprocess
import sys
import time
import traceback
from typing import Dict, List, Optional

from cadet.core.apps import get_app
from cadet.core.launcher import LocalLauncher
from cadet.metrics import device_summary
from cadet.network.netwaggle.emulator import Emulator
from cadet.network.netwaggle.generator import generate
from cadet.orchestrator import lifecycle
from cadet.orchestrator.runner import CLIENT_READINESS_TIMEOUT_S, preflight
from cadet.orchestrator.state import StateStore
from cadet.roles.client import (
    CLIENT_READY_MARKER,
    CLIENT_START_AFTER_SCENARIO,
    CLIENT_START_DEFAULT,
)
from cadet.roles.common import output_dir, select_run_config

# Substring a server prints once it is listening; must match the orchestrator's.
SERVER_READY_MARKER = "Waiting for connection"
# Sim hooks run in the container's root namespace, so an ordinary local launcher
# is the right one; role processes use Emulator.popen instead.
_LAUNCHER = LocalLauncher()
RUN_MARKER = "run.json"


def address_overrides(config, topology) -> List[str]:
    """Return ``--set`` overrides placing each emulated node on its shaped address.

    The server binds ``node.host`` and the client connects to it, so rewriting
    that one field is enough to move the link's traffic onto the emulated
    network. Nodes outside the topology are left alone.
    """
    index_of = {node.name: i for i, node in enumerate(config.nodes)}
    overrides = []
    for host in topology.hosts:
        overrides.append("nodes.{}.host={}".format(index_of[host.name], host.link_ip))
    return overrides


def _start_sim_hooks(config, model_dirs, log_dir, log, startup_deadline=None):
    """Start the app's sim-side hooks, outside the emulated network.

    A hook such as ScenarioRunner drives the simulator, which is framework
    plumbing: it must reach the simulator directly rather than through an
    emulated link. They therefore run in the emulation container's root
    namespace, which has the container's ordinary route to the simulator, and not
    inside a Mininet host.

    Returns the handles in start order; the caller stops them in reverse.
    """
    handles = []
    for spec in get_app(config.app.plugin).sim_hooks(config, model_dirs):
        if spec.log_path is None:
            spec.log_path = os.path.join(log_dir, "{}.log".format(spec.name))
        log("[emulate] sim hook: {}".format(spec.name))
        handle = _LAUNCHER.launch(spec)
        handles.append(handle)
        budget = spec.readiness_timeout_s
        if startup_deadline is not None:
            budget = lifecycle.remaining_startup_s(
                startup_deadline, spec.readiness_timeout_s, spec.name
            )
        if not _LAUNCHER.wait_ready(handle, timeout_s=budget):
            for started in reversed(handles):
                _LAUNCHER.stop(started)
            if startup_deadline is not None and time.monotonic() >= startup_deadline:
                raise lifecycle.StartupLimitExceeded(
                    "startup limit reached waiting for {}".format(spec.name)
                )
            raise RuntimeError(
                "sim hook {!r} failed to become ready; see {}".format(spec.name, spec.log_path)
            )
    return handles


def _apply_transport_tuning(emulator, config, topology, log) -> None:
    """Apply each emulated node's transport tuning inside its own namespace.

    This is the only place it can take effect for this backend: the sysctls are
    per network namespace, and the emulated hosts are namespaces created after
    the container started, so neither a docker flag nor a host setting reaches
    them.
    """
    from cadet.network.tuning import namespace_commands

    by_name = {node.name: node for node in config.nodes}
    for host in topology.hosts:
        node = by_name.get(host.name)
        if node is None:
            continue
        for argv in namespace_commands(node, routes=emulator.peer_routes(host.name)):
            output = emulator.run_in_host(host.name, argv).strip()
            log(
                "[emulate] tuning {}: {}{}".format(
                    host.name, " ".join(argv), " -> " + output if output else ""
                )
            )


class EmulatedRunResult:
    """Outcome of one emulated run."""

    def __init__(
        self,
        run_id: str,
        out_dir: str,
        ok: bool,
        detail: str,
        emulator_status: Optional[Dict[str, object]] = None,
        reason: str = "",
    ):
        self.run_id = run_id
        self.out_dir = out_dir
        self.ok = ok
        self.detail = detail
        self.emulator_status = emulator_status or {}
        # Why the run ended; see cadet.orchestrator.lifecycle REASON_* values.
        self.reason = reason

    def as_dict(self) -> Dict[str, object]:
        return {
            "run_id": self.run_id,
            "out_dir": self.out_dir,
            "status": "complete" if self.ok else "failed",
            "detail": self.detail,
            "reason": self.reason,
            "emulator": self.emulator_status,
        }


def _stage_modules() -> List[str]:
    """The pipeline stage modules this process was told to import.

    Read from the environment, where ``emulate`` publishes what
    ``--stage-module`` named, so a role's argv can carry it into an emulated
    host whose process inherits nothing from here.
    """
    value = os.environ.get("CADET_STAGE_MODULES", "")
    return [part for part in value.split(os.pathsep) if part]


def _write_device_summary(out_dir: str, log) -> None:
    """Reduce this run's collector CSVs to one per-node record, and log it.

    Writes nothing for a run that collected nothing. Failures are logged rather
    than raised: the samples are already on disk, and a run that measured its
    devices must not be reported as failed because the roll-up of those samples
    could not be written.
    """
    try:
        summary = device_summary.summarise_run(out_dir)
        if not summary:
            return
        path = device_summary.write_summary(out_dir)
        for line in device_summary.summary_lines(summary):
            log(line)
        if path is None:
            log(
                "[metrics] WARNING: the per-node device summary could not be "
                "written; the collector CSVs in the run directory are intact."
            )
    except Exception as exc:  # noqa: BLE001 - see the docstring
        log(
            "[metrics] WARNING: the per-node device summary was skipped "
            "({}: {}); the collector CSVs in the run directory are "
            "intact.".format(type(exc).__name__, exc)
        )


def _provenance() -> Dict[str, object]:
    """Which code produced this run, in the form an unemulated run records it.

    A checkout answers through git. An emulated run is built inside the
    emulation image, where there is no git and the commit sits in an OCI label
    only the daemon can read, so the image also carries it in the environment
    and a record that falls back to it says where the answer came from.
    """
    from cadet.orchestrator import placement

    commit, dirty = placement.repo_revision()
    if commit:
        return {"cadet": {"commit": commit, "dirty": bool(dirty)}}
    image_commit = os.environ.get("CADET_IMAGE_REVISION", "").strip()
    if image_commit and image_commit != "unknown":
        return {"cadet": {"commit": image_commit, "dirty": False, "from": "image"}}
    return {}


def run_emulated(
    config_path: str,
    overrides: Optional[List[str]] = None,
    run_index: int = 0,
    rep: int = 0,
    app_modules: Optional[List[str]] = None,
    model_dirs: Optional[List[str]] = None,
    interactive: bool = False,
    readiness_timeout_s: float = 120.0,
    log=print,
    allow_experimental: Optional[bool] = None,
) -> EmulatedRunResult:
    """Build the emulated network, run one experiment inside it, tear it down."""
    overrides = list(overrides or [])
    app_modules = list(app_modules or [])
    model_dirs = list(model_dirs or [])
    # Startup is timed from here: building the topology is part of what
    # max_startup_s bounds, as are the cleanup and preflight below.
    started_at = time.monotonic()
    timing = lifecycle.RunTiming()

    config, _ = select_run_config(config_path, overrides, run_index)
    topology = generate(config)
    if not topology.hosts:
        raise RuntimeError(
            "no link in this config uses network.mode: netwaggle, so there is "
            "nothing to emulate; run it with cadet run instead"
        )

    # The roles must address each other on the emulated network, so the address
    # overrides are appended after the caller's (and so win over them).
    effective = overrides + address_overrides(config, topology)

    # Re-resolve against the effective overrides: they change the config and so
    # its hash, and the role processes will derive this same id. Using the
    # pre-override id here would put run.json in a different directory from the
    # metrics the roles write.
    effective_config, run_id_base = select_run_config(config_path, effective, run_index)
    run_id = "{}-r{}".format(run_id_base, rep)
    out_dir = output_dir(effective_config, run_id)
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log("[emulate] topology: {}".format(topology))
    log("[emulate] addresses: {}".format(json.dumps(topology.address_map())))

    # Pre-run cleanup, as on the host path: a previous run in this same
    # container may have left a role process or a bound port behind. Scoped the
    # same way too - only pids this store recorded, or command lines that are
    # CADET's own, are killed.
    store = StateStore()
    try:
        hook_specs = get_app(config.app.plugin).sim_hooks(config, model_dirs)
        lifecycle.kill_orphans(
            lifecycle.orphan_patterns(config, hook_specs),
            lifecycle.listen_ports(config),
            log,
            owned=store.abandoned_pids(),
        )
    except Exception:  # noqa: BLE001 - cleanup must never fail a run
        # With the traceback, for the reason given at the host path's own call
        # site: the cleanup is where signals are sent, so a swallowed exception
        # there hides both what failed and what it had already killed.
        log("[emulate] orphan cleanup skipped:\n{}".format(traceback.format_exc().rstrip()))

    # The host path's preflight, reused: model artifacts must resolve and a
    # namespace-only tuning must not be asked of a local node. It also emits the
    # throughput warning. Ports are skipped - an emulated server binds inside
    # its own namespace, where the container's own busy ports do not reach.
    preflight(
        effective_config,
        model_dirs,
        log,
        check_ports=False,
        allow_experimental=allow_experimental,
    )
    from cadet.config.implemented import allow_experimental as _resolve_allow

    allowed = _resolve_allow(effective_config, allow_experimental)

    servers = [n for n in config.nodes if n.role == "server"]
    clients = [n for n in config.nodes if n.role == "client"]
    loads = [n for n in config.nodes if n.role == "load"]
    emulated = {h.name for h in topology.hosts}

    end_on, grace_s = lifecycle.end_rule(config)
    # The readiness waits this run will make; the startup limit defaults to
    # their sum.
    readiness_timeouts = [readiness_timeout_s] * len(
        [n for n in config.nodes if n.role == "server" and n.name in emulated]
    )
    readiness_timeouts += [
        spec.readiness_timeout_s
        for spec in get_app(config.app.plugin).sim_hooks(config, model_dirs)
    ]
    client_start = getattr(config.experiment, "client_start", CLIENT_START_DEFAULT)
    if client_start != CLIENT_START_AFTER_SCENARIO:
        # Each client's ready marker is a readiness wait like any other, so the
        # startup budget has to allow for it.
        readiness_timeouts += [CLIENT_READINESS_TIMEOUT_S] * len(
            [n for n in config.nodes if n.role == "client" and n.name in emulated]
        )
    startup_deadline = started_at + lifecycle.startup_limit_s(config, readiness_timeouts)
    ok = False
    detail = ""
    reason = ""
    hooks = []
    guards = []
    # The subset of the guards whose clean exit can end a run; see
    # experiment.end_on.
    sim_hook_procs = []
    workers = []
    client_log_paths = []
    emulator = Emulator(topology, interactive=interactive)
    # A terminating signal must reach the teardown below; restored in the finally.
    restore_signals = lifecycle.install_interrupt_guard()
    try:
        # Inside the try: an interrupt between building the topology and the
        # teardown used to leave the bridge and its veths behind, because these
        # two calls sat outside it.
        emulator.build()
        emulator.start()
        _apply_transport_tuning(emulator, config, topology, log)

        # Servers first, then the sim hooks: the original starts the detection
        # server before ScenarioRunner so the scenario's own clock does not run
        # while the model is still loading.
        for node in servers:
            if node.name not in emulated:
                continue
            log_path = os.path.join(log_dir, "server_{}.log".format(node.name))
            proc = _spawn(
                emulator,
                node.name,
                "serve",
                config_path,
                effective,
                run_index,
                app_modules,
                log_path,
                extra=["--node", node.name] + [a for d in model_dirs for a in ("--model-dir", d)],
            )
            log("[emulate] started serve:{} pid {}".format(node.name, proc.pid))
            store.add_proc(run_id, out_dir, "serve:{}".format(node.name), proc.pid)
            guards.append(("serve:{}".format(node.name), proc))
            # Cut to what is left of the startup budget, as on the host path.
            budget = lifecycle.remaining_startup_s(
                startup_deadline, readiness_timeout_s, "serve:{}".format(node.name)
            )
            if not lifecycle.wait_for_marker(
                log_path, SERVER_READY_MARKER, proc=proc, timeout=budget
            ):
                if time.monotonic() >= startup_deadline:
                    raise lifecycle.StartupLimitExceeded(
                        "startup limit reached waiting for serve:{}".format(node.name)
                    )
                raise RuntimeError(
                    "server {!r} did not become ready; see {}".format(node.name, log_path)
                )

        def _start_hooks():
            """Start the sim hooks and block on their marker.

            Called inside the try: the emulator is already running, so a hook
            that fails to come up must still go through the teardown below.
            """
            handles = _start_sim_hooks(config, model_dirs, log_dir, log, startup_deadline)
            hooks.extend(handles)
            guards.extend((h.spec.name, h.proc) for h in handles)
            sim_hook_procs.extend((h.spec.name, h.proc) for h in handles)
            for handle in handles:
                store.add_proc(run_id, out_dir, handle.spec.name, handle.pid)

        def _start_workers(wait_ready):
            """Start the clients and load generators inside their emulated hosts.

            ``wait_ready`` blocks on each client's own ready marker, which is
            what lets the sim hook start only once every client can do nothing
            more without the scene.
            """
            for node in clients:
                if node.name not in emulated:
                    continue
                log_path = os.path.join(log_dir, "client_{}.log".format(node.name))
                client_log_paths.append(log_path)
                proc = _spawn(
                    emulator,
                    node.name,
                    "client",
                    config_path,
                    effective,
                    run_index,
                    app_modules,
                    log_path,
                    extra=["--node", node.name, "--rep", str(rep)],
                )
                workers.append(("client:{}".format(node.name), proc))
                if not wait_ready:
                    continue
                budget = lifecycle.remaining_startup_s(
                    startup_deadline,
                    CLIENT_READINESS_TIMEOUT_S,
                    "client:{}".format(node.name),
                )
                if not lifecycle.wait_for_marker(
                    log_path, CLIENT_READY_MARKER, proc=proc, timeout=budget
                ):
                    if time.monotonic() >= startup_deadline:
                        raise lifecycle.StartupLimitExceeded(
                            "startup limit reached waiting for client:{}".format(node.name)
                        )
                    raise RuntimeError(
                        "client {!r} did not report ready; see {}".format(node.name, log_path)
                    )
                log("[emulate] client:{} ready".format(node.name))
            for node in loads:
                if node.name not in emulated:
                    continue
                log_path = os.path.join(log_dir, "load_{}.log".format(node.name))
                workers.append(
                    ("load:{}".format(node.name), _spawn(
                        emulator,
                        node.name,
                        "load",
                        config_path,
                        effective,
                        run_index,
                        app_modules,
                        log_path,
                    ))
                )
            for name, proc in workers:
                store.add_proc(run_id, out_dir, name, proc.pid)

        # The order experiment.client_start asks for. The default is the
        # original's: the scenario first, the clients after it. See the host
        # path for what opting into the other order buys and costs.
        if client_start == CLIENT_START_AFTER_SCENARIO:
            _start_hooks()
            timing.startup_s = time.monotonic() - started_at
            _start_workers(wait_ready=False)
        else:
            _start_workers(wait_ready=True)
            _start_hooks()
            # Startup ends once the scenario is up: the clients are already
            # running, and waiting for the ego is part of the run rather than
            # part of starting it.
            timing.startup_s = time.monotonic() - started_at

        active_at = time.monotonic()
        outcome = lifecycle.wait_for_workers(
            workers,
            lifecycle.active_limit_s(config),
            guards,
            log,
            sim_hooks=sim_hook_procs,
            end_on=end_on,
            grace_s=grace_s,
        )
        timing.active_s = time.monotonic() - active_at
        ok = outcome.ok
        detail = str(outcome)
        reason = outcome.reason
        if reason == lifecycle.REASON_ACTIVE_LIMIT:
            timing.limit = lifecycle.LIMIT_ACTIVE
        # A run whose workers all exited cleanly but produced nothing is not a
        # success; see the host path.
        if ok:
            failures = lifecycle.client_output_failures(out_dir, config, log)
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
        log("[emulate] {}; tearing down".format(detail))
    except (KeyboardInterrupt, lifecycle.RunInterrupted) as exc:
        ok = False
        reason = lifecycle.REASON_INTERRUPTED
        # KeyboardInterrupt carries no message, so a bare str(exc) records
        # "interrupted: " and says nothing about what arrived.
        detail = "{}: {}".format(
            lifecycle.REASON_INTERRUPTED, str(exc) or type(exc).__name__
        )
        log("[emulate] {}; tearing down".format(detail))
    finally:
        if timing.startup_s is None:
            timing.startup_s = time.monotonic() - started_at
        teardown_at = time.monotonic()
        # Emulator.stop() terminates every role process it started, so the only
        # thing left to stop here is the sim hooks, which run outside it.
        for handle in reversed(hooks):
            _LAUNCHER.stop(handle)
        status = emulator.stop()
        log("[emulate] teardown: {}".format(status.detail or status.state))
        store.clear(run_id)
        restore_signals()
        timing.teardown_s = time.monotonic() - teardown_at

    for path in client_log_paths:
        lifecycle.print_final_client_report(path, log)
    # The same roll-up and the same provenance an unemulated run records. An
    # emulated run is a run: a reader of its directory asks the same questions
    # of it, and asked them of a record that answered neither. Found by the
    # acceptance matrix, whose metrics check wanted both from every cell.
    _write_device_summary(out_dir, log)
    result = EmulatedRunResult(run_id, out_dir, ok, detail, status.as_dict(), reason)
    record = result.as_dict()
    record.update(timing.as_dict())
    record["allow_experimental"] = bool(allowed)
    record.update(_provenance())
    with open(os.path.join(out_dir, RUN_MARKER), "w") as fh:
        json.dump(record, fh, indent=2)
    log(
        "[emulate] run {} {} ({}) [{}]".format(
            run_id, "ok" if ok else "FAILED", detail, timing
        )
    )
    return result


def _spawn(
    emulator, node_name, sub, config_path, overrides, run_index, app_modules, log_path, extra=None
):
    """Start one role process inside its emulated host, logging to a file."""
    argv = [sys.executable, "-u", "-m", "cadet", sub, config_path]
    for module in app_modules:
        argv += ["--app-module", module]
    # The stage modules travel on the argv, as the app modules do. A role in an
    # emulated host is started through the emulator rather than inherited from
    # this process, so a staged pipeline role reached an empty registry and
    # died on the first stage its config named. The same was true of a
    # containerised role, and is fixed the same way.
    for module in _stage_modules():
        argv += ["--stage-module", module]
    for override in overrides:
        argv += ["--set", override]
    argv += ["--run", str(run_index)]
    argv += list(extra or [])
    handle = open(log_path, "w")
    return emulator.popen(node_name, argv, stdout=handle, stderr=subprocess.STDOUT)
