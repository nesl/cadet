#!/usr/bin/env python3
"""Run the four-layer acceptance matrix (tests/acceptance/matrix.yaml).

For each valid cell it launches the cell's config, runs the acceptance checks
(a..d defined in the spec), and writes acceptance_report.{md,json}: pass / fail /
blocked per cell, the failing check and its error, wall time, and the config's
line count. `na` and `pending` cells are reported, not run.

Design notes
------------
* Resumable: `--resume` skips cells that already PASSED in the report at
  `--report-dir`; everything else (fail/blocked/not-run) is retried.
* Filterable: `--cell ID` (repeatable, prefix match), `--layer LAYER=VALUE`
  (repeatable), `--status valid|pending|na`.
* Honest classification: a cell that cannot run because a dependency is absent
  in this environment (no CARLA server, no second machine, no
  containerised senders for netem, no Mininet container for netwaggle, no live
  stream, no FM endpoint) is recorded `blocked` with the missing dependency, NOT
  `fail`. A `fail` means the run or a check went wrong with the dependency
  present -- that is what gets reported.
* Private scratch only: outputs go under `--out` (default: a mkdtemp), never the
  repo tree.
* It drives the CLI as a subprocess (`<python> -m cadet ...`, or `--cadet-cmd`),
  so a crashing run cannot take the harness down.

This script is environment-agnostic: point `--python` at the interpreter that
has the CADET deps (or run it with that interpreter directly).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CHECK_ORDER = [
    "a_completes_and_tears_down",
    "b_metrics_well_formed",
    "c_fm_analysis",
    "d_dashboard_renders",
]


# --------------------------------------------------------------------------- #
# spec loading
# --------------------------------------------------------------------------- #
def load_spec(path: Path) -> dict:
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


def line_count(path: Path) -> int:
    try:
        with open(path) as fh:
            return sum(1 for _ in fh)
    except OSError:
        return -1


# --------------------------------------------------------------------------- #
# resolve a cell's config (for prechecks and to locate its run dir)
# --------------------------------------------------------------------------- #
# The placeholder a distributed cell's remote node carries in the spec. A real
# machine's name is a property of whoever runs the matrix and not of CADET, so
# the spec names none and this is substituted at run time.
PLACEHOLDER_SECOND_MACHINE = "edge-host"

# Where the real address comes from, in order. The environment variable is for
# one-off runs; the file is for a machine that runs the matrix repeatedly, and
# is untracked so that an address never reaches a commit.
SECOND_MACHINE_ENV = "CADET_SECOND_MACHINE"
LOCAL_OVERRIDE_FILE = "tests/acceptance/local.yaml"

# The fields the override may carry for that machine, as the key in the file and
# the node field it sets. The host is the one that must be supplied; an account
# and a key are only needed where they differ from the spec's defaults.
SECOND_MACHINE_FIELDS = (
    ("host", "host"),
    ("user", "user"),
    ("ssh_identity", "ssh_identity"),
    ("venv", "venv"),
)


def second_machine_settings() -> dict:
    """Return the operator's settings for the remote machine, if any.

    ``$CADET_SECOND_MACHINE`` names the address alone, which is the common
    case. ``tests/acceptance/local.yaml`` carries a ``second_machine`` mapping
    for a machine whose account or key differs from the spec's default. The
    environment wins, so a one-off run can point somewhere else without
    editing the file.
    """
    settings = {}
    path = Path(REPO) / LOCAL_OVERRIDE_FILE
    if path.exists():
        import yaml

        with open(path) as handle:
            local = yaml.safe_load(handle) or {}
        block = local.get("second_machine") or {}
        if isinstance(block, str):
            block = {"host": block}
        for key, _field in SECOND_MACHINE_FIELDS:
            if block.get(key):
                settings[key] = str(block[key])
    from_env = os.environ.get(SECOND_MACHINE_ENV, "").strip()
    if from_env:
        settings["host"] = from_env
    return settings


def second_machine_overrides(cell: dict, settings: dict) -> list[str]:
    """Return ``--set`` overrides pointing a cell's remote nodes at a machine.

    The nodes are found by the placeholder they carry rather than by name,
    because which node is the remote one is the cell's business: most cells put
    the server there, and a cell with two remote nodes would need both.
    Returns nothing when the cell has no remote node, so a single-host cell is
    unaffected by the setting being present.
    """
    if not settings.get("host"):
        return []
    import yaml

    with open(Path(REPO) / cell["config"]) as handle:
        raw = yaml.safe_load(handle) or {}
    overrides = []
    for index, node in enumerate(raw.get("nodes") or []):
        if not isinstance(node, dict) or node.get("host") != PLACEHOLDER_SECOND_MACHINE:
            continue
        for key, field in SECOND_MACHINE_FIELDS:
            value = settings.get(key)
            if value:
                # By index, because an override path indexes a list only by
                # number: a quoted name is read as a mapping key, which
                # replaces the whole node with the fields named here and drops
                # its name and role.
                overrides.append("nodes[{}].{}={}".format(index, field, value))
    return overrides


def missing_second_machine(cell: dict) -> bool:
    """True if this cell needs a remote machine and none has been supplied."""
    with open(Path(REPO) / cell["config"]) as handle:
        import yaml

        raw = yaml.safe_load(handle) or {}
    nodes = raw.get("nodes") or []
    return any(
        isinstance(n, dict) and n.get("host") == PLACEHOLDER_SECOND_MACHINE for n in nodes
    )


def resolve_cell(cell: dict, out_root: str, extra_set: list[str] = ()):
    """Return (config_obj, run_dir, run_id) or raise. Uses the real loader.

    The operator's own ``--set`` overrides are applied here as well as to the
    cell's command. They have to be: a config can name a value the operator
    supplies rather than one the repository can know - the simulator's address
    is the standing case, written as a placeholder in the config - and a
    precheck that resolved the config without them read the placeholder and
    refused a cell whose run would have worked.
    """
    from cadet.config import loader

    overrides = ["outputs.root={}".format(out_root)] + list(extra_set)
    run_cfgs, _ = loader.load_run_configs(cell["config"], overrides)
    cfg = run_cfgs[0]
    rid = loader.run_id(cfg, 0)
    run_dir = Path(out_root) / cfg.experiment.name / cfg.experiment.app / rid
    return cfg, run_dir, rid


def _child_env() -> dict:
    """The environment a cell's command runs with.

    The repository goes on the import path *ahead of* whatever is already
    there, rather than replacing it: an environment may carry a path a cell
    needs - the simulator's client egg is mounted and put on PYTHONPATH rather
    than installed - and replacing the variable dropped it, so a CARLA cell
    failed on `import carla`.
    """
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(REPO)] + [p for p in existing.split(os.pathsep) if p and p != str(REPO)]
    return dict(os.environ, PYTHONPATH=os.pathsep.join(parts))


def locate_run(out_root: str, cfg, predicted: Path) -> tuple[Path, str]:
    """Return (run_dir, run_id) for the run a cell just produced.

    The predicted directory is right whenever the config the orchestrator
    resolved is the one on disk, and that is not so for a cell that places a
    role in a container: the orchestrator rewrites those nodes' hosts to the
    per-run container network's addresses before launching, the run id is a
    hash of the resolved config, and so the id is not predictable from the
    config file alone. Rather than reproduce that rewriting here, the run is
    found by looking for the marker under this cell's own experiment and app
    directory, newest first.
    """
    if (predicted / "run.json").exists():
        return predicted, predicted.name
    parent = Path(out_root) / cfg.experiment.name / cfg.experiment.app
    markers = sorted(
        parent.glob("*/run.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if markers:
        found = markers[0].parent
        return found, found.name
    return predicted, predicted.name


# --------------------------------------------------------------------------- #
# environment prechecks -> a blocked reason, or None if the cell can be tried
# --------------------------------------------------------------------------- #
def _tcp_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with contextlib.closing(socket.create_connection((host, port), timeout)):
            return True
    except OSError:
        return False


def precheck(cell: dict, cfg) -> str | None:
    src = cell["source"]
    net = cell["network"]
    infra = cell["infra"]

    # CARLA source needs a reachable simulator.
    if src == "carla":
        carla = getattr(cfg.source, "carla", None)
        host = getattr(carla, "host", None) if carla else None
        port = getattr(carla, "port", 2000) if carla else 2000
        if not host or host == "CARLA_HOST" or not _tcp_open(host, int(port)):
            return "no CARLA server reachable at {}:{}".format(host, port)

    # live source needs a stream URL reachable or a camera device present. An
    # emulated cell is the exception: its stream is named by an address inside
    # the emulation - the bypass gateway, which is the host as seen from an
    # emulated node - and that address does not exist out here, so a connect
    # from this process says nothing about whether the role will reach it.
    #
    # The exception skips this one check and not the rest. It used to return,
    # which also skipped the emulation check below, so an emulated live cell on
    # a machine with no Mininet was reported `fail` with an import error
    # instead of `blocked` for the container it needs - the one dependency it
    # was most obviously missing.
    emulated_live = src == "live" and cell.get("run_with") == "emulate"
    if src == "live" and not emulated_live:
        url = getattr(cfg.source, "url", None)
        dev = getattr(cfg.source, "device", None)
        if url:
            m = re.match(r"\w+://([^/:]+):?(\d+)?", url)
            host = m.group(1) if m else None
            port = int(m.group(2)) if (m and m.group(2)) else 554
            if not host or not _tcp_open(host, port):
                return "no live stream reachable at {}".format(url)
        elif dev is not None:
            if not Path("/dev/video{}".format(dev)).exists():
                return "no capture device /dev/video{}".format(dev)
        else:
            return "live source names neither url nor device"

    # replay source needs its frames fetched to disk.
    if src == "replay":
        p = getattr(cfg.source, "path", None)
        if p and not (Path(p).is_absolute() and Path(p).exists() or (REPO / p).exists()):
            return ("replay frames not present at {}"
                    " (run examples/distributed_sensing/fetch_frames.py)").format(p)

    # distributed_hil needs the second machine reachable over ssh.
    if infra == "distributed_hil":
        server = next((n for n in cfg.nodes if n.role == "server"), None)
        host = getattr(server, "host", None) if server else None
        sshp = getattr(server, "ssh_port", None) or 22
        if not host or not _tcp_open(host, int(sshp)):
            return "no second machine reachable at {} (ssh/{})".format(host, sshp)

    # netem needs the qdisc installed in the sending node's namespace. A node
    # placed in a container carries NET_ADMIN on that container, so the
    # capability belongs to the node and not to this process; it is this
    # process's only when a shaped node runs here as a plain local process.
    if net == "netem":
        images = {
            (getattr(n, "container", None) or "cadet:dev")
            for n in cfg.nodes
            if getattr(n, "launcher", "local") == "docker"
        }
        for image in sorted(images):
            if not _docker_image_present(image):
                return "netem needs containerised senders; image {} not built".format(image)
        from cadet.orchestrator.placement import shaping_nodes

        shaped = set(shaping_nodes(cfg))
        local_shaped = [
            n.name for n in cfg.nodes
            if n.name in shaped and getattr(n, "launcher", "local") == "local"
        ]
        if local_shaped and os.geteuid() != 0 and not _has_net_admin():
            return "netem on local node(s) {} needs NET_ADMIN here".format(local_shaped)

    # netwaggle must run inside the privileged Mininet emulation container.
    if net in ("netwaggle", "netwaggle_wifi"):
        try:
            import mininet  # noqa: F401
        except Exception:
            return "netwaggle needs the privileged Mininet-WiFi emulation container (cadet emulate)"
    return None


def _docker_image_present(tag: str) -> bool:
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", tag],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
        )
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _has_net_admin() -> bool:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("CapEff:"):
                    cap = int(line.split()[1], 16)
                    return bool(cap & (1 << 12))  # CAP_NET_ADMIN
    except OSError:
        pass
    return False


# --------------------------------------------------------------------------- #
# launch one cell
# --------------------------------------------------------------------------- #
def launch(cell: dict, out_root: str, py: str, extra_set: list[str], timeout: int) -> dict:
    verb = "emulate" if cell.get("run_with") == "emulate" else "run"
    argv = [py, "-m", "cadet", verb, cell["config"],
            "--set", "outputs.root={}".format(out_root)]
    for s in extra_set:
        argv += ["--set", s]
    if cell.get("app_module"):
        argv += ["--app-module", cell["app_module"]]
    if cell.get("stage_module"):
        argv += ["--stage-module", cell["stage_module"]]
    env = _child_env()
    t0 = time.time()
    try:
        proc = subprocess.run(argv, cwd=str(REPO), env=env, timeout=timeout,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True)
        return {"rc": proc.returncode, "out": proc.stdout[-4000:], "wall": time.time() - t0,
                "argv": " ".join(argv[1:])}
    except subprocess.TimeoutExpired as exc:
        return {"rc": 124, "out": (exc.output or "")[-4000:] if isinstance(exc.output, str) else "",
                "wall": time.time() - t0, "argv": " ".join(argv[1:]), "timeout": True}


# --------------------------------------------------------------------------- #
# checks a..d
# --------------------------------------------------------------------------- #
# Stage modules whose terminus logs a decision per frame instead of acting on
# one. An open-loop cell has no stop to succeed at, so its evidence that the
# application did something is the decision log, not a client verdict.
OPEN_LOOP_STAGE_MODULES = ("examples.av_pipeline.openloop",)


def open_loop_cell(cell: dict) -> bool:
    """True if this cell's terminus only records decisions."""
    return cell.get("stage_module") in OPEN_LOOP_STAGE_MODULES


def client_verdicts(run_dir: Path, rec: dict) -> dict:
    """Return each client's own verdict, from run.json or from the files.

    Preferring the marker means a reader of the record alone sees what the
    harness saw. The files are read as a fallback so that a run produced by an
    older orchestrator - which wrote the files but not the field - is still
    judged on what its application reported.
    """
    clients = rec.get("clients")
    if isinstance(clients, dict) and clients:
        return clients
    found = {}
    for path in sorted(run_dir.glob("client_outcome_*.json")):
        node = path.name[len("client_outcome_") : -len(".json")]
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and "ok" in record:
            found[node] = {"ok": bool(record.get("ok")), "detail": record.get("detail") or ""}
    return found


def decision_rows(run_dir: Path) -> int:
    """Return how many decisions an open-loop run recorded, header excluded."""
    rows = 0
    for path in sorted(run_dir.glob("decisions_*.csv")):
        try:
            lines = [line for line in path.read_text().splitlines() if line.strip()]
        except OSError:
            continue
        rows += max(0, len(lines) - 1)
    return rows


def check_a(run_dir: Path, rid: str, py: str, cfg=None, cell=None) -> tuple[bool, str]:
    marker = run_dir / "run.json"
    if not marker.exists():
        return False, "run.json not written (run produced no marker)"
    rec = json.loads(marker.read_text())
    if rec.get("status") != "complete":
        return False, "status={!r} detail={!r}".format(rec.get("status"), rec.get("detail"))
    if not (rec.get("reason") or rec.get("detail")):
        return False, "no end reason recorded"

    # The application's own verdict, not just the orchestrator's. A run whose
    # processes exited and whose teardown was clean is not a run that did what
    # it was for: a braking client whose ego was removed before it stopped exits
    # 0 and tears down cleanly, and used to read as a pass.
    verdicts = client_verdicts(run_dir, rec)
    failed = sorted(node for node, v in verdicts.items() if not v.get("ok"))
    if failed:
        return False, "the application reported failure: {}".format(
            "; ".join("{}: {}".format(n, verdicts[n].get("detail") or "no detail") for n in failed)
        )
    if not verdicts:
        if cell is not None and open_loop_cell(cell):
            rows = decision_rows(run_dir)
            if rows < 1:
                return False, (
                    "open-loop cell recorded no decisions "
                    "(decisions_*.csv absent or empty)"
                )
        elif cfg is not None and any(getattr(n, "role", None) == "client" for n in cfg.nodes):
            return False, "no client verdict recorded (client_outcome_<node>.json absent)"
    # teardown: no active orchestrator run, no leftover container named for this run.
    leftovers = []
    try:
        r = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=10)
        leftovers = [n for n in r.stdout.split() if rid in n]
    except (OSError, subprocess.TimeoutExpired):
        pass
    if leftovers:
        return False, "leftover containers: {}".format(leftovers)
    detail = "complete; reason={!r}; teardown clean".format(rec.get("reason") or rec.get("detail"))
    if verdicts:
        detail += "; application ok ({})".format(", ".join(sorted(verdicts)))
    elif cell is not None and open_loop_cell(cell):
        detail += "; {} decisions logged".format(decision_rows(run_dir))
    return True, detail


def check_b(run_dir: Path, cfg) -> tuple[bool, str]:
    problems = []
    nodes = list(cfg.nodes)
    node_names = [n.name for n in nodes]
    clients = [n.name for n in nodes if n.role == "client"]
    servers = [n.name for n in nodes if n.role == "server"]
    requested = list(getattr(cfg.metrics, "collectors", []) or [])

    # per-frame record: braking writes *_latency_location_log.csv, dataset_vlm
    # samples_<client>.csv, the distributed sensing app's events_<client>.csv --
    # the name is app-specific.
    # Accept any per-client CSV that is not a device collector; e2e latency is
    # checked separately via the link report's measured block.
    coll_prefixes = {"cpu", "gpu", "energy", "bandwidth", "server_metrics", "link_report"}
    logs = list(run_dir.glob("*_latency_location_log.csv"))
    for c in clients:
        suffix = "_{}.csv".format(c)
        for p in run_dir.glob("*{}".format(suffix)):
            if p.name[:-len(suffix)] not in coll_prefixes:
                logs.append(p)
    if not logs:
        problems.append("no per-frame record (per-client app CSV) present")
    else:
        try:
            with open(logs[0]) as fh:
                rows = list(csv.reader(fh))
            if len(rows) < 2:
                problems.append("per-frame record {} has no data rows".format(logs[0].name))
        except OSError as e:
            problems.append("per-frame record unreadable: {}".format(e))

    # A link report per client, with a measured block, required when the cell
    # has links to report on. Both client runtimes write one: the pipeline's
    # did not, which was a gap in it rather than a property of staged
    # pipelines, and it now reports the same measured block the standalone
    # client does. A `none` cell has no link, so nothing is asked of it.
    has_links = bool(getattr(cfg, "links", None))
    if has_links:
        for c in clients:
            lr = run_dir / "link_report_{}.json".format(c)
            if not lr.exists():
                problems.append("no link_report_{}.json (emulated link present)".format(c))
                continue
            try:
                rep = json.loads(lr.read_text())
                if "measured" not in rep:
                    problems.append("link_report_{} lacks measured block".format(c))
            except (OSError, ValueError) as e:
                problems.append("link_report_{} unreadable: {}".format(c, e))

    # requested sampling collectors per node
    for coll in [x for x in requested if x in ("gpu", "cpu", "energy", "bandwidth")]:
        for nm in node_names:
            if not (run_dir / "{}_{}.csv".format(coll, nm)).exists():
                problems.append("missing {}_{}.csv".format(coll, nm))
    # queue = server per-request record
    for s in servers:
        if not (run_dir / "server_metrics_{}.csv".format(s)).exists():
            problems.append("missing server_metrics_{}.csv (queue)".format(s))

    # per-node device summary
    ds = run_dir / "device_summary.json"
    if not ds.exists():
        problems.append("no device_summary.json")
    else:
        try:
            summ = json.loads(ds.read_text())
            have = summ if isinstance(summ, dict) else {}
            # one entry per node (tolerate either {node: {...}} or {"nodes": {...}})
            entries = have.get("nodes", have)
            missing = [n for n in node_names if n not in entries]
            if missing:
                problems.append("device_summary missing nodes: {}".format(missing))
        except (OSError, ValueError) as e:
            problems.append("device_summary unreadable: {}".format(e))

    # provenance: the repository's commit and dirty flag, and - for a run that
    # placed a role in a container - what that image was. A node placed locally
    # or in a venv runs no image, so a run with no `images` block is only a gap
    # when some node asked for a container.
    warn = ""
    marker = run_dir / "run.json"
    if marker.exists():
        rec = json.loads(marker.read_text())
        cad = rec.get("cadet") or {}
        if "commit" not in cad or "dirty" not in cad:
            problems.append("run.json provenance missing commit/dirty")
        containerised = [
            n.name for n in nodes if getattr(n, "launcher", "local") in ("docker", "ssh_docker")
        ]
        recorded = rec.get("images") or {}
        missing_images = [n for n in containerised if n not in recorded]
        if missing_images:
            problems.append("run.json records no image for {}".format(missing_images))
        without_digest = [n for n, facts in sorted(recorded.items()) if not facts.get("id")]
        if without_digest:
            warn = " [warn: image recorded by tag only for {}]".format(without_digest)

    # Role logs present, and no uncaught traceback in a log CADET owns. A sim
    # hook is an external program that CADET only starts: ScenarioRunner raises
    # inside its own tick handler when a scenario destroys an actor mid-tick
    # (`dictionary changed size during iteration`), which says nothing about
    # the run CADET produced. So a hook's traceback is reported and does not
    # fail the cell; a role's does.
    logdir = run_dir / "logs"
    hook_logs = {"scenario_runner.log", "sim.log"}
    if not logdir.is_dir() or not any(logdir.iterdir()):
        problems.append("no role logs")
    else:
        hook_tracebacks = []
        for lf in logdir.glob("*.log"):
            if "Traceback (most recent call last)" in lf.read_text(errors="replace"):
                if lf.name in hook_logs:
                    hook_tracebacks.append(lf.name)
                else:
                    problems.append("traceback in {}".format(lf.name))
        if hook_tracebacks:
            warn += " [warn: traceback in the sim hook's own log: {}]".format(
                ", ".join(sorted(hook_tracebacks))
            )

    if problems:
        return False, "; ".join(problems)
    return True, "metrics/logs/provenance well-formed" + warn


_LAT_RE = re.compile(r"measured latency \(median over link reports\): ([\d.]+) ms")
_ENERGY_RE = re.compile(r"(\S+) energy:.*?energy_j(?:_total)?=([\d.]+)", re.I)


def _ground_truth(run_dir: Path, py: str) -> dict:
    """Offline reduction via `cadet summarize --context-only` -> the values the
    FM answer must match (median e2e latency ms, highest-energy node)."""
    env = _child_env()
    r = subprocess.run([py, "-m", "cadet", "summarize", str(run_dir), "--context-only"],
                       cwd=str(REPO), env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, timeout=120)
    txt = r.stdout
    gt = {"rc": r.returncode}
    m = _LAT_RE.search(txt)
    if m:
        gt["median_e2e_ms"] = float(m.group(1))
    # highest-energy node from the device section (energy_j per node)
    energies = {}
    for line in txt.splitlines():
        # `cadet summarize` reports a node's energy as `energy_j_total=`, so an
        # expression anchored on `energy_j=` matches no line and the check then
        # asks the answer to name a node the ground truth never found.
        em = _ENERGY_RE.search(line)
        if em:
            energies[em.group(1)] = float(em.group(2))
    if energies:
        gt["top_energy_node"] = max(energies, key=energies.get)
        gt["energies"] = energies
    gt["context_bytes"] = len(txt)
    return gt


def check_c(run_dir: Path, cfg, py: str) -> tuple[str, str]:
    """Returns (state, detail) where state in {pass, blocked, fail}."""
    gt = _ground_truth(run_dir, py)
    if gt.get("rc") != 0 or "median_e2e_ms" not in gt:
        return "fail", "offline reduction failed (rc={}, ctx={}B)".format(
            gt.get("rc"), gt.get("context_bytes"))
    gt_desc = "median_e2e={}ms top_energy_node={}".format(
        gt.get("median_e2e_ms"), gt.get("top_energy_node"))

    backend = getattr(cfg.analysis, "backend", "none")
    endpoint = getattr(cfg.analysis, "endpoint", None)
    key_env = getattr(cfg.analysis, "api_key_env", "CADET_FM_API_KEY")
    have_key = bool(os.environ.get(key_env))
    if backend == "none" or not endpoint or not have_key:
        return "blocked", ("no FM endpoint (analysis.backend={}, key_env {} {}); "
                           "ground truth verified offline: {}").format(
            backend, key_env, "set" if have_key else "unset", gt_desc)

    # Endpoint configured: run summarize + ask and compare programmatically.
    env = _child_env()
    rs = subprocess.run([py, "-m", "cadet", "summarize", str(run_dir)],
                        cwd=str(REPO), env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, timeout=180)
    if rs.returncode != 0 or not rs.stdout.strip():
        return "fail", "summarize rc={} empty={}".format(rs.returncode, not rs.stdout.strip())
    q = "what was the median end-to-end latency and which node used the most energy?"
    if "top_energy_node" not in gt:
        q = "what was the median end-to-end latency?"
    ra = subprocess.run([py, "-m", "cadet", "ask", str(run_dir), q],
                        cwd=str(REPO), env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, timeout=180)
    if ra.returncode != 0:
        return "fail", "ask rc={}".format(ra.returncode)
    ans = ra.stdout
    if not ans.strip():
        return "fail", "the endpoint returned an empty answer"

    # What this check is CADET's to answer for: the endpoint was reached, the
    # context was built from this run's own files, an answer came back, and the
    # answer is about that context rather than about nothing. A value from the
    # context appearing in the answer is what distinguishes the two.
    grounded = _answer_cites_context(ans, run_dir, py)
    if not grounded:
        return "fail", (
            "the answer cites no value from this run's context, so it is not "
            "an answer about this run"
        )

    # Whether the model read those numbers correctly is the model's to answer
    # for, and it varies between runs on identical data. Recorded, with its
    # answer and the ground truth, and not a verdict on the cell.
    nums = [float(x) for x in re.findall(r"([\d.]+)\s*ms", ans)]
    lat_ok = any(abs(n - gt["median_e2e_ms"]) <= max(5.0, 0.10 * gt["median_e2e_ms"]) for n in nums)
    # A cell whose nodes read no energy source has no highest-energy node, so
    # there is nothing for the answer to name.
    node_ok = "top_energy_node" not in gt or gt["top_energy_node"] in ans
    detail = "answered and grounded in this run's context; expected {}".format(gt_desc)
    if lat_ok and node_ok:
        return "pass", detail + "; model correct"
    return "pass", "{}; MODEL WRONG (lat_ok={} node_ok={}): {}".format(
        detail, lat_ok, node_ok, " ".join(ans.split())[:300]
    )


def _answer_cites_context(answer: str, run_dir: Path, py: str) -> bool:
    """True if the answer repeats a number the run's own context contains.

    The weakest check that still means "about this run": a model that was
    handed no context, or answered from its own invention, has no reason to
    produce a figure that appears in these files. Numbers are compared as the
    strings the context holds, with a tolerance of nothing: a near miss is the
    model's arithmetic, which is not what this is asking about.
    """
    # The context itself, not `cadet summarize`: summarize is a model call and
    # returns the model's prose. What this needs is the record the model was
    # given, which is what render_context builds.
    try:
        from cadet.analysis.context import render_context

        ctx = render_context(str(run_dir), 200000)
    except Exception:  # noqa: BLE001 - a context that cannot be built is not grounding
        return False
    # Figures worth citing: at least three characters, so a bare "0" or a node
    # count does not count as grounding.
    context_numbers = {n for n in re.findall(r"\d+\.\d+|\d{3,}", ctx)}
    if not context_numbers:
        return True  # nothing to cite; do not punish the answer for it
    answer_numbers = set(re.findall(r"\d+\.\d+|\d{3,}", answer))
    return bool(context_numbers & answer_numbers)


def check_d(run_dir: Path, rid: str) -> tuple[bool, str]:
    try:
        from cadet.dashboard import render as drender
        from cadet.dashboard import runs as druns
        rec = druns.read_run(str(run_dir))
        html = drender.run_view(rec, str(run_dir))
        cellhash = rid.split("-")[0]
        if html and cellhash in html:
            return True, "run_view rendered {} bytes".format(len(html))
        return False, "render produced no run_id-bearing HTML"
    except Exception as e:  # render is pure; any exception is a real failure
        return False, "render raised {}: {}".format(type(e).__name__, e)


# --------------------------------------------------------------------------- #
# per-cell driver
# --------------------------------------------------------------------------- #
def run_cell(cell: dict, out_root: str, py: str, extra_set: list[str], timeout: int) -> dict:
    cid = cell["id"]
    cfgpath = Path(REPO) / cell["config"]
    lines = line_count(cfgpath)
    res = {"id": cid, "source": cell["source"], "processing": cell["processing"],
           "network": cell["network"], "infra": cell["infra"],
           "launcher": cell.get("launcher", "local"), "config": cell["config"],
           "config_lines": lines, "checks": {}, "wall_s": 0.0}

    if lines > 25:
        res["warn"] = "config has {} lines (> 25 guideline)".format(lines)

    # The remote machine's address, which the spec deliberately does not carry.
    # Prepended to the operator's own overrides so that an explicit --set still
    # wins, and reported as blocked rather than failed when a cell needs a
    # second machine and none was named: that is a missing dependency, which is
    # what `blocked` means.
    settings = second_machine_settings()
    machine_set = second_machine_overrides(cell, settings)
    if not machine_set and missing_second_machine(cell):
        res["overall"] = "blocked"
        res["blocked_reason"] = (
            "this cell places a node on a second machine and none was named; "
            "set ${} or second_machine in {}".format(
                SECOND_MACHINE_ENV, LOCAL_OVERRIDE_FILE
            )
        )
        return res
    extra_set = machine_set + list(extra_set)

    # resolve + precheck
    try:
        cfg, run_dir, rid = resolve_cell(cell, out_root, extra_set)
    except Exception as e:
        res["overall"] = "fail"
        res["error"] = "config resolve failed: {}: {}".format(type(e).__name__, e)
        return res
    res["run_id"] = rid

    blocked = precheck(cell, cfg)
    if blocked:
        res["overall"] = "blocked"
        res["blocked_reason"] = blocked
        return res

    # launch
    out = launch(cell, out_root, py, extra_set, timeout)
    res["wall_s"] = round(out["wall"], 2)
    res["launch_cmd"] = out["argv"]
    if out["rc"] != 0:
        tail = out["out"].strip().splitlines()[-6:]
        res["overall"] = "fail"
        res["error"] = "run exited {}{}: {}".format(
            out["rc"], " (timeout)" if out.get("timeout") else "", " / ".join(tail))
        return res

    # The run id a containerised cell produced is not the one predicted from
    # its config, so the marker is found rather than assumed.
    run_dir, rid = locate_run(out_root, cfg, run_dir)
    res["run_id"] = rid

    # checks
    ok_a, d_a = check_a(run_dir, rid, py, cfg, cell)
    res["checks"]["a_completes_and_tears_down"] = {"ok": ok_a, "detail": d_a}
    ok_b, d_b = check_b(run_dir, cfg)
    res["checks"]["b_metrics_well_formed"] = {"ok": ok_b, "detail": d_b}
    st_c, d_c = check_c(run_dir, cfg, py)
    res["checks"]["c_fm_analysis"] = {"ok": st_c == "pass", "state": st_c, "detail": d_c}
    ok_d, d_d = check_d(run_dir, rid)
    res["checks"]["d_dashboard_renders"] = {"ok": ok_d, "detail": d_d}

    hard_fail = (not ok_a) or (not ok_b) or (not ok_d) or (st_c == "fail")
    if hard_fail:
        res["overall"] = "fail"
        failing = [k for k in CHECK_ORDER
                   if (k == "c_fm_analysis" and st_c == "fail") or
                      (k != "c_fm_analysis" and not res["checks"][k]["ok"])]
        res["error"] = "; ".join("{}: {}".format(k, res["checks"][k]["detail"]) for k in failing)
    elif st_c == "blocked":
        res["overall"] = "pass_partial"  # a..b..d green; FM check blocked (no endpoint)
    else:
        res["overall"] = "pass"
    return res


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def write_reports(spec: dict, results: list[dict], na_cells: list[dict],
                  pending_cells: list[dict], report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "acceptance_report.json").write_text(json.dumps(
        {"results": results, "na": na_cells, "pending": pending_cells}, indent=2))

    by = {}
    for r in results:
        by.setdefault(r["overall"], []).append(r)
    lines = ["# CADET acceptance matrix report", ""]
    lines.append("| overall | count |")
    lines.append("|---|---|")
    for k in ("pass", "pass_partial", "blocked", "fail", "not_run"):
        lines.append("| {} | {} |".format(k, len(by.get(k, []))))
    lines.append("| n/a | {} |".format(len(na_cells)))
    lines.append("| pending | {} |".format(len(pending_cells)))
    lines.append("")

    # The model's own accuracy, reported beside the cells rather than inside
    # their verdicts. A cell passes on CADET having produced a grounded answer;
    # whether the model read the numbers correctly varies between runs on
    # identical data, so it is a rate to watch and not a pass or a fail.
    asked = [
        r for r in results
        if (r.get("checks", {}).get("c_fm_analysis", {}).get("detail", "")).startswith(
            "answered and grounded"
        )
    ]
    if asked:
        wrong = [r for r in asked if "MODEL WRONG" in r["checks"]["c_fm_analysis"]["detail"]]
        lines.append("## Analysis model correctness")
        lines.append("")
        lines.append(
            "{} of {} answers correct ({:.0f}%). A wrong answer does not fail a "
            "cell: the check is that CADET reached the endpoint, built the "
            "context from the run's own files and got an answer grounded in "
            "it.".format(
                len(asked) - len(wrong), len(asked),
                100.0 * (len(asked) - len(wrong)) / len(asked),
            )
        )
        lines.append("")
        if wrong:
            lines.append("| cell | what the model got wrong |")
            lines.append("|---|---|")
            for r in wrong:
                detail = r["checks"]["c_fm_analysis"]["detail"]
                detail = detail.split("MODEL WRONG", 1)[1].lstrip(": ")
                lines.append("| {} | {} |".format(r["id"], detail.replace("|", "/")[:200]))
            lines.append("")

    lines.append("## Cells")
    lines.append("")
    lines.append("| cell | src | proc | net | infra | launcher | result | lines | wall s | note |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        note = r.get("error") or r.get("blocked_reason") or ""
        if r["overall"] in ("pass", "pass_partial"):
            c = r["checks"].get("c_fm_analysis", {})
            note = c.get("detail", "") if r["overall"] == "pass_partial" else ""
        note = (note[:120] + "...") if len(note) > 123 else note
        lines.append("| {} | {} | {} | {} | {} | {} | **{}** | {} | {} | {} |".format(
            r["id"], r["source"], r["processing"], r["network"], r["infra"],
            r.get("launcher", "local"), r["overall"], r.get("config_lines", "?"),
            r.get("wall_s", 0), note.replace("|", "/")))
    for r in pending_cells:
        lines.append("| {} | {} | {} | {} | {} | - | _pending_ | - | - | {} |".format(
            r["id"], r["source"], r["processing"], r["network"], r["infra"],
            r.get("reason", "").replace("|", "/")))
    for r in na_cells:
        lines.append("| {} | {} | {} | {} | {} | - | _n/a_ | - | - | rule: {} |".format(
            r["id"], r["source"], r["processing"], r.get("network", "*"),
            r.get("infra", "*"), r.get("rule", "")))
    (report_dir / "acceptance_report.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matrix", default=str(REPO / "tests/acceptance/matrix.yaml"))
    ap.add_argument("--out", default=None, help="scratch outputs root (default: mkdtemp)")
    ap.add_argument("--report-dir", default=None,
                    help="where report + resume state live (default: --out)")
    ap.add_argument("--python", default=sys.executable, help="interpreter with CADET deps")
    ap.add_argument("--cell", action="append", default=[],
                    help="cell id prefix filter (repeatable)")
    ap.add_argument("--layer", action="append", default=[],
                    help="LAYER=VALUE filter, e.g. network=delay (repeatable)")
    ap.add_argument("--status", default="valid", choices=["valid", "pending", "na", "all"])
    ap.add_argument("--set", dest="extra_set", action="append", default=[],
                    help="extra --set passthrough")
    ap.add_argument("--timeout", type=int, default=600, help="per-cell run timeout s")
    ap.add_argument("--resume", action="store_true", help="skip cells that already PASSED")
    args = ap.parse_args(argv)

    out_root = args.out or tempfile.mkdtemp(prefix="cadet_acc_")
    report_dir = Path(args.report_dir or out_root)
    spec = load_spec(Path(args.matrix))
    cells = spec["cells"]

    def layer_match(c):
        for f in args.layer:
            k, _, v = f.partition("=")
            key = {"source": "source", "processing": "processing",
                   "network": "network", "infra": "infrastructure"}.get(k, k)
            key = "infra" if key == "infrastructure" else key
            if str(c.get(key)) != v:
                return False
        return True

    def cell_match(c):
        return (not args.cell) or any(c["id"].startswith(p) for p in args.cell)

    na_cells = [c for c in cells if c.get("status") == "na"]
    pending_cells = [c for c in cells if c.get("status") == "pending"]
    valid = [c for c in cells if c.get("status") == "valid" and cell_match(c) and layer_match(c)]
    # Every valid cell, filtered or not: the report is about the matrix, and a
    # cell this pass did not select still has whatever a previous pass found.
    # The matrix cannot be covered by one pass, because its tiers need
    # different environments - a local role needs this host, a containerised
    # one needs the image, an emulated one needs the privileged emulation
    # image - so a report that showed only the pass would never show the whole.
    all_valid = [c for c in cells if c.get("status") == "valid"]

    prior = {}
    rfile = report_dir / "acceptance_report.json"
    if args.resume and rfile.exists():
        with contextlib.suppress(Exception):
            for r in json.loads(rfile.read_text()).get("results", []):
                prior[r["id"]] = r

    results = []
    todo = valid if args.status in ("valid", "all") else []
    print("acceptance: {} valid cells, out={}".format(len(todo), out_root), file=sys.stderr)
    for cell in todo:
        if args.resume and prior.get(cell["id"], {}).get("overall") == "pass":
            results.append(prior[cell["id"]])
            print("  = {} (resumed pass)".format(cell["id"]), file=sys.stderr)
            continue
        r = run_cell(cell, out_root, args.python, args.extra_set, args.timeout)
        results.append(r)
        print("  {} {} ({}s){}".format(
            {"pass": "PASS", "pass_partial": "PASS*", "blocked": "----",
             "fail": "FAIL"}.get(r["overall"], "?"),
            r["id"], r.get("wall_s", 0),
            "  " + (r.get("error") or r.get("blocked_reason") or "")), file=sys.stderr)

    done = {r["id"] for r in results}
    carried = [prior[c["id"]] for c in all_valid if c["id"] not in done and c["id"] in prior]
    not_run = [
        {
            "id": c["id"],
            "source": c["source"],
            "processing": c["processing"],
            "network": c["network"],
            "infra": c["infra"],
            "launcher": c.get("launcher", "local"),
            "config": c.get("config", ""),
            "config_lines": -1,
            "checks": {},
            "wall_s": 0.0,
            "overall": "not_run",
            "blocked_reason": "not selected by any pass recorded here",
        }
        for c in all_valid
        if c["id"] not in done and c["id"] not in prior
    ]
    order = {c["id"]: i for i, c in enumerate(all_valid)}
    results = sorted(results + carried + not_run, key=lambda r: order.get(r["id"], 0))
    write_reports(spec, results, na_cells, pending_cells, report_dir)
    print("\nreport: {}".format(report_dir / "acceptance_report.md"), file=sys.stderr)
    return 1 if any(r.get("overall") == "fail" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
