"""CADET command-line interface.

What this is for:
    The ``cadet`` command. Subcommands:
      - ``validate`` / ``show``: check and inspect a config.
      - ``serve`` / ``client`` / ``load``: run a single role process (used by the
        orchestrator, or by hand).
      - ``run`` / ``sweep``: orchestrate a whole run / matrix (start servers +
        clients, collect, tear down; sweep resumes completed cells).
      - ``status``: report orchestrator-started processes.
      - ``dashboard``: serve a local page over a tree of run directories.
      - ``variance``: reduce a run tree's repetitions into one variance report.
      - ``summarize`` / ``ask``: read a run or a sweep and put it to a model.
      - ``down``: recover from a run that never reached its teardown, removing
        the processes, containers, networks, namespaces, bridges, veths and
        qdiscs CADET owns.

What it contains:
    - ``cmd_*``: one function per subcommand.
    - :func:`build_parser` / :func:`main`: argparse wiring and entry point.

How it fits into CADET:
    Thin front-end over :mod:`cadet.config.loader`. Exposed as the ``cadet``
    console script and runnable as ``python -m cadet``. Returns a process exit
    code (0 ok, non-zero on error) so it composes in scripts and CI.

How to configure it:
    ``cadet validate path/to/config.yaml [--set a.b=1 --set c=d]``.
"""

import argparse
import os
import sys
from typing import List, Optional

import yaml
from pydantic import ValidationError

from cadet.config.loader import (
    ConfigError,
    format_validation_error,
    load_run_configs,
    resolve_app_params,
    run_id,
)


def _load_app_modules(args) -> None:
    """Import --app-module modules (and $CADET_APP_MODULES) to register plugins."""
    import os

    from cadet.core.apps import import_app_modules

    mods = list(getattr(args, "app_module", None) or [])
    env = os.environ.get("CADET_APP_MODULES")
    if env:
        mods.extend(m for m in env.split(os.pathsep) if m)
    import_app_modules(mods)


def _load_stage_modules(args) -> None:
    """Import --stage-module modules (and $CADET_STAGE_MODULES) to register stages.

    The orchestrator spawns the role processes and threads ``--app-module`` on their
    argv, but not ``--stage-module``; so ``run``/``sweep`` publish the stage modules
    through ``$CADET_STAGE_MODULES``, which the spawned ``serve``/``client`` read here
    (the same pattern as ``$CADET_APP_MODULES``). Keeps the orchestrator untouched.
    """
    import os

    from cadet.pipeline.registry import import_stage_modules

    mods = list(getattr(args, "stage_module", None) or [])
    env = os.environ.get("CADET_STAGE_MODULES")
    if env:
        mods.extend(m for m in env.split(os.pathsep) if m)
    if mods:
        import_stage_modules(mods)


def _publish_stage_env(args) -> None:
    """Publish --stage-module to $CADET_STAGE_MODULES so spawned roles inherit it."""
    import os

    mods = list(getattr(args, "stage_module", None) or [])
    if mods:
        existing = os.environ.get("CADET_STAGE_MODULES", "")
        parts = [p for p in (existing.split(os.pathsep) if existing else []) if p]
        for m in mods:
            if m not in parts:
                parts.append(m)
        os.environ["CADET_STAGE_MODULES"] = os.pathsep.join(parts)


def _missing_path(exc: FileNotFoundError, named: str) -> str:
    """Return the file that is actually missing, which is not always the named one.

    A config that ``extends`` a base names the base as a path of its own, so the
    file that cannot be opened is often the base rather than the config on the
    command line. Reporting the named file for a missing base sends the reader
    to a file that is there.
    """
    missing = getattr(exc, "filename", None)
    if not missing or os.path.realpath(missing) == os.path.realpath(named):
        return named
    return "{} (extended by {})".format(missing, named)


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a config file; print a summary or readable errors."""
    try:
        run_configs, _base = load_run_configs(args.config, args.set or [])
    except FileNotFoundError as exc:
        print(
            "error: config not found: {}".format(_missing_path(exc, args.config)),
            file=sys.stderr,
        )
        return 2
    except ConfigError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    except ValidationError as exc:
        print("invalid config: {}".format(args.config), file=sys.stderr)
        print(format_validation_error(exc), file=sys.stderr)
        return 1

    # The config is schema-valid; now reject any option that is declared but not
    # implemented yet (never silently ignored).
    from cadet.config.implemented import (
        allow_experimental,
        experimental_features,
        unimplemented_features,
    )

    for cfg in run_configs:
        # `validate` takes the opt-in from the config, not a flag: it is checking
        # whether *this config* is runnable, and a config that needs unverified
        # features should say so in the config.
        allowed = allow_experimental(cfg, getattr(args, "allow_experimental", None))
        missing = unimplemented_features(cfg, allow_experimental=allowed)
        if missing:
            print("invalid config: {}".format(args.config), file=sys.stderr)
            for feature in missing:
                print("  will not run: {}".format(feature), file=sys.stderr)
            return 1
        if allowed:
            for feature in experimental_features(cfg):
                print("  EXPERIMENTAL, accepted by opt-in: {}".format(feature))

    # The app plugin, and then its params against the plugin's model. Checked
    # whether or not --app-module was given, because "valid" has to mean "would
    # run": a config naming a plugin nothing has registered validated clean and
    # then failed at `cadet run`, which taught a reader that validate does not
    # mean much. The plugin lookup's own error names the registered plugins and
    # how to name a module, so there is nothing to add to it here.
    try:
        _load_app_modules(args)
        for cfg in run_configs:
            resolve_app_params(cfg)
    except ConfigError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    except ValidationError as exc:
        print("invalid app params in {}:".format(args.config), file=sys.stderr)
        print(format_validation_error(exc), file=sys.stderr)
        return 1

    # If stage modules were given, check each pipeline impl against the registry.
    _load_stage_modules(args)
    from cadet.pipeline.validate import validate_pipeline

    for cfg in run_configs:
        problems = validate_pipeline(cfg)
        if problems:
            print("invalid pipeline in {}:".format(args.config), file=sys.stderr)
            for p in problems:
                print("  {}".format(p), file=sys.stderr)
            return 1

    n = len(run_configs)
    print("OK: {} valid ({} run config{})".format(args.config, n, "" if n == 1 else "s"))
    # Show the resolved run ids for the first repetition so sweeps are visible.
    for cfg in run_configs:
        print("  - {}  [{}]".format(cfg.experiment.name, run_id(cfg, 0)))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Print one resolved run config as YAML (sweep run ``--run``)."""
    try:
        run_configs, _base = load_run_configs(args.config, args.set or [])
    except FileNotFoundError as exc:
        print(
            "error: config not found: {}".format(_missing_path(exc, args.config)),
            file=sys.stderr,
        )
        return 2
    except ConfigError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    except ValidationError as exc:
        print("invalid config: {}".format(args.config), file=sys.stderr)
        print(format_validation_error(exc), file=sys.stderr)
        return 1

    if args.run < 0 or args.run >= len(run_configs):
        print(
            "error: --run {} out of range (0..{})".format(args.run, len(run_configs) - 1),
            file=sys.stderr,
        )
        return 2

    cfg = run_configs[args.run]
    # --resolved fills in every default; otherwise show only fields actually set.
    payload = cfg.model_dump(mode="json", by_alias=True, exclude_unset=not args.resolved)
    print("# run_id: {}".format(run_id(cfg, 0)))
    print(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False).rstrip())
    return 0


def _model_search_dirs(extra: Optional[List[str]]) -> List[str]:
    """Model artifact search path: --model-dir(s), $CADET_MODEL_DIR, then cwd."""
    import os

    dirs = list(extra or [])
    env = os.environ.get("CADET_MODEL_DIR")
    if env:
        dirs.extend(env.split(os.pathsep))
    dirs.append(os.getcwd())
    return dirs


def cmd_serve(args: argparse.Namespace) -> int:
    """Run a server node for a config (blocks serving one client)."""
    _load_app_modules(args)  # register the app plugin(s) named by --app-module
    _load_stage_modules(args)  # register pipeline stages named by --stage-module
    from cadet.roles.server import build_server

    try:
        runtime = build_server(
            args.config,
            args.set or [],
            args.run,
            _model_search_dirs(args.model_dir),
            node_name=args.node,
            rep=args.rep,
            bind_host=args.bind_host,
        )
    except (ConfigError, ValidationError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    runtime.run()
    return 0


def cmd_client(args: argparse.Namespace) -> int:
    """Run a client node for a config (blocks until the run ends)."""
    _load_app_modules(args)  # register the app plugin(s) named by --app-module
    _load_stage_modules(args)  # register pipeline stages named by --stage-module
    from cadet.roles.client import build_client

    try:
        runtime = build_client(
            args.config, args.set or [], args.run, args.rep, node_name=args.node
        )
    except (ConfigError, ValidationError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    runtime.run()
    return 0


def cmd_load(args: argparse.Namespace) -> int:
    """Run the load generator (N synthetic clients -> server(s))."""
    _load_app_modules(args)
    from cadet.roles.load import build_load

    try:
        runtime = build_load(args.config, args.set or [], args.run)
    except (ConfigError, ValidationError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    runtime.run()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Orchestrate one run: preflight, start servers+clients, collect, tear down."""
    _load_app_modules(args)
    _publish_stage_env(args)  # so spawned serve/client inherit --stage-module
    _load_stage_modules(args)
    from cadet.orchestrator import Orchestrator, PreflightError
    from cadet.orchestrator.placement import PlacementError
    from cadet.sim.containers import SimSideError

    orch = Orchestrator()
    try:
        result = orch.run_one(
            args.config,
            args.set or [],
            args.run,
            args.rep,
            list(getattr(args, "app_module", None) or []),
            _model_search_dirs(args.model_dir),
            allow_experimental=getattr(args, "allow_experimental", False) or None,
        )
    except (
        ConfigError,
        ValidationError,
        PreflightError,
        PlacementError,
        SimSideError,
    ) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    return 0 if result.ok else 1


def cmd_emulate(args: argparse.Namespace) -> int:
    """Run one experiment with its roles inside an emulated network.

    Intended to be invoked inside the emulation container, which is the only
    place that can put a process into a Mininet host namespace. The host-side
    orchestrator reaches it with ``docker exec``.
    """
    _load_app_modules(args)
    # As `run` and `sweep` do, and for the same two reasons: this process has to
    # resolve a `pipeline` block's stages itself, and the roles it starts in the
    # emulated hosts are told which modules to import. Without these two lines
    # `--stage-module` was accepted here and did nothing, and a staged pipeline
    # in an emulated network died on the first stage its config named.
    _publish_stage_env(args)
    _load_stage_modules(args)
    from cadet.network.netwaggle.session import run_emulated
    from cadet.orchestrator import PreflightError
    from cadet.sim.containers import SimSideError

    try:
        result = run_emulated(
            args.config,
            args.set or [],
            args.run,
            args.rep,
            list(getattr(args, "app_module", None) or []),
            _model_search_dirs(args.model_dir),
            interactive=args.interactive,
            allow_experimental=getattr(args, "allow_experimental", False) or None,
        )
    except (ConfigError, ValidationError, PreflightError, SimSideError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 3
    return 0 if result.ok else 1


def cmd_sweep(args: argparse.Namespace) -> int:
    """Orchestrate every run x rep in the config; --resume skips completed ones."""
    _load_app_modules(args)
    _publish_stage_env(args)  # so spawned serve/client inherit --stage-module
    _load_stage_modules(args)
    from cadet.orchestrator import Orchestrator, PreflightError
    from cadet.orchestrator.placement import PlacementError
    from cadet.sim.containers import SimSideError

    orch = Orchestrator()
    try:
        results = orch.sweep(
            args.config,
            args.set or [],
            list(getattr(args, "app_module", None) or []),
            _model_search_dirs(args.model_dir),
            reps=args.reps,
            allow_experimental=getattr(args, "allow_experimental", False) or None,
            resume=True if getattr(args, "resume", False) else None,
        )
    except (
        ConfigError,
        ValidationError,
        PreflightError,
        PlacementError,
        SimSideError,
    ) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    failed = [r for r in results if not r.ok]
    print("sweep: {}/{} runs ok".format(len(results) - len(failed), len(results)))
    return 0 if not failed else 1


def cmd_status(args: argparse.Namespace) -> int:
    """Print the orchestrator's recorded active runs (processes it started)."""
    from cadet.orchestrator import Orchestrator
    from cadet.orchestrator.state import pid_alive

    data = Orchestrator().status()
    if not data:
        print("no active runs")
        return 0
    for rid, rec in sorted(data.items()):
        procs = rec.get("procs", [])
        alive = sum(1 for p in procs if p.get("pid") and pid_alive(p["pid"]))
        print("{}  dir={}  procs={} ({} alive)".format(rid, rec.get("dir"), len(procs), alive))
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    """Remove every CADET-owned leftover and report each one.

    Printing the list is the point as much as the removal: the operator has to
    be able to see what was taken down, because the alternative - a cleanup that
    reports only a count - is how an indiscriminate kill goes unnoticed.
    """
    from cadet.orchestrator import Orchestrator

    found = Orchestrator().down()
    if not found:
        print("nothing left to clean up")
        return 0
    for leftover in found:
        print("removed {}".format(leftover))
    failed = [item for item in found if not item.removed]
    print("{} leftover(s) removed, {} could not be".format(len(found) - len(failed), len(failed)))
    return 1 if failed else 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Serve the local dashboard over a tree of run directories.

    Reads only what a run already wrote and serves it from the standard
    library's HTTP server, with nothing fetched at page load: the page is shown
    on machines with no network. Does not start, change or tear down a run.
    """
    import os

    from cadet.dashboard.server import serve

    if not os.path.isdir(args.runs_dir):
        print("not a directory: {}".format(args.runs_dir), file=sys.stderr)
        return 2
    if args.refresh < 1:
        print("error: --refresh must be at least 1 second", file=sys.stderr)
        return 2
    return serve(args.runs_dir, host=args.host, port=args.port, refresh=args.refresh)


def cmd_variance(args: argparse.Namespace) -> int:
    """Reduce a tree of repetitions into one variance record.

    Reads only what a run already wrote; writes the report into the run tree
    (a variance report is a result, so it does not belong in the repository).
    """
    import os

    from cadet.repro import reduce_tree, render_markdown, write_reports

    root = args.run_dir
    if not os.path.isdir(root):
        print("not a directory: {}".format(root), file=sys.stderr)
        return 2
    report = reduce_tree(root)
    if not report["repetition_dirs"]:
        print(
            "no repetition found under {}: a repetition is a directory with a "
            "run.json".format(root),
            file=sys.stderr,
        )
        return 1
    if args.stdout:
        print(render_markdown(report))
    out_dir = args.out or root
    for path in write_reports(report, out_dir):
        print("wrote {}".format(path))
    # A metric missing from some repetitions, or a run that did not complete, is
    # a finding; the exit code says so without the caller parsing the report.
    findings = [f for cell in report["cells"] for f in cell["findings"]]
    return 3 if findings else 0


def _analyse(args: argparse.Namespace, question: Optional[str]) -> int:
    """Shared body of ``summarize`` and ``ask``: reduce, then (maybe) send.

    The two differ only in the question, so they are one function: a summary
    that was produced by a different path from an answer would be comparable to
    neither.

    Reads a run tree and writes nothing to it. The backend is resolved through
    :func:`cadet.analysis.make_backend`, which is the module-level name the
    tests replace with a stub - so the suite exercises this whole path without
    opening a socket.
    """
    import os

    from pydantic import ValidationError as _ValidationError

    import cadet.analysis as analysis

    if not os.path.isdir(args.path):
        print("not a directory: {}".format(args.path), file=sys.stderr)
        return 2
    try:
        config = analysis.load_analysis_config(args.config, args.path)
    except analysis.NotConfigured as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    except _ValidationError as exc:
        print("invalid `analysis` section:", file=sys.stderr)
        print(format_validation_error(exc), file=sys.stderr)
        return 1

    try:
        record = analysis.render_context(args.path, config.max_context_bytes)
    except analysis.ContextError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2

    # --context-only needs no backend and sends nothing: it is how a reader
    # sees exactly what would be sent before deciding to send it, and it works
    # in the default state where everything else refuses.
    if args.context_only:
        print(record, end="" if record.endswith("\n") else "\n")
        print(
            "({} bytes of context, budget {})".format(
                len(record.encode("utf-8")), config.max_context_bytes
            ),
            file=sys.stderr,
        )
        return 0

    try:
        backend = analysis.make_backend(config)
    except analysis.NotConfigured as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    try:
        print(analysis.respond(record, question, backend))
    except analysis.BackendError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    """Summarise one run directory, or every sweep in a tree of them."""
    return _analyse(args, None)


def cmd_ask(args: argparse.Namespace) -> int:
    """Answer one question about a run directory or a tree of them."""
    return _analyse(args, args.question)


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="cadet", description="CADET experiment toolkit")
    sub = parser.add_subparsers(dest="command")

    def _allow_experimental(p):
        p.add_argument(
            "--allow-experimental",
            action="store_true",
            help=(
                "accept config options that are implemented but not verified "
                "end to end; recorded in run.json. Options with no "
                "implementation at all stay refused."
            ),
        )

    def _app_module(p):
        p.add_argument(
            "--app-module",
            action="append",
            metavar="MODULE",
            help="import an app plugin module (repeatable); e.g. examples.braking.flat.app",
        )

    def _stage_module(p):
        p.add_argument(
            "--stage-module",
            action="append",
            metavar="MODULE",
            help="import a pipeline stage module (repeatable); e.g. cadet.pipeline.stages.braking",
        )

    v = sub.add_parser("validate", help="validate an experiment config")
    v.add_argument("config", help="path to a YAML config")
    v.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="override a dotted config path (repeatable)",
    )
    _app_module(v)
    _stage_module(v)
    v.set_defaults(func=cmd_validate)

    s = sub.add_parser("show", help="print a fully resolved config")
    s.add_argument("config", help="path to a YAML config")
    s.add_argument("--resolved", action="store_true", help="fill in all defaults")
    s.add_argument("--run", type=int, default=0, help="sweep run index to show (default 0)")
    s.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="override a dotted config path (repeatable)",
    )
    _app_module(s)
    s.set_defaults(func=cmd_show)

    srv = sub.add_parser("serve", help="run a server node for a config")
    srv.add_argument("config", help="path to a YAML config")
    srv.add_argument("--node", default="server", help="node name to run (role server)")
    srv.add_argument("--run", type=int, default=0, help="sweep run index (default 0)")
    srv.add_argument("--rep", type=int, default=0, help="repetition index (default 0)")
    srv.add_argument("--model-dir", action="append", help="model artifact search dir (repeatable)")
    srv.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config path")
    srv.add_argument(
        "--bind-host",
        default=None,
        help=(
            "address to listen on, overriding the node's configured host. Set by "
            "the placement for a role in a container that does not hold the "
            "address its peers reach it at; not a config override, because the "
            "config is what both sides derive the run id from"
        ),
    )
    _app_module(srv)
    _stage_module(srv)
    srv.set_defaults(func=cmd_serve)

    cl = sub.add_parser("client", help="run a client node for a config")
    cl.add_argument("config", help="path to a YAML config")
    cl.add_argument("--node", default="client", help="node name to run (role client)")
    cl.add_argument("--run", type=int, default=0, help="sweep run index (default 0)")
    cl.add_argument("--rep", type=int, default=0, help="repetition index (default 0)")
    cl.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config path")
    _app_module(cl)
    _stage_module(cl)
    cl.set_defaults(func=cmd_client)

    ld = sub.add_parser("load", help="run the load generator (N synthetic clients)")
    ld.add_argument("config", help="path to a YAML config")
    ld.add_argument("--run", type=int, default=0, help="sweep run index (default 0)")
    ld.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config path")
    _app_module(ld)
    ld.set_defaults(func=cmd_load)

    rn = sub.add_parser("run", help="orchestrate one run (servers + clients, teardown)")
    rn.add_argument("config", help="path to a YAML config")
    rn.add_argument("--run", type=int, default=0, help="sweep run index (default 0)")
    rn.add_argument("--rep", type=int, default=0, help="repetition index (default 0)")
    rn.add_argument("--model-dir", action="append", help="model artifact search dir (repeatable)")
    rn.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config path")
    _app_module(rn)
    _stage_module(rn)
    _allow_experimental(rn)
    rn.set_defaults(func=cmd_run)

    em = sub.add_parser(
        "emulate",
        help="run inside an emulated network (use within the emulation container)",
    )
    em.add_argument("config", help="path to a YAML config")
    em.add_argument("--run", type=int, default=0, help="sweep run index (default 0)")
    em.add_argument("--rep", type=int, default=0, help="repetition index (default 0)")
    em.add_argument("--model-dir", action="append", help="model artifact search dir (repeatable)")
    em.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config path")
    em.add_argument(
        "--interactive",
        action="store_true",
        help="drop into the Mininet CLI instead of running the roles (debugging)",
    )
    _app_module(em)
    _stage_module(em)
    _allow_experimental(em)
    em.set_defaults(func=cmd_emulate)

    sw = sub.add_parser("sweep", help="orchestrate all runs x reps")
    sw.add_argument("config", help="path to a YAML config")
    sw.add_argument("--reps", type=int, default=None, help="override repetitions per run")
    sw.add_argument(
        "--resume",
        action="store_true",
        help="skip cells that already completed (off by default: the original re-ran them)",
    )
    sw.add_argument("--model-dir", action="append", help="model artifact search dir (repeatable)")
    sw.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config path")
    _app_module(sw)
    _stage_module(sw)
    _allow_experimental(sw)
    sw.set_defaults(func=cmd_sweep)

    va = sub.add_parser(
        "variance",
        help="reduce N repetitions under a run tree into a variance report",
    )
    va.add_argument("run_dir", help="directory holding the repetition directories")
    va.add_argument(
        "--out",
        default=None,
        metavar="DIR",
        help="where to write the report (default: the directory being reduced)",
    )
    va.add_argument(
        "--stdout", action="store_true", help="also print the Markdown report"
    )
    va.set_defaults(func=cmd_variance)

    # Read where argparse needs them so ``cadet dashboard --help`` states the
    # real defaults, without every other subcommand importing an HTTP server.
    # Loopback by default: a run tree is measurement data on someone's machine
    # and the default must not publish it to the network the laptop is on.
    from cadet.dashboard.server import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_REFRESH_S

    db = sub.add_parser(
        "dashboard",
        help="serve a local page over a tree of run directories",
    )
    db.add_argument("runs_dir", help="directory holding the run directories to read")
    db.add_argument(
        "--host",
        default=DEFAULT_HOST,
        metavar="HOST",
        help="address to bind (default: %(default)s, loopback only)",
    )
    db.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        metavar="PORT",
        help="port to bind (default: %(default)s)",
    )
    db.add_argument(
        "--refresh",
        type=int,
        default=DEFAULT_REFRESH_S,
        metavar="SECONDS",
        help=(
            "how often the live view re-requests itself while a run or a sweep "
            "is in progress (default: %(default)s)"
        ),
    )
    db.set_defaults(func=cmd_dashboard)

    def _analysis_args(p, path_help):
        p.add_argument("path", help=path_help)
        p.add_argument(
            "--config",
            default=None,
            metavar="FILE",
            help=(
                "config whose `analysis` section says where to send the record "
                "(default: the run directory's own config.yaml)"
            ),
        )
        p.add_argument(
            "--context-only",
            action="store_true",
            help="print the record that would be sent, and send nothing",
        )

    sm = sub.add_parser(
        "summarize",
        help="summarise a run directory or a tree of them with a model",
    )
    _analysis_args(sm, "a run directory, or a directory holding run directories")
    sm.set_defaults(func=cmd_summarize)

    ak = sub.add_parser(
        "ask",
        help="ask a model one question about a run directory or a tree of them",
    )
    _analysis_args(ak, "a run directory, or a directory holding run directories")
    ak.add_argument("question", help="the question, in quotes")
    ak.set_defaults(func=cmd_ask)

    st = sub.add_parser("status", help="show active orchestrator runs")
    st.set_defaults(func=cmd_status)

    dn = sub.add_parser(
        "down",
        help="recover from an interrupted run: remove CADET-owned leftovers",
    )
    dn.set_defaults(func=cmd_down)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the ``cadet`` console script and ``python -m cadet``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
