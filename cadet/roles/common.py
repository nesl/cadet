"""Shared wiring for the client/server runtimes.

What this is for:
    Small helpers both role runtimes use: select one run config from a file
    (overrides + sweep run index), look up nodes/links, and prepare the output
    directory (creating it and saving the resolved config alongside results).

What it contains:
    - :func:`select_run_config` -> (config, run_id_base)
    - :func:`node_by_role`, :func:`link_into_node`
    - :func:`output_dir` (pure path), :func:`prepare_output_dir` (creates it)
    - :class:`StopSignal`: turn the orchestrator's stop signal into a flag a
      role's own loop can act on, so it reaches its own shutdown path, and then
      hand the process back to the signal so its exit status still says what
      happened to it

How it fits into CADET:
    Used by :mod:`cadet.roles.client` and :mod:`cadet.roles.server`. Output paths
    come only from the resolved config's ``outputs`` section — never a curated
    results directory.
"""

import os
import signal
import sys
from typing import Callable, List, Tuple

import yaml

from cadet.config.loader import apply_overrides, config_hash, expand_sweep, load_config_data
from cadet.config.schema import Config


class StopSignal:
    """A role's stop signal: first a flag its own loop acts on, then its exit.

    The orchestrator stops a role with SIGTERM and escalates to SIGKILL after a
    grace. Python's default action for SIGTERM ends the process at once, so a
    role that is stopped rather than finishing by itself never ran its own
    shutdown path: a server's ``finally`` did not execute and its per-request
    metrics were never written. Measured: the braking app wrote
    ``server_metrics_*.csv`` because its app stops the loop itself after the
    brake, while the reference app, whose server waits for work until it is
    stopped, produced none at all.

    :meth:`install` makes the handler set a flag through ``on_stop``; the loop
    notices on its next turn and unwinds normally. Raising from the handler
    would skip the same shutdown path by another route, which is also why the
    loop's wait has to be bounded rather than indefinite.

    :meth:`finish` then **hands the process back to the signal**, so its exit
    status is the one a signalled process has. That is not cosmetic: the
    orchestrator reads a worker's exit code during the active phase to tell a
    guard that finished its job (code 0 - the braking server is single-shot by
    design) from one that died under it (non-zero - ``node exited
    unexpectedly``). A server that flushed its records and then exited 0 would
    make an operator's ``kill`` look like a clean finish, which is exactly the
    failure the lifecycle matrix exists to catch.

    Signals can only be installed from the main thread, and a role process is
    one process per role, so this is used from the role's own ``run``. Off the
    main thread (a test driving a runtime in-process) :meth:`install` is a
    no-op rather than an error, and :meth:`finish` then has nothing to do.
    """

    def __init__(self):
        self.signum = None

    def install(self, on_stop: Callable[[], None], log=None) -> None:
        """Install the handler for SIGTERM and SIGINT."""
        log = log or (lambda m: None)

        def handler(signum, _frame):
            self.signum = signum
            log("[role] stop signal {}; finishing".format(signum))
            on_stop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                # Not the main thread, or a platform without the signal.
                return

    def finish(self) -> None:
        """Exit as a signalled process, now that the records are written."""
        if self.signum is None:
            return
        signum, self.signum = self.signum, None
        # Dying by a signal does not flush Python's buffers, and a role's log is
        # read by a person after a failure: without this the shutdown lines are
        # lost whenever stdout is a file, which it always is under the launcher.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (ValueError, OSError):
                pass
        try:
            signal.signal(signum, signal.SIG_DFL)
        except (ValueError, OSError):
            return
        os.kill(os.getpid(), signum)


def select_run_config(path: str, overrides: List[str], run_index: int) -> Tuple[Config, str]:
    """Load + override + expand a config and return the ``run_index``-th run.

    Returns ``(config, run_id_base)`` where ``run_id_base`` is the config hash
    (the per-repetition ``-r<rep>`` suffix is added by the caller).
    """
    data = apply_overrides(load_config_data(path), overrides)
    runs = expand_sweep(data)
    if run_index < 0 or run_index >= len(runs):
        raise IndexError("run index {} out of range (0..{})".format(run_index, len(runs) - 1))
    cfg = runs[run_index]
    return cfg, config_hash(cfg)


def node_by_role(config: Config, role: str):
    """Return the first node with ``role`` (or ``None``)."""
    for node in config.nodes:
        if node.role == role:
            return node
    return None


def link_into_node(config: Config, node_name: str):
    """Return the first link whose ``to`` endpoint is ``node_name`` (or ``None``)."""
    for link in config.links:
        if link.to == node_name:
            return link
    return None


def output_dir(config: Config, run_id: str) -> str:
    """Return the output directory path for a run (does not create it).

    Directory = ``outputs.root/<outputs.naming formatted>``. Kept pure so the
    orchestrator can locate a run's outputs (e.g. for resume) without side effects.
    """
    rel = config.outputs.naming.format(
        experiment=config.experiment.name, app=config.app.plugin, run_id=run_id
    )
    return os.path.join(config.outputs.root, rel)


def prepare_output_dir(config: Config, run_id: str) -> str:
    """Create the output directory for this run and save the resolved config.

    Directory = :func:`output_dir`. The resolved config is written as
    ``config.yaml`` next to the results.
    """
    out_dir = output_dir(config, run_id)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as fh:
        yaml.safe_dump(config.model_dump(mode="json", by_alias=True), fh, sort_keys=False)
    return out_dir
