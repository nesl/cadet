"""Run-state store for ``cadet status`` / ``cadet down``.

A JSON record under ``<cwd>/.cadet/state.json`` tracks what an orchestrator run
has started: a map of ``run_id -> {dir, procs, netem_interfaces}``, with one pid
file per launched process under ``<cwd>/.cadet/pids/<run_id>/<name>.pid``. The
pid files make ownership checkable: a pattern match on a command line cannot
distinguish a CADET role from any other process listening on the same port, so
cleanup needs direct evidence that CADET started something; a file written at
launch is that evidence, and it persists even if the orchestrator dies mid-run.
Used by :mod:`cadet.orchestrator.runner` (records each process at launch and
clears on teardown) and by :func:`cadet.orchestrator.recovery.sweep` (reads the
pid files and interfaces to power ``cadet down``). The state root is fixed at
``<cwd>/.cadet`` so ``cadet status`` and ``cadet down`` need no config.
"""

import json
import os
import shutil
import signal
from typing import Any, Dict, List, Optional

STATE_DIRNAME = ".cadet"
STATE_FILENAME = "state.json"
# One file per launched process, named after the role. It holds the pid and,
# separated by a space, the process's start time in clock ticks since boot.
#
# The start time is what makes the record checkable after the fact. A pid is
# reused; a (pid, start time) pair is not. Without it a recorded pid is only
# evidence that *something* with that number was ours once, so a cleanup acting
# on the record alone could signal a process that merely inherited the number -
# and the record was the only evidence of ownership the cleanup had.
#
# A file holding a bare pid is still read, and read as "no start time", so a
# store written before this is not invalidated by it: it simply gets the
# weaker check it has always had.
PID_DIRNAME = "pids"
PID_SUFFIX = ".pid"


class StateStore:
    """A JSON file mapping run_id -> run record (processes started for that run)."""

    def __init__(self, root: str = None):
        """Locate the store: the argument, else ``CADET_STATE_ROOT``, else the cwd.

        The working directory is the right default - a run's state belongs with
        the checkout it was started from - but it is the wrong *only* option.
        Every orchestrator started in one checkout shares one store, which is
        how a pre-run cleanup once killed a concurrent run's server, and two
        tests in one suite share it for the same reason. The environment
        variable gives a caller a way to say otherwise without threading a
        parameter through every construction site.

        The variable was already being set by this project's own probe scripts,
        where it did nothing at all until now.
        """
        base = root if root is not None else (
            os.environ.get("CADET_STATE_ROOT") or os.getcwd()
        )
        self.dir = os.path.join(base, STATE_DIRNAME)
        self.path = os.path.join(self.dir, STATE_FILENAME)
        self.pid_root = os.path.join(self.dir, PID_DIRNAME)

    def pid_dir(self, run_id: str) -> str:
        """Return the directory holding ``run_id``'s pid files."""
        return os.path.join(self.pid_root, run_id)

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}

    def _save(self, data: Dict[str, Any]) -> None:
        """Replace the record file in one step, never rewrite it in place.

        Two orchestrators started in the same directory share this file, and
        ``owner`` is the only thing that stops a pre-run cleanup killing the
        other one's live run (see :meth:`abandoned_pids`). An in-place rewrite
        has a window in which the file is truncated or half-written; a reader in
        that window parses nothing, falls back to "no runs recorded", and so
        reads every recorded pid as unowned - exactly the state the owner check
        exists to prevent. Writing beside it and renaming has no such window,
        and leaves the previous contents intact if the write itself fails.
        """
        os.makedirs(self.dir, exist_ok=True)
        pending = self.path + ".new"
        with open(pending, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        os.replace(pending, self.path)

    def record(self, run_id: str, out_dir: str, procs: List[Dict[str, Any]]) -> None:
        """Record (or replace) the process list for ``run_id``."""
        data = self.load()
        entry = data.setdefault(run_id, {})
        entry["dir"] = out_dir
        entry["owner"] = os.getpid()
        entry["procs"] = procs
        self._save(data)
        for proc in procs:
            if proc.get("pid"):
                self._write_pid(run_id, proc.get("name") or str(proc["pid"]), proc["pid"])

    def add_proc(self, run_id: str, out_dir: str, name: str, pid: Optional[int]) -> None:
        """Record one process as soon as it is started.

        Called per launch rather than once per run: a run that is interrupted
        between its first and last process must still leave a complete record of
        what it started, which is the case the old end-of-startup record
        missed.
        """
        if not pid:
            return
        data = self.load()
        entry = data.setdefault(run_id, {"dir": out_dir, "procs": []})
        entry["dir"] = out_dir
        # Which orchestrator is supervising this run; see abandoned_pids().
        entry["owner"] = os.getpid()
        procs = entry.setdefault("procs", [])
        procs.append({"name": name, "pid": pid})
        self._save(data)
        self._write_pid(run_id, name, pid)

    def record_netem(self, run_id: str, out_dir: str, interfaces: List[str]) -> None:
        """Record the interfaces this run installed netem qdiscs on."""
        if not interfaces:
            return
        data = self.load()
        entry = data.setdefault(run_id, {"dir": out_dir, "procs": []})
        entry["dir"] = out_dir
        entry["netem_interfaces"] = sorted(set(interfaces))
        self._save(data)

    def recorded_pids(self) -> List[int]:
        """Return every pid in every pid file, whether or not it is still alive."""
        pids = []
        for run_id in self._pid_runs():
            pids.extend(self.pids_for(run_id))
        return pids

    def abandoned_pids(self) -> List[int]:
        """Return the recorded pids a pre-run cleanup may kill.

        The store lives in the working directory, so two orchestrators started
        in the same repository share it, and "recorded" is therefore not the
        same as "mine to kill". A run's processes are fair game when this
        process is the one that recorded them, or when the orchestrator that did
        is no longer alive - that run is abandoned and its processes are the
        orphans the cleanup exists for. A run another live orchestrator
        recorded is left alone: killing it was measured to fail every
        concurrent run with the server SIGTERMed during warm-up.

        ``recorded_pids`` keeps its meaning for ``cadet down``, where sweeping
        everything is the point.
        """
        data = self.load()
        pids = []
        for run_id in self._pid_runs():
            owner = data.get(run_id, {}).get("owner")
            if owner not in (None, os.getpid()) and pid_alive(int(owner)):
                continue
            pids.extend(self.pids_for(run_id))
        return pids

    def pids_for(self, run_id: str) -> List[int]:
        """Return the pids recorded for one run."""
        return [pid for pid, _started in self.records_for(run_id)]

    def records_for(self, run_id: str, directory: Optional[str] = None):
        """Return ``(pid, start_ticks)`` for one run; ``start_ticks`` may be None.

        ``None`` for a pid file written before the start time was recorded, or
        for one whose process had already gone when it was written. A caller
        that verifies the start time must treat ``None`` as "cannot verify" and
        decide what that means for it, rather than as "verified".
        """
        directory = directory or self.pid_dir(run_id)
        out = []
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            return out
        for name in names:
            if not name.endswith(PID_SUFFIX):
                continue
            try:
                with open(os.path.join(directory, name)) as fh:
                    fields = fh.read().split()
                pid = int(fields[0])
            except (OSError, ValueError, IndexError):
                continue  # a truncated pid file is no evidence of ownership
            started = None
            if len(fields) > 1:
                try:
                    started = int(fields[1])
                except ValueError:
                    started = None
            out.append((pid, started))
        return out

    def recorded_process_records(self):
        """Return ``(pid, start_ticks)`` for every recorded process."""
        found = []
        for run_id in self._pid_runs():
            found.extend(self.records_for(run_id))
        return found

    def recorded_netem_interfaces(self) -> List[str]:
        """Return every interface any recorded run installed a netem qdisc on."""
        interfaces = []
        for rec in self.load().values():
            interfaces.extend(rec.get("netem_interfaces", []))
        return sorted(set(interfaces))

    def _pid_runs(self) -> List[str]:
        try:
            return sorted(os.listdir(self.pid_root))
        except OSError:
            return []

    def _write_pid(self, run_id: str, name: str, pid: int) -> None:
        directory = self.pid_dir(run_id)
        os.makedirs(directory, exist_ok=True)
        # The role name carries a colon ("serve:server"); keep the file name flat.
        safe = name.replace(os.sep, "_").replace(":", "_")
        # Read now rather than later: the process is alive at this moment, and
        # after it exits there is nothing to read. None when it has already
        # gone, which writes a bare pid and is the pre-existing behaviour.
        from cadet.orchestrator.lifecycle import process_start_ticks

        started = process_start_ticks(pid)
        with open(os.path.join(directory, safe + PID_SUFFIX), "w") as fh:
            fh.write("{} {}".format(pid, started) if started is not None else str(pid))

    def clear(self, run_id: str) -> None:
        """Drop ``run_id``'s pid files, then remove it from the store (idempotent).

        The pid files go first. They are what a cleanup reads, and the record
        entry is what says who owns them, so removing the entry first would
        leave pid files behind that no entry claims - and an unclaimed pid file
        is one a cleanup may kill. The reverse leftover, an entry with no pid
        files, is read by nothing.
        """
        shutil.rmtree(self.pid_dir(run_id), ignore_errors=True)
        data = self.load()
        if run_id in data:
            del data[run_id]
            self._save(data)

    def clear_all(self) -> None:
        """Remove every recorded run."""
        if os.path.exists(self.path):
            os.remove(self.path)
        shutil.rmtree(self.pid_root, ignore_errors=True)


def pid_alive(pid: int) -> bool:
    """True if a process group / pid is still alive."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


def kill_pid(pid: int) -> None:
    """Terminate a process group by leader pid (SIGTERM), ignoring if gone."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
