"""Run-lifecycle steps, ported from the original orchestrator scripts.

The original experiment orchestrators already solved the lifecycle of a braking
run: clear orphans and the server port, start the server and ScenarioRunner,
block on a marker line in the scenario log, run the client to completion, then
kill the scenario's whole process group. This module is that logic, ported once
so both the host orchestrator and the emulated session share one ordering and one
teardown. Their hard-coded constants (port 10000, script names) became config
fields; the readiness marker and its timeout come from the hook's
:class:`LaunchSpec`. Three behaviours here are additions without an original
equivalent: wall-clock limits on each phase (:func:`startup_limit_s`,
:func:`active_limit_s`), treating a sim hook or server failure as the end of the
run, and an explicit run-end rule (``experiment.end_on``). The original's ``fuser
-k`` port cleanup was narrowed: only CADET-owned processes are killed; a foreign
holder is named by preflight and refused. Used by :mod:`cadet.orchestrator.runner`
and :mod:`cadet.network.netwaggle.session`. Phase limits come from
``experiment.max_startup_s`` (defaults to the sum of readiness timeouts) and
``experiment.max_active_s`` (defaults to ``source.scenario.timeout_s`` plus a
margin). ``experiment.end_on`` and ``experiment.end_grace_s`` control when and
how cleanly a run stops.
"""

import contextlib
import json
import os
import signal
import socket
import time
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# The original's wait_for_line default: how long to wait with no new log output
# before giving up on a readiness marker.
WAIT_FOR_LINE_TIMEOUT_S = 60.0
# How long the worker phase may exceed the scenario's own timeout before it is
# stopped. The scenario ends itself at timeout_s; this is only slack around it.
ACTIVE_LIMIT_MARGIN_S = 60.0
# Used when neither experiment.max_active_s nor source.scenario.timeout_s is set.
DEFAULT_ACTIVE_LIMIT_S = 600.0
# Used for the startup phase when the config sets no limit and the run waits on
# no readiness marker at all, so there is no sum to take.
DEFAULT_STARTUP_LIMIT_S = 120.0
# How often the worker wait checks the deadline and the guard processes.
_POLL_S = 0.2
# How long to wait for a killed process to actually release a port. The original
# covered this with a fixed sleep after its cleanup; waiting on the condition
# does the same job without guessing how long the kernel needs.
PORT_RELEASE_TIMEOUT_S = 5.0

# How long a killed process is given to exit on SIGTERM before SIGKILL.
KILL_GRACE_S = 3.0

# Reason strings recorded in run.json when a run did not end on its own.
REASON_OK = "workers exited"
REASON_ACTIVE_LIMIT = "active time limit"
REASON_STARTUP_LIMIT = "startup time limit"
REASON_NODE_EXITED = "node exited unexpectedly"
REASON_SIM_HOOK_EXIT = "sim hook exited"
REASON_INTERRUPTED = "interrupted"
# Every worker exited cleanly and none of them produced any output. Exiting zero
# having done nothing is still exiting zero, so this is the one end state that
# cannot be read off an exit code.
REASON_NO_OUTPUT = "no output"

# Values for RunTiming.limit: which phase's limit ended the run, if either did.
LIMIT_STARTUP = "startup"
LIMIT_ACTIVE = "active"


class StartupLimitExceeded(Exception):
    """The startup phase ran past ``experiment.max_startup_s``."""


class RunTiming:
    """How long each phase of a run took, and which limit ended it.

    Recorded in ``run.json`` so a run that hit a limit can be told apart from
    one that was merely slow, without re-reading the logs: the phase durations
    say where the time went and ``limit`` says whether a bound was reached.
    """

    def __init__(self):
        self.startup_s: Optional[float] = None
        self.active_s: Optional[float] = None
        self.teardown_s: Optional[float] = None
        # LIMIT_STARTUP / LIMIT_ACTIVE, or None if no limit ended the run.
        self.limit: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        def _round(value):
            return None if value is None else round(float(value), 2)

        return {
            "startup_s": _round(self.startup_s),
            "active_s": _round(self.active_s),
            "teardown_s": _round(self.teardown_s),
            "limit": self.limit,
        }

    def __str__(self) -> str:
        parts = [
            "{}={}s".format(name, "?" if value is None else "{:.1f}".format(value))
            for name, value in (
                ("startup", self.startup_s),
                ("active", self.active_s),
                ("teardown", self.teardown_s),
            )
        ]
        text = " ".join(parts)
        return text + (" limit={}".format(self.limit) if self.limit else "")

# ``experiment.end_on`` values: what counts as the end of a run.
END_ON_CLIENT_REPORT = "client_report"
END_ON_SIM_HOOK_EXIT = "sim_hook_exit"
END_ON_FIRST = "first"
END_ON_VALUES = (END_ON_CLIENT_REPORT, END_ON_SIM_HOOK_EXIT, END_ON_FIRST)


class RunInterrupted(Exception):
    """A terminating signal arrived while a run was active."""


def install_interrupt_guard():
    """Turn SIGTERM and SIGHUP into :class:`RunInterrupted`; return the undo.

    Default SIGTERM handling ends the process immediately, so a ``finally``
    block never runs and an interrupted run leaves its processes, qdiscs and
    emulated network behind. Raising instead routes the interruption through the
    teardown the run already has, which is why this adds no teardown logic of
    its own. SIGINT already raises :class:`KeyboardInterrupt` and is left alone.

    Returned as a function rather than a context manager because the callers
    install it around a ``try``/``finally`` they already have, and wrapping that
    in another block would re-indent the whole run for no behavioural gain;
    :func:`interrupt_guard` is the block form for callers that want one.
    Signals cannot be installed from a non-main thread, where this is a no-op.
    """

    def _raise(signum, _frame):
        raise RunInterrupted("signal {}".format(signum))

    previous = {}
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            previous[sig] = signal.signal(sig, _raise)
        except (ValueError, OSError):  # not the main thread, or no such signal
            pass

    def _restore():
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    return _restore


@contextlib.contextmanager
def interrupt_guard():
    """Block form of :func:`install_interrupt_guard`."""
    restore = install_interrupt_guard()
    try:
        yield
    finally:
        restore()


def wait_for_line(file_handle, target_line, timeout=WAIT_FOR_LINE_TIMEOUT_S) -> bool:
    """Read forward through an open log until ``target_line`` appears.

    Ported unchanged from the original orchestrators: it reads incrementally
    from wherever the handle is, so a marker already written is still seen, and
    the timeout applies to waiting for *new* output rather than to the total
    wait. A scenario that keeps logging therefore keeps its barrier open, which
    is the behaviour the original runs depended on.
    """
    start_time = time.time()
    while True:
        line = file_handle.readline()
        if not line:
            if time.time() - start_time > timeout:
                return False
            time.sleep(0.1)
            continue
        if target_line in line:
            return True


def wait_for_marker(
    log_path: str,
    target_line: str,
    proc=None,
    timeout: float = WAIT_FOR_LINE_TIMEOUT_S,
) -> bool:
    """Wait for ``target_line`` in ``log_path``, giving up if ``proc`` exits.

    The barrier itself is :func:`wait_for_line`. The process check is the
    addition: the original would sit in ``readline`` for the full timeout after
    ScenarioRunner had already died, so a crash cost a minute per run and
    reported only "did not start properly".
    """
    deadline = time.time() + timeout
    while not os.path.exists(log_path):
        if proc is not None and proc.poll() is not None:
            return False
        if time.time() > deadline:
            return False
        time.sleep(0.1)
    with open(log_path) as handle:
        while True:
            line = handle.readline()
            if line:
                if target_line in line:
                    return True
                continue
            if proc is not None and proc.poll() is not None:
                # Drain whatever was written between the last read and the exit
                # before concluding the marker never arrived.
                return target_line in handle.read()
            if time.time() > deadline:
                return False
            time.sleep(0.1)


# How many of a log's last lines a readiness failure quotes.
READINESS_FAILURE_LINES = 3


def readiness_failure(name: str, proc=None, log_path: Optional[str] = None) -> str:
    """Describe a failed readiness wait: who, whether it died, and its last words.

    A wait can fail two ways that call for opposite fixes - the process exited,
    or it is alive and never printed its marker - and the message used to say
    neither. The log tail is included because the log a sweep writes is per-rep
    and the next rep overwrites the directory, so by the time the failure is
    read the evidence can be gone.
    """
    parts = [repr(name)]
    if proc is not None:
        code = proc.poll()
        parts.append("still running" if code is None else "exited rc={}".format(code))
    if log_path:
        parts.append("see {}".format(log_path))
        try:
            with open(log_path) as handle:
                lines = [line.strip() for line in handle if line.strip()]
        except OSError:
            lines = []
        if lines:
            parts.append(
                "last: {}".format(" | ".join(lines[-READINESS_FAILURE_LINES:]))
            )
    return "; ".join(parts)


def orphan_patterns(config, sim_hook_specs: Sequence = ()) -> List[str]:
    """Return ``pkill -f`` patterns for the processes a previous run may have left.

    The original listed its two script names and ``scenario_runner`` literally.
    Here the role processes are always ``cadet <sub>`` and the sim-side programs
    come from the app's own hook specs, so the list is derived rather than
    hard-coded.
    """
    patterns = ["cadet serve", "cadet client", "cadet load"]
    for spec in sim_hook_specs:
        argv = list(getattr(spec, "argv", ()) or ())
        # The script path is the identifying argument; argv[0] is an interpreter.
        for arg in argv[1:]:
            if arg.endswith(".py"):
                patterns.append(os.path.basename(arg))
                break
    return patterns


def listen_ports(config) -> List[int]:
    """Return the ports this config's listening nodes will bind.

    These are the ports the original freed with ``fuser -k 10000/tcp``, taken
    from the config instead of a constant.
    """
    ports = []
    for node in config.nodes:
        if node.role in ("server", "sim") and node.port is not None:
            ports.append(int(node.port))
    return ports


def process_cmdline(pid: int) -> str:
    """Return ``pid``'s command line, or "" if it is gone or unreadable."""
    try:
        with open("/proc/{}/cmdline".format(pid), "rb") as handle:
            return handle.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return ""


def process_cwd(pid: int) -> str:
    """Return ``pid``'s working directory, or "" if it is gone or unreadable.

    Evidence of ownership that does not depend on a record: a role started from
    a checkout runs with that checkout as its working directory, so a
    pattern-matched process can be attributed to a tree rather than guessed at.
    Unreadable for a process owned by another account, which is the right
    answer - a process this account cannot inspect is not one it should kill.
    """
    # readlink and not realpath: realpath does not raise for a path that is not
    # there, it returns what it was given - so a dead pid came back as the
    # probe path itself, which is a truthy answer meaning "unknown".
    try:
        return os.path.realpath(os.readlink("/proc/{}/cwd".format(pid)))
    except OSError:
        return ""


def process_start_ticks(pid: int):
    """Return ``pid``'s start time in clock ticks since boot, or ``None``.

    Field 22 of ``/proc/<pid>/stat``. It is what makes a recorded pid checkable
    after the fact: a pid is reused, a (pid, start time) pair is not, so a
    cleanup can tell the process it recorded from whatever now holds that
    number.

    Parsed from after the last ``)`` rather than by splitting the whole line,
    because field 2 is the executable name in parentheses and may itself contain
    spaces and parentheses.
    """
    try:
        with open("/proc/{}/stat".format(pid)) as handle:
            text = handle.read()
    except OSError:
        return None
    close = text.rfind(")")
    if close < 0:
        return None
    fields = text[close + 1 :].split()
    # After the ")" the remaining fields are 3..N, so field 22 is index 19.
    if len(fields) < 20:
        return None
    try:
        return int(fields[19])
    except ValueError:
        return None


def all_pids() -> List[int]:
    """Return every pid visible in /proc."""
    pids = []
    try:
        names = os.listdir("/proc")
    except OSError:
        return pids
    for name in names:
        if name.isdigit():
            pids.append(int(name))
    return pids


def pids_matching(pattern: str) -> List[int]:
    """Return the pids whose command line contains ``pattern``, excluding ours.

    The original used ``pkill -f``, which also matches the shell running the
    cleanup and so kills itself. Reading /proc instead makes it possible to
    exclude this process and its group, to report what was killed, and to work
    in an image that has no procps.
    """
    own = {os.getpid(), os.getppid()}
    try:
        own.add(os.getpgid(0))
    except OSError:
        pass
    found = []
    for pid in all_pids():
        if pid in own:
            continue
        if pattern in process_cmdline(pid):
            found.append(pid)
    return found


def listening_socket_inodes(port: int) -> Set[int]:
    """Return the inodes of the sockets listening on ``port``, from /proc/net.

    ``/proc/net/tcp`` writes the local address as ``HEXIP:HEXPORT`` and the
    state as a two-digit hex code, of which 0A is LISTEN.
    """
    inodes: Set[int] = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path) as handle:
                next(handle, None)  # the header row
                for line in handle:
                    fields = line.split()
                    if len(fields) < 10 or fields[3] != "0A":
                        continue
                    local = fields[1].rsplit(":", 1)
                    if len(local) != 2:
                        continue
                    try:
                        if int(local[1], 16) != port:
                            continue
                        inodes.add(int(fields[9]))
                    except ValueError:
                        continue
        except OSError:
            continue
    return inodes


def port_holders(port: int) -> List[Tuple[int, str]]:
    """Return ``(pid, command line)`` for every process listening on ``port``.

    This is the information the original threw away: it passed the port to
    ``fuser -k`` and never learned whose socket it had just closed. Resolved
    from /proc alone - the listening socket's inode, then the descriptors
    pointing at it - because the images this runs in have neither ``fuser`` nor
    ``ss``, and a cleanup that silently cannot identify a holder is the
    failure mode this removes.
    """
    inodes = listening_socket_inodes(port)
    if not inodes:
        return []
    wanted = {"socket:[{}]".format(inode) for inode in inodes}
    holders = []
    for pid in all_pids():
        directory = "/proc/{}/fd".format(pid)
        try:
            descriptors = os.listdir(directory)
        except OSError:
            continue  # another user's process, or it exited while we looked
        for fd in descriptors:
            try:
                target = os.readlink(os.path.join(directory, fd))
            except OSError:
                continue
            if target in wanted:
                holders.append((pid, process_cmdline(pid)))
                break
    return holders


def describe_port_holders(port: int) -> str:
    """Return a readable "pid N (cmdline)" list for whoever holds ``port``."""
    holders = port_holders(port)
    if holders:
        return ", ".join("pid {} ({})".format(pid, cmd or "unknown") for pid, cmd in holders)
    if listening_socket_inodes(port):
        # Something is listening but its descriptors are not readable from here:
        # another user's process, or another namespace.
        return "a process this user cannot inspect (another user, or another namespace)"
    return "no listening process (socket in TIME_WAIT, or bound in another namespace)"


def is_cadet_owned(pid: int, owned: Iterable[int] = (), patterns: Sequence[str] = ()) -> bool:
    """True if ``pid`` was recorded by this orchestrator **and still looks like it**.

    The record is the evidence of ownership, and a command line cannot replace
    it: every CADET server's is ``cadet serve``, so matching on it alone cannot
    distinguish a leftover of our own from a live process belonging to someone
    else's run on the same machine - and killing the latter is what a pre-run
    cleanup must never do. Measured, before that was narrowed: two orchestrators
    in one container, and every run of the second failed with the first's server
    SIGTERMed during model warm-up.

    But a record is not sufficient either, because **pids are reused**. A stale
    record names a number, and by the time a later run reads it the kernel may
    have given that number to something else - including a role belonging to a
    run in another checkout, which is how this was found: the route equivalence
    test failed about one run in five with ``serve:server exited with code
    -15``, while other streams were running their own suites on the same
    machine. Nothing in the record can notice that; the process can.

    So ownership is the record **and** a command line that still matches one of
    ``patterns``. With no patterns given the check is the record alone, which is
    what ``cadet down`` wants, where sweeping everything recorded is the point.

    A process that looks like a role but is not recorded is reported by
    :func:`looks_like_a_role` and left alone.
    """
    if pid not in set(owned):
        return False
    if not patterns:
        return True
    return looks_like_a_role(pid, patterns)


def looks_like_a_role(pid: int, patterns: Sequence[str]) -> bool:
    """True if ``pid``'s command line is a CADET role or an app's sim program.

    Evidence for a report, not for a kill: see :func:`is_cadet_owned`.
    """
    cmdline = process_cmdline(pid)
    if not cmdline:
        return False
    return any(pattern in cmdline for pattern in patterns)


def kill_pids(pids: Iterable[int], log=print) -> List[int]:
    """SIGTERM a pid's process group, escalating to SIGKILL. Returns those killed."""
    killed = []
    for pid in sorted(set(pids)):
        if not _signal_pid(pid, signal.SIGTERM):
            continue
        killed.append(pid)
    deadline = time.monotonic() + KILL_GRACE_S
    while killed and time.monotonic() < deadline:
        if not any(_pid_alive(pid) for pid in killed):
            return killed
        time.sleep(0.1)
    for pid in killed:
        if _pid_alive(pid):
            _signal_pid(pid, signal.SIGKILL)
    return killed


def _signal_pid(pid: int, sig) -> bool:
    """Signal ``pid``'s process group, falling back to the pid. False if gone."""
    try:
        os.killpg(os.getpgid(pid), sig)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, sig)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    """True if ``host:port`` can be bound, i.e. nothing is listening there."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def wait_for_ports_free(
    ports: Sequence[int], timeout_s: float = PORT_RELEASE_TIMEOUT_S, log=print
) -> bool:
    """Wait until every port is actually free, after the holders were killed.

    Killing a process does not release its listening socket synchronously, so a
    bind check run straight after the cleanup still sees the port in use and the
    run is refused for a port that is about to become free. The original papered
    over this with a fixed sleep between runs; this waits on the condition
    instead. Returns False if a port is still held when the time runs out, which
    means something CADET did not start is holding it.
    """
    if not ports:
        return True
    deadline = time.monotonic() + timeout_s
    while True:
        still_held = [port for port in ports if not port_free(port)]
        if not still_held:
            return True
        if time.monotonic() >= deadline:
            log("[orch] ports still held after cleanup: {}".format(still_held))
            return False
        time.sleep(0.1)


def kill_orphans(
    patterns: Sequence[str],
    ports: Sequence[int],
    log=print,
    owned: Iterable[int] = (),
) -> List[Tuple[int, int, str]]:
    """Kill the leftover processes CADET owns, and free only the ports they held.

    Ported from the original's ``kill_orphans``, with its one indiscriminate
    step narrowed. The original ran ``pkill -f`` over its script names and then
    ``fuser -k`` over its port, which frees the port whoever is holding it. Here
    a port is freed only by killing a holder that is CADET's own - recorded in
    ``owned``, or matching one of ``patterns`` - so a run can never take down a
    process the experimenter started.

    Every kill is best-effort: "nothing to kill" is the normal case and must not
    fail the run. Returns the holders left alone, as ``(port, pid, cmdline)``,
    which is what preflight reports when it then refuses the run.
    """
    # Recorded, alive, and still looking like the role it was recorded as. The
    # last clause is what keeps a reused pid from being killed; see
    # :func:`is_cadet_owned`.
    targets: Set[int] = {
        pid for pid in owned if _pid_alive(pid) and is_cadet_owned(pid, owned, patterns)
    }
    # Processes that look like ours but were not recorded by this orchestrator.
    # Reported, never killed: another run on this machine looks exactly like
    # this, and a cleanup that cannot tell the difference must not guess.
    strays: List[Tuple[int, str]] = []
    for pattern in patterns:
        for pid in pids_matching(pattern):
            if pid not in targets:
                strays.append((pid, process_cmdline(pid)))

    foreign: List[Tuple[int, int, str]] = []
    for port in ports:
        for pid, cmdline in port_holders(port):
            if is_cadet_owned(pid, owned, patterns):
                targets.add(pid)
            else:
                foreign.append((port, pid, cmdline))

    # Logged before the kill, while the command lines can still be read.
    for pid in sorted(targets):
        try:
            group = os.getpgid(pid)
        except OSError:
            group = -1
        log(
            "[orch] orphan cleanup: killing pid {} (pgid {}): {}".format(
                pid, group, process_cmdline(pid) or "unknown"
            )
        )
    killed = kill_pids(targets, log=log)
    # Only wait on the ports a kill was supposed to free. A port held by a
    # foreign process will never come free here, and waiting on it would spend
    # the timeout to learn what preflight is about to say anyway.
    foreign_ports = {port for port, _pid, _cmd in foreign}
    wait_for_ports_free([p for p in ports if p not in foreign_ports], log=log)
    log(
        "[orch] orphan cleanup: killed {} process(es), {} port(s) checked".format(
            len(killed), len(ports)
        )
    )
    for port, pid, cmdline in foreign:
        log(
            "[orch] port {} is held by pid {} ({}), which this run did not "
            "start; left alone".format(port, pid, cmdline or "unknown")
        )
    for pid, cmdline in sorted(set(strays)):
        log(
            "[orch] pid {} looks like a CADET role ({}) but this run did not "
            "start it; left alone".format(pid, cmdline or "unknown")
        )
    return foreign


def print_final_client_report(client_log_path: str, log=print) -> None:
    """Echo the client's final report, as the original did after every run."""
    if not os.path.exists(client_log_path):
        return
    try:
        with open(client_log_path) as fh:
            inside_report = False
            for line in fh:
                if "FINAL REPORT" in line:
                    inside_report = True
                if inside_report:
                    log(line.rstrip())
    except OSError:
        pass


def client_output_failures(out_dir: str, config=None, log=print) -> List[str]:
    """Return a reason per client that reported producing no output.

    The verdict is the client's, written beside its metrics: it holds the counts
    and its app declares what counts as output (see
    :func:`cadet.core.apps.AppPlugin.client_success`). A missing file is not a
    failure - a client that never got far enough to write one has already failed
    the run for a reason that is recorded.

    Whether that verdict *fails the run* is ``experiment.require_output``, off
    by default because the original judged a run by the client's exit code
    alone. With it off the verdict is still logged: the information is the
    client's own and worth having, it just does not change the outcome.
    """
    from cadet.roles.client import CLIENT_OUTCOME_PREFIX

    # One verdict per client node, so a run with several clients is judged by
    # all of them: a single sensor that produced nothing is a result about that
    # sensor, and reading only one file would hide it.
    try:
        names = sorted(
            name
            for name in os.listdir(out_dir)
            if name.startswith(CLIENT_OUTCOME_PREFIX) and name.endswith(".json")
        )
    except OSError:
        return []
    failures = []
    for name in names:
        path = os.path.join(out_dir, name)
        try:
            with open(path) as handle:
                record = json.load(handle)
        except (OSError, ValueError) as exc:
            log("[orch] could not read {}: {}".format(path, exc))
            continue
        if record.get("ok", True):
            continue
        failures.append(record.get("detail") or "client produced no output")
    if not failures:
        return []
    detail = "; ".join(failures)
    if config is not None and not config.experiment.require_output:
        log(
            "[orch] the client reports no output ({}); not failing the run, "
            "because experiment.require_output is off".format(detail)
        )
        return []
    return [detail]


def active_limit_s(config) -> float:
    """Return the wall-clock limit for the worker phase, in seconds.

    ``experiment.max_active_s`` wins; otherwise the scenario's own timeout plus
    :data:`ACTIVE_LIMIT_MARGIN_S`, so the limit always sits outside the
    scenario's own ending and only catches a run that failed to end at all.
    """
    explicit = config.experiment.max_active_s
    if explicit is not None:
        return float(explicit)
    scenario = getattr(getattr(config, "source", None), "scenario", None)
    timeout_s = getattr(scenario, "timeout_s", None) if scenario is not None else None
    if timeout_s is not None:
        return float(timeout_s) + ACTIVE_LIMIT_MARGIN_S
    return DEFAULT_ACTIVE_LIMIT_S


def startup_limit_s(config, readiness_timeouts: Sequence[float] = ()) -> float:
    """Return the wall-clock limit for everything before the workers start.

    ``experiment.max_startup_s`` wins; otherwise the sum of the readiness
    timeouts this run will wait on. The sum is the right default because it is
    already the phase's bound, taken one step at a time: a startup that every
    individual wait would have allowed can never be refused by the total, and a
    phase that stalls between the waits still cannot run past it.
    """
    explicit = config.experiment.max_startup_s
    if explicit is not None:
        return float(explicit)
    total = sum(float(t) for t in readiness_timeouts)
    return total if total > 0 else DEFAULT_STARTUP_LIMIT_S


def remaining_startup_s(deadline: float, step_timeout: float, what: str) -> float:
    """Return how long a readiness wait may take, given the startup deadline.

    Raises :class:`StartupLimitExceeded` if the budget is already spent, which
    is what makes the limit a bound on the phase rather than a note written
    afterwards.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise StartupLimitExceeded(
            "startup limit reached before {} became ready".format(what)
        )
    return min(float(step_timeout), remaining)


def end_rule(config) -> Tuple[str, float]:
    """Return ``(end_on, grace_s)`` for this config, with the schema's defaults."""
    experiment = getattr(config, "experiment", None)
    end_on = getattr(experiment, "end_on", None) or END_ON_CLIENT_REPORT
    grace = getattr(experiment, "end_grace_s", None)
    return end_on, float(grace if grace is not None else 0.0)


class RunOutcome:
    """Why a run ended, and with what exit codes."""

    def __init__(self, codes: List[Optional[int]], reason: str, detail: str = ""):
        self.codes = codes
        self.reason = reason
        self.detail = detail

    @property
    def ok(self) -> bool:
        """True if the run ended the way its ``end_on`` rule says it should.

        For ``client_report`` that means the workers exited cleanly. For a run
        ended by its sim hook the workers were stopped on purpose, so their exit
        codes are absent rather than zero; what disqualifies the run is a worker
        that had already failed.
        """
        if self.reason == REASON_OK:
            return bool(self.codes) and all(code == 0 for code in self.codes)
        if self.reason == REASON_SIM_HOOK_EXIT:
            return all(code in (0, None) for code in self.codes)
        return False

    def as_dict(self) -> Dict[str, object]:
        return {"exit_codes": self.codes, "reason": self.reason, "detail": self.detail}

    def __str__(self) -> str:
        text = "{}: exit codes {}".format(self.reason, self.codes)
        return "{} ({})".format(text, self.detail) if self.detail else text


def wait_for_workers(
    workers: Sequence[Tuple[str, object]],
    limit_s: float,
    guards: Sequence[Tuple[str, object]] = (),
    log=print,
    sim_hooks: Sequence[Tuple[str, object]] = (),
    end_on: str = END_ON_CLIENT_REPORT,
    grace_s: float = 0.0,
) -> RunOutcome:
    """Block until the run's ``end_on`` rule fires, the limit passes, or a guard dies.

    ``workers`` are the processes whose completion *is* the run under the
    ``client_report`` rule (clients and load generators); the original ran its
    single client in the foreground, which is that rule. ``guards`` are
    processes that must stay alive for the run to mean anything - the sim hooks
    and the servers - and ``sim_hooks`` is the subset whose *clean* exit can end
    a run, under the ``sim_hook_exit`` and ``first`` rules.

    Without an explicit rule a clean sim-hook exit only got recorded, so a
    scenario that had finished left the client streaming into a world that no
    longer had an ego, until the run limit caught it minutes later. ``end_on``
    makes the choice a config field instead of an accident of which process
    happened to exit first, and ``grace_s`` gives the workers a bounded chance
    to write their own report before they are stopped.
    """
    deadline = time.monotonic() + limit_s
    # Guards that finished cleanly while the workers were still running. The run
    # may be allowed to continue, but it is recorded either way: a sim hook that
    # ended early stops evaluating its own criteria.
    departed = []
    hook_names = {name for name, _proc in sim_hooks}
    hook_exit_at = None
    while True:
        codes = [proc.poll() for _name, proc in workers]
        workers_running = any(code is None for code in codes)

        # Guards are checked before the workers' own completion, not after. A
        # client often exits cleanly when its server disappears, so checking
        # completion first would report a run whose server died mid-stream as a
        # success.
        #
        # Only a *failed* guard invalidates the run. A guard that exits zero has
        # finished its job: the braking server is single-shot by design, serving
        # one client and returning, so its clean exit is the normal end of a run
        # rather than a fault. A guard that exits zero early instead leaves the
        # workers with nothing to talk to, and the run limit below catches that.
        for name, proc in guards:
            code = proc.poll()
            if code is None:
                continue
            if code != 0:
                log("[orch] {} exited (code {}) while the run was active".format(name, code))
                return RunOutcome(
                    codes,
                    REASON_NODE_EXITED,
                    "{} exited with code {}".format(name, code),
                )
            if workers_running and name not in departed:
                departed.append(name)
                log("[orch] {} finished while the run was still active".format(name))
                if name in hook_names and hook_exit_at is None:
                    hook_exit_at = time.monotonic()

        # The workers finishing ends the run under every rule but
        # ``sim_hook_exit``, where the scenario - not the client - defines the
        # run, and a client that returns early simply waits for it.
        if not workers_running and (end_on != END_ON_SIM_HOOK_EXIT or hook_exit_at is not None):
            detail = ""
            if departed:
                detail = "finished during the run: {}".format(", ".join(departed))
            return RunOutcome(codes, REASON_OK, detail)

        # A sim hook that has finished ends the run under the two rules that say
        # so, once the workers have had their grace period to finish by
        # themselves. Reaching here means they did not.
        if hook_exit_at is not None and end_on in (END_ON_SIM_HOOK_EXIT, END_ON_FIRST):
            waited = time.monotonic() - hook_exit_at
            if waited >= grace_s:
                log(
                    "[orch] sim hook finished {:.1f}s ago and the workers are still "
                    "running; ending the run".format(waited)
                )
                return RunOutcome(
                    codes,
                    REASON_SIM_HOOK_EXIT,
                    "{} finished; workers stopped after a {:.0f}s grace".format(
                        ", ".join(departed), grace_s
                    ),
                )

        if time.monotonic() >= deadline:
            log("[orch] active limit of {:.0f}s reached; stopping".format(limit_s))
            return RunOutcome(
                codes, REASON_ACTIVE_LIMIT, "no completion within {:.0f}s".format(limit_s)
            )

        time.sleep(_POLL_S)
