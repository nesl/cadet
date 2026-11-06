"""Remove what an interrupted run left behind, and only what CADET owns.

What this is for:
    ``cadet down`` after a run that did not reach its teardown - killed with
    SIGKILL, or stopped before the signal handler was installed. The run's own
    teardown is the normal path and this is the recovery path; it exists because
    a half-removed emulated network, a bound port or a leftover qdisc silently
    corrupts the next run.

What it contains:
    - :data:`Leftover` and :func:`sweep`: find and remove the leftovers.
    - one finder per category: processes, containers, docker networks, network
      namespaces, OVS bridges, veth interfaces, netem qdiscs.

How it fits into CADET:
    :meth:`cadet.orchestrator.runner.Orchestrator.down` calls :func:`sweep`, so
    ``cadet down`` is a recovery command rather than only a process killer.
    The emulated network's own forced teardown is reused rather than rewritten:
    :func:`cadet.network.netwaggle.emulator.force_cleanup` is the step the
    emulator already runs when Mininet's ``stop()`` hangs.

    Everything here is scoped by ownership, never by "remove all of this kind":
      * processes - a pid CADET recorded in a pid file *here*, and nothing else.
        A role-looking command line is not evidence of ownership: every role's
        is ``cadet <sub>``, so it also matches a live run in another checkout.
        Such a process is named in the report and left running;
      * containers and docker networks - the **run-scoped** name shape the
        docker launcher and the placement assign,
        ``cadet-<config hash>-r<rep>-<suffix>``. The bare ``cadet-`` prefix is
        deliberately not enough: it also matches long-lived containers somebody
        started by hand, and removing one of those is exactly the
        indiscriminate behaviour this module exists to avoid;
      * namespaces, bridges and veths - the names the topology generator
        assigns (``n<N>`` hosts and the ``bps0`` bypass switch);
      * qdiscs - only the interfaces a run recorded in the state store.
    A docker network or a bridge somebody else created is therefore left alone.

How to configure it:
    Nothing. The state root is ``<cwd>/.cadet``, so ``cadet down`` is run from
    the same directory as the run it is recovering.
"""

import os
import re
import shutil
import subprocess
from typing import List, Optional

from cadet.config.loader import RUN_ID_HASH_LEN
from cadet.core.docker_launcher import OWNED_CONTAINER_PREFIX
from cadet.network.netwaggle.topology import ALIAS_PREFIX, BYPASS_SWITCH
from cadet.orchestrator import lifecycle
from cadet.orchestrator.state import StateStore
from cadet.sim.containers import SHARED_SCOPE_LEN, SHARED_SCOPE_PREFIX

# A run's own containers and networks: the docker launcher names a role
# container "cadet-<run_id>-<node>" and the placement names the network
# "cadet-<run_id>-net", where run_id is "<config hash>-r<rep>". Matching that
# shape rather than the bare prefix is what keeps a hand-started container of a
# similar name out of the sweep.
OWNED_RUN_OBJECT_RE = re.compile(
    r"^{}[0-9a-f]{{{}}}-r\d+-.+$".format(re.escape(OWNED_CONTAINER_PREFIX), RUN_ID_HASH_LEN)
)

# The reusable simulator's objects: a CARLA server and the plumbing network it
# sits on, named after the launch settings rather than after a run, because the
# point of reuse is that they outlive the run that started them. Nothing else
# removes them, so `cadet down` does, and they are the one leftover that is
# deliberate until it is not.
OWNED_SIM_OBJECT_RE = re.compile(
    r"^{}{}[0-9a-f]{{{}}}-(?:carla|plumbing)$".format(
        re.escape(OWNED_CONTAINER_PREFIX),
        re.escape(SHARED_SCOPE_PREFIX),
        SHARED_SCOPE_LEN,
    )
)

# Network objects the topology generator names: hosts/stations/APs are
# "<ALIAS_PREFIX><index>" and the unshaped bypass switch is BYPASS_SWITCH. Their
# interfaces take the node name plus "-eth<N>", which is Mininet's own scheme.
OWNED_NODE_RE = re.compile(
    r"^(?:{}\d+|{})$".format(re.escape(ALIAS_PREFIX), re.escape(BYPASS_SWITCH))
)
OWNED_INTF_RE = re.compile(
    r"^(?:{}\d+|{})-eth\d+$".format(re.escape(ALIAS_PREFIX), re.escape(BYPASS_SWITCH))
)


class Leftover:
    """One thing that was found and removed (or found and could not be)."""

    def __init__(self, kind: str, name: str, removed: bool = True, detail: str = ""):
        self.kind = kind
        self.name = name
        self.removed = removed
        self.detail = detail

    def __str__(self) -> str:
        text = "{}: {}".format(self.kind, self.name)
        if not self.removed:
            text += " (NOT removed: {})".format(self.detail or "unknown reason")
        return text


def _run(argv: List[str]) -> Optional[subprocess.CompletedProcess]:
    """Run a command, returning None if the tool is not installed here."""
    try:
        return subprocess.run(argv, capture_output=True, text=True, check=False)
    except OSError:
        return None


def sweep(patterns=("cadet serve", "cadet client", "cadet load"), log=print) -> List[Leftover]:
    """Remove every CADET-owned leftover this host can see. Returns what it did.

    The order matters: processes go first so nothing is still writing to an
    interface or holding a namespace open, then the containers that hold
    namespaces of their own, then the network objects, and the qdiscs last
    because removing an interface removes its qdisc with it.
    """
    found: List[Leftover] = []
    store = StateStore()

    found.extend(_processes(store, patterns, log))
    found.extend(_containers(log))
    found.extend(_networks(log))
    found.extend(_emulated_network(log))
    found.extend(_qdiscs(store, log))

    store.clear_all()
    return found


def _owning_tree(store: StateStore) -> str:
    """The directory this store belongs to - the checkout a role was started in.

    The store lives at ``<tree>/.cadet``, so its parent is the tree. Used to
    attribute a process the record has lost: see :func:`_processes`.
    """
    return os.path.dirname(os.path.realpath(store.dir))


def _processes(store: StateStore, patterns, log) -> List[Leftover]:
    """Kill this checkout's role processes; name any others, kill nothing else.

    Two kinds of evidence, because the record alone turned out to be necessary
    but not sufficient.

    **The record, verified.** A recorded pid is killed, but its start time is
    checked first where one was recorded: a pid is reused and a (pid, start
    time) pair is not, so a record whose process has been replaced by an
    unrelated one is now detected rather than acted on. A record written before
    start times existed has none, and gets the weaker check it always had.

    **The working directory, for what the record has lost.** The patterns used
    to kill as well, and on their own they cannot: every CADET role's command
    line is ``cadet <sub>``, so a pattern match selects every run on the
    machine - including a live one from another checkout that this store has no
    record of, because the store lives in the working directory. Measured: this
    step, run from an unrelated directory whose store had recorded nothing at
    all, SIGTERMed the server and the client of a live run in another checkout.
    That was the "server killed by SIGTERM" flake in the end-to-end suite.

    Removing the pattern kill closed that and opened a smaller hole: a role of
    *this* checkout whose pid file was lost could no longer be reaped at all.
    So a pattern-matched process is now attributed by its working directory,
    which is the checkout it was started in and is evidence no record is needed
    for. One in this tree is ours and is killed; one anywhere else, or one whose
    directory cannot be read - which is what another account's process looks
    like - is reported and left running.
    """
    records = [(pid, started) for pid, started in store.recorded_process_records()]
    recorded = {pid for pid, _s in records}
    recorded.discard(os.getpid())
    tree = _owning_tree(store)
    stale = []
    for pid, started in records:
        if pid == os.getpid() or started is None:
            continue
        now = lifecycle.process_start_ticks(pid)
        if now is not None and now != started:
            # The number is in our record and the process behind it is not ours.
            stale.append(pid)
    for pid in stale:
        recorded.discard(pid)
        log(
            "[down] pid {} is recorded but was started by something else "
            "(pid reuse); left alone".format(pid)
        )
    found: List[Leftover] = []
    alive = [pid for pid in recorded if lifecycle.process_cmdline(pid)]
    if alive:
        killed = lifecycle.kill_pids(alive, log=log)
        found.extend(Leftover("process", "pid {}".format(pid)) for pid in killed)
    seen = set(recorded)
    for pattern in patterns:
        for pid in lifecycle.pids_matching(pattern):
            if pid in seen:
                continue
            seen.add(pid)
            cwd = lifecycle.process_cwd(pid)
            if cwd and os.path.realpath(cwd) == tree:
                # Ours by the directory it was started in, with no record left
                # to say so. This is the case the pattern kill used to cover.
                killed = lifecycle.kill_pids([pid], log=log)
                for victim in killed:
                    found.append(
                        Leftover(
                            "process",
                            "pid {}".format(victim),
                            True,
                            "unrecorded, attributed to this tree",
                        )
                    )
                continue
            cmdline = lifecycle.process_cmdline(pid)
            if not cmdline:
                continue  # it exited while we looked; nothing to report
            log(
                "[down] pid {} looks like a CADET role ({}) but nothing recorded "
                "here started it; left alone".format(pid, cmdline)
            )
            found.append(
                Leftover(
                    "process",
                    "pid {} ({})".format(pid, cmdline),
                    removed=False,
                    detail="not recorded here; it may belong to a live run elsewhere",
                )
            )
    return found


# Docker's listing field differs by object: containers have .Names, networks
# have .Name. Getting it wrong makes the template fail and the listing come back
# empty, which is a cleanup that silently does nothing.
_NAME_FIELD = {"container": "{{.Names}}", "network": "{{.Name}}"}
# `docker container ls` lists only *running* containers, which excludes exactly
# the leftovers this sweep exists to remove: a crashed or interrupted run leaves
# containers that have stopped, or that were created and never started, and
# those are invisible without -a. `docker network ls` has no such distinction
# and takes no flag here.
_LIST_FLAGS = {"container": ["-a"], "network": []}


def _docker_names(kind: str) -> List[str]:
    """List the docker ``container`` or ``network`` names a CADET run created.

    Includes containers that are not running; see :data:`_LIST_FLAGS`.
    """
    result = _run(["docker", kind, "ls"] + _LIST_FLAGS[kind] + ["--format", _NAME_FIELD[kind]])
    if result is None or result.returncode != 0:
        return []
    return [
        name
        for name in result.stdout.split()
        if OWNED_RUN_OBJECT_RE.match(name) or OWNED_SIM_OBJECT_RE.match(name)
    ]


def _containers(log) -> List[Leftover]:
    """Remove the role containers a run created, matched by their run-scoped name."""
    out = []
    for name in _docker_names("container"):
        result = _run(["docker", "rm", "-f", name])
        removed = result is not None and result.returncode == 0
        out.append(Leftover("container", name, removed, _stderr(result)))
    return out


def _networks(log) -> List[Leftover]:
    """Remove the per-run docker networks a placement created.

    This is exactly the leftover that outlives a run invisibly: nothing notices
    a stale network until the next run tries to create one of the same name and
    docker refuses.
    """
    out = []
    for name in _docker_names("network"):
        result = _run(["docker", "network", "rm", name])
        removed = result is not None and result.returncode == 0
        out.append(Leftover("docker network", name, removed, _stderr(result)))
    return out


def _emulated_network(log) -> List[Leftover]:
    """Remove the namespaces, bridges and veths the emulated topology creates."""
    out: List[Leftover] = []

    # The emulator's own forced teardown: its wireless processes, Mininet's
    # cleanup, and the hwsim module. Reused rather than reimplemented.
    from cadet.network.netwaggle.emulator import force_cleanup

    for name in force_cleanup(scoped=True):
        out.append(Leftover("emulator", name))

    result = _run(["ip", "netns", "list"])
    if result is not None:
        for line in result.stdout.splitlines():
            name = line.split()[0] if line.strip() else ""
            if name and OWNED_NODE_RE.match(name):
                rc = _run(["ip", "netns", "delete", name])
                out.append(Leftover("netns", name, rc is not None and rc.returncode == 0))

    if shutil.which("ovs-vsctl"):
        result = _run(["ovs-vsctl", "list-br"])
        if result is not None:
            for name in result.stdout.split():
                if OWNED_NODE_RE.match(name):
                    rc = _run(["ovs-vsctl", "--if-exists", "del-br", name])
                    out.append(Leftover("ovs bridge", name, rc is not None and rc.returncode == 0))

    result = _run(["ip", "-o", "link", "show"])
    if result is not None:
        for line in result.stdout.splitlines():
            # "3: n0-eth0@if2: <BROADCAST,...>" -> the name before any @peer.
            parts = line.split(":", 2)
            if len(parts) < 2:
                continue
            name = parts[1].strip().split("@")[0]
            if OWNED_INTF_RE.match(name):
                rc = _run(["ip", "link", "delete", name])
                out.append(Leftover("veth", name, rc is not None and rc.returncode == 0))
    return out


def _qdiscs(store: StateStore, log) -> List[Leftover]:
    """Remove the root qdisc on each interface a run recorded.

    Scoped by record, not by inspection: an interface can hold a qdisc somebody
    else installed, and there is no way to tell from the qdisc itself who put it
    there. Only an interface a CADET run wrote down is cleared.
    """
    from cadet.network import netem

    out = []
    for interface in store.recorded_netem_interfaces():
        failures = []
        for argv in netem.teardown_commands(interface):
            result = _run(argv)
            if result is not None and result.returncode != 0:
                failures.append(_stderr(result))
        out.append(Leftover("qdisc", interface, not failures, "; ".join(failures)))
    return out


def _stderr(result) -> str:
    if result is None:
        return "tool not installed"
    return (result.stderr or result.stdout or "").strip().splitlines()[0] if (
        result.stderr or result.stdout
    ) else ""
