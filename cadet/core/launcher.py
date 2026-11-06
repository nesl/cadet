"""Launcher interface: start/stop/supervise a role process on a node.

What this is for:
    Abstracts *how* a role process is started on a node, independent of whether
    the node is local, a docker container, ssh-into-docker, or ssh-into-venv. The
    orchestrator builds a :class:`LaunchSpec` per node and hands it to a launcher;
    only the launcher knows how to turn that into a running process.

What it contains:
    - :class:`LaunchStatus`: status constants returned by :meth:`Launcher.poll`.
    - :class:`LaunchSpec`: everything needed to start one process (argv, cwd, env,
      log file, optional readiness marker).
    - :class:`Handle`: a started process + its spec and log path.
    - :class:`Launcher`: abstract ``launch``/``wait_ready``/``poll``/``stop``.
    - :class:`LocalLauncher`: spawns real local subprocesses (the only launcher
      implemented today).
    - :data:`IMAGE_PYTHON` / :data:`IMAGE_PYTHON_VERSION`, and
      :func:`role_executable` / :func:`role_python_version`: which interpreter a
      role placed with a given launcher kind will actually run under, which is
      not in general the one running the orchestrator.

How it fits into CADET:
    The orchestrator (:mod:`cadet.orchestrator`) selects a launcher per
    ``config.nodes[*].launcher`` and uses it to bring servers, clients and sim
    hooks up and down in order, tearing everything down on exit. The other
    launcher kinds implement this same interface: ``docker`` and ``ssh_docker``
    in :mod:`cadet.core.docker_launcher`, ``ssh_venv`` in
    :mod:`cadet.core.venv_launcher`.

How to configure it:
    Launcher choice comes from ``nodes[*].launcher``; process details come from the
    spec the orchestrator builds. This module defines behaviour, not values.
"""

import os
import signal
import subprocess
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

# The interpreter inside a CADET image, and the version that interpreter is.
# The image installs python3.8 and symlinks it as `python` and `python3` (see
# docker/Dockerfile), and the CARLA client egg it bakes in is py3.7, which loads
# under 3.8 and under nothing newer - so the pair is a property of the image and
# its egg together and not a free choice. Pinned against the Dockerfile in a
# test, so the two cannot drift apart silently.
IMAGE_PYTHON = "python3"
IMAGE_PYTHON_VERSION = "3.8"

# The launcher kind that runs a role as a plain child of this process.
LOCAL_LAUNCHER = "local"


def role_executable(launcher_kind: str) -> str:
    """Return the interpreter that will run a role placed with this launcher.

    A local role is a child of this process, so it runs the same interpreter.
    Every other kind starts the role somewhere this process's interpreter does
    not exist, so the value is a *statement* about the interpreter there rather
    than a path any of them uses verbatim: the docker launchers strip the prefix
    from argv, since the image's entry point is ``cadet``, and the venv launcher
    substitutes the venv's own interpreter path, which only it knows. Stating it
    is what the version check reads.
    """
    if launcher_kind == LOCAL_LAUNCHER:
        return sys.executable
    return IMAGE_PYTHON


def role_python_version(launcher_kind: str) -> str:
    """Return the "major.minor" Python a role placed with this launcher gets.

    This - not the orchestrator's own version - is the question the CARLA client
    egg check has to ask. The egg is imported by the role that opens the
    simulator and never by the orchestrator, so asking about the orchestrator
    refused a sound run whenever the two differed, which is exactly the case
    where the orchestrator runs on a host python and its roles run in the image.

    For ``ssh_venv`` this is a requirement and not an observation: a venv is
    whatever version it was created with, so the placement asks the venv its
    version and refuses one that is not this
    (:func:`cadet.orchestrator.placement.require_venv`). Without that the check
    above would be sound and its premise false.
    """
    if launcher_kind == LOCAL_LAUNCHER:
        return "{}.{}".format(*sys.version_info[:2])
    return IMAGE_PYTHON_VERSION


class LaunchStatus:
    """String status constants for a launched role."""

    STOPPED = "stopped"
    RUNNING = "running"
    FAILED = "failed"


class LaunchSpec:
    """Everything needed to start one role process.

    ``readiness_marker`` is a substring the process prints to its log once it is
    ready to accept work (e.g. a server that is listening); the orchestrator waits
    for it before starting dependents. ``None`` means "ready as soon as started".
    """

    def __init__(
        self,
        name: str,
        argv: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        log_path: Optional[str] = None,
        readiness_marker: Optional[str] = None,
        readiness_timeout_s: float = 60.0,
    ):
        self.name = name
        self.argv = argv
        self.cwd = cwd
        self.env = env
        self.log_path = log_path
        self.readiness_marker = readiness_marker
        self.readiness_timeout_s = readiness_timeout_s


class Handle:
    """A started process, its spec, and the open log file handle."""

    def __init__(self, spec: LaunchSpec, proc, log_fh):
        self.spec = spec
        self.proc = proc
        self.log_fh = log_fh

    @property
    def pid(self) -> Optional[int]:
        return getattr(self.proc, "pid", None)


class Launcher(ABC):
    """Abstract process launcher for a role on a node."""

    @abstractmethod
    def launch(self, spec: LaunchSpec) -> Handle:
        """Start the process described by ``spec`` and return a :class:`Handle`."""

    @abstractmethod
    def wait_ready(self, handle: Handle, timeout_s: Optional[float] = None) -> bool:
        """Block until the handle's readiness marker appears (or it exits/timeouts).

        ``timeout_s`` overrides the spec's own timeout, which the orchestrator
        uses to cut a wait down to what is left of ``experiment.max_startup_s``.
        """

    @abstractmethod
    def poll(self, handle: Handle) -> str:
        """Return one of :class:`LaunchStatus` for a handle."""

    @abstractmethod
    def stop(self, handle: Handle, grace_s: float = 5.0) -> None:
        """Stop the process (idempotent); escalate to kill after ``grace_s``."""


def child_env(spec: LaunchSpec) -> Dict[str, str]:
    """Return the environment overrides for ``spec``'s child process.

    Child stdout is forced unbuffered. Every launcher decides readiness by
    watching for a marker in the child's log, and block buffering kicks in as
    soon as stdout is a file rather than a tty - which it always is here - so a
    marker that is already printed can sit unseen until the wait times out. The
    startup timeouts attributed to ScenarioRunner were this.

    ``SDL_NO_SIGNAL_HANDLERS`` is the second one, and it is a teardown fix.
    SDL installs its own SIGINT and SIGTERM handlers when it initialises, so a
    role that renders anything through pygame stops responding to the signal
    this orchestrator stops it with: its own handler never runs, and it is
    SIGKILLed when the grace expires. Measured on the braking client with the
    ported visualiser: teardown 5.2 s with SDL's handlers
    against 0.2 s without, on every backend, in every case where a worker has
    to be stopped rather than exiting by itself.

    A spec's own variables win over it: they are the app's, and the app knows
    what its hook needs.
    """
    return dict(
        {"PYTHONUNBUFFERED": "1", "SDL_NO_SIGNAL_HANDLERS": "1"}, **(spec.env or {})
    )


class LocalLauncher(Launcher):
    """Spawns local subprocesses and supervises them via their log + exit code."""

    def launch(self, spec: LaunchSpec) -> Handle:
        log_fh = open(spec.log_path, "w") if spec.log_path else subprocess.DEVNULL
        env = dict(os.environ, **child_env(spec))
        # New process group so we can signal the whole tree on teardown.
        proc = subprocess.Popen(
            spec.argv,
            cwd=spec.cwd,
            env=env,
            stdout=log_fh if log_fh is not subprocess.DEVNULL else subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return Handle(spec, proc, log_fh if log_fh is not subprocess.DEVNULL else None)

    def wait_ready(self, handle: Handle, timeout_s: Optional[float] = None) -> bool:
        """Block on the spec's readiness marker using the ported barrier.

        The barrier itself lives in :mod:`cadet.orchestrator.lifecycle`, ported
        from the original orchestrators, so the host path, the emulated path and
        this launcher all wait in the same way.
        """
        from cadet.orchestrator.lifecycle import wait_for_marker

        marker = handle.spec.readiness_marker
        if marker is None:
            return handle.proc.poll() is None
        if not handle.spec.log_path:
            return handle.proc.poll() is None
        return wait_for_marker(
            handle.spec.log_path,
            marker,
            proc=handle.proc,
            timeout=handle.spec.readiness_timeout_s if timeout_s is None else timeout_s,
        )

    def poll(self, handle: Handle) -> str:
        code = handle.proc.poll()
        if code is None:
            return LaunchStatus.RUNNING
        return LaunchStatus.STOPPED if code == 0 else LaunchStatus.FAILED

    def stop(self, handle: Handle, grace_s: float = 5.0) -> None:
        proc = handle.proc
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        if handle.log_fh is not None:
            try:
                handle.log_fh.close()
            except OSError:
                pass
