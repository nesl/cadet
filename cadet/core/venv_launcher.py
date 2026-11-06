"""Launcher that starts a role as a plain process in a virtualenv on another machine.

What this is for:
    The remaining placement in ``nodes[*].launcher``: a machine that has no
    docker, or an account that is not allowed to use it, but does have a Python
    environment with CADET installed in it. The role is an ordinary process
    there, started over the same SSH channel the container launcher uses, so the
    orchestrator's ordering, readiness and teardown logic is unchanged.

What it contains:
    - :class:`SshVenvLauncher`: the launcher. It builds the remote shell command
      that records the role's pid and then ``exec``s the venv's interpreter, and
      it stops the role by that pid.
    - :func:`venv_python`: the interpreter inside a virtualenv.
    - :func:`venv_facts` and :class:`VenvError`: what the far venv is - its
      Python version, where its CADET is installed and which commit that copy
      records - asked in one round trip, because the answers decide whether the
      run can proceed at all.
    - :func:`describe_venv`: that answer as one log line.

How it fits into CADET:
    :class:`cadet.orchestrator.placement.Placement` builds one of these per
    ``ssh_venv`` node, after refusing a venv that cannot run a role
    (:func:`cadet.orchestrator.placement.require_venv`) and warning about one
    whose code is not this repository's
    (:func:`cadet.orchestrator.placement.warn_stale_venv`). The outputs tree,
    the run's config and the model artifacts are staged on that machine by the
    placement through the same classes the container launcher uses, because a
    venv role has the same problem a container one does: the paths the
    orchestrator was given name nothing there.

    **This launcher ships no code.** It requires CADET to be installed in the
    venv already, which is what the field name says and what a virtualenv is;
    the orchestrator's working tree is never copied. The cost of that choice is
    the one images have - the version that runs is whatever is installed - so it
    is answered the same way: the installed copy's commit is read and compared
    with this repository's, and a mismatch is warned about rather than refused.
    A run's app modules (``--app-module``) have to be importable in that venv
    for the same reason.

How to configure it:
    Per node: ``launcher: ssh_venv``, ``venv`` (the virtualenv's path on that
    machine), ``host``, ``user``, ``ssh_port`` and ``ssh_identity``. There is no
    ``container``: nothing is containerised here.
"""

import os
import shlex
import subprocess
from typing import Dict, List, Optional

from cadet.core.launcher import Handle, Launcher, LaunchSpec, LaunchStatus, child_env
from cadet.core.ssh import RemoteError, SshTarget

# Where a virtualenv keeps its interpreter. Naming the interpreter rather than
# sourcing `activate` is deliberate and is the robust form: a venv interpreter
# sets its own sys.prefix, so `<venv>/bin/python -m cadet` runs with that
# environment's packages without a shell having been "activated" at all - and
# `activate` is bash-specific, while the remote shell is whatever the account's
# login shell happens to be.
VENV_BIN = "bin"
VENV_INTERPRETER = "python"
# What the remote probe prints, in order: the venv's Python "major.minor", the
# directory its CADET is installed from, the commit that directory records (or
# an empty line), and whether it has uncommitted changes. One round trip,
# because every one of these costs a connection and all four are wanted
# together.
#
# The commit is read with git in that directory, which is the venv's answer to
# the OCI revision label an image carries: an editable install points at a
# checkout and can be asked, while a copied-and-installed tree has no history
# and is recorded as unknown - the same two cases as an image with and without
# the label.
VENV_PROBE = "\n".join(
    [
        "import os, subprocess, sys",
        "import cadet",
        "print('{}.{}'.format(*sys.version_info[:2]))",
        "root = os.path.dirname(os.path.dirname(os.path.abspath(cadet.__file__)))",
        "print(root)",
        "def git(args):",
        "    try:",
        "        done = subprocess.run(['git'] + args, cwd=root,",
        "                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)",
        "    except OSError:",
        "        return ''",
        "    if done.returncode != 0:",
        "        return ''",
        "    return done.stdout.decode('utf-8', 'replace').strip()",
        "print(git(['rev-parse', 'HEAD']))",
        "print('dirty' if git(['status', '--porcelain']) else 'clean')",
    ]
)
# How long the probe is given. Importing the CADET package is all it does, but
# that import is from an environment whose disk may be cold.
PROBE_TIMEOUT_S = 60.0
# Exit code the remote wrapper uses when the venv has no interpreter, so the
# refusal can say that rather than relaying a shell's "not found".
NO_INTERPRETER_RC = 127


class VenvError(Exception):
    """The virtualenv a node names cannot run a role."""


def venv_python(venv: str) -> str:
    """Return the interpreter path inside ``venv``."""
    return "{}/{}/{}".format(venv.rstrip("/"), VENV_BIN, VENV_INTERPRETER)


def venv_facts(machine, venv: str) -> Dict[str, object]:
    """Return what the venv on ``machine`` is, or raise :class:`VenvError`.

    ``{"python": "3.8", "root": "/path/to/install", "revision": "abc1234..." or
    ``None``, "dirty": bool}``. ``machine`` is a
    :class:`~cadet.core.staging.TargetMachine`, so this works unchanged against
    a venv on this machine - which is what makes it testable without a second
    one.

    It raises rather than returning a partial answer: every field is needed
    before a role is started, and a venv that cannot answer cannot run one.
    """
    python = venv_python(venv)
    # `test -x` first, so a missing venv is reported as a missing venv instead
    # of as whatever the remote shell says about a path it cannot execute.
    script = "test -x {0} || exit {1}; exec {0} -c {2}".format(
        shlex.quote(python), NO_INTERPRETER_RC, shlex.quote(VENV_PROBE)
    )
    code, output = machine.run(["sh", "-c", script], timeout_s=PROBE_TIMEOUT_S)
    if code == NO_INTERPRETER_RC:
        raise VenvError(
            "no interpreter at {} on {}".format(python, machine.describe())
        )
    lines = [line.rstrip() for line in output.splitlines()]
    if code != 0 or len(lines) < 4:
        raise VenvError(
            "the venv {} on {} cannot import cadet: {}".format(
                venv, machine.describe(), output.strip() or "rc=" + str(code)
            )
        )
    version, root, revision, dirty = lines[-4], lines[-3], lines[-2], lines[-1]
    return {
        "python": version.strip(),
        "root": root.strip(),
        "revision": revision.strip() or None,
        "dirty": dirty.strip() == "dirty",
    }


def describe_venv(facts: Dict[str, object]) -> str:
    """Return a one-line description of a venv, for a log line or a run record."""
    revision = facts.get("revision")
    return "python {}, cadet from {} ({}{})".format(
        facts.get("python"),
        facts.get("root"),
        str(revision)[:10] if revision else "no recorded commit",
        ", dirty" if facts.get("dirty") else "",
    )


class SshVenvLauncher(SshTarget, Launcher):
    """Starts each role as a process in a virtualenv on another machine, over SSH.

    What is sent is one ``sh -c`` command that records the role's pid in a file
    and then ``exec``s the venv's interpreter, so the pid in that file *is* the
    role. That file is the whole of the teardown story and is why it exists:
    killing the local ``ssh`` client does not reliably kill what it started on
    the far side, which is the same problem the container launcher solves by
    removing the container by name. Here the name is a pid, recorded by the
    remote shell itself rather than guessed at from a command line - every
    role's command line is ``cadet <sub>``, so it matches any other run's too.

    The role's stdout and stderr come back down the SSH connection into the
    run's own log file, so the readiness barrier and the failure reports read
    the same file they would for a local role.

    ``dry_run`` makes every operation return the command it would have run
    without contacting anything, which is how a two-host setup can be checked
    before any remote access exists.
    """

    def __init__(
        self,
        host: str,
        venv: str,
        run_dir: str = "",
        install_root: str = "",
        user: Optional[str] = None,
        port: Optional[int] = None,
        identity: Optional[str] = None,
        proxy_jump: Optional[str] = None,
        dry_run: bool = False,
        connect_timeout_s: int = 10,
    ):
        SshTarget.__init__(
            self,
            host=host,
            user=user,
            port=port,
            identity=identity,
            proxy_jump=proxy_jump,
            connect_timeout_s=connect_timeout_s,
        )
        self.venv = venv
        # A directory on the other machine, under that account's home, holding
        # this run's pid files. Passed in rather than derived here because the
        # placement has already asked that machine where its home is, and every
        # such question is another connection.
        # Where the installed CADET lives on that machine, which is the
        # directory the role runs in. It matters for one reason: an app module
        # named with --app-module is a path into a project, and
        # import_app_modules looks for it in the working directory. A role
        # started over SSH begins in the account's home, where `examples` is not
        # importable, so `--app-module examples.distributed_sensing.app` failed with
        # `ModuleNotFoundError: No module named 'examples'` even though the
        # install was an editable checkout that had it. The container launchers
        # set a workdir for the same reason; this is theirs.
        #
        # Set by the placement from the venv probe, which already reports it.
        self.install_root = install_root
        # Optional here and set by the placement once the remote account's home
        # is known, because the directory is under that home and asking for it
        # needs this launcher to exist first. A required argument made the two
        # orderings irreconcilable and the placement silently constructed
        # without it; _pid_path refuses rather than building a path under "".
        self.run_dir = run_dir
        self.dry_run = dry_run

    # -- what runs on the other machine --------------------------------------
    def python(self) -> str:
        """Return the interpreter this launcher starts roles with."""
        return venv_python(self.venv)

    def pid_file(self, spec: LaunchSpec) -> str:
        """Return the file on the other machine holding this role's pid."""
        # Spec names are "<sub>:<node>", e.g. "serve:server"; the colon is legal
        # in a path and confusing in one.
        if not self.run_dir:
            raise RemoteError(
                "this launcher has no run directory yet, so there is nowhere to "
                "write the role's pid file. The placement sets it once the "
                "remote account's home is known; a launcher built outside the "
                "placement has to be given one."
            )
        return "{}/{}.pid".format(self.run_dir.rstrip("/"), spec.name.replace(":", "-"))

    def role_argv(self, spec: LaunchSpec) -> List[str]:
        """Return the role's argv with the venv's interpreter in front.

        The orchestrator builds ``<python> -m cadet <sub> ...`` for local
        execution, where ``<python>`` is what
        :func:`cadet.core.launcher.role_executable` states for this launcher
        kind - a name, not a path, because the path is a property of the venv
        and only this launcher knows it. So the name is substituted and nothing
        else is touched: the role runs the same command it would anywhere else.
        """
        argv = list(spec.argv)
        if len(argv) > 1 and argv[1] == "-m":
            argv[0] = self.python()
        return argv

    def remote_script(self, spec: LaunchSpec) -> str:
        """Return the shell command the other machine runs to start this role.

        ``echo $$`` before ``exec`` is the point: the shell's own pid becomes
        the role's, because ``exec`` replaces the shell rather than forking, so
        the recorded pid is the process to signal and not a parent of it.

        The spec's environment travels with it, as it does locally, through
        ``env`` - so the unbuffered stdout the readiness barrier depends on and
        the SDL signal-handler fix teardown depends on apply here too.
        """
        argv = ["env"] + [
            "{}={}".format(key, value) for key, value in sorted(child_env(spec).items())
        ]
        argv += self.role_argv(spec)
        prefix = ""
        if self.install_root:
            # Fail rather than run the role from the wrong directory: a role
            # that starts in the home directory looks like it works until an
            # app module cannot be found, and that error names the module
            # rather than the directory.
            prefix = "cd {} && ".format(shlex.quote(self.install_root))
        return "{}mkdir -p {} && echo $$ > {} && exec {}".format(
            prefix,
            shlex.quote(self.run_dir),
            shlex.quote(self.pid_file(spec)),
            " ".join(shlex.quote(part) for part in argv),
        )

    def stop_script(self, spec: LaunchSpec, grace_s: float = 5.0) -> str:
        """Return the shell command that stops this role on the other machine.

        SIGTERM, then SIGKILL once the grace period is up, then the pid file -
        the same escalation the local launcher does, done where the process is.
        It exits 0 for a role that is already gone, because teardown runs after
        a normal completion as well as after a failure, so a pid file whose
        process has exited is the ordinary case and not an error.
        """
        return (
            'p=$(cat {pid} 2>/dev/null); '
            'if [ -n "$p" ]; then '
            'kill -TERM "$p" 2>/dev/null; '
            'i=0; while [ $i -lt {grace} ] && kill -0 "$p" 2>/dev/null; do '
            'sleep 1; i=$((i+1)); done; '
            'kill -0 "$p" 2>/dev/null && kill -KILL "$p" 2>/dev/null; '
            'fi; '
            # The pid file first, then the directory non-recursively, so the
            # removal succeeds only once this run's last role has gone and never
            # takes a concurrent run's directory with it - the same rule the
            # outputs and config staging follow.
            # The run directory, then its parent, both non-recursively and in
            # that order - the same rule RemoteOutputs and StagedRunConfig
            # follow. Without the second rmdir the parent was left behind
            # empty, which is not harmful but is a leftover the other two do
            # not leave, and "nothing on that machine" should mean the same
            # thing for all three.
            'rm -f {pid}; rmdir {dir} 2>/dev/null; rmdir {parent} 2>/dev/null; exit 0'.format(
                pid=shlex.quote(self.pid_file(spec)),
                grace=max(1, int(grace_s)),
                dir=shlex.quote(self.run_dir),
                parent=shlex.quote(os.path.dirname(self.run_dir.rstrip("/"))),
            )
        )

    def command(self, spec: LaunchSpec) -> List[str]:
        """Return the full local command that starts this role (useful for dry runs)."""
        return self.remote_command(["sh", "-c", self.remote_script(spec)])

    def stop_command(self, spec: LaunchSpec, grace_s: float = 5.0) -> List[str]:
        """Return the full local command that stops this role."""
        return self.remote_command(["sh", "-c", self.stop_script(spec, grace_s)])

    # -- the Launcher interface ----------------------------------------------
    def launch(self, spec: LaunchSpec) -> Handle:
        if self.dry_run:
            raise RuntimeError(
                "this launcher is in dry-run, which contacts nothing and so "
                "cannot start a role. Inspect the command it would have run "
                "with command(); clear dry_run to run it."
            )
        log_fh = open(spec.log_path, "w") if spec.log_path else None
        proc = subprocess.Popen(
            self.command(spec),
            # ssh reads its own stdin and would compete with the orchestrator's
            # terminal for it; no role reads stdin.
            stdin=subprocess.DEVNULL,
            stdout=log_fh if log_fh is not None else subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return Handle(spec, proc, log_fh)

    def wait_ready(self, handle: Handle, timeout_s: Optional[float] = None) -> bool:
        """Block on the spec's readiness marker, using the ported barrier.

        The marker is read from the local log file, which is where the role's
        output arrives: it travels back over the SSH connection exactly as a
        local child's would through a pipe, so this is the same wait the local
        launcher makes, made by the same code.
        """
        from cadet.orchestrator.lifecycle import wait_for_marker

        marker = handle.spec.readiness_marker
        if marker is None or not handle.spec.log_path:
            return handle.proc.poll() is None
        return wait_for_marker(
            handle.spec.log_path,
            marker,
            proc=handle.proc,
            timeout=handle.spec.readiness_timeout_s if timeout_s is None else timeout_s,
        )

    def poll(self, handle: Handle) -> str:
        """Return the role's status as the ssh client reports it.

        The client's exit code is the remote command's, so a role that failed is
        reported as failed. An ssh that could not connect exits 255, which is
        also a failure - and one whose reason is in the log file.
        """
        code = handle.proc.poll()
        if code is None:
            return LaunchStatus.RUNNING
        return LaunchStatus.STOPPED if code == 0 else LaunchStatus.FAILED

    def stop(self, handle: Handle, grace_s: float = 5.0) -> None:
        """Stop the remote role by its pid, then reap the local ssh client.

        In that order, and not the other: killing the ssh client first would
        leave the role running on the other machine with nothing left that knows
        its pid, which is exactly the leftover this launcher must not create.
        """
        if self.dry_run:
            return
        subprocess.run(
            self.stop_command(handle.spec, grace_s), capture_output=True, text=True
        )
        if handle.proc.poll() is None:
            try:
                handle.proc.terminate()
                handle.proc.wait(timeout=grace_s)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    handle.proc.kill()
                except OSError:
                    pass
        if handle.log_fh is not None:
            try:
                handle.log_fh.close()
            except OSError:
                pass
