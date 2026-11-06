"""The SSH channel to the machine a node lives on.

What this is for:
    Two launchers put a role on another machine: one wraps it in a container
    (:class:`~cadet.core.docker_launcher.SshDockerLauncher`) and one runs it as a
    process in a virtualenv (:class:`~cadet.core.venv_launcher.SshVenvLauncher`).
    What they place differs; how they reach the machine does not, and the
    reaching is where the decisions that matter are. This module holds that half
    once so the two cannot drift apart - a second copy of it would be a second
    place for the host-key policy to be relaxed in.

What it contains:
    - :class:`RemoteError`: the machine could not be asked or prepared.
    - :class:`SshTarget`: the ``ssh`` invocation, the quoting that survives the
      remote shell, a one-shot command runner, and the few facts a remote
      placement has to ask of the other machine rather than assume.

How it fits into CADET:
    :class:`~cadet.orchestrator.placement.Placement` builds one of these per
    remote node, by way of the launcher it is mixed into, and everything that
    needs the other machine - :class:`~cadet.core.staging.TargetMachine`, the
    outputs staging, the staged config, the clock probe - goes through it.

How to configure it:
    Per node: ``user``, ``host``, ``ssh_port`` and ``ssh_identity``. The
    connect timeout is a constructor argument, not a config field: it bounds a
    failure rather than describing the experiment.
"""

import shlex
import subprocess
from typing import List, Optional


class RemoteError(Exception):
    """The machine a remote node lives on could not be asked or prepared."""


class SshTarget:
    """One machine reached over SSH, and the commands that ask it anything.

    SSH options are fixed to fail rather than prompt: batch mode so a missing key
    is an error instead of a password prompt, and an explicit connect timeout.
    ``known_hosts`` is never modified — host-key policy stays the operator's.

    Mixed into a launcher rather than owned by one, because what runs on the
    far side is the launcher's business and getting there is not.
    """

    def __init__(
        self,
        host: str,
        user: Optional[str] = None,
        port: Optional[int] = None,
        identity: Optional[str] = None,
        proxy_jump: Optional[str] = None,
        connect_timeout_s: int = 10,
    ):
        self.host = host
        self.user = user
        self.port = port
        self.identity = identity
        self.proxy_jump = proxy_jump
        self.connect_timeout_s = connect_timeout_s
        self._facts = None

    def ssh_prefix(self) -> List[str]:
        """Return the ``ssh`` invocation that fronts every remote command."""
        argv = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout={}".format(self.connect_timeout_s),
        ]
        if self.identity:
            argv += ["-i", self.identity]
        if self.proxy_jump:
            argv += ["-J", self.proxy_jump]
        if self.port:
            argv += ["-p", str(self.port)]
        target = "{}@{}".format(self.user, self.host) if self.user else self.host
        argv.append(target)
        return argv

    def remote_command(self, argv: List[str]) -> List[str]:
        """Return the local argv that runs ``argv`` on the remote machine.

        The remote side runs through a shell, so the command is quoted as one
        argument; otherwise the remote shell would re-split it and the flags
        would be lost.
        """
        return self.ssh_prefix() + [" ".join(shlex.quote(part) for part in argv)]

    def run_remote(self, argv: List[str], timeout_s: Optional[float] = None):
        """Run one command on the remote machine; return ``(rc, output)``.

        Used for the few things a remote placement has to ask the other machine
        rather than assume: where its account's home is, which uid it is, and
        the staging directories' creation and removal.
        """
        try:
            result = subprocess.run(
                self.remote_command(argv),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return 255, "{}: {}".format(type(exc).__name__, exc)
        return result.returncode, (result.stdout or "") + (result.stderr or "")

    def remote_facts(self):
        """Return ``{"home", "uid", "gid"}`` for the remote account, cached.

        The orchestrator's own values are the wrong answer on another machine:
        the home directory is where the staging tree can be created, and the
        uid/gid are what a container has to run as for the role to be able to
        write into it. Asked once per launcher and cached, because every call
        costs an SSH round trip.
        """
        if self._facts is not None:
            return self._facts
        code, output = self.run_remote(
            ["sh", "-c", "cd && pwd && id -u && id -g"], timeout_s=self.connect_timeout_s + 10
        )
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if code != 0 or len(lines) < 3:
            raise RemoteError(
                "cannot read the remote account's home and uid over "
                "{!r}: {}".format(" ".join(self.ssh_prefix()), output.strip() or "rc=" + str(code))
            )
        home, uid, gid = lines[-3], lines[-2], lines[-1]
        if not home.startswith("/") or not uid.isdigit() or not gid.isdigit():
            raise RemoteError(
                "the remote account reported an unusable home/uid/gid: "
                "{!r} {!r} {!r}".format(home, uid, gid)
            )
        self._facts = {"home": home, "uid": uid, "gid": gid}
        return self._facts
