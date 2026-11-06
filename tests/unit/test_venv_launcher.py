"""Tests for the ssh_venv launcher: what it sends, what it proves, what it refuses.

What this covers:
    The command the launcher builds (the venv's own interpreter substituted for
    the name the orchestrator states, ``-m cadet`` kept because there is no
    image entry point here, the spec's environment carried through ``env``, the
    SSH options that fail rather than prompt), the probe that decides whether a
    venv can run a role at all, the refusals built on it, the staleness warning
    that stands in for an image's recorded commit, and what the placement tells
    a venv role - which is a path for everything a containerised role is given a
    mount point for.

**The remote side is executed, not only inspected.** The payload this launcher
sends is a shell command, so the tests that matter run that exact command
through ``sh`` against a fake venv on this machine: the pid file ends up
holding the pid of the *role* (because the shell ``exec``s rather than forks),
the environment arrives, and the stop command this launcher would send kills
that process and removes the pid file and its directory. That is the whole of
the teardown contract, checked with real processes and no SSH.

What it does **not** cover, and cannot: SSH itself, a second machine, a second
kernel, and therefore none of the four lifecycle cases. Those need a machine
and a key this session has neither of, and the evidence for them belongs beside
``nodes.launcher`` in :mod:`cadet.config.implemented` once someone has run
them. ``ssh_venv`` is refused until then.
"""

import os
import re
import signal
import subprocess
import sys
import time

import pytest

from cadet.core.launcher import IMAGE_PYTHON, LaunchSpec, role_python_version
from cadet.core.staging import TargetMachine
from cadet.core.venv_launcher import (
    NO_INTERPRETER_RC,
    SshVenvLauncher,
    VenvError,
    describe_venv,
    venv_facts,
    venv_python,
)
from cadet.orchestrator.placement import (
    CONTAINER_BIND_HOST,
    CONTAINER_OUTPUTS,
    PlacementError,
    require_venv,
    warn_stale_venv,
)

VENV = "/opt/venvs/cadet"
RUN_DIR = "/home/elsewhere/cadet-run/0123456789-r0"


def spec(name="serve:server", log_path=None):
    """The argv the orchestrator builds for a role placed with this launcher.

    Its interpreter is what ``role_executable('ssh_venv')`` states, which is a
    name and not a path - the path is the venv's and only the launcher knows it.
    """
    return LaunchSpec(
        name=name,
        argv=[IMAGE_PYTHON, "-m", "cadet", "serve", "/home/elsewhere/cadet-configs/x.yaml"],
        log_path=log_path,
    )


def launcher(**kwargs):
    params = {
        "host": "10.0.0.6",
        "venv": VENV,
        "run_dir": RUN_DIR,
        "user": "an-account",
        "dry_run": True,
    }
    params.update(kwargs)
    return SshVenvLauncher(**params)


# --- what runs on the other machine --------------------------------------
def test_the_venvs_own_interpreter_replaces_the_stated_one():
    argv = launcher().role_argv(spec())
    assert argv[0] == "{}/bin/python".format(VENV) == venv_python(VENV)
    # And nothing else moves: unlike the image, a venv has no `cadet` entry
    # point to hand the subcommand to, so the module invocation is kept.
    assert argv[1:] == ["-m", "cadet", "serve", "/home/elsewhere/cadet-configs/x.yaml"]


def test_the_interpreter_is_named_rather_than_the_venv_activated():
    """`activate` is bash-specific and the remote login shell is whatever the
    account's is; a venv interpreter sets its own sys.prefix, so naming it is
    enough and is the form that works in any shell."""
    script = launcher().remote_script(spec())
    assert "activate" not in script
    assert "{}/bin/python".format(VENV) in script


def test_the_remote_script_records_the_pid_before_it_execs():
    script = launcher().remote_script(spec())
    # The order is the contract: $$ is the shell's pid, and `exec` makes it the
    # role's. Recording it after would record a pid that no longer exists.
    assert script.index("echo $$") < script.index("exec ")
    assert "mkdir -p" in script
    assert script.index("mkdir -p") < script.index("echo $$")


def test_the_pid_file_is_per_role_and_under_the_runs_own_directory():
    assert launcher().pid_file(spec("serve:server")) == RUN_DIR + "/serve-server.pid"
    assert launcher().pid_file(spec("client:car")) == RUN_DIR + "/client-car.pid"


def test_the_pid_file_is_not_inside_the_outputs_tree():
    """The outputs tree is fetched back into the run directory; a pid file in it
    would turn up in a run's results."""
    assert "outputs" not in launcher().pid_file(spec())


def test_the_specs_environment_travels_with_the_command():
    script = launcher().remote_script(spec())
    # Unbuffered stdout is what the readiness barrier depends on, and the SDL
    # setting is what teardown depends on; both are child_env's.
    assert "PYTHONUNBUFFERED=1" in script
    assert "SDL_NO_SIGNAL_HANDLERS=1" in script
    assert " env " in " " + script


def test_the_whole_remote_command_is_one_quoted_argument():
    """Otherwise the remote shell re-splits it and the command falls apart."""
    cmd = launcher().command(spec())
    assert cmd[-1].startswith("sh -c ")
    assert len([part for part in cmd if part.startswith("sh -c ")]) == 1


# --- the SSH channel, which is the container launcher's ------------------
def test_ssh_fails_rather_than_prompting():
    cmd = " ".join(launcher().command(spec()))
    assert "BatchMode=yes" in cmd
    assert "ConnectTimeout=" in cmd


def test_known_hosts_is_never_configured_away():
    """Host-key policy stays the operator's; no launcher may relax it."""
    cmd = " ".join(launcher().command(spec()))
    assert "StrictHostKeyChecking" not in cmd
    assert "UserKnownHostsFile" not in cmd


def test_the_target_identity_port_and_jump_host_are_passed():
    cmd = launcher(identity="/k", port=2222, proxy_jump="an-account@gateway").command(spec())
    assert "an-account@10.0.0.6" in cmd
    assert "-i" in cmd and "/k" in cmd
    assert "-p" in cmd and "2222" in cmd
    assert "-J" in cmd and "an-account@gateway" in cmd


def test_dry_run_refuses_to_launch_but_still_builds_the_command():
    held = launcher()
    assert held.command(spec())
    with pytest.raises(RuntimeError) as err:
        held.launch(spec())
    assert "dry-run" in str(err.value)


# --- the remote side, executed ------------------------------------------
def _fake_venv(tmp_path, body):
    """A directory shaped like a venv whose interpreter is ``body``."""
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text(body)
    python.chmod(0o755)
    return str(tmp_path / "venv")


ROLE_STUB = """#!/bin/sh
echo "argv: $*"
echo "unbuffered=$PYTHONUNBUFFERED sdl=$SDL_NO_SIGNAL_HANDLERS"
echo "listening"
exec sleep 30
"""


def _wait_for(predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    return None


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_the_launch_script_runs_and_the_recorded_pid_is_the_role(tmp_path):
    """The payload this launcher sends, executed: `sh -c <script>`.

    No SSH and no second machine - the script is the same string either way,
    and running it here is what shows that `exec` keeps the pid, that the pid
    file holds the role's own and not a parent's, and that the environment
    arrives.

    One thing is a surrogate: here the role's parent is this test, so a
    terminated role stays a zombie until it is reaped and the stop command's
    `kill -0` wait therefore runs its full grace period. On the far side the
    parent is sshd, which reaps the session's child to report its exit status,
    so the wait ends as soon as the role does.
    """
    venv = _fake_venv(tmp_path, ROLE_STUB)
    run_dir = str(tmp_path / "run")
    held = SshVenvLauncher(host="h", venv=venv, run_dir=run_dir, dry_run=True)
    log = tmp_path / "role.log"
    role = spec(log_path=str(log))
    with open(str(log), "w") as fh:
        proc = subprocess.Popen(
            ["sh", "-c", held.remote_script(role)],
            stdout=fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
    try:
        pid_file = held.pid_file(role)
        assert _wait_for(lambda: os.path.exists(pid_file)), "no pid file was written"
        pid = int(_wait_for(lambda: open(pid_file).read().strip()))
        # The recorded pid is the process still running, which is what makes it
        # the handle teardown needs.
        assert _alive(pid)
        # And it is the role itself: every step of the chain execs, so the pid
        # the shell recorded is the one the final program runs under.
        comm = "/proc/{}/comm".format(pid)
        if os.path.exists(comm):
            assert open(comm).read().strip() == "sleep"
        text = _wait_for(lambda: "listening" in open(str(log)).read() and open(str(log)).read())
        assert text, "the role's output did not reach the log"
        assert "unbuffered=1 sdl=1" in text
        assert "-m cadet serve" in text

        # Now the teardown this launcher would send.
        done = subprocess.run(
            ["sh", "-c", held.stop_script(role, grace_s=5)],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
        # SIGTERM did it: the role was stopped inside its grace period and the
        # escalation to SIGKILL was not needed, which is the same escalation
        # the local launcher makes and the behaviour a role's own handler
        # depends on.
        assert proc.wait(timeout=30) == -signal.SIGTERM
        assert not os.path.exists(pid_file), "the pid file was left behind"
        assert not os.path.exists(run_dir), "the pid directory was left behind"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


def test_the_stop_script_is_quiet_about_a_role_that_has_already_exited(tmp_path):
    """Teardown runs after a normal completion too, so a pid file whose process
    is gone is the ordinary case and must not fail the teardown."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    held = SshVenvLauncher(host="h", venv="/v", run_dir=str(run_dir), dry_run=True)
    role = spec()
    with open(held.pid_file(role), "w") as fh:
        # A pid that cannot exist, so this is the "already exited" case. Not 0,
        # which is not a free choice: `kill 0` signals the caller's own process
        # group, so a pid file holding it would make a teardown kill the test
        # run - and it is what an interrupted `echo $$` could plausibly leave.
        fh.write("2147483647\n")
    done = subprocess.run(
        ["sh", "-c", held.stop_script(role)], capture_output=True, text=True
    )
    assert done.returncode == 0
    assert not os.path.exists(held.pid_file(role))


def test_an_empty_pid_file_is_not_a_kill_of_everything(tmp_path):
    """An unquoted empty pid would make `kill -TERM $p` into `kill -TERM`, and a
    truncated pid file is exactly what an interrupted launch leaves."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    held = SshVenvLauncher(host="h", venv="/v", run_dir=str(run_dir), dry_run=True)
    role = spec()
    open(held.pid_file(role), "w").close()
    done = subprocess.run(
        ["sh", "-c", held.stop_script(role)], capture_output=True, text=True
    )
    assert done.returncode == 0
    assert "usage" not in (done.stderr or "").lower()


def test_stop_kills_the_remote_role_before_reaping_the_local_client(monkeypatch):
    """The other order leaves the role running with nothing left that knows its
    pid, which is the leftover this launcher exists to not create."""
    events = []

    class _Proc:
        def poll(self):
            return None

        def terminate(self):
            events.append("terminate-local")

        def wait(self, timeout=None):
            return 0

    from cadet.core import venv_launcher as module

    def _run(argv, **_kwargs):
        events.append("remote:" + argv[-1])
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", _run)
    held = launcher(dry_run=False)
    role = spec()
    held.stop(module.Handle(role, _Proc(), None))
    assert len(events) == 2
    assert events[0].startswith("remote:")
    assert "kill -TERM" in events[0] and "kill -KILL" in events[0]
    assert events[1] == "terminate-local"


# --- the probe, against a real interpreter -------------------------------
def _local_machine():
    """``TargetMachine`` with no launcher: the probe runs on this machine.

    Which is the point - the probe is the same command either way, so it can be
    proved here and only its transport is unproven.
    """
    return TargetMachine(log=lambda *_a: None)


def test_the_probe_reports_a_real_venvs_python_install_and_commit(tmp_path):
    venv = _fake_venv(tmp_path, "")
    os.remove(venv_python(venv))
    os.symlink(sys.executable, venv_python(venv))
    facts = venv_facts(_local_machine(), venv)
    assert facts["python"] == "{}.{}".format(*sys.version_info[:2])
    # Where its CADET is installed, which is what the commit is read from.
    assert os.path.isdir(os.path.join(str(facts["root"]), "cadet"))
    assert facts["revision"] is None or re.match(r"\A[0-9a-f]{40}\Z", str(facts["revision"]))
    assert facts["dirty"] in (True, False)
    assert "python {}".format(facts["python"]) in describe_venv(facts)


def test_a_venv_that_does_not_exist_is_named_as_such(tmp_path):
    with pytest.raises(VenvError) as err:
        venv_facts(_local_machine(), str(tmp_path / "nothing-here"))
    assert "no interpreter at" in str(err.value)
    assert NO_INTERPRETER_RC == 127


def test_a_venv_without_cadet_in_it_is_a_separate_failure(tmp_path):
    venv = _fake_venv(tmp_path, "#!/bin/sh\necho 'No module named cadet' >&2\nexit 1\n")
    with pytest.raises(VenvError) as err:
        venv_facts(_local_machine(), venv)
    assert "cannot import cadet" in str(err.value)
    # The venv's own words are kept: they are what says which of the many ways
    # an import fails this one was.
    assert "No module named cadet" in str(err.value)


# --- the refusals the placement makes from it ---------------------------
class _Node:
    def __init__(self, name="server", venv=VENV):
        self.name = name
        self.venv = venv
        self.launcher = "ssh_venv"
        self.host = "10.0.0.6"
        self.gpu_index = None


class _FakeMachine:
    """A machine that answers the probe however a test needs it to."""

    def __init__(self, reply=None, code=0):
        self.reply = reply
        self.code = code
        self.commands = []

    def describe(self):
        return "the other machine"

    def run(self, argv, timeout_s=None):
        self.commands.append(list(argv))
        return self.code, self.reply or ""


def _probe_reply(python="3.8", root="/opt/cadet", revision="", dirty="clean"):
    return "\n".join([python, root, revision, dirty]) + "\n"


def test_a_node_that_names_no_venv_is_refused_before_anything_is_contacted():
    machine = _FakeMachine()
    with pytest.raises(PlacementError) as err:
        require_venv(machine, _Node(venv=None))
    assert "names no virtualenv" in str(err.value)
    assert "nodes[*].venv" in str(err.value)
    assert machine.commands == []


def test_a_venv_that_cannot_run_a_role_is_refused_and_told_how_to_be_fixed():
    machine = _FakeMachine(reply="ImportError: no cadet", code=1)
    with pytest.raises(PlacementError) as err:
        require_venv(machine, _Node())
    message = str(err.value)
    # The decision this launcher made, in the refusal: nothing is shipped, so
    # the venv has to have CADET installed in it.
    assert "not shipped" in message
    assert "-m venv" in message and "pip install" in message
    assert VENV in message


def test_a_venv_on_another_python_is_refused_rather_than_warned_about():
    """The launcher kind *states* a version and the simulator version check
    reasons from that statement, so a venv that is not it would make the check
    sound with a false premise."""
    wanted = role_python_version("ssh_venv")
    machine = _FakeMachine(reply=_probe_reply(python="3.11"))
    with pytest.raises(PlacementError) as err:
        require_venv(machine, _Node())
    assert "python 3.11" in str(err.value)
    assert wanted in str(err.value)


def test_a_venv_that_can_run_a_role_is_accepted_and_described():
    machine = _FakeMachine(
        reply=_probe_reply(revision="a" * 40, dirty="dirty")
    )
    facts = require_venv(machine, _Node())
    assert facts["python"] == role_python_version("ssh_venv")
    assert facts["revision"] == "a" * 40
    assert facts["dirty"] is True


# --- the staleness warning, which is this launcher's analogue of an image's
def _warn(monkeypatch, facts, head="b" * 40, dirty=False):
    from cadet.orchestrator import placement as module

    monkeypatch.setattr(module, "repo_revision", lambda *_a, **_k: (head, dirty))
    lines = []
    return module.warn_stale_venv(facts, lines.append, _FakeMachine(), VENV), lines


def test_an_install_from_another_commit_is_warned_about_not_refused(monkeypatch):
    facts = {"python": "3.8", "root": "/opt/cadet", "revision": "a" * 40, "dirty": False}
    lines, logged = _warn(monkeypatch, facts)
    assert len(lines) == 1
    assert lines == logged
    assert "WARNING" in lines[0]
    assert "aaaaaaaaaa" in lines[0] and "bbbbbbbbbb" in lines[0]
    # Named, because the mismatch alone sends the reader looking for the cause
    # of a change that appears not to work.
    assert "pip install" in lines[0]


def test_an_install_with_no_recorded_commit_says_the_question_cannot_be_answered(
    monkeypatch,
):
    facts = {"python": "3.8", "root": "/opt/cadet", "revision": None, "dirty": False}
    lines, _logged = _warn(monkeypatch, facts)
    assert len(lines) == 1
    assert "records no commit" in lines[0]


def test_a_matching_install_against_a_dirty_tree_is_still_warned_about(monkeypatch):
    facts = {"python": "3.8", "root": "/opt/cadet", "revision": "b" * 40, "dirty": False}
    lines, _logged = _warn(monkeypatch, facts, dirty=True)
    assert len(lines) == 1
    assert "uncommitted changes" in lines[0]


def test_a_matching_clean_install_is_not_warned_about(monkeypatch):
    facts = {"python": "3.8", "root": "/opt/cadet", "revision": "b" * 40, "dirty": False}
    lines, _logged = _warn(monkeypatch, facts)
    assert lines == []


def test_an_install_dirty_on_its_own_machine_is_warned_about(monkeypatch):
    """It reports a commit and is not running it, which is the one case the
    image analogue cannot have."""
    facts = {"python": "3.8", "root": "/opt/cadet", "revision": "b" * 40, "dirty": True}
    lines, _logged = _warn(monkeypatch, facts)
    assert any("changes of its own" in line for line in lines)


def test_nothing_is_warned_about_when_this_tree_is_not_a_checkout(monkeypatch):
    """An installed copy of CADET need not be a git checkout, and a warning that
    compares against nothing is noise."""
    facts = {"python": "3.8", "root": "/opt/cadet", "revision": None, "dirty": False}
    lines, _logged = _warn(monkeypatch, facts, head=None)
    assert lines == []


# --- what the placement tells a venv role -------------------------------
class _Outputs:
    root = "outputs"


class _Config:
    def __init__(self, nodes):
        self.nodes = list(nodes)
        self.links = []
        self.outputs = _Outputs()


class _StagedConfig:
    def __init__(self, path):
        self.path = path


class _StubLauncher:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.calls = []

    def run_remote(self, argv, timeout_s=None):
        self.calls.append(list(argv))
        return 0, ""


def _placement_with_a_venv_node(tmp_path):
    from cadet.orchestrator.placement import Placement

    node = _Node()
    place = Placement(_Config([node]), "0123456789-r0", str(tmp_path), log=lambda *_a: None)
    # What _prepare_venv records, without the machine it would have asked.
    remote_outputs = "/home/elsewhere/cadet-outputs/0123456789-r0"
    place.venv_outputs[node.name] = remote_outputs
    place.staged_configs[node.name] = _StagedConfig(
        "/home/elsewhere/cadet-configs/0123456789-r0.yaml"
    )
    place.remote_outputs[node.name] = None
    place.venv_launchers[node.name] = _StubLauncher(RUN_DIR)
    place.venv_facts[node.name] = {
        "python": "3.8",
        "root": "/opt/cadet",
        "revision": "c" * 40,
        "dirty": False,
    }
    return place, node, remote_outputs


def test_a_venv_role_is_told_the_outputs_tree_itself_not_a_mount(tmp_path):
    place, node, remote_outputs = _placement_with_a_venv_node(tmp_path)
    assert place.role_overrides_for(node) == ["outputs.root={}".format(remote_outputs)]
    # Which is not the containerised answer: there is no mount to name.
    assert CONTAINER_OUTPUTS not in place.role_overrides_for(node)[0]


def test_a_venv_role_is_told_the_staged_configs_own_path(tmp_path):
    place, node, _outputs = _placement_with_a_venv_node(tmp_path)
    assert place.role_config_path(node) == "/home/elsewhere/cadet-configs/0123456789-r0.yaml"


def test_a_venv_server_binds_an_address_it_holds_on_its_own_machine(tmp_path):
    place, node, _outputs = _placement_with_a_venv_node(tmp_path)
    node.role = "server"
    assert place.role_args(node) == ["--bind-host", CONTAINER_BIND_HOST]


def test_a_venv_role_is_told_the_staged_model_directory_on_its_machine(tmp_path):
    place, node, _outputs = _placement_with_a_venv_node(tmp_path)
    node.role = "server"

    class _Staging:
        path = "/home/elsewhere/.cache/cadet-models/runs/0123456789-r0"
        machine = None

    place.model_staging[node.name] = _Staging()
    assert place.role_args(node)[-2:] == ["--model-dir", _Staging.path]


def test_the_pid_directory_is_removed_when_the_placement_stops(tmp_path):
    place, node, _outputs = _placement_with_a_venv_node(tmp_path)
    stub = place.venv_launchers[node.name]
    place.remote_outputs.clear()  # fetching is RemoteOutputs' own test
    place.staged_configs.clear()
    removed = place.stop()
    assert ["rmdir", "--", RUN_DIR] in stub.calls
    assert "venv-run-dir:{}".format(node.name) in removed
    # Non-recursive, so a concurrent run's directory is never taken with it.
    assert not any("-rf" in part for call in stub.calls for part in call)


def test_the_venv_is_recorded_in_the_placements_status(tmp_path):
    place, node, _outputs = _placement_with_a_venv_node(tmp_path)
    status = place.status()
    assert node.name in status["venvs"]
    assert "cccccccccc" in status["venvs"][node.name]


# --- a config that names no venv node is unaffected ---------------------
def test_a_placement_with_no_venv_node_contacts_nothing(tmp_path):
    from cadet.orchestrator.placement import Placement

    class _Local:
        name = "client"
        launcher = "local"
        role = "client"
        host = "localhost"
        port = None
        venv = None
        container = None
        gpu_index = None
        transport_tuning = None

    place = Placement(_Config([_Local()]), "0123456789-r0", str(tmp_path), log=lambda *_a: None)
    assert place.venv_nodes() == []
    assert place.offmachine_nodes() == []
    assert place.start() == {}
    assert place.role_overrides_for(_Local()) == []
    assert place.role_config_path(_Local()) is None
    assert place.role_args(_Local()) == []
    assert place.stop() == []
    assert place.status()["venvs"] == {}
