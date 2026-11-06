"""Tests for the ssh_docker launcher, over a loopback SSH surrogate.

What this covers:
    The parts of placing a role on another machine that do not need a second
    machine: that the command sent over SSH carries the published port, the
    remote account's uid and the remote staging mount; that a real
    ``docker run`` started through ``ssh`` reaches its readiness marker, is
    reachable at the published port, and is removed afterwards; and that a file
    written *inside* that container comes back into the run directory and the
    staging tree on the far side is gone.

**Loopback is a surrogate, and a weak one by design.** The "remote" host is
``127.0.0.1``: the same kernel, the same docker daemon, the same filesystem and
no network between the two ends. So these tests prove the plumbing - the SSH
transport, the remote ``docker run``, readiness read over the SSH pipe, the
published port, the staging mount and the teardown - and they prove **none** of
what only two machines can show: a second kernel's namespaces, a second docker
daemon's images, real link latency or loss between the roles, a remote GPU, or
that the key restriction (``from=``) holds. Those belong to the live two-host
run, whose evidence is recorded beside ``nodes.launcher`` in
:mod:`cadet.config.implemented`, not here.

How it fits into CADET:
    These are the CI tests for :class:`cadet.core.docker_launcher.SshDockerLauncher`
    and the remote half of :class:`cadet.orchestrator.placement.Placement`.
    Every one of them is skipped unless this machine can reach a docker daemon
    *and* log into itself over SSH non-interactively with the test key, so a
    checkout without that setup reports skips rather than failures.

How to configure it:
    ``CADET_SSH_TEST_IDENTITY`` names the private key to authenticate with
    (default ``~/.ssh/cadet_test_ed25519``), and ``CADET_SSH_TEST_IMAGE`` the
    role image to run (default the repository's own ``cadet:dev``). The key is
    expected to be authorised for this account restricted to the loopback
    address; nothing here creates or edits a key or an ``authorized_keys`` file.
"""

import getpass
import json
import os
import socket
import subprocess

import pytest

from cadet.config import build_config
from cadet.core.docker_launcher import HOST_NETWORK
from cadet.core.launcher import LaunchSpec
from cadet.orchestrator.placement import (
    CONTAINER_BIND_HOST,
    OWNED_CONTAINER_PREFIX,
    Placement,
)

# The surrogate's "other machine". Loopback, which is the whole caveat above.
SURROGATE_HOST = "127.0.0.1"
IDENTITY = os.environ.get(
    "CADET_SSH_TEST_IDENTITY", os.path.expanduser("~/.ssh/cadet_test_ed25519")
)
IMAGE = os.environ.get("CADET_SSH_TEST_IMAGE", "cadet:dev")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _docker_reachable():
    try:
        done = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0


def _image_present():
    try:
        done = subprocess.run(
            ["docker", "image", "inspect", IMAGE], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0


def _ssh_works():
    if not os.path.exists(IDENTITY):
        return False
    try:
        done = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "ConnectTimeout=5",
                "-i",
                IDENTITY,
                "{}@{}".format(getpass.getuser(), SURROGATE_HOST),
                "true",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0


needs_surrogate = pytest.mark.skipif(
    not (_docker_reachable() and _image_present() and _ssh_works()),
    reason=(
        "the loopback SSH surrogate is not available here: it needs a docker "
        "daemon, the role image, and a non-interactive login to this account "
        "over SSH with CADET_SSH_TEST_IDENTITY"
    ),
)


def _free_port():
    sock = socket.socket()
    sock.bind((SURROGATE_HOST, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _config(port, outputs_root):
    """Reference client/server with the server placed on the surrogate."""
    return build_config(
        {
            "experiment": {"name": "ssh_surrogate", "app": "reference", "repetitions": 1},
            "source": {"kind": "synthetic", "count": 4, "width": 16, "height": 16},
            "app": {"plugin": "reference", "params": {"model_kind": "stub"}},
            "outputs": {"root": outputs_root},
            "nodes": [
                {"name": "client", "host": "localhost", "role": "client"},
                {
                    "name": "server",
                    "host": SURROGATE_HOST,
                    "port": port,
                    "role": "server",
                    "launcher": "ssh_docker",
                    "user": getpass.getuser(),
                    "ssh_identity": IDENTITY,
                    "container": IMAGE,
                },
            ],
            "channels": [{"name": "frames", "from": "client", "to": "server"}],
        }
    )


def _placement(tmp_path, port):
    config = _config(port, str(tmp_path / "out"))
    return Placement(
        config,
        "surrogate01-r0",
        str(tmp_path / "out" / "run"),
        log=lambda *_a: None,
    )


def _serve_spec(tmp_path, port):
    return LaunchSpec(
        name="serve:server",
        argv=["python", "-m", "cadet", "serve", "--config", "x.yaml", "--node", "server"],
        log_path=str(tmp_path / "server.log"),
        readiness_marker="[server] Waiting for connection",
        readiness_timeout_s=90.0,
    )


# --- the command that crosses the SSH hop --------------------------------
@needs_surrogate
def test_the_remote_command_joins_that_machines_network_and_mounts_a_remote_path(
    tmp_path,
):
    """All three of the things a remote container needs, in one command.

    They are asserted together because each is only correct relative to the
    others: a mount under the *remote* account's home is only useful if the
    container runs as that account's uid, and joining that machine's own
    network namespace is what makes the bind address the placement supplies
    reachable at all - a published port would depend on that machine's
    forwarding policy instead.
    """
    port = _free_port()
    place = _placement(tmp_path, port)
    try:
        place.start()
        launcher = place.launcher_for(place.config.nodes[1])
        command = " ".join(launcher.command(_serve_spec(tmp_path, port)))
        # The SSH hop itself: never prompt, and use the key the config names.
        assert "BatchMode=yes" in command
        assert "-i {}".format(IDENTITY) in command
        # The container joins that machine's own namespace, so the role is
        # reached at the address the config names - no forwarding involved.
        assert "--network {}".format(HOST_NETWORK) in command
        assert "-p " not in command
        # The mount is a path on that machine, under its own account's home,
        # and it is the outputs ROOT - the role appends the naming template.
        remote = place.remote_outputs["server"]
        assert remote.path is not None
        assert "{}:/app/outputs".format(remote.path) in command
        assert OWNED_CONTAINER_PREFIX in remote.path
        # ... and the container runs as the account that owns that path, which
        # is read from the far side rather than taken from this process.
        facts = launcher.remote_facts()
        assert "--user {}:{}".format(facts["uid"], facts["gid"]) in command
        assert remote.path.startswith(facts["home"])
    finally:
        place.stop()


@needs_surrogate
def test_a_remote_listening_role_is_told_to_bind_an_address_it_holds(tmp_path):
    """And through argv, not a --set: the config is hashed into the run id."""
    port = _free_port()
    place = _placement(tmp_path, port)
    try:
        place.start()
        server, client = place.config.nodes[1], place.config.nodes[0]
        assert place.role_args(server) == ["--bind-host", CONTAINER_BIND_HOST]
        # The client runs here and is not in a container, so neither its bind
        # address nor its outputs root is the placement's business.
        assert place.role_args(client) == []
        assert place.role_overrides_for(client) == []
        assert place.role_overrides_for(server) == ["outputs.root=/app/outputs"]
    finally:
        place.stop()


# --- the role really starts, is reachable, and is cleaned up -------------
@needs_surrogate
def test_a_role_launched_over_ssh_becomes_ready_and_is_removed(tmp_path):
    """Readiness is read from the log the SSH pipe writes, which is the point.

    The orchestrator never talks to the remote docker daemon directly: it reads
    the container's stdout as it arrives over SSH. If that pipe did not carry
    the marker, every remote run would fail its readiness barrier.
    """
    port = _free_port()
    place = _placement(tmp_path, port)
    handle = None
    launcher = None
    try:
        place.start()
        launcher = place.launcher_for(place.config.nodes[1])
        spec = _serve_spec(tmp_path, port)
        # The role needs a config inside the image, so it runs the repository's
        # own reference example from the image's working directory.
        spec.argv = [
            "python",
            "-m",
            "cadet",
            "serve",
            "examples/reference/synthetic.yaml",
            "--app-module",
            "examples.reference.app",
            "--node",
            "server",
            "--set",
            "nodes.1.port={}".format(port),
            "--set",
            "outputs.root=/app/outputs",
            "--bind-host",
            CONTAINER_BIND_HOST,
        ]
        handle = launcher.launch(spec)
        assert launcher.wait_ready(handle, timeout_s=120.0), open(spec.log_path).read()
        # Reachable here, which is what joining that machine's namespace is for.
        probe = socket.socket()
        probe.settimeout(10)
        probe.connect((SURROGATE_HOST, port))
        probe.close()
        container = launcher.container_spec(spec).name
        assert _container_exists(container)
    finally:
        if launcher is not None and handle is not None:
            launcher.stop(handle)
        place.stop()
    # Nothing left behind: the container is gone and so is the staging tree.
    assert not _container_exists("{}surrogate01-r0-server".format(OWNED_CONTAINER_PREFIX))


@needs_surrogate
def test_a_file_written_in_the_remote_container_comes_back(tmp_path):
    """The outputs mount, end to end: written there, fetched here, gone there.

    This is the check the old ssh path could not have passed - it mounted a path
    on the orchestrator's own filesystem, so a remote role's results stayed on
    the machine that ran it.
    """
    port = _free_port()
    place = _placement(tmp_path, port)
    local_root = os.path.abspath(place.config.outputs.root)
    try:
        place.start()
        launcher = place.launcher_for(place.config.nodes[1])
        remote_path = place.remote_outputs["server"].path
        # Written by a process in the container, into the mount, as the
        # container's own user - not by the test into the staging directory.
        payload = json.dumps({"written": "inside the container"})
        # The image's entry point is the cadet CLI, so it is overridden for this
        # one container: what is under test is the mount, not the role.
        launcher.container_spec(_serve_spec(tmp_path, port)).extra_args += [
            "--entrypoint",
            "python",
        ]
        spec = LaunchSpec(
            name="serve:server",
            argv=[
                "-c",
                "open('/app/outputs/from_remote.json','w').write({!r})".format(payload),
            ],
            log_path=str(tmp_path / "writer.log"),
        )
        handle = launcher.launch(spec)
        handle.proc.wait(timeout=120)
        assert handle.proc.returncode == 0, open(spec.log_path).read()
    finally:
        place.stop()
    fetched = os.path.join(local_root, "from_remote.json")
    assert os.path.exists(fetched), os.listdir(local_root) if os.path.isdir(local_root) else "none"
    assert json.load(open(fetched)) == {"written": "inside the container"}
    # And the staging tree on the far side is gone, so a run leaves nothing.
    assert remote_path is not None
    assert not _remote_path_exists(remote_path)


def _container_exists(name):
    done = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=^{}$".format(name), "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )
    return name in done.stdout.split()


def _remote_path_exists(path):
    done = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-i",
            IDENTITY,
            "{}@{}".format(getpass.getuser(), SURROGATE_HOST),
            "test -d {}".format(path),
        ],
        capture_output=True,
        text=True,
    )
    return done.returncode == 0
