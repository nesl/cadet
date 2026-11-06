"""Unit tests for the docker and ssh_docker launchers.

What this covers:
    The command each launcher builds — image, per-node GPU pinning, NET_ADMIN
    only where traffic is shaped, the outputs mount, container naming, the
    interpreter prefix being dropped in favour of the image entry point — and the
    ssh variant's options, ProxyJump and dry-run refusal.

Nothing is executed: only :meth:`command` is exercised, so these run anywhere
without docker or ssh.
"""

import pytest

from cadet.core.docker_launcher import (
    DEFAULT_IMAGE,
    ContainerSpec,
    DockerLauncher,
    SshDockerLauncher,
    container_spec_for,
)
from cadet.core.launcher import LaunchSpec


def spec(name="serve:server"):
    # The orchestrator builds argv for local execution, interpreter and all.
    return LaunchSpec(
        name=name,
        argv=["/usr/bin/python3", "-m", "cadet", "serve", "cfg.yaml", "--node", "server"],
    )


def test_interpreter_prefix_is_replaced_by_the_image_entry_point():
    cmd = DockerLauncher().command(spec())
    assert "-m" not in cmd
    assert "/usr/bin/python3" not in cmd
    assert cmd[-5:] == ["serve", "cfg.yaml", "--node", "server"][-4:] or cmd[-4:] == [
        "serve",
        "cfg.yaml",
        "--node",
        "server",
    ]


def test_default_image_when_the_node_names_none():
    assert DEFAULT_IMAGE in DockerLauncher().command(spec())


def test_a_pinned_gpu_is_device_zero_inside_the_container():
    """`--gpus device=N` filters through the device cgroup and renumbers the
    survivor to 0, so naming N again hides the only GPU the container has -
    measured as device_count() 0 on a two-GPU host."""
    launcher = DockerLauncher({"server": ContainerSpec(gpu_index=1)})
    cmd = " ".join(launcher.command(spec()))
    assert "--gpus device=1" in cmd
    assert "CUDA_VISIBLE_DEVICES=0" in cmd


def test_a_privileged_container_keeps_the_hosts_gpu_numbering():
    """Privileged bypasses the cgroup, so every GPU is visible and the host's
    index is the one that selects the intended device."""
    spec_ = ContainerSpec(gpu_index=1, extra_args=["--privileged"])
    cmd = " ".join(DockerLauncher({"server": spec_}).command(spec()))
    assert "CUDA_VISIBLE_DEVICES=1" in cmd


def test_no_gpu_flags_when_the_node_asks_for_none():
    cmd = " ".join(DockerLauncher({"server": ContainerSpec()}).command(spec()))
    assert "--gpus" not in cmd
    assert "CUDA_VISIBLE_DEVICES" not in cmd


def test_net_admin_only_for_nodes_that_shape_traffic():
    shaped = DockerLauncher({"server": ContainerSpec(needs_net_admin=True)})
    plain = DockerLauncher({"server": ContainerSpec(needs_net_admin=False)})
    assert "--cap-add" in shaped.command(spec())
    assert "--cap-add" not in plain.command(spec())


def test_outputs_are_bind_mounted_so_results_survive_the_container():
    launcher = DockerLauncher({"server": ContainerSpec(outputs_dir="/host/out")})
    assert "/host/out:/app/outputs" in " ".join(launcher.command(spec()))


def test_container_name_ties_the_container_to_its_run_and_node():
    class Node:
        name = "server"
        container = None
        gpu_index = None

    cspec = container_spec_for(Node(), run_id="abc123-r0")
    assert cspec.name == "cadet-abc123-r0-server"


def test_node_image_overrides_the_default():
    class Node:
        name = "server"
        container = "cadet:arm64"
        gpu_index = None

    assert container_spec_for(Node(), "r0").image == "cadet:arm64"


# --- ssh variant ----------------------------------------------------------
def ssh_launcher(**kwargs):
    params = {"host": "10.0.0.6", "user": "an-account", "dry_run": True}
    params.update(kwargs)
    return SshDockerLauncher(**params)


def test_ssh_command_fails_rather_than_prompting():
    cmd = " ".join(ssh_launcher().command(spec()))
    assert "BatchMode=yes" in cmd
    assert "ConnectTimeout=" in cmd


def test_ssh_target_and_proxy_jump():
    cmd = ssh_launcher(proxy_jump="an-account@gateway", identity="/k", port=2222).command(spec())
    assert "an-account@10.0.0.6" in cmd
    assert "-J" in cmd and "an-account@gateway" in cmd
    assert "-i" in cmd and "/k" in cmd
    assert "-p" in cmd and "2222" in cmd


def test_remote_docker_command_is_one_quoted_argument():
    """Otherwise the remote shell re-splits it and the flags are lost."""
    cmd = ssh_launcher().command(spec())
    remote = cmd[-1]
    assert remote.startswith("docker run")
    assert "serve" in remote


def test_dry_run_refuses_to_launch_but_still_builds_the_command():
    launcher = ssh_launcher()
    assert launcher.command(spec())  # inspectable
    with pytest.raises(RuntimeError) as err:
        launcher.launch(spec())
    assert "dry-run" in str(err.value)


def test_known_hosts_is_never_configured_away():
    """Host-key policy stays the operator's; the launcher must not relax it."""
    cmd = " ".join(ssh_launcher().command(spec()))
    assert "StrictHostKeyChecking" not in cmd
    assert "UserKnownHostsFile" not in cmd


# --- which image a node with no `container` runs ---------------------------
class _PlainNode:
    """A node that names no image, so the default applies to it."""

    name = "server"
    container = None
    gpu_index = None


def test_the_node_image_comes_from_the_environment_when_set(monkeypatch):
    """`cadet:dev` is a moving tag, so a result cannot be attributed by naming it.

    Pointing the variable at an immutable tag pins every placed node in one
    place, rather than a --set per node per command.
    """
    from cadet.core import docker_launcher as dl

    monkeypatch.setenv(dl.DEFAULT_IMAGE_ENV, "cadet:demo-2")
    assert dl.default_image() == "cadet:demo-2"
    assert container_spec_for(_PlainNode(), "0123456789-r0").image == "cadet:demo-2"


def test_a_node_that_names_its_image_is_not_overridden(monkeypatch):
    """The config is more specific than the machine, so it wins."""
    from cadet.core import docker_launcher as dl

    monkeypatch.setenv(dl.DEFAULT_IMAGE_ENV, "cadet:demo-2")
    class _Named(_PlainNode):
        container = "cadet-md:dev"

    assert container_spec_for(_Named(), "0123456789-r0").image == "cadet-md:dev"


def test_the_default_is_unchanged_with_no_variable(monkeypatch):
    from cadet.core import docker_launcher as dl

    monkeypatch.delenv(dl.DEFAULT_IMAGE_ENV, raising=False)
    assert dl.default_image() == dl.DEFAULT_IMAGE_TAG == "cadet:dev"


def test_the_variable_is_read_per_call_not_at_import(monkeypatch):
    """A wrapper sets it after this module is imported and must still be obeyed."""
    from cadet.core import docker_launcher as dl

    monkeypatch.delenv(dl.DEFAULT_IMAGE_ENV, raising=False)
    assert dl.default_image() == "cadet:dev"
    monkeypatch.setenv(dl.DEFAULT_IMAGE_ENV, "cadet:later")
    assert dl.default_image() == "cadet:later"
