"""Pin which interpreter a role runs under, and the facts that answer it.

What this is for:
    The CARLA client egg is imported by the role that opens the simulator, never
    by the orchestrator, so "which Python will import the egg" is a question
    about where that role is *placed*. Asking it about the orchestrator refused
    sound arrangements - an orchestrator on a host Python placing its roles in
    the image - and these tests pin the answer so it cannot drift back.

What it contains:
    - the launcher-kind answers for an interpreter path and its version;
    - that the role argv is built with the placement's interpreter;
    - that the egg check accepts a placement whose Python suits the egg even
      when this process's does not;
    - that the image's pinned Python agrees with what the Dockerfile installs,
      and that the image carries the docker client the wrapper needs.

How it fits into CADET:
    Unit tests over :mod:`cadet.core.launcher`, :mod:`cadet.orchestrator.runner`
    and :mod:`cadet.sim.versions`, plus two assertions about
    ``docker/Dockerfile`` and ``scripts/cadet`` as the wrapper's premises.
"""

import os
import subprocess
import sys

from cadet.core.launcher import (
    IMAGE_PYTHON,
    IMAGE_PYTHON_VERSION,
    role_executable,
    role_python_version,
)
from cadet.orchestrator.runner import Orchestrator
from cadet.sim import versions

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCKERFILE = os.path.join(REPO, "docker", "Dockerfile")
WRAPPER = os.path.join(REPO, "scripts", "cadet")

THIS_PYTHON = "{}.{}".format(*sys.version_info[:2])


def test_a_local_role_runs_this_interpreter():
    """A local role is a child of this process, so it is the same Python."""
    assert role_executable("local") == sys.executable
    assert role_python_version("local") == THIS_PYTHON


def test_a_placed_role_runs_the_image_interpreter():
    """Every other placement starts the role inside the image."""
    for kind in ("docker", "ssh_docker", "ssh_venv"):
        assert role_executable(kind) == IMAGE_PYTHON
        assert role_python_version(kind) == IMAGE_PYTHON_VERSION


def test_role_argv_takes_the_interpreter_it_is_given():
    """The caller names the interpreter, because the caller knows the node."""
    runner = Orchestrator(log=lambda *a: None)
    argv = runner._base_argv("client", "c.yaml", [], 0, [], executable="python3")
    assert argv[0] == "python3"
    # Unnamed is this process's own, which is what a local placement gets.
    assert runner._base_argv("client", "c.yaml", [], 0, [])[0] == sys.executable


class _Egg:
    host_path = None

    def __init__(self, name):
        self.name = name


class _Carla:
    image = None
    host = "127.0.0.1"
    port = 2000

    def __init__(self, egg):
        self.client_egg = egg


class _Source:
    kind = "carla"
    scenario = None

    def __init__(self, carla):
        self.carla = carla


class _Config:
    def __init__(self, egg_name):
        self.source = _Source(_Carla(_Egg(egg_name)))


def test_the_egg_check_follows_the_placement_not_this_process():
    """A py3.7 egg is fine for a role placed in the image, whatever we are.

    The egg is not on this machine, so the "not found" line is expected and is
    not what this test is about; the importability line is.
    """
    config = _Config("carla-0.9.10-py3.7-linux-x86_64.egg")

    placed = versions.check_versions(
        config, interpreter=role_python_version("docker")
    )
    assert not [p for p in placed if "will not import" in p]

    # The same config, asked about an interpreter that cannot load the egg.
    refused = versions.check_versions(config, interpreter="3.10")
    assert [p for p in refused if "will not import" in p]


def test_the_image_python_is_the_one_the_dockerfile_installs():
    """The pinned version is only true while the image agrees with it."""
    with open(DOCKERFILE) as fh:
        text = fh.read()
    assert "python{}".format(IMAGE_PYTHON_VERSION) in text
    # And the symlink that makes IMAGE_PYTHON name it.
    assert "/usr/local/bin/{}".format(IMAGE_PYTHON) in text


def test_the_image_carries_the_docker_client_the_wrapper_needs():
    """A client only, from a stage of its own, at a pinned version."""
    with open(DOCKERFILE) as fh:
        text = fh.read()
    assert "ARG DOCKER_CLI_VERSION=" in text
    assert "COPY --from=dockercli" in text
    assert "/usr/local/bin/docker" in text


def test_the_wrapper_is_runnable_and_valid_shell():
    """It is the documented way to run, so a syntax error in it is a bug."""
    assert os.access(WRAPPER, os.X_OK)
    assert subprocess.call(["bash", "-n", WRAPPER]) == 0
