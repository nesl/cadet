"""Start the scenario driver in its own container, through the launcher interface.

What this is for:
    The orchestrator already knows how to start a sim hook, wait for its
    readiness marker, supervise it and stop it - it does all four through
    :class:`cadet.core.launcher.Launcher`. The scenario driver needs exactly
    that treatment but in a container of its own, so what is missing is a
    launcher, not a second code path through the run. When the scenario starts,
    and what happens when it does not become ready, stay where they were.

What it contains:
    - :class:`ScenarioLauncher`: a :class:`~cadet.core.launcher.LocalLauncher`
      that starts the process inside this run's scenario container.
    - :func:`scenario_launcher_for`: the launcher a config asks for, or ``None``
      when the scenario is to stay on the host.
    - :func:`scenario_mounts`: what that container needs mounted, and why.

How it fits into CADET:
    :meth:`cadet.orchestrator.runner.Orchestrator.run_one` picks the launcher
    for its hook specs with :func:`scenario_launcher_for` and otherwise treats
    the handle exactly as it treats a host hook's. Only ``launch`` differs from
    the local launcher: readiness is still the marker in the log, supervision is
    still the process's exit, and the container is removed by
    :meth:`cadet.sim.containers.SimSide.stop` with the rest of the simulator
    side. The interpreter is not this module's business either - the config says
    whether the scenario is containerised, so
    :func:`cadet.sim.scenario_command.scenario_runner_command` can pick the
    right one when it builds the argv.

How to configure it:
    ``source.scenario.runner.image`` selects the image and is what makes the
    scenario containerised at all; without it the hook runs on the host as
    before. ``source.scenario.runner.path`` is where the tree is mounted inside
    that container, and :data:`TREE_HOST_PATH_ENV` says where it is found on
    this machine - a machine's path, so it comes from the environment rather
    than from an experiment's config. :data:`API_HOST_PATH_ENV` is the same for
    the CARLA PythonAPI tree, which a scenario needs for its ``agents`` imports
    when the image carries only the client egg.
"""

import os
from typing import List, Optional, Tuple

from cadet.core.launcher import Handle, LaunchSpec, LocalLauncher, child_env
from cadet.sim.containers import (
    CONTAINER_CARLA_API,
    CONTAINER_SCENARIO_RUNNER,
    SimSideError,
)
from cadet.sim.versions import TREE_HOST_PATH_ENV, tree_host_path

# Where the repository is mounted in the scenario container. The image installs
# the app at this path, and a scenario's ``--additionalScenario`` and
# ``--configFile`` are written relative to the repository root, so mounting the
# run's own checkout here is what makes those two resolve to the files this run
# is testing rather than to whatever the image happened to be built from.
CONTAINER_REPO = "/app"

# TREE_HOST_PATH_ENV and tree_host_path live in cadet.sim.versions, beside
# egg_directory: "where is this piece on this machine" is one question and the
# version check asks it too, of the same tree, to take its digest.

# Where the CARLA PythonAPI tree lives on this machine, for the same reason and
# in the same form. ScenarioRunner needs more of the client distribution than
# the egg: the egg provides the `carla` module, but the tree also ships
# `agents`, which ScenarioRunner imports for its own behaviours, and that is
# not in the egg. Unset means the image carries the whole distribution itself.
API_HOST_PATH_ENV = "CADET_CARLA_PYTHONAPI_DIR"

# The directory inside that tree that goes on the path. The CARLA client
# distribution is laid out PythonAPI/carla/{agents,dist}, so the importable
# parent of `agents` is the `carla` subdirectory and not the tree's own root.
API_PACKAGE_SUBDIR = "carla"


def api_host_path():
    """Return the host directory holding the CARLA PythonAPI tree, if named."""
    return os.environ.get(API_HOST_PATH_ENV) or None


def runner_pin(config):
    """Return ``source.scenario.runner``, or None for a config without one."""
    return getattr(getattr(getattr(config, "source", None), "scenario", None), "runner", None)


def scenario_mounts(config, repo_root: str) -> List[Tuple[str, str]]:
    """Return the ``(host path, container path)`` pairs a scenario container needs.

    Two, each for its own reason. The ScenarioRunner tree is mounted rather than
    baked because the version pin is a digest of the tree *as the run sees it*,
    so baking it would check one copy and run another. The repository is mounted
    because the scenario's definition and config file are files in it, named
    relative to its root.

    The CARLA client egg is deliberately **not** here: the image bakes it at a
    fixed path of its own, which is the one case where a container path is the
    image's convention rather than a setting (see
    :data:`cadet.sim.versions.CONTAINER_CARLA_EGG_DIR`).
    """
    host_tree = tree_host_path()
    if not host_tree:
        raise SimSideError(
            "source.scenario.runner.image asks for the scenario to run in a "
            "container, and the ScenarioRunner tree has to be mounted into it, "
            "but {} is not set. It is a path on this machine, so it is an "
            "environment variable rather than a config field: the same config "
            "has to run on a host that keeps the tree elsewhere.".format(
                TREE_HOST_PATH_ENV
            )
        )
    if not os.path.isdir(host_tree):
        raise SimSideError(
            "{}={!r} is not a directory, so there is no ScenarioRunner tree to "
            "mount into the scenario container.".format(TREE_HOST_PATH_ENV, host_tree)
        )
    container_tree = getattr(runner_pin(config), "path", None) or CONTAINER_SCENARIO_RUNNER
    mounts = [(host_tree, container_tree), (repo_root, CONTAINER_REPO)]
    host_api = api_host_path()
    if host_api:
        if not os.path.isdir(host_api):
            raise SimSideError(
                "{}={!r} is not a directory, so there is no CARLA PythonAPI "
                "tree to mount into the scenario container.".format(
                    API_HOST_PATH_ENV, host_api
                )
            )
        mounts.append((host_api, CONTAINER_CARLA_API))
    return mounts


def scenario_pythonpath(config) -> str:
    """Return the PYTHONPATH the scenario needs inside its container.

    The repository, because a scenario's definition imports from it, and the
    CARLA client distribution when one is mounted, because ScenarioRunner
    imports `agents` from it and the baked egg carries only the `carla` module.
    Appended to rather than replacing what the image sets, since the image puts
    the app on the path itself.
    """
    parts = [CONTAINER_REPO]
    if api_host_path():
        parts.insert(0, os.path.join(CONTAINER_CARLA_API, API_PACKAGE_SUBDIR))
    return ":".join(parts)


class ScenarioLauncher(LocalLauncher):
    """Runs one sim hook inside this run's scenario container.

    Subclasses the local launcher rather than reimplementing the interface:
    ``wait_ready``, ``poll`` and ``stop`` are already right, because the
    handle's process is the ``docker exec`` whose lifetime is the scenario's and
    whose output is the log the readiness barrier reads.
    """

    def __init__(
        self,
        sim_side,
        image: str,
        mounts,
        pythonpath: str = CONTAINER_REPO,
        workdir: str = CONTAINER_REPO,
    ):
        self.sim_side = sim_side
        self.image = image
        self.mounts = list(mounts)
        self.pythonpath = pythonpath
        self.workdir = workdir

    def launch(self, spec: LaunchSpec) -> Handle:
        """Start ``spec`` in the scenario container and return its handle.

        The spec's environment goes through :func:`child_env` so a containerised
        hook gets the same unbuffering a host hook does: the readiness barrier
        reads the log, and a marker sitting in a buffer reads as a startup
        timeout rather than as a slow scenario.
        """
        # The spec's own environment wins: it is the app's, and the app knows
        # what its hook needs. PYTHONPATH is the container's business, so it is
        # underneath rather than on top.
        env = {"PYTHONPATH": self.pythonpath}
        env.update(child_env(spec))
        proc = self.sim_side.start_scenario(
            argv=spec.argv,
            log_path=spec.log_path,
            image=self.image,
            mounts=self.mounts,
            env=env,
            workdir=self.workdir,
        )
        return Handle(spec, proc, None)


def scenario_launcher_for(config, sim_side, repo_root: str) -> Optional[ScenarioLauncher]:
    """Return the launcher for this config's scenario hooks, or ``None``.

    ``None`` means "leave the hook on the host", which is what a config without
    ``source.scenario.runner.image`` asks for, and what every run did before the
    scenario had a container of its own.
    """
    if sim_side is None:
        return None
    image = getattr(runner_pin(config), "image", None)
    if not image:
        return None
    return ScenarioLauncher(
        sim_side=sim_side,
        image=image,
        mounts=scenario_mounts(config, repo_root),
        pythonpath=scenario_pythonpath(config),
    )
