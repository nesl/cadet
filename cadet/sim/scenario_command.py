"""Build the scenario driver's command line from the config.

What this is for:
    The original drove the scenario through a three-line shell script,
    ``scenario_runner/run_scenario.sh``, whose last line is the whole of it::

        python3 scenario_runner.py --scenario $1 --reloadWorld --timeout 10000 --output

    CADET took that command from an environment variable instead, which meant
    the framework had no scenario command of its own - faithful or otherwise -
    and every run's fidelity depended on an operator reproducing those flags by
    hand. Dropping ``--reloadWorld`` leaves the previous run's actors in the
    world; naming no port sends the scenario to CARLA's default 2000 while the
    client talks to the configured one, so the client finds no ego and the run
    produces nothing. Both were observed.

What it contains:
    - :func:`scenario_runner_command`: the original's argv, from the config.
    - :data:`READY_MARKER`: the line the orchestrator waits for.

How it fits into CADET:
    An app's ``sim_hooks`` calls this when ``CADET_SCENARIO_RUNNER_CMD`` is
    unset, which makes the default the original's behaviour and the environment
    variable an override rather than the only route.

How to configure it:
    ``source.scenario.runner.path`` for the tree, and ``source.scenario.name``,
    ``definition`` and ``config_file`` for what to run. The simulator address
    comes from ``source.carla.host`` / ``source.carla.port``.

    The interpreter is the caller's to name. It defaults to the one running the
    orchestrator, which is what a scenario started on the host wants; a
    scenario started in its own container is given that container's instead,
    because the point of the container is that its dependency set is not this
    process's. See :data:`CONTAINER_PYTHON`.
"""

import os
import sys
from typing import List, Optional

from cadet.core.launcher import IMAGE_PYTHON

# The line ScenarioRunner prints once the scenario is live and the ego exists.
# The original waited for this with the scenario's name appended; the name is
# left off here so one marker serves every scenario.
READY_MARKER = "ScenarioManager: Running scenario"

# The original's flags, kept as named constants so a change to either is a
# change to a documented deviation rather than an edit to a string.
ORIGINAL_TIMEOUT_S = 10000
ORIGINAL_FLAGS = ("--reloadWorld", "--output")

# The interpreter inside a scenario container. The original's script said
# `python3` and so does the image, which installs ScenarioRunner's pinned
# dependencies for it; this process's own interpreter is not on that container's
# filesystem at all. It is the CADET image either way, so the name comes from
# cadet.core.launcher rather than being spelled a second time here.
CONTAINER_PYTHON = IMAGE_PYTHON


def scenario_runner_command(config, python: Optional[str] = None) -> Optional[List[str]]:
    """Return the scenario driver's argv, or ``None`` if the config names none.

    ``None`` means there is nothing to run - a config with no scenario, or no
    ScenarioRunner tree - and the caller then starts no sim hook, which is what
    a non-CARLA app wants.

    ``python`` is the interpreter to run the tree with. ``None`` means "the one
    this config implies": the container's when the scenario declares an image of
    its own, and this process's otherwise. An interpreter is only correct where
    its dependencies are installed, and a containerised scenario's are not
    here.

    ``--host`` and ``--port`` are the one addition to the original's flags. The
    original hard-coded ``localhost:2000`` by omitting them; here the simulator
    address is configurable, and a scenario that connects to a different
    simulator than the client spawns the ego somewhere the client cannot see it.
    """
    scenario = getattr(getattr(config, "source", None), "scenario", None)
    if scenario is None:
        return None
    name = getattr(scenario, "name", None)
    runner = getattr(scenario, "runner", None)
    tree = getattr(runner, "path", None)
    if not name or not tree:
        return None

    # The interpreter follows from the config when the caller names none: a
    # scenario with an image of its own runs in that container, where this
    # process's interpreter does not exist. Deciding it here rather than
    # rewriting argv[0] later keeps one answer to "what runs the tree".
    if python is None:
        python = CONTAINER_PYTHON if getattr(runner, "image", None) else sys.executable

    argv = [
        python,
        os.path.join(tree, "scenario_runner.py"),
        "--host",
        str(config.source.carla.host),
        "--port",
        str(config.source.carla.port),
        "--scenario",
        str(name),
    ]
    if getattr(scenario, "definition", None):
        argv += ["--additionalScenario", str(scenario.definition)]
    if getattr(scenario, "config_file", None):
        argv += ["--configFile", str(scenario.config_file)]
    argv += ["--timeout", str(ORIGINAL_TIMEOUT_S)]
    argv += list(ORIGINAL_FLAGS)
    return argv
