"""Declare the simulator-side versions together, and refuse a mismatch.

A CARLA server image, a client egg and a ScenarioRunner tree have to agree,
and nothing checked that they did. This module declares the checks together
and runs them before a run starts. It provides :func:`source_digest` (a
16-character hex digest of a source tree by file contents), :func:`egg_python_version`
and :func:`egg_carla_version` (what an egg's filename says), :func:`local_image_present`
(whether a declared image exists locally), and :func:`check_versions` (all
checks, returning readable mismatches). Called from ``preflight`` on both run
paths; a mismatch is a refusal, not a warning, because unlike an
over-subscribed link there is no experiment for which running mismatched
simulator pieces is the intent. The ScenarioRunner pin is a content digest
rather than a commit because the same files live in different repositories with
different histories; their commit hashes differ by design and will never agree,
but the file contents do - and content is what the check needs. Configured via
``source.carla.image``, ``source.carla.client_egg.name``,
``source.scenario.runner.path`` and ``source.scenario.runner.source_digest``.

The egg is looked for in ``source.carla.client_egg.host_path`` when set,
and otherwise at :data:`CONTAINER_CARLA_EGG_DIR`, the fixed path the image
bakes it at. One is a host directory and the other a container directory, and
only one applies to a given run - which is why they are two separate fields
rather than one meaning whichever the reader assumed.

The ScenarioRunner tree has the same two readings and so the same rule: the
digest is taken of the tree at :func:`tree_directory`, which is the host
directory named by :data:`TREE_HOST_PATH_ENV` when a run mounts the tree into a
scenario container, and ``source.scenario.runner.path`` otherwise.
"""

import hashlib
import os
import re
import subprocess
import sys
from typing import List, Optional

# Files whose contents define a ScenarioRunner tree. Documentation images and
# sample logs are deliberately excluded: they do not change behaviour, and
# including them would make the digest depend on whether a run had written
# sample output into the tree.
DIGEST_SUFFIXES = (".py", ".json", ".xml", ".sh")
# Length of the hex digest kept. Long enough that a collision is not a practical
# concern for a handful of trees, short enough to read and paste into a config.
DIGEST_CHARS = 16

# An egg is named e.g. carla-0.9.10-py3.7-linux-x86_64.egg.
_EGG_RE = re.compile(r"^carla-(?P<carla>[0-9.]+)-py(?P<py>[0-9.]+)-")

# Where the image bakes the CARLA client egg. A fixed convention of the image
# rather than a config field: it is the same in every run, and making it
# configurable is what let a host path and a container path be confused for one
# another.
CONTAINER_CARLA_EGG_DIR = "/opt/carla/PythonAPI"

# Where the ScenarioRunner tree lives on *this machine*. A host path is a
# property of the machine and not of an experiment - the same config has to run
# on a host that keeps the tree elsewhere - so it is an environment variable
# rather than a config field, for the same reason as CADET_STATE_ROOT.
TREE_HOST_PATH_ENV = "CADET_SCENARIO_RUNNER_DIR"


def tree_host_path() -> Optional[str]:
    """Return the host directory holding the ScenarioRunner tree, if named."""
    return os.environ.get(TREE_HOST_PATH_ENV) or None


def tree_directory(pin) -> Optional[str]:
    """Return the directory to read the ScenarioRunner tree from.

    The host path wins when it is set, because that is the tree the run will
    actually mount, and the digest has to be of the tree the run uses. The pin's
    own ``path`` is the fallback and is what a run whose orchestrator and
    scenario share a filesystem has always used.

    These are not interchangeable: with a containerised scenario, ``path`` is a
    path inside that container and does not exist out here at all. Reading it
    anyway is how the pinned digest came to be reported as uncheckable on a
    machine where the tree was present all along.
    """
    return tree_host_path() or getattr(pin, "path", None)


def egg_directory(client_egg) -> str:
    """Return the directory to look for the client egg in.

    The egg's own ``host_path`` wins when set, because that is the directory the
    run will actually mount; otherwise it is the image's fixed path, which is
    the image's convention and not a setting. Exactly one of the two applies at
    a time, which is why they are two fields of one block rather than one field
    meaning whichever the reader assumed.
    """
    host_path = getattr(client_egg, "host_path", None)
    return host_path or CONTAINER_CARLA_EGG_DIR


def source_digest(tree_path: str, suffixes=DIGEST_SUFFIXES) -> Optional[str]:
    """Return a content digest of ``tree_path``, or None if it does not exist.

    Each file contributes its path *and* its contents, so moving a file changes
    the digest as much as editing it. The walk is sorted, so the digest does not
    depend on directory order.
    """
    if not os.path.isdir(tree_path):
        return None
    entries = []
    for root, dirs, files in os.walk(tree_path):
        dirs[:] = sorted(d for d in dirs if d not in (".git", "__pycache__"))
        for name in sorted(files):
            if not name.endswith(suffixes):
                continue
            full = os.path.join(root, name)
            rel = os.path.relpath(full, tree_path)
            try:
                with open(full, "rb") as handle:
                    body = handle.read()
            except OSError:
                continue  # unreadable files cannot be part of a reproducible pin
            entries.append((rel, hashlib.sha256(body).hexdigest()))
    outer = hashlib.sha256()
    for rel, digest in entries:
        outer.update(rel.encode("utf-8"))
        outer.update(b" ")
        outer.update(digest.encode("ascii"))
        outer.update(b"\n")
    return outer.hexdigest()[:DIGEST_CHARS]


def egg_python_version(egg_name: str) -> Optional[str]:
    """Return the Python version an egg's filename declares, e.g. "3.7"."""
    match = _EGG_RE.match(os.path.basename(egg_name))
    return match.group("py") if match else None


def egg_carla_version(egg_name: str) -> Optional[str]:
    """Return the CARLA version an egg's filename declares, e.g. "0.9.10"."""
    match = _EGG_RE.match(os.path.basename(egg_name))
    return match.group("carla") if match else None


def image_version(image: str) -> Optional[str]:
    """Return the version an image reference's tag declares, e.g. "0.9.10.1"."""
    if ":" not in image:
        return None
    tag = image.rsplit(":", 1)[1]
    return tag if re.match(r"^[0-9][0-9.]*$", tag) else None


def _api_version(version: Optional[str]) -> Optional[str]:
    """Return the CARLA client-API version, i.e. the first three components.

    CARLA numbers its releases ``0.9.<N>``, so the release is identified by
    three components and not two: 0.9.10 and 0.9.12 are different releases with
    different client APIs. A fourth component is a patch of the same release -
    the 0.9.10.1 server ships the 0.9.10 client - so it is dropped.
    """
    if not version:
        return None
    return ".".join(version.split(".")[:3])


def _py_version(version: Optional[str]) -> Optional[str]:
    """Return "3.7" from "3.7"/"3.7.9"; Python compatibility is major.minor."""
    if not version:
        return None
    parts = version.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else version


def local_image_present(image: str, docker: str = "docker") -> Optional[bool]:
    """True/False if the image is/is not local; None if docker cannot be asked."""
    try:
        result = subprocess.run(
            [docker, "image", "inspect", image],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None  # no docker client here, e.g. inside a role container
    return result.returncode == 0


def check_versions(config, interpreter=None) -> List[str]:
    """Return one readable line per simulator-side version mismatch.

    Empty means everything the config declared agrees with what is on disk.
    Each check is skipped, not failed, when the config does not declare the
    thing it would check - a config that declares no image and no scenario
    runner is making no claim to verify.

    ``interpreter`` is the "major.minor" Python that will *import the client
    egg*: the role that opens the simulator, not this process. The caller knows
    where that role is placed and so has to say. The default is this process's
    own version, which is the role's only when the role is a local child of it.
    See :func:`cadet.core.launcher.role_python_version`.
    """
    source = getattr(config, "source", None)
    if source is None or getattr(source, "kind", None) != "carla":
        return []
    carla = source.carla
    pin = source.scenario.runner if source.scenario is not None else None

    problems: List[str] = []
    # A fallback only: this process's version is the role's when the role is a
    # local child of it, and a different interpreter otherwise.
    interpreter = interpreter or "{}.{}".format(*sys.version_info[:2])

    # 1) The CARLA server image must be here, since the orchestrator starts it.
    #    No image means no launch, so there is nothing to check.
    present = local_image_present(carla.image) if carla.image else None
    if present is False:
        problems.append(
            "source.carla.image {!r} is not present locally; pull it or correct "
            "the declared version".format(carla.image)
        )

    # 2) The client egg must exist where the config says, and its name must
    #    agree with the server image on CARLA's release. A fourth component is
    #    a patch of the same release (the 0.9.10.1 server ships the 0.9.10
    #    client), so it is dropped and the first three are compared - 0.9.10 and
    #    0.9.12 are different releases and must not be called a match.
    egg = carla.client_egg.name if carla.client_egg else None
    if egg:
        egg_path = (
            egg
            if os.path.isabs(egg)
            else os.path.join(egg_directory(carla.client_egg), egg)
        )
        if not os.path.exists(egg_path):
            problems.append(
                "source.carla.client_egg.name not found at {}".format(egg_path)
            )
    client_carla = _api_version(egg_carla_version(egg)) if egg else None
    server_carla = _api_version(image_version(carla.image)) if carla.image else None
    if client_carla and server_carla and client_carla != server_carla:
        problems.append(
            "CARLA client/server mismatch: egg {!r} is for {} but "
            "source.carla.image {!r} is {}".format(
                os.path.basename(egg), client_carla, carla.image, server_carla
            )
        )

    # 3) The egg has to be importable by the interpreter that will import it.
    #    The 0.9.10 egg is py3.7: it loads under 3.8 but not 3.10, which is how
    #    an image rebuild once left no CARLA client at all.
    egg_py = egg_python_version(egg) if egg else None
    if egg_py and interpreter:
        if _py_version(interpreter) != _py_version(egg_py):
            if tuple(int(p) for p in interpreter.split(".")[:2]) > (3, 8):
                problems.append(
                    "source.carla.client_egg.name {!r} is a py{} egg and this "
                    "interpreter is "
                    "py{}; the CARLA client will not import".format(
                        os.path.basename(egg), egg_py, interpreter
                    )
                )

    # 4) ScenarioRunner, pinned by content (see the module docstring). Read
    #    where the tree is on this machine, which is not where it is mounted in
    #    the scenario container; see tree_directory.
    #
    #    `pin` is None when the config declares no scenario at all, which is a
    #    config that drives an already-running simulator and runs no scenario of
    #    its own. It declares no pin, so there is nothing to check - the same
    #    "no claim, no check" rule as every other step here.
    if pin is not None and pin.source_digest:
        tree = tree_directory(pin)
        actual = source_digest(tree)
        if actual is None:
            problems.append(
                "the ScenarioRunner tree was not found at {!r}, so its pinned "
                "digest cannot be checked. That is {}, or "
                "source.scenario.runner.path when it is unset.".format(
                    tree, TREE_HOST_PATH_ENV
                )
            )
        elif actual != pin.source_digest:
            problems.append(
                "ScenarioRunner content mismatch at {}: declared digest {} but "
                "the files there digest to {}. The pin is over file contents, "
                "not a commit id, so this means the files really differ.".format(
                    pin.path, pin.source_digest, actual
                )
            )
    return problems
