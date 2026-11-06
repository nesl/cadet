"""Put a containerised role's model artifacts on the machine it runs on.

What this is for:
    A role is told where to look for model artifacts with ``--model-dir``, and
    the paths it is given are paths on the *orchestrator's* host. For a role in
    a container that is the wrong filesystem: the container sees neither the
    host's search directories nor, when the node lives on another machine, the
    host's disk at all. So an app that loads a model from the registry could not
    run in a container, locally or remotely, however correct its config was.
    This module stages the artifacts the registry names onto the machine the
    container runs on, and the placement mounts the result.

What it contains:
    - :class:`TargetMachine`: "the machine a role runs on" - this one, or
      another one reached through an :class:`~cadet.core.ssh.SshTarget`, which
      is either remote launcher. Commands and file copies go through it, so the
      staging logic is written once and does not branch on where the node lives
      or on how its role is started there.
    - :class:`ModelStaging`: a per-run directory of artifacts on that machine,
      filled from a per-machine content-addressed cache so a second run does
      not re-ship what is already there.
    - :data:`CONTAINER_MODELS`: where the staged directory is mounted.

How it fits into CADET:
    :class:`cadet.orchestrator.placement.Placement` stages before it starts a
    container and removes the per-run directory in its teardown, in the same
    place and the same order as the outputs tree it already manages, and through
    the same remote machinery (``run_remote``/``remote_facts``). The two are
    separate classes because they flow in opposite directions and have different
    lifetimes: outputs are written on the far machine and fetched back, and
    nothing of them is kept; artifacts are sent out and their *cache* is kept
    deliberately, because re-shipping a model on every run of a sweep would cost
    more than the run.

How to configure it:
    Nothing to configure. The artifacts staged are the ones
    ``models.registry`` names for the device of the node that hosts the model,
    which is the same set ``preflight`` already insists on finding.
"""

import hashlib
import os
import re
import shutil
import subprocess
from typing import Dict, Iterable, List, Optional, Tuple

# Where a staged model directory is mounted inside the container. On the
# search path because the placement also passes it as a --model-dir.
CONTAINER_MODELS = "/app/models"
# Directory under the target account's home holding the cache and the per-run
# directories. Under `.cache` deliberately, and not `~/cadet-models`, which was
# tried first and collided: that is exactly the name somebody keeps their own
# weights in and passes as a `--model-dir`, so the cache's digest-named entries
# landed in a directory the operator owns and curates. A cache belongs where
# caches go, and the name still says whose it is.
MODELS_DIRNAME = ".cache/cadet-models"
# Subdirectory of the above holding one directory per run.
RUNS_DIRNAME = "runs"
# How much of the digest goes into a cache entry's name. 16 hex characters of
# SHA-256 is 64 bits: collisions are not a practical concern for a handful of
# model files per machine, and a short name keeps the paths readable.
DIGEST_CHARS = 16
# Read size for digesting. Model artifacts run to hundreds of megabytes, so
# they are not read into memory whole.
_CHUNK = 1024 * 1024
# A run id becomes a path component and the argument of a remove, so it is
# checked rather than trusted.
_SAFE_RUN_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*\Z")
# Timeout for the directory operations. Generous: the copy of a large artifact
# over SSH happens under it.
OP_TIMEOUT_S = 1800.0


class StagingError(Exception):
    """A model artifact could not be placed on the machine that needs it."""


def artifact_digest(path: str) -> str:
    """Return ``(size, sha256-prefix)`` joined as a cache entry's name stem.

    Content, not metadata. Caching on mtime or on the artifact's name alone is
    what would eventually hand a run the *wrong weights*: a retrained file keeps
    its name, and a file copied or restored keeps neither its mtime nor any
    relation to what a previous run used. A name that contains a digest of the
    bytes cannot go stale - different content is simply a different path - and
    the size is checked again on the far side, which is what catches a copy that
    was interrupted and left a short file under the right name.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()[:DIGEST_CHARS]


def cache_entry_name(path: str) -> str:
    """Return the content-addressed file name for an artifact."""
    return "{}-{}".format(artifact_digest(path), os.path.basename(path))


class TargetMachine:
    """The machine a container runs on: this one, or one reached over SSH.

    Constructed with no launcher for a role on this machine and with a launcher
    that mixes in :class:`~cadet.core.ssh.SshTarget` - either remote one - for a
    role on another. Everything the staging needs is expressed as :meth:`run`
    and :meth:`send`, so there is one staging implementation rather than a local
    one and a remote one that drift apart.
    """

    def __init__(self, launcher=None, log=print):
        self.launcher = launcher
        self.log = log
        self._home: Optional[str] = None

    @property
    def is_remote(self) -> bool:
        return self.launcher is not None

    def describe(self) -> str:
        """A short name for this machine, for a log line."""
        if self.launcher is None:
            return "this machine"
        return "{}".format(self.launcher.host)

    def home(self) -> str:
        """The home directory of the account the container will run as."""
        if self._home is None:
            if self.launcher is None:
                self._home = os.path.expanduser("~")
            else:
                self._home = self.launcher.remote_facts()["home"]
        return self._home

    def run(self, argv: List[str], timeout_s: float = OP_TIMEOUT_S) -> Tuple[int, str]:
        """Run a command on this machine; return ``(returncode, output)``."""
        if self.launcher is not None:
            return self.launcher.run_remote(argv, timeout_s=timeout_s)
        try:
            done = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout_s
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return 255, "{}: {}".format(type(exc).__name__, exc)
        return done.returncode, (done.stdout or "") + (done.stderr or "")

    def size_of(self, path: str) -> Optional[int]:
        """Return the size of ``path`` on this machine, or None if it is absent."""
        if self.launcher is None:
            try:
                return os.path.getsize(path)
            except OSError:
                return None
        code, out = self.run(["stat", "-c", "%s", "--", path])
        if code != 0:
            return None
        try:
            return int(out.strip().splitlines()[0])
        except (ValueError, IndexError):
            return None

    def send(self, local_path: str, dest_path: str) -> None:
        """Copy a local file to ``dest_path`` on this machine, atomically.

        Written to a sibling ``.part`` and renamed, so an interrupted copy
        cannot leave a short file under the name a later run would trust. On
        this machine a hard link is tried first: the artifact is already here,
        and linking a few hundred megabytes costs nothing while copying it
        twice per machine would.
        """
        partial = dest_path + ".part"
        if self.launcher is None:
            try:
                os.link(local_path, dest_path)
                return
            except OSError:
                pass  # different filesystem, or it already exists
            try:
                shutil.copyfile(local_path, partial)
                os.replace(partial, dest_path)
            except OSError as exc:
                raise StagingError(
                    "cannot stage {} to {}: {}".format(local_path, dest_path, exc)
                )
            return
        command = "cat > {0}.part && mv {0}.part {0}".format(_quote(dest_path))
        argv = self.launcher.ssh_prefix() + [command]
        try:
            with open(local_path, "rb") as fh:
                done = subprocess.run(
                    argv, stdin=fh, capture_output=True, text=True, timeout=OP_TIMEOUT_S
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise StagingError(
                "cannot send {} to {}: {}".format(local_path, self.describe(), exc)
            )
        if done.returncode != 0:
            raise StagingError(
                "cannot send {} to {}: {}".format(
                    local_path, self.describe(), (done.stderr or done.stdout).strip()
                )
            )


def _quote(path: str) -> str:
    """Quote a path for the remote shell. Paths here are ours and plain."""
    import shlex

    return shlex.quote(path)


class ModelStaging:
    """One run's model artifacts on one machine, drawn from a per-machine cache.

    Two directories under the target account's home:

    ``.cache/cadet-models/<digest>-<name>``
        the cache. One entry per distinct *content*, kept between runs, which is
        the whole point of the cache check: a sweep of twenty repetitions, or a
        second experiment on the same machine, ships nothing.

    ``.cache/cadet-models/runs/<run_id>/<name>``
        what the container mounts, read-only. The artifacts appear here under
        the plain names the registry asked for, because that is what the
        registry looks for, and they are hard links to the cache entries so the
        per-run view costs no space. Removed at teardown; the cache is not.
    """

    def __init__(self, machine: TargetMachine, run_id: str, log=print):
        if not _SAFE_RUN_ID.match(run_id):
            raise StagingError(
                "run id {!r} is not a safe path component, and it would be part "
                "of a path created and removed on {}".format(run_id, machine.describe())
            )
        self.machine = machine
        self.run_id = run_id
        self.log = log
        self.path: Optional[str] = None
        self._cache_root: Optional[str] = None
        self._runs_root: Optional[str] = None

    def stage(self, artifacts: Iterable[str]) -> Optional[str]:
        """Put each local artifact path on the machine; return the mount source.

        Returns None when there is nothing to stage, so a run of an app with no
        models does not create directories or mounts it has no use for.
        """
        paths = [p for p in dict.fromkeys(artifacts) if p]
        if not paths:
            return None
        home = self.machine.home().rstrip("/")
        self._cache_root = "{}/{}".format(home, MODELS_DIRNAME)
        self._runs_root = "{}/{}".format(self._cache_root, RUNS_DIRNAME)
        self.path = "{}/{}".format(self._runs_root, self.run_id)
        code, out = self.machine.run(["mkdir", "-p", "--", self._cache_root, self.path])
        if code != 0:
            raise StagingError(
                "cannot create the model staging directory {} on {}: {}".format(
                    self.path, self.machine.describe(), out.strip() or "rc=" + str(code)
                )
            )
        shipped: List[str] = []
        reused: List[str] = []
        for local_path in paths:
            if not os.path.exists(local_path):
                raise StagingError(
                    "model artifact {} does not exist on this host, so it cannot "
                    "be staged".format(local_path)
                )
            entry = "{}/{}".format(self._cache_root, cache_entry_name(local_path))
            want = os.path.getsize(local_path)
            have = self.machine.size_of(entry)
            if have == want:
                reused.append(os.path.basename(local_path))
            else:
                if have is not None:
                    # Right content-addressed name, wrong length: a previous
                    # copy was cut short. Replace it rather than trust it.
                    self.machine.run(["rm", "-f", "--", entry])
                self.machine.send(local_path, entry)
                confirmed = self.machine.size_of(entry)
                if confirmed != want:
                    raise StagingError(
                        "staged {} to {} but it is {} bytes there against {} "
                        "here".format(
                            local_path, self.machine.describe(), confirmed, want
                        )
                    )
                shipped.append(os.path.basename(local_path))
            link = "{}/{}".format(self.path, os.path.basename(local_path))
            code, out = self.machine.run(["ln", "-f", "--", entry, link])
            if code != 0:
                # No hard link across devices, and some filesystems refuse them
                # outright; a copy within the machine is correct either way.
                code, out = self.machine.run(["cp", "-f", "--", entry, link])
                if code != 0:
                    raise StagingError(
                        "cannot place {} in {} on {}: {}".format(
                            os.path.basename(local_path),
                            self.path,
                            self.machine.describe(),
                            out.strip() or "rc=" + str(code),
                        )
                    )
        self.log(
            "[place] models on {}: {} staged{}{}".format(
                self.machine.describe(),
                len(paths),
                ", shipped " + ", ".join(sorted(shipped)) if shipped else "",
                ", cached " + ", ".join(sorted(reused)) if reused else "",
            )
        )
        return self.path

    def remove(self) -> bool:
        """Remove this run's directory, keeping the cache. Returns whether it is gone."""
        if self.path is None:
            return True
        code, out = self.machine.run(["rm", "-rf", "--", self.path])
        if code != 0:
            self.log(
                "[place] WARNING: model staging directory {} on {} was not "
                "removed: {}".format(
                    self.path, self.machine.describe(), out.strip() or "rc=" + str(code)
                )
            )
            return False
        self.path = None
        # Non-recursive, so it succeeds only when this was the last run's
        # directory. The cache root is left alone deliberately - it is what
        # makes the next run cheap - so a machine keeps `cadet-models` with the
        # artifacts in it and nothing else.
        if self._runs_root:
            self.machine.run(["rmdir", "--", self._runs_root])
        return True


def artifacts_for(config, model_dirs: List[str]) -> List[str]:
    """Return the local paths of the artifacts this config's roles will load.

    Everything in ``models.registry`` for the device of the node that hosts the
    model, which is the set ``preflight`` already requires to resolve. An entry
    that does not resolve is skipped rather than raised on: preflight reports
    missing artifacts with a better message, and staging should not be the thing
    that fails first.
    """
    from cadet.core.model_registry import ModelRegistry, device_for_models

    if not getattr(config.models, "registry", None):
        return []
    device = device_for_models(config)
    registry = ModelRegistry.from_config(config.models, device)
    found: List[str] = []
    for name in sorted(config.models.registry):
        try:
            resolved = registry.resolve(name, model_dirs, device)
        except KeyError:
            # No artifact for this device; another node's device may have one.
            continue
        if resolved.path:
            found.append(os.path.abspath(resolved.path))
    return found


def staged_model_args() -> List[str]:
    """Return the ``--model-dir`` argv a role in a container should be given."""
    return ["--model-dir", CONTAINER_MODELS]


def describe_staging(stagings: Dict[str, ModelStaging]) -> Dict[str, object]:
    """Return a snapshot of what was staged, for a run record."""
    return {
        node: {"machine": staging.machine.describe(), "dir": staging.path}
        for node, staging in sorted(stagings.items())
    }
