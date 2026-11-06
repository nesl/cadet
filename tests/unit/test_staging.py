"""Tests for staging model artifacts onto the machine a container runs on.

What this covers:
    That the cache entry's name is content-addressed and so cannot go stale;
    that a second run on the same machine reuses the cache rather than copying
    again; that a cache entry of the right name but the wrong length is replaced
    rather than trusted; that the per-run directory presents the artifacts under
    the plain names the registry looks for and is removed at teardown while the
    cache is kept; and which roles are given the staged directory on the command
    line.

The machine here is this one, with no launcher, which is not a weaker case than
a remote one for any of the above: the whole point of
:class:`cadet.core.staging.TargetMachine` is that the staging logic does not
branch on where the node lives, so what is asserted here is the same code a
remote node runs. What it does not cover is the SSH transport underneath -
``tests/e2e/test_ssh_docker.py`` does that, and the live two-host runs in the
multi-device report do it for real.
"""

import os

import pytest

from cadet.core.staging import (
    CONTAINER_MODELS,
    MODELS_DIRNAME,
    RUNS_DIRNAME,
    ModelStaging,
    StagingError,
    TargetMachine,
    artifact_digest,
    artifacts_for,
    cache_entry_name,
    staged_model_args,
)


class _HereWithHome(TargetMachine):
    """This machine, but with its home redirected into a temporary directory."""

    def __init__(self, home, log=lambda *_a: None):
        super().__init__(launcher=None, log=log)
        self._home = str(home)


def _artifact(tmp_path, name, body=b"weights"):
    path = tmp_path / name
    path.write_bytes(body)
    return str(path)


# --- the cache key -------------------------------------------------------
def test_the_cache_entry_name_follows_the_content_not_the_name(tmp_path):
    """Caching on the name or the mtime is what would hand a run the wrong
    weights: a retrained artifact keeps its filename."""
    first = _artifact(tmp_path, "model.pt", b"one")
    name_one = cache_entry_name(first)
    with open(first, "wb") as fh:
        fh.write(b"two")
    name_two = cache_entry_name(first)
    assert name_one != name_two
    # Both still end in the plain name, which is what makes a staged directory
    # readable and what the registry looks for once it is linked in.
    assert name_one.endswith("-model.pt") and name_two.endswith("-model.pt")


def test_identical_content_under_different_names_shares_nothing_and_differs_only_by_name(
    tmp_path,
):
    a = _artifact(tmp_path, "a.pt", b"same")
    b = _artifact(tmp_path, "b.pt", b"same")
    assert artifact_digest(a) == artifact_digest(b)
    assert cache_entry_name(a) != cache_entry_name(b)


# --- staging, reuse, and repair -----------------------------------------
def test_staging_presents_plain_names_and_caches_the_content(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    art = _artifact(tmp_path, "yolo.pt", b"x" * 2048)
    machine = _HereWithHome(home)
    staging = ModelStaging(machine, "abc123-r0")
    staged = staging.stage([art])

    # The mount source holds the artifact under the name the registry asks for.
    assert staged == str(home / MODELS_DIRNAME / RUNS_DIRNAME / "abc123-r0")
    assert os.path.exists(os.path.join(staged, "yolo.pt"))
    with open(os.path.join(staged, "yolo.pt"), "rb") as fh:
        assert fh.read() == b"x" * 2048
    # ... and the cache holds it under its content-addressed name.
    cache = home / MODELS_DIRNAME / cache_entry_name(art)
    assert cache.exists()

    # Teardown removes the run's directory and keeps the cache, which is the
    # whole reason the cache exists.
    assert staging.remove() is True
    assert not os.path.exists(staged)
    assert cache.exists()


def test_a_second_run_on_the_same_machine_reuses_the_cache(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    art = _artifact(tmp_path, "yolo.pt", b"y" * 1024)
    machine = _HereWithHome(home)
    lines = []
    ModelStaging(machine, "run1-r0", log=lines.append).stage([art])
    second = ModelStaging(machine, "run2-r0", log=lines.append)
    second.stage([art])
    assert "shipped" in lines[0] or "staged" in lines[0]
    # The second run reports the artifact as cached, not shipped.
    assert "cached yolo.pt" in lines[1]
    assert "shipped" not in lines[1]


def test_a_short_cache_entry_is_replaced_rather_than_trusted(tmp_path):
    """The name is content-addressed, so a wrong-length file under it can only
    be an interrupted copy. Trusting it would feed a run a truncated model."""
    home = tmp_path / "home"
    home.mkdir()
    art = _artifact(tmp_path, "yolo.pt", b"z" * 4096)
    cache_dir = home / MODELS_DIRNAME
    cache_dir.mkdir(parents=True)
    entry = cache_dir / cache_entry_name(art)
    entry.write_bytes(b"z" * 10)  # as if a copy had been cut short

    staging = ModelStaging(_HereWithHome(home), "run-r0")
    staged = staging.stage([art])
    assert entry.stat().st_size == 4096
    assert os.path.getsize(os.path.join(staged, "yolo.pt")) == 4096


def test_nothing_is_created_when_there_is_nothing_to_stage(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    staging = ModelStaging(_HereWithHome(home), "run-r0")
    assert staging.stage([]) is None
    assert not (home / MODELS_DIRNAME).exists()
    assert staging.remove() is True


def test_a_missing_artifact_is_refused_with_its_path(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    staging = ModelStaging(_HereWithHome(home), "run-r0")
    with pytest.raises(StagingError, match="does not exist on this host"):
        staging.stage([str(tmp_path / "absent.pt")])


def test_an_unsafe_run_id_is_refused_before_any_path_is_built(tmp_path):
    """The run id ends up in the argument of a remove on another machine."""
    with pytest.raises(StagingError, match="not a safe path component"):
        ModelStaging(_HereWithHome(tmp_path), "../../etc")


# --- what the config asks for -------------------------------------------
def _braking_like_config(tmp_path):
    from cadet.config import build_config

    return build_config(
        {
            "experiment": {"name": "stage", "app": "reference", "repetitions": 1},
            "source": {"kind": "synthetic", "count": 2},
            "app": {"plugin": "reference", "params": {"model_kind": "stub"}},
            "models": {
                "model": "det",
                "registry": {
                    "det": {"per_device": {"cpu": "det.pt", "gpu": "det.engine"}},
                    "labels": {"per_device": {"cpu": "labels.names", "gpu": "labels.names"}},
                },
            },
            "nodes": [
                {"name": "client", "host": "localhost", "role": "client"},
                {
                    "name": "server",
                    "host": "127.0.0.1",
                    "port": 15010,
                    "role": "server",
                    "device": "cpu",
                },
            ],
        }
    )


def test_the_artifacts_staged_are_the_registry_entries_that_resolve(tmp_path):
    """Every registry entry for the hosting node's device, and only those: the
    gpu artifact of the same model is not what this run will load."""
    search = tmp_path / "models"
    search.mkdir()
    (search / "det.pt").write_bytes(b"cpu-weights")
    (search / "labels.names").write_bytes(b"a\nb\n")
    (search / "det.engine").write_bytes(b"gpu-weights")

    found = artifacts_for(_braking_like_config(tmp_path), [str(search)])
    assert sorted(os.path.basename(p) for p in found) == ["det.pt", "labels.names"]


def test_an_unresolved_entry_is_left_to_preflight(tmp_path):
    """Staging must not be what fails first: preflight names missing artifacts
    with a better message than a copy error would."""
    search = tmp_path / "models"
    search.mkdir()
    (search / "det.pt").write_bytes(b"cpu-weights")
    found = artifacts_for(_braking_like_config(tmp_path), [str(search)])
    assert [os.path.basename(p) for p in found] == ["det.pt"]


# --- who is told about it ------------------------------------------------
def test_the_staged_directory_is_passed_as_a_model_search_dir():
    assert staged_model_args() == ["--model-dir", CONTAINER_MODELS]
