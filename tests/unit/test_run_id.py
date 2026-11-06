"""Tests for run_id derivation and per-repetition seeds.

Covers: run_id stability, the ``-r<rep>`` suffix, that ``outputs`` is excluded
from the hash while real fields are not, and the seed-per-repetition rule.
"""

from conftest import EXAMPLE_CONFIGS

from cadet.config import build_config, config_hash, load_config_data, run_id, seed_for_rep


def _braking_cfg():
    return build_config(load_config_data(str(EXAMPLE_CONFIGS[0])))


def test_run_id_is_stable_and_suffixed():
    cfg = _braking_cfg()
    assert run_id(cfg, 0) == run_id(cfg, 0)
    assert run_id(cfg, 3).endswith("-r3")
    # hash part is shared across repetitions; only the suffix changes.
    assert run_id(cfg, 0).rsplit("-r", 1)[0] == run_id(cfg, 1).rsplit("-r", 1)[0]


def test_outputs_excluded_from_hash():
    data = load_config_data(str(EXAMPLE_CONFIGS[0]))
    base = build_config(data)
    data2 = load_config_data(str(EXAMPLE_CONFIGS[0]))
    data2["outputs"]["root"] = "somewhere-else"
    changed_outputs = build_config(data2)
    assert config_hash(base) == config_hash(changed_outputs)


def test_real_field_change_changes_hash():
    data = load_config_data(str(EXAMPLE_CONFIGS[0]))
    base = build_config(data)
    data2 = load_config_data(str(EXAMPLE_CONFIGS[0]))
    data2["source"]["carla"]["port"] = 2002
    assert config_hash(base) != config_hash(data2 and build_config(data2))


def test_seed_per_repetition():
    cfg = _braking_cfg()  # seed: 0 in the example
    assert seed_for_rep(cfg, 0) == 0
    assert seed_for_rep(cfg, 5) == 5


def test_seed_none_stays_none():
    data = load_config_data(str(EXAMPLE_CONFIGS[0]))
    data["experiment"]["seed"] = None
    cfg = build_config(data)
    assert seed_for_rep(cfg, 3) is None


# --- which code produced a run -------------------------------------------
def test_the_run_record_names_the_commit_that_produced_it(tmp_path, monkeypatch):
    """A run id is reproducible against the version that wrote it, not a later one.

    The hash covers the resolved config *as that version's schema read it*, so
    adding a field or changing a default moves every id uniformly. Measured on
    this repository: 53 of 74 recorded run directories no longer recompute to
    their own id, and each recomputes to the id current code gives that same
    cell. Without the commit in the record a reader cannot tell a drifted id
    from a mismatched one.
    """
    import json

    from cadet.orchestrator import placement as placement_mod
    from cadet.orchestrator.runner import Orchestrator

    monkeypatch.setattr(placement_mod, "repo_revision", lambda *_a: ("a" * 40, False))
    orch = Orchestrator.__new__(Orchestrator)
    orch._write_marker(str(tmp_path), "0123456789-r0", True, "ok")

    record = json.load(open(str(tmp_path / "run.json")))
    assert record["cadet"] == {"commit": "a" * 40, "dirty": False}


def test_a_dirty_tree_is_recorded_as_such(tmp_path, monkeypatch):
    """A commit plus uncommitted edits does not identify the code that ran."""
    import json

    from cadet.orchestrator import placement as placement_mod
    from cadet.orchestrator.runner import Orchestrator

    monkeypatch.setattr(placement_mod, "repo_revision", lambda *_a: ("b" * 40, True))
    orch = Orchestrator.__new__(Orchestrator)
    orch._write_marker(str(tmp_path), "0123456789-r0", True, "ok")
    assert json.load(open(str(tmp_path / "run.json")))["cadet"]["dirty"] is True


def test_a_tree_that_is_not_a_checkout_records_no_commit(tmp_path, monkeypatch):
    """An installed copy need not be a checkout; absent beats a guessed value."""
    import json

    from cadet.orchestrator import placement as placement_mod
    from cadet.orchestrator.runner import Orchestrator

    monkeypatch.setattr(placement_mod, "repo_revision", lambda *_a: (None, False))
    orch = Orchestrator.__new__(Orchestrator)
    orch._write_marker(str(tmp_path), "0123456789-r0", True, "ok")
    assert "cadet" not in json.load(open(str(tmp_path / "run.json")))


def test_the_record_names_the_image_each_containerised_node_ran(tmp_path, monkeypatch):
    """A tag is not provenance: it is rebuilt, and the digest tells runs apart.

    The repository's own commit says nothing about the code inside a container,
    so a run that placed a role in one records what that image was: the tag the
    node asked for, the digest the daemon resolved it to, and the commit the
    image records.
    """
    import json

    from cadet.orchestrator import placement as placement_mod
    from cadet.orchestrator.runner import Orchestrator

    monkeypatch.setattr(placement_mod, "repo_revision", lambda *_a: ("c" * 40, False))
    orch = Orchestrator.__new__(Orchestrator)
    orch._write_marker(
        str(tmp_path),
        "0123456789-r0",
        True,
        "ok",
        images={
            "server": {"image": "cadet:dev", "id": "sha256:" + "d" * 64, "revision": "e" * 40},
            "client": {"image": "cadet:dev", "id": "sha256:" + "d" * 64},
        },
    )

    record = json.load(open(str(tmp_path / "run.json")))
    assert record["images"]["server"]["id"] == "sha256:" + "d" * 64
    assert record["images"]["server"]["revision"] == "e" * 40
    # The node whose image records no commit carries no revision key, rather
    # than one holding a guess.
    assert "revision" not in record["images"]["client"]


def test_a_run_that_placed_no_container_records_no_images(tmp_path, monkeypatch):
    """Absent, not empty: an empty map reads as "asked and found none"."""
    import json

    from cadet.orchestrator import placement as placement_mod
    from cadet.orchestrator.runner import Orchestrator

    monkeypatch.setattr(placement_mod, "repo_revision", lambda *_a: ("f" * 40, False))
    orch = Orchestrator.__new__(Orchestrator)
    orch._write_marker(str(tmp_path), "0123456789-r0", True, "ok", images={})
    assert "images" not in json.load(open(str(tmp_path / "run.json")))


def test_an_orchestrator_in_the_image_records_the_commit_the_image_carries(
    tmp_path, monkeypatch
):
    """There is no git in the image, and no provenance is the worse answer.

    The image records its commit in an OCI label, which is readable only by
    asking the daemon, so a process inside the container cannot see it. The
    same value is in the environment for this reason, and the record says the
    answer came from the image rather than from a checkout.
    """
    import json

    from cadet.orchestrator import placement as placement_mod
    from cadet.orchestrator.runner import Orchestrator

    monkeypatch.setattr(placement_mod, "repo_revision", lambda *_a: (None, False))
    monkeypatch.setenv("CADET_IMAGE_REVISION", "9" * 40)
    orch = Orchestrator.__new__(Orchestrator)
    orch._write_marker(str(tmp_path), "0123456789-r0", True, "ok")

    record = json.load(open(str(tmp_path / "run.json")))
    assert record["cadet"] == {"commit": "9" * 40, "dirty": False, "from": "image"}


def test_an_unbuilt_revision_is_not_recorded_as_a_commit(tmp_path, monkeypatch):
    """The build argument defaults to `unknown`, which is not a commit."""
    import json

    from cadet.orchestrator import placement as placement_mod
    from cadet.orchestrator.runner import Orchestrator

    monkeypatch.setattr(placement_mod, "repo_revision", lambda *_a: (None, False))
    monkeypatch.setenv("CADET_IMAGE_REVISION", "unknown")
    orch = Orchestrator.__new__(Orchestrator)
    orch._write_marker(str(tmp_path), "0123456789-r0", True, "ok")
    assert "cadet" not in json.load(open(str(tmp_path / "run.json")))
