"""Tests for the simulator-side version pin.

The property that matters is *why* the ScenarioRunner pin is a content digest:
the same files live in two repositories with different commit ids, so a hash pin
could never hold for both. These tests pin the digest's behaviour - it notices
an edit, a rename and a deletion, and it ignores things that cannot change
behaviour - plus the client/server agreement checks.
"""

import os

import pytest

from cadet.sim import versions


def _tree(root, files):
    for rel, body in files.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            handle.write(body)
    return str(root)


# --- the content digest -------------------------------------------------
def test_digest_is_stable_for_the_same_contents(tmp_path):
    a = _tree(tmp_path / "a", {"srunner/x.py": "print(1)\n", "d/s.json": "{}\n"})
    b = _tree(tmp_path / "b", {"srunner/x.py": "print(1)\n", "d/s.json": "{}\n"})
    assert versions.source_digest(a) == versions.source_digest(b)


def test_digest_notices_an_edit(tmp_path):
    a = _tree(tmp_path / "a", {"x.py": "print(1)\n"})
    before = versions.source_digest(a)
    with open(os.path.join(a, "x.py"), "w") as handle:
        handle.write("print(2)\n")
    assert versions.source_digest(a) != before


def test_digest_notices_a_rename(tmp_path):
    # The path is hashed alongside the contents, so moving a file is a change.
    a = _tree(tmp_path / "a", {"x.py": "print(1)\n"})
    before = versions.source_digest(a)
    os.rename(os.path.join(a, "x.py"), os.path.join(a, "y.py"))
    assert versions.source_digest(a) != before


def test_digest_notices_a_deletion(tmp_path):
    a = _tree(tmp_path / "a", {"x.py": "print(1)\n", "y.py": "print(2)\n"})
    before = versions.source_digest(a)
    os.remove(os.path.join(a, "y.py"))
    assert versions.source_digest(a) != before


def test_digest_ignores_what_cannot_change_behaviour(tmp_path):
    # Documentation images and sample logs are excluded on purpose: including
    # them would make the pin depend on whether a run had written output into
    # the tree.
    a = _tree(tmp_path / "a", {"x.py": "print(1)\n"})
    before = versions.source_digest(a)
    _tree(tmp_path / "a", {"Docs/img/diagram.png": "not really a png", "m/data/run.log": "noise"})
    assert versions.source_digest(a) == before


def test_digest_ignores_directory_order(tmp_path):
    a = _tree(tmp_path / "a", {"b/x.py": "1\n", "a/y.py": "2\n"})
    b = _tree(tmp_path / "b", {"a/y.py": "2\n", "b/x.py": "1\n"})
    assert versions.source_digest(a) == versions.source_digest(b)


def test_digest_is_none_for_a_missing_tree(tmp_path):
    assert versions.source_digest(str(tmp_path / "nope")) is None


# --- what an egg's and an image's name declare ---------------------------
def test_egg_name_parsing():
    egg = "carla-0.9.10-py3.7-linux-x86_64.egg"
    assert versions.egg_carla_version(egg) == "0.9.10"
    assert versions.egg_python_version(egg) == "3.7"


def test_egg_name_parsing_tolerates_a_full_path():
    egg = "/opt/carla/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg"
    assert versions.egg_carla_version(egg) == "0.9.10"


def test_egg_name_parsing_returns_none_for_something_else():
    assert versions.egg_carla_version("not-an-egg.zip") is None


def test_image_version_from_the_tag():
    assert versions.image_version("carlasim/carla:0.9.10.1") == "0.9.10.1"
    assert versions.image_version("carlasim/carla:latest") is None
    assert versions.image_version("carlasim/carla") is None


# --- the checks ----------------------------------------------------------
class _Carla:
    def __init__(self, image):
        self.image = image
        self.client_egg = None


class _Pin:
    def __init__(self, path, source_digest=None, commit=None):
        self.path = path
        self.source_digest = source_digest
        self.commit = commit


class _Egg:
    def __init__(self, name, host_path):
        self.name = name
        # The *host* path, which is the half a test can stage files in. Left
        # unset it would fall back to the image's fixed container path, which
        # does not exist outside the image.
        self.host_path = host_path


class _Scenario:
    def __init__(self, pin):
        self.runner = pin


class _Sim:
    """A CARLA source carrying what the version checks read."""

    kind = "carla"

    def __init__(self, image, api, egg, pin):
        self.carla = _Carla(image)
        self.carla.client_egg = _Egg(egg, api)
        self.scenario = _Scenario(pin)


class _Config:
    def __init__(self, sim=None):
        self.source = sim


def test_no_simulator_settings_make_no_claims(monkeypatch):
    assert versions.check_versions(_Config(None)) == []


def test_a_config_with_no_scenario_declares_no_pin(tmp_path, monkeypatch):
    # A config that drives an already-running simulator runs no scenario of its
    # own, so there is no ScenarioRunner pin to check. Reading one anyway was an
    # AttributeError out of the preflight rather than a readable refusal.
    monkeypatch.setattr(versions, "local_image_present", lambda *_a, **_k: True)
    egg = "carla-0.9.10-py3.7-linux-x86_64.egg"
    open(os.path.join(tmp_path, egg), "w").close()
    sim = _Sim("carlasim/carla:0.9.10.1", str(tmp_path), egg, None)
    sim.scenario = None
    assert versions.check_versions(_Config(sim), interpreter="3.8") == []


def test_a_patch_release_server_matches_a_minor_release_client(tmp_path, monkeypatch):
    # 0.9.10.1 server against a 0.9.10 client egg: the client API is stable
    # within a minor, so this must not be reported.
    monkeypatch.setattr(versions, "local_image_present", lambda *_a, **_k: True)
    egg = "carla-0.9.10-py3.7-linux-x86_64.egg"
    open(os.path.join(tmp_path, egg), "w").close()
    sim = _Sim("carlasim/carla:0.9.10.1", str(tmp_path), egg, _Pin(str(tmp_path)))
    assert versions.check_versions(_Config(sim), interpreter="3.8") == []


def test_a_different_release_is_a_mismatch(tmp_path, monkeypatch):
    # 0.9.10 and 0.9.12 are different CARLA releases with different client
    # APIs, so comparing only major.minor would wrongly call them a match.
    monkeypatch.setattr(versions, "local_image_present", lambda *_a, **_k: True)
    egg = "carla-0.9.10-py3.7-linux-x86_64.egg"
    open(os.path.join(tmp_path, egg), "w").close()
    sim = _Sim("carlasim/carla:0.9.12", str(tmp_path), egg, _Pin(str(tmp_path)))
    problems = versions.check_versions(_Config(sim), interpreter="3.8")
    assert any("client/server mismatch" in p for p in problems)


def test_a_missing_egg_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(versions, "local_image_present", lambda *_a, **_k: True)
    sim = _Sim(
        "carlasim/carla:0.9.10.1",
        str(tmp_path),
        "carla-0.9.10-py3.7-linux-x86_64.egg",
        _Pin(str(tmp_path)),
    )
    problems = versions.check_versions(_Config(sim), interpreter="3.8")
    assert any("client_egg.name not found" in p for p in problems)


def test_an_interpreter_too_new_for_the_egg_is_reported(tmp_path, monkeypatch):
    # This is the failure that once left an image with no CARLA client at all:
    # the py3.7 egg loads under 3.8 but not under 3.10.
    monkeypatch.setattr(versions, "local_image_present", lambda *_a, **_k: True)
    egg = "carla-0.9.10-py3.7-linux-x86_64.egg"
    open(os.path.join(tmp_path, egg), "w").close()
    sim = _Sim("carlasim/carla:0.9.10.1", str(tmp_path), egg, _Pin(str(tmp_path)))
    problems = versions.check_versions(_Config(sim), interpreter="3.10")
    assert any("will not import" in p for p in problems)
    # 3.8 is fine, which is the whole reason the image is built on 3.8.
    assert versions.check_versions(_Config(sim), interpreter="3.8") == []


def test_a_missing_image_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(versions, "local_image_present", lambda *_a, **_k: False)
    egg = "carla-0.9.10-py3.7-linux-x86_64.egg"
    open(os.path.join(tmp_path, egg), "w").close()
    sim = _Sim("carlasim/carla:0.9.10.1", str(tmp_path), egg, _Pin(str(tmp_path)))
    problems = versions.check_versions(_Config(sim), interpreter="3.8")
    assert any("not present locally" in p for p in problems)


def test_a_changed_scenario_runner_tree_is_caught(tmp_path, monkeypatch):
    monkeypatch.setattr(versions, "local_image_present", lambda *_a, **_k: True)
    egg = "carla-0.9.10-py3.7-linux-x86_64.egg"
    open(os.path.join(tmp_path, egg), "w").close()
    sr = _tree(tmp_path / "sr", {"srunner/x.py": "print(1)\n"})
    pinned = versions.source_digest(sr)
    sim = _Sim("carlasim/carla:0.9.10.1", str(tmp_path), egg, _Pin(sr, pinned))
    assert versions.check_versions(_Config(sim), interpreter="3.8") == []

    with open(os.path.join(sr, "srunner/x.py"), "w") as handle:
        handle.write("print(2)\n")
    problems = versions.check_versions(_Config(sim), interpreter="3.8")
    assert any("content mismatch" in p for p in problems)
    # The message has to say it is contents, not a commit, or the reader will
    # assume a version bump and go looking for the wrong thing.
    assert any("not a commit id" in p for p in problems)


def test_a_missing_scenario_runner_tree_is_caught(tmp_path, monkeypatch):
    monkeypatch.setattr(versions, "local_image_present", lambda *_a, **_k: True)
    egg = "carla-0.9.10-py3.7-linux-x86_64.egg"
    open(os.path.join(tmp_path, egg), "w").close()
    sim = _Sim(
        "carlasim/carla:0.9.10.1",
        str(tmp_path),
        egg,
        _Pin(str(tmp_path / "gone"), "deadbeefdeadbeef"),
    )
    problems = versions.check_versions(_Config(sim), interpreter="3.8")
    # The refusal names the directory it looked in, and the variable that
    # overrides it: with a containerised scenario the tree is somewhere else on
    # this machine, and the pins path is a path inside that container.
    assert len(problems) == 1
    assert str(tmp_path / "gone") in problems[0]
    assert versions.TREE_HOST_PATH_ENV in problems[0]


def test_the_host_tree_is_digested_when_it_is_named(tmp_path, monkeypatch):
    """The digest is of the tree the run mounts, not of the pins container path.

    A containerised scenario mounts the tree at a path inside that container,
    so reading the pins own path out here finds nothing - which was reported
    as an uncheckable digest on a machine where the tree was present all along.
    """
    monkeypatch.setattr(versions, "local_image_present", lambda *_a, **_k: True)
    egg = "carla-0.9.10-py3.7-linux-x86_64.egg"
    open(os.path.join(tmp_path, egg), "w").close()
    tree = tmp_path / "scenario_runner"
    tree.mkdir()
    (tree / "scenario_runner.py").write_text("# the tree\n")
    digest = versions.source_digest(str(tree))

    monkeypatch.setenv(versions.TREE_HOST_PATH_ENV, str(tree))
    sim = _Sim(
        "carlasim/carla:0.9.10.1",
        str(tmp_path),
        egg,
        _Pin("/opt/scenario_runner", digest),
    )
    assert versions.check_versions(_Config(sim), interpreter="3.8") == []


def test_an_unpinned_tree_is_not_checked(tmp_path, monkeypatch):
    # source_digest: None means "do not check", so a config that makes no claim
    # is not failed for it.
    monkeypatch.setattr(versions, "local_image_present", lambda *_a, **_k: True)
    egg = "carla-0.9.10-py3.7-linux-x86_64.egg"
    open(os.path.join(tmp_path, egg), "w").close()
    sim = _Sim("carlasim/carla:0.9.10.1", str(tmp_path), egg, _Pin("/nonexistent", None))
    assert versions.check_versions(_Config(sim), interpreter="3.8") == []

# --- where the egg is looked for ----------------------------------------
def test_the_egg_directory_defaults_to_the_image_path():
    # No host path set: the image bakes the egg at a fixed location, so there is
    # nothing to configure and nothing to get wrong.
    class _NoHost:
        carla_client_egg_host_path = None

    assert versions.egg_directory(_NoHost()) == versions.CONTAINER_CARLA_EGG_DIR


def test_a_host_path_overrides_the_image_path():
    # Set, it is unambiguously a host directory, mounted over the fixed path -
    # which is the distinction the removed carla_python_api could not express.
    class _WithHost:
        host_path = "/somewhere/on/the/host"

    assert versions.egg_directory(_WithHost()) == "/somewhere/on/the/host"
