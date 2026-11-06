"""Tests for what a run records about the images its nodes ran.

A role placed in a container runs the image's copy of the code, so the
repository's own commit does not identify what ran. These cover
:func:`cadet.orchestrator.placement.image_identity`: which of the tag, the
content digest and the recorded commit it reports, and what it leaves out when
the daemon or the image cannot answer.
"""

def test_an_images_identity_is_the_tag_the_digest_and_the_recorded_commit(monkeypatch):
    """What a run records about an image, and what it leaves out when unknown."""
    from cadet.orchestrator import placement as placement_mod

    calls = []

    class _Result(object):
        def __init__(self, returncode, stdout):
            self.returncode = returncode
            self.stdout = stdout

    def _fake_docker(argv, docker="docker"):
        calls.append(list(argv))
        if "{{.Id}}" in argv:
            return _Result(0, "sha256:" + "a" * 64 + "\n")
        return _Result(0, "b" * 40 + "\n")

    monkeypatch.setattr(placement_mod, "_docker", _fake_docker)
    identity = placement_mod.image_identity("cadet:dev")
    assert identity == {
        "image": "cadet:dev",
        "id": "sha256:" + "a" * 64,
        "revision": "b" * 40,
    }

    # An image carrying no revision label is recorded by tag and digest, with no
    # revision key at all: a label that says "<no value>" is not a commit.
    def _no_label(argv, docker="docker"):
        if "{{.Id}}" in argv:
            return _Result(0, "sha256:" + "c" * 64)
        return _Result(0, "<no value>")

    monkeypatch.setattr(placement_mod, "_docker", _no_label)
    bare = placement_mod.image_identity("cadet:dev")
    assert bare == {"image": "cadet:dev", "id": "sha256:" + "c" * 64}

    # A daemon that cannot be asked leaves the tag the node named and nothing
    # invented beside it.
    def _no_daemon(argv, docker="docker"):
        return _Result(1, "")

    monkeypatch.setattr(placement_mod, "_docker", _no_daemon)
    assert placement_mod.image_identity("cadet:dev") == {"image": "cadet:dev"}


def test_an_unreachable_machine_is_not_reported_as_one_without_the_image():
    """An ssh that could not run says nothing about what is on the far side.

    Measured while running the distributed cells: the orchestrator ran in a
    container whose uid had no passwd entry, ssh refused with "No user exists
    for uid" and exited 255, and the refusal that came back named the image as
    absent from the second machine - which had it. The advice was to build or
    copy an image that was already there, so the reader is sent to the wrong
    problem. Docker's own "no such image" is exit 1 and still means missing.
    """
    import pytest

    from cadet.orchestrator import placement as placement_mod

    class _Machine(object):
        def __init__(self, code, out):
            self._code = code
            self._out = out

        def describe(self):
            return "the second machine"

        def run(self, argv, timeout_s=None):
            return self._code, self._out

    unreachable = _Machine(255, "No user exists for uid 1001\n")
    with pytest.raises(placement_mod.PlacementError) as caught:
        placement_mod.image_present("cadet:dev", unreachable)
    message = str(caught.value)
    assert "could not reach the second machine" in message
    assert "No user exists for uid 1001" in message

    # Docker answering "no such image" is still a missing image, and is still
    # reported as one rather than as a machine that could not be reached.
    assert placement_mod.image_present("cadet:dev", _Machine(1, "")) is False
    assert placement_mod.image_present("cadet:dev", _Machine(0, "sha256:x")) is True
