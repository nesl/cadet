"""Tests for ``--set key=value`` overrides and dotted-path setting.

Covers: scalar/typed value parsing, nested dotted paths, list-index paths, and
that a malformed override raises a readable error.
"""

import pytest
from conftest import EXAMPLE_CONFIGS

from cadet.config import build_config, load_config_data
from cadet.config.loader import ConfigError, apply_overrides, set_by_path


def _braking_dict():
    return load_config_data(str(EXAMPLE_CONFIGS[0]))


def test_override_scalar_and_type_parsing():
    data = apply_overrides(_braking_dict(), ["source.carla.port=2001"])
    cfg = build_config(data)
    assert cfg.source.carla.port == 2001  # parsed as int, not "2001"


def test_override_bool_and_list():
    data = apply_overrides(
        _braking_dict(),
        ["channels[0].send_meta=true", "app.params.target_classes=[car, person]"],
    )
    cfg = build_config(data)
    assert cfg.channels[0].send_meta is True
    assert cfg.app.params["target_classes"] == ["car", "person"]


def test_override_list_index_path():
    data = apply_overrides(_braking_dict(), ["nodes[1].host=example-host"])
    cfg = build_config(data)
    assert cfg.nodes[1].host == "example-host"


def test_override_creates_nested_dict_key():
    data = {"a": {}}
    set_by_path(data, "a.b.c", 5)
    assert data == {"a": {"b": {"c": 5}}}


def test_malformed_override_rejected():
    with pytest.raises(ConfigError):
        apply_overrides(_braking_dict(), ["no_equals_sign"])


def test_out_of_range_index_rejected():
    with pytest.raises(ConfigError):
        set_by_path({"xs": [1]}, "xs.5", 9)


def test_a_list_element_can_be_named_instead_of_counted():
    """``nodes[server].host`` means the node named server.

    A position is a property of how the config happens to be written: insert a
    node above and every numbered path below it addresses a different node,
    while still parsing. The name is what the rest of a config identifies a
    node by - links name their endpoints, placements name their nodes - so an
    override path can too. Quoted and bare spellings mean the same thing.
    """
    for path in ("nodes['server'].host", 'nodes["server"].host', "nodes[server].host"):
        data = apply_overrides(_braking_dict(), [path + "=198.51.100.7"])
        cfg = build_config(data)
        server = [n for n in cfg.nodes if n.name == "server"][0]
        assert server.host == "198.51.100.7", path
        # Everything else about that node survives, and so does the other node.
        assert server.role == "server"
        assert [n.name for n in cfg.nodes] == [n.name for n in build_config(_braking_dict()).nodes]


def test_naming_a_whole_list_element_is_refused_rather_than_replacing_it():
    """The dangerous case: a path that ends on the element itself.

    Setting the element would drop every field not mentioned. That used to be
    what happened - a path like this created a top-level key called
    ``nodes['server']`` holding only the fields given, and the run then failed
    pydantic validation about extra inputs, which named neither the node nor
    the override. The refusal says what to write instead.
    """
    with pytest.raises(ConfigError) as caught:
        apply_overrides(_braking_dict(), ["nodes['server']={host: 198.51.100.7}"])
    message = str(caught.value)
    assert "names a whole list element" in message
    assert "<field>" in message

    # And nothing was changed on the way to refusing.
    data = _braking_dict()
    try:
        apply_overrides(data, ["nodes['server']={host: 198.51.100.7}"])
    except ConfigError:
        pass
    assert data == _braking_dict()


def test_an_unknown_or_ambiguous_name_says_which_names_exist():
    """A misspelt node is a mistake in the override, not a new node."""
    with pytest.raises(ConfigError) as caught:
        apply_overrides(_braking_dict(), ["nodes['serverr'].host=198.51.100.7"])
    message = str(caught.value)
    assert "no element named 'serverr'" in message
    assert "server" in message  # the names that do exist are listed
    assert "[0]" in message  # and the other valid form is named

    # Two elements with one name cannot be told apart, so the path is refused
    # rather than resolved to whichever comes first.
    data = {"nodes": [{"name": "same", "host": "a"}, {"name": "same", "host": "b"}]}
    with pytest.raises(ConfigError) as caught:
        set_by_path(data, "nodes['same'].host", "198.51.100.7")
    assert "2 elements are named 'same'" in str(caught.value)
    assert data["nodes"][0]["host"] == "a"


def test_links_are_nameable_too_and_indices_still_work():
    """The same rule for every named list, and nothing about indices changes."""
    data = apply_overrides(_braking_dict(), ["links['frames_link'].network.mode=netem"])
    assert build_config(data).links[0].network.mode == "netem"

    by_index = apply_overrides(_braking_dict(), ["links[0].network.mode=netem"])
    assert build_config(by_index).links[0].network.mode == "netem"

    with pytest.raises(ConfigError) as caught:
        apply_overrides(_braking_dict(), ["nodes[9].host=198.51.100.7"])
    assert "out of range" in str(caught.value)


def test_the_suggestions_the_config_report_prints_are_usable_overrides():
    """What a user is told to type has to work when typed.

    ``cadet.config.implemented`` prints a ``--set`` path for each node's
    launcher and each link's mode. It printed ``nodes['server'].launcher=...``,
    which was not a path the loader could follow, so following the advice
    produced an error about extra inputs.
    """
    cfg = build_config(_braking_dict())
    suggestions = ["nodes[{!r}].launcher={}".format(n.name, n.launcher) for n in cfg.nodes]
    suggestions += [
        "links[{!r}].network.mode={}".format(link.name, link.network.mode)
        for link in cfg.links
    ]
    assert suggestions, "the config under test has neither nodes nor links"
    for suggestion in suggestions:
        # Raises if the path cannot be followed; the point is that it does not.
        build_config(apply_overrides(_braking_dict(), [suggestion]))
