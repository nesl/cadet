"""Loading, overriding, sweeping and identifying CADET configs.

What this is for:
    Turns a YAML file into validated :class:`~cadet.config.schema.Config`
    objects. Applies ``--set key=value`` command-line overrides, expands a
    ``sweep`` into one concrete config per parameter combination, and derives a
    stable ``run_id`` and per-repetition seed.

What it contains:
    - :func:`load_yaml`, :func:`load_config_data` (resolves ``extends:`` bases by
      deep-merge), :func:`apply_overrides`, :func:`set_by_path`
    - :func:`build_config` (dict -> ``Config``)
    - :func:`expand_sweep` (base dict -> list of concrete ``Config``)
    - :func:`config_hash`, :func:`run_id`, :func:`seed_for_rep`
    - :class:`ConfigError` and :func:`format_validation_error` for readable errors

How it fits into CADET:
    ``cadet validate`` (``cadet.cli``) and, later, the runner/orchestrator call
    these helpers. Overrides and sweeps are applied to the raw mapping *before*
    validation, so every produced config is typed and checked.

How to configure it:
    Overrides are ``dotted.path=yaml_value`` (list indices allowed, e.g.
    ``nodes[1].host=localhost``). Sweep axes are the same dotted paths mapped to
    value lists under ``sweep.axes``.
"""

import copy
import hashlib
import itertools
import json
import os
import re
from typing import Any, Dict, List, Tuple

import yaml
from pydantic import ValidationError

from cadet.config.schema import Config

# Number of hex chars kept from the config digest. Short but collision-safe for
# the scale of a sweep; this is an identifier, never a timestamp.
RUN_ID_HASH_LEN = 10

# A bracketed subscript in an override path. The inner part is a list index
# when it is all digits and a node's name otherwise, with optional quotes so
# that both nodes[server] and nodes['server'] mean the same thing. Names are
# matched rather than positions because a position is a property of how the
# config happens to be written: inserting a node above renumbers every path
# below it, and the path that still parses is then quietly about a different
# node.
_INDEX_RE = re.compile(r"\[\s*(?:'([^']*)'|\"([^\"]*)\"|([^\]]*?))\s*\]")


def _subscript(match: "re.Match[str]") -> str:
    """Return the dotted form of one bracketed subscript."""
    inner = next(group for group in match.groups() if group is not None)
    return "." + inner


class ConfigError(Exception):
    """Raised for unreadable/undecodable configs or bad override strings."""


def load_yaml(path: str) -> Dict[str, Any]:
    """Parse a YAML file into a mapping (raises :class:`ConfigError`)."""
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping, got {}".format(type(data).__name__))
    return data


# Discriminator keys for the schema's tagged unions. If a child changes one of
# these, it has selected a different variant, so the subtree is replaced wholesale
# rather than merged (merging e.g. a synthetic source onto a replay source would
# leak synthetic-only keys into the replay model).
_DISCRIMINATORS = ("kind", "plugin")


def _is_variant_switch(base: Dict[str, Any], override: Dict[str, Any]) -> bool:
    """True if ``override`` selects a different union variant than ``base``."""
    for disc in _DISCRIMINATORS:
        if disc in base and disc in override and base[disc] != override[disc]:
            return True
    return False


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` (override wins; lists replace).

    Dict values are merged key-by-key; any non-dict value (including lists) from
    ``override`` replaces the base value wholesale — we never try to merge lists,
    which keeps composition predictable. A dict that switches a discriminated-union
    variant (``kind``/``plugin``) also replaces the base subtree wholesale.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
            and not _is_variant_switch(result[key], value)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config_data(path: str, _seen: Tuple[str, ...] = ()) -> Dict[str, Any]:
    """Load a YAML config, resolving any ``extends:`` base(s) by deep-merge.

    ``extends`` may be a single path or a list of paths, each relative to the
    config file that declares it. Bases are merged left-to-right, then the current
    file is merged on top (so the child always wins). Supports nested extends and
    detects cycles.
    """
    real = os.path.realpath(path)
    if real in _seen:
        chain = " -> ".join(_seen + (real,))
        raise ConfigError("circular extends: {}".format(chain))
    data = load_yaml(path)
    bases = data.pop("extends", None)
    if bases is None:
        return data
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list):
        raise ConfigError("extends must be a path or list of paths, got {!r}".format(bases))

    here = os.path.dirname(real)
    merged: Dict[str, Any] = {}
    for base in bases:
        base_path = base if os.path.isabs(base) else os.path.join(here, base)
        merged = _deep_merge(merged, load_config_data(base_path, _seen + (real,)))
    return _deep_merge(merged, data)


def _split_path(path: str) -> List[Any]:
    """Split a dotted path into keys, turning ``a[0].b`` / ``a.0.b`` into tokens.

    Integer tokens address list indices; everything else is a dict key.
    """
    normalized = _INDEX_RE.sub(_subscript, path)  # a[0] -> a.0, a['b'] -> a.b
    tokens: List[Any] = []
    for part in normalized.split("."):
        if part == "":
            raise ConfigError("empty path segment in {!r}".format(path))
        tokens.append(int(part) if part.isdigit() else part)
    return tokens


def _named_element(items: list, name: str, path: str) -> Dict[str, Any]:
    """Return the one element of ``items`` whose ``name`` field is ``name``.

    Raises rather than guessing. A path that cannot be resolved to exactly one
    element is a mistake in the override, and the alternatives - creating an
    element, or replacing one - both produce a config the author did not ask
    for. Replacing is the worse of the two: it silently drops every field the
    element had, and the failure then surfaces as schema validation about
    fields that were never mentioned.
    """
    matches = [
        item for item in items if isinstance(item, dict) and item.get("name") == name
    ]
    if len(matches) == 1:
        return matches[0]
    known = [
        str(item.get("name"))
        for item in items
        if isinstance(item, dict) and item.get("name") is not None
    ]
    if not matches:
        raise ConfigError(
            "no element named {!r} in path {!r}. Named: {}. Use a name that is "
            "there, or an index such as [0]".format(
                name, path, ", ".join(known) if known else "(none)"
            )
        )
    raise ConfigError(
        "{} elements are named {!r} in path {!r}, so the path does not say "
        "which. Use an index such as [0]".format(len(matches), name, path)
    )


def set_by_path(data: Dict[str, Any], path: str, value: Any) -> None:
    """Set ``value`` at ``path`` inside the nested ``data`` (mutates in place).

    Intermediate dict keys are created if missing; list indices must already
    exist (we never guess list length). A list may also be subscripted by an
    element's ``name`` - ``nodes[server].host`` and ``nodes['server'].host``
    both mean the node named ``server`` - which is how a node is identified
    everywhere else in a config, and which survives a node being inserted
    above it.
    """
    tokens = _split_path(path)
    current: Any = data
    for i, token in enumerate(tokens[:-1]):
        nxt = tokens[i + 1]
        if isinstance(token, int):
            if not isinstance(current, list) or token >= len(current):
                raise ConfigError("list index {} out of range in path {!r}".format(token, path))
            current = current[token]
        elif isinstance(current, list):
            current = _named_element(current, token, path)
        else:
            if not isinstance(current, dict):
                raise ConfigError("cannot descend into {!r} in path {!r}".format(token, path))
            if token not in current:
                current[token] = [] if isinstance(nxt, int) else {}
            current = current[token]
    last = tokens[-1]
    if isinstance(last, int):
        if not isinstance(current, list) or last >= len(current):
            raise ConfigError("list index {} out of range in path {!r}".format(last, path))
        current[last] = value
    elif isinstance(current, list):
        # The path ends on a named element, so what it asks for is to replace
        # that whole element. Refused: the fields not mentioned would be
        # dropped, and an override that silently discards half a node is worse
        # than one that does not run. Naming the field is unambiguous.
        _named_element(current, last, path)
        raise ConfigError(
            "path {!r} names a whole list element, which would replace it and "
            "drop its other fields. Name the field to set, such as "
            "{}.<field>".format(path, path)
        )
    else:
        if not isinstance(current, dict):
            raise ConfigError("cannot set key {!r} in path {!r}".format(last, path))
        current[last] = value


def apply_overrides(data: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply ``key=value`` overrides to a copy of ``data`` and return it.

    The value is parsed as a YAML scalar, so ``5``, ``true`` and ``[a, b]`` get
    their natural types.
    """
    result = copy.deepcopy(data)
    for item in overrides:
        if "=" not in item:
            raise ConfigError("override {!r} is not of the form key=value".format(item))
        key, raw_value = item.split("=", 1)
        set_by_path(result, key.strip(), yaml.safe_load(raw_value))
    return result


# Keys that moved when the simulator settings folded into the Data Source
# section. A config naming one of these is refused with the key that replaced
# it, because a silent alias would leave two spellings in circulation and the
# other sessions' configs are rebased onto this one.
MOVED_KEYS = {
    "sim.carla.image": "source.carla.image",
    "sim.carla.rpc_port": "source.carla.port",
    "sim.carla.gpu_index": "source.carla.gpu_index",
    "sim.carla.quality": "source.carla.quality",
    "sim.carla.offscreen": "source.carla.offscreen",
    "sim.client_egg": "source.carla.client_egg.name",
    "sim.carla_client_egg_host_path": "source.carla.client_egg.host_path",
    "sim.scenario_runner.path": "source.scenario.runner.path",
    "sim.scenario_runner.source_digest": "source.scenario.runner.source_digest",
    "sim.scenario_runner.commit": "source.scenario.runner.commit",
    "sim.scenario_runner.image": "source.scenario.runner.image",
    "source.simulator_host": "source.carla.host",
    "source.simulator_port": "source.carla.port",
    "source.simulator_timeout": "source.carla.timeout",
}

# Where the whole section went, for a `sim.*` key that is not listed above: an
# unlisted key must be refused too, or a config could carry a key nothing reads.
MOVED_SECTIONS = {"sim": "source.carla (and source.scenario.runner)"}


def _moved_key_paths(data: Dict[str, Any]) -> List[str]:
    """Return the dotted paths in ``data`` that name a key that has moved.

    A leaf is reported rather than the section holding it, so a config that
    spells out four moved keys is told about four replacements instead of being
    told once that a section is gone. The section itself is only reported for a
    key that has no listed replacement - which is how an unlisted key inside a
    moved section is still refused.
    """
    found = []

    def walk(node, prefix):
        for key, value in node.items():
            path = "{}.{}".format(prefix, key) if prefix else str(key)
            if path in MOVED_KEYS:
                found.append(path)
                continue
            if isinstance(value, dict) and value:
                walk(value, path)
                continue
            if path.split(".", 1)[0] in MOVED_SECTIONS:
                found.append(path)

    if isinstance(data, dict):
        walk(data, "")
    return found


def reject_moved_keys(data: Dict[str, Any]) -> None:
    """Raise :class:`ConfigError` naming the replacement for each moved key."""
    paths = _moved_key_paths(data)
    if not paths:
        return
    lines = []
    for path in sorted(set(paths)):
        if path in MOVED_KEYS:
            lines.append("  {} is now {}".format(path, MOVED_KEYS[path]))
        else:
            section = path.split(".", 1)[0]
            lines.append(
                "  {} is gone: that section moved to {}".format(
                    path, MOVED_SECTIONS[section]
                )
            )
    raise ConfigError(
        "this config uses keys that have moved:\n"
        + "\n".join(lines)
        + "\n\nscripts/migrate_simulator_keys.py rewrites a config or a shell "
        "script in place."
    )


def build_config(data: Dict[str, Any]) -> Config:
    """Validate a mapping into a :class:`Config` (raises ``ValidationError``)."""
    reject_moved_keys(data)
    return Config.model_validate(data)


def resolve_app_params(config: Config):
    """Validate ``config.app.params`` against the registered plugin's model.

    The app plugin (registered via :mod:`cadet.core.apps`, typically by importing
    an ``examples/*`` module) declares ``params_model``. This validates the raw
    params against it and returns the typed instance. Raises :class:`ConfigError`
    if the plugin is not registered, or ``ValidationError`` if params are invalid.
    """
    from cadet.core.apps import get_app

    try:
        plugin = get_app(config.app.plugin)
    except KeyError as exc:
        # get_app's own message names what is registered and gives an example of
        # naming a module, so it is carried through rather than replaced by a
        # shorter one that says less.
        raise ConfigError(str(exc))
    model = getattr(plugin, "params_model", None)
    if model is None:
        raise ConfigError("app plugin {!r} declares no params_model".format(config.app.plugin))
    return model.model_validate(config.app.params)


def expand_sweep(data: Dict[str, Any]) -> List[Config]:
    """Expand ``sweep.axes`` into the cartesian product of concrete configs.

    Each produced config has its swept paths set and an empty ``sweep``. With no
    axes, this returns a single config. Validation happens per combination.
    """
    axes: Dict[str, List[Any]] = (data.get("sweep") or {}).get("axes", {}) or {}
    if not axes:
        return [build_config(data)]

    paths = list(axes.keys())
    value_lists = [axes[p] for p in paths]
    configs: List[Config] = []
    for combo in itertools.product(*value_lists):
        variant = copy.deepcopy(data)
        variant["sweep"] = {"axes": {}}  # concrete config carries no sweep
        for path, value in zip(paths, combo):
            set_by_path(variant, path, value)
        configs.append(build_config(variant))
    return configs


# Sections a run's identity is not a function of. ``outputs`` says where results
# are written; ``analysis`` says what may read them back afterwards. Neither is
# part of what was run, and including either would mean that pointing the
# foundation-model interface at a different endpoint - or at one at all - gave
# every cell of a sweep a new run id, so `--resume` would re-run work that was
# already done and an existing output tree would stop matching its own config.
IDENTITY_EXCLUDES = ("outputs", "analysis")


def config_hash(config: Config) -> str:
    """Short, stable digest of a resolved config, excluding
    :data:`IDENTITY_EXCLUDES`.

    Keys are sorted so the digest is order-independent.
    """
    payload = config.model_dump(mode="json", by_alias=True)
    for section in IDENTITY_EXCLUDES:
        payload.pop(section, None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:RUN_ID_HASH_LEN]


def run_id(config: Config, rep: int) -> str:
    """Identifier for one repetition: ``<config-hash>-r<rep>``."""
    return "{}-r{}".format(config_hash(config), rep)


def seed_for_rep(config: Config, rep: int) -> Any:
    """Per-repetition seed: ``experiment.seed + rep`` (or ``None`` if unset)."""
    base = config.experiment.seed
    if base is None:
        return None
    return base + rep


def format_validation_error(exc: ValidationError) -> str:
    """Render a pydantic ``ValidationError`` as readable ``loc: message`` lines."""
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()))
        lines.append("  {}: {}".format(loc or "<root>", err.get("msg", "invalid")))
    return "\n".join(lines)


def load_run_configs(path: str, overrides: List[str]) -> Tuple[List[Config], Config]:
    """Load, override and expand a config file into concrete run configs.

    Returns ``(run_configs, base_config)`` where ``base_config`` is the single
    pre-expansion config (useful for reporting).
    """
    data = apply_overrides(load_config_data(path), overrides)
    base = build_config(data)
    return expand_sweep(data), base
