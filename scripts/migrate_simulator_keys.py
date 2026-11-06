"""Rewrite the simulator config keys that moved under the Data Source section.

What this is for:
    The simulator settings used to live in a top-level ``sim`` section and in
    three ``source.simulator_*`` keys. They now sit under ``source.carla`` and
    ``source.scenario.runner``, and the loader refuses the old spellings rather
    than aliasing them - so every config, and every script that passes a key
    with ``--set``, has to be rewritten once. This does that rewrite in place.

What it contains:
    - :data:`KEY_MOVES`: old dotted key to new, imported from
      :mod:`cadet.config.loader` so the refusal and the rewrite cannot drift.
    - :func:`migrate_script_text`: a command line carries the dotted path
      verbatim, so it is replaced verbatim - longest key first, or
      ``sim.carla`` would be rewritten inside ``source.carla.port``.
    - :func:`migrate_yaml_text`: a config carries the keys as nested blocks, so
      they are collected and re-emitted under their new parents.

    The YAML pass works on lines rather than through a parser, because a round
    trip through a YAML library discards every comment and reorders what it
    keeps, and these files are read by people. It handles the shapes this
    project writes - a ``sim:`` block, the three flat ``source.simulator_*``
    keys, and an existing ``source.scenario:`` block to nest ``runner:`` into -
    and reports anything else instead of guessing.

How it fits into CADET:
    Not part of a run. Run once per tree:

        python scripts/migrate_simulator_keys.py examples/ scripts/

    ``--check`` reports what would change, writes nothing, and exits non-zero,
    which is what a session should run before its next hand-off.

How to configure it:
    Nothing. The mapping is the schema's.
"""

import argparse
import io
import os
import re
import sys

try:  # pragma: no cover - present in a checkout, absent in a stray copy
    from cadet.config.loader import MOVED_KEYS as KEY_MOVES
except Exception:  # pragma: no cover
    KEY_MOVES = {
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

YAML_SUFFIXES = (".yaml", ".yml")
SCRIPT_SUFFIXES = (".sh", ".py", ".md", ".rst")

# sim.carla.* and the flat source keys land here, under these names.
_CARLA_RENAMES = {"rpc_port": "port", "simulator_host": "host",
                  "simulator_port": "port", "simulator_timeout": "timeout"}


def migrate_script_text(text):
    """Rewrite dotted keys as a command line spells them."""
    changed = []
    for old in sorted(KEY_MOVES, key=len, reverse=True):
        new = KEY_MOVES[old]
        pattern = re.compile(r"(?<![\w.])" + re.escape(old) + r"(?![\w.])")
        text, count = pattern.subn(new, text)
        if count:
            changed.append((old, new, count))
    return text, changed


class _Line:
    __slots__ = ("raw", "indent", "key", "rest")

    def __init__(self, raw):
        self.raw = raw
        body = raw.rstrip("\n")
        self.indent = len(body) - len(body.lstrip())
        stripped = body.strip()
        if stripped and not stripped.startswith("#") and ":" in stripped:
            self.key = stripped.split(":", 1)[0].strip()
            self.rest = stripped.split(":", 1)[1]
        else:
            self.key, self.rest = None, ""


def _block_end(lines, start, indent):
    """Index just past the block whose header is at ``start``."""
    i = start + 1
    while i < len(lines):
        line = lines[i]
        if line.key is None and not line.raw.strip():
            i += 1
            continue
        if line.indent <= indent and line.raw.strip():
            break
        i += 1
    return i


def migrate_yaml_text(text):
    """Return ``(new_text, changed, unhandled)`` for one config's contents."""
    lines = [_Line(raw) for raw in text.splitlines(keepends=True)]
    changed = []
    unhandled = []

    carla = []  # (name, rest) to emit under source.carla
    runner = []  # (name, rest) to emit under source.scenario.runner
    drop = set()  # line indices to remove

    # 1. the top-level sim: block
    for i, line in enumerate(lines):
        if line.indent == 0 and line.key == "sim":
            end = _block_end(lines, i, 0)
            sub = None
            child_indent = next(
                (lines[j].indent for j in range(i + 1, end) if lines[j].key is not None), 2
            )
            for j in range(i + 1, end):
                inner = lines[j]
                if inner.key is None:
                    continue
                # A nested block ends when the indentation comes back to the
                # section's own children; without this the keys after it are
                # still read as belonging to it, which put `client_egg` inside
                # `carla:` as a flat key.
                if sub is not None and inner.indent <= child_indent:
                    sub = None
                if inner.key in ("carla", "scenario_runner"):
                    sub = inner.key
                    continue
                if sub == "carla":
                    carla.append((_CARLA_RENAMES.get(inner.key, inner.key), inner.rest))
                elif sub == "scenario_runner":
                    runner.append((inner.key, inner.rest))
                elif inner.key == "client_egg":
                    carla.append(("client_egg", ""))
                    carla.append(("  name", inner.rest))
                elif inner.key == "carla_client_egg_host_path":
                    carla.append(("  host_path", inner.rest))
                else:
                    unhandled.append(inner.raw.strip())
            drop.update(range(i, end))
            changed.append(("sim:", "source.carla / source.scenario.runner", 1))
            break

    # 2. the three flat keys, inside source:
    for i, line in enumerate(lines):
        if i in drop or line.key not in ("simulator_host", "simulator_port", "simulator_timeout"):
            continue
        carla.append((_CARLA_RENAMES[line.key], line.rest))
        changed.append(("source." + line.key, KEY_MOVES["source." + line.key], 1))
        drop.add(i)

    # Two sources for one key is a config whose own settings disagreed - the
    # launch port and the client's connect port were separate fields with
    # nothing tying them, and a config could set 3000 and 2000. Collapsing them
    # is the point of the move, but which value a run meant is not this
    # script's to guess.
    names = [name.strip() for name, _rest in carla]
    for name in sorted(set(names)):
        if names.count(name) > 1:
            values = [rest.strip() for n, rest in carla if n.strip() == name]
            unhandled.append(
                "two values for source.carla.{}: {} - the old config set the "
                "launch port and the client's connect port separately; pick "
                "one".format(name, " and ".join(v for v in values if v))
            )
    if unhandled:
        return text, changed, unhandled

    if not carla and not runner:
        return text, changed, unhandled

    # 3. where to put them: the source: block, and its scenario: block if any
    source_at = next(
        (i for i, line in enumerate(lines) if line.indent == 0 and line.key == "source"),
        None,
    )
    if source_at is None:
        unhandled.append("no top-level source: block to move the keys into")
        return text, changed, unhandled
    source_end = _block_end(lines, source_at, 0)
    inner_indent = next(
        (ln.indent for ln in lines[source_at + 1 : source_end] if ln.key is not None), 2
    )
    scenario_at = next(
        (
            i
            for i in range(source_at + 1, source_end)
            if lines[i].key == "scenario" and lines[i].indent == inner_indent
        ),
        None,
    )

    # Build the surviving lines, then insert the new blocks by position. The
    # first version anchored each insertion on "the last line of the block",
    # which breaks exactly when that line is one the migration deleted - the
    # flat keys at the end of a source block were dropped and the carla block
    # was then never emitted.
    kept = [(i, lines[i].raw) for i in range(len(lines)) if i not in drop]
    pad = " " * inner_indent

    carla_block = []
    if carla:
        carla_block.append("{}carla:\n".format(pad))
        for name, rest in carla:
            carla_block.append("{}  {}:{}\n".format(pad, name, rest))

    runner_block = []
    if runner and scenario_at is not None:
        inner = " " * (inner_indent + 2)
        runner_block.append("{}runner:\n".format(inner))
        for name, rest in runner:
            runner_block.append("{}  {}:{}\n".format(inner, name, rest))
    elif runner:
        runner_block.append("{}scenario:\n".format(pad))
        runner_block.append("{}  runner:\n".format(pad))
        for name, rest in runner:
            runner_block.append("{}    {}:{}\n".format(pad, name, rest))

    def insert_at(original_end):
        """Output position just past the last surviving line before a boundary."""
        position = 0
        for n, (original_index, _raw) in enumerate(kept):
            if original_index < original_end:
                position = n + 1
        return position

    out = [raw for _i, raw in kept]
    # The source block's end first, then the scenario block's: inserting the
    # later position first keeps the earlier one valid.
    source_insert = insert_at(source_end)
    scenario_insert = (
        insert_at(_block_end(lines, scenario_at, inner_indent))
        if scenario_at is not None
        else None
    )
    if carla_block and (scenario_insert is None or source_insert >= scenario_insert):
        out[source_insert:source_insert] = carla_block
        if runner_block and scenario_insert is not None:
            out[scenario_insert:scenario_insert] = runner_block
        elif runner_block:
            out[source_insert:source_insert] = runner_block
    else:
        if runner_block and scenario_insert is not None:
            out[scenario_insert:scenario_insert] = runner_block
        if carla_block:
            out[source_insert:source_insert] = carla_block
        elif runner_block and scenario_insert is None:
            out[source_insert:source_insert] = runner_block

    return "".join(out), changed, unhandled


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="files or directories to rewrite")
    parser.add_argument("--check", action="store_true", help="report only; write nothing")
    args = parser.parse_args(argv)

    # Files that hold the mapping itself: rewriting them turns every entry into
    # new-to-new and the script silently stops migrating anything. It rewrote
    # its own table the first time it was run over `scripts/`.
    SELF = ("scripts/migrate_simulator_keys.py", "cadet/config/loader.py")

    targets = []
    for path in args.paths:
        if os.path.isdir(path):
            for root, _dirs, names in os.walk(path):
                if "__pycache__" in root:
                    continue
                targets += [
                    os.path.join(root, n)
                    for n in names
                    if n.endswith(YAML_SUFFIXES + SCRIPT_SUFFIXES)
                ]
        else:
            targets.append(path)

    touched, needs_hand = 0, []
    for path in sorted(set(targets)):
        if any(os.path.abspath(path).endswith(tail) for tail in SELF):
            print("{}: skipped - it defines the mapping".format(path))
            continue
        text = io.open(path, encoding="utf-8").read()
        if path.endswith(YAML_SUFFIXES):
            new_text, changed, unhandled = migrate_yaml_text(text)
            # A config's comments tell a reader which key to pass with --set,
            # and a comment naming a key that no longer exists is worse than no
            # comment. The command-line pass handles them: a dotted path in
            # prose is spelled exactly as it is on a command line.
            new_text, comment_changes = migrate_script_text(new_text)
            changed += comment_changes
            if unhandled:
                needs_hand.append((path, unhandled))
        else:
            new_text, changed = migrate_script_text(text)
        if new_text == text:
            continue
        touched += 1
        print("{}:".format(path))
        for old, new, count in changed:
            print("    {} -> {} ({}x)".format(old, new, count))
        if not args.check:
            io.open(path + ".migrated", "w", encoding="utf-8").write(new_text)
            os.replace(path + ".migrated", path)

    for path, unhandled in needs_hand:
        print("{}: needs a hand:".format(path))
        for line in unhandled:
            print("    {}".format(line))
    if needs_hand:
        return 1
    return 1 if (args.check and touched) else 0


if __name__ == "__main__":
    sys.exit(main())
