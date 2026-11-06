#!/usr/bin/env python3
"""Generate docs/config-reference.md from the CADET config schema.

What this is for:
    A single, re-runnable source of truth for the config reference. It introspects
    the pydantic models in :mod:`cadet.config.schema` and emits one Markdown
    section per model (fields, types, defaults, required-ness), marking options
    that are declared but not built or not yet verified (the three tiers in
    :mod:`cadet.config.implemented`).

How to run:
    python scripts/gen_config_reference.py           # writes docs/config-reference.md
    python scripts/gen_config_reference.py --check    # fail if the file is stale

Run it after changing the schema; CI (later) runs ``--check`` to keep docs honest.
"""

import argparse
import enum
import os
import sys
import typing

import typing_extensions as te

# Make the repo importable when run from anywhere.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from pydantic import BaseModel  # noqa: E402

from cadet.config import schema as S  # noqa: E402
from cadet.config.implemented import EXPERIMENTAL, IMPLEMENTED  # noqa: E402

SEMANTICS = """\
## Link backend semantics

`links[*].network.mode` selects *where* a link is emulated, and that changes what
the other fields mean. The two families are not interchangeable and neither is
made to imitate the other.

| | `delay` | `netem`, `netwaggle` |
|---|---|---|
| layer | **message-level**: the transport's delay line holds whole application messages | **packet-level**: a kernel qdisc holds packets, and TCP reacts |
| `distribution` | sampled per message, per the configured distribution | the mean and jitter of a netem delay, applied per packet |
| `direction: forward` | delays request messages only; replies are untouched | delays *every* packet leaving the sender, including the handshake and ACKs |
| `direction: reverse` | delays reply messages only | delays every packet leaving the listener |
| `loss` | **drops application messages**; a dropped message is never answered | **drops packets**; TCP retransmits, so the message still arrives |
| `bandwidth_mbps` | serialization delay computed per message | a real rate limit on the queue |
| needs | nothing | `netem`: NET_ADMIN. `netwaggle`: the emulation container |

### What this means in practice

**A configured delay is reproduced at the message level, and at the packet
level once Nagle is out of the way.** Measured on a 60 ms forward link with the
reference app at 20 fps: `delay` gave p50 60.5 ms, while `netem` and
`netwaggle` gave p50 ≈ 150 ms. That is not emulator error — an isolated
request/response probe over the same qdisc measures 60.1 ms. It is TCP: with
small paced messages, Nagle holds a write until the previous one is
acknowledged, and the acknowledgement is itself behind the configured delay.

**`tcp_nodelay` is needed on both ends, not just the sender.** Each end holds
its own small writes, so each contributes part of the error. The same run, with
the option set on neither node, on one, and on both:

| `tcp_nodelay` | p50 | min |
|---|---|---|
| neither | 150.7 ms | 99.5 ms |
| client only | 110.4 ms | 68.6 ms |
| server only | 90.9 ms | 60.6 ms |
| both | **60.5 ms** | 60.3 ms |

**The send rate decides whether it matters at all.** Nagle only has something
to wait for while a previous small write is unacknowledged, so the error
appears when the send interval approaches the shaped round trip. The same
60 ms link, `tcp_nodelay` unset, swept across that boundary:

| send interval | p50 |
|---|---|
| 200 ms (5 fps) | 60.5 ms |
| 100 ms (10 fps) | 60.5 ms |
| 71 ms (14 fps) | 143.2 ms (mixed: `min` still 60.4 ms) |
| 50 ms (20 fps) | 150.2 ms |
| 33 ms (30 fps) | 150.7 ms |

So a packet-level measurement taken at a send interval comfortably above the
shaped round trip needs nothing; one taken near or below it, without
`tcp_nodelay` on both ends, measures the link plus an extra round trip. The
message-level backend is unaffected either way — it holds whole application
messages in process, so TCP's write batching never enters.

**Configured loss means different things.** `delay` loses messages, so a 1% loss
produced 1.25% unanswered frames. `netem` and `netwaggle` lose packets under a
reliable transport, so the same 1% produced **0% unanswered** and showed up as
retransmission latency in p99 instead. If an experiment needs observable message
loss, it needs `delay` or an unreliable transport.

**Choose by question, not by fidelity.** Use `delay` to state a per-message
latency distribution and have it honoured. Use `netem` or `netwaggle` to ask what
a real protocol stack does on a delayed, lossy, rate-limited path. `netwaggle`
additionally gives a real topology, a wireless medium and propagation model, and
hosts in separate namespaces.

Every run writes `link_report_<node>.json` per client, next to its metrics,
holding the configured
parameters beside the measured per-message latency and unanswered counts, so the
gap is visible per run rather than assumed.

## Shared media: `connectors`

A `links` entry is point to point: one `from`, one `to`, its own L2 segment and
its own qdiscs. Some media are not that shape. Several stations associating with
one access point share one medium, sit in **one subnet**, and reach each other
over the air with no L3 hop and no route between them - which is exactly what
two point-to-point links in series cannot express.

A `connectors` entry is that shape. Nodes name the connector they attach to
(`nodes[*].connector`) and, on an access point, where they sit
(`nodes[*].position`, `"x,y,z"` in metres). Two types:

| `type` | what it is | its members |
|---|---|---|
| `ap` | an access point, with the cell's radio parameters in `wifi` and its own `position` | stations, each at its own position, which is what the propagation model acts on |
| `switch` | the same idea without radios: a wired shared segment | wired hosts, attached with a plain veth |

Three things follow, and they are the point rather than side effects.

**A connector runs nothing.** It is deliberately not a `nodes` entry: there is
no role to dispatch on it and no address on it to bind. In Mininet-WiFi an
access point is a switch in the root namespace, so a role process placed on one
dies with `Cannot assign requested address`; keeping the two sections apart is
what makes that config unexpressible rather than merely discouraged.

**A cell's members share a subnet and get no route to each other.** Each member
carries the shaped subnet's own prefix on its connector-facing interface, so the
others are on-link. A point-to-point endpoint still gets a `/32` plus a host
route to its peer, and the two coexist in one subnet by longest match, so a node
on a cell *and* on a link sends each peer's traffic out of the right interface.

**A connector is shaped by its medium, not by a qdisc.** A cell's loss and delay
come from the propagation model and the distances between positions, which is
why such an experiment varies the distances. A connector therefore has no
`distribution`, `loss` or `bandwidth_mbps`; a path that needs those is a `links`
entry. Both sections can appear in one config, and a config that mentions no
connector behaves exactly as it did before they existed.

Connectors exist only in the `netwaggle` backend, like a `netwaggle` link: under
`cadet run` the topology is not built and the nodes talk over the host's own
stack.

"""

_LITERAL_ORIGINS = {te.Literal, getattr(typing, "Literal", None)}

OUT_PATH = os.path.join(REPO, "docs", "config-reference.md")

# Top-level entry model; the generator walks outward from here.
ROOT = S.Config

# Map a path in IMPLEMENTED to the schema field it constrains, so we can annotate
# "planned" values inline in the reference.
_IMPL_BY_FIELD = {
    ("Config", "transport"): None,
    ("TransportConfig", "protocol"): "transport.protocol",
    ("TransportConfig", "mode"): "transport.mode",
    ("TransportConfig", "encoding"): "transport.encoding",
    ("MetricsConfig", "collectors"): "metrics.collectors",
    ("CarlaSource", "sync_mode"): "source.sync_mode",
    ("Node", "launcher"): "nodes.launcher",
    ("NetworkModel", "mode"): "links.network.mode",
    ("AnalysisConfig", "backend"): "analysis.backend",
}


def _literal_values(annot):
    """Return the scalar values of a Literal (recursing through list[Literal])."""
    origin = te.get_origin(annot)
    if origin in _LITERAL_ORIGINS:
        return list(te.get_args(annot))
    for arg in te.get_args(annot):
        vals = _literal_values(arg)
        if vals:
            return vals
    return []


def _type_name(annot, accepted=None) -> str:
    """A compact human-readable name for a type annotation.

    ``accepted`` limits the values printed for a gated enumeration to the ones
    a run may ask for; see :func:`_accepted_values`.
    """
    origin = te.get_origin(annot)
    if origin in _LITERAL_ORIGINS:
        values = accepted if accepted else te.get_args(annot)
        return "one of {{{}}}".format(", ".join(repr(a) for a in values))
    if origin is typing.Union:
        args = [a for a in te.get_args(annot) if a is not type(None)]
        inner = " | ".join(_type_name(a) for a in args)
        optional = type(None) in te.get_args(annot)
        return "{} (optional)".format(inner) if optional else inner
    if origin in (list, typing.List):
        return "list[{}]".format(_type_name(te.get_args(annot)[0]))
    if origin in (dict, typing.Dict):
        k, v = te.get_args(annot)
        return "map[{} -> {}]".format(_type_name(k), _type_name(v))
    if isinstance(annot, type) and issubclass(annot, BaseModel):
        return annot.__name__
    if isinstance(annot, type) and issubclass(annot, enum.Enum):
        return "enum"
    return getattr(annot, "__name__", str(annot).replace("typing.", ""))


def _nested_models(annot):
    """Yield BaseModel subclasses referenced by a type annotation (shallow)."""
    if isinstance(annot, type) and issubclass(annot, BaseModel):
        yield annot
        return
    for arg in typing.get_args(annot):
        yield from _nested_models(arg)


def _accepted_values(key: str, allowed):
    """Return the values of ``allowed`` that ``cadet validate`` accepts.

    The reference documents what CADET will run, so a value the gate refuses is
    left out of the field's type rather than listed with a label saying it
    cannot be used. The gate stays the authority either way: a config naming a
    value that is not here is refused by name.
    """
    accepted = set(IMPLEMENTED.get(key, set())) | set(EXPERIMENTAL.get(key, set()))
    kept = [value for value in allowed if value in accepted]
    return kept or list(allowed)


def _gate_key(model_name: str, field_name: str):
    """Return the gate path constraining this field, when the gate names one."""
    key = _IMPL_BY_FIELD.get((model_name, field_name))
    if key is None or (key not in IMPLEMENTED and key not in EXPERIMENTAL):
        return None
    return key


def _default_str(field) -> str:
    if field.is_required():
        return "— (required)"
    default = field.default
    factory = field.default_factory
    if factory is not None:
        try:
            default = factory()
        except Exception:
            return "(factory)"
    if isinstance(default, BaseModel):
        return "(defaults)"
    return "`{}`".format(default)


def _render_model(model, seen, lines):
    if model.__name__ in seen:
        return
    seen.add(model.__name__)
    doc = (model.__doc__ or "").strip().split("\n")[0]
    lines.append("### `{}`".format(model.__name__))
    if doc:
        lines.append("")
        lines.append(doc)
    lines.append("")
    lines.append("| field | type | default | notes |")
    lines.append("|---|---|---|---|")
    children = []
    for name, field in model.model_fields.items():
        alias = field.alias or name
        annot = field.annotation
        key = _gate_key(model.__name__, name)
        type_name = _type_name(annot, _accepted_values(key, _literal_values(annot)) if key else None)
        lines.append(
            "| `{}` | {} | {} | {} |".format(alias, type_name, _default_str(field), "")
        )
        children.extend(_nested_models(annot))
    lines.append("")
    for child in children:
        _render_model(child, seen, lines)


def generate() -> str:
    lines = [
        "# CADET config reference",
        "",
        "Generated from `cadet/config/schema.py` by `scripts/gen_config_reference.py`.",
        "Do not edit by hand — re-run the script.",
        "",
        "Each enumerated field lists the values a run may ask for. `cadet",
        "validate` names anything else a config asks for and refuses the run.",
        "",
    ]
    # The field tables alone cannot convey that the same field means different
    # things per backend, so the semantics section is emitted ahead of them.
    lines.extend(SEMANTICS.splitlines())
    _render_model(ROOT, set(), lines)
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if docs are stale")
    args = ap.parse_args()
    content = generate()
    if args.check:
        existing = open(OUT_PATH).read() if os.path.exists(OUT_PATH) else ""
        if existing != content:
            print(
                "config-reference.md is stale; run scripts/gen_config_reference.py", file=sys.stderr
            )
            return 1
        print("config-reference.md is up to date")
        return 0
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        fh.write(content)
    print("wrote {}".format(OUT_PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
