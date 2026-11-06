"""netem backend: shape a link's traffic with tc inside the sending node.

What this is for:
    Implements ``links[*].network.mode: netem``. Where the ``netwaggle`` backend
    builds a whole emulated network, this one changes nothing about the topology:
    it installs a ``tc`` queueing discipline on the sending node's own interface
    and filters it to the link peer's address and port, so only that flow is
    delayed, dropped and rate-limited. Everything else on the same interface
    (channels, the simulator connection, log shipping) stays in the unshaped
    default class.

    That is what makes this the simple option: no namespaces, no switches, no
    privileged emulation container. A node's container needs only NET_ADMIN.

    This is **packet-level** emulation. The qdisc holds packets, so the configured
    delay applies to every packet leaving the sender on this flow (data, the
    handshake and ACKs alike), and ``loss`` drops packets that TCP retransmits
    rather than application messages. A request/response workload therefore pays
    the configured delay more than once unless Nagle is disabled
    (``nodes.*.transport_tuning.tcp_nodelay``), and configured loss shows up as
    p99 latency rather than as unanswered messages. That is the transport
    reacting to the link, which is the point of emulating at this layer; the
    message-level alternative is ``mode: delay``. See "Link backend semantics" in
    ``docs/config-reference.md``.

What it contains:
    - :class:`ShapeRule`: one flow to shape (peer address, which port to match,
      and the :class:`~cadet.network.shaping.Shaping` to apply).
    - :func:`rules_for_node`: the rules a given node must install, derived from
      the config's links and routes and that node's direction of each. A
      composed route is one rule on the node at the start of the path, carrying
      the hops' composed shaping and matching the far end: the connection is end
      to end, so no packet of it ever leaves a waypoint.
    - :func:`build_commands` / :func:`teardown_commands`: the ``tc`` command
      lines, as argv lists.
    - :class:`NetemBackend`: applies and removes them through a caller-supplied
      runner, so the same code works locally or over ``docker exec``.
    - :func:`local_runner`, :func:`interface_for` and :func:`sessions_for_config`:
      the glue the orchestrator uses to install the rules in the right place.

    Nothing here executes a command on import, and no function runs ``tc``
    itself: a backend returns argv and the orchestrator's launcher runs it in the
    right place. That keeps the whole mapping testable without root.

How it fits into CADET:
    The orchestrator asks :func:`rules_for_node` what each node needs, then uses
    :class:`NetemBackend` to apply before the role starts and remove on
    teardown. Ordering matters: apply before traffic flows, remove even when the
    run fails, which is why teardown is a single idempotent qdisc delete.

How to configure it:
    Per link via ``links[*].network`` with ``mode: netem``. Delay, jitter and
    loss come from the distribution and ``loss``; ``bandwidth_mbps`` becomes the
    shaped class's rate.
"""

import subprocess
from typing import Callable, Dict, List, Optional, Tuple

from cadet.config.routes import hops_in_direction, route_hops
from cadet.network.shaping import Shaping, shaping_from_path
from cadet.network.shaping import shaping_from_network as _shaping_from_network

NETEM_MODE = "netem"

# Handle numbers for the qdisc tree installed on a shaped interface. One root
# htb with a wide-open default class, and one shaped class per rule.
ROOT_HANDLE = "1:"
DEFAULT_CLASSID = "1:999"
FIRST_SHAPED_MINOR = 10

# Rate given to the default (unshaped) class and to shaped classes that set no
# bandwidth. htb requires a rate on every class, so this stands in for
# "unlimited"; it is far above any link this emulates, so it never binds.
UNLIMITED_RATE = "10gbit"

# Which hook a rule's qdisc hangs from. A node shapes what it sends on its own
# egress; a direction whose sender is on another machine is shaped on the
# receiver's ingress instead, which is the subject of :func:`ingress_commands`.
EGRESS = "egress"
INGRESS = "ingress"

# The ingress qdisc's fixed handle, which is what `tc qdisc del dev X ingress`
# removes. One per interface, so the name needs no run in it.
INGRESS_HANDLE = "ffff:"


def ifb_name(interface: str) -> str:
    """Return the intermediate device that carries ``interface``'s ingress.

    An interface's ingress cannot hold a shaping tree directly - the qdisc there
    classifies and redirects, it does not queue - so the tree lives on an
    intermediate functional block device and the ingress hook mirrors traffic to
    it. The name is derived from the interface so it is the same on every call
    and distinct per interface, and kept inside the kernel's 15-character limit.
    """
    return "ifb-{}".format(interface)[:15]


class ShapeRule:
    """One flow to shape on a node's interface.

    ``match`` says which port identifies the flow:

    ``"dport"``
        the peer's listening port. Used by the side that initiates; a client's
        requests go to ``server:port``.
    ``"sport"``
        the local listening port. Used by the listener for its replies, which
        leave from ``server:port`` towards an ephemeral client port, so the
        destination port cannot identify them.

    ``hook`` says which direction of the interface the rule shapes.
    ``"egress"`` matches the address as the destination, which is what a node
    sending to a peer needs. ``"ingress"`` matches it as the source, for a
    direction whose sender runs on another machine: that sender's own interface
    is that machine's, and shaping it would shape everything else on it, so the
    receiver shapes what arrives from that address instead. Per-direction
    latency, loss and rate are what the link states either way.
    """

    def __init__(
        self,
        link_name: str,
        peer_ip: str,
        port: int,
        shaping: Shaping,
        match: str = "dport",
        hook: str = EGRESS,
    ):
        if match not in ("dport", "sport"):
            raise ValueError("match must be 'dport' or 'sport', got {!r}".format(match))
        if hook not in (EGRESS, INGRESS):
            raise ValueError(
                "hook must be {!r} or {!r}, got {!r}".format(EGRESS, INGRESS, hook)
            )
        self.link_name = link_name
        self.peer_ip = peer_ip
        self.port = port
        self.shaping = shaping
        self.match = match
        self.hook = hook

    def __repr__(self) -> str:
        return "ShapeRule({} {} {}:{} {} {})".format(
            self.link_name, self.hook, self.peer_ip, self.port, self.match, self.shaping
        )


def _resolve_ip(host: str) -> str:
    """Map a config host value to an address usable in a tc filter.

    ``localhost`` and the empty host both mean the loopback address here; a tc
    filter needs a literal address, and a name would not match.
    """
    if host in ("localhost", ""):
        return "127.0.0.1"
    return host


def rules_for_node(config, node_name: str, offmachine=()) -> List[ShapeRule]:
    """Return the shaping rules ``node_name`` must install on its interface.

    A link's ``direction`` decides which side shapes. ``forward`` shapes the
    request path, so the rule belongs on the ``from`` node and matches the
    peer's listening port. ``reverse`` shapes replies, so it belongs on the
    ``to`` node and matches its own listening port as the source. ``both``
    installs a rule on each side.

    ``offmachine`` names the nodes whose own interface is another machine's. A
    direction whose sender is one of them cannot be shaped where it is sent:
    that interface carries everything else on that machine. Such a direction is
    shaped where it arrives instead - an ingress rule on the node at the other
    end, matching the sender's address as the source - so every qdisc stays on
    an interface this orchestrator owns. The flow matched is the same flow, and
    the link's latency, loss and rate are what the configuration states.

    Nodes named only by ``channels`` get no rules at all, which is how channel
    traffic stays unshaped: there is nothing to match it.
    """
    offmachine = set(offmachine or ())
    by_name = {n.name: n for n in config.nodes}
    rules: List[ShapeRule] = []

    # A composed route replaces its hops: one connection, one sender, one qdisc.
    # A relayed route does not, because each of its hops is a real connection
    # between two processes and is shaped as the link it is.
    composed = [
        route
        for route in (getattr(config, "routes", None) or [])
        if route.realise == "compose"
    ]
    hops_of_composed = {name for route in composed for name in route.hops}

    for route in composed:
        hops = route_hops(config, route)
        if any(hop.network.mode != NETEM_MODE for hop in hops):
            continue
        for direction, shaper, listener in (
            ("forward", route.from_, route.to),
            ("reverse", route.to, route.to),
        ):
            acting = hops_in_direction(hops, direction)
            if not acting or node_name != shaper:
                continue
            shaping = shaping_from_path([hop.network for hop in acting])
            if shaping.is_empty:
                continue
            listen_port = getattr(by_name.get(listener), "port", None)
            if listen_port is None:
                raise ValueError(
                    "route {!r} shapes the {} path but node {!r} declares no "
                    "port to match".format(route.name, direction, listener)
                )
            peer = route.to if direction == "forward" else route.from_
            rules.append(
                ShapeRule(
                    link_name=route.name,
                    peer_ip=_resolve_ip(by_name[peer].host),
                    port=listen_port,
                    shaping=shaping,
                    match="dport" if direction == "forward" else "sport",
                )
            )

    for link in config.links:
        if link.name in hops_of_composed:
            continue
        if link.network.mode != NETEM_MODE:
            continue
        shaping = shaping_from_network(link.network)
        if shaping.is_empty:
            continue
        direction = getattr(link, "direction", "forward")
        sender, listener = link.from_, link.to
        listen_port = getattr(by_name.get(listener), "port", None)

        # (which direction, who sends it, who receives it, the port match)
        for wanted, shaper, receiver, match in (
            ("forward", sender, listener, "dport"),
            ("reverse", listener, sender, "sport"),
        ):
            if direction not in (wanted, "both"):
                continue
            if listen_port is None:
                raise ValueError(
                    "link {!r} shapes the {} path but node {!r} declares no "
                    "port to match".format(link.name, wanted, listener)
                )
            if shaper in offmachine:
                # The sender is on another machine: shape it where it arrives.
                if node_name != receiver:
                    continue
                rules.append(
                    ShapeRule(
                        link_name=link.name,
                        peer_ip=_resolve_ip(by_name[shaper].host),
                        port=listen_port,
                        shaping=shaping,
                        match=match,
                        hook=INGRESS,
                    )
                )
                continue
            if node_name != shaper:
                continue
            rules.append(
                ShapeRule(
                    link_name=link.name,
                    peer_ip=_resolve_ip(by_name[receiver].host),
                    port=listen_port,
                    shaping=shaping,
                    match=match,
                )
            )
    return rules


# The mapping from a configured distribution to kernel parameters is shared
# with the netwaggle backend; see cadet.network.shaping for why netem's jitter
# needs a rate to stay in order.
shaping_from_network = _shaping_from_network


def egress_rules(rules: List[ShapeRule]) -> List[ShapeRule]:
    """The rules shaping what the node sends."""
    return [rule for rule in rules if rule.hook == EGRESS]


def ingress_rules(rules: List[ShapeRule]) -> List[ShapeRule]:
    """The rules shaping what arrives from a sender on another machine."""
    return [rule for rule in rules if rule.hook == INGRESS]


def ingress_commands(interface: str, rules: List[ShapeRule]) -> List[List[str]]:
    """Return the commands shaping what arrives on ``interface``.

    An interface's ingress hook classifies and redirects; it cannot queue, so
    the shaping tree lives on an intermediate functional block device and every
    arriving packet is mirrored to it. The tree there is the same htb root, wide
    default class and netem leaf per rule that egress uses - what differs is
    only that a rule matches its peer as the *source*, because these packets are
    arriving from it.

    The device is created here rather than assumed: it lives in the namespace of
    the node's own container, so it appears with the container and goes with it.
    Returns an empty list when there is nothing to install.
    """
    if not rules:
        return []
    device = ifb_name(interface)
    commands: List[List[str]] = [
        ["ip", "link", "add", "name", device, "type", "ifb"],
        ["ip", "link", "set", "dev", device, "up"],
        ["tc", "qdisc", "add", "dev", interface, "handle", INGRESS_HANDLE, "ingress"],
        # Everything arriving is mirrored to the device, where the filters below
        # decide what is shaped; unmatched traffic lands in the default class and
        # is untouched, exactly as on egress.
        [
            "tc", "filter", "add", "dev", interface,
            "parent", INGRESS_HANDLE,
            "protocol", "ip", "prio", "1", "u32",
            "match", "u32", "0", "0",
            "action", "mirred", "egress", "redirect", "dev", device,
        ],
    ]
    return commands + _tree_commands(device, rules)


def build_commands(interface: str, rules: List[ShapeRule]) -> List[List[str]]:
    """Return the ``tc`` commands installing ``rules`` on ``interface``.

    The tree is an htb root whose default class is effectively unlimited, plus
    one shaped class per rule with a netem leaf. Unmatched traffic lands in the
    default class and is untouched, so channels sharing the interface are not
    shaped. Returns an empty list when there is nothing to install.

    Egress rules only: what arrives from a sender on another machine is shaped
    by :func:`ingress_commands`, which puts the same tree on an intermediate
    device.
    """
    rules = egress_rules(rules)
    if not rules:
        return []
    return _tree_commands(interface, rules)


def _tree_commands(device: str, rules: List[ShapeRule]) -> List[List[str]]:
    """Return the htb-plus-netem tree for ``rules`` on ``device``."""
    interface = device

    commands: List[List[str]] = [
        # default <minor> routes unclassified traffic to the unshaped class.
        [
            "tc",
            "qdisc",
            "add",
            "dev",
            interface,
            "root",
            "handle",
            ROOT_HANDLE,
            "htb",
            "default",
            DEFAULT_CLASSID.split(":")[1],
        ],
        [
            "tc",
            "class",
            "add",
            "dev",
            interface,
            "parent",
            ROOT_HANDLE,
            "classid",
            DEFAULT_CLASSID,
            "htb",
            "rate",
            UNLIMITED_RATE,
        ],
    ]

    for index, rule in enumerate(rules):
        minor = FIRST_SHAPED_MINOR + index
        classid = "1:{}".format(minor)
        rate = (
            "{}mbit".format(rule.shaping.bandwidth_mbps)
            if rule.shaping.bandwidth_mbps
            else UNLIMITED_RATE
        )
        commands.append(
            [
                "tc",
                "class",
                "add",
                "dev",
                interface,
                "parent",
                ROOT_HANDLE,
                "classid",
                classid,
                "htb",
                "rate",
                rate,
            ]
        )
        netem_args = rule.shaping.netem_args()
        if netem_args:
            commands.append(
                [
                    "tc",
                    "qdisc",
                    "add",
                    "dev",
                    interface,
                    "parent",
                    classid,
                    "handle",
                    "{}:".format(minor * 10),
                    "netem",
                ]
                + netem_args
            )
        commands.append(
            [
                "tc",
                "filter",
                "add",
                "dev",
                interface,
                "protocol",
                "ip",
                "parent",
                ROOT_HANDLE,
                "prio",
                str(index + 1),
                "u32",
                "match",
                "ip",
                # The peer is the destination of what this node sends, and the
                # source of what arrives from it.
                "dst" if rule.hook == EGRESS else "src",
                "{}/32".format(rule.peer_ip),
                "match",
                "ip",
                rule.match,
                str(rule.port),
                "0xffff",
                "flowid",
                classid,
            ]
        )
    return commands


def teardown_commands(interface: str, ingress: bool = False) -> List[List[str]]:
    """Return the commands removing everything this backend installed.

    Deleting a root qdisc takes the classes, leaf qdiscs and filters with it.
    ``--force`` is not used; instead each delete tolerates its object being
    absent, so teardown is idempotent and safe to run after a failed apply.

    ``ingress`` adds the removal of the ingress hook and its intermediate
    device, in that order, because the device cannot go while traffic is still
    redirected to it. Asked for rather than always emitted, so a node that
    shaped only what it sends tears down exactly what it built.
    """
    commands = [["tc", "qdisc", "del", "dev", interface, "root"]]
    if ingress:
        device = ifb_name(interface)
        commands += [
            ["tc", "qdisc", "del", "dev", interface, "ingress"],
            ["tc", "qdisc", "del", "dev", device, "root"],
            ["ip", "link", "del", device],
        ]
    return commands


class NetemBackend:
    """Applies and removes netem shaping through a caller-supplied runner.

    ``runner`` takes an argv list and returns ``(returncode, output)``. The
    orchestrator supplies one that runs locally or inside the node's container,
    so this class never decides where a command executes.
    """

    def __init__(
        self,
        interface: str,
        rules: List[ShapeRule],
        runner: Callable[[List[str]], object],
    ):
        self.interface = interface
        self.rules = rules
        self.runner = runner
        self._applied = False

    def apply(self) -> List[str]:
        """Install the qdisc tree. Returns the commands run, for the run record.

        Any failure removes what was already installed, so a partially built
        tree never shapes traffic in a way the config did not ask for.
        """
        if not self.rules:
            return []
        run_log: List[str] = []
        commands = build_commands(self.interface, self.rules) + ingress_commands(
            self.interface, self.rules
        )
        for argv in commands:
            code, output = self._invoke(argv)
            run_log.append(" ".join(argv))
            if code != 0:
                self.remove()
                raise RuntimeError("netem setup failed: {} -> {}".format(" ".join(argv), output))
        self._applied = True
        return run_log

    def remove(self) -> None:
        """Remove the qdisc tree; tolerates it being absent already."""
        has_ingress = bool(ingress_rules(self.rules))
        for argv in teardown_commands(self.interface, ingress=has_ingress):
            self._invoke(argv)
        self._applied = False

    @property
    def applied(self) -> bool:
        """True once :meth:`apply` has completed successfully."""
        return self._applied

    def _invoke(self, argv: List[str]):
        result = self.runner(argv)
        if isinstance(result, tuple):
            return result
        # A runner may return just a return code.
        return result, ""

    def __enter__(self) -> "NetemBackend":
        self.apply()
        return self

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
        self.remove()
        return None


# --- orchestrator glue ----------------------------------------------------
def local_runner(argv: List[str]) -> Tuple[int, str]:
    """Run a tc command on this machine. The default runner for local runs."""
    result = subprocess.run(argv, capture_output=True, text=True)
    return result.returncode, (result.stderr or result.stdout).strip()


def interface_for(ip: str, runner: Optional[Callable[[List[str]], object]] = None) -> str:
    """Return the interface the kernel would use to reach ``ip``.

    Asks the routing table rather than guessing, because the answer differs by
    deployment: a local run between two processes on one machine leaves by the
    loopback, while a containerised node leaves by its bridge interface. Falls
    back to ``lo`` when the lookup cannot be parsed, since that is the only
    sensible default for an address on this host.
    """
    run = runner or local_runner
    code, output = run(["ip", "route", "get", ip])
    if code == 0:
        fields = output.split()
        if "dev" in fields:
            return fields[fields.index("dev") + 1]
    return "lo"


def sessions_for_config(
    config,
    node_names: Optional[List[str]] = None,
    runner: Optional[Callable[[List[str]], object]] = None,
    interface_resolver: Optional[Callable[[str], str]] = None,
    offmachine=(),
) -> List[NetemBackend]:
    """Build one :class:`NetemBackend` per (node, interface) the config needs.

    Rules are grouped by the interface that reaches their peer, because a node
    may hold links whose peers are reached by different interfaces and each
    interface carries its own qdisc tree. Returns an empty list when no link
    uses ``mode: netem``, so the caller can treat it as "nothing to do".

    ``offmachine`` names the nodes running on another machine; a direction they
    send is shaped where it arrives. See :func:`rules_for_node`.
    """
    run = runner or local_runner
    resolve = interface_resolver or (lambda ip: interface_for(ip, run))
    names = node_names if node_names is not None else [n.name for n in config.nodes]

    grouped: Dict[str, List[ShapeRule]] = {}
    for node_name in names:
        for rule in rules_for_node(config, node_name, offmachine=offmachine):
            grouped.setdefault(resolve(rule.peer_ip), []).append(rule)

    return [NetemBackend(intf, rules, run) for intf, rules in sorted(grouped.items())]


def apply_sessions(sessions: List[NetemBackend]) -> List[str]:
    """Apply every session, rolling back all of them if any one fails.

    Partial shaping is worse than none: it would quietly measure a different
    experiment from the configured one. On failure every session already applied
    is removed before the error propagates.
    """
    applied: List[NetemBackend] = []
    log: List[str] = []
    try:
        for session in sessions:
            log.extend(session.apply())
            applied.append(session)
    except Exception:
        for session in reversed(applied):
            session.remove()
        raise
    return log


def remove_sessions(sessions: List[NetemBackend]) -> None:
    """Remove every session; tolerates sessions that were never applied."""
    for session in reversed(sessions):
        session.remove()
