"""Generator: a CADET config becomes an emulated-network topology.

What this is for:
    Translates the declarative part of a CADET experiment — ``nodes``,
    ``connectors``, ``links`` and ``channels`` — into the
    :class:`~cadet.network.netwaggle.topology.Topology` the emulator builds. This
    is the single place that decides what enters the emulated network and what
    stays out of it.

What it contains:
    - :class:`UnsupportedLink`: raised when a link asks for something the
      emulator cannot express, rather than quietly approximating it.
    - :func:`shaping_for`: one link's network model becomes per-direction
      shaping.
    - :func:`generate`: the whole config becomes a topology.

How it fits into CADET:
    The orchestrator calls :func:`generate` for any run whose config has a link
    with ``network.mode: netwaggle``, hands the result to the emulator, and
    applies :meth:`Topology.address_map` to the run config so role processes use
    the shaped addresses.

    The design rule the advisor set — only traffic that mimics a real network
    flow crosses the emulator — is enforced structurally here, not by
    convention:

    * a node that is an endpoint of a ``link``, or that attaches to a
      ``connector``, becomes an emulated host;
    * a ``connector`` becomes infrastructure and never a host: it runs no role,
      so there is nothing to dispatch on it;
    * a node that appears only in ``channels`` is never added to the topology,
      so it has no emulated path at all;
    * every emulated host also gets an unshaped bypass interface that carries
      the default route, so channel traffic from a host that *does* have a link
      still leaves without being shaped.

How to configure it:
    Nothing here is configured directly. It reads ``config.links[*].network``
    (mode, medium, distribution, loss, bandwidth, wifi), ``config.connectors[*]``
    (type, position, wifi) and ``config.nodes[*]`` (including ``connector`` and
    ``position``).
"""

from typing import Dict, List, Tuple

from cadet.network.netwaggle.topology import (
    ALIAS_PREFIX,
    EmulatedConnector,
    EmulatedHost,
    EmulatedLink,
    EmulatedPath,
    Shaping,
    Topology,
    WifiNode,
    WifiSpec,
)
from cadet.network.shaping import UnrepresentableDelay, shaping_from_network

# Mode marking a link as belonging to this backend.
NETWAGGLE_MODE = "netwaggle"


class UnsupportedLink(Exception):
    """A link requests behaviour the emulated backend cannot reproduce."""


def _shaping_from_network(net) -> Shaping:
    """Build one direction's :class:`Shaping` from a link's network model.

    The mapping itself is shared with the ``netem`` backend so the two cannot
    drift apart; see :mod:`cadet.network.shaping`. Its refusals are re-raised as
    :class:`UnsupportedLink` so a caller already handling emulated-topology
    problems catches this one too.
    """
    try:
        return shaping_from_network(net)
    except UnrepresentableDelay as exc:
        raise UnsupportedLink(str(exc)) from exc


def shaping_for(link) -> Tuple[Shaping, Shaping]:
    """Return ``(forward, reverse)`` shaping for a CADET link.

    ``direction`` decides which side carries the qdisc. netem shapes egress, so
    delaying only requests means a qdisc on the sending endpoint's interface and
    nothing on the receiver's — which is why the two directions are kept apart
    instead of handing Mininet one symmetric parameter set.
    """
    shaped = _shaping_from_network(link.network)
    direction = getattr(link, "direction", "forward")
    if direction == "forward":
        return shaped, Shaping()
    if direction == "reverse":
        return Shaping(), shaped
    if direction == "both":
        # A fresh object per side: the emulator mutates neither, but sharing one
        # instance across directions would make later edits act on both.
        return shaped, _shaping_from_network(link.network)
    raise UnsupportedLink("unknown link direction {!r}".format(direction))


def _wifi_from_network(link, net) -> WifiSpec:
    """Build a :class:`WifiSpec` from a link whose medium is wireless."""
    wifi = getattr(net, "wifi", None)
    if wifi is None:
        raise UnsupportedLink(
            "link {!r} has network.medium: wifi but no network.wifi block".format(link.name)
        )
    nodes = {
        name: WifiNode(role=entry.role, position=entry.position)
        for name, entry in wifi.nodes.items()
    }
    endpoints = {link.from_, link.to}
    declared = set(nodes)
    if not declared.issubset(endpoints):
        raise UnsupportedLink(
            "link {!r} declares wifi roles for {} which are not its endpoints {}".format(
                link.name, sorted(declared - endpoints), sorted(endpoints)
            )
        )
    if declared != endpoints:
        raise UnsupportedLink(
            "link {!r} must declare a wifi role for both endpoints; missing {}".format(
                link.name, sorted(endpoints - declared)
            )
        )
    return WifiSpec(
        nodes=nodes,
        ssid=wifi.ssid,
        mode=wifi.mode,
        channel=wifi.channel,
        propagation_model=wifi.propagation_model,
        propagation_exp=wifi.propagation_exp,
        propagation_params=dict(wifi.propagation_params),
        mobility_trace=wifi.mobility_trace,
    )


def _radio_for(conn) -> WifiSpec:
    """Build the cell-wide :class:`WifiSpec` of an access-point connector.

    ``nodes`` stays empty on purpose: in a cell the access point is the
    connector and each station's position lives on its own host, so there are no
    per-endpoint radio roles to carry. A connector with no ``wifi`` block gets
    the medium defaults, which is the same set the schema would have filled in.
    """
    wifi = getattr(conn, "wifi", None)
    if wifi is None:
        return WifiSpec()
    return WifiSpec(
        ssid=wifi.ssid,
        mode=wifi.mode,
        channel=wifi.channel,
        propagation_model=wifi.propagation_model,
        propagation_exp=wifi.propagation_exp,
        propagation_params=dict(wifi.propagation_params),
        mobility_trace=wifi.mobility_trace,
    )


def emulated_node_names(config) -> List[str]:
    """Return the node names that belong in the emulated topology, in order.

    Two things qualify: being an endpoint of a ``netwaggle`` link, and attaching
    to a ``connector``. A connector exists only in this backend, so attaching to
    one is itself the statement that the node is emulated - which is how the
    Wi-Fi example has a topology with no ``links`` section at all.

    Nodes that appear solely in ``channels`` are deliberately absent: with no
    emulated host they have no emulated path, which is how plumbing is kept off
    the emulator.
    """
    names: List[str] = []
    for link in config.links:
        if link.network.mode != NETWAGGLE_MODE:
            continue
        for endpoint in (link.from_, link.to):
            if endpoint not in names:
                names.append(endpoint)
    for node in config.nodes:
        if getattr(node, "connector", None) and node.name not in names:
            names.append(node.name)
    return names


def generate(
    config,
    link_subnet: str = "10.60.0.0/24",
    bypass_subnet: str = "10.61.0.0/24",
) -> Topology:
    """Turn a CADET ``Config`` into a :class:`Topology`.

    Raises :class:`UnsupportedLink` if any ``netwaggle`` link asks for something
    netem or Mininet-WiFi cannot express. Returns a topology with no hosts and
    no links when the config has neither a ``netwaggle`` link nor a node on a
    connector, which the orchestrator treats as "no emulator needed".
    """
    names = emulated_node_names(config)
    by_name = {n.name: n for n in config.nodes}
    declared_connectors = {c.name: c for c in getattr(config, "connectors", None) or []}

    link_base = link_subnet.rsplit(".", 1)[0]
    bypass_base = bypass_subnet.rsplit(".", 1)[0]

    # Wireless roles are needed when allocating hosts, so collect them first.
    wifi_specs: Dict[str, WifiSpec] = {}
    for link in config.links:
        if link.network.mode != NETWAGGLE_MODE:
            continue
        if getattr(link.network, "medium", "wired") == "wifi":
            wifi_specs[link.name] = _wifi_from_network(link, link.network)

    roles: Dict[str, str] = {}
    for spec in wifi_specs.values():
        for node_name, entry in spec.nodes.items():
            existing = roles.get(node_name)
            if existing is not None and existing != entry.role:
                raise UnsupportedLink(
                    "node {!r} is declared as {!r} on one wireless link and {!r} on "
                    "another; a node has one radio role".format(node_name, existing, entry.role)
                )
            roles[node_name] = entry.role

    # A node on an access point is a station, and nothing is ever the access
    # point: that is the connector, which is not a node at all. The schema
    # refuses a node that is on a cell and also carries a link's radio role, so
    # these two sources of a radio role cannot disagree here.
    for node in config.nodes:
        attached = getattr(node, "connector", None)
        if not attached:
            continue
        conn = declared_connectors.get(attached)
        if conn is None:
            raise UnsupportedLink(
                "node {!r} attaches to unknown connector {!r}".format(node.name, attached)
            )
        if conn.type == "ap":
            roles[node.name] = "station"

    hosts: List[EmulatedHost] = []
    for index, name in enumerate(names):
        if name not in by_name:
            raise UnsupportedLink("link endpoint {!r} is not a declared node".format(name))
        # Host addresses start at .1; the bypass gateway takes .254.
        node = by_name[name]
        hosts.append(
            EmulatedHost(
                name=name,
                alias="{}{}".format(ALIAS_PREFIX, index),
                link_ip="{}.{}".format(link_base, index + 1),
                bypass_ip="{}.{}".format(bypass_base, index + 1),
                wifi_role=roles.get(name),
                connector=getattr(node, "connector", None),
                position=getattr(node, "position", None),
            )
        )

    # Connectors continue the hosts' alias numbering rather than starting their
    # own, so every object the emulator names is "<ALIAS_PREFIX><index>" and the
    # recovery sweep that removes leftovers by that shape covers an access point
    # and a switch without being told about them.
    connectors: List[EmulatedConnector] = []
    for offset, conn in enumerate(getattr(config, "connectors", None) or []):
        connectors.append(
            EmulatedConnector(
                name=conn.name,
                alias="{}{}".format(ALIAS_PREFIX, len(hosts) + offset),
                kind=conn.type,
                position=conn.position,
                wifi=_radio_for(conn) if conn.type == "ap" else None,
            )
        )

    links: List[EmulatedLink] = []
    for link in config.links:
        if link.network.mode != NETWAGGLE_MODE:
            continue
        forward, reverse = shaping_for(link)
        links.append(
            EmulatedLink(
                name=link.name,
                endpoint_a=link.from_,
                endpoint_b=link.to,
                forward=forward,
                reverse=reverse,
                medium=getattr(link.network, "medium", "wired"),
                wifi=wifi_specs.get(link.name),
            )
        )

    paths: List[EmulatedPath] = []
    by_link_name = {link.name: link for link in config.links}
    for route in getattr(config, "routes", None) or []:
        hops = [by_link_name[name] for name in route.hops]
        if any(hop.network.mode != NETWAGGLE_MODE for hop in hops):
            continue
        if route.realise != "compose":
            # A relayed route's hops each carry their own connection between two
            # processes, so they need no transit routing: every packet is
            # addressed to the next process, not through it.
            continue
        path = EmulatedPath(
            name=route.name,
            endpoint_a=route.from_,
            endpoint_b=route.to,
            hop_links=[hop.name for hop in hops],
        )
        path.waypoints = [hop.to for hop in hops[:-1]]
        paths.append(path)

    return Topology(
        hosts=hosts,
        links=links,
        paths=paths,
        connectors=connectors,
        link_subnet=link_subnet,
        bypass_subnet=bypass_subnet,
        bypass_gateway="{}.254".format(bypass_base),
    )
