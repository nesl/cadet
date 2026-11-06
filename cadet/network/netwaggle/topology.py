"""Typed description of an emulated network topology.

What this is for:
    A backend-neutral, dependency-free description of what the emulator should
    build: which hosts exist, which links connect them, how each direction of
    each link is shaped, and which wireless parameters apply. Nothing here
    imports Mininet, so the generator and its tests run on any interpreter,
    without root and without the emulator installed.

What it contains:
    - :class:`~cadet.network.shaping.Shaping`, re-exported: one direction's
      delay/jitter/loss/bandwidth, shared with the netem backend.
    - :class:`WifiNode` and :class:`WifiSpec`: wireless roles, positions,
      propagation model and mobility for a wireless link.
    - :class:`EmulatedConnector`: a shared medium nodes attach to - an access
      point or a wired switch - which runs nothing itself.
    - :class:`EmulatedHost`: one emulated host, with its shaped address, its
      unshaped bypass address and the connector it attaches to.
    - :class:`EmulatedLink`: one link, with per-direction shaping.
    - :class:`EmulatedPath`: a chain of links an end-to-end flow crosses, so the
      emulator can route through the hosts between its ends.
    - :class:`Topology`: the whole description, plus addressing for the bypass
      network and the resolved short names the emulator must use.

How it fits into CADET:
    :mod:`cadet.network.netwaggle.generator` turns a CADET ``Config`` into a
    :class:`Topology`; :mod:`cadet.network.netwaggle.emulator` consumes one and
    drives Mininet-WiFi. The orchestrator never builds Mininet objects directly.

How to configure it:
    Not user-facing. Values come from ``config.links[*]``, ``config.nodes[*]``
    and ``config.connectors[*]`` via the generator; see
    ``docs/config-reference.md`` for the config fields.
"""

from typing import Dict, List, Optional

from cadet.network.shaping import Shaping

# Mininet rejects or silently mishandles long node names (its own docs
# recommend a short string followed by an integer), so the generator assigns
# aliases of this form and keeps the mapping back to CADET node names.
ALIAS_PREFIX = "n"
BYPASS_SWITCH = "bps0"


class WifiNode:
    """One endpoint's wireless role and position.

    ``position`` is an ``"x,y,z"`` string in metres, as Mininet-WiFi expects.
    It is required for stations and access points whenever a propagation model
    is in use, because the model derives signal strength from distance.
    """

    def __init__(self, role: str, position: Optional[str] = None):
        if role not in ("ap", "station"):
            raise ValueError("wifi role must be 'ap' or 'station', got {!r}".format(role))
        self.role = role
        self.position = position


class WifiSpec:
    """Wireless medium parameters for one link.

    The propagation model is what makes position meaningful: without it (and
    without wmediumd running) a wireless link degrades to a plain shared medium
    and the configured coordinates are ignored.

    Used two ways. A point-to-point wireless link fills ``nodes`` with its two
    endpoints' radio roles. A cell's connector states the medium alone and
    leaves ``nodes`` empty: in a cell the access point is the connector, not a
    node, and each station's position lives on its own
    :class:`EmulatedHost`.
    """

    def __init__(
        self,
        nodes: Optional[Dict[str, WifiNode]] = None,
        ssid: str = "cadet",
        mode: str = "g",
        channel: int = 1,
        propagation_model: str = "logDistance",
        propagation_exp: float = 3.5,
        propagation_params: Optional[Dict[str, float]] = None,
        mobility_trace: Optional[str] = None,
    ):
        self.nodes = dict(nodes or {})
        self.ssid = ssid
        self.mode = mode
        self.channel = channel
        self.propagation_model = propagation_model
        self.propagation_exp = propagation_exp
        self.propagation_params = propagation_params or {}
        self.mobility_trace = mobility_trace

    def role_of(self, node_name: str) -> str:
        """Return ``"ap"`` or ``"station"`` for a CADET node name."""
        entry = self.nodes.get(node_name)
        if entry is None:
            raise KeyError("no wifi role declared for node {!r}".format(node_name))
        return entry.role

    def ap_name(self) -> str:
        """Return the CADET node name acting as the access point."""
        for name, entry in self.nodes.items():
            if entry.role == "ap":
                return name
        raise ValueError("wifi link has no node with role 'ap'")


class EmulatedConnector:
    """A shared medium that nodes attach to, and that runs nothing.

    This is the distinction ported from NetWaggle, whose config has two
    sections: hosts, each added with ``addStation`` or ``addHost`` and each
    naming the connector it attaches to, and connectors, each an ``AP`` or a
    ``switch``. Hosts run things; connectors are infrastructure.

    The distinction is not cosmetic. In Mininet-WiFi an access point is a switch
    in the root namespace with no address of its own, so it cannot hold a role
    process: an earlier attempt that made the cell's gateway *be* the access
    point died on the server's bind with ``Cannot assign requested address``.
    Keeping connectors out of the host list is what makes that unexpressible.

    ``kind`` is ``"ap"`` or ``"switch"``, and the only difference between them is
    radios: an access point carries a :class:`WifiSpec` and its attached nodes
    are stations, a switch carries none and its attached nodes are wired. Every
    other property - one subnet, no route between members, nothing running on
    the connector - is shared, which is why both go through the same code.
    """

    def __init__(
        self,
        name: str,
        alias: str,
        kind: str,
        position: Optional[str] = None,
        wifi: Optional[WifiSpec] = None,
    ):
        if kind not in ("ap", "switch"):
            raise ValueError("connector kind must be 'ap' or 'switch', got {!r}".format(kind))
        if kind == "switch" and wifi is not None:
            raise ValueError("connector {!r} is a switch and has no radios".format(name))
        if kind == "ap" and wifi is None:
            raise ValueError("connector {!r} is an access point and needs a WifiSpec".format(name))
        self.name = name  # CADET connector name
        self.alias = alias  # short Mininet-safe name
        self.kind = kind
        self.position = position
        self.wifi = wifi

    @property
    def medium(self) -> str:
        """``"wifi"`` for an access point, ``"wired"`` for a switch."""
        return "wifi" if self.kind == "ap" else "wired"

    def __repr__(self) -> str:
        return "EmulatedConnector({}/{} {})".format(self.name, self.alias, self.kind)


class EmulatedHost:
    """One host inside the emulated topology.

    Each host has two addresses, and the split is the whole point:

    ``link_ip``
        on the shaped network. Traffic to another host's ``link_ip`` crosses the
        emulated links and is delayed, dropped and rate-limited as configured.
    ``bypass_ip``
        on an unshaped bridge that also carries the default route. Framework
        plumbing — simulator frames, model fetches, log shipping — travels here
        and is never shaped, which is what keeps emulation restricted to traffic
        that mimics a real network flow.
    """

    def __init__(
        self,
        name: str,
        alias: str,
        link_ip: str,
        bypass_ip: str,
        wifi_role: Optional[str] = None,
        connector: Optional[str] = None,
        position: Optional[str] = None,
    ):
        self.name = name  # CADET node name
        self.alias = alias  # short Mininet-safe name
        self.link_ip = link_ip
        self.bypass_ip = bypass_ip
        self.wifi_role = wifi_role  # "ap" | "station" | None (wired)
        # Name of the EmulatedConnector this host attaches to, if any. A host
        # with a connector is on a shared medium; one without it reaches its
        # peers over point-to-point links.
        self.connector = connector
        # "x,y,z" in metres, for a station in a cell. The per-link spelling
        # keeps its position in the link's WifiSpec instead.
        self.position = position

    def __repr__(self) -> str:
        return "EmulatedHost({}/{} link={} bypass={}{})".format(
            self.name,
            self.alias,
            self.link_ip,
            self.bypass_ip,
            " on=" + self.connector if self.connector else "",
        )


class EmulatedLink:
    """One emulated link between two hosts, shaped per direction.

    ``forward`` shapes traffic leaving ``endpoint_a`` (so, a -> b) and
    ``reverse`` shapes traffic leaving ``endpoint_b``. Keeping the two separate
    is what lets a CADET link's ``direction`` field mean what it says: netem on
    an interface affects egress only, so a one-way delay is a qdisc on one side.
    """

    def __init__(
        self,
        name: str,
        endpoint_a: str,
        endpoint_b: str,
        forward: Shaping,
        reverse: Shaping,
        medium: str = "wired",
        wifi: Optional[WifiSpec] = None,
    ):
        if medium not in ("wired", "wifi"):
            raise ValueError("medium must be 'wired' or 'wifi', got {!r}".format(medium))
        if medium == "wifi" and wifi is None:
            raise ValueError("link {!r} has medium 'wifi' but no WifiSpec".format(name))
        self.name = name
        self.endpoint_a = endpoint_a
        self.endpoint_b = endpoint_b
        self.forward = forward
        self.reverse = reverse
        self.medium = medium
        self.wifi = wifi

    def __repr__(self) -> str:
        return "EmulatedLink({} {}->{} {} fwd={} rev={})".format(
            self.name, self.endpoint_a, self.endpoint_b, self.medium, self.forward, self.reverse
        )


class EmulatedPath:
    """A chain of links one end-to-end flow crosses, in order.

    A composed route (``routes[*].realise: compose``) keeps a single connection
    between its ends, so for the flow to cross the hops at all the hosts between
    them have to forward it: each end needs a route to the far end **via** its
    neighbour, and every waypoint needs IP forwarding turned on. That is the
    difference from the packet-level ``netem`` backend, where one sender means
    one qdisc and the hops have to be collapsed into it: here each hop is a real
    interface with its own qdisc, so the kernel composes them by carrying the
    packet across both.

    ``hop_links`` names :class:`EmulatedLink` entries in path order;
    ``waypoints`` are the hosts between the ends, derived from them.
    """

    def __init__(self, name: str, endpoint_a: str, endpoint_b: str, hop_links: List[str]):
        if len(hop_links) < 2:
            raise ValueError("path {!r} needs at least two hops".format(name))
        self.name = name
        self.endpoint_a = endpoint_a
        self.endpoint_b = endpoint_b
        self.hop_links = list(hop_links)
        self.waypoints: List[str] = []

    def __repr__(self) -> str:
        return "EmulatedPath({} {}->{} via {})".format(
            self.name, self.endpoint_a, self.endpoint_b, self.waypoints
        )


class Topology:
    """A complete emulated-network description.

    ``link_subnet`` and ``bypass_subnet`` must differ, because the separation of
    shaped from unshaped traffic is done by longest-prefix routing inside each
    host: peers on the link subnet leave by the shaped interface, and everything
    else follows the default route out of the bypass interface.
    """

    def __init__(
        self,
        hosts: List[EmulatedHost],
        links: List[EmulatedLink],
        link_subnet: str = "10.60.0.0/24",
        bypass_subnet: str = "10.61.0.0/24",
        bypass_gateway: str = "10.61.0.254",
        paths: Optional[List["EmulatedPath"]] = None,
        connectors: Optional[List["EmulatedConnector"]] = None,
    ):
        self.hosts = hosts
        self.links = links
        self.paths = paths or []
        self.connectors = connectors or []
        self.link_subnet = link_subnet
        self.bypass_subnet = bypass_subnet
        self.bypass_gateway = bypass_gateway

    @property
    def prefix_len(self) -> str:
        """Prefix length of the shaped subnet, as a string (for ``ip`` calls)."""
        return self.link_subnet.split("/")[1]

    @property
    def bypass_prefix_len(self) -> str:
        """Prefix length of the bypass subnet, as a string."""
        return self.bypass_subnet.split("/")[1]

    def host(self, name: str) -> EmulatedHost:
        """Return the host with this CADET node name."""
        for h in self.hosts:
            if h.name == name:
                return h
        raise KeyError("no emulated host named {!r}".format(name))

    def connector(self, name: str) -> EmulatedConnector:
        """Return the connector with this CADET connector name."""
        for c in self.connectors:
            if c.name == name:
                return c
        raise KeyError("no connector named {!r}".format(name))

    def hosts_on(self, connector_name: str) -> List[EmulatedHost]:
        """Return the hosts attached to one connector, in topology order."""
        return [h for h in self.hosts if h.connector == connector_name]

    def shares_subnet(self, a: str, b: str) -> bool:
        """True if these two hosts are on one shared medium, so neither routes.

        That is the property a cell has and two point-to-point links in series
        do not: both ends are on the same connector, both carry the shaped
        subnet's own prefix, and each is therefore on-link for the other. No
        route is installed between them and no hop separates them.
        """
        host_a, host_b = self.host(a), self.host(b)
        return host_a.connector is not None and host_a.connector == host_b.connector

    def cell_cidr(self, name: str) -> str:
        """Return the address and prefix a connector member carries on that connector.

        This is where the difference between the two media is decided, so it
        lives here rather than inline in the emulator:

        * a **connector-attached** host gets the shaped subnet's own prefix, so
          every other host on that connector is on-link. Nothing installs a
          route between cell members, because there is nothing to route;
        * a **point-to-point** endpoint gets a ``/32`` plus a host route to its
          peer, so a node holding two links does not end up with two interfaces
          claiming one subnet and an arbitrary choice of egress.

        The two coexist in one shaped subnet by longest prefix: a ``/32`` peer
        route is always more specific than a cell's on-link prefix, so a host
        that is both on a cell and on a point-to-point link still sends each
        peer's traffic out of the right interface.
        """
        host = self.host(name)
        if host.connector is None:
            raise KeyError("host {!r} is not attached to a connector".format(name))
        return "{}/{}".format(host.link_ip, self.prefix_len)

    def has_wireless(self) -> bool:
        """True if any medium is wireless (selects the WiFi driver).

        A cell counts: an access-point connector needs Mininet-WiFi even when
        the topology holds no ``links`` at all.
        """
        return any(lk.medium == "wifi" for lk in self.links) or any(
            c.kind == "ap" for c in self.connectors
        )

    def address_map(self) -> Dict[str, Dict[str, str]]:
        """Return ``{node_name: {"link_ip":…, "bypass_ip":…}}``.

        The orchestrator applies this to the run config so role processes bind
        and connect on the shaped addresses, while channel and simulator
        endpoints keep the addresses the user configured.
        """
        return {h.name: {"link_ip": h.link_ip, "bypass_ip": h.bypass_ip} for h in self.hosts}

    def forwarders(self) -> List[str]:
        """Return the hosts that have to forward, in topology order.

        A waypoint of a composed path routes somebody else's packets, which a
        host does not do by default.
        """
        names = []
        for path in self.paths:
            for name in path.waypoints:
                if name not in names:
                    names.append(name)
        return names

    def __repr__(self) -> str:
        return "Topology(hosts={} connectors={} links={} paths={} wireless={})".format(
            len(self.hosts),
            len(self.connectors),
            len(self.links),
            len(self.paths),
            self.has_wireless(),
        )


__all__ = [
    "ALIAS_PREFIX",
    "BYPASS_SWITCH",
    "EmulatedConnector",
    "EmulatedHost",
    "EmulatedLink",
    "EmulatedPath",
    "Shaping",
    "Topology",
    "WifiNode",
    "WifiSpec",
]
