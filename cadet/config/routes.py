"""Resolving a node's path: one link, or a route across several.

A role, the netem rule builder and the link report all have to answer the same
question - "what network path leads into this node?" - and before routes existed
the answer was a single line in each of them (the first link whose ``to`` is the
node). A route makes the answer two-valued, so it is derived here once rather
than three times.

Contains :func:`path_into_node` (the lookup every caller wants),
:func:`route_hops` (a route's hops as :class:`~cadet.config.schema.Link`
objects, in order), :func:`hops_in_direction` (the hops a given direction
perturbs) and :func:`is_route`. Nothing here samples, shapes or connects: it
reads the config and returns config objects, so it has no side effects and
needs no network.

Configured by the ``routes`` section; see ``docs/config-reference.md``.
"""

from typing import List, Optional, Tuple

from cadet.config.schema import Config, Link, Route


def is_route(path) -> bool:
    """Return True if ``path`` is a :class:`Route` rather than a single link."""
    return isinstance(path, Route)


def route_hops(config: Config, route: Route) -> List[Link]:
    """Return ``route``'s hops as links, in path order.

    The schema has already checked that every hop names a declared link and that
    they chain, so this does not re-check: a missing hop here would be a bug in
    the validator, not a config error.
    """
    by_name = {link.name: link for link in config.links}
    return [by_name[name] for name in route.hops]


def hops_in_direction(hops: List[Link], direction: str) -> List[Link]:
    """Return the hops that perturb ``direction`` (``forward`` or ``reverse``).

    A hop declares its own ``direction``, so an asymmetric path - a fast
    downlink and a slow uplink - is one route whose hops are shaped on different
    sides, rather than two routes.
    """
    return [hop for hop in hops if hop.direction in (direction, "both")]


def paths_into_node(config: Config, node_name: str) -> List[Tuple[str, object]]:
    """Return ``(peer, path)`` for every path whose destination is ``node_name``.

    A node can be the far end of several paths at once - N sensors into one
    gateway, an ego and a roadside feed into one server - and those paths are
    not the same link. :func:`path_into_node` answers with one of them, which is
    all a single-client server needs; this answers with all of them, so a server
    can shape each inbound connection by the path its own peer is on.

    The peer is the path's origin: ``from`` for a link, and for a route the node
    the route starts at, because a composed route is one connection from that
    node and a relayed route's last hop starts at the waypoint.
    """
    out: List[Tuple[str, object]] = []
    for route in config.routes:
        if route.to != node_name:
            continue
        if route.realise == "relay":
            hops = route_hops(config, route)
            out.append((hops[-1].from_, route))
        else:
            out.append((route.from_, route))
    hop_names = {name for route in config.routes for name in route.hops}
    for link in config.links:
        if link.to != node_name or link.name in hop_names:
            continue
        out.append((link.from_, link))
    return out


def path_into_node(config: Config, node_name: str):
    """Return the path whose destination is ``node_name``, or ``None``.

    A route is preferred over a link, and the schema refuses a config where both
    connect the same pair, so the preference can never silence a link that was
    meant to apply.
    """
    for route in config.routes:
        if route.to == node_name:
            return route
    for link in config.links:
        if link.to == node_name:
            return link
    return None


def path_name(path) -> Optional[str]:
    """Return a path's name, whichever kind it is."""
    return None if path is None else path.name
