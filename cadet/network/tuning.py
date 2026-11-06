"""Per-node transport tuning, applied only inside that node's own namespace.

What this is for:
    Packet-level emulation includes the transport's reaction to a link, and for
    a request/response workload of small messages that reaction is substantial:
    on a 60 ms forward-shaped link, the same traffic measured 141 ms with kernel
    defaults and 110 ms with Nagle disabled. Those are properties of the TCP
    stack, not of the emulated link, so they belong in the config as explicit
    per-node knobs rather than being silently inherited or silently changed.

    Defaults are deliberately the kernel's. CADET does not pick a TCP policy on
    the experimenter's behalf; it makes the policy visible and selectable, and
    records it in the run.

What it contains:
    - :data:`SYSCTL_KEYS`: which tunables are sysctls (and so need a namespace).
    - :func:`socket_options`: the setsockopt triples for a node.
    - :func:`apply_socket_options`: apply them to a socket.
    - :func:`sysctl_settings` / :func:`route_settings`: namespace-scoped
      settings, as ``(key, value)`` pairs and as argv.
    - :func:`namespace_commands`: the commands to run inside a node's namespace.
    - :func:`docker_sysctl_args`: the same settings as ``docker run`` flags.
    - :func:`requires_namespace`: whether a node's tuning needs one, used to
      refuse a setting that would otherwise leak to the host.

How it fits into CADET:
    ``nodes[*].transport_tuning`` feeds this module.
    :mod:`cadet.core.transport` applies the socket options when it creates a
    socket; :class:`~cadet.network.netwaggle.emulator.Emulator` runs
    :func:`namespace_commands` inside each emulated host;
    :class:`~cadet.core.docker_launcher.DockerLauncher` passes
    :func:`docker_sysctl_args` to ``docker run``. The orchestrator's preflight
    refuses a sysctl or route setting for a node whose launcher is ``local``,
    because there is no namespace there and the change would be host-wide.

    **Nothing here ever writes a host-wide setting.** The sysctls involved are
    per-network-namespace, which was verified on this platform: setting
    ``tcp_slow_start_after_idle`` inside a container (whether by ``docker
    --sysctl`` at creation or by ``sysctl -w`` in a privileged container with its
    own namespace) leaves the host's value unchanged.

How to configure it:
    Per node::

        nodes:
          - name: client
            transport_tuning:
              tcp_nodelay: true                 # socket option, no namespace needed
              tcp_slow_start_after_idle: false  # sysctl, needs a namespace
              initcwnd: 20                      # route attribute, needs a namespace
"""

import socket
from typing import Dict, List, Optional, Tuple

# Tunables realised as sysctls. These are per-network-namespace in Linux, so a
# node that asks for one must own a namespace; see :func:`requires_namespace`.
SYSCTL_KEYS = {
    "tcp_slow_start_after_idle": "net.ipv4.tcp_slow_start_after_idle",
}


def _tuning_of(node):
    """Return a node's tuning block, or None when it has none."""
    return getattr(node, "transport_tuning", None)


def socket_options(node) -> List[Tuple[int, int, int]]:
    """Return ``(level, option, value)`` triples for this node's sockets.

    Only ``tcp_nodelay`` is a socket option; it needs no namespace and no
    privilege, which is why it is the one knob that works under every launcher.
    An unset field yields nothing, so the kernel default stands.
    """
    tuning = _tuning_of(node)
    if tuning is None:
        return []
    options: List[Tuple[int, int, int]] = []
    if tuning.tcp_nodelay is not None:
        options.append((socket.IPPROTO_TCP, socket.TCP_NODELAY, 1 if tuning.tcp_nodelay else 0))
    return options


def apply_socket_options(sock, node) -> List[str]:
    """Apply this node's socket options to ``sock``; return what was applied.

    Failures are swallowed deliberately: a platform that rejects an option
    should not fail the run, and the returned list records what actually took
    effect so the run report cannot overstate it.
    """
    applied: List[str] = []
    for level, option, value in socket_options(node):
        try:
            sock.setsockopt(level, option, value)
            applied.append("TCP_NODELAY={}".format(value))
        except OSError:
            continue
    return applied


def sysctl_settings(node) -> Dict[str, str]:
    """Return the ``{sysctl key: value}`` this node asks for."""
    tuning = _tuning_of(node)
    if tuning is None:
        return {}
    settings: Dict[str, str] = {}
    if tuning.tcp_slow_start_after_idle is not None:
        settings[SYSCTL_KEYS["tcp_slow_start_after_idle"]] = (
            "1" if tuning.tcp_slow_start_after_idle else "0"
        )
    return settings


def route_settings(node) -> Optional[int]:
    """Return the requested ``initcwnd``, or None.

    initcwnd is an attribute of a route, not a socket or a sysctl, so it is
    applied by rewriting the route inside the node's namespace.
    """
    tuning = _tuning_of(node)
    if tuning is None:
        return None
    return tuning.initcwnd


def requires_namespace(node) -> bool:
    """True when this node's tuning can only be applied inside a namespace.

    ``tcp_nodelay`` alone does not require one. Anything else does, and applying
    it without one would change the host.
    """
    return bool(sysctl_settings(node)) or route_settings(node) is not None


def namespace_commands(node, routes: Optional[List[Tuple[str, str]]] = None) -> List[List[str]]:
    """Return the commands to run inside this node's namespace.

    ``routes`` matters only for ``initcwnd``, which is an attribute of a route
    and so must name one. Each entry is ``(destination, interface)``; the
    netwaggle backend gives a host one /32 route per link peer, so there is
    normally one entry per peer. With no routes supplied the setting is skipped
    rather than guessed at, because rewriting the wrong route would change
    traffic the config never mentioned — the bypass default route in particular
    must stay at the kernel default, like the rest of the plumbing.
    """
    commands: List[List[str]] = []
    for key, value in sorted(sysctl_settings(node).items()):
        commands.append(["sysctl", "-qw", "{}={}".format(key, value)])
    initcwnd = route_settings(node)
    if initcwnd is not None:
        for destination, interface in routes or []:
            commands.append(
                ["ip", "route", "change", destination, "dev", interface, "initcwnd", str(initcwnd)]
            )
    return commands


def docker_sysctl_args(node) -> List[str]:
    """Return ``--sysctl k=v`` flags for ``docker run``.

    Docker applies these at container creation, inside the container's own
    network namespace, which is why this is the right mechanism for a
    containerised node: an unprivileged container has a read-only ``/proc/sys``
    and could not set them afterwards.
    """
    args: List[str] = []
    for key, value in sorted(sysctl_settings(node).items()):
        args += ["--sysctl", "{}={}".format(key, value)]
    return args


def describe(node) -> Dict[str, object]:
    """Return this node's effective tuning, for the run record.

    Fields the config left unset are reported as ``None`` rather than as the
    kernel's value: CADET does not know the kernel default and should not claim
    one.
    """
    tuning = _tuning_of(node)
    if tuning is None:
        return {}
    return {
        "tcp_nodelay": tuning.tcp_nodelay,
        "tcp_slow_start_after_idle": tuning.tcp_slow_start_after_idle,
        "initcwnd": tuning.initcwnd,
    }
