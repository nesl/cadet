"""Emulator: drives Mininet-WiFi from a :class:`Topology`, as a library.

Adapted from the NESL NetWaggle testbed (the IoCT-Testbed-Simulation project),
which established the approach: emulate link characteristics in Mininet while
application processes live in network namespaces, and keep traffic that is not
part of the emulated network off it. The structure here differs in the ways that
mattered for using it as a library inside an orchestrator rather than as a
script:

* **No interactive CLI on the default path.** The original ended in a blocking
  ``CLI(net)`` and only tore down when an operator exited it, so nothing could
  drive it programmatically. Here ``start()`` returns once the topology is up;
  the CLI is available behind ``interactive=True`` for debugging.
* **Real process handles.** The original passed ``os.system()``'s wait status
  around as if it were a pid and later sent SIGKILL to it, which signals an
  unrelated process (and, since ``os.system`` returns 0 on success, usually the
  caller's whole process group). Role processes here are started with
  :meth:`mininet.node.Host.popen`, so every handle is a real
  :class:`subprocess.Popen`.
* **A bypass interface per host.** The original gave a virtual host exactly one
  veth and no second path, so framework plumbing could not avoid the emulated
  network. Every host here also joins an unshaped bridge that carries the
  default route.
* **Per-direction shaping.** The original set one latency per inter-switch link.
  Here each direction of each link is shaped separately, because netem shapes
  egress and a one-way delay therefore belongs on one side only.
* **Explicit addressing on the radio.** The original let ``addStation(ip=...)``
  place a station's address and then bridged a veth onto the same interface.
  Every host here also holds an unshaped bypass veth, which is added before the
  radios exist, so which interface Mininet-WiFi calls the default one is not
  something to rely on. The cell address is written onto ``<node>-wlan0``
  directly - the interface name the original itself assumed - after the network
  is built.
* **Bounded teardown.** Mininet-WiFi's ``net.stop()`` was observed to hang with
  a wireless topology, leaving radios, interfaces, hostapd and wmediumd behind.
  :meth:`Emulator.stop` bounds it and then forcibly removes what it finds.

What this contains:
    - :class:`EmulatorError`
    - :class:`EmulatorStatus`: a snapshot of what is running.
    - :class:`Emulator`: ``build`` / ``start`` / ``stop`` / ``status``, plus
      :meth:`popen` to run a process inside an emulated host. Its two node
      namespaces mirror the original's config: hosts, which run things, and
      connectors, which are infrastructure and run nothing.

How it fits into CADET:
    The orchestrator builds a topology with
    :mod:`cadet.network.netwaggle.generator`, passes it here, and launches role
    processes with :meth:`popen` so they sit inside the emulated hosts. Mininet
    imports are deliberately deferred to :meth:`build`, so this module can be
    imported (and the generator tested) on a machine without the emulator.

How to configure it:
    ``Emulator(topology, interactive=False, stop_timeout_s=30.0)``. The topology
    carries all network parameters; see the generator for how a config becomes
    one.
"""

import os
import shutil
import subprocess
import threading
from typing import Dict, List, Optional, Tuple

from cadet.network.netwaggle.topology import BYPASS_SWITCH, Topology

# Processes the emulator owns and must not leave behind. Matched by exact
# process name, never by command line: a pattern match on the command line also
# matches the shell running the cleanup, which then kills itself.
EMULATOR_PROCESSES = ("hostapd", "wmediumd", "wpa_supplicant")

# debugfs must be mounted for Mininet-WiFi to enumerate hwsim radios under
# /sys/kernel/debug/ieee80211; in a container it usually is not mounted.
DEBUGFS_PATH = "/sys/kernel/debug"


class EmulatorError(Exception):
    """The emulated network could not be built, started or torn down."""


class EmulatorStatus:
    """A snapshot of the emulator's state, safe to serialise into a run record."""

    def __init__(
        self,
        state: str,
        hosts: Optional[List[str]] = None,
        links: Optional[List[str]] = None,
        wireless: bool = False,
        processes: Optional[Dict[str, Optional[int]]] = None,
        detail: str = "",
    ):
        self.state = state  # "empty" | "built" | "running" | "stopped" | "degraded"
        self.hosts = hosts or []
        self.links = links or []
        self.wireless = wireless
        self.processes = processes or {}
        self.detail = detail

    def as_dict(self) -> Dict[str, object]:
        """Return a plain dict for logging or the run record."""
        return {
            "state": self.state,
            "hosts": self.hosts,
            "links": self.links,
            "wireless": self.wireless,
            "processes": self.processes,
            "detail": self.detail,
        }

    def __repr__(self) -> str:
        return "EmulatorStatus({} hosts={} links={})".format(
            self.state, len(self.hosts), len(self.links)
        )


def bypass_intf_name(alias: str) -> str:
    """Return the name of a host's interface onto the plumbing bridge.

    Outside the ``-wlan`` namespace on purpose: see :meth:`Emulator._add_bypass`.
    Short enough for the kernel's 15-character interface-name limit for every
    alias the topology generates, which is why it is the alias and not the
    node's own name.
    """
    return "{}-bps".format(alias)


def _run(argv, check=False):
    """Run a short command, returning its CompletedProcess; never raises on rc."""
    result = subprocess.run(argv, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise EmulatorError(
            "{} failed: {}".format(" ".join(argv), (result.stderr or result.stdout).strip())
        )
    return result


def ensure_debugfs() -> bool:
    """Mount debugfs if it is missing, so hwsim radios can be enumerated.

    Returns True if debugfs is available afterwards. Wireless topologies fail
    with a bare "Cannot find device" otherwise, which is hard to trace back.
    """
    if os.path.isdir(os.path.join(DEBUGFS_PATH, "ieee80211")):
        return True
    _run(["mount", "-t", "debugfs", "none", DEBUGFS_PATH])
    return os.path.isdir(os.path.join(DEBUGFS_PATH, "ieee80211"))


def force_cleanup(scoped: bool = False, names: Optional[List[str]] = None) -> List[str]:
    """Remove emulator leftovers. Returns a list of what was cleaned.

    This is what the emulator runs when Mininet's own ``net.stop()` hangs, and
    what ``cadet down`` runs to recover from a run that never reached its
    teardown at all.

    ``scoped`` is the difference between those two callers. Unscoped is the
    emulator's own call: it owns every bridge and namespace in its container, so
    it clears them all and runs Mininet's global ``mn -c``. Scoped is the
    recovery call, which may be running on a host that is doing other things: it
    stops only the processes the emulator starts, and leaves the named objects
    to :func:`cadet.orchestrator.recovery.sweep`, which removes them by the
    names the topology generator assigned. Running ``mn -c`` there would also
    tear down an unrelated Mininet, which is precisely the indiscriminate
    behaviour the scoping exists to avoid.

    An unscoped call must run in a network namespace the emulator owns. In the
    host's own namespace it would delete that machine's bridges and namespaces,
    and a wireless topology needs the host's namespace, because the hwsim radios
    are created there and a container with a namespace of its own cannot see
    them at all. So a wireless teardown passes ``names``, the objects the
    topology generator named, and this removes those and nothing else: the
    bridge of each connector, the bypass bridge, and any namespace carrying one
    of those names.

    Killing the emulator's processes is scoped by something else: the process
    namespace. Each of :data:`EMULATOR_PROCESSES` is matched by name, so the
    emulation container must keep its own process namespace even when it shares
    the host's network namespace. Sharing both would put the machine's own
    ``wpa_supplicant`` in range of this call.
    """
    removed: List[str] = []

    for name in EMULATOR_PROCESSES:
        # -x matches the process name exactly, so the shell running this
        # cleanup is never itself a match.
        if _run(["pkill", "-9", "-x", name]).returncode == 0:
            removed.append("process:{}".format(name))

    if scoped and names:
        existing_bridges = set(_run(["ovs-vsctl", "list-br"]).stdout.split())
        listed = _run(["ip", "netns", "list"]).stdout.splitlines()
        existing_netns = {line.split()[0] for line in listed if line.strip()}
        for name in names:
            if name in existing_bridges:
                _run(["ovs-vsctl", "--if-exists", "del-br", name])
                removed.append("bridge:{}".format(name))
            if name in existing_netns:
                _run(["ip", "netns", "delete", name])
                removed.append("netns:{}".format(name))

    if not scoped:
        if shutil.which("mn"):
            _run(["mn", "-c"])
            removed.append("mn -c")

        bridges = _run(["ovs-vsctl", "list-br"]).stdout.split()
        for bridge in bridges:
            _run(["ovs-vsctl", "--if-exists", "del-br", bridge])
            removed.append("bridge:{}".format(bridge))

        netns = _run(["ip", "netns", "list"]).stdout.split("\n")
        for line in netns:
            name = line.split()[0] if line.strip() else ""
            if name:
                _run(["ip", "netns", "delete", name])
                removed.append("netns:{}".format(name))

    # Both callers drop the hwsim module, and the scoped one has to. Its radios
    # survive a hung stop, Mininet-WiFi then finds the module already loaded and
    # does not reload it with the radio count the next topology needs, and that
    # run dies inside wmediumd's txpower update with an AttributeError that says
    # nothing about radios. Leaving it loaded for the sake of scoping just moves
    # the damage to the next run, and this was measured: a crashed run left it
    # loaded, `cadet down` reported everything cleaned, and the following
    # wireless run failed that way.
    #
    # It is safe in a scoped sweep because `rmmod` refuses a module that is in
    # use, so a live emulation beside this one keeps its radios and nothing is
    # reported as removed. What is left behind when it succeeds is a module at
    # refcount 0 that no longer breaks anything.
    if _run(["rmmod", "mac80211_hwsim"]).returncode == 0:
        removed.append("module:mac80211_hwsim")

    return removed


class Emulator:
    """Builds, runs and tears down one emulated network for one run."""

    def __init__(
        self,
        topology: Topology,
        interactive: bool = False,
        stop_timeout_s: float = 30.0,
    ):
        self.topology = topology
        self.interactive = interactive
        self.stop_timeout_s = stop_timeout_s
        self._net = None
        self._nat = None
        self._nodes: Dict[str, object] = {}  # CADET node name -> Mininet node
        # CADET connector name -> Mininet AP/switch. Kept apart from _nodes
        # because a connector runs nothing: popen() and run_in_host() must not
        # be able to find one.
        self._connectors: Dict[str, object] = {}
        self._procs: Dict[str, subprocess.Popen] = {}
        self._state = "empty"
        self._detail = ""

    # -- lifecycle -------------------------------------------------------------
    def build(self):
        """Construct the topology. Imports Mininet here, not at module import.

        Wired links are point-to-point: each is its own L2 segment, each
        endpoint interface carries the node's address as a /32, and a host route
        points at the peer. That gives one address per node (so a CADET
        ``node.host`` stays single-valued) while still allowing a node to hold
        several links with different shaping, which a shared subnet could not.

        A connector is the other shape, and the two coexist. Its members carry
        the shaped subnet's own prefix instead of a /32, so they are on-link for
        each other with no route and no hop between them; see
        :meth:`Topology.cell_cidr`. Connectors are added first, as in the
        original, because a wired member's link needs one to attach to.
        """
        topo = self.topology
        if not topo.hosts:
            self._state = "empty"
            return self

        wireless = topo.has_wireless()
        if wireless and not ensure_debugfs():
            raise EmulatorError(
                "wireless topology requested but debugfs is unavailable at {}; "
                "the container needs --privileged and debugfs mounted".format(DEBUGFS_PATH)
            )

        try:
            from mininet.link import Link, TCLink
            from mininet.node import OVSKernelSwitch
        except ImportError as exc:  # pragma: no cover - depends on the image
            raise EmulatorError("Mininet is not installed in this environment: {}".format(exc))

        if wireless:
            try:
                from mn_wifi.link import wmediumd
                from mn_wifi.net import Mininet_wifi
                from mn_wifi.wmediumdConnector import interference
            except ImportError as exc:  # pragma: no cover
                raise EmulatorError("Mininet-WiFi is not installed: {}".format(exc))
            # The wmediumd medium with the interference model is what makes the
            # propagation model and node positions govern the link. Without it
            # a wireless link is a plain shared medium and positions are inert.
            net = Mininet_wifi(link=wmediumd, wmediumd_mode=interference)
        else:
            from mininet.net import Mininet

            # No controller: standalone OVS bridges learn and forward like plain
            # switches. CADET needs netem on links, not OpenFlow, so dropping
            # the controller removes a process and a class of failure.
            net = Mininet(controller=None, switch=OVSKernelSwitch, link=TCLink, autoSetMacs=True)

        self._net = net
        self._add_connectors(net)
        self._add_hosts(net, wireless)
        # The radios are configured before any wired interface is added, and the
        # order is load-bearing. Mininet-WiFi assigns a station's radios into
        # the node's own port numbering, overwriting whatever interface objects
        # already sit there: a veth added first survived in the kernel and
        # vanished from the node's interface list, which is what the emulator
        # looks an interface up in. Measured on a station with one radio and one
        # veth: adding the veth first left the node reporting `n0-wlan0` and
        # `n0-wlan2` and no veth at all, and adding it after left `n0-wlan0` and
        # `n0-bps`, which is what the topology asked for.
        self._add_associations(net)
        if wireless:
            self._configure_wireless(net)
        self._add_links(net, TCLink, wireless, Link)
        self._add_bypass(net, Link)

        self._state = "built"
        return self

    def _add_hosts(self, net, wireless: bool):
        """Add each emulated host, as a station or access point when wireless."""
        for host in self.topology.hosts:
            if wireless and host.wifi_role == "ap":
                node = net.addAccessPoint(
                    host.alias,
                    ssid=self._ssid_for(host.name),
                    mode=self._wifi_attr(host.name, "mode", "g"),
                    channel=str(self._wifi_attr(host.name, "channel", 1)),
                    position=self._position_for(host.name),
                    failMode="standalone",
                    # No `ip=`. An access point is a switch in the root
                    # namespace, so it has no address to hold and a role process
                    # placed on it has nothing to bind - which is the whole
                    # reason a cell is a connector here and not a node wearing
                    # the access point's hat. A node that has to accept
                    # connections in a cell is a station on a connector, and
                    # gets its address like every other station.
                )
            elif wireless and host.wifi_role == "station":
                node = net.addStation(
                    host.alias,
                    ip="{}/{}".format(host.link_ip, self.topology.prefix_len),
                    position=self._position_for(host.name),
                )
            else:
                node = net.addHost(host.alias)
            self._nodes[host.name] = node

    def _add_connectors(self, net):
        """Add each connector: a shared medium that nodes attach to and runs nothing.

        This is the original's ``addAllConnectors``, with its two types kept and
        its three parallel arrays of inter-connector links left out, because
        CADET already spells a shaped path between two things as a ``links``
        entry.

        Neither type takes an address. An access point is a switch with radios,
        a switch is one without them, and in both cases the members' addresses
        live on the members. That is what makes the distinction load-bearing
        rather than cosmetic: there is no way to ask for a role on a connector,
        so the bind failure that comes from running a server on an access point
        is not expressible.
        """
        for conn in self.topology.connectors:
            if conn.kind == "ap":
                node = net.addAccessPoint(
                    conn.alias,
                    ssid=conn.wifi.ssid,
                    mode=conn.wifi.mode,
                    channel=str(conn.wifi.channel),
                    position=conn.position,
                    failMode="standalone",
                )
            else:
                node = net.addSwitch(conn.alias, failMode="standalone")
            self._connectors[conn.name] = node

    def _add_bypass(self, net, wired):
        """Attach every host to an unshaped bridge carrying the default route.

        This is the plumbing path. It is a separate L2 segment with no qdisc, and
        a NAT node on it provides the default route out of the emulation, so a
        host can reach the simulator or a model store without that traffic
        crossing an emulated link.
        """
        topo = self.topology
        bypass = net.addSwitch(BYPASS_SWITCH, failMode="standalone")
        for host in topo.hosts:
            node = self._nodes[host.name]
            # No TC parameters: this link must stay unshaped by construction.
            # Two things are named rather than defaulted, and a cell does not
            # work without either. The class, because a wireless emulation's
            # default link class is the radio medium, so defaulting it builds a
            # radio where a veth belongs. And the interface's own name, because
            # a station numbers a new port into the ``-wlan`` namespace: the
            # veth came up as ``<node>-wlan1``, inside the names the wireless
            # configuration then assigns radios from, and the host was left
            # holding two radios and no interface facing this bridge.
            net.addLink(
                node, bypass, cls=wired, intfName1=bypass_intf_name(host.alias)
            )
        # Mininet's NAT masquerades onto the root namespace's default route and
        # installs the default route inside each host, which is exactly the
        # behaviour wanted here — plumbing leaves by the bypass.
        self._nat = net.addNAT(
            name="nat0",
            connect=bypass,
            ip="{}/{}".format(topo.bypass_gateway, topo.bypass_prefix_len),
        )

    def _add_links(self, net, tclink, wireless: bool, wired=None):
        """Add the shaped links, with each direction's qdisc on its own side."""
        for link in self.topology.links:
            if link.medium == "wifi":
                continue  # association is created by the wireless configuration
            a = self._nodes[link.endpoint_a]
            b = self._nodes[link.endpoint_b]
            params1 = link.forward.as_mininet_params()
            params2 = link.reverse.as_mininet_params()
            # params1 shapes egress from endpoint_a (so a -> b) and params2
            # egress from endpoint_b. cls=TCLink is needed even when one side is
            # unshaped, because the other side carries a qdisc.
            net.addLink(a, b, cls=tclink, params1=params1, params2=params2)

        # Attach each connector's members. Only the wired kind is here: a
        # station's attachment is its association, which is a wireless link and
        # has to exist before the radios are configured, so it is added in
        # _add_associations. A switch's member gets a plain veth with no TC
        # parameters, because a connector is shaped by its own medium - a cell's
        # loss and delay come from the propagation model and the distances - and
        # not by a per-member qdisc.
        for host in self.topology.hosts:
            if host.connector is None:
                continue
            conn = self.topology.connector(host.connector)
            if conn.kind != "switch":
                continue
            kwargs = {"cls": wired} if wired is not None else {}
            net.addLink(self._nodes[host.name], self._connectors[conn.name], **kwargs)

    def _add_associations(self, net):
        """Link every station to the access point it associates with.

        In Mininet-WiFi an association **is** a link: ``addLink`` between a
        station and an access point is what configures the radio to join that
        cell. Without it the radio comes up DOWN and ``iw dev`` reports "Not
        connected" however correct the rest of the topology is - addresses on
        the right interfaces, the subnet on-link, the access point's radio on
        its bridge - and every member's neighbour entry for every other stays
        INCOMPLETE. A radio is not an association, which is what the connector
        model first assumed; the reference script adds this link for a member of
        either connector kind, and for a cell's member it is the only link.

        No class is named, deliberately, where every other ``addLink`` in this
        module names one: the wireless medium is the emulation's own default
        link class, and naming a wired class here is exactly what would turn the
        association back into a veth.
        """
        for host in self.topology.hosts:
            if host.connector is None:
                continue
            conn = self.topology.connector(host.connector)
            if conn.kind != "ap":
                continue
            net.addLink(self._nodes[host.name], self._connectors[conn.name])
        for link in self.topology.links:
            if link.medium != "wifi":
                continue
            ap_name = link.wifi.ap_name()
            ap = self._nodes[ap_name]
            for name in (link.endpoint_a, link.endpoint_b):
                if name != ap_name:
                    net.addLink(self._nodes[name], ap)

    def _configure_wireless(self, net):
        """Apply the propagation model and build the wireless nodes.

        One model applies to the whole emulation, as in the original, so a cell's
        connector is consulted first and a point-to-point wireless link only if
        there is no cell.
        """
        spec = next(
            iter(
                [c.wifi for c in self.topology.connectors if c.kind == "ap"]
                + [lk.wifi for lk in self.topology.links if lk.medium == "wifi"]
            )
        )
        params = {"model": spec.propagation_model, "exp": spec.propagation_exp}
        params.update(spec.propagation_params)
        net.setPropagationModel(**params)
        net.configureWifiNodes()

    def _wifi_spec_for(self, node_name: str):
        """Return the WifiSpec governing this node's radio, or None if it has none.

        A node on an access point takes the cell's medium; otherwise the first
        wireless link it is an endpoint of.
        """
        host = next((h for h in self.topology.hosts if h.name == node_name), None)
        if host is not None and host.connector is not None:
            conn = self.topology.connector(host.connector)
            if conn.wifi is not None:
                return conn.wifi
        for link in self.topology.links:
            if link.medium == "wifi" and node_name in (link.endpoint_a, link.endpoint_b):
                return link.wifi
        return None

    def _wifi_attr(self, node_name: str, attr: str, default):
        spec = self._wifi_spec_for(node_name)
        return getattr(spec, attr, default) if spec else default

    def _ssid_for(self, node_name: str) -> str:
        spec = self._wifi_spec_for(node_name)
        return spec.ssid if spec else "cadet"

    def _position_for(self, node_name: str) -> Optional[str]:
        """Return this node's ``"x,y,z"``, from its own host or its link's spec.

        A station in a cell carries its own position, because the cell's medium
        has no per-endpoint entries; the point-to-point spelling keeps both
        endpoints' positions inside the link's spec.
        """
        host = next((h for h in self.topology.hosts if h.name == node_name), None)
        if host is not None and host.position:
            return host.position
        spec = self._wifi_spec_for(node_name)
        if spec is None:
            return None
        entry = spec.nodes.get(node_name)
        return entry.position if entry else None

    def start(self, interactive: Optional[bool] = None):
        """Start the network and finish host addressing. Returns immediately.

        Unlike the original script this does not block: it brings the topology
        up, configures addresses and routes, and returns. Pass
        ``interactive=True`` (or set it on the constructor) to drop into the
        Mininet CLI instead, which is for debugging only.
        """
        if self._state == "empty":
            return self
        if self._net is None:
            raise EmulatorError("start() called before build()")

        net = self._net
        net.build()
        if self.topology.has_wireless():
            # Mininet-WiFi's own start is the wireless half of bringing a cell
            # up, and there is no doing it by hand: it starts the access points,
            # which are not among `net.switches` and so are not covered by the
            # loop below, and it runs the mobility check that settles each
            # station's association. Starting the L2 nodes by hand instead left
            # every radio DOWN and every station reporting "Not connected",
            # which reads exactly like a cell with no path through its access
            # point. It adds no controller of its own - `self.controllers` stays
            # empty, which is what the standalone bridges below want as well.
            net.start()
        else:
            # Switches must be started explicitly. There is no controller, so
            # each one comes up as a standalone learning bridge; skipping this
            # leaves the bypass switch non-forwarding and every unshaped path
            # silently dead.
            for switch in net.switches:
                switch.start([])
            # Every connector is a switch without radios here, and a switch
            # that is not started is a non-forwarding bridge.
            for conn in self.topology.connectors:
                node = self._connectors.get(conn.name)
                if node is not None:
                    node.start([])
        # Masquerade the bypass network onto the container's default route, so a
        # host can reach the simulator or a model store off-topology. This also
        # installs a default route in each host, which _configure_addressing
        # then re-applies to be explicit about which interface it points at.
        if self._nat is not None:
            self._nat.configDefault()
        self._configure_addressing()
        self._install_ordered_shaping()
        self._state = "running"

        use_cli = self.interactive if interactive is None else interactive
        if use_cli:  # pragma: no cover - operator-driven debugging path
            self._run_cli(net)
        return self

    def _run_cli(self, net):
        """Open the Mininet CLI (debug only)."""
        if self.topology.has_wireless():
            from mn_wifi.cli import CLI
        else:
            from mininet.cli import CLI
        CLI(net)

    def _install_ordered_shaping(self):
        """Replace the qdisc on each jittered wired link so it keeps packet order.

        TCLink's own tree cannot do this: its bandwidth stage runs before netem,
        leaving netem free to let one packet overtake another. A single netem
        with a ``rate`` delays, jitters, drops and then serialises, which is the
        only arrangement measured to hold the flow in order.

        Interfaces without jitter are left exactly as TCLink built them.
        """
        topo = self.topology
        for link in topo.links:
            if link.medium == "wifi":
                continue
            for near, far, shaping in (
                (link.endpoint_a, link.endpoint_b, link.forward),
                (link.endpoint_b, link.endpoint_a, link.reverse),
            ):
                if not shaping.jitters:
                    continue
                node = self._nodes[near]
                intf = self._shaped_intf(node, far)
                if intf is None:
                    continue
                argv = " ".join(shaping.netem_args())
                node.cmd("tc qdisc replace dev {} root netem {}".format(intf, argv))

    def _configure_addressing(self):
        """Give each host its shaped address and routes, plus the bypass route.

        Two shapes, and the difference between them is the point of the
        connector model:

        * a **point-to-point** interface gets the node's address as a /32 and an
          explicit route to the peer. Keeping the prefix at /32 means a node
          holding two links does not end up with two interfaces claiming the
          same subnet, which would make the kernel's choice of egress interface
          arbitrary;
        * a **connector member** gets the shaped subnet's own prefix and **no
          route at all**. Every other member is then on-link: a frame from one
          station to another goes out over the air, with no L3 hop in between
          and nothing to route it. That is what makes a cell a cell rather than
          two links in series, and it is why this loop installs an address and
          stops there.

        The two live in one shaped subnet without interfering, because a /32
        peer route always beats a member's on-link prefix by longest match.
        """
        topo = self.topology
        for link in topo.links:
            if link.medium == "wifi":
                continue
            for near, far in (
                (link.endpoint_a, link.endpoint_b),
                (link.endpoint_b, link.endpoint_a),
            ):
                node = self._nodes[near]
                near_host = topo.host(near)
                far_host = topo.host(far)
                intf = self._shaped_intf(node, far)
                if intf is None:
                    continue
                node.cmd("ip addr flush dev {}".format(intf))
                node.cmd("ip addr add {}/32 dev {}".format(near_host.link_ip, intf))
                node.cmd("ip link set {} up".format(intf))
                node.cmd("ip route replace {} dev {}".format(far_host.link_ip, intf))

        for host in topo.hosts:
            if host.connector is None:
                continue
            conn = topo.connector(host.connector)
            node = self._nodes[host.name]
            intf = self._connector_intf(node, conn)
            if intf is None:
                continue
            # Flushed first because a station may already carry the address
            # addStation was given, and on whichever interface Mininet-WiFi
            # considered the default one at the time.
            node.cmd("ip addr flush dev {}".format(intf))
            node.cmd("ip addr add {} dev {}".format(topo.cell_cidr(host.name), intf))
            node.cmd("ip link set {} up".format(intf))
            # Deliberately no `ip route`: see this method's docstring.

        self._configure_paths()

        for host in topo.hosts:
            node = self._nodes[host.name]
            bypass_intf = self._bypass_intf(node)
            if bypass_intf is None:
                # Not a tolerable absence: _add_bypass attaches every host to
                # the bridge, so no interface facing it means the lookup missed
                # one, and the host would come up with no plumbing path.
                raise EmulatorError(
                    "{} has no interface facing the {} bridge (interfaces: {})".format(
                        host.name,
                        BYPASS_SWITCH,
                        ", ".join(i.name for i in node.intfList()) or "none",
                    )
                )
            node.cmd("ip addr flush dev {}".format(bypass_intf))
            added = node.cmd(
                "ip addr add {}/{} dev {}".format(
                    host.bypass_ip, topo.bypass_prefix_len, bypass_intf
                )
            )
            node.cmd("ip link set {} up".format(bypass_intf))
            # Read back rather than trust the add. A host whose bypass carries
            # no address still looks healthy from the outside - its shaped
            # interfaces are up and a server on it binds - and the first thing
            # that fails is some later fetch over the plumbing, a long way from
            # the cause. The read-back turns that into an error here, naming the
            # interface and whatever the add had to say.
            if not node.cmd("ip -4 -o addr show dev {}".format(bypass_intf)).strip():
                raise EmulatorError(
                    "bypass address {}/{} did not land on {} of {}: {}".format(
                        host.bypass_ip,
                        topo.bypass_prefix_len,
                        bypass_intf,
                        host.name,
                        added.strip() or "no output from ip addr add",
                    )
                )
            # Default route on the bypass: everything that is not a link peer,
            # which is precisely the framework plumbing, leaves unshaped.
            node.cmd(
                "ip route replace default via {} dev {}".format(topo.bypass_gateway, bypass_intf)
            )

    def _configure_paths(self):
        """Route each composed path across its hops, and let waypoints forward.

        Per-link addressing already gives every host a /32 on each shaped
        interface and a route to its neighbour, so a direct link needs nothing
        more. A composed path's two ends are not neighbours: the flow has to be
        carried *through* the hosts between them, which takes two things the
        kernel does not do by itself - a route to the far end via the next hop
        at each end, and ``ip_forward`` on every waypoint, since a host does not
        forward somebody else's packets by default.

        The shaping is not composed here, unlike the ``netem`` backend: each hop
        is a real interface with its own qdisc, so the packet pays each hop as it
        crosses it and the kernel does the composition.
        """
        topo = self.topology
        for name in topo.forwarders():
            node = self._nodes.get(name)
            if node is None:
                continue
            node.cmd("sysctl -w net.ipv4.ip_forward=1")

        for path in topo.paths:
            chain = self._path_nodes(path)
            if chain is None:
                continue
            # Each end needs the far end via its own neighbour, in both
            # directions: a reply that cannot find its way back is a path that
            # works once and then stalls on the handshake.
            for near, neighbour, far in (
                (chain[0], chain[1], chain[-1]),
                (chain[-1], chain[-2], chain[0]),
            ):
                node = self._nodes[near]
                intf = self._shaped_intf(node, neighbour)
                if intf is None:
                    continue
                node.cmd(
                    "ip route replace {} via {} dev {}".format(
                        topo.host(far).link_ip, topo.host(neighbour).link_ip, intf
                    )
                )

    def _path_nodes(self, path):
        """Return the host names a path crosses, in order, or None if it cannot.

        A hop naming a link the topology does not hold means the path was built
        from links the emulator is not emulating, which is a generator bug
        rather than a config error, so it is skipped rather than guessed at.
        """
        by_name = {link.name: link for link in self.topology.links}
        chain = [path.endpoint_a]
        for hop_name in path.hop_links:
            hop = by_name.get(hop_name)
            if hop is None:
                return None
            chain.append(hop.endpoint_b if hop.endpoint_a == chain[-1] else hop.endpoint_a)
        if chain[-1] != path.endpoint_b or len(chain) < 3:
            return None
        return chain

    @staticmethod
    def _link_peer_node(intf) -> Optional[str]:
        """Return the node name at the other end of ``intf``'s link, or ``None``.

        ``None`` for a link that has no node object at the other end, which is
        not an error and is the case this exists for: a **wireless** link's
        ``intf1``/``intf2`` are interface *names* (plain strings), because a
        station associating with an access point is not a veth pair and has no
        peer interface object to point at. Reading ``.node`` off one raised
        ``AttributeError: 'str' object has no attribute 'node'`` and took the
        whole wireless path down before addressing - the wired path never hit it,
        which is why an emulated wired run passed while every Wi-Fi run failed.

        A wireless interface therefore matches neither a shaped peer nor the
        bypass switch, which is correct: both of those are wired veth links that
        this emulator creates itself.
        """
        link = intf.link
        if link is None:
            return None
        other = link.intf1 if link.intf2 == intf else link.intf2
        node = getattr(other, "node", None)
        return getattr(node, "name", None)

    def _shaped_intf(self, node, peer_name: str) -> Optional[str]:
        """Return the interface on ``node`` that faces ``peer_name``."""
        peer_alias = self.topology.host(peer_name).alias
        for intf in node.intfList():
            if intf.link is None or intf.name == "lo":
                continue
            if self._link_peer_node(intf) == peer_alias:
                return intf.name
        return None

    def _connector_intf(self, node, connector) -> Optional[str]:
        """Return the interface on ``node`` that faces this connector.

        A **switch** member is a veth pair like any other, so its peer is found
        the same way the bypass switch's is.

        A **station**'s interface is its radio, and that is looked up by name
        rather than by following a link. Mininet-WiFi names it
        ``<node>-wlan0`` - the name the original script itself relies on - and a
        wireless link holds interface *names* rather than interface objects, so
        there is no peer to follow (see :meth:`_link_peer_node`).
        """
        if connector.kind == "ap":
            radio = "{}-wlan0".format(node.name)
            return radio if any(intf.name == radio for intf in node.intfList()) else None
        for intf in node.intfList():
            if intf.link is None or intf.name == "lo":
                continue
            if self._link_peer_node(intf) == connector.alias:
                return intf.name
        return None

    def _bypass_intf(self, node) -> Optional[str]:
        """Return the interface on ``node`` that faces the bypass switch."""
        for intf in node.intfList():
            if intf.link is None or intf.name == "lo":
                continue
            if self._link_peer_node(intf) == BYPASS_SWITCH:
                return intf.name
        return None

    def run_in_host(self, node_name: str, argv: List[str]) -> str:
        """Run a command to completion inside an emulated host, returning output.

        For one-off namespace setup such as a sysctl or a route change. Use
        :meth:`popen` for role processes, which hands back a real process handle.
        """
        if self._state != "running":
            raise EmulatorError("run_in_host() requires a running emulator")
        node = self._nodes.get(node_name)
        if node is None:
            raise EmulatorError("no emulated host named {!r}".format(node_name))
        return node.cmd(" ".join(argv))

    def peer_routes(self, node_name: str) -> List[Tuple[str, str]]:
        """Return ``(destination, interface)`` for each shaped route of a host.

        These are the routes an initcwnd setting applies to: the per-peer /32
        routes carrying the emulated links, plus the on-link subnet route a
        connector member reaches its cell over, and never the bypass default
        route. The cell's entry is the subnet rather than a peer because a
        member has no per-peer route - that is the property the cell exists for -
        so the route the setting has to name is the on-link one.
        """
        routes: List[Tuple[str, str]] = []
        node = self._nodes.get(node_name)
        if node is None:
            return routes
        host = next((h for h in self.topology.hosts if h.name == node_name), None)
        if host is not None and host.connector is not None:
            conn = self.topology.connector(host.connector)
            intf = self._connector_intf(node, conn)
            if intf is not None:
                routes.append((self.topology.link_subnet, intf))
        for link in self.topology.links:
            if link.medium == "wifi":
                continue
            if node_name == link.endpoint_a:
                peer = link.endpoint_b
            elif node_name == link.endpoint_b:
                peer = link.endpoint_a
            else:
                continue
            intf = self._shaped_intf(node, peer)
            if intf is not None:
                routes.append((self.topology.host(peer).link_ip, intf))
        return routes

    # -- running processes -----------------------------------------------------
    def popen(self, node_name: str, argv: List[str], **kwargs) -> subprocess.Popen:
        """Start a process inside an emulated host and return its real handle.

        This is the replacement for the original's ``os.system`` plus
        wait-status-as-pid: the returned object is a real
        :class:`subprocess.Popen` whose ``pid`` is the process's own, so it can
        be polled, waited on and signalled correctly.
        """
        if self._state != "running":
            raise EmulatorError("popen() requires a running emulator")
        node = self._nodes.get(node_name)
        if node is None:
            raise EmulatorError("no emulated host named {!r}".format(node_name))
        proc = node.popen(argv, **kwargs)
        self._procs["{}:{}".format(node_name, proc.pid)] = proc
        return proc

    def _stop_processes(self, grace_s: float = 5.0):
        """Terminate every process this emulator started, then kill stragglers."""
        for proc in self._procs.values():
            if proc.poll() is None:
                try:
                    proc.terminate()
                except OSError:
                    pass
        for proc in self._procs.values():
            if proc.poll() is None:
                try:
                    proc.wait(timeout=grace_s)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                    except OSError:
                        pass
        self._procs.clear()

    # -- teardown --------------------------------------------------------------
    def stop(self) -> EmulatorStatus:
        """Tear the network down, bounding Mininet's own stop and then forcing it.

        ``net.stop()`` is run on a worker thread and given ``stop_timeout_s``.
        Mininet-WiFi has been observed to hang there on wireless topologies,
        leaving hwsim radios, wlan interfaces, hostapd and wmediumd running; when
        that happens the thread is abandoned and :meth:`_force_cleanup` removes
        what is left. The returned status says which path was taken, so a run
        record shows whether teardown was clean.
        """
        if self._state in ("empty", "stopped"):
            self._state = "stopped"
            return self.status()

        self._stop_processes()

        outcome = {"done": False, "error": None}

        def _worker():
            try:
                self._net.stop()
                outcome["done"] = True
            except Exception as exc:  # teardown must not raise into the caller
                outcome["error"] = "{}: {}".format(type(exc).__name__, exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        thread.join(self.stop_timeout_s)

        if outcome["done"]:
            self._state = "stopped"
            self._detail = "clean"
        else:
            reason = outcome["error"] or "net.stop() exceeded {}s".format(self.stop_timeout_s)
            leftovers = self._force_cleanup()
            self._state = "stopped"
            self._detail = "forced after {}; removed: {}".format(
                reason, ", ".join(leftovers) or "nothing"
            )

        self._net = None
        self._nodes.clear()
        return self.status()

    def _force_cleanup(self) -> List[str]:
        """Remove emulator leftovers. Returns a list of what was cleaned.

        A wireless topology runs in the host's network namespace, because that
        is where hwsim creates its radios. The machine's own bridges and
        namespaces are in that namespace too, so the unscoped sweep would take
        them with it; a wireless teardown removes this topology's own objects by
        name instead.
        """
        if self.topology.has_wireless():
            return force_cleanup(scoped=True, names=self._owned_names())
        return force_cleanup(scoped=False)

    def _owned_names(self) -> List[str]:
        """Return the names this topology gave the objects it created."""
        names = [BYPASS_SWITCH]
        names.extend(connector.alias for connector in self.topology.connectors)
        names.extend(host.alias for host in self.topology.hosts)
        return names

    # -- introspection ---------------------------------------------------------
    def status(self) -> EmulatorStatus:
        """Return a snapshot of the emulator's state."""
        return EmulatorStatus(
            state=self._state,
            hosts=[h.name for h in self.topology.hosts],
            links=[lk.name for lk in self.topology.links],
            wireless=self.topology.has_wireless(),
            processes={key: proc.poll() for key, proc in self._procs.items()},
            detail=self._detail,
        )
