"""Energy collector: power over the run, and energy per inference.

What this is for:
    The paper's energy figure. Its definition is the original's, taken from
    ``energy_monitoring/plot_energy_per_detection.py``, which computes

        energies = latencies * avg_power

    that is, **energy per inference = that inference's latency x the average
    power over that inference's window**. The original averaged power over the
    whole model run because its power log and its timing CSV were separate files
    joined afterwards. Here the sampler runs in the process that times the
    inference, so the same definition is evaluated over each inference's own
    window instead of over the run - see :meth:`EnergyCollector.window`. The run
    integral is kept beside it, so the two are comparable.

    The sampling interval default is the original's as well: its power log came
    from ``nvidia-smi ... -lms 10``, so 10 ms, not a number chosen here.

What it contains:
    - :class:`PowerReader` and the per-device readers:
      :class:`NvmlPowerReader`, :class:`SmiPowerReader`,
      :class:`TegrastatsPowerReader`, :class:`TegraSysfsPowerReader`,
      :class:`RaplPowerReader`.
    - :func:`select_power_reader`: ``auto`` in device order.
    - :class:`InferenceEnergy`, :class:`EnergyCollector`,
      :func:`build_energy_collector`.

How it fits into CADET:
    Built by :func:`cadet.metrics.collectors.build_collectors` when
    ``metrics.collectors`` names ``energy``, and run by the role process on the
    node it measures. Writes ``energy_<node>.csv`` into the run directory: one
    row per sample, carrying the instantaneous power, the energy of that
    sampling interval and the run total so far. :mod:`cadet.roles.server` asks
    the live collector for :meth:`EnergyCollector.window` per request, so the
    per-request record carries that inference's own energy.

    **One machine, two nodes.** Power is a property of a device, not of a
    process, so two roles placed on the same machine both measure that machine:
    their ``energy_<node>.csv`` files describe the same draw and must not be
    added together. Per-inference energy is unaffected - it is attributed from
    the window, and whatever else ran in that window is part of what the device
    drew. Nodes on separate machines each measure their own.

    **Device coverage.** The NVIDIA readers are exercised live. The two Jetson
    readers are **ports** of ``tegrastats_logger.py`` and
    ``jetson_power_logger.py``: the rails they name, the units, the cadence and
    what they do when a read fails all come from those scripts, not from the
    sysfs or tegrastats documentation, and where the two disagree the original
    wins, and each disagreement is recorded in the reader's own docstring.
    They are still **verified on hardware: no** - no Jetson is reachable from
    this environment, so no reading here has come off a board.

    :class:`RaplPowerReader` has no original; it is written against the powercap
    interface and is unit-tested, and its domains are detected on this x86 host
    but its counters are root-only here, so it too is unverified on hardware.

    :class:`RpiProxyPowerReader` is a port of the method in ``rpi_yolo.py``, and
    the one thing to know about it is that **the quantity is not power**. See its
    own docstring: it is reported in its own column and never as watts.

How to configure it:
    ``metrics.energy.source`` (``auto`` | ``nvml`` | ``nvidia-smi`` |
    ``tegrastats`` | ``tegra_sysfs`` | ``rapl`` | ``rpi_proxy``),
    ``metrics.energy.devices`` (device indices for the GPU readers), and
    ``metrics.energy.interval_ms``, which when unset takes **each source's own
    original cadence** (:attr:`PowerReader.DEFAULT_INTERVAL_MS`): 10 ms for the
    GPU and Jetson readers, which is what every original used, and 100 ms for
    the Raspberry Pi proxy, which is what its original used.
    ``metrics.energy.tegrastats_sudo`` controls the ``sudo`` the original ran
    ``tegrastats`` under.
"""

import glob
import os
import re
import shutil
import subprocess
import threading
from typing import Dict, List, Optional, Sequence, Tuple

from cadet.metrics.collectors import ORIGINAL_SAMPLE_INTERVAL_MS, Collector
from cadet.metrics.gpu import NvmlReader, SmiReader

# Jetson rails, from the original `tegrastats_logger.py`. It targets the AGX
# Orin format and names exactly two rails - the GPU-and-SOC rail and the
# CPU-and-CV rail - in milliwatts, and writes a sample only when BOTH matched.
# It does not read the board input rail (VDD_IN) and it does not sum the rails
# it finds, so neither does this: the total here is GPU + CPU, which is what the
# original's two columns add up to and what its averages were taken over.
TEGRASTATS_GPU_RAIL = "VDD_GPU_SOC"
TEGRASTATS_CPU_RAIL = "VDD_CPU_CV"
_TEGRASTATS_GPU_RE = re.compile(r"VDD_GPU_SOC\s+(\d+)mW")
_TEGRASTATS_CPU_RE = re.compile(r"VDD_CPU_CV\s+(\d+)mW")

# The INA device and channels the original `jetson_power_logger.py` reads, by
# fixed path: one INA3221 at i2c address 0-0040, three channels, milliwatts.
# The channel-to-rail mapping is the original's, and it is not discoverable from
# sysfs - the files are called in_power0/1/2_input and nothing in the tree says
# which rail each one meters.
JETSON_INA_DEVICE = "/sys/bus/i2c/devices/0-0040/iio_device"
JETSON_INA_CHANNELS = (
    ("gpu", "in_power0_input"),
    ("cpu", "in_power1_input"),
    ("soc", "in_power2_input"),
)

# The i2c address above is the one board the original ran on, and an address is
# the one thing about it guaranteed not to carry to another carrier board. This
# pattern is tried only when the original's fixed path is absent, and it keeps
# the original's channel order and naming.
JETSON_INA_DEVICE_GLOB = "/sys/bus/i2c/devices/*/iio_device"

# The powercap tree RAPL exposes on x86. A package domain is "intel-rapl:N"; the
# nested "intel-rapl:N:M" are its subdomains (core, uncore, dram) and are left
# out so their energy is not counted twice inside the package's.
RAPL_ROOT = "/sys/class/powercap"
_RAPL_PACKAGE = re.compile(r"^intel-rapl(?:-mmio)?:\d+$")


class PowerReader:
    """What the energy collector needs from a power source.

    :attr:`DEFAULT_INTERVAL_MS` is the cadence the reader's own original
    sampled at, so an unset ``metrics.energy.interval_ms`` reproduces each
    source rather than one number for all of them.

    :attr:`IS_PHYSICAL` says whether what the reader returns is watts. It is
    false for exactly one reader, and the collector uses it to keep a quantity
    that is not power out of every column named for power or energy.

    :meth:`detail` returns whatever the reader can say about the last read
    beyond the single total - the per-rail milliwatts on a Jetson, for instance,
    which are the columns the original logged.
    """

    SOURCE = ""
    DEFAULT_INTERVAL_MS = ORIGINAL_SAMPLE_INTERVAL_MS
    IS_PHYSICAL = True

    def available(self) -> Tuple[bool, str]:
        """Return ``(can_read, reason)``."""
        return False, "not implemented"

    def open(self) -> None:
        """Start whatever the reader needs."""

    def read_w(self) -> Optional[float]:
        """Return the node's instantaneous power in watts, or ``None``."""
        return None

    def detail(self) -> Dict[str, object]:
        """Extra columns describing the last read. Empty by default."""
        return {}

    def close(self) -> None:
        """Release the reader."""


class NvmlPowerReader(PowerReader):
    """Total GPU power through NVML, summed over the selected devices."""

    SOURCE = "nvml"

    def __init__(self, devices=None, nvml=None):
        self._gpu = NvmlReader(devices=devices, nvml=nvml)

    def available(self):
        return self._gpu.available()

    def open(self) -> None:
        self._gpu.open()

    def read_w(self) -> Optional[float]:
        watts = [row["power_w"] for row in self._gpu.read() if row["power_w"] is not None]
        return sum(watts) if watts else None

    def close(self) -> None:
        self._gpu.close()


class SmiPowerReader(PowerReader):
    """Total GPU power from the ``nvidia-smi`` stream - the original's source.

    The original logged ``power.draw`` from exactly this stream. Several devices
    are summed, because the node's energy is what the figure is about; a config
    that means one GPU names it in ``metrics.energy.devices``.
    """

    SOURCE = "nvidia-smi"

    def __init__(self, interval_ms=ORIGINAL_SAMPLE_INTERVAL_MS, devices=None, binary=None):
        self._gpu = SmiReader(interval_ms=interval_ms, devices=devices, binary=binary)

    def available(self):
        return self._gpu.available()

    def open(self) -> None:
        self._gpu.open()

    def read_w(self) -> Optional[float]:
        watts = [row["power_w"] for row in self._gpu.read() if row["power_w"] is not None]
        return sum(watts) if watts else None

    def close(self) -> None:
        self._gpu.close()


class TegrastatsPowerReader(PowerReader):
    """Jetson rail power from a long-running ``tegrastats`` - a port.

    **Verified on hardware: no.** This is a port of
    ``energy_monitoring/tegrastats_logger.py``, read from the machine that holds
    it, but no Jetson is reachable from this environment, so no reading here has
    come off a board.

    What the original does, and what this therefore does:

    * runs ``sudo tegrastats --interval 10`` as one long-lived process and reads
      its lines as they arrive (``tegrastats`` paces itself);
    * matches **two named rails**, ``VDD_GPU_SOC`` and ``VDD_CPU_CV``, in
      milliwatts, with the instantaneous value - not the running average that
      follows the slash;
    * writes a sample only when **both** matched, so a line that carries neither
      is not a zero-power sample;
    * does not read the board input rail and does not sum whatever rails it
      finds. The total here is GPU + CPU, which is what the original's two
      columns add to and what its averages were taken over.

    The ``sudo`` is the original's, because ``tegrastats`` needs root. A role
    already running as root, or one with no ``sudo``, sets
    ``metrics.energy.tegrastats_sudo: false``.
    """

    SOURCE = "tegrastats"

    def __init__(self, interval_ms=ORIGINAL_SAMPLE_INTERVAL_MS, binary=None, use_sudo=True):
        self.interval_ms = max(1, int(interval_ms))
        self.binary = binary or "tegrastats"
        self.use_sudo = bool(use_sudo)
        self._proc = None
        self._thread = None
        self._latest: Optional[Dict[str, float]] = None
        self._lock = threading.Lock()

    def available(self) -> Tuple[bool, str]:
        if shutil.which(self.binary) is None:
            return False, "{} is not on PATH (not a Jetson?)".format(self.binary)
        if self.use_sudo and shutil.which("sudo") is None:
            return False, "sudo is not on PATH; set metrics.energy.tegrastats_sudo: false"
        return True, ""

    def argv(self) -> List[str]:
        """The command the reader runs (the original's, inspectable and tested)."""
        argv = [self.binary, "--interval", str(self.interval_ms)]
        return (["sudo"] + argv) if self.use_sudo else argv

    def open(self) -> None:
        self._proc = subprocess.Popen(
            self.argv(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
            bufsize=1,  # line buffered, as the original asked for
        )
        self._thread = threading.Thread(target=self._pump, name="tegrastats", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        stream = self._proc.stdout if self._proc else None
        if stream is None:
            return
        for line in stream:
            rails = self.parse_rails(line)
            if rails is None:
                continue
            with self._lock:
                self._latest = rails

    @staticmethod
    def parse_rails(line: str) -> Optional[Dict[str, float]]:
        """The two rails' milliwatts from one line, or ``None`` if not both there.

        ``None`` rather than a partial reading is the original's rule (``if gpu
        and cpu``), and it matters: a line missing a rail is a line that says
        nothing about that rail, not one that says it drew nothing.
        """
        gpu = _TEGRASTATS_GPU_RE.search(line or "")
        cpu = _TEGRASTATS_CPU_RE.search(line or "")
        if not (gpu and cpu):
            return None
        return {"gpu": float(gpu.group(1)), "cpu": float(cpu.group(1))}

    @staticmethod
    def parse_w(line: str) -> Optional[float]:
        """Total watts from one line: the two rails the original adds up."""
        rails = TegrastatsPowerReader.parse_rails(line)
        if rails is None:
            return None
        return (rails["gpu"] + rails["cpu"]) / 1000.0

    def read_w(self) -> Optional[float]:
        with self._lock:
            rails = self._latest
        if rails is None:
            return None
        return (rails["gpu"] + rails["cpu"]) / 1000.0

    def detail(self) -> Dict[str, object]:
        """The original's own columns: the two rails, in milliwatts."""
        with self._lock:
            rails = self._latest
        if rails is None:
            return {}
        return {"rail_gpu_mw": rails["gpu"], "rail_cpu_mw": rails["cpu"]}

    def close(self) -> None:
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:  # noqa: BLE001 - teardown of a measurement helper
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass


class TegraSysfsPowerReader(PowerReader):
    """Jetson rail power from the INA channels in sysfs - a port.

    **Verified on hardware: no.** This is a port of
    ``energy_monitoring/jetson_power_logger.py``; no Jetson is reachable from
    this environment.

    What the original does, and what this therefore does:

    * reads three **named channels** by fixed path under one INA3221 at i2c
      address ``0-0040`` - ``in_power0_input`` is GPU, ``in_power1_input`` is
      CPU, ``in_power2_input`` is SOC - in milliwatts. That mapping is the
      original's and is not discoverable from sysfs: nothing in the tree says
      which rail a channel meters;
    * samples every 10 ms, and totals all three channels (unlike the
      ``tegrastats`` original, which reads only two: the two originals meter
      different rail sets, so the two sources are **not** interchangeable);
    * logs the three channels as separate columns, which are kept here.

    One divergence, deliberate and recorded: when a read fails the original
    writes ``-1`` in every channel and carries on. Those rows then enter the mean
    its own analysis takes, dragging the average power down by an amount that
    depends on how often the read failed. This returns no sample instead, so the
    failure is a gap in the record rather than a number in the average.
    """

    SOURCE = "tegra_sysfs"

    def __init__(self, device: str = JETSON_INA_DEVICE, device_glob: str = JETSON_INA_DEVICE_GLOB):
        self.device = device
        self.device_glob = device_glob
        self._paths: List[Tuple[str, str]] = []  # (rail name, file)
        self._latest: Optional[Dict[str, float]] = None

    def channel_paths(self) -> List[Tuple[str, str]]:
        """``(rail, path)`` for the three channels, by the original's fixed path.

        Falls back to the same channel names under another i2c address only when
        the original's address is absent, because an address is the one thing
        about it that cannot carry to a different carrier board.
        """
        for directory in [self.device] + sorted(glob.glob(self.device_glob)):
            candidates = [
                (rail, os.path.join(directory, name)) for rail, name in JETSON_INA_CHANNELS
            ]
            if all(os.path.exists(path) for _rail, path in candidates):
                return candidates
        return []

    def available(self) -> Tuple[bool, str]:
        self._paths = self.channel_paths()
        if not self._paths:
            return False, "no INA power channels at {} or {}".format(self.device, self.device_glob)
        unreadable = [p for _rail, p in self._paths if not os.access(p, os.R_OK)]
        if unreadable:
            return False, "INA channel is not readable ({})".format(unreadable[0])
        return True, ""

    def open(self) -> None:
        if not self._paths:
            self._paths = self.channel_paths()

    def read_w(self) -> Optional[float]:
        rails = {}
        for rail, path in self._paths:
            try:
                with open(path) as handle:
                    rails[rail] = float(handle.read().strip())
            except (OSError, ValueError):
                # The original would write -1 here; see the class docstring for
                # why this reports nothing instead.
                self._latest = None
                return None
        self._latest = rails
        return sum(rails.values()) / 1000.0

    def detail(self) -> Dict[str, object]:
        """The original's own columns: the three channels, in milliwatts."""
        if not self._latest:
            return {}
        return {"rail_{}_mw".format(rail): value for rail, value in self._latest.items()}


class RpiProxyPowerReader(PowerReader):
    """The Raspberry Pi method from ``rpi_yolo.py`` - and it is **not power**.

    **Verified on hardware: no**, and more importantly: this reader does not
    measure power, and nothing it produces is written to a column named for
    watts or joules. It is here because it is what the original used, so it is
    the only way to reproduce that device's published figures, and because the
    alternative - inventing a power model - would be worse.

    The original's ``get_power_draw()`` is, verbatim in effect::

        cpu  = psutil.cpu_percent(interval=None)          # percent
        mem  = psutil.virtual_memory().percent            # percent
        disk = read_bytes + write_bytes                   # CUMULATIVE bytes
        return cpu * 0.5 + mem * 0.3 + (disk / 1e6) * 0.2

    Three things are wrong with reading that as watts, and all three are the
    original's, not this port's:

    1. **It has no units.** Two percentages and a byte count, weighted and
       added. The logger writes the result with a ``" W"`` suffix; the author's
       own later script (``transfer/rpi_power_plot_correct.py``) divides the same
       values by 1000 with the comment "in millijoules (mistakenly)", i.e. reads
       them as milliwatts. The originals disagree with each other by a factor of
       1000 about what the number is.
    2. **The disk term is a cumulative counter, never differenced.** It only
       grows, so the "power" of an idle machine climbs for as long as the process
       lives, and the term's size depends on how much the machine had read since
       boot rather than on anything the run did.
    3. **The sampler ran for a fixed 10 s** regardless of how long the model run
       took, so for any longer run the average covers only the beginning.

    So the value is reported as :attr:`SOURCE` ``rpi_proxy`` in its own
    ``activity_index`` column, with ``power_w``, ``energy_j`` and
    ``energy_j_total`` left empty, and :meth:`EnergyCollector.window` returns
    nothing for it. The arithmetic is reproduced exactly - the same weights, the
    same undifferenced counter, the original's 100 ms cadence - so the original's
    numbers can be reproduced from the column; it is the *unit* that is not
    asserted.
    """

    SOURCE = "rpi_proxy"
    IS_PHYSICAL = False
    # The original's `time.sleep(0.1)` between samples.
    DEFAULT_INTERVAL_MS = 100

    # The original's weights, named rather than inline so the port is checkable
    # against the three lines above.
    CPU_WEIGHT = 0.5
    MEM_WEIGHT = 0.3
    DISK_WEIGHT = 0.2
    DISK_SCALE = 1e6  # bytes -> the original's "MB" term

    def __init__(self, psutil=None):
        self._psutil = psutil
        self._latest: Optional[float] = None

    def _module(self):
        if self._psutil is None:
            import psutil  # noqa: PLC0415 - optional dependency, imported on use

            self._psutil = psutil
        return self._psutil

    def available(self) -> Tuple[bool, str]:
        try:
            self._module()
        except ImportError as exc:
            return False, "psutil is not installed ({})".format(exc)
        return True, ""

    def open(self) -> None:
        # The original primes nothing, so its first cpu_percent() is the average
        # since the process started rather than over an interval. Priming here
        # would change the first sample's value, so it is left as the original
        # has it; the first sample is simply the one to distrust.
        self._module()

    def index(self) -> Optional[float]:
        """The original's weighted sum. Not watts; see the class docstring."""
        module = self._module()
        try:
            counters = module.disk_io_counters()
            disk_bytes = counters.read_bytes + counters.write_bytes
        except (AttributeError, TypeError):
            # disk_io_counters() returns None where the platform has no
            # counters, which the original would have crashed on.
            return None
        cpu = module.cpu_percent(interval=None)
        mem = module.virtual_memory().percent
        return (
            cpu * self.CPU_WEIGHT
            + mem * self.MEM_WEIGHT
            + (disk_bytes / self.DISK_SCALE) * self.DISK_WEIGHT
        )

    def read_w(self) -> Optional[float]:
        """Always ``None``: this reader has no watts to report."""
        self._latest = self.index()
        return None

    def detail(self) -> Dict[str, object]:
        if self._latest is None:
            return {}
        return {"activity_index": round(self._latest, 4)}


class RaplPowerReader(PowerReader):
    """CPU-package power from the x86 RAPL energy counters.

    RAPL reports **energy**, not power, as a microjoule counter per package, so
    power is the counter's increase divided by the time between two reads. The
    first read therefore returns ``None``: there is no interval yet. The counter
    wraps at ``max_energy_range_uj``, which is handled rather than showing up as
    a large negative sample.

    The counters are root-readable only on most distributions; a node that
    cannot read them refuses rather than reporting zero.
    """

    SOURCE = "rapl"

    def __init__(self, root: str = RAPL_ROOT, clock=None):
        self.root = root
        self._clock = clock  # injectable, so the differentiation is testable
        self._domains: List[Tuple[str, Optional[float]]] = []
        self._previous: Optional[Tuple[float, float]] = None

    def _now(self) -> float:
        if self._clock is not None:
            return self._clock()
        import time  # noqa: PLC0415 - kept local so the clock stays injectable

        return time.monotonic()

    def domains(self) -> List[Tuple[str, Optional[float]]]:
        """Return ``(energy_uj path, wrap point)`` for each package domain."""
        found = []
        try:
            names = sorted(os.listdir(self.root))
        except OSError:
            return found
        for name in names:
            if not _RAPL_PACKAGE.match(name):
                continue
            energy = os.path.join(self.root, name, "energy_uj")
            if not os.path.exists(energy):
                continue
            wrap = None
            try:
                with open(os.path.join(self.root, name, "max_energy_range_uj")) as handle:
                    wrap = float(handle.read().strip())
            except (OSError, ValueError):
                wrap = None
            found.append((energy, wrap))
        return found

    def available(self) -> Tuple[bool, str]:
        self._domains = self.domains()
        if not self._domains:
            return False, "no RAPL package domains under {}".format(self.root)
        unreadable = [path for path, _wrap in self._domains if not os.access(path, os.R_OK)]
        if unreadable:
            return False, "RAPL counters are not readable ({})".format(unreadable[0])
        return True, ""

    def open(self) -> None:
        if not self._domains:
            self._domains = self.domains()
        self._previous = None

    def _total_uj(self) -> Tuple[float, float]:
        """Summed counter and the wrap span, so a wrap can be corrected for."""
        total = 0.0
        span = 0.0
        for path, wrap in self._domains:
            with open(path) as handle:
                total += float(handle.read().strip())
            span += wrap if wrap else 0.0
        return total, span

    def read_w(self) -> Optional[float]:
        try:
            total, span = self._total_uj()
        except (OSError, ValueError):
            return None
        now = self._now()
        previous, self._previous = self._previous, (now, total)
        if previous is None:
            return None  # first read only opens the interval
        elapsed = now - previous[0]
        if elapsed <= 0:
            return None
        delta = total - previous[1]
        if delta < 0:
            # The counter wrapped. span is the sum of the domains' wrap points,
            # which is the right correction while every domain wraps at its own.
            delta += span
            if delta < 0:
                return None
        return (delta / 1e6) / elapsed


# ``auto`` order. NVIDIA first because this project's GPU nodes are the ones the
# paper measures; the Jetson readers next (tegrastats before sysfs, because its
# original is the one the project's Jetson work used); RAPL last, as the x86
# CPU-only fallback. A reader that cannot read is skipped, not reported as zero.
#
# ``rpi_proxy`` is deliberately NOT in this list. It would be available on every
# machine with psutil, so ``auto`` would reach it on any node where nothing else
# could be read - and it does not measure power. A config that wants it names it.
_AUTO_ORDER = ("nvml", "nvidia-smi", "tegrastats", "tegra_sysfs", "rapl")

# Every source a config may name, including the ones ``auto`` will not choose.
SOURCES = _AUTO_ORDER + ("rpi_proxy",)


def make_power_reader(
    source: str,
    interval_ms: int = ORIGINAL_SAMPLE_INTERVAL_MS,
    devices: Optional[Sequence[int]] = None,
    tegrastats_sudo: bool = True,
) -> PowerReader:
    """Construct one named power reader (no availability check)."""
    if source == "nvml":
        return NvmlPowerReader(devices=devices)
    if source == "nvidia-smi":
        return SmiPowerReader(interval_ms=interval_ms, devices=devices)
    if source == "tegrastats":
        return TegrastatsPowerReader(interval_ms=interval_ms, use_sudo=tegrastats_sudo)
    if source == "tegra_sysfs":
        return TegraSysfsPowerReader()
    if source == "rapl":
        return RaplPowerReader()
    if source == "rpi_proxy":
        return RpiProxyPowerReader()
    raise ValueError("unknown energy source {!r}".format(source))


class UnavailablePowerReader(PowerReader):
    """Stands in when ``auto`` found nothing, carrying what it tried.

    A reader rather than ``None`` so the refusal reaches the operator through
    the same path as every other unreadable collector, naming each source that
    was tried and why it did not work.
    """

    SOURCE = "none"

    def __init__(self, reasons: Sequence[str]):
        self.reasons = list(reasons)

    def available(self) -> Tuple[bool, str]:
        return False, "no power source on this node: {}".format("; ".join(self.reasons))


def select_power_reader(
    source: str = "auto",
    interval_ms: int = ORIGINAL_SAMPLE_INTERVAL_MS,
    devices: Optional[Sequence[int]] = None,
    tegrastats_sudo: bool = True,
    skip_sources: Sequence[str] = (),
) -> PowerReader:
    """Return the power reader for ``source``.

    A named source is used as named, so a measurement cannot silently change
    source between runs - including when it is one of ``skip_sources``, which
    only narrows what ``auto`` will consider. ``auto`` tries :data:`_AUTO_ORDER`
    in order, skipping those, and returns the first that reports itself
    available.
    """
    if source != "auto":
        return make_power_reader(
            source,
            interval_ms=interval_ms,
            devices=devices,
            tegrastats_sudo=tegrastats_sudo,
        )
    reasons = []
    for candidate in _AUTO_ORDER:
        if candidate in skip_sources:
            continue
        reader = make_power_reader(
            candidate,
            interval_ms=interval_ms,
            devices=devices,
            tegrastats_sudo=tegrastats_sudo,
        )
        ok, reason = reader.available()
        if ok:
            return reader
        reader.close()
        reasons.append("{}: {}".format(candidate, reason))
    return UnavailablePowerReader(reasons)


class InferenceEnergy:
    """One inference's energy, as the original defines it.

    ``energy_j = latency_s * avg_power_w``, with ``avg_power_w`` averaged over
    the samples inside that inference's window. :attr:`samples` is how many
    samples that average came from, so a figure drawn from one sample of a
    sub-interval inference is distinguishable from one drawn from forty.
    """

    __slots__ = ("latency_s", "avg_power_w", "energy_j", "samples")

    def __init__(self, latency_s, avg_power_w, energy_j, samples):
        self.latency_s = latency_s
        self.avg_power_w = avg_power_w
        self.energy_j = energy_j
        self.samples = samples

    def __repr__(self) -> str:
        return "InferenceEnergy({:.6f}s, {}W, {}J, n={})".format(
            self.latency_s, self.avg_power_w, self.energy_j, self.samples
        )


class EnergyCollector(Collector):
    """Samples power into ``energy_<node>.csv`` and integrates it over the run.

    Each row carries the instantaneous power, the energy of the interval that
    sample stands for (``power_w`` x the gap since the previous sample, the
    rectangle rule) and the run total so far, so the run integral is readable
    straight off the file's last row.

    The samples are also kept in memory, which is what :meth:`window` needs: the
    process that times an inference asks this collector for that inference's
    energy while the run is still going, on the monotonic clock both already
    use, so nothing has to be aligned afterwards.
    """

    NAME = "energy"
    # The last row carries energy_j_total, the run integral, so it is the row a
    # reader takes the run's energy from. Flushed per sample for the same reason
    # as the bandwidth collector: a killed role must not under-report a total.
    FLUSH_EVERY_SAMPLE = True
    # The rail columns are the Jetson originals' own columns, and
    # ``activity_index`` is the Raspberry Pi proxy's. Each is empty for the
    # sources that do not produce it, which is how one file per collector
    # survives sources that report different things.
    COLUMNS = (
        "source",
        "power_w",
        "interval_s",
        "energy_j",
        "energy_j_total",
        "rail_gpu_mw",
        "rail_cpu_mw",
        "rail_soc_mw",
        "activity_index",
    )

    # The sources that measure a GPU's power. On a node that was given no GPU
    # these cannot be what `auto` chooses, and the CPU package counters are the
    # measurement that is actually available there.
    GPU_SOURCES = ("nvml", "nvidia-smi")

    def __init__(self, csv_path, interval_s, reader: PowerReader, node="", logger=None):
        super().__init__(csv_path, interval_s, node=node, logger=logger)
        self.reader = reader
        self.energy_j_total = 0.0
        self._samples: List[Tuple[float, float]] = []  # (t_monotonic, watts)
        self._lock = threading.Lock()
        self._previous_t: Optional[float] = None

    def applicable(self):
        """Not applicable only when nothing on this node reports power.

        A node with no GPU still has power to report where the CPU package
        counters are readable, so the fallback happens when the reader is
        chosen (see :func:`build_energy_collector`) rather than here. This is
        the end of that line: no GPU, no readable CPU counters, nothing else -
        so there is no measurement to make, and a run that asked every node for
        energy should not stop because one node has no instrumentation.
        """
        ok, reason = self.reader.available()
        if ok:
            return True, ""
        if getattr(self.reader, "SOURCE", "") == UnavailablePowerReader.SOURCE:
            return False, "not applicable: {}".format(reason)
        # A named source that cannot be read is a failure, not an absence: the
        # config asked for that source specifically.
        return True, ""

    def available(self):
        return self.reader.available()

    def setup(self) -> None:
        self.reader.open()

    def teardown(self) -> None:
        self.reader.close()

    def read(self) -> List[Dict[str, object]]:
        import time  # noqa: PLC0415 - one clock read per sample, kept local

        watts = self.reader.read_w()
        now = time.monotonic()
        if not self.reader.IS_PHYSICAL:
            # A source that does not report watts gets its own column and
            # nothing else: no power, no joules, no run integral. Keeping the
            # quantity out of those columns is the whole reason this branch
            # exists - see :class:`RpiProxyPowerReader`.
            detail = self.reader.detail()
            if not detail:
                return []
            previous, self._previous_t = self._previous_t, now
            interval = (now - previous) if previous is not None else self.interval_s
            row = {
                "source": self.reader.SOURCE,
                "power_w": None,
                "interval_s": round(interval, 6),
                "energy_j": None,
                "energy_j_total": None,
            }
            row.update(detail)
            return [row]
        if watts is None:
            # Nothing to integrate. A row is still not written, because a gap in
            # the record is honest and a zero would be read as "drew no power".
            return []
        previous, self._previous_t = self._previous_t, now
        interval = (now - previous) if previous is not None else self.interval_s
        energy = watts * interval
        self.energy_j_total += energy
        with self._lock:
            self._samples.append((now, watts))
        row = {
            "source": self.reader.SOURCE,
            "power_w": round(watts, 4),
            "interval_s": round(interval, 6),
            "energy_j": round(energy, 6),
            "energy_j_total": round(self.energy_j_total, 6),
        }
        # Whatever else the reader can say about this sample: on a Jetson, the
        # per-rail milliwatts its original logged as separate columns.
        row.update(self.reader.detail())
        return [row]

    def samples(self) -> List[Tuple[float, float]]:
        """A copy of the ``(t_monotonic, watts)`` samples taken so far."""
        with self._lock:
            return list(self._samples)

    def window(self, start_monotonic: float, end_monotonic: float) -> Optional[InferenceEnergy]:
        """Energy attributed to one window: its length x the average power in it.

        This is the original's definition from
        ``plot_energy_per_detection.py`` (``energies = latencies *
        avg_power``), evaluated over this window rather than over the whole run.
        The arguments are ``time.monotonic()`` readings, the clock the server's
        per-request records already use.

        A window shorter than the sampling interval can contain no sample; the
        nearest sample is then used, because the power the device drew during a
        3 ms inference is better described by the reading next to it than by
        nothing at all. :attr:`InferenceEnergy.samples` is 0 in that case, so
        such a row is identifiable. ``None`` means no sample had been taken at
        all - the window is outside the collector's lifetime - or that this
        node's source does not report watts at all.
        """
        if not self.reader.IS_PHYSICAL:
            # No joules can be attributed from a quantity that is not watts.
            return None
        latency = max(0.0, float(end_monotonic) - float(start_monotonic))
        samples = self.samples()
        if not samples:
            return None
        inside = [w for t, w in samples if start_monotonic <= t <= end_monotonic]
        if inside:
            average = sum(inside) / len(inside)
            count = len(inside)
        else:
            midpoint = (start_monotonic + end_monotonic) / 2.0
            average = min(samples, key=lambda s: abs(s[0] - midpoint))[1]
            count = 0
        return InferenceEnergy(latency, average, latency * average, count)


def build_energy_collector(config, node: str, csv_path: str, logger=None) -> EnergyCollector:
    """Build the energy collector for one node from ``metrics.energy``.

    An unset ``interval_ms`` takes the chosen source's **own** original cadence
    rather than one number for every source: 10 ms for the GPU and Jetson
    readers, 100 ms for the Raspberry Pi proxy. The resolved value is logged, so
    a run's record says what it sampled at.
    """
    log = logger or (lambda message: None)
    options = config.metrics.energy
    configured_ms = getattr(options, "interval_ms", None)
    source = getattr(options, "source", "auto")
    devices = getattr(options, "devices", None)
    sudo = getattr(options, "tegrastats_sudo", True)
    # The interval is wanted twice: the nvidia-smi reader paces its own stream
    # with it, and the sampling loop uses it. The reader is built with whatever
    # the config asked for, then the loop takes the reader's own default if the
    # config asked for nothing.
    # A node with no GPU cannot be measured by a GPU power source, so `auto`
    # does not try them there: it would report each as unavailable and then
    # reach the CPU counters anyway, with the reasons for two sources that were
    # never applicable recorded as though they had failed.
    from cadet.metrics.gpu import node_has_gpu  # noqa: PLC0415 - avoids a cycle

    has_gpu = node_has_gpu(config, node)
    reader = select_power_reader(
        source=source,
        interval_ms=configured_ms or ORIGINAL_SAMPLE_INTERVAL_MS,
        devices=devices,
        tegrastats_sudo=sudo,
        skip_sources=() if has_gpu else EnergyCollector.GPU_SOURCES,
    )
    interval_ms = configured_ms or reader.DEFAULT_INTERVAL_MS
    if configured_ms is None:
        log(
            "[metrics] energy source {} at its original {} ms cadence".format(
                reader.SOURCE, interval_ms
            )
        )
    if not reader.IS_PHYSICAL:
        # Said once, loudly, in the run's own log, because the CSV's column
        # names are the only other place this is visible.
        log(
            "[metrics] energy source {} is NOT a power measurement; recorded as "
            "activity_index, with no watts and no joules".format(reader.SOURCE)
        )
    return EnergyCollector(
        csv_path, interval_ms / 1000.0, reader=reader, node=node, logger=log
    )
