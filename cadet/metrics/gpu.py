"""GPU collector: utilization, memory and power, per device, on the node itself.

What this is for:
    The system-level half of the paper's measurements for an NVIDIA node: how
    busy the GPU was, how much of its memory the run used, and how much power it
    drew, sampled while the run happens on that node.

What it contains:
    - :class:`SmiReader`: the ported reader. The original sampled the GPU with a
      long-running ``nvidia-smi --format=csv -lms <interval>`` and read its
      lines; this is that, widened from ``power.draw`` to the full row.
    - :class:`NvmlReader`: the same numbers through NVML, which needs no
      subprocess and resolves finer. Preferred when the binding is installed.
    - :func:`select_reader`: ``auto`` picks NVML and falls back to
      ``nvidia-smi``, as the original used.
    - :class:`GpuCollector` and :func:`build_gpu_collector`.

How it fits into CADET:
    Built by :func:`cadet.metrics.collectors.build_collectors` when
    ``metrics.collectors`` names ``gpu``, and run by the role process on that
    node. Writes ``gpu_<node>.csv`` into the run directory. Every row carries the
    ``source`` column, so a reading is never separated from which reader
    produced it: NVML and ``nvidia-smi`` do not round identically, and
    ``nvidia-smi`` reports ``N/A`` for power on cards that cannot measure it.

How to configure it:
    ``metrics.gpu.interval_ms`` (default the original's 10 ms),
    ``metrics.gpu.source`` (``auto`` | ``nvml`` | ``nvidia-smi``) and
    ``metrics.gpu.devices`` (device indices; unset means every visible device).
"""

import shutil
import subprocess
import threading
from typing import Dict, List, Optional, Sequence, Tuple

from cadet.metrics.collectors import ORIGINAL_SAMPLE_INTERVAL_MS, Collector

# The fields asked of nvidia-smi, in the order they come back. ``index`` first
# so a row can be attributed to a device when several are visible.
SMI_FIELDS = (
    "index",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
)

# The measurement columns a reader returns, shared by both readers so the CSV
# does not change shape with the source.
READING_COLUMNS = (
    "gpu_index",
    "util_gpu_pct",
    "util_mem_pct",
    "mem_used_mb",
    "mem_total_mb",
    "power_w",
)


def _number(text: str) -> Optional[float]:
    """Parse one nvidia-smi cell, or ``None`` for ``N/A`` and ``[Not Supported]``."""
    cleaned = (text or "").strip()
    if not cleaned or cleaned.startswith("[") or cleaned.upper().startswith("N/A"):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


class GpuReader:
    """What a GPU collector needs from a reader."""

    SOURCE = ""

    def available(self) -> Tuple[bool, str]:
        """Return ``(can_read, reason)``."""
        return False, "not implemented"

    def open(self) -> None:
        """Start whatever the reader needs."""

    def read(self) -> List[Dict[str, object]]:
        """Return one row per device, keyed by :data:`READING_COLUMNS`."""
        return []

    def close(self) -> None:
        """Release the reader."""


class SmiReader(GpuReader):
    """Reads a long-running ``nvidia-smi`` CSV stream (the ported path).

    The original ran ``nvidia-smi --query-gpu=timestamp,power.draw --format=csv
    -lms 10`` into a file and joined that file to the detection timings
    afterwards. The stream is kept - it is one process for the whole run rather
    than one per sample, which is what makes a 10 ms interval affordable - but
    the lines are consumed here and stamped with this process's own clocks
    instead of nvidia-smi's own timestamp. That is the second alignment step the
    collectors exist to remove: a sample is timestamped on the clock the
    inference records use, so nothing has to be matched up later.

    A reader thread keeps the newest line per device; :meth:`read` returns those.
    nvidia-smi paces the stream itself, so a collector sampling slower than the
    stream sees the most recent reading and one sampling faster repeats a
    reading rather than blocking.
    """

    SOURCE = "nvidia-smi"

    def __init__(self, interval_ms: int = ORIGINAL_SAMPLE_INTERVAL_MS, devices=None, binary=None):
        self.interval_ms = max(1, int(interval_ms))
        self.devices = list(devices) if devices else None
        self.binary = binary or "nvidia-smi"
        self._proc = None
        self._thread = None
        self._latest: Dict[int, Dict[str, object]] = {}
        self._lock = threading.Lock()

    def available(self) -> Tuple[bool, str]:
        if shutil.which(self.binary) is None:
            return False, "{} is not on PATH".format(self.binary)
        return True, ""

    def argv(self) -> List[str]:
        """The command the reader runs (inspectable, and asserted in a test)."""
        argv = [
            self.binary,
            "--query-gpu={}".format(",".join(SMI_FIELDS)),
            "--format=csv,noheader,nounits",
            "-lms",
            str(self.interval_ms),
        ]
        if self.devices:
            argv += ["-i", ",".join(str(d) for d in self.devices)]
        return argv

    def open(self) -> None:
        self._proc = subprocess.Popen(
            self.argv(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
        self._thread = threading.Thread(target=self._pump, name="nvidia-smi-stream", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        """Keep the newest parsed line per device index."""
        stream = self._proc.stdout if self._proc else None
        if stream is None:
            return
        for line in stream:
            row = self.parse(line)
            if row is None:
                continue
            with self._lock:
                self._latest[int(row["gpu_index"])] = row

    @staticmethod
    def parse(line: str) -> Optional[Dict[str, object]]:
        """Parse one ``--format=csv,noheader,nounits`` row, or ``None``."""
        cells = [c.strip() for c in (line or "").split(",")]
        if len(cells) != len(SMI_FIELDS):
            return None
        index = _number(cells[0])
        if index is None:
            return None
        return {
            "gpu_index": int(index),
            "util_gpu_pct": _number(cells[1]),
            "util_mem_pct": _number(cells[2]),
            "mem_used_mb": _number(cells[3]),
            "mem_total_mb": _number(cells[4]),
            "power_w": _number(cells[5]),
        }

    def read(self) -> List[Dict[str, object]]:
        with self._lock:
            return [dict(self._latest[index]) for index in sorted(self._latest)]

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


class NvmlReader(GpuReader):
    """Reads the same numbers through NVML, with no subprocess.

    Preferred over the stream when the binding is importable: NVML returns
    milliwatts rather than nvidia-smi's two decimals, and a sample costs a
    library call instead of a line of pipe. The original had no NVML path, so
    this is the added one, which is why the ``source`` column exists.
    """

    SOURCE = "nvml"

    def __init__(self, devices=None, nvml=None):
        self.devices = list(devices) if devices else None
        self._nvml = nvml  # injectable, so the path is testable without a GPU
        self._handles: List[Tuple[int, object]] = []
        self._initialised = False

    def _module(self):
        if self._nvml is None:
            import pynvml  # noqa: PLC0415 - optional dependency, imported on use

            self._nvml = pynvml
        return self._nvml

    def available(self) -> Tuple[bool, str]:
        try:
            module = self._module()
        except ImportError as exc:
            return False, "NVML binding not installed ({})".format(exc)
        try:
            module.nvmlInit()
            self._initialised = True
        except Exception as exc:  # noqa: BLE001 - NVML raises its own error type
            return False, "nvmlInit failed ({})".format(exc)
        return True, ""

    def open(self) -> None:
        module = self._module()
        if not self._initialised:
            module.nvmlInit()
            self._initialised = True
        count = module.nvmlDeviceGetCount()
        indices = self.devices if self.devices else list(range(count))
        self._handles = [(i, module.nvmlDeviceGetHandleByIndex(i)) for i in indices]

    def read(self) -> List[Dict[str, object]]:
        module = self._module()
        rows = []
        for index, handle in self._handles:
            rates = module.nvmlDeviceGetUtilizationRates(handle)
            memory = module.nvmlDeviceGetMemoryInfo(handle)
            try:
                milliwatts = module.nvmlDeviceGetPowerUsage(handle)
                watts = milliwatts / 1000.0
            except Exception:  # noqa: BLE001 - not every card reports power
                watts = None
            rows.append(
                {
                    "gpu_index": index,
                    "util_gpu_pct": float(rates.gpu),
                    "util_mem_pct": float(rates.memory),
                    "mem_used_mb": memory.used / (1024.0 * 1024.0),
                    "mem_total_mb": memory.total / (1024.0 * 1024.0),
                    "power_w": watts,
                }
            )
        return rows

    def close(self) -> None:
        if not self._initialised:
            return
        try:
            self._module().nvmlShutdown()
        except Exception:  # noqa: BLE001 - teardown
            pass
        self._initialised = False


def select_reader(
    source: str = "auto",
    interval_ms: int = ORIGINAL_SAMPLE_INTERVAL_MS,
    devices: Optional[Sequence[int]] = None,
) -> GpuReader:
    """Return the reader for ``source``.

    ``auto`` prefers NVML and falls back to the ``nvidia-smi`` stream the
    original used; a named source is used as named, so a config can pin the
    reader when a comparison must not depend on which happened to be installed.
    """
    if source == "nvml":
        return NvmlReader(devices=devices)
    if source == "nvidia-smi":
        return SmiReader(interval_ms=interval_ms, devices=devices)
    nvml = NvmlReader(devices=devices)
    ok, _reason = nvml.available()
    if ok:
        return nvml
    nvml.close()
    return SmiReader(interval_ms=interval_ms, devices=devices)


class GpuCollector(Collector):
    """Samples GPU utilization, memory and power into ``gpu_<node>.csv``."""

    NAME = "gpu"
    COLUMNS = ("source",) + READING_COLUMNS

    def __init__(
        self, csv_path, interval_s, reader: GpuReader, node="", logger=None, has_gpu=True
    ):
        super().__init__(csv_path, interval_s, node=node, logger=logger)
        self.reader = reader
        # Whether this node was given a GPU at all. A config asks for
        # collectors once for the whole experiment, so a node that was placed
        # on the CPU is still asked for GPU figures; it has none to give, and
        # that is not the same as being unable to read the ones it has.
        self.has_gpu = has_gpu

    def applicable(self):
        if not self.has_gpu:
            return False, "not applicable: no GPU on this node"
        return True, ""

    def available(self):
        # Reached only for a node that has a GPU, so an unreadable reader here
        # is a real failure: the node was meant to produce these figures.
        return self.reader.available()

    def setup(self) -> None:
        self.reader.open()

    def read(self) -> List[Dict[str, object]]:
        rows = self.reader.read()
        for row in rows:
            row["source"] = self.reader.SOURCE
        return rows

    def teardown(self) -> None:
        self.reader.close()


def node_has_gpu(config, node: str) -> bool:
    """Return whether the named node was placed on a GPU.

    The node's own declaration, not a probe of the machine: a node declared
    ``device: cpu`` is given no GPU access even where the host has one - a
    containerised node is launched without them - so what the config says is
    what the node has. A node this config does not mention is treated as having
    one, which keeps a caller that collects outside a placement behaving as it
    did.
    """
    for candidate in getattr(config, "nodes", None) or ():
        if getattr(candidate, "name", None) == node:
            return getattr(candidate, "device", "cpu") == "gpu"
    return True


def build_gpu_collector(config, node: str, csv_path: str, logger=None) -> GpuCollector:
    """Build the GPU collector for one node from ``metrics.gpu``."""
    options = config.metrics.gpu
    interval_ms = getattr(options, "interval_ms", ORIGINAL_SAMPLE_INTERVAL_MS)
    source = getattr(options, "source", "auto")
    devices = getattr(options, "devices", None)
    reader = select_reader(source=source, interval_ms=interval_ms, devices=devices)
    return GpuCollector(
        csv_path,
        interval_ms / 1000.0,
        reader=reader,
        node=node,
        logger=logger,
        has_gpu=node_has_gpu(config, node),
    )
