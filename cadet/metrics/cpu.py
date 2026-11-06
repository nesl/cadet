"""CPU collector: host and per-process CPU and memory, on the node itself.

What this is for:
    The CPU half of the paper's system-level measurements. Two scopes, because
    they answer different questions: the host figures say how loaded the node
    was, and the per-process figures say how much of that was this role. On a
    node that shares a machine with a simulator or another stream's container,
    the host number alone cannot be attributed to the run.

What it contains:
    - :class:`CpuCollector`: samples both scopes through psutil.
    - :func:`build_cpu_collector`.

How it fits into CADET:
    Built by :func:`cadet.metrics.collectors.build_collectors` when
    ``metrics.collectors`` names ``cpu``, and run by the role process on that
    node. Writes ``cpu_<node>.csv`` into the run directory.

    The process scope is the role process and its children, which is what the
    role is: a client's detection work and a server's model both run in
    processes this one started, so charging only the calling process would
    under-report exactly the work being measured.

How to configure it:
    ``metrics.cpu.interval_ms`` (default 100 ms; see
    :data:`cadet.metrics.collectors.COUNTER_SAMPLE_INTERVAL_MS` for why a
    percentage needs a window) and ``metrics.cpu.per_process`` (set it false to
    record the host scope only).
"""

from typing import Dict, List, Tuple

from cadet.metrics.collectors import COUNTER_SAMPLE_INTERVAL_MS, Collector

_MB = 1024.0 * 1024.0


class CpuCollector(Collector):
    """Samples host and per-process CPU and memory into ``cpu_<node>.csv``.

    ``psutil`` is injectable so the collector can be exercised against a fake
    reader: the paths that matter here (priming the percentage counters, summing
    children, a process that exits mid-run) are not GPU- or host-specific, and
    testing them against the real module would make the test a measurement.
    """

    NAME = "cpu"
    COLUMNS = (
        "source",
        "cpu_pct_host",
        "cpu_count",
        "mem_pct_host",
        "mem_used_mb_host",
        "cpu_pct_process",
        "rss_mb_process",
        "threads_process",
        "children_process",
    )

    def __init__(self, csv_path, interval_s, node="", per_process=True, logger=None, psutil=None):
        super().__init__(csv_path, interval_s, node=node, logger=logger)
        self.per_process = bool(per_process)
        self._psutil = psutil
        self._proc = None

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

    def setup(self) -> None:
        module = self._module()
        # Both percentages are the share of CPU time since the previous call, so
        # the first call only starts the window. Priming here means the first
        # written sample is a real measurement rather than the meaningless
        # since-process-start average psutil returns otherwise.
        module.cpu_percent(interval=None)
        if self.per_process:
            self._proc = module.Process()
            self._proc.cpu_percent(interval=None)

    def read(self) -> List[Dict[str, object]]:
        module = self._module()
        memory = module.virtual_memory()
        row = {
            "source": "psutil",
            "cpu_pct_host": module.cpu_percent(interval=None),
            "cpu_count": module.cpu_count(),
            "mem_pct_host": getattr(memory, "percent", None),
            "mem_used_mb_host": _round_mb(getattr(memory, "used", None)),
            "cpu_pct_process": None,
            "rss_mb_process": None,
            "threads_process": None,
            "children_process": None,
        }
        if self._proc is not None:
            row.update(self._process_scope(module))
        return [row]

    def _process_scope(self, module) -> Dict[str, object]:
        """This role's own CPU and memory, including the children it started."""
        try:
            children = self._proc.children(recursive=True)
        except Exception:  # noqa: BLE001 - psutil raises per-platform errors
            children = []
        cpu = self._proc.cpu_percent(interval=None)
        rss = self._proc.memory_info().rss
        threads = self._proc.num_threads()
        for child in children:
            try:
                # A child seen for the first time has no window either, so its
                # first contribution is zero rather than its lifetime average.
                cpu += child.cpu_percent(interval=None)
                rss += child.memory_info().rss
                threads += child.num_threads()
            except Exception:  # noqa: BLE001 - a child may exit between calls
                continue
        return {
            "cpu_pct_process": round(cpu, 2),
            "rss_mb_process": _round_mb(rss),
            "threads_process": threads,
            "children_process": len(children),
        }


def _round_mb(value):
    """Bytes to megabytes, tolerating ``None``."""
    return None if value is None else round(value / _MB, 3)


def build_cpu_collector(config, node: str, csv_path: str, logger=None) -> CpuCollector:
    """Build the CPU collector for one node from ``metrics.cpu``."""
    options = config.metrics.cpu
    interval_ms = getattr(options, "interval_ms", COUNTER_SAMPLE_INTERVAL_MS)
    per_process = getattr(options, "per_process", True)
    return CpuCollector(
        csv_path,
        interval_ms / 1000.0,
        node=node,
        per_process=per_process,
        logger=logger,
    )
