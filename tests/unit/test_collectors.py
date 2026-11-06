"""Tests for the per-node system-metric collectors.

Two things are under test here, and they are different in kind.

The first is the **default**: ``metrics.collectors`` is empty, so an
unconfigured run starts no sampling thread, opens no reader and writes no extra
file. That is the fidelity rule, and a default that drifts is what pinning it
prevents, so it is pinned on the schema, on every config in the repository and
on both role runtimes.

The second is each collector's behaviour against a **fake reader**. The readers
are the part that cannot be tested by running them: the GPU path needs a GPU,
and the Jetson paths need a Jetson, which this environment does not have (those
two are ported and unit-tested only - *verified on hardware: no*). So every
reader is injectable, the sampling loop, the CSV shape, the differencing and the
energy arithmetic are checked against recorded or synthetic readings, and the
one thing left for a live run is whether the real reader returns numbers.
"""

import csv
import os
import time

import pytest
import yaml
from conftest import REPO_ROOT

from cadet.config.loader import load_run_configs
from cadet.config.schema import Config, MetricsConfig
from cadet.core.transport import FramedSocketTransport, counter_snapshot, reset_counters
from cadet.metrics.bandwidth import BandwidthCollector
from cadet.metrics.collectors import (
    _FLUSH_AFTER_S,
    CLOCK_COLUMNS,
    COUNTER_SAMPLE_INTERVAL_MS,
    ORIGINAL_SAMPLE_INTERVAL_MS,
    Collector,
    CollectorError,
    CollectorSet,
    build_collectors,
    csv_path_for,
)
from cadet.metrics.cpu import CpuCollector
from cadet.metrics.energy import (
    JETSON_INA_CHANNELS,
    EnergyCollector,
    PowerReader,
    RaplPowerReader,
    RpiProxyPowerReader,
    TegrastatsPowerReader,
    TegraSysfsPowerReader,
    UnavailablePowerReader,
    make_power_reader,
    select_power_reader,
)
from cadet.metrics.gpu import GpuCollector, NvmlReader, SmiReader, select_reader

REFERENCE_CONFIG = REPO_ROOT / "examples" / "reference" / "synthetic.yaml"


def _rows(path):
    """Read a collector CSV back as a list of dicts."""
    with open(path) as handle:
        return list(csv.DictReader(handle))


# --- the default: nothing enabled -----------------------------------------
def test_no_collectors_by_default():
    # The whole point of the section: a config that says nothing about metrics
    # measures exactly what it measured before the collectors existed.
    assert MetricsConfig().collectors == []
    assert Config.model_fields["metrics"].default_factory().collectors == []


def test_the_per_collector_defaults_are_the_originals():
    metrics = MetricsConfig()
    # The original's power log came from `nvidia-smi ... -lms 10`.
    assert metrics.gpu.interval_ms == ORIGINAL_SAMPLE_INTERVAL_MS == 10
    # Energy takes each source's OWN original cadence when unset, so the field
    # itself has no number in it.
    assert metrics.energy.interval_ms is None
    # The original `sudo tegrastats` kept its sudo.
    assert metrics.energy.tegrastats_sudo is True
    # The counters have no original; 100 ms, because a percentage or a rate
    # needs a window to divide by.
    assert metrics.cpu.interval_ms == COUNTER_SAMPLE_INTERVAL_MS == 100
    assert metrics.bandwidth.interval_ms == COUNTER_SAMPLE_INTERVAL_MS == 100
    # A reader is never pinned by default, and every row says which one ran.
    assert metrics.gpu.source == "auto"
    assert metrics.energy.source == "auto"


# The configs allowed to enable collectors, and why each one is. A config that
# measures devices is not a drifted default - it is a different experiment - but
# it has to be a deliberate entry here rather than a line somebody added to a
# config that reproduces an original.
MEASURING_CONFIGS = {
    # Its subject IS the measurement: the point of putting one node on a second
    # machine is that its device figures are that machine's rather than a share
    # of this one's, and that the two clocks differ. A run of it with no
    # collectors would demonstrate nothing it exists to demonstrate.
    "examples/distributed_sensing/two_host.yaml",
}


def test_no_config_in_the_repository_enables_a_collector_by_accident():
    # Pinned across the repository, not only on the model: a default restored in
    # the schema and then re-enabled in an example config would still change
    # what an "unconfigured" run of that example does. The exceptions are named
    # above, so enabling a collector in a config is an edit to this list.
    roots = [REPO_ROOT / "configs", REPO_ROOT / "examples"]
    checked = 0
    allowed = set()
    for root in roots:
        for path in sorted(root.rglob("*.yaml")):
            data = yaml.safe_load(path.read_text()) or {}
            collectors = (data.get("metrics") or {}).get("collectors")
            relative = path.relative_to(REPO_ROOT).as_posix()
            if relative in MEASURING_CONFIGS:
                allowed.add(relative)
                assert collectors, "{} is listed as measuring but enables none".format(relative)
            else:
                assert not collectors, "{} enables collectors {}".format(relative, collectors)
            checked += 1
    assert checked > 0
    # A listed config that has been deleted or renamed leaves a stale exception
    # that would quietly excuse a future file of the same name.
    assert allowed == MEASURING_CONFIGS


def test_build_collectors_builds_nothing_for_a_default_config():
    runs, _base = load_run_configs(str(REFERENCE_CONFIG), [])
    collected = build_collectors(runs[0], "client", "/nonexistent")
    # No directory was created and no reader opened, which is why the path above
    # can be one that does not exist.
    assert len(collected) == 0
    assert collected.names() == []


def test_queue_is_not_a_sampling_collector():
    # `queue` is the server's per-request record, written for every run; naming
    # it does not start a sampler.
    runs, _base = load_run_configs(str(REFERENCE_CONFIG), ["metrics.collectors=[queue]"])
    assert build_collectors(runs[0], "server", "/nonexistent").names() == []


# --- naming ----------------------------------------------------------------
def test_per_node_csv_naming_matches_the_other_per_node_files():
    assert csv_path_for("/out/run", "gpu", "server") == "/out/run/gpu_server.csv"
    assert csv_path_for("/out/run", "energy", "sensor3") == "/out/run/energy_sensor3.csv"


# --- the sampling loop -----------------------------------------------------
class _Counting(Collector):
    """A collector whose reader is a counter, so the loop can be tested alone."""

    NAME = "counting"
    COLUMNS = ("value",)

    def __init__(self, csv_path, interval_s, rows_per_sample=1, **kwargs):
        super().__init__(csv_path, interval_s, **kwargs)
        self.reads = 0
        self.opened = 0
        self.closed = 0
        self.rows_per_sample = rows_per_sample

    def setup(self):
        self.opened += 1

    def teardown(self):
        self.closed += 1

    def read(self):
        self.reads += 1
        return [{"value": self.reads} for _ in range(self.rows_per_sample)]


def test_the_loop_samples_writes_both_clocks_and_closes(tmp_path):
    path = str(tmp_path / "counting_node.csv")
    collector = _Counting(path, 0.01, node="node-a")
    collector.start()
    time.sleep(0.1)
    collector.stop()

    rows = _rows(path)
    assert collector.opened == 1 and collector.closed == 1
    # Several samples in 100 ms at a 10 ms interval, and not an unbounded number.
    assert 3 <= len(rows) <= 40
    assert list(rows[0]) == list(CLOCK_COLUMNS) + ["value"]
    for row in rows:
        assert row["node"] == "node-a"
        # Both clocks, on every row: the inference records are on one and the
        # per-frame log on the other.
        assert float(row["t_wall"]) > 1_000_000_000  # a real epoch reading
        assert float(row["t_monotonic"]) > 0
    stamps = [float(r["t_monotonic"]) for r in rows]
    assert stamps == sorted(stamps)


def test_one_sample_can_be_several_rows(tmp_path):
    # A two-GPU node is two rows per sample; a node with no open link is none.
    path = str(tmp_path / "counting_node.csv")
    collector = _Counting(path, 0.01, rows_per_sample=2, node="n")
    collector.start()
    time.sleep(0.05)
    collector.stop()
    assert len(_rows(path)) % 2 == 0


def test_a_failing_read_does_not_end_the_run(tmp_path):
    class _Broken(_Counting):
        def read(self):
            super().read()
            raise OSError("the sensor went away")

    logged = []
    collector = _Broken(str(tmp_path / "c_n.csv"), 0.01, logger=logged.append)
    collector.start()
    time.sleep(0.05)
    collector.stop()  # must not raise
    assert any("read failed" in line for line in logged)


def test_a_collector_that_cannot_read_refuses_loudly(tmp_path):
    class _Unavailable(_Counting):
        def available(self):
            return False, "no sensor here"

    collector = _Unavailable(str(tmp_path / "c_n.csv"), 0.01)
    with pytest.raises(CollectorError) as excinfo:
        collector.start()
    assert "no sensor here" in str(excinfo.value)
    # Nothing was written: an empty CSV is indistinguishable from a node that
    # genuinely drew no power, which is the confusion the refusal prevents.
    assert not os.path.exists(str(tmp_path / "c_n.csv"))


def test_a_set_stops_what_it_started_when_one_refuses(tmp_path):
    class _Unavailable(_Counting):
        NAME = "unavailable"

        def available(self):
            return False, "nope"

    good = _Counting(str(tmp_path / "good_n.csv"), 0.01)
    bad = _Unavailable(str(tmp_path / "bad_n.csv"), 0.01)
    with pytest.raises(CollectorError):
        CollectorSet([good, bad]).start()
    assert good.closed == 1  # started, then stopped again


# --- the GPU reader --------------------------------------------------------
def test_the_smi_command_is_the_originals_streaming_query():
    reader = SmiReader(interval_ms=10, devices=[0, 1])
    argv = reader.argv()
    # The original ran a long-running `nvidia-smi --format=csv -lms 10`; the
    # stream is kept, the query widened past power.draw.
    assert "-lms" in argv and argv[argv.index("-lms") + 1] == "10"
    assert "--format=csv,noheader,nounits" in argv
    assert "power.draw" in argv[1] and "utilization.gpu" in argv[1]
    assert argv[argv.index("-i") + 1] == "0,1"


def test_the_smi_row_parser_handles_units_and_absent_readings():
    row = SmiReader.parse("1, 97, 43, 4937, 49140, 204.72")
    assert row == {
        "gpu_index": 1,
        "util_gpu_pct": 97.0,
        "util_mem_pct": 43.0,
        "mem_used_mb": 4937.0,
        "mem_total_mb": 49140.0,
        "power_w": 204.72,
    }
    # A card that cannot meter power reports [N/A]; that is absent, not zero.
    assert SmiReader.parse("0, 0, 0, 188, 24576, [N/A]")["power_w"] is None
    assert SmiReader.parse("garbage") is None


class _FakeGpuReader:
    """Returns recorded readings for two devices."""

    SOURCE = "fake"

    def __init__(self):
        self.opened = self.closed = 0

    def available(self):
        return True, ""

    def open(self):
        self.opened += 1

    def close(self):
        self.closed += 1

    def read(self):
        return [
            {
                "gpu_index": 0,
                "util_gpu_pct": 11.0,
                "util_mem_pct": 2.0,
                "mem_used_mb": 188.0,
                "mem_total_mb": 24576.0,
                "power_w": 24.65,
            },
            {
                "gpu_index": 1,
                "util_gpu_pct": 100.0,
                "util_mem_pct": 43.0,
                "mem_used_mb": 4937.0,
                "mem_total_mb": 49140.0,
                "power_w": 204.72,
            },
        ]


def test_the_gpu_csv_records_which_reader_produced_the_numbers(tmp_path):
    path = str(tmp_path / "gpu_server.csv")
    reader = _FakeGpuReader()
    collector = GpuCollector(path, 0.01, reader=reader, node="server")
    collector.start()
    time.sleep(0.05)
    collector.stop()
    rows = _rows(path)
    assert rows, "the collector wrote no samples"
    assert {r["gpu_index"] for r in rows} == {"0", "1"}
    # NVML and nvidia-smi do not round identically, so a reading is never
    # separated from the reader that produced it.
    assert {r["source"] for r in rows} == {"fake"}
    assert rows[0]["power_w"] == "24.65"
    assert reader.opened == 1 and reader.closed == 1


class _FakeNvml:
    """The handful of NVML calls the reader makes."""

    def __init__(self, count=1, power_mw=150_000, fail_power=False):
        self.count = count
        self.power_mw = power_mw
        self.fail_power = fail_power
        self.init_calls = 0
        self.shutdown_calls = 0

    def nvmlInit(self):
        self.init_calls += 1

    def nvmlShutdown(self):
        self.shutdown_calls += 1

    def nvmlDeviceGetCount(self):
        return self.count

    def nvmlDeviceGetHandleByIndex(self, index):
        return "handle{}".format(index)

    def nvmlDeviceGetUtilizationRates(self, _handle):
        return type("Rates", (), {"gpu": 77, "memory": 12})()

    def nvmlDeviceGetMemoryInfo(self, _handle):
        return type("Mem", (), {"used": 2 * 1024 * 1024, "total": 8 * 1024 * 1024})()

    def nvmlDeviceGetPowerUsage(self, _handle):
        if self.fail_power:
            raise RuntimeError("not supported")
        return self.power_mw


def test_the_nvml_reader_reports_watts_and_megabytes():
    nvml = _FakeNvml(count=2, power_mw=150_500)
    reader = NvmlReader(nvml=nvml)
    assert reader.available() == (True, "")
    reader.open()
    rows = reader.read()
    assert [r["gpu_index"] for r in rows] == [0, 1]
    assert rows[0]["power_w"] == pytest.approx(150.5)
    assert rows[0]["mem_used_mb"] == pytest.approx(2.0)
    assert rows[0]["util_gpu_pct"] == 77.0
    reader.close()
    assert nvml.shutdown_calls == 1


def test_a_card_without_a_power_sensor_reports_absent_power():
    reader = NvmlReader(nvml=_FakeNvml(fail_power=True))
    reader.available()
    reader.open()
    assert reader.read()[0]["power_w"] is None


def test_a_named_gpu_source_is_used_as_named():
    # A comparison must not silently change reader between runs.
    assert select_reader("nvidia-smi").SOURCE == "nvidia-smi"
    assert select_reader("nvml").SOURCE == "nvml"


# --- the CPU reader --------------------------------------------------------
class _FakeProcess:
    def __init__(self, cpu=20.0, rss=100 * 1024 * 1024, threads=3, children=()):
        self._cpu = cpu
        self._rss = rss
        self._threads = threads
        self._children = list(children)
        self.cpu_percent_calls = 0

    def cpu_percent(self, interval=None):
        self.cpu_percent_calls += 1
        return self._cpu

    def memory_info(self):
        return type("Info", (), {"rss": self._rss})()

    def num_threads(self):
        return self._threads

    def children(self, recursive=False):
        return list(self._children)


class _FakePsutil:
    def __init__(self, process=None):
        self._process = process or _FakeProcess()
        self.cpu_percent_calls = 0

    def cpu_percent(self, interval=None):
        self.cpu_percent_calls += 1
        return 42.5

    def cpu_count(self):
        return 8

    def virtual_memory(self):
        return type("VM", (), {"percent": 31.0, "used": 4 * 1024 * 1024 * 1024})()

    def Process(self, *_args):
        return self._process


def test_the_cpu_collector_primes_both_percentages_then_measures(tmp_path):
    child = _FakeProcess(cpu=5.0, rss=10 * 1024 * 1024, threads=1)
    process = _FakeProcess(cpu=20.0, rss=100 * 1024 * 1024, threads=3, children=[child])
    psutil = _FakePsutil(process=process)
    collector = CpuCollector(
        str(tmp_path / "cpu_server.csv"), 0.01, node="server", psutil=psutil
    )
    collector.setup()
    # Priming: the first call to either only opens the window, so a measurement
    # has happened before the first row is written.
    assert psutil.cpu_percent_calls == 1
    assert process.cpu_percent_calls == 1

    row = collector.read()[0]
    assert row["source"] == "psutil"
    assert row["cpu_pct_host"] == 42.5
    assert row["cpu_count"] == 8
    assert row["mem_used_mb_host"] == pytest.approx(4096.0)
    # The role is the process and the children it started: a client's detection
    # work and a server's model both run in processes this one spawned.
    assert row["cpu_pct_process"] == pytest.approx(25.0)
    assert row["rss_mb_process"] == pytest.approx(110.0)
    assert row["threads_process"] == 4
    assert row["children_process"] == 1


def test_a_child_that_exits_mid_sample_is_skipped_not_fatal():
    class _Gone(_FakeProcess):
        def cpu_percent(self, interval=None):
            raise RuntimeError("no such process")

    process = _FakeProcess(cpu=7.0, children=[_Gone()])
    collector = CpuCollector("/dev/null", 0.01, psutil=_FakePsutil(process=process))
    collector.setup()
    assert collector.read()[0]["cpu_pct_process"] == pytest.approx(7.0)


def test_the_host_scope_can_be_recorded_alone():
    collector = CpuCollector("/dev/null", 0.01, per_process=False, psutil=_FakePsutil())
    collector.setup()
    row = collector.read()[0]
    assert row["cpu_pct_host"] == 42.5
    assert row["cpu_pct_process"] is None


def test_the_cpu_collector_says_so_when_psutil_is_absent():
    class _NoPsutil(CpuCollector):
        def _module(self):
            raise ImportError("No module named 'psutil'")

    ok, reason = _NoPsutil("/dev/null", 0.01).available()
    assert ok is False and "psutil" in reason


# --- energy: the original's definition ------------------------------------
class _ScriptedPower(PowerReader):
    """Returns a scripted series of watt readings."""

    SOURCE = "scripted"

    def __init__(self, readings):
        self.readings = list(readings)
        self.index = 0

    def available(self):
        return True, ""

    def read_w(self):
        if self.index >= len(self.readings):
            return self.readings[-1] if self.readings else None
        value = self.readings[self.index]
        self.index += 1
        return value


def _energy_collector(readings, interval_s=0.01):
    return EnergyCollector("/dev/null", interval_s, reader=_ScriptedPower(readings))


def test_energy_per_inference_is_latency_times_average_power_in_its_window():
    # The original (energy_monitoring/plot_energy_per_detection.py) computes
    #     energies = latencies * avg_power
    # and this is that, evaluated over one inference's own window.
    collector = _energy_collector([])
    collector._samples = [(10.0, 100.0), (10.1, 200.0), (10.2, 300.0), (10.5, 999.0)]

    result = collector.window(10.0, 10.2)
    assert result.samples == 3
    assert result.avg_power_w == pytest.approx(200.0)  # (100+200+300)/3
    assert result.latency_s == pytest.approx(0.2)
    assert result.energy_j == pytest.approx(0.2 * 200.0)
    # The identity itself, not just the number: joules = seconds x watts.
    assert result.energy_j == pytest.approx(result.latency_s * result.avg_power_w)


def test_an_inference_shorter_than_the_sampling_interval_uses_the_nearest_sample():
    collector = _energy_collector([])
    collector._samples = [(10.0, 50.0), (10.2, 150.0)]
    result = collector.window(10.11, 10.13)
    # No sample falls inside a 20 ms window of a 200 ms-apart series. The
    # nearest reading describes that window better than nothing does, and
    # samples == 0 is what makes such a row identifiable afterwards.
    assert result.samples == 0
    assert result.avg_power_w == pytest.approx(150.0)
    assert result.energy_j == pytest.approx(0.02 * 150.0)


def test_a_window_outside_the_collectors_life_has_no_energy():
    assert _energy_collector([]).window(1.0, 2.0) is None


def test_the_run_integral_accumulates_power_times_elapsed(tmp_path):
    path = str(tmp_path / "energy_server.csv")
    collector = EnergyCollector(path, 0.01, reader=_ScriptedPower([100.0] * 1000), node="server")
    collector.start()
    time.sleep(0.2)
    collector.stop()
    rows = _rows(path)
    assert rows, "the collector wrote no samples"
    # Each row carries the interval it stands for and the total so far, so the
    # run integral is readable off the last row.
    total = float(rows[-1]["energy_j_total"])
    seconds = sum(float(r["interval_s"]) for r in rows)
    assert total == pytest.approx(100.0 * seconds, rel=1e-3)
    assert {r["source"] for r in rows} == {"scripted"}


def test_a_reading_that_is_absent_is_a_gap_not_a_zero(tmp_path):
    # A zero would be read as "this node drew no power", which is a different
    # claim from "nothing could be read".
    path = str(tmp_path / "energy_n.csv")
    collector = EnergyCollector(path, 0.01, reader=_ScriptedPower([None, None]))
    collector.start()
    time.sleep(0.05)
    collector.stop()
    assert _rows(path) == []
    assert collector.energy_j_total == 0.0


# --- energy: the Jetson paths (ports; never run on hardware) --------------
#
# Every expectation below is read off the original logger scripts, not off the
# tegrastats or sysfs documentation. Where the two disagree the original is
# what is asserted.
def test_tegrastats_reads_the_two_rails_the_original_names():
    # A real AGX Orin line. The original matches VDD_GPU_SOC and VDD_CPU_CV in
    # mW, taking the instantaneous value before the slash, not the average.
    line = (
        "RAM 3043/7850MB SWAP 0/3925MB CPU [9%@1190,6%@1190] GR3D_FREQ 0% "
        "VDD_GPU_SOC 2520mW/2400mW VDD_CPU_CV 1096mW/1050mW VDD_IN 4148mW/4148mW"
    )
    assert TegrastatsPowerReader.parse_rails(line) == {"gpu": 2520.0, "cpu": 1096.0}
    # Total = GPU + CPU. NOT VDD_IN (4148 mW), which the original never reads,
    # and not the sum of every rail on the line.
    assert TegrastatsPowerReader.parse_w(line) == pytest.approx(3.616)


def test_tegrastats_needs_both_rails_or_reports_nothing():
    # The original's `if gpu and cpu`: a line missing a rail says nothing about
    # it, which is not the same as saying it drew nothing.
    assert TegrastatsPowerReader.parse_rails("VDD_GPU_SOC 2520mW/2400mW") is None
    assert TegrastatsPowerReader.parse_w("VDD_CPU_CV 1096mW/1050mW") is None
    assert TegrastatsPowerReader.parse_w("RAM 3043/7850MB SWAP 0/3925MB") is None


def test_tegrastats_does_not_guess_at_the_older_unitless_format():
    # The original's regex requires the mW suffix and the Orin rail names, so a
    # board printing the older form yields no samples rather than a guess.
    assert TegrastatsPowerReader.parse_w("POM_5V_IN 4148/4148 POM_5V_GPU 500/480") is None


def test_tegrastats_runs_the_originals_command_including_sudo():
    argv = TegrastatsPowerReader(interval_ms=10).argv()
    assert argv == ["sudo", "tegrastats", "--interval", "10"]
    # Escapable, for a role that is already root or has no sudo.
    assert TegrastatsPowerReader(interval_ms=10, use_sudo=False).argv() == [
        "tegrastats",
        "--interval",
        "10",
    ]


def test_tegrastats_keeps_the_originals_per_rail_columns():
    reader = TegrastatsPowerReader()
    reader._latest = {"gpu": 2520.0, "cpu": 1096.0}
    assert reader.detail() == {"rail_gpu_mw": 2520.0, "rail_cpu_mw": 1096.0}
    assert TegrastatsPowerReader().detail() == {}


def _jetson_tree(tmp_path, address="0-0040", values=(1500, 2500, 800)):
    """A fake of the sysfs path the original reads, by its own path shape."""
    device = tmp_path / "sys" / "bus" / "i2c" / "devices" / address / "iio_device"
    device.mkdir(parents=True)
    for (_rail, name), value in zip(JETSON_INA_CHANNELS, values):
        (device / name).write_text("{}\n".format(value))
    return device


def test_the_sysfs_reader_reads_the_three_channels_by_the_originals_path(tmp_path):
    device = _jetson_tree(tmp_path)
    reader = TegraSysfsPowerReader(device=str(device), device_glob=str(tmp_path / "absent" / "*"))
    assert reader.available() == (True, "")
    reader.open()
    # GPU + CPU + SOC, in mW, totalled: 1500 + 2500 + 800.
    assert reader.read_w() == pytest.approx(4.8)
    # The channel-to-rail mapping is the original's; nothing in sysfs states it.
    assert reader.detail() == {
        "rail_gpu_mw": 1500.0,
        "rail_cpu_mw": 2500.0,
        "rail_soc_mw": 800.0,
    }


def test_the_sysfs_reader_falls_back_to_another_i2c_address(tmp_path):
    # The hard-coded address is the one thing about the original's path that
    # cannot carry to a different carrier board.
    _jetson_tree(tmp_path, address="1-0041", values=(10, 20, 30))
    reader = TegraSysfsPowerReader(
        device=str(tmp_path / "sys" / "bus" / "i2c" / "devices" / "0-0040" / "iio_device"),
        device_glob=str(tmp_path / "sys" / "bus" / "i2c" / "devices" / "*" / "iio_device"),
    )
    assert reader.available()[0] is True
    reader.open()
    assert reader.read_w() == pytest.approx(0.06)


def test_a_failed_sysfs_read_is_a_gap_not_the_originals_sentinel(tmp_path):
    # The original writes -1 in every channel and carries on, and those rows
    # then enter the mean its own analysis takes. Recorded divergence: no
    # sample, so the failure cannot move the average.
    device = _jetson_tree(tmp_path)
    reader = TegraSysfsPowerReader(device=str(device), device_glob=str(tmp_path / "none" / "*"))
    reader.available()
    reader.open()
    (device / "in_power1_input").write_text("not a number\n")
    assert reader.read_w() is None
    assert reader.detail() == {}


def test_a_node_with_no_ina_channels_refuses_rather_than_reporting_zero(tmp_path):
    reader = TegraSysfsPowerReader(
        device=str(tmp_path / "absent"), device_glob=str(tmp_path / "absent" / "*")
    )
    ok, reason = reader.available()
    assert ok is False and "no INA power channels" in reason


# --- energy: the Raspberry Pi method, ported and NOT called power ---------
class _RpiPsutil:
    """psutil as the original uses it: a percentage, a percentage, and a counter."""

    def __init__(self, cpu=40.0, mem=50.0, read_bytes=3_000_000, write_bytes=1_000_000):
        self._cpu = cpu
        self._mem = mem
        self.read_bytes = read_bytes
        self.write_bytes = write_bytes

    def cpu_percent(self, interval=None):
        return self._cpu

    def virtual_memory(self):
        return type("VM", (), {"percent": self._mem, "used": 0})()

    def disk_io_counters(self):
        return type("IO", (), {"read_bytes": self.read_bytes, "write_bytes": self.write_bytes})()


def test_the_rpi_index_is_the_originals_arithmetic_exactly():
    reader = RpiProxyPowerReader(psutil=_RpiPsutil())
    # 40*0.5 + 50*0.3 + (4 000 000/1e6)*0.2 = 20 + 15 + 0.8
    assert reader.index() == pytest.approx(35.8)


def test_the_rpi_index_climbs_with_the_undifferenced_disk_counter():
    # The original never differences the counter, so the value grows with what
    # the machine has read since boot. Ported as-is; this test is the evidence
    # for why it is not reported as power (note Y in the audit).
    fake = _RpiPsutil(read_bytes=0, write_bytes=0)
    reader = RpiProxyPowerReader(psutil=fake)
    first = reader.index()
    fake.read_bytes = 500_000_000
    assert reader.index() > first
    assert reader.index() - first == pytest.approx(100.0)


def test_the_rpi_reader_never_reports_watts():
    reader = RpiProxyPowerReader(psutil=_RpiPsutil())
    assert reader.IS_PHYSICAL is False
    # read_w() drives the read and still returns no watts, by contract.
    assert reader.read_w() is None
    assert reader.detail() == {"activity_index": pytest.approx(35.8)}
    # And its cadence is its own original's 100 ms, not the GPU readers' 10 ms.
    assert reader.DEFAULT_INTERVAL_MS == 100


def test_the_rpi_proxy_writes_no_power_and_no_joules(tmp_path):
    path = str(tmp_path / "energy_pi.csv")
    collector = EnergyCollector(
        path, 0.01, reader=RpiProxyPowerReader(psutil=_RpiPsutil()), node="pi"
    )
    collector.start()
    time.sleep(0.05)
    collector.stop()
    rows = _rows(path)
    assert rows, "the collector wrote no samples"
    for row in rows:
        assert row["activity_index"] == "35.8"
        # The columns named for a physical quantity stay empty, which is the
        # whole point: nothing here asserts watts or joules.
        assert row["power_w"] == ""
        assert row["energy_j"] == ""
        assert row["energy_j_total"] == ""
        assert row["source"] == "rpi_proxy"
    assert collector.energy_j_total == 0.0
    # No per-inference energy can be attributed from a non-power quantity.
    assert collector.window(0.0, 1e12) is None


def test_auto_never_selects_the_rpi_proxy():
    # It would be "available" on any machine with psutil, so auto reaching it
    # would silently replace a measurement with an index.
    import cadet.metrics.energy as energy_module

    assert "rpi_proxy" not in energy_module._AUTO_ORDER
    assert "rpi_proxy" in energy_module.SOURCES
    # Named explicitly, it is built.
    assert make_power_reader("rpi_proxy").SOURCE == "rpi_proxy"


# --- energy: RAPL ----------------------------------------------------------
def _rapl_tree(tmp_path, energy_uj, wrap=1_000_000):
    root = tmp_path / "powercap"
    package = root / "intel-rapl:0"
    package.mkdir(parents=True)
    (package / "energy_uj").write_text(str(energy_uj))
    (package / "max_energy_range_uj").write_text(str(wrap))
    # A subdomain, which must NOT be counted: its energy is already inside the
    # package's, so adding it would double-count.
    subdomain = package / "intel-rapl:0:0"
    subdomain.mkdir()
    (subdomain / "energy_uj").write_text("12345")
    (subdomain / "max_energy_range_uj").write_text(str(wrap))
    return root, package


def test_rapl_counts_packages_and_not_their_subdomains(tmp_path):
    root, _package = _rapl_tree(tmp_path, 1_000_000)
    reader = RaplPowerReader(root=str(root))
    assert reader.available()[0] is True
    assert [os.path.basename(os.path.dirname(p)) for p, _w in reader.domains()] == ["intel-rapl:0"]


def test_rapl_is_a_counter_so_the_first_read_only_opens_the_interval(tmp_path):
    root, package = _rapl_tree(tmp_path, 1_000_000)
    clock = [100.0]
    reader = RaplPowerReader(root=str(root), clock=lambda: clock[0])
    reader.available()
    reader.open()
    assert reader.read_w() is None  # no interval yet
    # 2 J more over 2 s is 1 W.
    (package / "energy_uj").write_text(str(1_000_000 + 2_000_000))
    clock[0] = 102.0
    assert reader.read_w() == pytest.approx(1.0)


def test_rapl_corrects_for_the_counter_wrapping(tmp_path):
    root, package = _rapl_tree(tmp_path, 900_000, wrap=1_000_000)
    clock = [0.0]
    reader = RaplPowerReader(root=str(root), clock=lambda: clock[0])
    reader.available()
    reader.open()
    reader.read_w()
    # Wrapped: 0.9 J -> 0.1 J is 0.2 J consumed, not a negative sample.
    (package / "energy_uj").write_text("100000")
    clock[0] = 1.0
    assert reader.read_w() == pytest.approx(0.2)


def test_a_node_with_no_rapl_refuses(tmp_path):
    ok, reason = RaplPowerReader(root=str(tmp_path / "absent")).available()
    assert ok is False and "no RAPL package domains" in reason


def test_a_named_power_source_is_used_as_named_and_auto_explains_itself():
    assert select_power_reader("rapl").SOURCE == "rapl"
    assert select_power_reader("tegrastats").SOURCE == "tegrastats"
    assert select_power_reader("rpi_proxy").SOURCE == "rpi_proxy"
    # The sudo the original ran tegrastats under is threaded through.
    assert select_power_reader("tegrastats", tegrastats_sudo=False).argv()[0] == "tegrastats"
    with pytest.raises(ValueError):
        select_power_reader("a-meter-on-the-bench")
    # The stand-in that `auto` returns when nothing is readable names every
    # source it tried, so the refusal is diagnosable.
    reader = UnavailablePowerReader(["nvml: not installed", "rapl: not readable"])
    ok, reason = reader.available()
    assert ok is False and "nvml" in reason and "rapl" in reason


# --- bandwidth -------------------------------------------------------------
def test_the_transport_counts_payload_bytes_in_both_directions():
    reset_counters()
    try:
        import socket
        import threading

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        host, port = listener.getsockname()

        accepted = {}

        def accept():
            conn, _addr = listener.accept()
            accepted["transport"] = FramedSocketTransport(conn, label="server-side")

        thread = threading.Thread(target=accept)
        thread.start()
        client = FramedSocketTransport.connect(host, port, label="client-side")
        thread.join()
        server = accepted["transport"]

        client.send(b"x" * 1000)
        assert server.recv() == b"x" * 1000
        server.send(b"y" * 7)
        assert client.recv() == b"y" * 7

        counters = {c.label: c for c in counter_snapshot()}
        # Payload bytes only: the 4-byte length prefix is excluded, so framing
        # overhead stays derivable from the message count.
        assert counters["client-side"].bytes_sent == 1000
        assert counters["client-side"].messages_sent == 1
        assert counters["client-side"].bytes_recv == 7
        assert counters["server-side"].bytes_recv == 1000
        assert counters["server-side"].bytes_sent == 7
        client.close()
        server.close()
        listener.close()
    finally:
        reset_counters()


class _FakeCounters:
    def __init__(self, label, bytes_sent, messages_sent, bytes_recv, messages_recv):
        self.label = label
        self.bytes_sent = bytes_sent
        self.messages_sent = messages_sent
        self.bytes_recv = bytes_recv
        self.messages_recv = messages_recv


def test_the_bandwidth_rows_difference_the_counters_and_rate_them(tmp_path):
    series = [
        [_FakeCounters("uplink", 0, 0, 0, 0)],
        [_FakeCounters("uplink", 125_000, 10, 500, 2)],
    ]
    ticks = iter([0.0, 1.0])
    collector = BandwidthCollector(
        str(tmp_path / "bandwidth_client.csv"),
        0.01,
        node="client",
        snapshot=lambda: series.pop(0),
        clock=lambda: next(ticks),
    )
    first = collector.read()
    second = collector.read()

    sent_first = next(r for r in first if r["direction"] == "sent")
    # The first sample of a link has no window to divide by, so it carries the
    # totals without claiming a rate.
    assert sent_first["mbps"] is None
    assert sent_first["interval_s"] is None

    sent_second = next(r for r in second if r["direction"] == "sent")
    assert sent_second["bytes_total"] == 125_000
    assert sent_second["bytes_delta"] == 125_000
    assert sent_second["messages_delta"] == 10
    assert sent_second["interval_s"] == pytest.approx(1.0)
    # 125 000 B over 1 s is 1 Mbit/s.
    assert sent_second["mbps"] == pytest.approx(1.0)
    recv_second = next(r for r in second if r["direction"] == "recv")
    assert recv_second["bytes_delta"] == 500


def test_both_directions_are_reported_per_link(tmp_path):
    collector = BandwidthCollector(
        "/dev/null",
        0.01,
        snapshot=lambda: [
            _FakeCounters("a", 1, 1, 2, 1),
            _FakeCounters("b", 3, 1, 4, 1),
        ],
    )
    rows = collector.read()
    assert {(r["link"], r["direction"]) for r in rows} == {
        ("a", "sent"),
        ("a", "recv"),
        ("b", "sent"),
        ("b", "recv"),
    }


# --- what the configuration asks for becomes collectors -------------------
def test_the_collectors_a_config_names_are_built_for_that_node(tmp_path):
    runs, _base = load_run_configs(
        str(REFERENCE_CONFIG), ["metrics.collectors=[cpu,bandwidth,queue]"]
    )
    collected = build_collectors(runs[0], "server", str(tmp_path))
    assert collected.names() == ["cpu", "bandwidth"]
    assert collected.get("cpu").csv_path == str(tmp_path / "cpu_server.csv")
    assert collected.get("bandwidth").interval_s == pytest.approx(0.1)
    assert collected.get("energy") is None


def test_the_configured_interval_reaches_the_collector(tmp_path):
    runs, _base = load_run_configs(
        str(REFERENCE_CONFIG),
        ["metrics.collectors=[cpu]", "metrics.cpu.interval_ms=250"],
    )
    collected = build_collectors(runs[0], "n", str(tmp_path))
    assert collected.get("cpu").interval_s == pytest.approx(0.25)


def test_the_energy_cadence_defaults_to_the_chosen_sources_own_original(tmp_path):
    # Unset: each source's own original cadence, not one number for all.
    runs, _base = load_run_configs(
        str(REFERENCE_CONFIG),
        ["metrics.collectors=[energy]", "metrics.energy.source=rpi_proxy"],
    )
    collected = build_collectors(runs[0], "pi", str(tmp_path))
    assert collected.get("energy").interval_s == pytest.approx(0.1)

    runs, _base = load_run_configs(
        str(REFERENCE_CONFIG),
        ["metrics.collectors=[energy]", "metrics.energy.source=tegrastats"],
    )
    jetson = build_collectors(runs[0], "jet", str(tmp_path)).get("energy")
    assert jetson.interval_s == pytest.approx(0.01)

    # Set: the configured value wins for whatever source was chosen.
    runs, _base = load_run_configs(
        str(REFERENCE_CONFIG),
        [
            "metrics.collectors=[energy]",
            "metrics.energy.source=rpi_proxy",
            "metrics.energy.interval_ms=250",
        ],
    )
    pinned = build_collectors(runs[0], "pi", str(tmp_path)).get("energy")
    assert pinned.interval_s == pytest.approx(0.25)


# --- the final partial window ----------------------------------------------
#
# A collector stops when the run stops, which is never on a sampling boundary.
# Everything below drives that deliberately, because it is the one condition the
# rest of this file's tests cannot reach: they either sample many times or not
# at all. It was found across two machines, where the two ends of a link stopped
# at different moments and disagreed on totals while agreeing per message - on
# one host they stop together, so the totals matched and this hid.
def test_the_final_partial_window_is_written(tmp_path):
    path = str(tmp_path / "counting_node.csv")
    # An interval long enough that stopping is certainly mid-interval.
    collector = _Counting(path, 10.0, node="n")
    collector.start()
    collector.stop()
    rows = _rows(path)
    # Two samples: the loop's first, and the one that closes the window. Without
    # the second, a collector of a run shorter than its interval writes one row
    # and a collector stopped mid-interval loses everything since its last tick.
    assert len(rows) == 2
    assert collector.reads == 2


def test_bandwidth_totals_include_traffic_after_the_last_tick(tmp_path):
    # The reported symptom, in miniature: bytes move after the final interval
    # tick, and the totals in the file must still account for them.
    counters = [_FakeCounters("uplink", 1000, 5, 0, 0)]
    collector = BandwidthCollector(
        str(tmp_path / "bandwidth_n.csv"), 10.0, node="n", snapshot=lambda: list(counters)
    )
    collector.start()
    # Four more messages, as in the two-machine run that found this.
    counters[0] = _FakeCounters("uplink", 1000 + 4 * 198, 9, 0, 0)
    collector.stop()

    sent = [r for r in _rows(str(tmp_path / "bandwidth_n.csv")) if r["direction"] == "sent"]
    assert sent, "no rows written"
    # The last row is the authority on totals, and it has to be the post-traffic
    # one: 1000 + 4 x 198 bytes over 9 messages.
    assert int(sent[-1]["bytes_total"]) == 1000 + 4 * 198
    assert int(sent[-1]["messages_total"]) == 9
    assert int(sent[-1]["bytes_delta"]) == 4 * 198


def test_the_energy_integral_includes_the_final_partial_window(tmp_path):
    # energy_j_total is a run integral, so a missing final window is an
    # undercount of a headline number rather than one absent reading.
    path = str(tmp_path / "energy_n.csv")
    collector = EnergyCollector(path, 10.0, reader=_ScriptedPower([100.0] * 10), node="n")
    collector.start()
    time.sleep(0.05)
    collector.stop()
    rows = _rows(path)
    assert len(rows) == 2
    # The second row's interval is the partial window, and it is integrated.
    assert float(rows[1]["interval_s"]) == pytest.approx(0.05, abs=0.03)
    total = float(rows[-1]["energy_j_total"])
    assert total == pytest.approx(100.0 * sum(float(r["interval_s"]) for r in rows), rel=1e-6)
    assert total > float(rows[0]["energy_j_total"])


def test_two_ends_that_stop_at_different_moments_still_agree_on_totals(tmp_path):
    # The two-machine shape, on one host: the same link counted by two
    # collectors stopped at different times. Totals are cumulative, so the two
    # files must agree however far apart the stops are.
    shared = {"bytes": 0, "msgs": 0}

    def snapshot():
        return [_FakeCounters("link", shared["bytes"], shared["msgs"], 0, 0)]

    near = BandwidthCollector(str(tmp_path / "bandwidth_a.csv"), 10.0, node="a", snapshot=snapshot)
    far = BandwidthCollector(str(tmp_path / "bandwidth_b.csv"), 10.0, node="b", snapshot=snapshot)
    near.start()
    far.start()
    shared["bytes"], shared["msgs"] = 39_299, 200
    near.stop()
    shared["bytes"], shared["msgs"] = 39_299, 200  # no further traffic
    far.stop()

    def total(name):
        rows = [r for r in _rows(str(tmp_path / name)) if r["direction"] == "sent"]
        return int(rows[-1]["bytes_total"]), int(rows[-1]["messages_total"])

    assert total("bandwidth_a.csv") == (39_299, 200)
    assert total("bandwidth_b.csv") == (39_299, 200)


def test_rows_are_flushed_on_a_time_bound_not_only_a_row_count(tmp_path):
    # A role can be SIGKILLed or die with its container, and stop() then never
    # runs. A row-count bound alone means the loss is a function of the sampling
    # rate: at bandwidth's 100 ms, 100 rows is two and a half seconds of record.
    # The time bound makes it a known quantity, so this reads the file while the
    # collector is still running - which is what a killed role leaves behind.
    path = str(tmp_path / "counting_node.csv")
    collector = _Counting(path, 0.02, node="n")
    collector.start()
    time.sleep(_FLUSH_AFTER_S + 0.3)
    on_disk = _rows(path)  # read WITHOUT stopping the collector
    collector.stop()
    assert on_disk, "nothing reached the file before the collector was stopped"
    assert len(on_disk) < collector.rows_written + 1


def test_a_forced_flush_puts_the_buffer_on_disk_at_once(tmp_path):
    path = str(tmp_path / "counting_node.csv")
    collector = _Counting(path, 10.0, node="n")
    collector.start()
    assert _rows(path) == []  # the first row is still buffered
    collector.flush()
    assert len(_rows(path)) == 1
    collector.stop()


def test_a_set_flushes_every_collector_it_started(tmp_path):
    first = _Counting(str(tmp_path / "a_n.csv"), 10.0)
    second = _Counting(str(tmp_path / "b_n.csv"), 10.0)
    collected = CollectorSet([first, second])
    collected.start()
    collected.flush()
    assert len(_rows(str(tmp_path / "a_n.csv"))) == 1
    assert len(_rows(str(tmp_path / "b_n.csv"))) == 1
    collected.stop()


def test_the_collectors_whose_last_row_is_a_total_flush_every_sample(tmp_path):
    # The condition the two-machine run actually hit: the role never got to
    # stop() at all. Everything below reads the file while the collector is
    # still running, which is what a killed role leaves behind.
    assert BandwidthCollector.FLUSH_EVERY_SAMPLE is True
    assert EnergyCollector.FLUSH_EVERY_SAMPLE is True
    # And not for the readings that are independent samples, where a lost tail
    # is a gap rather than a wrong total.
    assert GpuCollector.FLUSH_EVERY_SAMPLE is False
    assert CpuCollector.FLUSH_EVERY_SAMPLE is False

    counters = [_FakeCounters("uplink", 39_299, 200, 21_088, 200)]
    collector = BandwidthCollector(
        str(tmp_path / "bandwidth_n.csv"), 0.01, node="n", snapshot=lambda: list(counters)
    )
    collector.start()
    time.sleep(0.05)
    # No stop(): the process is imagined to die here.
    sent = [r for r in _rows(str(tmp_path / "bandwidth_n.csv")) if r["direction"] == "sent"]
    assert sent, "a killed role left no totals on disk"
    assert int(sent[-1]["bytes_total"]) == 39_299
    assert int(sent[-1]["messages_total"]) == 200
    collector.stop()


def test_a_killed_energy_collector_still_has_its_integral_on_disk(tmp_path):
    path = str(tmp_path / "energy_n.csv")
    collector = EnergyCollector(path, 0.01, reader=_ScriptedPower([100.0] * 1000), node="n")
    collector.start()
    time.sleep(0.08)
    rows = _rows(path)  # again, without stopping
    assert rows, "a killed role left no integral on disk"
    assert float(rows[-1]["energy_j_total"]) > 0
    collector.stop()
