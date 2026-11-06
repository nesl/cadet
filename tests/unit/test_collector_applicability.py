"""Tests for collectors on a node that has nothing for them to measure.

What this is for
    A config asks for collectors once, for the whole experiment, and the nodes
    it places are not alike: a detection server gets a GPU, the client driving
    it does not. Asking every node for GPU power therefore asks some nodes for
    a measurement they cannot make. That is not the same as a node that was
    meant to make it and failed, and the two must not share an outcome: the
    first is an absence to record, the second is a run to stop.

    This was found in the acceptance matrix. The netem cells place their
    sender in a container, that container is given no GPU because the node is
    declared ``device: cpu``, and the run died on
    ``nvidia-smi is not on PATH`` - for a node that never had a GPU to read.

What it contains
    Cases for both directions: the GPU-less node records the absence and the
    run continues, and the GPU node still fails loudly when its reader cannot
    be read.

How it fits into CADET
    Unit tests over :mod:`cadet.metrics.collectors`, :mod:`cadet.metrics.gpu`,
    :mod:`cadet.metrics.energy` and :mod:`cadet.metrics.device_summary`.
"""

import csv
import json

import pytest

from cadet.metrics.collectors import NOT_APPLICABLE_SUFFIX, CollectorError
from cadet.metrics.device_summary import summarise_run, write_summary
from cadet.metrics.gpu import GpuCollector, node_has_gpu


class _UnreadableReader:
    """A GPU reader on a machine without the tool it needs."""

    SOURCE = "nvidia-smi"

    def available(self):
        return False, "nvidia-smi is not on PATH"

    def open(self):
        raise AssertionError("must not be opened when it is unavailable")

    def read(self):
        return []

    def close(self):
        pass


def _collector(tmp_path, has_gpu, node="client"):
    return GpuCollector(
        str(tmp_path / "gpu_{}.csv".format(node)),
        0.01,
        reader=_UnreadableReader(),
        node=node,
        has_gpu=has_gpu,
    )


def test_a_node_with_no_gpu_records_the_absence_instead_of_failing(tmp_path):
    """The run goes on, and says why the file is empty."""
    collector = _collector(tmp_path, has_gpu=False)

    collector.start()  # must not raise
    collector.stop()  # must be safe with no sampling thread

    csv_path = tmp_path / "gpu_client.csv"
    marker = tmp_path / ("gpu_client.csv" + NOT_APPLICABLE_SUFFIX)
    assert csv_path.exists(), "the file is still written, so it is not just missing"
    assert marker.exists(), "and the reason is recorded beside it"
    assert "no GPU on this node" in marker.read_text()

    # Header and no samples: well-formed for anything that reads it.
    rows = list(csv.reader(csv_path.open()))
    assert len(rows) == 1
    assert rows[0][:3] == ["t_wall", "t_monotonic", "node"]


def test_a_gpu_node_that_cannot_read_still_fails_loudly(tmp_path):
    """The case the absence path must not swallow.

    This node was given a GPU and was meant to produce these figures. Reporting
    the gap as "not applicable" would turn a broken measurement into a quiet
    one, which is the failure this whole distinction exists to avoid.
    """
    collector = _collector(tmp_path, has_gpu=True)

    with pytest.raises(CollectorError) as caught:
        collector.start()

    message = str(caught.value)
    assert "cannot read on node 'client'" in message
    assert "nvidia-smi is not on PATH" in message
    assert not (tmp_path / ("gpu_client.csv" + NOT_APPLICABLE_SUFFIX)).exists()


def test_whether_a_node_has_a_gpu_comes_from_the_config():
    """The node's own declaration, not a probe of the machine.

    A containerised node is launched without GPU access when it is declared
    ``device: cpu``, even on a host with eight of them, so the config is the
    only thing that knows.
    """

    class _Node:
        def __init__(self, name, device):
            self.name = name
            self.device = device

    class _Config:
        nodes = [_Node("client", "cpu"), _Node("server", "gpu")]

    assert node_has_gpu(_Config(), "server") is True
    assert node_has_gpu(_Config(), "client") is False
    # A node this config does not place is left as it was before.
    assert node_has_gpu(_Config(), "elsewhere") is True


def test_the_device_summary_states_the_absence(tmp_path):
    """A reader of the summary sees "nothing to measure", not nothing at all."""
    (tmp_path / "gpu_client.csv").write_text("t_wall,t_monotonic,node,source\n")
    (tmp_path / ("gpu_client.csv" + NOT_APPLICABLE_SUFFIX)).write_text(
        "not applicable: no GPU on this node\n"
    )

    summary = summarise_run(str(tmp_path))

    assert summary["nodes"]["client"]["gpu"] == {
        "not_applicable": "not applicable: no GPU on this node"
    }

    path = write_summary(str(tmp_path))
    assert path is not None
    with open(path) as handle:
        assert "no GPU on this node" in json.load(handle)["nodes"]["client"]["gpu"][
            "not_applicable"
        ]


def test_the_marker_suffix_matches_the_one_the_collectors_write():
    """The summary pins the name rather than importing it; keep them equal."""
    from cadet.metrics import device_summary

    assert device_summary._NOT_APPLICABLE_SUFFIX == NOT_APPLICABLE_SUFFIX


def test_energy_on_a_gpu_less_node_falls_back_to_the_cpu_counters():
    """A node with no GPU still has power to report, where it can be read.

    `auto` must not reach for the GPU sources on a node that has none: it would
    report each as unavailable on the way past and record the reasons as though
    two real sources had failed.
    """
    from cadet.metrics import energy

    tried = []

    class _Reader:
        DEFAULT_INTERVAL_MS = 100

        def __init__(self, source, ok):
            self.SOURCE = source
            self._ok = ok

        def available(self):
            tried.append(self.SOURCE)
            return (True, "") if self._ok else (False, "not here")

        def close(self):
            pass

    def _make(source, **_kwargs):
        # Only the CPU package counters answer on this machine.
        return _Reader(source, ok=(source == "rapl"))

    original = energy.make_power_reader
    energy.make_power_reader = _make
    try:
        chosen = energy.select_power_reader(
            source="auto", skip_sources=energy.EnergyCollector.GPU_SOURCES
        )
    finally:
        energy.make_power_reader = original

    assert chosen.SOURCE == "rapl"
    assert "nvml" not in tried and "nvidia-smi" not in tried, tried


def test_energy_is_not_applicable_when_nothing_on_the_node_reports_power(tmp_path):
    """The end of the fallback line: no GPU, no readable counters, nothing."""
    from cadet.metrics.energy import EnergyCollector, UnavailablePowerReader

    collector = EnergyCollector(
        str(tmp_path / "energy_client.csv"),
        0.01,
        reader=UnavailablePowerReader(["rapl: permission denied"]),
        node="client",
    )

    applies, reason = collector.applicable()
    assert applies is False
    assert "not applicable" in reason
    assert "rapl: permission denied" in reason

    collector.start()
    collector.stop()
    marker = tmp_path / ("energy_client.csv" + NOT_APPLICABLE_SUFFIX)
    assert marker.exists()
    assert "rapl" in marker.read_text()


def test_a_named_energy_source_that_cannot_be_read_is_still_a_failure(tmp_path):
    """Asking for one source by name is asking for that source.

    The fallback is what `auto` means. A config that names `rapl` and cannot
    read it has a broken measurement, not an absent one.
    """
    from cadet.metrics.energy import EnergyCollector

    class _NamedButUnreadable:
        SOURCE = "rapl"

        def available(self):
            return False, "permission denied"

        def close(self):
            pass

    collector = EnergyCollector(
        str(tmp_path / "energy_server.csv"),
        0.01,
        reader=_NamedButUnreadable(),
        node="server",
    )

    applies, _reason = collector.applicable()
    assert applies is True, "a named source is not excused by the absence path"
    with pytest.raises(CollectorError):
        collector.start()
