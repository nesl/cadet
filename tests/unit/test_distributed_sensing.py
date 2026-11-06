"""Unit tests for the distributed sensing example.

Covers the three things that could silently drift: every config in
``examples/distributed_sensing`` stays loadable and expands to the matrix the README states,
the app's placement parameter decides what goes on the wire and where the
detector runs, and the Wi-Fi config still puts the whole cell on one access
point that runs nothing. Nothing here starts a process or opens a socket; the
live paths are exercised by running the quick config.
"""

import os

import pytest
import yaml

import examples.distributed_sensing.app as sensing
from cadet.config.loader import load_run_configs
from cadet.config.schema import Config
from cadet.network.netwaggle.generator import generate

EXAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples",
    "distributed_sensing",
)

# Config name -> how many runs its sweep expands to. The counts are the example's
# contract with its README, so a config that gains or loses an axis fails here.
RUN_COUNTS = {
    "quick.yaml": 1,
    "edge_2.yaml": 6,
    "edge_5.yaml": 6,
    "edge_10.yaml": 6,
    "cloud_2.yaml": 3,
    "cloud_5.yaml": 3,
    "cloud_10.yaml": 3,
    "wifi_5.yaml": 1,
    # The same six cells as edge_5.yaml, on the packet-level backend, at the
    # medians edge_5 uses - the tails differ because netem cannot represent
    # edge_5's, which the config's header explains.
    "netem_5.yaml": 6,
    # edge_5.yaml with the processing node enforcing a freshness policy, so the
    # two matrices read against each other cell for cell.
    "queue_5.yaml": 6,
    # One cell of edge_5 on the emulated-network backend, over a wired medium:
    # the same cell measured at a third layer. One run, because the point is the
    # layer and not a sweep.
    "emulated_5.yaml": 1,
    "replay.yaml": 6,
    # One run: it has no sweep. Its axis is the placement of a *node* on a
    # second machine, which is a topology and not a value a sweep can vary.
    "two_host.yaml": 1,
}


def _load(name):
    return load_run_configs(os.path.join(EXAMPLE_DIR, name), [])[0]


@pytest.mark.parametrize("name,expected", sorted(RUN_COUNTS.items()))
def test_config_expands_to_the_documented_run_count(name, expected):
    runs = _load(name)
    assert len(runs) == expected
    for config in runs:
        assert config.app.plugin == "distributed_sensing"
        sensing.SensingParams.model_validate(config.app.params)


def test_every_config_in_the_directory_is_covered():
    """A new config has to be added to the table, not quietly left untested."""
    present = {
        name
        for name in os.listdir(EXAMPLE_DIR)
        if name.endswith(".yaml") and not name.startswith("_")
    }
    assert present == set(RUN_COUNTS)


def test_edge_sweep_crosses_placement_with_access_quality():
    runs = _load("edge_5.yaml")
    seen = {(cfg.app.params["placement"], cfg.links[0].network.distribution.p50_ms) for cfg in runs}
    assert seen == {
        (placement, p50) for placement in ("sensor", "gateway") for p50 in (8.0, 35.0, 90.0)
    }


def test_every_sensor_uplink_states_the_same_quality():
    """The cell is one access network, so its links cannot disagree."""
    for name in ("edge_2.yaml", "edge_5.yaml", "edge_10.yaml"):
        for config in _load(name):
            uplinks = [link for link in config.links if link.name.startswith("uplink_")]
            assert len(uplinks) == len([n for n in config.nodes if n.role == "client"])
            assert len({link.network.model_dump_json() for link in uplinks}) == 1


def test_cloud_config_routes_every_sensor_through_the_gateway():
    for config in _load("cloud_5.yaml"):
        gateway = next(node for node in config.nodes if node.name == "gateway")
        assert gateway.role == "waypoint"
        assert config.app.params["placement"] == "cloud"
        sensors = [node.name for node in config.nodes if node.role == "client"]
        assert len(config.routes) == len(sensors)
        for route in config.routes:
            assert route.realise == "compose"
            assert route.to == "cloud"
            assert route.via == ["gateway"]
            assert len(route.hops) == 2


SENSORS = ["sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5"]


def test_wifi_config_puts_the_whole_cell_on_one_access_point():
    """The access point is infrastructure: not a node, and running nothing."""
    config = _load("wifi_5.yaml")[0]
    topology = generate(config)
    assert [conn.name for conn in topology.connectors] == ["cell_ap"]
    assert topology.connector("cell_ap").kind == "ap"
    assert "cell_ap" not in {node.name for node in config.nodes}
    # Every member is a station, the gateway included, and it keeps its role.
    assert {host.wifi_role for host in topology.hosts} == {"station"}
    assert sorted(h.name for h in topology.hosts_on("cell_ap")) == sorted(
        SENSORS + ["gateway"]
    )
    assert next(n for n in config.nodes if n.name == "gateway").role == "server"


def test_wifi_cell_is_one_subnet_with_no_hop_between_its_members():
    """A cell, not five point-to-point links: no link, no route, one subnet."""
    config = _load("wifi_5.yaml")[0]
    topology = generate(config)
    assert topology.links == []
    names = [host.name for host in topology.hosts]
    assert all(topology.shares_subnet(a, b) for a in names for b in names)
    cidrs = [topology.cell_cidr(name) for name in names]
    assert len(set(cidrs)) == len(cidrs)
    assert {cidr.split("/")[1] for cidr in cidrs} == {topology.prefix_len}


def test_wifi_stations_sit_at_different_distances():
    config = _load("wifi_5.yaml")[0]
    positions = [n.position for n in config.nodes if n.name in SENSORS]
    assert len(set(positions)) == len(positions)


def test_params_refuse_a_placement_with_no_declared_cost():
    with pytest.raises(ValueError):
        sensing.SensingParams(placement="cloud", detect_ms={"sensor": 10.0})


def test_detector_cost_follows_the_placement():
    costs = {"sensor": 120.0, "gateway": 40.0, "cloud": 12.0}
    for placement, expected in costs.items():
        params = sensing.SensingParams(placement=placement, detect_ms=costs)
        assert params.detector_cost_s == pytest.approx(expected / 1000.0)
        assert params.detects_on_sensor == (placement == "sensor")


def _blank_frame(height=48, width=64):
    import numpy as np

    return np.zeros((height, width, 3), dtype="uint8")


def test_stub_detector_is_deterministic_for_a_frame():
    detector = sensing.StubDetector(0.0)
    first = detector.detect(_blank_frame())
    second = detector.detect(_blank_frame())
    assert first == second
    assert len(first) <= sensing.STUB_MAX_DETECTIONS
    for box in first:
        assert box["w"] == sensing.STUB_BOX_PX


def test_frame_survives_the_encode_decode_round_trip():
    frame = _blank_frame(32, 48)
    restored = sensing.decode_frame(sensing.encode_frame(frame, 80))
    assert restored.shape == frame.shape


def test_wire_message_round_trips():
    message = {"sensor": "s00001", "seq": 3, "capture_t": 1.5}
    assert sensing.unpack(sensing.pack(message)) == message


class _Frame:
    """The source frame shape the client runtime hands an app."""

    def __init__(self, image, capture_time):
        self.image = image
        self.meta = {"capture_time": capture_time, "index": 0}


def _config(tmp_path, placement, name="sensing_test"):
    """Build a config for the app with results pointed at a temporary directory."""
    data = yaml.safe_load(
        """
experiment: {name: PLACEHOLDER, app: distributed_sensing, repetitions: 1, seed: 0}
source: {kind: synthetic, count: 2, width: 64, height: 48}
models: {warmup: false}
nodes:
  - {name: sensor_1, host: localhost, role: client, device: cpu}
  - {name: gateway, host: 127.0.0.1, port: 15999, role: server, device: cpu}
app: {plugin: distributed_sensing, params: {detector: stub}}
""".replace("PLACEHOLDER", name)
    )
    data["app"]["params"]["placement"] = placement
    data["app"]["params"]["detect_ms"] = {"sensor": 0.0, "gateway": 0.0, "cloud": 0.0}
    data["outputs"] = {"root": str(tmp_path), "naming": "{experiment}/{app}/{run_id}"}
    return Config.model_validate(data)


class _Ctx:
    def __init__(self):
        self.transport = None


def test_sensor_sends_boxes_when_it_detects_and_a_frame_when_it_does_not(tmp_path):
    frame = _Frame(_blank_frame(), 1.0)
    sizes = {}
    for placement in ("sensor", "gateway"):
        config = _config(tmp_path, placement)
        app = sensing.SensorApp(config, "run", 0, str(tmp_path / "events.csv"))
        app.transport = None
        if placement == "sensor":
            app._detector = sensing.StubDetector(0.0)
        payload = app.process_frame(frame, _Ctx())
        message = sensing.unpack(payload)
        sizes[placement] = len(payload)
        if placement == "sensor":
            assert "detections" in message and "frame" not in message
        else:
            assert "frame" in message and "detections" not in message
    assert sizes["sensor"] < sizes["gateway"]


def test_processing_node_records_age_of_information(tmp_path):
    config = _config(tmp_path, "gateway")
    server = sensing.InferenceApp(config, [])
    server.setup()
    frame = _Frame(_blank_frame(), 1.0)
    sensor = sensing.SensorApp(config, "run", 0, str(tmp_path / "events.csv"))
    payload = sensor.process_frame(frame, _Ctx())
    reply = sensing.unpack(server.handle(payload))
    server.teardown()
    assert reply["seq"] == 0
    # Capture time is in the past, so the age at arrival is positive.
    assert reply["aoi_ms"] > 0
    rows = list(_read_csv(server.csv_path))
    assert len(rows) == 1
    assert rows[0]["detected_here"] == "True"
    assert float(rows[0]["bytes_in"]) == len(payload)


def test_processing_node_passes_through_boxes_detected_on_the_sensor(tmp_path):
    config = _config(tmp_path, "sensor")
    server = sensing.InferenceApp(config, [])
    server.setup()
    sensor = sensing.SensorApp(config, "run", 0, str(tmp_path / "events.csv"))
    sensor._detector = sensing.StubDetector(0.0)
    payload = sensor.process_frame(_Frame(_blank_frame(), 1.0), _Ctx())
    expected = len(sensing.unpack(payload)["detections"])
    reply = sensing.unpack(server.handle(payload))
    server.teardown()
    assert reply["detections"] == expected
    rows = list(_read_csv(server.csv_path))
    assert rows[0]["detected_here"] == "False"


def test_sensor_event_log_is_named_after_its_node(tmp_path):
    config = _config(tmp_path, "gateway")
    app = sensing.SensorApp(config, "run", 0, str(tmp_path / "events.csv"), node_name="sensor_3")
    assert os.path.basename(app.csv_path) == "events_sensor_3.csv"


def test_sensors_without_a_node_name_still_differ_between_containers(tmp_path):
    """The bug this guards is silent, which is why it is pinned.

    A sensor used to name itself after its process id. That is distinct between
    host processes and identical between containers - every node container's pid
    namespace starts at 1 - so five containerised sensors all called themselves
    the same thing, wrote to one file, and the run reported one sensor while
    each of the five wrote a well-formed log. The fallback therefore carries the
    host name, which in a container is the container.
    """
    import socket

    fallback = sensing.sensor_id()
    assert socket.gethostname() in fallback
    # And a node name, when given, is used verbatim rather than decorated.
    assert sensing.sensor_id("sensor_3") == "sensor_3"


def test_client_success_needs_an_answered_frame(tmp_path):
    config = _config(tmp_path, "gateway")
    app = sensing.SensorApp(config, "run", 0, str(tmp_path / "events.csv"))
    ok, detail = sensing.SensingApp.client_success(app)
    assert not ok and "no frames" in detail
    app._sent = 1
    ok, detail = sensing.SensingApp.client_success(app)
    assert not ok and "no result" in detail
    app._answered = 1
    ok, _detail = sensing.SensingApp.client_success(app)
    assert ok


def _read_csv(path):
    import csv

    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def test_gathered_sweep_run_dirs_recompute_to_their_own_id(tmp_path):
    """Every run dir in a gathered placement sweep recomputes to its own run id.

    The sweep wrapper (``examples/distributed_sensing/placement_sweep.py``) copies each sweep run
    directory into one tree for the dashboard. A run's identity is the hash of its
    saved ``config.yaml`` (minus ``outputs``); if the save does not round-trip, or
    the gathering copies one cell's config into another cell's directory, a dir
    would recompute to a sibling cell's id. This reproduces that whole path - write
    each cell's config the way the runtime does, gather with the wrapper's own
    copy, then recompute - and asserts identity is preserved and distinct.
    """
    import shutil

    from cadet.config.loader import config_hash, run_id

    # Write each cell's resolved config the way cadet/roles/common.py does, into a
    # per-cell run directory named by its run id, for both halves of the sweep.
    source_root = tmp_path / "source"
    run_dirs = []
    for cfg_name in ("edge_5.yaml", "cloud_5.yaml"):
        for config in load_run_configs(os.path.join(EXAMPLE_DIR, cfg_name), [])[0]:
            rid = run_id(config, 0)
            run_dir = source_root / rid
            run_dir.mkdir(parents=True)
            with open(run_dir / "config.yaml", "w") as fh:
                yaml.safe_dump(config.model_dump(mode="json", by_alias=True), fh, sort_keys=False)
            run_dirs.append(run_dir)

    # Gather, exactly as placement_sweep.rows_for does: copytree into one tree,
    # keyed by the run dir's own basename.
    gathered = tmp_path / "gathered"
    gathered.mkdir()
    for run_dir in run_dirs:
        shutil.copytree(run_dir, gathered / run_dir.name)

    # Every gathered dir's config.yaml must recompute to the id it is named by,
    # and the ids must be distinct (no cell collapsed onto a sibling).
    seen = set()
    gathered_dirs = sorted(gathered.iterdir())
    assert len(gathered_dirs) == 9  # 2 placements x 3 access + cloud x 3
    for run_dir in gathered_dirs:
        recorded = run_dir.name.split("-r")[0]
        data = yaml.safe_load((run_dir / "config.yaml").read_text())
        recomputed = config_hash(Config.model_validate(data))
        assert recomputed == recorded, "{} recomputes to {}".format(run_dir.name, recomputed)
        seen.add(recomputed)
    assert len(seen) == len(gathered_dirs)
