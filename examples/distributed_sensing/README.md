# Distributed sensing: where should object detection run?

N sensor nodes stream frames over a contested access network into one edge
gateway, and the gateway can be replaced by a cloud node reached through it. The
detector is the same stage with a different host: it runs on every sensor, on
the gateway, or in the cloud. The example measures what that choice costs.

The example needs no CARLA and no simulator, so it runs under a plain Python
install. From a fresh clone, install CADET and the example's extras (numpy,
pyyaml, and the matplotlib the plot needs) once, then drive it with the `cadet`
console script (equivalently `python -m cadet`):

```bash
pip install -e .[examples]        # once, in a fresh clone

# Validate, then run the quick config: two sensors, generated frames, no download.
cadet validate examples/distributed_sensing/quick.yaml --app-module examples.distributed_sensing.app
cadet run      examples/distributed_sensing/quick.yaml --app-module examples.distributed_sensing.app

# The whole placement comparison in ONE command: on-device vs edge vs cloud,
# across three access qualities. Runs the edge and cloud sweeps, writes the
# summary table, the crossover plot, and a dashboard-ready run tree. ~75 s for
# five sensors on this host.
python examples/distributed_sensing/placement_sweep.py --sensors 5
#   -> outputs/sensing_placement_5/summary.csv   (one row per run; a reader's table)
#   -> outputs/sensing_placement_5/summary.txt   (the console table)
#   -> outputs/sensing_placement_5/crossover.png (task metric + latency vs placement)
#   -> outputs/sensing_placement_5/<run_id>/sensing_summary.json  (per-run; the dashboard reads these)

# See the sweep in the dashboard (reads the per-run sensing_summary.json sidecars).
cadet dashboard outputs/sensing_placement_5

# Or drive the two sweeps and the reducer yourself.
cadet sweep examples/distributed_sensing/edge_5.yaml  --app-module examples.distributed_sensing.app
cadet sweep examples/distributed_sensing/cloud_5.yaml --app-module examples.distributed_sensing.app
python examples/distributed_sensing/summarise.py outputs/sensing_edge_5/distributed_sensing/*-r0
```

Configs that launch their own simulator use `scripts/cadet` instead - the
container wrapper that carries the docker client and the CARLA egg; this example
needs neither and runs under the plain `cadet` above.

The sweep output format - directory layout, the `summary.csv` schema, the
per-run `sensing_summary.json` sidecar, and which per-run file holds the task
metric, the latency and the placement - is written down in the repository-root
the columns listed in `placement_sweep.py`, which the dashboard reads.

## What it measures

Four quantities, per run:

- End-to-end latency, from a frame's capture on the sensor to the detection
  result arriving back at it.
- Age of information, from a frame's capture to its arrival at the node that
  processes it. This is the one the framework does not already collect; the
  section on metrics below says how it is derived.
- Detection rate, the fraction of frames whose result carried at least one box.
- Bytes per link, the payload each sensor put on its uplink. The placement
  decides what that payload is: boxes when the sensor detects, an encoded frame
  when it does not.

The fan-in is what the comparison turns on. Every sensor is its own process with
its own connection, and the processing node holds one detector behind one queue,
so N sensors share one compute budget while each sensor has its own.

## The configs

| Config | Topology | Sensors | Sweeps | Source |
|---|---|---|---|---|
| `quick.yaml` | sensors into gateway | 2 | nothing (one run) | generated |
| `edge_2.yaml`, `edge_5.yaml`, `edge_10.yaml` | sensors into gateway | 2, 5, 10 | placement (sensor, gateway) x access quality (3) | generated |
| `cloud_2.yaml`, `cloud_5.yaml`, `cloud_10.yaml` | sensors into gateway waypoint into cloud | 2, 5, 10 | access quality (3) | generated |
| `replay.yaml` | as `edge_5.yaml` | 5 | as `edge_5.yaml` | cached images |
| `netem_5.yaml` | sensors into gateway | 5 | as `edge_5.yaml`, on the netem backend | generated |
| `queue_5.yaml` | sensors into gateway | 5 | as `edge_5.yaml`, queue freshness policy | generated |
| `emulated_5.yaml` | sensors into gateway | 5 | nothing (one run), on the emulated wired backend | generated |
| `wifi_5.yaml` | one Wi-Fi cell into the gateway | 5 | nothing (one run) | generated |

`placement_sweep.py` is not a config: it runs `edge_N.yaml` and `cloud_N.yaml`
together and reduces both into one table and the crossover plot, so the three
placements that cannot share a sweep file are compared in one command.

Four configs differ by network backend, one per `links.network.mode`, and the
privilege each needs is part of what they demonstrate:

| config | mode | layer | what it needs |
| --- | --- | --- | --- |
| `edge_*`, `cloud_*`, `quick` | `delay` | whole messages, in the sending process | nothing - the quick start |
| `netem_5.yaml` | `netem` | packets, on the sending node's interface | each node in a container with NET_ADMIN (`launcher: docker`) |
| `emulated_5.yaml` | `netwaggle`, `medium: wired` | packets, on an emulated link between emulated hosts | the emulation container (privileged), `cadet emulate` |
| `wifi_5.yaml` | `netwaggle`, one `ap` connector | as above, plus a propagation model, with the whole cell on one shared medium | the emulation container **and** `mac80211_hwsim` in the host kernel |

The first three have been run live. See "Features used" for the fourth.

Three axes need six sweeping files rather than one. A sweep axis sets one dotted
path per value, and the sensor count changes the node, link and route lists
together, so it cannot be an axis; the placement changes the topology as well,
because "detector on the gateway" ends the path at the gateway while "detector
in the cloud" continues through it. The three access qualities, declared in
every sweeping config, are:

| Quality | `p50_ms` | `p99_ms` | `loss` | `bandwidth_mbps` |
|---|---|---|---|---|
| good | 8 | 25 | 0 | 24 |
| contested | 35 | 160 | 0.02 | 6 |
| severe | 90 | 500 | 0.08 | 1.5 |

The cloud configs add a fixed backhaul at `p50_ms: 25`, `p99_ms: 90`,
`loss: 0.005`, `bandwidth_mbps: 50`, and sweep the access quality, so a cloud
run and an edge run at the same quality differ in the path and not in the axis.

Detector cost is a config input, not a measurement: `app.params.detect_ms` in
`_base.yaml` gives the sensor 120 ms per frame, the gateway 40 ms and the cloud
12 ms, standing for a constrained sensor, an edge box and a cloud accelerator.
The shipped configs use `detector: stub`, which pays that cost and returns a box
count derived from the frame's bytes, so no weights are downloaded and the
detection rate is reproducible. Set `detector: yolo` to resolve a real model
through the model registry instead.

## What the runs showed

Measured on this host with the `delay` backend, 5 sensors, 20 generated frames
per sensor at 10 fps, from one `python examples/distributed_sensing/placement_sweep.py
--sensors 5` (which ran `edge_5.yaml` and `cloud_5.yaml`). Latencies are the
median over the answered frames of all five sensors. The source runs free of the
simulator clock, so exact numbers vary a few percent run to run; the plot it
writes, `outputs/sensing_placement_5/crossover.png`, shows the same shape.

| Placement | e2e p50, good | contested | severe | AoI p50, good | contested | severe | Uplink |
|---|---|---|---|---|---|---|---|
| sensor | 166.3 ms | 221.3 ms | 346.3 ms | 156.1 ms | 180.5 ms | 222.3 ms | 15.0 KB |
| gateway | 1147.2 ms | 1189.9 ms | 1159.6 ms | 1080.6 ms | 1065.5 ms | 1035.4 ms | 54.9 KB |
| cloud | 126.2 ms | 196.1 ms | 360.3 ms | 77.2 ms | 110.8 ms | 181.8 ms | 54.9 KB |

The gateway placement is an order of magnitude worse here because it is over
capacity: five sensors at 10 fps offer 50 frames per second, and one gateway
detector at 40 ms per frame serves 25, which is arithmetic over the values in
`edge_5.yaml` and `_base.yaml`. The cloud placement, at 12 ms per frame, serves
83 and stays ahead of the offered load. The sensor placement never queues at
all, because each sensor runs its own detector and its own source loop.

The sensor count decides which placement wins. At good access quality:

| Sensors | sensor | gateway | cloud |
|---|---|---|---|
| 2 (`placement_sweep.py --sensors 2`) | 161.5 ms | 99.5 ms | 110.0 ms |
| 5 (`placement_sweep.py --sensors 5`) | 166.3 ms | 1147.2 ms | 126.2 ms |
| 10 (`placement_sweep.py --sensors 10`) | 166.2 ms | 2674.4 ms | 465.5 ms |

At two sensors the gateway wins, because the detector's own cost dominates and
the gateway's is lower than the sensor's with less network in front of it than
the cloud's. By five sensors the gateway has saturated and the cloud wins. By ten
the gateway is 2.6 s and even the cloud's composed paths have begun to cost (the
backhaul is per-sensor), so the sensor placement - flat near 166 ms across all
three counts - is now the lowest. That flatness is what placing the detector on
the node that produced the frame buys: it is the one placement whose latency does
not depend on how many other sensors share the cell.

Bytes separate the placements more sharply on real frames than on generated
ones, because a generated frame is one flat colour and encodes to 533 bytes. The
same sweep over 20 cached images per sensor
(`cadet sweep examples/distributed_sensing/replay.yaml --app-module examples.distributed_sensing.app`) put
14.9 KB on the five uplinks with
the detector on the sensors and 6752 KB with the detector on the gateway, a
factor of 453, and the gateway run's median end-to-end latency at good access
quality was 1301.2 ms against the sensor run's 138.5 ms.

The quick config is the smallest run: `cadet run examples/distributed_sensing/quick.yaml
--app-module examples.distributed_sensing.app` answered 40 of 40 frames with a median
end-to-end latency of 76.9 ms and a median age of information of 26.1 ms, in
2.6 s of worker time (3.5 s wall, including startup and teardown).

Detection rate was 0.8 of frames carrying at least one box on the generated
source, and 0.75 to 0.8 on the cached images, in every run above. It is a
property of the stub detector rather than of the placement, and it is reported
so that a placement that drops frames can be told from one that does not.

## Metrics and where they come from

Each run writes three kinds of file under
`outputs/<experiment>/distributed_sensing/<run_id>/`:

- `events_<node>.csv`, one per sensor, one row per transmitted frame:
  `capture_time`, `send_time`, `bytes_sent`, `sensor_detect_ms`,
  `remote_detect_ms`, `detections`, `aoi_ms`, `reply_time`, `e2e_latency_ms`,
  `round_trip_ms`. The file is named after the sensor's node, which the client
  runtime passes to the app. It used to be named after the process id, and that
  was wrong in a way nothing reported: a pid is distinct between host processes
  and identical between containers, so a run with containerised sensors kept one
  sensor's log and reported one sensor.
- `arrivals_<config_hash>.csv`, one per run, beside the run directories: one row
  per frame the processing node received, with `arrival_time`, `aoi_ms`,
  `bytes_in`, `detections`, `detect_ms` and `detected_here`.
- `link_report_<node>.json`, written by the framework: the configured link or route
  parameters beside the per-message latencies one sensor observed.

Age of information needs no new collector. The framework's per-frame log
(`cadet/metrics/frame_log.py`) carries no arrival timestamp for the receiving
node and its columns are specific to the braking experiment, so it cannot be
post-processed into an age. The quantity is instead computed where both halves
exist: the sensor stamps capture time into the payload, the processing node
stamps arrival, and the difference is written to `arrivals_<config_hash>.csv`
and returned to the sensor. That holds because every role of a run shares one
host clock. Across hosts the difference between the two clocks is part of the
age: `metrics.collectors: [clock_offset]` records that difference, per run and
per node, with its uncertainty, so an age computed across machines can be
corrected by whoever reads it - the arrival stamps themselves are left on the
clock that wrote them.

`summarise.py` reduces a run to one row. Each column it prints comes from a
named column of those CSVs and from the resolved `config.yaml` the runtime saves
beside them.

With `--write` it also writes `sensing_summary.json` into each run directory:

```bash
python examples/distributed_sensing/summarise.py --write outputs/sensing_edge_5/distributed_sensing/*-r0
cadet dashboard outputs/sensing_edge_5
```

That file is what puts this example's numbers into the dashboard's sweep matrix
and its crossover plot. The dashboard reads every `*_summary.json` in a run
directory and takes the first numeric entry as the run's headline metric, so it
needs to know nothing about this example - and `e2e_p50_ms` is written first for
that reason. The step is separate from the run, and after it, because the
quantities need both the per-sensor logs and the processing node's arrival log
and no single role has both.

## Source data and its license

The replay config reads images that `fetch_frames.py` downloads at run time into
`datasets/distributed_sensing/coco-val2017`, a path `.gitignore` already excludes. Nothing
downloaded is committed or redistributed.

```bash
python examples/distributed_sensing/fetch_frames.py --limit 20
cadet sweep examples/distributed_sensing/replay.yaml --app-module examples.distributed_sensing.app
```

The fetcher reads the archive's entries in stream order and closes the
connection once it has enough, so 20 images cost 3.5 MB of a 778 MB archive.

The source is the COCO validation split. VisDrone was the first choice, and its
terms rule it out for this example. The only license
statement the VisDrone project publishes is on its own site
(<https://aiskyeye.com/data-protection/>), which states that "this dataset is
made available for academic use only" and that it "is distributed under the
Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License", with
copyright reserved by the AISKYEYE team at the Lab of Machine Learning and Data
Mining, Tianjin University. The `VisDrone-Dataset` repository carries no license
file at all. CADET is open source and its users are not limited to academic or
non-commercial work, so an example that depends on a non-commercial dataset
would hand them a restriction the example cannot lift. The downloads are also
served from Google Drive behind a confirm-token page, which an automated fetch
cannot rely on.

COCO's terms (<https://cocodataset.org/#termsofuse>) license the annotations
under Creative Commons Attribution 4.0 and state that the consortium "does not
own the copyright of the images" and that "use of the images must abide by the
Flickr Terms of Use". The example uses images only, caches them locally and
redistributes none of them. The archive is served over plain HTTP from the
project's own bucket with no authentication and no interstitial.

To use another dataset, pass `--url` to the fetcher and point
`source.path` at the result. VisDrone remains usable for work that falls within
its terms, by fetching it by hand into the same cache directory.

## Features used

Everything the configs ask for is marked implemented in
`cadet/config/implemented.py`, with one exception named below:

- `source.kind: synthetic` and `replay`
- `nodes.launcher: local`
- `links.network.mode: delay` for the sweeping configs, `netem` for
  `netem_5.yaml`, `netwaggle` with `medium: wired` for `emulated_5.yaml`
- `routes.realise: compose` for the cloud configs
- `transport.protocol: tcp`, `transport.mode: frame`

`wifi_5.yaml` adds two: `connectors[*].type: ap` and the Wi-Fi medium. No
config here sets `experiment.allow_experimental`. That one has no `links` at
all - its cell is one shared medium rather than a set of point-to-point paths -
which is why the feature it asks for is named on the connector rather than on a
link's medium.

**Running `wifi_5.yaml` takes one thing the wired configs do not.** The
emulation container must share the host's network namespace, because
`mac80211_hwsim` creates its radios there and a container with a namespace of
its own sees none of them: measured on this six-radio topology, the host's
`/sys/class/ieee80211` listed every radio and the container's listing was
empty. The container keeps its own process namespace, and it also needs the
host's `/lib/modules`, the host's Open vSwitch socket directory and
`/sys/kernel/debug` mounted. The wireless note in
`cadet/network/netwaggle/__init__.py` gives the arrangement and the reasons.

Running it that way is how three faults in the cell were found and fixed: the
plumbing interface was being overwritten by the radios, nothing associated the
stations, and the access point was never started. What the cell measures now:
every station associated, 20 of 20 messages answered per sensor, and 100
arrivals totalling 55,870 bytes at the gateway, with the round trip from
42.6 ms at its fastest and a median of 1157.8 ms, which is five sensors
queueing at one gateway rather than the air. Its topology is asserted in
`tests/unit/test_distributed_sensing.py`.

`emulated_5.yaml` is the wired form of the same thing and has been run: the same
gateway-placement cell reads 1246.9 ms median end-to-end against 1213.3 ms on
`netem` and 1176.2 ms on `delay`, so each layer adds a little; and both kernel
backends answered all 100 frames where `delay` lost 6, because configured loss
becomes retransmission latency there rather than a message that never arrives.
Two costs of that run, neither a defect in the config: the emulation container
needs `ovsdb-server` running (`service openvswitch-switch start`), and it is
privileged and runs as root, so a run with the repository mounted leaves
**root-owned** output files in it - unlike every other placement, which runs as
the invoking account. Give the emulated run an outputs directory of its own, or
expect to take ownership of what it wrote.

A plain `cadet run` of any `netwaggle` config applies no shaping and warns,
because that backend shapes in the kernel rather than in the transport. The
unprivileged path is the `delay` backend, which the quick start and the
placement sweep use.

A composed route gives one end-to-end connection per sensor, over hops whose
delays add, so a number from one path is that sensor's own path rather than a
share of a contended backhaul. That is also why "detector on the gateway" and "detector in the cloud" are
different topologies rather than one: with `compose` the gateway runs nothing, so
a gateway that processes what it forwards cannot be a waypoint.
