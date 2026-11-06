# CADET config reference

Generated from `cadet/config/schema.py` by `scripts/gen_config_reference.py`.
Do not edit by hand — re-run the script.

Each enumerated field lists the values a run may ask for. `cadet
validate` names anything else a config asks for and refuses the run.

## Link backend semantics

`links[*].network.mode` selects *where* a link is emulated, and that changes what
the other fields mean. The two families are not interchangeable and neither is
made to imitate the other.

| | `delay` | `netem`, `netwaggle` |
|---|---|---|
| layer | **message-level**: the transport's delay line holds whole application messages | **packet-level**: a kernel qdisc holds packets, and TCP reacts |
| `distribution` | sampled per message, per the configured distribution | the mean and jitter of a netem delay, applied per packet |
| `direction: forward` | delays request messages only; replies are untouched | delays *every* packet leaving the sender, including the handshake and ACKs |
| `direction: reverse` | delays reply messages only | delays every packet leaving the listener |
| `loss` | **drops application messages**; a dropped message is never answered | **drops packets**; TCP retransmits, so the message still arrives |
| `bandwidth_mbps` | serialization delay computed per message | a real rate limit on the queue |
| needs | nothing | `netem`: NET_ADMIN. `netwaggle`: the emulation container |

### What this means in practice

**A configured delay is reproduced at the message level, and at the packet
level once Nagle is out of the way.** Measured on a 60 ms forward link with the
reference app at 20 fps: `delay` gave p50 60.5 ms, while `netem` and
`netwaggle` gave p50 ≈ 150 ms. That is not emulator error — an isolated
request/response probe over the same qdisc measures 60.1 ms. It is TCP: with
small paced messages, Nagle holds a write until the previous one is
acknowledged, and the acknowledgement is itself behind the configured delay.

**`tcp_nodelay` is needed on both ends, not just the sender.** Each end holds
its own small writes, so each contributes part of the error. The same run, with
the option set on neither node, on one, and on both:

| `tcp_nodelay` | p50 | min |
|---|---|---|
| neither | 150.7 ms | 99.5 ms |
| client only | 110.4 ms | 68.6 ms |
| server only | 90.9 ms | 60.6 ms |
| both | **60.5 ms** | 60.3 ms |

**The send rate decides whether it matters at all.** Nagle only has something
to wait for while a previous small write is unacknowledged, so the error
appears when the send interval approaches the shaped round trip. The same
60 ms link, `tcp_nodelay` unset, swept across that boundary:

| send interval | p50 |
|---|---|
| 200 ms (5 fps) | 60.5 ms |
| 100 ms (10 fps) | 60.5 ms |
| 71 ms (14 fps) | 143.2 ms (mixed: `min` still 60.4 ms) |
| 50 ms (20 fps) | 150.2 ms |
| 33 ms (30 fps) | 150.7 ms |

So a packet-level measurement taken at a send interval comfortably above the
shaped round trip needs nothing; one taken near or below it, without
`tcp_nodelay` on both ends, measures the link plus an extra round trip. The
message-level backend is unaffected either way — it holds whole application
messages in process, so TCP's write batching never enters.

**Configured loss means different things.** `delay` loses messages, so a 1% loss
produced 1.25% unanswered frames. `netem` and `netwaggle` lose packets under a
reliable transport, so the same 1% produced **0% unanswered** and showed up as
retransmission latency in p99 instead. If an experiment needs observable message
loss, it needs `delay` or an unreliable transport.

**Choose by question, not by fidelity.** Use `delay` to state a per-message
latency distribution and have it honoured. Use `netem` or `netwaggle` to ask what
a real protocol stack does on a delayed, lossy, rate-limited path. `netwaggle`
additionally gives a real topology, a wireless medium and propagation model, and
hosts in separate namespaces.

Every run writes `link_report_<node>.json` per client, next to its metrics,
holding the configured
parameters beside the measured per-message latency and unanswered counts, so the
gap is visible per run rather than assumed.

## Shared media: `connectors`

A `links` entry is point to point: one `from`, one `to`, its own L2 segment and
its own qdiscs. Some media are not that shape. Several stations associating with
one access point share one medium, sit in **one subnet**, and reach each other
over the air with no L3 hop and no route between them - which is exactly what
two point-to-point links in series cannot express.

A `connectors` entry is that shape. Nodes name the connector they attach to
(`nodes[*].connector`) and, on an access point, where they sit
(`nodes[*].position`, `"x,y,z"` in metres). Two types:

| `type` | what it is | its members |
|---|---|---|
| `ap` | an access point, with the cell's radio parameters in `wifi` and its own `position` | stations, each at its own position, which is what the propagation model acts on |
| `switch` | the same idea without radios: a wired shared segment | wired hosts, attached with a plain veth |

Three things follow, and they are the point rather than side effects.

**A connector runs nothing.** It is deliberately not a `nodes` entry: there is
no role to dispatch on it and no address on it to bind. In Mininet-WiFi an
access point is a switch in the root namespace, so a role process placed on one
dies with `Cannot assign requested address`; keeping the two sections apart is
what makes that config unexpressible rather than merely discouraged.

**A cell's members share a subnet and get no route to each other.** Each member
carries the shaped subnet's own prefix on its connector-facing interface, so the
others are on-link. A point-to-point endpoint still gets a `/32` plus a host
route to its peer, and the two coexist in one subnet by longest match, so a node
on a cell *and* on a link sends each peer's traffic out of the right interface.

**A connector is shaped by its medium, not by a qdisc.** A cell's loss and delay
come from the propagation model and the distances between positions, which is
why such an experiment varies the distances. A connector therefore has no
`distribution`, `loss` or `bandwidth_mbps`; a path that needs those is a `links`
entry. Both sections can appear in one config, and a config that mentions no
connector behaves exactly as it did before they existed.

Connectors exist only in the `netwaggle` backend, like a `netwaggle` link: under
`cadet run` the topology is not built and the nodes talk over the host's own
stack.

### `Config`

A complete CADET experiment configuration.

| field | type | default | notes |
|---|---|---|---|
| `experiment` | Experiment | — (required) |  |
| `source` | CarlaSource | ReplaySource | SyntheticSource | LiveSource | DatasetSource | — (required) |  |
| `models` | Models | (defaults) |  |
| `nodes` | list[Node] | — (required) |  |
| `channels` | list[Channel] | `[]` |  |
| `connectors` | list[Connector] | `[]` |  |
| `links` | list[Link] | `[]` |  |
| `routes` | list[Route] | `[]` |  |
| `transport` | TransportConfig | (defaults) |  |
| `metrics` | MetricsConfig | (defaults) |  |
| `app` | App | — (required) |  |
| `pipeline` | PipelineConfig (optional) | `None` |  |
| `sweep` | Sweep | (defaults) |  |
| `outputs` | Outputs | (defaults) |  |
| `analysis` | AnalysisConfig | (defaults) |  |

### `Experiment`

Identity and repetition policy for a run family.

| field | type | default | notes |
|---|---|---|---|
| `name` | str | — (required) |  |
| `app` | str | — (required) |  |
| `repetitions` | int | `1` |  |
| `seed` | int (optional) | `None` |  |
| `max_startup_s` | float (optional) | `None` |  |
| `max_active_s` | float (optional) | `None` |  |
| `end_on` | one of {'client_report', 'sim_hook_exit', 'first'} | `client_report` |  |
| `end_grace_s` | float | `5.0` |  |
| `client_start` | one of {'before_scenario', 'after_scenario'} | `after_scenario` |  |
| `require_output` | bool | `False` |  |
| `inter_run_pause_s` | float | `5.0` |  |
| `resume` | bool | `False` |  |
| `allow_experimental` | bool | `False` |  |

### `CarlaSource`

Input from a running CARLA simulator (the only live backend today).

| field | type | default | notes |
|---|---|---|---|
| `kind` | one of {'carla'} | `carla` |  |
| `carla` | CarlaSettings | (defaults) |  |
| `fps` | int | `10` |  |
| `sync_mode` | bool | `True` |  |
| `camera` | Camera | (defaults) |  |
| `scenario` | Scenario | (defaults) |  |
| `weather` | one of {'clear', 'light', 'high'} (optional) | `None` |  |

### `CarlaSettings`

How a run reaches CARLA, and - when it launches one - how it starts it.

| field | type | default | notes |
|---|---|---|---|
| `host` | str | `127.0.0.1` |  |
| `port` | int | `2000` |  |
| `timeout` | int | `10` |  |
| `image` | str (optional) | `None` |  |
| `gpu_index` | int (optional) | `None` |  |
| `quality` | one of {'Low', 'Epic'} | `Low` |  |
| `offscreen` | bool | `True` |  |
| `per_run` | bool | `False` |  |
| `client_egg` | CarlaClientEgg (optional) | `None` |  |

### `CarlaClientEgg`

Which CARLA client egg a run must use, and where to find it.

| field | type | default | notes |
|---|---|---|---|
| `name` | str | — (required) |  |
| `host_path` | str (optional) | `None` |  |

### `Camera`

Ego camera sensor geometry.

| field | type | default | notes |
|---|---|---|---|
| `width` | int | `960` |  |
| `height` | int | `512` |  |
| `fov` | float | `90.0` |  |
| `mount_x` | float | `2.5` |  |
| `mount_z` | float | `1.8` |  |
| `mount_pitch` | float | `-10.0` |  |
| `queue_size` | int | `10` |  |
| `fixed_rate` | bool | `False` |  |

### `Scenario`

Scenario geometry and lifetime (fields optional; apps use a subset).

| field | type | default | notes |
|---|---|---|---|
| `map` | str (optional) | `None` |  |
| `ego_vehicle` | str (optional) | `None` |  |
| `target_object` | str (optional) | `None` |  |
| `ego_speed` | str (optional) | `None` |  |
| `ego_spawn` | Spawn (optional) | `None` |  |
| `occluder` | Occluder (optional) | `None` |  |
| `pedestrian` | Pedestrian (optional) | `None` |  |
| `timeout_s` | float (optional) | `None` |  |
| `ego_role_name` | str (optional) | `hero` |  |
| `ego_attach_timeout_s` | float | `60.0` |  |
| `runner` | ScenarioRunnerPin (optional) | `None` |  |
| `name` | str (optional) | `None` |  |
| `definition` | str (optional) | `None` |  |
| `config_file` | str (optional) | `None` |  |

### `Spawn`

A world pose (metres, degrees) for spawning an actor.

| field | type | default | notes |
|---|---|---|---|
| `x` | float | — (required) |  |
| `y` | float | — (required) |  |
| `z` | float | `0.0` |  |
| `yaw` | float | `0.0` |  |

### `Occluder`

A parked actor used to block the ego's line of sight.

| field | type | default | notes |
|---|---|---|---|
| `model` | str | — (required) |  |
| `spawn` | Spawn | — (required) |  |

### `Pedestrian`

A pedestrian that holds, then crosses at ``speed_mps`` after ``t_start_s``.

| field | type | default | notes |
|---|---|---|---|
| `spawn` | Spawn | — (required) |  |
| `speed_mps` | float | — (required) |  |
| `t_start_s` | float | — (required) |  |

### `ScenarioRunnerPin`

Where the ScenarioRunner tree is, and which tree it must be.

| field | type | default | notes |
|---|---|---|---|
| `path` | str | — (required) |  |
| `source_digest` | str (optional) | `None` |  |
| `commit` | str (optional) | `None` |  |
| `image` | str (optional) | `None` |  |

### `ReplaySource`

Deterministic replay from a folder of images or a video file.

| field | type | default | notes |
|---|---|---|---|
| `kind` | one of {'replay'} | `replay` |  |
| `path` | str | — (required) |  |
| `fps` | float | `10.0` |  |
| `loop` | bool | `False` |  |
| `limit` | int (optional) | `None` |  |

### `SyntheticSource`

Generated frames for tests (no files, no hardware).

| field | type | default | notes |
|---|---|---|---|
| `kind` | one of {'synthetic'} | `synthetic` |  |
| `count` | int | `10` |  |
| `width` | int | `64` |  |
| `height` | int | `48` |  |
| `fps` | float | `0.0` |  |

### `LiveSource`

Live capture: a camera device index or a network stream URL.

| field | type | default | notes |
|---|---|---|---|
| `kind` | one of {'live'} | `live` |  |
| `device` | int (optional) | `None` |  |
| `url` | str (optional) | `None` |  |
| `fps` | float | `0.0` |  |
| `limit` | int (optional) | `None` |  |

### `DatasetSource`

Deterministic dataset replay for model evaluation (the "Dataset Mode").

| field | type | default | notes |
|---|---|---|---|
| `kind` | one of {'dataset'} | `dataset` |  |
| `task` | one of {'vqa', 'detection'} | `vqa` |  |
| `path` | str (optional) | `None` |  |
| `manifest` | str (optional) | `None` |  |
| `fps` | float | `0.0` |  |
| `limit` | int (optional) | `None` |  |
| `count` | int | `8` |  |

### `Models`

Pipeline component choices plus the logical-name -> artifact registry.

| field | type | default | notes |
|---|---|---|---|
| `detector` | str | `yolo` |  |
| `tracker` | str | `sort` |  |
| `predictor` | str | `linear` |  |
| `model` | str (optional) | `None` |  |
| `warmup` | bool | `True` |  |
| `registry` | map[str -> ModelEntry] | `{}` |  |

### `ModelEntry`

A logical model mapped to a concrete artifact per device type.

| field | type | default | notes |
|---|---|---|---|
| `per_device` | map[str -> str] | — (required) |  |

### `Node`

A compute node that hosts a role (client/server/sim).

| field | type | default | notes |
|---|---|---|---|
| `name` | str | — (required) |  |
| `host` | str | — (required) |  |
| `port` | int (optional) | `None` |  |
| `launcher` | one of {'local', 'docker', 'ssh_docker', 'ssh_venv'} | `local` |  |
| `role` | one of {'client', 'server', 'sim', 'leader', 'load', 'waypoint'} | — (required) |  |
| `device` | str | `cpu` |  |
| `gpu_index` | int (optional) | `None` |  |
| `scheduler` | one of {'fifo', 'round_robin'} (optional) | `None` |  |
| `extra_processing_ms` | float (optional) | `None` |  |
| `server_delay_mode` | one of {'blocking', 'link'} | `blocking` |  |
| `enforce_queue_policy` | bool | `False` |  |
| `inbound_block_at` | int (optional) | `None` |  |
| `clients` | int (optional) | `None` |  |
| `rate_hz` | float (optional) | `None` |  |
| `arrival` | one of {'constant', 'poisson'} (optional) | `None` |  |
| `frames` | int (optional) | `None` |  |
| `transport_tuning` | TransportTuning (optional) | `None` |  |
| `connector` | str (optional) | `None` |  |
| `position` | str (optional) | `None` |  |
| `container` | str (optional) | `None` |  |
| `venv` | str (optional) | `None` |  |
| `user` | str (optional) | `None` |  |
| `ssh_port` | int (optional) | `None` |  |
| `ssh_identity` | str (optional) | `None` |  |

### `TransportTuning`

TCP stack settings for one node, applied only in that node's namespace.

| field | type | default | notes |
|---|---|---|---|
| `tcp_nodelay` | bool (optional) | `None` |  |
| `tcp_slow_start_after_idle` | bool (optional) | `None` |  |
| `initcwnd` | int (optional) | `None` |  |

### `Channel`

A direct data path between two nodes (no network emulation).

| field | type | default | notes |
|---|---|---|---|
| `name` | str | — (required) |  |
| `from` | str | — (required) |  |
| `to` | str | — (required) |  |
| `stream_fps` | int (optional) | `None` |  |
| `max_queue` | int | `2` |  |
| `send_meta` | bool | `False` |  |
| `stop_on_brake` | bool | `False` |  |
| `device_id` | str (optional) | `None` |  |

### `Connector`

A shared medium that nodes attach to, and that runs nothing itself.

| field | type | default | notes |
|---|---|---|---|
| `name` | str | — (required) |  |
| `type` | one of {'ap', 'switch'} | — (required) |  |
| `position` | str (optional) | `None` |  |
| `wifi` | WifiMedium (optional) | `None` |  |

### `WifiMedium`

Radio parameters of one wireless medium, with no endpoints in them.

| field | type | default | notes |
|---|---|---|---|
| `ssid` | str | `cadet` |  |
| `mode` | one of {'a', 'b', 'g', 'n', 'ac'} | `g` |  |
| `channel` | int | `1` |  |
| `propagation_model` | one of {'logDistance', 'friis', 'twoRayGround', 'logNormalShadowing', 'ITU'} | `logDistance` |  |
| `propagation_exp` | float | `3.5` |  |
| `propagation_params` | map[str -> float] | `{}` |  |
| `mobility_trace` | str (optional) | `None` |  |

### `Link`

A network-emulated data path between two nodes.

| field | type | default | notes |
|---|---|---|---|
| `name` | str | — (required) |  |
| `from` | str | — (required) |  |
| `to` | str | — (required) |  |
| `direction` | one of {'forward', 'reverse', 'both'} | `forward` |  |
| `network` | NetworkModel | (defaults) |  |

### `NetworkModel`

Emulated-network behaviour applied to a link's traffic.

| field | type | default | notes |
|---|---|---|---|
| `mode` | one of {'none', 'delay', 'netem', 'netwaggle'} | `none` |  |
| `medium` | one of {'wired', 'wifi'} | `wired` |  |
| `distribution` | ConstantDist | NormalDist | LognormalDist | TraceDist | (defaults) |  |
| `loss` | float | `0.0` |  |
| `bandwidth_mbps` | float (optional) | `None` |  |
| `wifi` | WifiConfig (optional) | `None` |  |

### `ConstantDist`

Fixed per-packet delay.

| field | type | default | notes |
|---|---|---|---|
| `kind` | one of {'constant'} | `constant` |  |
| `value_ms` | float | `0.0` |  |

### `NormalDist`

Gaussian delay.

| field | type | default | notes |
|---|---|---|---|
| `kind` | one of {'normal'} | `normal` |  |
| `mean_ms` | float | — (required) |  |
| `std_ms` | float | — (required) |  |

### `LognormalDist`

Log-normal delay (long tail), stated as two percentiles.

| field | type | default | notes |
|---|---|---|---|
| `kind` | one of {'lognormal'} | `lognormal` |  |
| `p50_ms` | float | — (required) |  |
| `p99_ms` | float | — (required) |  |

### `TraceDist`

Replay delays from a recorded trace file.

| field | type | default | notes |
|---|---|---|---|
| `kind` | one of {'trace'} | `trace` |  |
| `path` | str | — (required) |  |

### `WifiConfig`

A point-to-point wireless link's medium plus the radio role of each end.

| field | type | default | notes |
|---|---|---|---|
| `ssid` | str | `cadet` |  |
| `mode` | one of {'a', 'b', 'g', 'n', 'ac'} | `g` |  |
| `channel` | int | `1` |  |
| `propagation_model` | one of {'logDistance', 'friis', 'twoRayGround', 'logNormalShadowing', 'ITU'} | `logDistance` |  |
| `propagation_exp` | float | `3.5` |  |
| `propagation_params` | map[str -> float] | `{}` |  |
| `mobility_trace` | str (optional) | `None` |  |
| `nodes` | map[str -> WifiNodeSpec] | — (required) |  |

### `WifiNodeSpec`

One endpoint's radio role and position on a wireless link.

| field | type | default | notes |
|---|---|---|---|
| `role` | one of {'ap', 'station'} | — (required) |  |
| `position` | str (optional) | `None` |  |

### `Route`

A path from one node to another across several links, in order.

| field | type | default | notes |
|---|---|---|---|
| `name` | str | — (required) |  |
| `from` | str | — (required) |  |
| `to` | str | — (required) |  |
| `hops` | list[str] | — (required) |  |
| `realise` | one of {'compose', 'relay'} | `compose` |  |
| `via` | list[str] (optional) | `None` |  |

### `TransportConfig`

Wire transport for node-to-node traffic.

| field | type | default | notes |
|---|---|---|---|
| `protocol` | one of {'tcp', 'websocket'} | `tcp` |  |
| `mode` | one of {'frame', 'stream'} | `frame` |  |
| `encoding` | one of {'safe', 'pickle'} | `safe` |  |

### `MetricsConfig`

Optional measurement collectors to run during an experiment.

| field | type | default | notes |
|---|---|---|---|
| `collectors` | list[one of {'gpu', 'cpu', 'bandwidth', 'queue', 'energy', 'clock_offset'}] | `[]` |  |
| `gpu` | GpuMetrics | (defaults) |  |
| `cpu` | CpuMetrics | (defaults) |  |
| `energy` | EnergyMetrics | (defaults) |  |
| `bandwidth` | BandwidthMetrics | (defaults) |  |
| `clock_offset` | ClockOffsetMetrics | (defaults) |  |

### `GpuMetrics`

Options for the ``gpu`` collector (utilization, memory, power).

| field | type | default | notes |
|---|---|---|---|
| `interval_ms` | int | `10` |  |
| `source` | one of {'auto', 'nvml', 'nvidia-smi'} | `auto` |  |
| `devices` | list[int] (optional) | `None` |  |

### `CpuMetrics`

Options for the ``cpu`` collector (host and per-process CPU and memory).

| field | type | default | notes |
|---|---|---|---|
| `interval_ms` | int | `100` |  |
| `per_process` | bool | `True` |  |

### `EnergyMetrics`

Options for the ``energy`` collector (power over the run, per inference).

| field | type | default | notes |
|---|---|---|---|
| `interval_ms` | int (optional) | `None` |  |
| `source` | one of {'auto', 'nvml', 'nvidia-smi', 'tegrastats', 'tegra_sysfs', 'rapl', 'rpi_proxy'} | `auto` |  |
| `devices` | list[int] (optional) | `None` |  |
| `tegrastats_sudo` | bool | `True` |  |

### `BandwidthMetrics`

Options for the ``bandwidth`` collector (bytes per link and direction).

| field | type | default | notes |
|---|---|---|---|
| `interval_ms` | int | `100` |  |

### `ClockOffsetMetrics`

Options for the ``clock_offset`` collector (per-node clock difference).

| field | type | default | notes |
|---|---|---|---|
| `method` | one of {'auto', 'round_trip', 'time_service'} | `auto` |  |
| `probes` | int | `11` |  |
| `timeout_s` | float | `15.0` |  |

### `App`

App selector + its raw parameters.

| field | type | default | notes |
|---|---|---|---|
| `plugin` | str | — (required) |  |
| `params` | map[str -> Any] | `{}` |  |

### `PipelineConfig`

The ordered list of stages that make up a staged application.

| field | type | default | notes |
|---|---|---|---|
| `stages` | list[StageSpec] | `[]` |  |

### `StageSpec`

One placed stage: an impl from the registry assigned to a node.

| field | type | default | notes |
|---|---|---|---|
| `name` | str | — (required) |  |
| `impl` | str | — (required) |  |
| `node` | str | — (required) |  |
| `params` | map[str -> Any] | `{}` |  |
| `queue_policy` | one of {'latest_only', 'fifo'} | `latest_only` |  |
| `deadline` | DeadlineSpec (optional) | `None` |  |

### `DeadlineSpec`

A per-stage time-to-decision budget (ported deadline-enforcement semantics).

| field | type | default | notes |
|---|---|---|---|
| `policy` | one of {'none', 'static', 'dynamic'} | `none` |  |
| `budget_ms` | float (optional) | `None` |  |

### `Sweep`

Cartesian sweep over dotted config paths -> list of values.

| field | type | default | notes |
|---|---|---|---|
| `axes` | map[str -> list[Any]] | `{}` |  |

### `Outputs`

Where run artifacts go. By policy, results/weights are never committed.

| field | type | default | notes |
|---|---|---|---|
| `root` | str | `outputs` |  |
| `naming` | str | `{experiment}/{app}/{run_id}` |  |
| `save` | list[str] | `[]` |  |

### `AnalysisConfig`

Where ``cadet summarize`` / ``cadet ask`` send a run's reduced record.

| field | type | default | notes |
|---|---|---|---|
| `backend` | one of {'none', 'openai_chat'} | `none` |  |
| `endpoint` | str (optional) | `None` |  |
| `model` | str (optional) | `None` |  |
| `api_key_env` | str | `CADET_FM_API_KEY` |  |
| `timeout_s` | float | `60.0` |  |
| `max_context_bytes` | int | `200000` |  |
| `temperature` | float | `0.0` |  |
