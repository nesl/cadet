"""Load generator role: N synthetic clients at a configured rate -> server(s).

What this is for:
    Drives concurrent-load experiments. A ``load`` node describes N synthetic
    clients, a send rate, and an arrival process (constant or Poisson). Each
    client connects to every ``server`` node (fan-out) and sends frames in the
    reference-app wire format, recording round-trip latency. Combined with the
    server's per-request metrics (queue wait vs load), this measures contention.

What it contains:
    - :class:`LoadRunner`: spawn clients, send at rate, collect latencies.
    - :func:`build_load`: construct from a config.

How it fits into CADET:
    ``cadet load <config> --app-module examples.reference.app``. Targets all
    ``server`` nodes in the config: ``clients=10, servers=[s]`` is 10->1;
    ``clients=1, servers=[s1,s2]`` is 1->2.
"""

import csv
import os
import random
import statistics
import time
from queue import Empty, Queue
from threading import Lock, Thread
from typing import List

from cadet.core.codec import decode as wire_decode
from cadet.core.codec import encode as wire_encode
from cadet.core.codec import set_wire_encoding_from_config
from cadet.core.transport import FramedSocketTransport
from cadet.roles.common import select_run_config

# Defaults used when a load node omits the corresponding field.
DEFAULT_CLIENTS = 1
DEFAULT_FRAMES = 10
DEFAULT_RATE_HZ = 10.0
# Generated frame geometry (H, W); matches SyntheticSource's default size.
LOAD_FRAME_H = 48
LOAD_FRAME_W = 64
# Per-client RNG seed base, offset by the client index for reproducibility.
CLIENT_SEED_BASE = 1000
# How long to connect-retry while a server is still loading its model.
CONNECT_RETRY_S = 30.0
# After sending all frames, how long to wait for outstanding replies.
DRAIN_WINDOW_S = 15.0
# Poll interval while draining the reply queue.
REPLY_POLL_S = 1.0


class LoadRunner:
    """Runs a fleet of synthetic clients against the config's server nodes."""

    def __init__(self, config, csv_path, logger=None):
        self.cfg = config
        self.csv_path = csv_path
        self.log = logger or (lambda m: print(m))
        self.load_node = next((n for n in config.nodes if n.role == "load"), None)
        if self.load_node is None:
            raise RuntimeError("config has no node with role 'load'")
        self.servers = [n for n in config.nodes if n.role == "server"]
        if not self.servers:
            raise RuntimeError("config has no server nodes to target")

    def run(self) -> None:
        import numpy as np

        clients = self.load_node.clients or DEFAULT_CLIENTS
        frames = self.load_node.frames or DEFAULT_FRAMES
        rate = self.load_node.rate_hz or DEFAULT_RATE_HZ
        arrival = self.load_node.arrival or "constant"
        results = []  # (client, target, frame_id, latency_s)
        lock = Lock()
        img = np.zeros((LOAD_FRAME_H, LOAD_FRAME_W, 3), dtype=np.uint8)

        def one_client(ci: int) -> None:
            for srv in self.servers:
                target = "{}:{}".format(srv.host, srv.port)
                transport = FramedSocketTransport.connect(
                    srv.host,
                    srv.port,
                    retry_s=CONNECT_RETRY_S,
                    protocol=self.cfg.transport.protocol,
                    mode=self.cfg.transport.mode,
                )
                sent = {}
                replies: "Queue" = Queue()

                def listen():
                    while True:
                        try:
                            raw = transport.recv()
                        except Exception:
                            break
                        if raw is None:
                            break
                        try:
                            replies.put(wire_decode(raw))
                        except Exception:
                            continue

                Thread(target=listen, daemon=True).start()
                rng = random.Random(CLIENT_SEED_BASE + ci)
                for fid in range(frames):
                    transport.send(wire_encode({"frame_id": fid, "image": img}))
                    sent[fid] = time.monotonic()
                    dt = rng.expovariate(rate) if arrival == "poisson" else 1.0 / rate
                    time.sleep(dt)
                deadline = time.monotonic() + DRAIN_WINDOW_S
                got = 0
                while got < frames and time.monotonic() < deadline:
                    try:
                        rep = replies.get(timeout=REPLY_POLL_S)
                    except Empty:
                        break
                    fid = rep.get("frame_id")
                    if fid in sent:
                        with lock:
                            results.append(
                                (ci, target, fid, round(time.monotonic() - sent[fid], 6))
                            )
                        got += 1
                transport.close()

        threads = [Thread(target=one_client, args=(ci,)) for ci in range(clients)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["client", "target", "frame_id", "latency_s"])
            for r in sorted(results):
                w.writerow(r)
        lats = [r[3] for r in results]
        med = statistics.median(lats) if lats else float("nan")
        self.log(
            "[load] clients={} servers={} replies={} median_latency={:.1f}ms csv={}".format(
                clients, len(self.servers), len(results), med * 1000, self.csv_path
            )
        )


def build_load(config_path: str, overrides: List[str], run_index: int, logger=None):
    """Construct a :class:`LoadRunner` from a config file."""
    config, run_id = select_run_config(config_path, overrides, run_index)
    # The wire encoding is a per-process choice the apps read through
    # cadet.core.codec, so it is selected once here, before any transport
    # exists. Selecting it later would leave a payload encoded one way and
    # decoded another.
    set_wire_encoding_from_config(config)
    out_dir = os.path.join(config.outputs.root, config.experiment.name, config.app.plugin)
    csv_path = os.path.join(out_dir, "load_{}.csv".format(run_id))
    return LoadRunner(config, csv_path, logger=logger)
