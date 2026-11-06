"""End-to-end framework tests: real `cadet serve` + `cadet client` processes.

No CARLA — the reference app over a synthetic source on loopback. Covers the
transport round trip (a reply per frame), and that a link's network delay is
applied while a plain channel bypasses it.
"""

import csv
import os
import socket
import subprocess
import sys
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF_CONFIG = os.path.join(REPO, "examples", "reference", "synthetic.yaml")
APP = "examples.reference.app"


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _run(tmp_path, extra_sets):
    """Run serve+client for the reference config; return the reply CSV rows."""
    os.makedirs(str(tmp_path), exist_ok=True)
    port = _free_port()
    out = str(tmp_path / "out")
    sets = ["outputs.root={}".format(out), "nodes[1].port={}".format(port)] + extra_sets
    env = dict(os.environ, PYTHONUNBUFFERED="1")

    def cmd(sub):
        c = [sys.executable, "-m", "cadet", sub, REF_CONFIG, "--app-module", APP]
        for s in sets:
            c += ["--set", s]
        return c

    slog = open(str(tmp_path / "server.log"), "w")
    server = subprocess.Popen(
        cmd("serve"), cwd=REPO, env=env, stdout=slog, stderr=subprocess.STDOUT
    )
    try:
        # wait for the server to be listening
        deadline = time.time() + 30
        while time.time() < deadline:
            if "Waiting for connection" in open(str(tmp_path / "server.log")).read():
                break
            if server.poll() is not None:
                raise RuntimeError("server exited early")
            time.sleep(0.2)
        client = subprocess.run(
            cmd("client"),
            cwd=REPO,
            env=env,
            timeout=60,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert client.returncode == 0, client.stdout.decode()
    finally:
        server.wait(timeout=10)
        slog.close()

    csvs = []
    for root, _d, files in os.walk(out):
        for f in files:
            if f.endswith("latency_location_log.csv"):  # the client's reply log
                csvs.append(os.path.join(root, f))
    assert len(csvs) == 1, csvs
    return list(csv.DictReader(open(csvs[0])))


def test_round_trip_reply_per_frame(tmp_path):
    rows = _run(tmp_path, ["source.count=8"])
    assert len(rows) == 8  # a reply for every frame
    assert all(r["latency_s"] not in ("", "None") for r in rows)
    assert {int(r["frame_id"]) for r in rows} == set(range(8))


def test_link_delay_applied_per_message_not_cumulative(tmp_path):
    """A 60 ms link adds ~60 ms per message (NOT cumulative) and a channel bypasses it.

    With the delay-line fix, each message is delayed ~60 ms independently, so the
    median stays near 60 ms over many frames (the old sleep-in-recv model would
    accumulate to hundreds of ms). A plain channel is much faster.
    """
    # Pace the source (fps) so frames don't burst and queue in the single-threaded
    # server; this isolates the per-message LINK delay from server queueing.
    base = _run(tmp_path / "chan", ["source.count=12", "source.fps=25"])
    base_med = sorted(float(r["latency_s"]) for r in base)[len(base) // 2]
    link = [
        "source.count=12",
        "source.fps=25",
        "links=[{name: l, from: client, to: server, network: {mode: delay, distribution: {kind: constant, value_ms: 60.0}}}]",  # noqa: E501
    ]
    delayed = _run(tmp_path / "link", link)
    med = sorted(float(r["latency_s"]) for r in delayed)[len(delayed) // 2]
    assert 0.055 <= med <= 0.15, "link delay not ~60ms per message (cumulative?): {}".format(med)
    assert base_med < med, "channel not faster than link: {} vs {}".format(base_med, med)


def test_same_app_over_replay_source(tmp_path):
    """The reference app runs unchanged over a replay source (vs synthetic)."""
    import cv2
    import numpy as np

    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for i in range(6):
        cv2.imwrite(str(imgs / "f{}.png".format(i)), np.full((16, 16, 3), i * 5, np.uint8))
    rows = _run(
        tmp_path,
        [
            "source={{kind: replay, path: {}, fps: 0.0}}".format(imgs),
        ],
    )
    assert len(rows) == 6
    assert all(r["latency_s"] not in ("", "None") for r in rows)


def test_bad_config_readable_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "experiment: {name: x, app: reference}\nsource: {kind: teleport}\nnodes: []\n"
        "app: {plugin: reference}\n"
    )
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    r = subprocess.run(
        [sys.executable, "-m", "cadet", "validate", str(bad), "--app-module", APP],
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert r.returncode != 0
    assert b"source" in r.stderr  # names the offending field


def test_preflight_failure_e2e(tmp_path):
    """A yolo server with a missing artifact fails fast with a readable error."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    out = str(tmp_path / "out")
    port = _free_port()
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "cadet",
            "serve",
            REF_CONFIG,
            "--app-module",
            APP,
            "--set",
            "app.params.model_kind=yolo",
            "--set",
            "models.model=ghost",
            "--set",
            "models.registry={ghost: {per_device: {cpu: ghost.pt}}}",
            "--set",
            "nodes[1].device=cpu",
            "--set",
            "nodes[1].port={}".format(port),
            "--set",
            "outputs.root={}".format(out),
            "--model-dir",
            str(tmp_path),
        ],
        cwd=REPO,
        env=env,
        timeout=60,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r.returncode != 0
    assert b"not found" in r.stdout.lower() or b"ghost" in r.stdout.lower()
