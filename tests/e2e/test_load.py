"""Multi-connection server + load tests (loopback, no CARLA).

Covers: two clients -> one server, one client -> two servers, load (queue wait
grows with the number of clients), and server-dies-mid-run (client exits cleanly
with the error recorded).
"""

import csv
import glob
import os
import socket
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = os.path.join(REPO, "examples", "reference", "synthetic.yaml")
LOAD = os.path.join(REPO, "examples", "reference", "load.yaml")
APP = "examples.reference.app"
ENV = None


def _env():
    global ENV
    if ENV is None:
        ENV = dict(os.environ, PYTHONUNBUFFERED="1")
    return ENV


def _port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _serve(config, out, sets, logpath, node=None):
    cmd = [
        sys.executable,
        "-m",
        "cadet",
        "serve",
        config,
        "--app-module",
        APP,
        "--set",
        "outputs.root=" + out,
    ]
    if node:
        cmd += ["--node", node]
    for s in sets:
        cmd += ["--set", s]
    lf = open(logpath, "w")
    p = subprocess.Popen(cmd, cwd=REPO, env=_env(), stdout=lf, stderr=subprocess.STDOUT)
    deadline = time.time() + 30
    while time.time() < deadline:
        if "Waiting for connection" in open(logpath).read():
            return p, lf
        if p.poll() is not None:
            raise RuntimeError("server exited early: " + open(logpath).read())
        time.sleep(0.2)
    raise RuntimeError("server never listened")


def _client(config, out, sets):
    cmd = [
        sys.executable,
        "-m",
        "cadet",
        "client",
        config,
        "--app-module",
        APP,
        "--set",
        "outputs.root=" + out,
    ]
    for s in sets:
        cmd += ["--set", s]
    return subprocess.Popen(
        cmd, cwd=REPO, env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )


def _server_metrics(out):
    f = glob.glob(os.path.join(out, "**", "server_metrics_*.csv"), recursive=True)
    assert f, "no server metrics in " + out
    return list(csv.DictReader(open(f[0])))


def test_two_clients_one_server(tmp_path):
    out = str(tmp_path / "out")
    port = _port()
    srv, lf = _serve(REF, out, ["nodes[1].port={}".format(port)], str(tmp_path / "s.log"))
    try:
        c1 = _client(REF, out, ["nodes[1].port={}".format(port), "source.count=6"])
        c2 = _client(REF, out, ["nodes[1].port={}".format(port), "source.count=6"])
        assert c1.wait(60) == 0 and c2.wait(60) == 0
    finally:
        srv.wait(15)
        lf.close()
    recs = _server_metrics(out)
    assert len(recs) == 12  # 6 + 6 requests served
    assert len({r["client_id"] for r in recs}) == 2  # two concurrent clients


def test_one_client_two_servers(tmp_path):
    out = str(tmp_path / "out")
    p1, p2 = _port(), _port()
    # Two server nodes; one load client fans out to both.
    two = (
        "nodes=[{name: s1, host: 127.0.0.1, port: %d, role: server, device: cpu},"
        "{name: s2, host: 127.0.0.1, port: %d, role: server, device: cpu},"
        "{name: gen, host: localhost, role: load, clients: 1, rate_hz: 200.0,"
        " arrival: constant, frames: 8}]" % (p1, p2)
    )
    s1, l1 = _serve(LOAD, out, [two], str(tmp_path / "s1.log"), node="s1")
    s2, l2 = _serve(LOAD, out, [two], str(tmp_path / "s2.log"), node="s2")
    try:
        ld = subprocess.run(
            [
                sys.executable,
                "-m",
                "cadet",
                "load",
                LOAD,
                "--app-module",
                APP,
                "--set",
                "outputs.root=" + out,
                "--set",
                two,
            ],
            cwd=REPO,
            env=_env(),
            timeout=60,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert ld.returncode == 0, ld.stdout.decode()
    finally:
        s1.wait(15)
        s2.wait(15)
        l1.close()
        l2.close()
    rows = list(
        csv.DictReader(open(glob.glob(os.path.join(out, "**", "load_*.csv"), recursive=True)[0]))
    )
    targets = {r["target"] for r in rows}
    assert len(targets) == 2, targets  # one client reached both servers
    assert len(rows) == 16  # 8 frames x 2 servers


def _load_mean_queue_wait(tmp_path, clients, tag):
    out = str(tmp_path / tag)
    port = _port()
    sets = [
        "nodes[0].port={}".format(port),
        "nodes[1].clients={}".format(clients),
        "nodes[1].frames=25",
        "nodes[1].rate_hz=120",
    ]
    srv, lf = _serve(LOAD, out, sets, str(tmp_path / (tag + ".log")))
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "cadet",
                "load",
                LOAD,
                "--app-module",
                APP,
                "--set",
                "outputs.root=" + out,
            ]
            + sum((["--set", s] for s in sets), []),
            cwd=REPO,
            env=_env(),
            timeout=90,
            stdout=subprocess.DEVNULL,
        )
    finally:
        srv.wait(20)
        lf.close()
    recs = _server_metrics(out)
    return statistics.mean(float(r["queue_wait_s"]) for r in recs)


def test_load_queue_wait_grows_with_clients(tmp_path):
    light = _load_mean_queue_wait(tmp_path, 1, "light")  # 1 client < capacity
    heavy = _load_mean_queue_wait(tmp_path, 10, "heavy")  # 10 clients > capacity
    assert heavy > light * 2, "queue wait did not grow with load: {} vs {}".format(light, heavy)


def test_server_dies_midrun(tmp_path):
    out = str(tmp_path / "out")
    port = _port()
    # Many frames, paced, so the run is still going when we kill the server.
    srv, lf = _serve(REF, out, ["nodes[1].port={}".format(port)], str(tmp_path / "s.log"))
    clog = tmp_path / "c.log"
    cmd = [
        sys.executable,
        "-m",
        "cadet",
        "client",
        REF,
        "--app-module",
        APP,
        "--set",
        "outputs.root=" + out,
        "--set",
        "nodes[1].port={}".format(port),
        "--set",
        "source.count=400",
        "--set",
        "source.fps=50",
    ]
    cf = open(str(clog), "w")
    client = subprocess.Popen(cmd, cwd=REPO, env=_env(), stdout=cf, stderr=subprocess.STDOUT)
    time.sleep(2.0)
    srv.kill()
    srv.wait(10)
    lf.close()
    rc = client.wait(30)  # must exit, not hang
    cf.close()
    log = open(str(clog)).read()
    assert "server closed mid-run" in log, log  # error recorded
    assert rc == 0  # clean exit (no crash)
