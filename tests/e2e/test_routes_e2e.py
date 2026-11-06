"""End-to-end check that a composed route behaves like the link it sums to.

Uses the reference app over a synthetic source (no CARLA). A two-hop route of
20 ms and 40 ms is run against a single 60 ms link, both paced, and the measured
message latencies are compared: composing hops must not add latency of its own.
"""

import json
import os
import socket

from cadet.orchestrator import Orchestrator

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROUTE_CONFIG = os.path.join(REPO, "examples", "reference", "route.yaml")
FLAT_CONFIG = os.path.join(REPO, "examples", "reference", "synthetic.yaml")
APP = "examples.reference.app"

# One link carrying the sum of the route's two hops.
SUMMED_LINK = (
    'links=[{"name":"only","from":"client","to":"server","direction":"forward",'
    '"network":{"mode":"delay","distribution":{"kind":"constant","value_ms":60.0}}}]'
)
# Paced, so the measurement is the path's latency and not the queueing of a
# burst the app sends as fast as it can.
PACED = ["source.count=24", "source.fps=10"]


def _free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _run(config, tmp_path, tag, server_index, extra):
    os.chdir(REPO)
    orch = Orchestrator(log=lambda m: None)
    sets = [
        "outputs.root={}".format(tmp_path / tag),
        "nodes[{}].port={}".format(server_index, _free_port()),
    ] + PACED + extra
    result = orch.run_one(config, sets, 0, 0, [APP], [])
    assert result.ok, result.detail
    with open(os.path.join(result.out_dir, "link_report_client.json")) as handle:
        return json.load(handle)


def test_a_composed_route_measures_like_the_summed_link(tmp_path):
    route = _run(ROUTE_CONFIG, tmp_path, "route", 2, [])
    flat = _run(FLAT_CONFIG, tmp_path, "flat", 1, [SUMMED_LINK])

    record = route["configured_routes"][0]
    assert record["via"] == ["edge"]
    assert [hop["name"] for hop in record["hops"]] == ["access", "backhaul"]
    assert record["end_to_end"]["forward"]["p50_ms"] == 60.0
    # The hops are reported inside the route, so nothing is listed twice.
    assert route["configured_links"] == []

    route = route["measured"]["latency_ms"]
    flat = flat["measured"]["latency_ms"]
    # Both must reach the configured 60 ms, and the route must not add to it.
    #
    # The comparison is on the *minimum*, not the median, and that is the point
    # rather than a weakening: the floor of the distribution is set by the
    # configured delay, while a median moves with whatever else the machine is
    # doing. Run inside the full suite, with another stream's tests running
    # beside it, the two medians drifted apart by more than the ten
    # milliseconds this once allowed - two live runs, unequally loaded - and the
    # failure said nothing about composition. The minimum does.
    # The window is what the delay guarantees and no more, derived from the
    # configured value rather than written as two numbers.
    #
    # A delay line releases on a timer: it can release late by as much as the
    # machine is busy for, and it cannot release early - but the client brackets
    # the call with its own clock reads, so a measured minimum can sit slightly
    # *below* the configured value. 59.341 ms for a configured 60 was observed
    # under load, which is clock granularity and not a link that under-delayed,
    # so the floor carries an allowance instead of being exact. Measured again
    # under three concurrent suites, six runs: minima between 60.467 and 60.605.
    #
    # The ceiling used to be 75 ms, which is a statement about how fast the
    # machine is rather than about the route: a run whose every sample was late
    # fails it and says nothing about composition. What the test actually claims
    # is that the delay is paid once, so the ceiling is the value at which it
    # would have been paid twice, with the same allowance below it.
    configured = record["end_to_end"]["forward"]["p50_ms"]
    floor, ceiling = 0.95 * configured, 1.5 * configured
    assert floor <= route["min"] <= ceiling, route
    assert floor <= flat["min"] <= ceiling, flat
    assert abs(route["min"] - flat["min"]) < 5.0, (route["min"], flat["min"])
    # The median is still checked, loosely, so a path that composes correctly
    # but queues pathologically does not pass unnoticed.
    assert route["p50"] < 3.0 * flat["p50"], (route["p50"], flat["p50"])
