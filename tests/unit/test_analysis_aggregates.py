"""Tests for the reductions the analysis context states above its tables.

What this is for
    The record holds the tables a question has to be answered *from*: energy
    per node per collector, latency per client, a message count and an outcome
    each. Doing the arithmetic across them is work, and a reader - a person
    skimming or a model answering - does it with whatever attention is left
    after parsing the tables. Measured across this repository's acceptance
    matrix, an analysis model answered 138 of 140 such questions correctly. The
    two it missed were enough to flip those cells' verdicts, and one was a cell
    that answered correctly in other runs on identical data - so what these
    reductions remove is the variance, not an inability to do arithmetic.

    So the arithmetic is done once, in the record. These tests pin it, because
    a wrong aggregate is worse than none: it is a confident wrong answer that
    reads as the record's own.

What it contains
    Cases for the energy ordering (including nodes that measured nothing), the
    percentiles, the message counts and the outcomes.

How it fits into CADET
    Unit tests over :mod:`cadet.analysis.context`.
"""

from cadet.analysis.context import run_aggregates


def _ctx(**overrides):
    base = {
        "device": {},
        "links_measured": {},
        "outcomes": {},
        "latency_ms": None,
        "path": None,
    }
    base.update(overrides)
    return base


def test_the_top_energy_node_is_the_one_that_drew_the_most():
    """Sorted, with the top named, so nobody has to compare the rows."""
    ctx = _ctx(
        device={
            "sensor_1": {"energy": {"energy_j_total": 184.189}},
            "gateway": {"energy": {"energy_j_total": 510.325}},
            "sensor_2": {"energy": {"energy_j_total": 187.567}},
        }
    )

    agg = run_aggregates(ctx)

    assert agg["top_energy_node"] == "gateway"
    assert [e["node"] for e in agg["energy_by_node"]] == ["gateway", "sensor_2", "sensor_1"]
    assert [e["energy_j"] for e in agg["energy_by_node"]] == [510.325, 187.567, 184.189]


def test_a_node_that_measured_nothing_is_listed_but_never_the_top():
    """A node with no power source has no energy, which is not zero energy.

    Ordering it as zero would make it the lowest; dropping it would make the
    record silent about a node that was asked. It is listed after the measured
    ones, with the reason it has no figure.
    """
    ctx = _ctx(
        device={
            "client": {"energy": {"not_applicable": "no power source on this node: rapl: denied"}},
            "server": {"energy": {"energy_j_total": 109.557}},
        }
    )

    agg = run_aggregates(ctx)

    assert agg["top_energy_node"] == "server"
    nodes = [e["node"] for e in agg["energy_by_node"]]
    assert nodes == ["server", "client"], "measured first, unmeasured after"
    assert agg["energy_by_node"][1]["energy_j"] is None
    assert "rapl" in agg["energy_by_node"][1]["why"]


def test_no_node_measured_power_means_no_top_node():
    """There is nothing to name, and naming one anyway would be an invention."""
    ctx = _ctx(
        device={
            "client": {"energy": {"not_applicable": "no GPU on this node"}},
            "server": {"cpu": {"samples": 10}},
        }
    )

    agg = run_aggregates(ctx)

    assert agg["top_energy_node"] is None
    assert [e["node"] for e in agg["energy_by_node"]] == ["client"]


def test_the_percentiles_come_from_the_link_report_as_recorded():
    """Copied, not recomputed, so the record and the reduction cannot disagree."""
    ctx = _ctx(
        links_measured={
            "client": {
                "reply_policy": "per-message",
                "messages_sent": 40,
                "messages_received": 40,
                "unanswered": 0,
                "latency_ms": {
                    "samples": 40,
                    "p50": 0.721,
                    "p99": 16.043,
                    "mean": 1.163,
                    "max": 16.043,
                },
            }
        }
    )

    agg = run_aggregates(ctx)

    assert agg["latency_by_client"] == [
        {
            "client": "client",
            "samples": 40,
            "p50_ms": 0.721,
            "p99_ms": 16.043,
            "mean_ms": 1.163,
            "max_ms": 16.043,
        }
    ]
    # p95 is absent rather than interpolated: the link report does not keep it
    # and this run has no per-frame record to compute it from.
    assert "p95_ms" not in agg["latency_by_client"][0]


def test_p95_is_computed_from_the_per_frame_record_when_there_is_one(tmp_path):
    """Nearest-rank over the measured samples, not a guess between p50 and p99."""
    rows = ["frame_id,latency_s,server_processing_time,result"]
    # 100 samples of 1..100 ms: the 95th by nearest rank is the 95th value.
    for i in range(1, 101):
        rows.append("{},{},0.0,x".format(i, i / 1000.0))
    (tmp_path / "abc123-r0_latency_location_log.csv").write_text("\n".join(rows) + "\n")

    ctx = _ctx(
        path=str(tmp_path),
        links_measured={
            "client": {"latency_ms": {"samples": 100, "p50": 50.0, "p99": 99.0}},
        },
    )

    agg = run_aggregates(ctx)

    assert agg["latency_by_client"][0]["p95_ms"] == 95.0


def test_the_message_counts_and_outcomes_are_stated_per_client():
    """Including the reply policy, without which "unanswered" reads as loss."""
    ctx = _ctx(
        links_measured={
            "client": {
                "reply_policy": "on-event",
                "messages_sent": 27,
                "messages_received": 1,
                "unanswered": 26,
            }
        },
        outcomes={"client": {"ok": False, "detail": "the ego was removed"}},
    )

    agg = run_aggregates(ctx)

    assert agg["messages_by_client"] == [
        {
            "client": "client",
            "reply_policy": "on-event",
            "sent": 27,
            "answered": 1,
            "unanswered": 26,
        }
    ]
    assert agg["outcomes"] == [
        {"client": "client", "ok": False, "detail": "the ego was removed"}
    ]


def test_an_empty_record_reduces_to_nothing_rather_than_to_zeroes():
    """A run that measured nothing must not read as a run that measured zero."""
    agg = run_aggregates(_ctx())
    assert agg["top_energy_node"] is None
    assert agg["energy_by_node"] == []
    assert agg["latency_by_client"] == []
    assert agg["messages_by_client"] == []
    assert agg["outcomes"] == []


def test_a_run_with_several_clients_states_its_one_latency_figure_too():
    """"The median latency of this run" is not any one client's p50.

    The distributed-sensing example has a client per sensor, and the run's
    figure is the median over their link reports. Stating only the per-client
    percentiles left that question answerable only by combining them - and
    measured against the acceptance matrix, a model reading the more prominent
    per-client lines quoted one of those instead, on three cells that had
    answered correctly before the per-client lines existed.
    """
    ctx = _ctx(
        latency_ms=57.752,
        links_measured={
            "sensor_1": {"latency_ms": {"samples": 20, "p50": 44.972, "p99": 64.65}},
            "sensor_2": {"latency_ms": {"samples": 20, "p50": 70.531, "p99": 86.794}},
        },
    )

    agg = run_aggregates(ctx)

    assert agg["median_e2e_ms"] == 57.752
    assert [item["client"] for item in agg["latency_by_client"]] == ["sensor_1", "sensor_2"]
    # The run's figure is not one of the clients' own.
    assert agg["median_e2e_ms"] not in [i["p50_ms"] for i in agg["latency_by_client"]]


def test_a_run_that_measured_no_latency_states_none():
    """Absent rather than zero, which would read as an immeasurably fast run."""
    agg = run_aggregates(_ctx())
    assert agg["median_e2e_ms"] is None
