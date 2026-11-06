"""Tests for the foundation-model interface: the record, the refusals, the seam.

What is pinned here, and why each of these and not a happy path:

* **No test reaches the network.** The autouse :func:`no_network` fixture
  replaces ``urllib.request.urlopen`` with something that fails the test, so a
  path that would open a socket cannot pass quietly; the three tests that drive
  the HTTP client replace it themselves with a recorder.
* **No key reaches the context.** Asserted against an environment variable, an
  app parameter and an ``analysis`` section that all hold the same sentinel, so
  the assertion fails if any one of the three routes opens.
* **No key reaches an error message.** A failing request's message is checked
  for the endpoint and the status *and* against the key, the word ``Bearer``
  and the header name.
* **The default state refuses.** With nothing configured both subcommands exit
  non-zero with a message naming what to set, rather than a traceback.
* Both shapes: one run directory and a tree of them, including a tree holding
  two sweeps, and a run whose files are malformed (which must become notes).

The fixture tree is written here rather than copied from a real output tree, so
these tests pin the file names and the column shapes the readers depend on.
"""

import json
import os
import urllib.error
import urllib.request

import pytest
import yaml

from cadet.analysis import backend as backend_mod
from cadet.analysis import context as context_mod
from cadet.analysis import session
from cadet.config.schema import AnalysisConfig

# The string that must never appear anywhere a model, a log or a config can see.
SENTINEL = "sk-do-not-leak-this-anywhere"

ENDPOINT = "http://127.0.0.1:9/v1/chat/completions"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any test that opens an HTTP connection.

    The brief's hard constraint is that no test touches the network. Asserting
    it is better than reviewing it: a future change that makes the context
    builder fetch something, or a backend constructed where a stub was meant,
    fails here instead of passing on a machine that happens to be offline.
    """

    def _refuse(*args, **kwargs):
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(urllib.request, "urlopen", _refuse)


# --- fixture run trees ----------------------------------------------------
def _config(placement, delay_ms, secret=None):
    """A resolved config of the shape the orchestrator saves into a run dir.

    Carries the three things that must not come out the other side: a node's
    address and account, an app parameter named like a credential, and an
    ``analysis`` section.
    """
    data = {
        "experiment": {"name": "fixture", "repetitions": 2},
        "source": {"kind": "synthetic", "fps": 10},
        "app": {"plugin": "fixture", "params": {"placement": placement}},
        "nodes": [
            {
                "name": "edge",
                "role": "client",
                "device": "cpu",
                "host": "10.11.12.13",
                "user": "an-account",
                "ssh_identity": "/home/an-account/.ssh/id_ed25519",
                "launcher": "ssh_docker",
            },
            {
                "name": "hub",
                "role": "server",
                "device": "gpu",
                "host": "10.11.12.14",
                "port": 9000,
                "launcher": "docker",
            },
        ],
        "links": [
            {
                "name": "uplink",
                "from": "edge",
                "to": "hub",
                "network": {
                    "mode": "delay",
                    "medium": "wired",
                    "distribution": {"kind": "normal", "mean_ms": delay_ms, "std_ms": 1.0},
                    "loss": 0.01,
                    "bandwidth_mbps": 10.0,
                },
            }
        ],
        "routes": [],
        "channels": [],
        "transport": {"protocol": "tcp", "mode": "frame", "encoding": "safe"},
        "metrics": {"collectors": ["cpu"]},
        "sweep": {"axes": {}},
        "analysis": {"backend": "none", "api_key_env": "CADET_FM_API_KEY"},
    }
    if secret is not None:
        data["app"]["params"]["api_key"] = secret
        data["analysis"]["endpoint"] = "https://example.invalid/v1/chat/completions"
    return data


def _write_run(root, run_id, placement="edge", delay_ms=20.0, secret=None, status="complete"):
    """Write one run directory holding every file the record reads."""
    path = os.path.join(root, run_id)
    os.makedirs(path)
    if status is not None:
        with open(os.path.join(path, "run.json"), "w") as handle:
            json.dump(
                {
                    "run_id": run_id,
                    "status": status,
                    "reason": "workers exited",
                    "detail": "exit codes [0]",
                    "startup_s": 1.5,
                    "active_s": 30.0,
                    "teardown_s": 2.0,
                },
                handle,
            )
    with open(os.path.join(path, "config.yaml"), "w") as handle:
        yaml.safe_dump(_config(placement, delay_ms, secret), handle)
    with open(os.path.join(path, "link_report_edge.json"), "w") as handle:
        json.dump(
            {
                "node": "edge",
                "configured_links": [{"name": "uplink", "loss": 0.01}],
                "measured": {
                    "reply_policy": "per-message",
                    "messages_sent": 100,
                    "messages_received": 97,
                    "unanswered": 3,
                    "unanswered_pct": 3.0,
                    "latency_ms": {
                        "samples": 97,
                        "p50": 41.5,
                        "p99": 88.25,
                        "mean": 44.0,
                        "min": 21.0,
                        "max": 88.25,
                    },
                },
            },
            handle,
        )
    with open(os.path.join(path, "client_outcome_edge.json"), "w") as handle:
        json.dump({"ok": True, "detail": "sent 100"}, handle)
    with open(os.path.join(path, "server_metrics_hub.csv"), "w") as handle:
        handle.write("queue_wait_s,inference_s,replied\n")
        for i in range(5):
            handle.write("0.0{},0.1{},True\n".format(i, i))
    with open(os.path.join(path, "fixture_summary.json"), "w") as handle:
        json.dump({"e2e_p50_ms": 42.0 + delay_ms, "lost": 3}, handle)
    with open(os.path.join(path, "cpu_hub.csv"), "w") as handle:
        handle.write("t_wall,t_monotonic,node,cpu_pct_host,rss_mb_process\n")
        for i in range(4):
            handle.write("100{0},{0}.0,hub,1{0}.5,110.5\n".format(i))
    return path


@pytest.fixture
def one_run(tmp_path):
    """A single run directory, pointed at directly."""
    return _write_run(str(tmp_path / "tree"), "aaa1-r0", secret=SENTINEL)


@pytest.fixture
def sweep_tree(tmp_path):
    """One sweep: two placements x two conditions, two repetitions each."""
    root = str(tmp_path / "sweep")
    index = 0
    for placement in ("edge", "cloud"):
        for delay in (20.0, 80.0):
            for rep in range(2):
                index += 1
                _write_run(
                    root,
                    "cell{}-r{}".format(index, rep),
                    placement=placement,
                    delay_ms=delay,
                )
    return root


# --- the record -----------------------------------------------------------
def test_a_single_run_directory_reads_as_one_run(one_run):
    context = context_mod.build_context(one_run)
    assert context["kind"] == "run"
    run = context["run"]
    assert run["run_id"] == "aaa1-r0"
    assert run["rep"] == 0
    assert run["status"] == "complete"
    assert run["placement"] == "edge"
    assert run["swept_ms"] == 20.0
    # Reduced, not raw: the server CSV's five rows are one stage summary.
    assert run["server"]["hub"]["requests"] == 5
    assert [s["name"] for s in run["server"]["hub"]["stages"]] == ["queue wait", "inference"]
    # The measured side of the link report, and the app's own sidecar.
    assert run["links_measured"]["edge"]["unanswered"] == 3
    assert run["links_measured"]["edge"]["latency_ms"]["p50"] == 41.5
    assert run["sidecars"]["e2e_p50_ms"] == 62.0
    assert run["task_metric"] == {"name": "e2e p50", "value": 62.0, "unit": "ms"}
    # The device collector had no roll-up file, so the series' own stats.
    assert run["device"]["hub"]["CPU (host)"]["peak"] == 13.5


def test_the_record_carries_no_address_account_or_credential(one_run, monkeypatch):
    """Three routes a key or an address could take, all closed at once.

    The environment variable, an app parameter named like a credential, and the
    ``analysis`` section of the run's own config each hold the same sentinel, so
    this fails if any one of them opens.
    """
    monkeypatch.setenv("CADET_FM_API_KEY", SENTINEL)
    text = context_mod.render_context(one_run, max_bytes=1000000)
    assert SENTINEL not in text
    assert "api_key" not in text
    # The addresses and the account the orchestrator reached the nodes with.
    for leaked in ("10.11.12.13", "10.11.12.14", "an-account", "id_ed25519"):
        assert leaked not in text, leaked
    # And the facts that are kept, so this is not passing by rendering nothing.
    assert "name=edge role=client device=cpu launcher=ssh_docker" in text
    assert "placement: edge" in text


def test_a_peer_address_in_the_device_rollup_is_rewritten(tmp_path):
    """The bandwidth collector keys a series by peer address; the key survives
    and the address does not."""
    path = _write_run(str(tmp_path / "t"), "bbb1-r0")
    with open(os.path.join(path, "device_summary.json"), "w") as handle:
        json.dump(
            {
                "nodes": {
                    "hub": {
                        "bandwidth": {
                            "links": {
                                "10.11.12.13:47386/recv": {"bytes_total": 11247.0},
                                "10.11.12.13:47386/sent": {"bytes_total": 2018.0},
                                "edge/recv": {"bytes_total": 7.0},
                            }
                        }
                    }
                }
            },
            handle,
        )
    text = context_mod.render_context(path, max_bytes=1000000)
    assert "10.11.12.13" not in text
    assert "peer1/recv" in text and "peer1/sent" in text
    # A key that is already a node name is left alone.
    assert "edge/recv" in text


def test_a_tree_of_runs_reads_as_a_sweep_matrix(sweep_tree):
    context = context_mod.build_context(sweep_tree)
    assert context["kind"] == "sweep"
    assert len(context["sweeps"]) == 1
    sweep = context["sweeps"][0]
    assert sweep["placements"] == ["cloud", "edge"]
    assert len(sweep["networks"]) == 2
    assert len(sweep["cells"]) == 4
    assert len(sweep["runs"]) == 8
    # Each cell is its repetitions, and its number is the median over them.
    for cell in sweep["cells"]:
        assert len(cell["runs"]) == 2
        assert cell["metric"]["median"] is not None
    text = context_mod.render_context(sweep_tree, max_bytes=1000000)
    assert "2 placement(s) x 2 network condition(s), 4 cell(s)" in text
    # Eight runs of reduced record, not eight runs of CSV rows.
    assert text.count("\nrun cell") == 8


def test_two_sweeps_in_one_tree_stay_two_matrices(tmp_path):
    """A sweep is a directory of runs, so two directories are two matrices and
    never one that averaged them together."""
    root = str(tmp_path / "trees")
    _write_run(os.path.join(root, "first"), "a1-r0", placement="edge")
    _write_run(os.path.join(root, "second"), "b1-r0", placement="cloud")
    context = context_mod.build_context(root)
    assert [sweep["name"] for sweep in context["sweeps"]] == ["first", "second"]


def test_a_malformed_run_becomes_notes_and_not_an_exception(tmp_path):
    path = os.path.join(str(tmp_path / "t"), "ccc1-r0")
    os.makedirs(path)
    with open(os.path.join(path, "config.yaml"), "w") as handle:
        handle.write("{not: yaml: at all\n")
    run = context_mod.build_context(path)["run"]
    assert run["status"] == "in-progress"
    assert any("config.yaml" in note for note in run["notes"])
    assert any("link report" in note for note in run["notes"])
    assert "notes:" in context_mod.render_context(path, max_bytes=1000000)


def test_a_path_with_no_run_in_it_is_the_one_refusal(tmp_path):
    empty = str(tmp_path / "nothing")
    os.makedirs(empty)
    with pytest.raises(context_mod.ContextError) as caught:
        context_mod.build_context(empty)
    assert "not a run directory" in str(caught.value)


def test_the_record_is_reduced_to_fit_the_budget_and_says_so(sweep_tree):
    full = context_mod.render_context(sweep_tree, max_bytes=1000000)
    budget = 4000
    reduced = context_mod.render_context(sweep_tree, max_bytes=budget)
    assert len(full.encode("utf-8")) > budget
    assert len(reduced.encode("utf-8")) <= budget
    assert "reduced to fit analysis.max_context_bytes" in reduced
    # Still the sweep's own answer: the matrix is what survives the reduction.
    assert "cells (each number a median" in reduced


def test_a_budget_nothing_fits_in_is_cut_and_says_where(sweep_tree):
    text = context_mod.render_context(sweep_tree, max_bytes=400)
    assert len(text.encode("utf-8")) <= 400
    assert "record cut at 400 bytes" in text


# --- the section is not part of what was run ------------------------------
def test_the_analysis_section_does_not_change_a_run_id(braking_config_path):
    """Where a question is sent is not a fact about the run that answered it.

    outputs was already excluded from the identity for the same reason. If
    analysis were in it, adding an endpoint would give every cell of a
    sweep a new run id: `--resume` would re-run work already done, and an
    existing output tree would stop matching its own saved config.
    """
    from cadet.config.loader import config_hash, load_run_configs

    plain, _base = load_run_configs(braking_config_path, [])
    configured, _base = load_run_configs(
        braking_config_path,
        [
            'analysis.backend=openai_chat',
            'analysis.endpoint=' + ENDPOINT,
            'analysis.model=a-model',
        ],
    )
    assert configured[0].analysis.backend == 'openai_chat'
    assert config_hash(configured[0]) == config_hash(plain[0])


# --- the backend seam -----------------------------------------------------
def test_the_default_state_is_no_backend_configured():
    with pytest.raises(backend_mod.NotConfigured) as caught:
        backend_mod.make_backend(AnalysisConfig())
    message = str(caught.value)
    # The refusal has to say what to set, or it is a dead end.
    for expected in (
        "analysis.backend: openai_chat",
        "analysis.endpoint",
        "analysis.model",
        "CADET_FM_API_KEY",
    ):
        assert expected in message, expected


def test_a_configured_backend_with_no_key_in_the_environment_refuses(monkeypatch):
    monkeypatch.delenv("CADET_FM_API_KEY", raising=False)
    config = AnalysisConfig(backend="openai_chat", endpoint=ENDPOINT, model="a-model")
    with pytest.raises(backend_mod.NotConfigured) as caught:
        backend_mod.make_backend(config)
    message = str(caught.value)
    assert "CADET_FM_API_KEY" in message
    assert "analysis.api_key_env" in message


def test_a_backend_with_no_endpoint_is_a_config_error():
    """Caught by the schema, which is where a config is told it is wrong."""
    with pytest.raises(Exception) as caught:
        AnalysisConfig(backend="openai_chat")
    assert "analysis.endpoint" in str(caught.value)


def test_an_incomplete_backend_object_still_names_the_missing_field():
    """The same check inside make_backend, for a config built in code."""

    class _Partial(object):
        backend = "openai_chat"
        endpoint = ENDPOINT
        model = ""
        api_key_env = "CADET_FM_API_KEY"

    with pytest.raises(backend_mod.NotConfigured) as caught:
        backend_mod.make_backend(_Partial())
    assert "analysis.model not set" in str(caught.value)


class _Recorder(object):
    """A stand-in for ``urllib.request.urlopen`` that records and replies."""

    def __init__(self, body=None, raising=None):
        self.body = body
        self.raising = raising
        self.request = None
        self.timeout = None

    def __call__(self, request, timeout=None):
        self.request = request
        self.timeout = timeout
        if self.raising is not None:
            raise self.raising
        return self

    # Context-manager and reader surface of an http.client.HTTPResponse.
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self.body).encode("utf-8")

    def getcode(self):
        return self.status


def _backend(monkeypatch, recorder):
    monkeypatch.setenv("CADET_FM_API_KEY", SENTINEL)
    monkeypatch.setattr(urllib.request, "urlopen", recorder)
    config = AnalysisConfig(
        backend="openai_chat", endpoint=ENDPOINT, model="a-model", timeout_s=7.5
    )
    return backend_mod.make_backend(config)


def test_the_key_goes_in_the_header_and_nowhere_else(monkeypatch):
    recorder = _Recorder(body={"choices": [{"message": {"content": "an answer"}}]})
    client = _backend(monkeypatch, recorder)
    assert client.complete("be brief", "the record") == "an answer"

    request = recorder.request
    assert request.get_method() == "POST"
    assert recorder.timeout == 7.5
    # The key is in the one place it belongs.
    assert request.get_header("Authorization") == "Bearer {}".format(SENTINEL)
    # And in none of the places it does not.
    body = request.data.decode("utf-8")
    assert SENTINEL not in body
    assert SENTINEL not in repr(client)
    sent = json.loads(body)
    assert sent["model"] == "a-model"
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert sent["messages"][1]["content"] == "the record"


def test_an_http_error_names_the_endpoint_and_the_status_and_no_header(monkeypatch):
    failure = urllib.error.HTTPError(ENDPOINT, 503, "Service Unavailable", {}, None)
    client = _backend(monkeypatch, _Recorder(raising=failure))
    with pytest.raises(backend_mod.BackendError) as caught:
        client.complete("be brief", "the record")
    message = str(caught.value)
    assert ENDPOINT in message
    assert "503" in message
    for forbidden in (SENTINEL, "Bearer", "Authorization"):
        assert forbidden not in message, forbidden


def test_a_timeout_names_the_endpoint_and_the_budget(monkeypatch):
    import socket

    client = _backend(monkeypatch, _Recorder(raising=socket.timeout()))
    with pytest.raises(backend_mod.BackendError) as caught:
        client.complete("be brief", "the record")
    message = str(caught.value)
    assert ENDPOINT in message
    assert "timed out after 7.5 s" in message
    assert SENTINEL not in message


def test_a_connection_that_never_got_a_status_says_so(monkeypatch):
    failure = urllib.error.URLError("Connection refused")
    client = _backend(monkeypatch, _Recorder(raising=failure))
    with pytest.raises(backend_mod.BackendError) as caught:
        client.complete("be brief", "the record")
    assert "before any HTTP status" in str(caught.value)
    assert SENTINEL not in str(caught.value)


@pytest.mark.parametrize(
    "body", [{}, {"choices": []}, {"choices": [{"message": {"content": "   "}}]}]
)
def test_a_reply_with_no_completion_is_an_error_and_not_an_empty_summary(monkeypatch, body):
    client = _backend(monkeypatch, _Recorder(body=body))
    with pytest.raises(backend_mod.BackendError) as caught:
        client.complete("be brief", "the record")
    assert ENDPOINT in str(caught.value)


# --- the analysis section, and the prompts --------------------------------
def test_the_section_comes_from_the_run_directory_when_no_config_is_named(one_run):
    config = backend_mod.load_analysis_config(None, one_run)
    # The fixture run's own config.yaml carries the section, backend none.
    assert config.backend == "none"
    assert config.api_key_env == "CADET_FM_API_KEY"


def test_a_named_config_wins_over_the_run_directory(tmp_path, one_run):
    path = str(tmp_path / "fm.yaml")
    with open(path, "w") as handle:
        yaml.safe_dump(
            {
                "analysis": {
                    "backend": "openai_chat",
                    "endpoint": ENDPOINT,
                    "model": "a-model",
                    "max_context_bytes": 4096,
                }
            },
            handle,
        )
    config = backend_mod.load_analysis_config(path, one_run)
    assert config.backend == "openai_chat"
    assert config.max_context_bytes == 4096


def test_a_tree_with_no_section_anywhere_is_the_default_state(tmp_path):
    assert backend_mod.load_analysis_config(None, str(tmp_path)).backend == "none"


def test_the_question_is_last_in_the_prompt_and_the_record_first():
    seen = {}

    class _Stub(backend_mod.Backend):
        def complete(self, system, user):
            seen["system"] = system
            seen["user"] = user
            return "ok"

    assert session.respond("THE RECORD", "why so slow?", _Stub()) == "ok"
    assert seen["user"].startswith("THE RECORD")
    assert seen["user"].rstrip().endswith("Question: why so slow?")
    assert "Answer only from the record" in seen["system"]
    # No question is the fixed summarise task, so two summaries are comparable.
    session.respond("THE RECORD", None, _Stub())
    assert session.SUMMARIZE_TASK in seen["user"]


# --- the two subcommands --------------------------------------------------
class _StubBackend(backend_mod.Backend):
    """The backend every CLI test injects. Opens nothing."""

    def __init__(self, answer="a stub answer", raising=None):
        self.answer = answer
        self.raising = raising
        self.system = None
        self.user = None

    def complete(self, system, user):
        self.system = system
        self.user = user
        if self.raising is not None:
            raise self.raising
        return self.answer


def _inject(monkeypatch, stub):
    """Replace the backend factory the CLI calls. The injection point."""
    import cadet.analysis

    monkeypatch.setattr(cadet.analysis, "make_backend", lambda config: stub)
    return stub


def _main(*argv):
    from cadet.cli import main

    return main(list(argv))


def test_summarize_refuses_in_the_default_state_without_a_traceback(one_run, capsys):
    assert _main("summarize", one_run) == 2
    err = capsys.readouterr().err
    assert "no analysis backend configured" in err
    assert "analysis.endpoint" in err
    assert "Traceback" not in err


def test_ask_refuses_in_the_default_state_too(sweep_tree, capsys):
    assert _main("ask", sweep_tree, "which placement won?") == 2
    assert "no analysis backend configured" in capsys.readouterr().err


def test_summarize_sends_the_record_to_the_injected_backend(one_run, capsys, monkeypatch):
    stub = _inject(monkeypatch, _StubBackend())
    assert _main("summarize", one_run) == 0
    assert capsys.readouterr().out.strip() == "a stub answer"
    assert "CADET run record" in stub.user
    assert session.SUMMARIZE_TASK in stub.user


def test_ask_puts_the_question_to_the_backend(sweep_tree, capsys, monkeypatch):
    stub = _inject(monkeypatch, _StubBackend(answer="cloud, above 40 ms"))
    assert _main("ask", sweep_tree, "which placement won?") == 0
    assert capsys.readouterr().out.strip() == "cloud, above 40 ms"
    assert "CADET sweep record" in stub.user
    assert stub.user.rstrip().endswith("Question: which placement won?")


def test_context_only_prints_the_record_and_needs_no_backend(one_run, capsys):
    assert _main("summarize", one_run, "--context-only") == 0
    captured = capsys.readouterr()
    assert "CADET run record" in captured.out
    assert "bytes of context" in captured.err


def test_a_backend_that_fails_is_reported_and_exits_non_zero(one_run, capsys, monkeypatch):
    failure = backend_mod.BackendError(
        "request to {} failed: HTTP 500 (response body not shown)".format(ENDPOINT)
    )
    _inject(monkeypatch, _StubBackend(raising=failure))
    assert _main("summarize", one_run) == 1
    err = capsys.readouterr().err
    assert "HTTP 500" in err
    assert ENDPOINT in err
    assert "Traceback" not in err


def test_a_path_that_is_not_a_directory_is_refused(tmp_path, capsys):
    assert _main("summarize", str(tmp_path / "nope")) == 2
    assert "not a directory" in capsys.readouterr().err


def test_an_unreadable_named_config_is_refused_not_ignored(one_run, tmp_path, capsys):
    path = str(tmp_path / "broken.yaml")
    with open(path, "w") as handle:
        handle.write("analysis: [not, a, mapping]\n")
    assert _main("summarize", one_run, "--config", path) == 2
    assert "is not a mapping" in capsys.readouterr().err


def test_an_invalid_analysis_section_reports_the_field(one_run, tmp_path, capsys):
    path = str(tmp_path / "bad.yaml")
    with open(path, "w") as handle:
        yaml.safe_dump({"analysis": {"backend": "telepathy"}}, handle)
    assert _main("summarize", one_run, "--config", path) == 1
    assert "invalid `analysis` section" in capsys.readouterr().err


def test_an_endpoint_that_takes_no_key_is_told_apart_from_a_missing_one(monkeypatch):
    """A model served locally authenticates nobody, and says so in the config.

    An empty ``api_key_env`` is the config stating that this endpoint takes no
    key: the backend builds, and no Authorization header is sent. A name that
    is set with nothing behind it stays the misconfiguration it was.
    """
    from cadet.analysis import backend as backend_mod

    monkeypatch.delenv("CADET_FM_API_KEY", raising=False)

    keyless = AnalysisConfig(
        backend="openai_chat",
        endpoint="http://127.0.0.1:11434/v1/chat/completions",
        model="a-local-model",
        api_key_env="",
    )
    built = backend_mod.make_backend(keyless)
    sent = {}

    def _urlopen(request, timeout=None):
        sent["headers"] = dict(request.headers)
        raise backend_mod.BackendError("not sent anywhere")

    monkeypatch.setattr(backend_mod.urllib.request, "urlopen", _urlopen)
    with pytest.raises(backend_mod.BackendError):
        built.complete("system", "user")
    assert "Authorization" not in sent["headers"]

    named = AnalysisConfig(
        backend="openai_chat",
        endpoint="https://example.invalid/v1/chat/completions",
        model="a-remote-model",
        api_key_env="CADET_FM_API_KEY",
    )
    with pytest.raises(backend_mod.NotConfigured) as refusal:
        backend_mod.make_backend(named)
    assert "CADET_FM_API_KEY" in str(refusal.value)
