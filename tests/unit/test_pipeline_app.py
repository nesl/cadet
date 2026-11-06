"""Unit tests for the built-in `pipeline` app (server + client), no sockets/CARLA.

Exercises PipelineServerApp.handle (run local segment, reply-on-event) and
PipelineClientApp's stage-driven control + duck-typed actuation against a fake
source, in-process. Live transport is covered by the loopback runtime tests and
the CARLA L2 run.
"""

import cadet.pipeline.stages.braking  # noqa: F401  (camera/brake_controller)
from cadet.config.schema import Config
from cadet.pipeline.app import PipelineClientApp, PipelineServerApp
from cadet.pipeline.messages import Detections, Frame, Message
from cadet.pipeline.registry import register_stage
from cadet.pipeline.runtime import deserialize_message, serialize_message
from cadet.pipeline.stage import Stage
from cadet.pipeline.stages.braking import BrakeDecision


@register_stage("t_app_echo")
class _Echo(Stage):
    """Server stage that returns a Detections every frame (every-frame reply)."""

    def process(self, msg):
        return Detections(detections=[("x", 1.0, 1)])


@register_stage("t_app_emit_once")
class _EmitOnce(Stage):
    """Server stage that emits a BrakeDecision only on the 2nd call."""

    def __init__(self, name, params=None):
        super().__init__(name, params)
        self.n = 0

    def process(self, msg):
        self.n += 1
        return BrakeDecision(brake=True, frame_id=msg.frame_id) if self.n >= 2 else None


def _cfg(stages, app_params=None):
    return Config.model_validate(
        {
            "experiment": {"name": "t", "app": "pipeline"},
            "source": {"kind": "synthetic"},
            "nodes": [
                {"name": "client", "host": "localhost", "role": "client"},
                {"name": "server", "host": "127.0.0.1", "port": 9100, "role": "server"},
            ],
            "app": {"plugin": "pipeline", "params": app_params or {}},
            "pipeline": {"stages": stages},
        }
    )


# --------------------------------------------------------------------------
# server app
# --------------------------------------------------------------------------
def test_server_handle_runs_segment_and_replies_on_event():
    cfg = _cfg(
        [
            {"name": "camera", "impl": "camera", "node": "client"},
            {"name": "decide", "impl": "t_app_emit_once", "node": "server"},
        ]
    )
    app = PipelineServerApp(cfg, model_dirs=[], logger=lambda m: None)
    app.setup()
    assert [ps.name for ps in app._local] == ["decide"]
    assert app._downstream is None
    # first frame: stage returns None -> no reply
    assert app.handle(serialize_message(Frame(frame_id=0))) is None
    # second frame: stage emits -> a serialized BrakeDecision
    reply = app.handle(serialize_message(Frame(frame_id=1)))
    assert reply is not None
    out = deserialize_message(reply)
    assert isinstance(out, BrakeDecision) and out.frame_id == 1
    assert app.stop_after_reply() is True  # default


def test_server_stop_after_reply_configurable():
    cfg = _cfg(
        [
            {"name": "camera", "impl": "camera", "node": "client"},
            {"name": "echo", "impl": "t_app_echo", "node": "server"},
        ],
        app_params={"stop_after_reply": False},
    )
    app = PipelineServerApp(cfg, model_dirs=[], logger=lambda m: None)
    app.setup()
    assert app.stop_after_reply() is False
    assert deserialize_message(app.handle(serialize_message(Frame(frame_id=0)))).detections


# --------------------------------------------------------------------------
# client app (fake source + fake ctx)
# --------------------------------------------------------------------------
class _Loc:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class _FakeSource:
    def __init__(self, speed, y):
        self._speed = speed
        self._y = y
        self.controls = []
        self.kicks = 0
        self.destroyed = False

    def vehicle_speed_mps(self):
        return self._speed

    def vehicle_location(self):
        return _Loc(0.0, self._y, 0.0)

    def apply_control(self, t, s, b):
        self.controls.append((t, s, b))

    def set_target_velocity(self, v):
        self.kicks += 1

    def destroy_obstacle(self):
        self.destroyed = True


class _Ctx:
    def __init__(self, source):
        self.source = source
        self.transport = None


def _control_params():
    return {
        "ego_vehicle": "audi.tt",
        "ego_speed": "40mph",
        "controllers": {
            "audi.tt": {"target_decel": 7.0, "max_decel": 7.3, "kp": 0.2, "max_ramp": 0.05}
        },
    }


def _client(source):
    cfg = _cfg(
        [
            {"name": "camera", "impl": "camera", "node": "client"},
            {"name": "decide", "impl": "t_app_emit_once", "node": "server"},
            {
                "name": "control",
                "impl": "brake_controller",
                "node": "client",
                "params": _control_params(),
            },
        ]
    )
    app = PipelineClientApp(
        cfg, run_id="r", rep=0, csv_path="/tmp/_pl_app_test.csv", logger=lambda m: None
    )
    app.setup(_Ctx(source))
    return app


def test_client_cruises_then_brakes_and_stops():
    src = _FakeSource(speed=18.0, y=95.0)
    app = _client(src)
    # pre-brake tick (above gate, no reply): cruises, applies control, not done
    app.translate_action(Frame(frame_id=0), None, _Ctx(src))
    assert src.controls and app.is_done(None) is False
    assert app._latched is False
    # reply arrives -> latch + destroy obstacle + brake metrics
    src._speed = 10.0
    app.translate_action(Frame(frame_id=1), BrakeDecision(brake=True, frame_id=1), _Ctx(src))
    assert app._latched is True and src.destroyed is True and app._transmit is False
    # ego comes to rest -> stop recorded, done
    src._speed = 0.05
    app.translate_action(Frame(frame_id=2), None, _Ctx(src))
    assert app.is_done(None) is True
    # all control commands are (throttle, steer, brake) triples
    assert all(len(c) == 3 for c in src.controls)


def test_client_process_frame_emits_wire_message():
    src = _FakeSource(speed=18.0, y=95.0)
    app = _client(src)
    payload = app.process_frame(Frame(image=7, frame_id=0, meta={"capture_time": 1.0}), _Ctx(src))
    msg = deserialize_message(payload)
    assert isinstance(msg, Message) and msg.image == 7  # camera passthrough carried the frame


def test_a_pipeline_client_ends_when_the_ego_is_removed():
    """The scenario ending first is an ending, not something to wait out.

    Measured on the distributed tier: the scenario destroyed the ego and
    exited, and this client went on sending into a broken connection until the
    active limit killed the run 120 s later. The run now ends there, and its
    verdict says why instead of reporting the frames it had already sent as a
    success.
    """
    from cadet.pipeline.app import PipelineApp, PipelineClientApp

    app = PipelineClientApp.__new__(PipelineClientApp)
    app._source_done = False
    app._done = False
    app._ego_gone = False
    app._ego_checked_at = 0.0
    app._ext = None
    app._last_sim_time = 0.0
    logged = []
    app.log = logged.append

    class _Gone(object):
        @staticmethod
        def ego_is_alive():
            return False

    app.source = _Gone()
    assert app.is_done(None) is True
    assert app._ego_gone is True
    assert any("no longer in the world" in line for line in logged)

    ok, detail = PipelineApp.client_success(app)
    assert ok is False
    assert "removed from the world" in detail


def test_a_pipeline_run_with_its_ego_present_is_unchanged():
    """The path must be invisible to every run that does not lose its ego.

    Nothing is logged, nothing is marked, the ending stays the one the run
    always reached, and a source with no ego at all - replay, dataset,
    synthetic - is never asked.
    """
    from cadet.pipeline.app import PipelineApp, PipelineClientApp

    def _fresh(source):
        app = PipelineClientApp.__new__(PipelineClientApp)
        app._source_done = False
        app._done = False
        app._ego_gone = False
        app._ego_checked_at = 0.0
        app._ext = None
        app._last_sim_time = 0.0
        app.log = lambda line: None
        app.source = source
        return app

    class _Live(object):
        @staticmethod
        def ego_is_alive():
            return True

    class _NoEgo(object):
        """A replay or synthetic source: no ego to have lost."""

    for source in (_Live(), _NoEgo()):
        app = _fresh(source)
        assert app.is_done(None) is False
        assert app._ego_gone is False

    # An exhausted source still ends the run, and still ends it first.
    app = _fresh(_NoEgo())
    app._source_done = True
    assert app.is_done(None) is True
    assert app._ego_gone is False

    # The verdict for a run that kept its ego is the base claim, unchanged.
    app = _fresh(_Live())

    class _Stats(object):
        sent = 12

    app.message_stats = lambda: _Stats()
    ok, detail = PipelineApp.client_success(app)
    assert ok is True
    assert "sent=12" in detail


def test_the_simulator_is_asked_at_most_once_a_second():
    """The question is a round trip, on the loop the latencies come from."""
    from cadet.core.source import EGO_CHECK_INTERVAL_S, ego_removed

    asked = []

    class _Live(object):
        @staticmethod
        def ego_is_alive():
            asked.append(1)
            return True

    source = _Live()
    checked_at = 0.0
    gone, checked_at = ego_removed(source, checked_at, now=1000.0)
    assert gone is False and checked_at == 1000.0 and len(asked) == 1

    # Inside the interval: not asked again, and the timestamp is left alone so
    # the next call is still measured from the question that was actually put.
    for moment in (1000.1, 1000.5, 1000.0 + EGO_CHECK_INTERVAL_S - 0.01):
        gone, checked_at = ego_removed(source, checked_at, now=moment)
        assert gone is False and checked_at == 1000.0
    assert len(asked) == 1

    gone, checked_at = ego_removed(source, checked_at, now=1000.0 + EGO_CHECK_INTERVAL_S)
    assert len(asked) == 2

    # A source that cannot be asked is never reported as having lost an ego.
    gone, unchanged = ego_removed(object(), 0.0, now=9999.0)
    assert gone is False and unchanged == 0.0
