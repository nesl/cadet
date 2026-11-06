"""The :class:`Stage` contract: one unit of pipeline work.

A stage is the smallest placeable unit of the pipeline, with a single
responsibility: turn one message into the next via ``process(msg) -> msg``.
Keeping the contract this small lets the builder place any stage on any node
and insert a transport boundary between stages transparently. Each subclass
declares its ``data_source`` (``perceived`` or ``ground_truth``) and implements
:meth:`process`; optional :meth:`setup` / :meth:`teardown` bracket a run.
Concrete stages register under an ``impl`` name with
:func:`cadet.pipeline.registry.register_stage`; the builder instantiates them
from ``pipeline.stages`` config. Per-stage ``params`` are available as
``self.params``; a stage reads only ``self.params`` and the message.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from cadet.pipeline.messages import Message

# A stage either consumes perceived data (detector/tracker output) or reads the
# simulator's ground truth directly; experiments must know which, so it is declared.
PERCEIVED = "perceived"
GROUND_TRUTH = "ground_truth"


class StageContext:
    """Setup-time context handed to a stage that needs more than its params.

    A pure stage ignores this; a stage that loads a model (the detector) reads the
    resolved ``config``, the ``node`` it runs on, the model search ``dirs`` and a
    ``logger``. Kept separate from ``params`` so the common case stays params-only.
    """

    def __init__(
        self,
        config=None,
        node: Optional[str] = None,
        model_dirs: Optional[List[str]] = None,
        logger=None,
    ):
        self.config = config
        self.node = node
        self.model_dirs = model_dirs or []
        self.log = logger or (lambda m: None)


class Stage(ABC):
    """Abstract base for a pipeline stage.

    Subclasses override :meth:`process` and may override :meth:`setup` and
    :meth:`teardown`. ``data_source`` declares whether the stage's output reflects
    perceived data or simulator ground truth.

    ``answers_every_message`` is for the run's link report, and only a terminal
    stage's answer matters: a stage that replies to every frame makes an
    unanswered frame a measurement of the link, and one that replies only when
    something happens makes an unanswered frame the normal case. The closed-loop
    brake controller answers on an event; the open-loop decision log answers
    every frame. Getting it wrong either hides a real loss or invents one, which
    is why the stage says so rather than the client guessing from the data.
    """

    data_source: str = PERCEIVED
    answers_every_message: bool = False

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}

    def setup(self, ctx: Optional["StageContext"] = None) -> None:
        """Acquire resources before the first message (load a model, connect).

        ``ctx`` is a :class:`StageContext` for stages that need the resolved config,
        node or model dirs; pure stages ignore it.
        """

    @abstractmethod
    def process(self, msg: Message) -> Optional[Message]:
        """Transform ``msg`` into the next message, or return ``None`` to drop it."""

    def teardown(self) -> None:
        """Release resources after the last message."""
