"""The foundation-model interface: answer questions about a run CADET wrote.

What this is for:
    CADET's analysis layer ends in files - a config, a run marker, link
    reports, per-request and per-frame CSVs, application summaries and device
    metrics. This package reads those files for one run or for a whole sweep,
    reduces them to a record small enough to send, and asks a model about it.
    It is log reading and question answering over what a run already wrote, and
    nothing else: nothing here triggers, activates, routes or dispatches
    anything, and no code path from a running experiment reaches it.

What it contains:
    - :mod:`cadet.analysis.context`: the reduction, and its rendering to text.
      Every file read goes through the dashboard's existing readers.
    - :mod:`cadet.analysis.backend`: the backend seam, an OpenAI-compatible
      chat-completions client over the standard library, and the resolution of
      the ``analysis`` config section into one.
    - :mod:`cadet.analysis.session`: the prompts.

How it fits into CADET:
    Behind ``cadet summarize <path>`` and ``cadet ask <path> "<question>"``
    (:mod:`cadet.cli`). Both accept a single run directory or a tree of them.
    Reads a run tree and writes nothing to it.

How to configure it:
    The ``analysis`` section of a config
    (:class:`cadet.config.schema.AnalysisConfig`): the endpoint and the model
    name, plus the name of the environment variable the key is read from. The
    default is ``backend: none``, in which state both subcommands refuse and
    say what to set. There is no config field for a key.
"""

from cadet.analysis.backend import (
    Backend,
    BackendError,
    ChatCompletionsBackend,
    NotConfigured,
    load_analysis_config,
    make_backend,
)
from cadet.analysis.context import (
    ContextError,
    build_context,
    render_context,
    render_one,
)
from cadet.analysis.session import SUMMARIZE_TASK, SYSTEM_PROMPT, respond

__all__ = [
    "Backend",
    "BackendError",
    "ChatCompletionsBackend",
    "ContextError",
    "NotConfigured",
    "SUMMARIZE_TASK",
    "SYSTEM_PROMPT",
    "build_context",
    "load_analysis_config",
    "make_backend",
    "render_context",
    "render_one",
    "respond",
]
