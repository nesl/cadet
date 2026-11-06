"""The two questions ``cadet summarize`` and ``cadet ask`` put to a model.

What this is for:
    Everything between a reduced run record (:mod:`cadet.analysis.context`) and
    a backend (:mod:`cadet.analysis.backend`): the standing instructions, the
    fixed summarise task, and how a user's question is attached.

What it contains:
    - :data:`SYSTEM_PROMPT`: the standing instructions, the whole of which is
      "answer from the record, and say so when the record does not".
    - :data:`SUMMARIZE_TASK`: what ``cadet summarize`` asks.
    - :func:`respond`: record plus question plus backend, in; the answer, out.

How it fits into CADET:
    Called by :mod:`cadet.cli`. Takes a :class:`cadet.analysis.backend.Backend`
    rather than making one, so every test injects a stub and none reaches the
    network. Writes nothing and reads no file.

How to configure it:
    Nothing. The prompts are not configuration: they are what makes an answer
    about a run's own records rather than about distributed systems in general,
    and a config field for them would make the provenance of an answer
    unreconstructible from the run tree.
"""

from typing import Any, Optional

# The standing instructions. Short on purpose: the long part of the prompt is
# the record, and every sentence added here is a sentence the record pays for.
SYSTEM_PROMPT = (
    "You are reading the record of an experiment run by CADET, a platform for "
    "evaluating distributed cooperative autonomy. The record below was reduced "
    "from the files one run or one sweep of runs wrote: its resolved config, "
    "its run marker, its link reports, its per-request and per-frame records, "
    "its application summary sidecars and its device metrics.\n"
    "\n"
    "Rules you follow without exception:\n"
    "- Answer only from the record. Every number you give must appear in it.\n"
    "- Where the record does not say, say that it does not, and name the part "
    "of it you looked at. Do not estimate, and do not fill a gap from general "
    "knowledge of networks or of machine learning.\n"
    "- Keep each number's unit as the record gives it, and do not convert "
    "between units.\n"
    "- The record's `notes` lines say which files were missing or unreadable. "
    "A measurement whose file is named there is not a measurement; say so "
    "rather than reporting it.\n"
    "- A configured network parameter and a measured one are different facts. "
    "Never report one as the other.\n"
    "- Be brief and concrete. No preamble."
)

# What `cadet summarize` asks. A fixed question rather than an empty prompt, so
# two summaries of two runs are comparable.
SUMMARIZE_TASK = (
    "Summarise this record: what was run, where the compute was placed and "
    "under what network condition, what the headline measurements were, and "
    "what in the record would make a reader doubt one of those measurements "
    "(a run that did not reach a terminal status, a file named in the notes, "
    "unanswered messages, a cell with fewer repetitions than its neighbours). "
    "If the record holds several runs, say what differs across them and which "
    "comparison the sweep supports."
)


def respond(record: str, question: Optional[str], backend: Any) -> str:
    """The answer to ``question`` about ``record``, from ``backend``.

    ``question`` of ``None`` is the summarise task. The record comes first in
    the prompt and the question last, which is the order that survives a long
    context.
    """
    task = "Question: {}".format(question.strip()) if question else SUMMARIZE_TASK
    return backend.complete(SYSTEM_PROMPT, "{}\n\n{}".format(record, task))
