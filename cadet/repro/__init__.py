"""Reproducibility: reduce a tree of repetitions into one variance record.

What this is for:
    The orchestrator already repeats a config (``experiment.repetitions``,
    ``cadet sweep``, ``--resume``), so what was missing was not a second runner
    but the *reduction*: reading a directory that holds N repetition
    directories and producing one record that says what varied and by how much.
    Nothing here instruments a run; it reads only the files a run already
    writes.

What it contains:
    - :mod:`cadet.repro.extract`: per-repetition reading - the run marker, the
      per-frame CSV, the app summary sidecar, the per-node link report and
      client outcome, the server's per-request CSV and the collector CSVs - plus
      the measurement context (send interval, warm-up, Nagle) that has to be
      stated beside a number for it to mean anything.
    - :mod:`cadet.repro.reduce`: grouping repetitions by the config cell they
      belong to, the across-repetition statistics (median, IQR, standard
      deviation, count, how many repetitions produced the metric at all), the
      bimodality hint, and the JSON / Markdown renderings.

How it fits into CADET:
    Driven by ``cadet variance <dir>`` (:mod:`cadet.cli`). Input is a directory
    under ``outputs.root``; output is a report written into that run tree -
    never into the repository, because a variance report is a result.

How to configure it:
    Nothing to configure: it takes a path. Which files it reads is fixed by
    what the roles write, and a file a run did not produce is recorded as
    missing rather than averaged over.
"""

from cadet.repro.extract import Repetition, read_repetition
from cadet.repro.reduce import (
    REPORT_JSON,
    REPORT_MD,
    MetricSummary,
    discover_repetitions,
    reduce_tree,
    render_json,
    render_markdown,
    write_reports,
)

__all__ = [
    "MetricSummary",
    "REPORT_JSON",
    "REPORT_MD",
    "Repetition",
    "discover_repetitions",
    "read_repetition",
    "reduce_tree",
    "render_json",
    "render_markdown",
    "write_reports",
]
