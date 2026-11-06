"""Sphinx configuration for the CADET documentation."""

import os
import re
import subprocess
import sys
from pathlib import Path

from docutils.parsers.rst import Directive

# Make the cadet package importable for autodoc.
sys.path.insert(0, os.path.abspath("../.."))

project = "CADET"
author = ""
release = ""

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

html_theme = "furo"
html_title = "CADET"

autodoc_member_order = "bysource"
autodoc_typehints = "description"
suppress_warnings = ["app.add_node"]

# Heavy runtime dependencies that are not installed in the docs venv.
autodoc_mock_imports = [
    "carla",
    "torch",
    "torchvision",
    "ultralytics",
    "cv2",
    "open3d",
    "filterpy",
    "lapsolver",
    "numpy",
]

# MyST — allow .md files in toctrees.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Omit copyright / version / timestamps from rendered output.
html_show_copyright = False
html_show_sphinx = False
html_last_updated_fmt = None

html_theme_options = {
    "footer_icons": [],
    "source_repository": "https://github.com/nesl/cadet/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

html_static_path = ["_static"]

html_js_files = [
    "sidebar-scroll.js",
]

html_css_files = [
    "cadet-table.css",
]

# ---------------------------------------------------------------------------
# Placeholder check: fail the build if any source file contains known
# placeholder patterns that should have been replaced before publishing.
# ---------------------------------------------------------------------------
_PLACEHOLDER_RE = re.compile(
    # Catch placeholder tokens:
    # - tokens with hyphens like <clone-url> or <HOST-IP>
    # - ALL-CAPS tokens like <HOST_IP>, <REPO_URL> (min 2 chars, all uppercase)
    # - [foo placeholder] style
    # - TODO, TBD
    # Excludes short lowercase CLI metavars like <config>, <n>, <dir>
    r"(<[A-Za-z][A-Za-z0-9_]*-[A-Za-z0-9_-]+>"   # hyphenated: <clone-url>
    r"|<[A-Z][A-Z0-9_]{1,}>"                       # ALL-CAPS: <HOST_IP>
    r"|\bTODO\b|\bTBD\b"
    r"|\[.*?placeholder\])",
)

_DOCS_SOURCE = os.path.abspath(os.path.dirname(__file__))
# Repo root is two levels above docs/source/
_REPO_ROOT = os.path.abspath(os.path.join(_DOCS_SOURCE, "..", ".."))

# ---------------------------------------------------------------------------
# Inline-YAML check: fail the build if any source file uses a code-block yaml
# directive instead of a literalinclude. Config snippets must live under
# examples/ or docs/snippets/ so they can be validated with cadet validate.
# ---------------------------------------------------------------------------
# Pages allowed to use inline yaml code-blocks.
# config-reference is machine-generated. All others are pre-existing pages
# that predate the literalinclude policy; new pages must use literalinclude.
_INLINE_YAML_ALLOWLIST = {
    "config-reference",
    "explanation/network-backends",
    "explanation/processing-core-layer",
    "explanation/reproducibility",
    "howto/cadet-down",
    "howto/link-distributions",
    "howto/model-registry",
    "howto/replay-source",
    "howto/sweeps",
    "howto/synthetic-source",
    "howto/transport-tuning",
    "tutorials/01-first-experiment",
    "tutorials/03-backends",
    "tutorials/04-multi-device",
    "tutorials/05-sweeps",
    "tutorials/06-write-an-app",
    "tutorials/07-v2v-v2i",
}

_INLINE_YAML_RE = re.compile(
    r"\.\.\s+code-block::\s+yaml"   # RST code-block directive
    r"|```\s*yaml",                  # MyST fenced block
    re.IGNORECASE,
)


def _check_inline_yaml(app, docname, source):
    """Sphinx source-read event: fail on inline YAML code blocks."""
    if docname in _INLINE_YAML_ALLOWLIST:
        return
    text = source[0]
    if _INLINE_YAML_RE.search(text):
        raise ValueError(
            "Inline YAML code block in {}: use literalinclude of a validated "
            "snippet under docs/snippets/ or examples/ instead".format(docname)
        )


def _check_placeholders(app, docname, source):
    """Sphinx source-read event: fail on placeholder text."""
    text = source[0]
    matches = _PLACEHOLDER_RE.findall(text)
    if matches:
        raise ValueError(
            "Placeholder text found in {}: {}".format(docname, matches)
        )


# ---------------------------------------------------------------------------
# Snippet validation: run cadet validate on every literalinclude'd .yaml file.
# A snippet a reader is shown is a snippet a reader can run, so a snippet that
# does not validate fails the build. There is no exempt category: the pages
# carry no "target behaviour" blocks and the build writes no burndown report,
# because a published page that lists what does not work is not something this
# project publishes.
# ---------------------------------------------------------------------------

# Registry of (rst_docname, yaml_abs_path) collected at source-read.
_SNIPPET_REFS: list = []

# The example app plugins the snippets name. The framework never imports
# example code, so `cadet validate` has to be told which modules register the
# plugins a config asks for -- exactly as a reader has to pass --app-module.
# Each module here imports with nothing but the base dependencies: the model
# backends they can use are imported lazily, inside the call that needs one.
_SNIPPET_APP_MODULES = (
    "examples.reference.app",
    "examples.distributed_sensing.app",
    "examples.braking.flat.app",
    "examples.v2v.app",
    "examples.v2i.app",
    "examples.dataset_vlm.app",
)


def _collect_literalincludes(app, docname, source):
    """Collect .yaml literalinclude paths for post-build validation."""
    text = source[0]
    for m in re.finditer(r"\.\.\s+literalinclude::\s+(\S+\.yaml)", text):
        rel = m.group(1)
        # Resolve relative to the .rst file's directory
        rst_dir = os.path.join(_DOCS_SOURCE, *docname.split("/")[:-1])
        abs_path = os.path.normpath(os.path.join(rst_dir, rel))
        _SNIPPET_REFS.append((docname, abs_path))


def _validate_snippets(app, exception):
    """Build-finished hook: validate every collected yaml snippet."""
    if exception:
        return

    app_module_args = []
    for module in _SNIPPET_APP_MODULES:
        app_module_args += ["--app-module", module]

    failures = []
    for docname, yaml_path in _SNIPPET_REFS:
        if not os.path.isfile(yaml_path):
            continue  # missing-file warning already emitted by Sphinx
        # Only validate YAML files under docs/snippets/ or examples/
        if not (
            yaml_path.startswith(os.path.join(_REPO_ROOT, "docs", "snippets"))
            or yaml_path.startswith(os.path.join(_REPO_ROOT, "examples"))
        ):
            continue
        result = subprocess.run(
            [sys.executable, "-m", "cadet", "validate", yaml_path] + app_module_args,
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        if result.returncode != 0:
            err = (result.stdout + result.stderr).strip()
            failures.append((docname, yaml_path, err))

    checked = len({path for _, path in _SNIPPET_REFS})
    print("\n[snippets] {} snippet(s) checked with cadet validate.".format(checked))

    if failures:
        msgs = [
            "{}: {} - {}".format(doc, os.path.relpath(path, _REPO_ROOT), err[:200])
            for doc, path, err in failures
        ]
        raise SystemExit(
            "Build error: {} snippet(s) fail cadet validate:\n".format(len(failures))
            + "\n".join(msgs)
        )


class RunTagDirective(Directive):
    """No-op directive that marks a following code block for run_docs_commands.py."""

    required_arguments = 1
    optional_arguments = 0
    has_content = False

    def run(self):
        return []


def _check_empty_codeblocks(app, exception):
    """Fail the build if any literalinclude or code-block rendered empty."""
    if exception:
        return
    import re

    outdir = Path(app.outdir)
    empty = []
    # Match highlight divs: <div class="highlight-* notranslate">...<pre>...</pre>
    pat = re.compile(
        r'<div class="highlight-[^"]*[^>]*>.*?<pre[^>]*>(.*?)</pre>',
        re.DOTALL,
    )
    for html_file in outdir.rglob("*.html"):
        text = html_file.read_text(encoding="utf-8", errors="replace")
        for m in pat.finditer(text):
            inner = re.sub(r"<[^>]+>", "", m.group(1)).strip()
            if not inner:
                empty.append(str(html_file.relative_to(outdir)))
    if empty:
        seen = sorted(set(empty))
        raise SystemExit(
            "Build error: empty code block(s) in: " + ", ".join(seen)
        )


def setup(app):
    app.add_directive("run-tag", RunTagDirective)
    app.connect("source-read", _check_inline_yaml)
    app.connect("source-read", _check_placeholders)
    app.connect("source-read", _collect_literalincludes)
    app.connect("build-finished", _validate_snippets)
    app.connect("build-finished", _check_empty_codeblocks)
