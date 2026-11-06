"""The model backend behind ``cadet summarize`` and ``cadet ask``.

What this is for:
    One seam between CADET and whatever answers a question about a run. A
    backend takes two strings and returns one, so a stub is a four-line class
    and **no test in the suite reaches the network**: the tests inject a stub,
    and the only code here that opens a socket is
    :class:`ChatCompletionsBackend`, which is exercised with
    ``urllib.request.urlopen`` replaced.

What it contains:
    - :class:`Backend`: the seam - :meth:`Backend.complete`.
    - :class:`ChatCompletionsBackend`: an OpenAI-compatible
      ``/chat/completions`` endpoint, or a local model behind the same shape,
      over the standard library and no new dependency (such a call is an HTTP
      POST with a JSON body).
    - :func:`make_backend`: the backend a resolved ``analysis`` section asks
      for, or :class:`NotConfigured` saying what to set.
    - :func:`load_analysis_config`: that section, out of a config file.
    - :class:`BackendError` / :class:`NotConfigured`.

How it fits into CADET:
    Called by :mod:`cadet.cli` for the two subcommands and by
    :mod:`cadet.analysis.session`, which is handed a backend rather than making
    one. Nothing else in ``cadet/`` imports it, and no run-time path does: a
    run does not call a model, a reader of a run does.

How to configure it:
    The ``analysis`` section of a config (see
    :class:`cadet.config.schema.AnalysisConfig`). ``analysis.backend`` defaults
    to ``none``, so **the default state is "no backend configured"** and both
    subcommands refuse with a message naming what to set.

The key, and where it is not:
    ``analysis.api_key_env`` names an environment variable; the key itself is
    read from the environment at the moment of the request and is held in the
    backend instance and nowhere else. It is **never** written to a config, a
    log, an error message or the model's context: no message here is built from
    a request header or a response body, and a failure names the endpoint and
    the HTTP status, which are the two facts a reader needs to tell a wrong URL
    from a refused request from an overloaded server.
"""

import json
import os
import socket
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

import yaml

# The one message the default state prints. Written out in full because it is
# the first thing most readers will see from this feature, and "no backend
# configured" on its own tells nobody what to do about it.
NO_BACKEND = (
    "no analysis backend configured. Set, in the `analysis` section of a "
    "config (pass one with --config, or run against a run directory whose "
    "config.yaml carries it):\n"
    "  analysis.backend: openai_chat\n"
    "  analysis.endpoint: <an OpenAI-compatible chat-completions URL>\n"
    "  analysis.model: <the model name that endpoint serves>\n"
    "and put the API key in the environment variable named by "
    "analysis.api_key_env (default {}). The key belongs in the environment "
    "and nowhere else: there is no config field for it."
)

# How long a reply may be. A bound rather than none, because an endpoint that
# streams until it is stopped is a cost and a hang, and a summary of one run
# does not need more.
MAX_REPLY_TOKENS = 1024


class BackendError(Exception):
    """A request to a configured backend did not produce an answer."""


class NotConfigured(BackendError):
    """No backend is configured, or its configuration is incomplete.

    Separate from :class:`BackendError` because it is not a failure of a
    request - nothing was sent - and the caller says what to set rather than
    reporting that something went wrong.
    """


class Backend(object):
    """Something that answers a prompt. The injection point.

    Two strings in, one string out. Deliberately not a client object with a
    session, a retry policy and a token budget: everything above this class
    works on text, so a test's stub is a class with one method and the suite
    never opens a socket.
    """

    def complete(self, system: str, user: str) -> str:
        """Answer ``user`` under the instructions ``system``."""
        raise NotImplementedError


class ChatCompletionsBackend(Backend):
    """An OpenAI-compatible ``/chat/completions`` endpoint over the stdlib.

    No new dependency: the call is one HTTP POST of a JSON body, which
    ``urllib.request`` makes. A local model served behind the same route shape
    is the same backend with a different endpoint, which is the reason the
    endpoint is configuration and the wire format is not.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str,
        timeout_s: float = 60.0,
        temperature: float = 0.0,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        # The one place the key lives in this process. Not in the config, not in
        # __repr__, not in any message this module raises.
        self._api_key = api_key

    def __repr__(self) -> str:
        """The endpoint and model, which is all of this that is printable."""
        return "ChatCompletionsBackend(endpoint={!r}, model={!r})".format(
            self.endpoint, self.model
        )

    def complete(self, system: str, user: str) -> str:
        """POST the two messages and return the reply's text.

        Every failure becomes a :class:`BackendError` naming the endpoint and
        the HTTP status. Neither the request headers nor the response body
        reach the message: the body of a 4xx from a proxy has been known to
        echo the request back.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": MAX_REPLY_TOKENS,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        # A model served on the machine the command runs on authenticates
        # nobody, and an endpoint that wants no key is told apart from one
        # whose key is missing by the config: an empty `analysis.api_key_env`
        # says this endpoint takes none, so no header is sent at all.
        if self._api_key:
            headers["Authorization"] = "Bearer {}".format(self._api_key)
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                status = getattr(response, "status", None) or response.getcode()
                body = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            raise BackendError(
                "request to {} failed: HTTP {} (response body not shown)".format(
                    self.endpoint, exc.code
                )
            )
        except socket.timeout:
            raise BackendError(
                "request to {} timed out after {:g} s (analysis.timeout_s)".format(
                    self.endpoint, self.timeout_s
                )
            )
        except urllib.error.URLError as exc:
            # No status to name: the request never reached a server. The reason
            # is the socket's own (refused, unknown host, TLS), which carries
            # nothing from the request.
            reason = exc.reason
            if isinstance(reason, socket.timeout):
                raise BackendError(
                    "request to {} timed out after {:g} s (analysis.timeout_s)".format(
                        self.endpoint, self.timeout_s
                    )
                )
            raise BackendError(
                "request to {} failed before any HTTP status: {}".format(self.endpoint, reason)
            )
        except OSError as exc:
            raise BackendError(
                "request to {} failed before any HTTP status: {}".format(
                    self.endpoint, exc.strerror or exc.__class__.__name__
                )
            )
        return self._text_of(body, status)

    def _text_of(self, body: str, status: Any) -> str:
        """The reply text out of a chat-completions body.

        A 200 with a shape this does not understand is a failure and not an
        empty answer: an empty summary printed as though a model had produced
        it is the one outcome worse than an error.
        """
        try:
            data = json.loads(body)
        except ValueError:
            raise BackendError(
                "response from {} was not JSON (HTTP {})".format(self.endpoint, status)
            )
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise BackendError(
                "response from {} held no completion (HTTP {})".format(self.endpoint, status)
            )
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise BackendError(
                "response from {} held no completion text (HTTP {})".format(self.endpoint, status)
            )
        return text.strip()


def make_backend(config: Any) -> Backend:
    """The backend a resolved ``analysis`` section asks for.

    Raises :class:`NotConfigured` for the default state and for every
    incomplete one, each naming the field or the environment variable that is
    missing. Nothing is sent and no socket is opened here.
    """
    backend = getattr(config, "backend", "none")
    if backend == "none":
        raise NotConfigured(NO_BACKEND.format(getattr(config, "api_key_env", "")))
    if backend != "openai_chat":
        raise NotConfigured("unknown analysis.backend: {!r}".format(backend))
    missing = [
        name for name in ("endpoint", "model") if not getattr(config, name, None)
    ]
    if missing:
        raise NotConfigured(
            "analysis.backend is {!r} but {} not set".format(
                backend, " and ".join("analysis." + name for name in missing)
            )
        )
    env = config.api_key_env
    # An empty name is the config saying this endpoint needs no key, which is
    # the state a model served locally is in. A name that is set and an
    # environment that does not hold it is the misconfiguration.
    key = ""
    if env:
        key = os.environ.get(env, "")
        if not key.strip():
            raise NotConfigured(
                "no API key in the environment: set ${} (named by "
                "analysis.api_key_env) to the key for {}".format(env, config.endpoint)
            )
    return ChatCompletionsBackend(
        endpoint=config.endpoint,
        model=config.model,
        api_key=key,
        timeout_s=config.timeout_s,
        temperature=config.temperature,
    )


def load_analysis_config(path: Optional[str], target: Optional[str] = None) -> Any:
    """The ``analysis`` section to use, as an :class:`AnalysisConfig`.

    ``path`` is an explicit config file. With none, the section is taken from
    ``<target>/config.yaml`` - the resolved config a run directory already
    holds - so a run can be summarised without naming a second file. Neither
    present, or neither carrying the section, gives the defaults, which is the
    "no backend configured" state.

    Only the ``analysis`` mapping is read, and it is validated on its own rather
    than through the whole :class:`cadet.config.schema.Config`: the file may be
    a resolved config saved by an older schema, and whether *that* still
    validates has nothing to do with where to send a question.
    """
    from cadet.config.schema import AnalysisConfig

    for candidate in (path, os.path.join(target, "config.yaml") if target else None):
        if not candidate:
            continue
        section = _analysis_section(candidate, required=candidate == path)
        if section is not None:
            return AnalysisConfig(**section)
    return AnalysisConfig()


def _analysis_section(path: str, required: bool) -> Optional[Dict[str, Any]]:
    """The ``analysis`` mapping in a YAML file, or None.

    ``required`` distinguishes the two callers: a config the user named must
    exist and parse, while the run directory's own ``config.yaml`` is a
    convenience and its absence is the ordinary case.
    """
    try:
        with open(path) as handle:
            data = yaml.safe_load(handle)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        if required:
            raise NotConfigured("cannot read {}: {}".format(path, exc))
        return None
    if not isinstance(data, dict):
        if required:
            raise NotConfigured("{} is not a config mapping".format(path))
        return None
    section = data.get("analysis")
    if section is None:
        return None
    if not isinstance(section, dict):
        raise NotConfigured("the `analysis` section of {} is not a mapping".format(path))
    return section
