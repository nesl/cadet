"""Decorator-based app plugin registry (framework, app-agnostic).

What this is for:
    Apps register themselves here. Each plugin advertises its ``name`` and its
    typed ``params_model``, so the loader can validate ``app.params`` against the
    right model. The framework ships no app registrations; apps live under
    ``examples/`` and register on import (e.g. ``--app-module examples.braking.flat.app``).

What it contains:
    - :class:`AppPlugin`: the abstract base every app subclasses.
    - :func:`register_app`: class decorator that adds a plugin to the registry.
    - :func:`get_app` / :func:`available_apps`: registry lookup.

How it fits into CADET:
    ``experiment.app``/``app.plugin`` name a registered plugin. The CLI imports the
    requested app module(s) before dispatch; :func:`cadet.config.loader.resolve_app_params`
    then validates the config's ``app.params`` against the plugin's model. Nothing
    under ``cadet/`` imports ``examples/``.
"""

import os
import sys
from abc import ABC
from typing import Dict, List, Type

from pydantic import BaseModel

_REGISTRY: Dict[str, Type["AppPlugin"]] = {}


class AppPlugin(ABC):
    """Base class for an app plugin.

    Subclasses set ``name`` (the plugin key used in configs) and ``params_model``
    (the pydantic model validating ``app.params`` for this app).
    """

    name: str = ""
    params_model: Type[BaseModel]

    @staticmethod
    def client_success(client_app):
        """Return ``(ok, detail)``: did this client actually produce output?

        Only the app knows what its output is, so the app declares it. The
        default is the weakest claim that still means something happened: the
        client transmitted at least one message. An app whose run is not
        meaningful without a reply - a decision coming back, a command acted on -
        overrides this and says so.

        This exists because a client that set up, sent nothing and exited
        cleanly used to be indistinguishable from one that ran the experiment:
        success was decided by exit codes, and exiting zero having done nothing
        is still exiting zero.

        ``client_app`` is the app's own client object, so it is read through the
        same ``message_stats()`` the link report uses rather than through a
        second accounting of the same thing.
        """
        stats_fn = getattr(client_app, "message_stats", None)
        if stats_fn is None:
            # An app that reports no statistics makes no claim, so there is
            # nothing to contradict. Saying "no output" here would fail a run
            # for not being instrumented.
            return True, "app reports no message statistics"
        stats = stats_fn()
        if stats is None:
            return True, "app reports no message statistics"
        if stats.sent < 1:
            return False, "client transmitted no messages"
        return True, "sent={}".format(stats.sent)

    @staticmethod
    def sim_hooks(config, model_search_dirs):
        """Sim-side processes to start before servers/clients (default: none).

        An app declares what its simulator needs (e.g. the braking app returns a
        ScenarioRunner :class:`~cadet.core.launcher.LaunchSpec` so the ego exists
        before the client attaches). Returns a list of ``LaunchSpec``; the
        orchestrator starts them first and tears them down last.
        """
        return []


# The app plugins that ship with the framework. These are imported on demand by
# get_app, so a config naming one does not have to ask for its module.
BUILTIN_APP_MODULES = ("cadet.pipeline.app",)
_builtin_apps_loaded = False


def register_app(cls: Type[AppPlugin]) -> Type[AppPlugin]:
    """Class decorator that registers an :class:`AppPlugin` by its ``name``."""
    if not getattr(cls, "name", ""):
        raise ValueError("app plugin {!r} must set a non-empty `name`".format(cls.__name__))
    if cls.name in _REGISTRY:
        raise ValueError("app plugin {!r} already registered".format(cls.name))
    _REGISTRY[cls.name] = cls
    return cls


class UnknownAppError(KeyError):
    """No app plugin is registered under the requested name.

    A ``KeyError`` subclass, because that is what :func:`get_app` has always
    raised and callers catch it; the message is readable so the name reaches the
    person running the command rather than only a traceback.
    """

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


def _load_builtin_apps() -> None:
    """Import the framework's own app plugins, once.

    A built-in that exists only once something imports it is not a built-in:
    ``app.plugin: pipeline`` failed with a bare KeyError unless the caller also
    passed ``--app-module cadet.pipeline.app`` for a plugin that ships here,
    and the two example configs that use it said nothing about needing to.

    Imported on the first lookup rather than at module import, because the
    plugin modules import the framework and this module is part of it. The flag
    is set before the import so a module that fails to import is not retried on
    every lookup, which would turn one error into many.
    """
    global _builtin_apps_loaded
    if _builtin_apps_loaded:
        return
    _builtin_apps_loaded = True
    import_app_modules(list(BUILTIN_APP_MODULES))


def get_app(name: str) -> Type[AppPlugin]:
    """Return the plugin registered under ``name``.

    Raises :class:`UnknownAppError`, a ``KeyError``, naming what is registered.
    """
    if name not in _REGISTRY:
        _load_builtin_apps()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise UnknownAppError(
            "no app plugin named {!r}. Registered: {}. An app that lives in "
            "examples/ is imported with --app-module (e.g. --app-module "
            "examples.braking.flat.app).".format(
                name, ", ".join(available_apps()) or "none"
            )
        )


def available_apps() -> List[str]:
    """Return the sorted names of all registered app plugins."""
    return sorted(_REGISTRY)


def import_app_modules(modules: List[str]) -> None:
    """Import each module so its ``@register_app`` plugins register.

    Used by the CLI (``--app-module``). Keeps ``cadet/`` free of ``examples/``
    imports: the app module path is supplied at run time.

    The working directory is searched, which is not the default and is the
    reason this is more than one line. An editable install puts the ``cadet``
    package on the path and nothing else, and a console script's own
    ``sys.path[0]`` is the directory the script lives in - the virtualenv's
    ``bin`` - so ``cadet run --app-module examples.distributed_sensing.app`` from a fresh clone
    could not import ``examples`` at all, while the same command as ``python -m
    cadet`` could, because that form puts the working directory on the path.
    That difference is invisible from the command line and the error it produced
    said only that no module was named ``examples``.

    A module named on the command line is a path into whatever project the
    caller is standing in, so the working directory is where it is looked for.
    It is appended rather than prepended: an installed module of the same name
    keeps winning, so a directory that happens to hold a ``json.py`` cannot
    shadow the standard library.
    """
    import importlib

    here = os.getcwd()
    added = here not in sys.path
    if added:
        sys.path.append(here)
    try:
        for mod in modules:
            importlib.import_module(mod)
    finally:
        if added:
            try:
                sys.path.remove(here)
            except ValueError:
                pass
