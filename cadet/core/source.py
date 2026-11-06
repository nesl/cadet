"""Source interface: where pipeline input frames come from.

What this is for:
    A uniform way to pull input frames regardless of backend: CARLA, replay,
    synthetic, and live. Mirrors the schema's discriminated ``source`` union.

What it contains:
    - :class:`Frame`: a minimal frame container (image + metadata).
    - :class:`Source`: the abstract ``open``/``next_frame``/``close``/``metadata``
      lifecycle.
    - :func:`ego_removed`: whether a simulated source's ego has left the world,
      asked at a bounded rate. Here rather than in either client app so the two
      cannot answer the same question differently.

How it fits into CADET:
    A runner opens a source built from ``config.source`` (see
    :mod:`cadet.sources`), pulls frames with :meth:`Source.next_frame`, feeds them
    through the pipeline, and closes it.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

# How often a client may ask the simulator whether its ego still exists. The
# answer is a round trip, and the question is asked from the loop the latency
# numbers are measured on, so it is asked once a second rather than per frame:
# a run that has lost its ego ends a second later than it could, which is
# nothing beside the alternative of waiting out the active limit.
EGO_CHECK_INTERVAL_S = 1.0


def ego_removed(source, checked_at, now=None):
    """Return ``(removed, checked_at)`` for a source whose ego may be gone.

    ``removed`` is True only when the source says so; ``checked_at`` is the
    timestamp to keep for the next call, unchanged when the rate limit meant no
    question was asked. A source with no ``ego_is_alive`` is never reported as
    having lost one, so a replay or synthetic source is unaffected.

    Why a client needs this at all: a scenario that ends before the vehicle
    stops destroys the ego, and the client then waits for a stop that cannot
    arrive until something else times out. Why it is asked of the source rather
    than of the handle: in the simulator version this was measured against,
    neither ``is_alive`` nor a lookup by id stops answering for an actor the
    server has destroyed - only the world's own actor list does, and
    :meth:`cadet.sources.carla.CarlaSource.ego_is_alive` is where that lives.
    """
    alive = getattr(source, "ego_is_alive", None)
    if alive is None:
        return False, checked_at
    moment = time.time() if now is None else now
    if moment - checked_at < EGO_CHECK_INTERVAL_S:
        return False, checked_at
    return (not alive()), moment


class Frame:
    """A single input frame: an image payload plus arbitrary metadata."""

    def __init__(self, image: Any = None, meta: Optional[Dict[str, Any]] = None):
        self.image = image
        self.meta = meta or {}


class Source(ABC):
    """Abstract frame source with an explicit open/close lifecycle."""

    @abstractmethod
    def open(self) -> None:
        """Acquire the backend (connect to sim, open file, start sensor)."""

    @abstractmethod
    def next_frame(self) -> Optional[Frame]:
        """Return the next :class:`Frame`, or ``None`` if none is available now.

        ``None`` is **not** by itself the end of the stream. A file or a
        generator has nothing left when it returns None, but a live source
        returns None whenever nothing has arrived yet - which is always true on
        the first tick, before its first sensor callback. Ask
        :meth:`exhausted` to tell the two apart.
        """

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Return static source metadata (resolution, fps, map, …)."""

    # -- optional two-phase startup -------------------------------------------
    # A source whose scene is created by something else - a simulator scenario
    # spawning the ego - cannot do all of its setup at once: the part that needs
    # the scene has to wait for it. These two hooks split that, and default to
    # doing nothing, so a source that owns everything it needs (a file, a
    # generator, a sensor) is unaffected and need not implement them.

    def prepare(self) -> None:
        """Do the setup that does not need the scene to exist yet.

        Called before the client announces it is ready. For a simulator source
        this is connecting to the simulator; for everything else there is
        nothing to do, and :meth:`open` remains the whole of the startup.
        """

    def await_scene(self, timeout_s: float) -> bool:
        """Block until the scene this source needs exists. True if it appeared.

        Called after the client has announced it is ready and the scenario has
        been started. The default returns True immediately: a source with no
        external scene is always ready.
        """
        return True

    def exhausted(self) -> bool:
        """True if no further frames will ever arrive.

        Only meaningful straight after :meth:`next_frame` returned ``None``,
        which is the one case where a consumer has to know whether to wait or to
        stop. The default is True, which is the behaviour a file or a generator
        wants and what the interface used to imply unconditionally; a live source
        overrides it, because its empty moments are not its end.
        """
        return True
