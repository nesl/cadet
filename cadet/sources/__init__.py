"""CADET input sources.

Concrete :class:`cadet.core.source.Source` backends and a factory that picks one
by ``config.source.kind``: ``carla`` (:mod:`cadet.sources.carla`), ``replay``
(:mod:`cadet.sources.replay`), ``synthetic`` (:mod:`cadet.sources.synthetic`),
``live`` (:mod:`cadet.sources.live`, the replay path with a capture reader),
``dataset`` (:mod:`cadet.sources.dataset`). ``carla`` is imported lazily so the
framework works without CARLA installed.
"""

from cadet.core.source import Source


def build_source(config, logger=None) -> Source:
    """Construct the Source for ``config.source.kind``."""
    kind = config.source.kind
    if kind == "carla":
        from cadet.sources.carla import CarlaSource

        return CarlaSource(config, logger=logger)
    if kind == "replay":
        from cadet.sources.replay import ReplaySource

        return ReplaySource(config, logger=logger)
    if kind == "live":
        from cadet.sources.live import LiveSource

        return LiveSource(config, logger=logger)
    if kind == "synthetic":
        from cadet.sources.synthetic import SyntheticSource

        return SyntheticSource(config, logger=logger)
    if kind == "dataset":
        from cadet.sources.dataset import DatasetSource

        return DatasetSource(config, logger=logger)
    raise ValueError("no source implementation for kind {!r}".format(kind))
