"""Client-side visualiser, ported from the original ``carla_wrapper/ol_visualizer.py``.

What this is for:
    The original braking client rendered a pygame window every control tick (the
    camera frame plus a throttle/steer/brake overlay) and optionally saved a PNG per
    captured frame. That per-tick work set the client loop's rate (~10 fps under
    ``SDL_VIDEODRIVER=dummy``), which in turn sets the wall-clock dt the per_loop
    brake controller runs on. Reproducing it is what makes a CADET braking run's loop
    rate - and so its stopping dynamics - match the original's. It is a port,
    not a sleep: the same render and the same PNG write, so the cost is the
    original's cost.

What it contains:
    - :class:`ClientVisualizer`: ``render`` (blit the frame + overlay, flip) and
      ``save_frame`` (one PNG per captured frame), both headless-safe.

How it fits into CADET:
    The braking clients (flat ``examples.braking.flat.app`` and the ``pipeline``
    driver) construct one when ``app.params.client_visualizer`` is true (the default,
    matching the original), call ``render`` each control tick and ``save_frame`` per
    captured frame, and ``close`` at the end. Kept in ``cadet/`` so the framework does
    not depend on the examples. Runs headless when ``SDL_VIDEODRIVER=dummy``.
"""

import os


class ClientVisualizer:
    """Per-tick pygame render + per-frame PNG save (ported from ol_visualizer.py)."""

    def __init__(self, width, height, out_dir, save_frames=True, logger=None):
        # Before the import, because SDL reads it when it initialises: without
        # it SDL installs its own SIGINT/SIGTERM handlers and the process stops
        # responding to the signal the orchestrator stops it with. The launcher
        # sets it for every child as well; this covers a visualiser built
        # in-process, by a test or an app that runs the client directly.
        os.environ.setdefault("SDL_NO_SIGNAL_HANDLERS", "1")
        import pygame

        self._pg = pygame
        self.width = width
        self.height = height
        self._log = logger or (lambda m: None)
        pygame.init()
        # SWSURFACE + the dummy SDL driver keep this headless; the original used the
        # same surface mode.
        self.display = pygame.display.set_mode((width, height), pygame.SWSURFACE)
        pygame.display.set_caption("CADET Visualizer")
        mono = pygame.font.match_font("ubuntumono") or pygame.font.get_default_font()
        self.font = pygame.font.Font(mono, 14)
        self._rgb = None
        self._save = save_frames
        self._out = out_dir
        if save_frames:
            os.makedirs(out_dir, exist_ok=True)

    def update_frame(self, rgb) -> None:
        """Hand the visualiser the latest camera frame (HxWx3 RGB); kept for render."""
        if rgb is not None:
            self._rgb = rgb

    def render(self, throttle, steer, brake, timestamp) -> None:
        """Blit the current frame + a throttle/steer/brake overlay and flip (per tick)."""
        pg = self._pg
        if self._rgb is not None:
            try:
                surface = pg.surfarray.make_surface(self._rgb.swapaxes(0, 1))
                self.display.blit(surface, (0, 0))
            except Exception as exc:  # a bad frame must not kill the run
                self._log("[viz] surface conversion failed: {}".format(exc))
        info = [
            "Timestamp: {:.2f}".format(timestamp),
            "Throttle : {:.2f}".format(throttle),
            "Steer    : {:.2f}".format(steer),
            "Brake    : {:.2f}".format(brake),
        ]
        info_surface = pg.Surface((220, self.height // 4))
        info_surface.set_alpha(100)
        self.display.blit(info_surface, (0, 0))
        v = 10
        for line in info:
            self.display.blit(self.font.render(line, True, (255, 255, 255)), (8, v))
            v += 18
        pg.display.flip()

    def save_frame(self, rgb, frame_id) -> None:
        """Write one PNG per captured frame (as the original did), under the run dir."""
        if not self._save or rgb is None:
            return
        import cv2

        path = os.path.join(self._out, "frame_{:05d}.png".format(frame_id))
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        try:
            self._pg.quit()
        except Exception:
            pass
