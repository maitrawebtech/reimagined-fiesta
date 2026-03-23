# ============================================================
# MNCS — ai_modules/air_drawing.py
# Air drawing: traces index finger path on a transparent canvas
# ============================================================

import cv2
import numpy as np
from config import (
    DRAWING_COLOR, DRAWING_THICKNESS, DRAWING_HISTORY_LIMIT,
    FRAME_WIDTH, FRAME_HEIGHT,
)


class AirDrawing:
    """
    Maintains a transparent drawing canvas overlaid on the live frame.
    Points are appended from the index finger tip position.
    """

    def __init__(self, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT):
        self._canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self._points: list[tuple | None] = []   # None = pen-up marker
        self._color     = DRAWING_COLOR
        self._thickness = DRAWING_THICKNESS
        self._active    = False
        self._width  = width
        self._height = height

    # ── Drawing lifecycle ─────────────────────────────────────

    def start(self):
        """Begin drawing (pen-down)."""
        self._active = True

    def stop(self):
        """Lift pen — inserts a gap marker."""
        if self._active:
            self._points.append(None)
        self._active = False

    def add_point(self, point: tuple):
        """
        Add an (x, y) coordinate to the drawing path.
        Only records when active.
        """
        if not self._active:
            return
        self._points.append(point)
        if len(self._points) > DRAWING_HISTORY_LIMIT:
            self._points.pop(0)
        self._redraw_canvas()

    def clear(self):
        """Erase all drawings."""
        self._canvas = np.zeros_like(self._canvas)
        self._points.clear()

    # ── Rendering ─────────────────────────────────────────────

    def _redraw_canvas(self):
        """Redraw all strokes onto the internal canvas."""
        self._canvas = np.zeros_like(self._canvas)
        prev = None
        for pt in self._points:
            if pt is None:
                prev = None
                continue
            if prev is not None:
                cv2.line(self._canvas, prev, pt, self._color, self._thickness)
                # Glow effect: thinner brighter center line
                cv2.line(self._canvas, prev, pt, (255, 255, 255),
                         max(1, self._thickness // 3))
            prev = pt

    def composite(self, frame: np.ndarray) -> np.ndarray:
        """
        Overlay the drawing canvas onto a live frame.
        Returns composited frame.
        """
        # Blend: where canvas has drawn pixels, overlay them
        mask = cv2.cvtColor(self._canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(mask)

        bg  = cv2.bitwise_and(frame, frame, mask=inv_mask)
        fg  = cv2.bitwise_and(self._canvas, self._canvas, mask=mask)
        return cv2.add(bg, fg)

    def set_color(self, color: tuple):
        """Change stroke color (BGR)."""
        self._color = color

    def set_thickness(self, thickness: int):
        self._thickness = max(1, thickness)

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def point_count(self) -> int:
        return len([p for p in self._points if p is not None])
