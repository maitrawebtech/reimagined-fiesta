# ============================================================
# MNCS — control/mouse_control.py
# Virtual mouse: converts finger coords → screen coords
# ============================================================

import pyautogui
import numpy as np
from config import (
    FRAME_WIDTH, FRAME_HEIGHT,
    SCREEN_WIDTH, SCREEN_HEIGHT,
    SMOOTHING_ALPHA,
)
from utils.math_utils import ewa_smooth, map_range

# Disable pyautogui fail-safe for smoother operation (optional)
pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0.0   # Remove built-in delay

# Auto-detect screen resolution
_SCREEN_W, _SCREEN_H = pyautogui.size()
if SCREEN_WIDTH:
    _SCREEN_W = SCREEN_WIDTH
if SCREEN_HEIGHT:
    _SCREEN_H = SCREEN_HEIGHT

# Camera region to use for mouse mapping
# Reducing this to a center ROI gives better usability
_CAM_ROI = {
    "x_min": int(FRAME_WIDTH  * 0.10),
    "x_max": int(FRAME_WIDTH  * 0.90),
    "y_min": int(FRAME_HEIGHT * 0.10),
    "y_max": int(FRAME_HEIGHT * 0.90),
}


class MouseController:
    """
    Maps index-finger tip position in camera frame to
    absolute screen coordinates with EWA smoothing.
    """

    def __init__(self):
        self._prev: tuple = (_SCREEN_W // 2, _SCREEN_H // 2)
        self._alpha: float = SMOOTHING_ALPHA

    def move(self, finger_px: tuple) -> tuple:
        """
        Move mouse to position corresponding to finger_px (camera coords).
        Returns the actual screen (x, y) used.
        """
        cx, cy = finger_px

        # Map camera ROI → screen
        sx = map_range(
            cx,
            _CAM_ROI["x_min"], _CAM_ROI["x_max"],
            0, _SCREEN_W
        )
        sy = map_range(
            cy,
            _CAM_ROI["y_min"], _CAM_ROI["y_max"],
            0, _SCREEN_H
        )

        # Clamp to screen bounds
        sx = max(0, min(_SCREEN_W - 1, sx))
        sy = max(0, min(_SCREEN_H - 1, sy))

        # EWA smoothing
        sx, sy = ewa_smooth(self._prev, (sx, sy), self._alpha)
        self._prev = (sx, sy)

        pyautogui.moveTo(int(sx), int(sy))
        return (int(sx), int(sy))

    def click(self, button: str = "left"):
        """Perform a mouse click."""
        pyautogui.click(button=button)

    def double_click(self):
        pyautogui.doubleClick()

    def right_click(self):
        pyautogui.rightClick()

    def set_alpha(self, alpha: float):
        """Adjust smoothing factor dynamically."""
        self._alpha = max(0.01, min(1.0, alpha))

    @property
    def screen_size(self) -> tuple:
        return (_SCREEN_W, _SCREEN_H)
