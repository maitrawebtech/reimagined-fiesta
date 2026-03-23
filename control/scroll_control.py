# ============================================================
# MNCS — control/scroll_control.py
# Scroll detection using two-finger gesture + wrist Y delta
# ============================================================

import time
import pyautogui
from config import SCROLL_SPEED, SCROLL_DEADZONE, FRAME_HEIGHT


class ScrollController:
    """
    Tracks wrist Y position over frames to determine
    scroll direction and magnitude when TWO_FINGERS is detected.
    """

    def __init__(self):
        self._prev_y: float | None = None
        self._last_scroll_time: float = 0.0
        self._cooldown: float = 0.12   # seconds between scroll events

    def update(self, wrist_y: int) -> int:
        """
        Call every frame while TWO_FINGERS is active.
        Returns scroll amount (positive = up, negative = down, 0 = no scroll).
        """
        now = time.time()
        norm_y = wrist_y / FRAME_HEIGHT   # normalize

        if self._prev_y is None:
            self._prev_y = norm_y
            return 0

        delta = self._prev_y - norm_y    # positive when hand moves up
        self._prev_y = norm_y

        if abs(delta) < SCROLL_DEADZONE:
            return 0

        if now - self._last_scroll_time < self._cooldown:
            return 0

        direction = 1 if delta > 0 else -1
        scroll_amount = int(direction * SCROLL_SPEED)

        pyautogui.scroll(scroll_amount)
        self._last_scroll_time = now
        return scroll_amount

    def reset(self):
        """Call when gesture changes away from TWO_FINGERS."""
        self._prev_y = None
