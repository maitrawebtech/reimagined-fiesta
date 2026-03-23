# ============================================================
# MNCS — control/macros.py
# Gesture-triggered application launcher & system macros
# ============================================================

import subprocess
import sys
import time
import pyautogui
from config import MACRO_APPS
from utils.math_utils import euclidean


class MacroEngine:
    """
    Maps gesture names to system-level macro actions.
    Includes cooldown to prevent rapid repeated triggering.
    """

    def __init__(self, cooldown: float = 2.0):
        self._cooldown   = cooldown
        self._last_fired: dict[str, float] = {}

    def trigger(self, gesture_name: str) -> str | None:
        """
        Execute the macro bound to gesture_name.
        Returns action description or None if on cooldown.
        """
        now = time.time()
        if now - self._last_fired.get(gesture_name, 0) < self._cooldown:
            return None

        self._last_fired[gesture_name] = now
        action = MACRO_APPS.get(gesture_name)
        if action is None:
            return None

        if action == "__SCREENSHOT__":
            return self._take_screenshot()

        return self._launch_app(action)

    def _launch_app(self, command: str) -> str:
        """Launch an application via subprocess."""
        try:
            if sys.platform == "win32":
                subprocess.Popen(command, shell=True)
            else:
                subprocess.Popen(command.split(), stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            return f"Launched: {command}"
        except Exception as e:
            return f"Launch failed: {e}"

    def _take_screenshot(self) -> str:
        """Save screenshot with timestamp."""
        ts = int(time.time())
        path = f"screenshot_{ts}.png"
        img = pyautogui.screenshot()
        img.save(path)
        return f"Screenshot saved: {path}"

    def run_key(self, hotkey: str) -> str:
        """Execute a keyboard shortcut (e.g. 'alt+f4')."""
        try:
            pyautogui.hotkey(*hotkey.split("+"))
            return f"Hotkey: {hotkey}"
        except Exception as e:
            return f"Hotkey failed: {e}"
