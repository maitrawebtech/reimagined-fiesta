# ============================================================
# MNCS — control/volume_control.py
# System volume via PyCaw (Windows) or fallbacks
# ============================================================

import sys
import subprocess
import logging
from config import VOLUME_MIN_DIST, VOLUME_MAX_DIST
from utils.math_utils import normalize

log = logging.getLogger(__name__)

_USE_PYCAW   = False
_pycaw_iface = None   # module-level so we initialise once

if sys.platform == "win32":
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        # ── Try the classic PyCaw path ───────────────────────
        try:
            _devices = AudioUtilities.GetSpeakers()
            _iface   = _devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            _pycaw_iface = cast(_iface, POINTER(IAudioEndpointVolume))
            _USE_PYCAW = True
            log.info("PyCaw initialised via GetSpeakers().Activate()")

        except (AttributeError, Exception) as e:
            log.warning(f"PyCaw classic init failed ({e}), trying fallback …")
            # Newer pycaw / comtypes builds surface the interface differently
            try:
                _pycaw_iface = AudioUtilities.GetMasterVolumeInterface()
                _USE_PYCAW = True
                log.info("PyCaw initialised via GetMasterVolumeInterface()")
            except Exception as e2:
                log.warning(f"PyCaw fallback also failed: {e2}. Volume control disabled.")

    except ImportError as e:
        log.warning(f"PyCaw not installed: {e}. Volume control disabled.")


class VolumeController:
    """
    Controls system master volume from pinch finger distance.
    Windows: PyCaw (two init strategies + graceful disable on failure)
    Linux:   amixer
    macOS:   osascript
    """

    def __init__(self):
        self._volume_pct: float = 0.5     # 0.0 – 1.0
        self._interface = _pycaw_iface    # may be None if PyCaw failed

        # Sync initial volume from OS
        if _USE_PYCAW and self._interface:
            try:
                self._volume_pct = self._interface.GetMasterVolumeLevelScalar()
            except Exception:
                pass

    # ── Public API ───────────────────────────────────────────

    def set_from_distance(self, distance_px: float) -> float:
        """Map pinch pixel distance to volume level 0–1. Returns new level."""
        vol = normalize(distance_px, VOLUME_MIN_DIST, VOLUME_MAX_DIST)
        self._apply(vol)
        return vol

    def increase(self, step: float = 0.05):
        self._apply(self._volume_pct + step)

    def decrease(self, step: float = 0.05):
        self._apply(self._volume_pct - step)

    def step(self, delta_pct: float):
        """Change volume by delta_pct points (e.g. +5 or -5)."""
        self._apply(self._volume_pct + delta_pct / 100.0)

    def get_volume_pct(self) -> float:
        """Return current volume as 0–100 float (for HUD display)."""
        if _USE_PYCAW and self._interface:
            try:
                self._volume_pct = self._interface.GetMasterVolumeLevelScalar()
            except Exception:
                pass
        return self._volume_pct * 100.0

    @property
    def current_volume(self) -> float:
        return self._volume_pct

    @property
    def current_volume_pct(self) -> int:
        return int(self._volume_pct * 100)

    # ── Internal ─────────────────────────────────────────────

    def _apply(self, level: float):
        level = max(0.0, min(1.0, level))
        self._volume_pct = level

        if _USE_PYCAW and self._interface:
            try:
                self._interface.SetMasterVolumeLevelScalar(level, None)
                return
            except Exception as e:
                log.error(f"PyCaw SetMasterVolumeLevelScalar failed: {e}")

        # Platform fallbacks (used when PyCaw unavailable)
        pct = int(level * 100)
        if sys.platform == "linux":
            subprocess.run(["amixer", "-q", "set", "Master", f"{pct}%"],
                           check=False)
        elif sys.platform == "darwin":
            subprocess.run(
                ["osascript", "-e", f"set volume output volume {pct}"],
                check=False
            )
        # Windows without PyCaw: silently no-op (no crash)
