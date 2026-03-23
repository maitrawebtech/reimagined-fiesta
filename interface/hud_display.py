# ============================================================
# MNCS — interface/hud_display.py
# Jarvis-style HUD overlay rendered on the live camera frame
# ============================================================

import cv2
import numpy as np
import time
from config import (
    HUD_COLOR_PRIMARY, HUD_COLOR_WARNING, HUD_COLOR_DIM, HUD_ALPHA,
    FRAME_WIDTH, FRAME_HEIGHT,
)


class HUDDisplay:
    """
    Draws a futuristic overlay on the OpenCV frame including:
    • Gesture label        • Action log
    • FPS counter          • Volume bar
    • Mode indicator       • Mouse position
    • Corner brackets      • Scan line
    """

    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX
    COLOR_PRI  = HUD_COLOR_PRIMARY
    COLOR_WARN = HUD_COLOR_WARNING
    COLOR_DIM  = HUD_COLOR_DIM

    def __init__(self):
        self._fps_history: list[float] = []
        self._last_time   = time.time()
        self._action_log: list[tuple[str, float]] = []   # (message, expire_time)
        self._scan_y      = 0   # animated scan line Y

    # ── Public render call ────────────────────────────────────

    def render(
        self,
        frame:          np.ndarray,
        gesture:        str  = "NONE",
        mode:           str  = "NORMAL",
        volume_pct:     int  = 50,
        mouse_pos:      tuple = (0, 0),
        action_msg:     str  | None = None,
        hand_detected:  bool = False,
    ) -> np.ndarray:
        """
        Composite the HUD onto frame (makes a copy — non-destructive).
        """
        overlay = frame.copy()
        h, w    = frame.shape[:2]

        # Background panels (semi-transparent)
        self._draw_top_bar(overlay, w)
        self._draw_bottom_bar(overlay, w, h)
        self._draw_side_panel(overlay, h)

        # Blend overlay with original
        cv2.addWeighted(overlay, HUD_ALPHA, frame, 1 - HUD_ALPHA, 0, frame)

        # --- Non-blended HUD elements (drawn on frame directly) ---
        fps = self._calc_fps()
        self._draw_corner_brackets(frame, w, h)
        self._draw_scan_line(frame, w, h)
        self._draw_title(frame)
        self._draw_fps(frame, fps)
        self._draw_gesture_label(frame, gesture, hand_detected)
        self._draw_mode_indicator(frame, mode, w)
        self._draw_volume_bar(frame, volume_pct, w, h)
        self._draw_mouse_pos(frame, mouse_pos, h)
        self._draw_action_log(frame, action_msg, w, h)

        return frame

    # ── Internal drawing helpers ──────────────────────────────

    def _draw_top_bar(self, overlay, w):
        cv2.rectangle(overlay, (0, 0), (w, 52), (0, 0, 0), -1)

    def _draw_bottom_bar(self, overlay, w, h):
        cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)

    def _draw_side_panel(self, overlay, h):
        cv2.rectangle(overlay, (0, 52), (220, h - 50), (0, 0, 0), -1)

    def _draw_title(self, frame):
        cv2.putText(frame, "MNCS // MAITRA NEURAL CONTROL",
                    (12, 34), self.FONT_BOLD, 0.65, self.COLOR_PRI, 1, cv2.LINE_AA)

    def _draw_fps(self, frame, fps):
        label = f"FPS: {fps:.0f}"
        cv2.putText(frame, label, (12, 70), self.FONT, 0.5, self.COLOR_PRI, 1, cv2.LINE_AA)

    def _draw_gesture_label(self, frame, gesture, hand_detected):
        color = self.COLOR_PRI if hand_detected else self.COLOR_DIM
        status = "● HAND DETECTED" if hand_detected else "○ NO HAND"
        cv2.putText(frame, status, (12, 95), self.FONT, 0.42, color, 1, cv2.LINE_AA)
        cv2.putText(frame, "GESTURE:", (12, 120), self.FONT, 0.42, self.COLOR_DIM, 1, cv2.LINE_AA)
        cv2.putText(frame, gesture, (12, 148), self.FONT_BOLD, 0.75,
                    self.COLOR_PRI, 1, cv2.LINE_AA)

    def _draw_mode_indicator(self, frame, mode, w):
        colors = {
            "NORMAL":  self.COLOR_PRI,
            "DRAWING": (0, 200, 255),
            "VOICE":   (200, 100, 255),
        }
        col = colors.get(mode, self.COLOR_PRI)
        label = f"MODE: {mode}"
        tw, _ = cv2.getTextSize(label, self.FONT_BOLD, 0.55, 1)
        x = w - tw[0] - 14
        cv2.putText(frame, label, (x, 34), self.FONT_BOLD, 0.55, col, 1, cv2.LINE_AA)

    def _draw_volume_bar(self, frame, volume_pct, w, h):
        bar_x  = 12
        bar_y  = h - 40
        bar_w  = 200
        bar_h  = 12
        filled = int(bar_w * volume_pct / 100)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      self.COLOR_DIM, 1)
        if filled > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                          self.COLOR_PRI, -1)
        cv2.putText(frame, f"VOL {volume_pct:3d}%",
                    (bar_x + bar_w + 10, bar_y + 10),
                    self.FONT, 0.42, self.COLOR_PRI, 1, cv2.LINE_AA)

    def _draw_mouse_pos(self, frame, mouse_pos, h):
        label = f"CURSOR  X:{mouse_pos[0]}  Y:{mouse_pos[1]}"
        cv2.putText(frame, label, (12, h - 10), self.FONT, 0.38,
                    self.COLOR_DIM, 1, cv2.LINE_AA)

    def _draw_action_log(self, frame, action_msg, w, h):
        now = time.time()
        if action_msg:
            self._action_log.append((action_msg, now + 3.0))

        # Purge expired
        self._action_log = [(m, e) for m, e in self._action_log if e > now]

        y = h - 80
        for msg, _ in reversed(self._action_log[-3:]):
            tw, _ = cv2.getTextSize(msg, self.FONT, 0.45, 1)
            x = w - tw[0] - 14
            cv2.putText(frame, f"▶ {msg}", (x, y), self.FONT, 0.45,
                        self.COLOR_WARN, 1, cv2.LINE_AA)
            y -= 22

    def _draw_corner_brackets(self, frame, w, h):
        size   = 28
        thick  = 2
        col    = self.COLOR_PRI
        corners = [
            ((0, 0), (size, 0), (0, size)),
            ((w - 1, 0), (w - 1 - size, 0), (w - 1, size)),
            ((0, h - 1), (size, h - 1), (0, h - 1 - size)),
            ((w - 1, h - 1), (w - 1 - size, h - 1), (w - 1, h - 1 - size)),
        ]
        for corner, p1, p2 in corners:
            cv2.line(frame, corner, p1, col, thick)
            cv2.line(frame, corner, p2, col, thick)

    def _draw_scan_line(self, frame, w, h):
        """Animated horizontal scan line for futuristic feel."""
        self._scan_y = (self._scan_y + 3) % h
        alpha_mask = np.zeros_like(frame)
        cv2.line(alpha_mask, (0, self._scan_y), (w, self._scan_y),
                 (0, 60, 40), 1)
        cv2.addWeighted(frame, 1.0, alpha_mask, 0.6, 0, frame)

    def _calc_fps(self) -> float:
        now = time.time()
        dt  = now - self._last_time
        self._last_time = now
        fps = 1.0 / (dt + 1e-9)
        self._fps_history.append(fps)
        if len(self._fps_history) > 30:
            self._fps_history.pop(0)
        return sum(self._fps_history) / len(self._fps_history)
