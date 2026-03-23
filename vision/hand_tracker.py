# ============================================================
# MNCS — vision/hand_tracker.py
# MediaPipe Hands wrapper: detects landmarks & handedness
# ============================================================

import cv2
import mediapipe as mp
import numpy as np
from config import (
    MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE
)
from utils.math_utils import landmark_to_px


class HandTracker:
    """
    Wraps MediaPipe Hands to provide per-frame landmark data.

    Usage:
        tracker = HandTracker()
        while True:
            ret, frame = cap.read()
            result = tracker.process(frame)
            if result:
                lm_list = result["landmarks_px"]
    """

    def __init__(self):
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils
        self._mp_style  = mp.solutions.drawing_styles

        self.hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
        )

    def process(self, frame: np.ndarray) -> dict | None:
        """
        Process a BGR frame.

        Returns dict with:
            landmarks_px  — list of 21 (x, y) pixel tuples
            landmarks_norm— list of 21 mediapipe landmark objects (normalized)
            handedness    — "Left" or "Right"
            raw_results   — full mediapipe result object
        Or None if no hand detected.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return None

        # Use first hand only (MAX_HANDS=1 by default)
        hand_lm   = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0].classification[0].label  # "Left"/"Right"

        landmarks_px   = [landmark_to_px(lm, w, h) for lm in hand_lm.landmark]
        landmarks_norm = list(hand_lm.landmark)

        return {
            "landmarks_px":   landmarks_px,
            "landmarks_norm": landmarks_norm,
            "handedness":     handedness,
            "raw_results":    results,
            "hand_lm_obj":    hand_lm,
        }

    def draw_landmarks(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Draw MediaPipe skeleton on frame (in-place)."""
        if result is None:
            return frame
        self._mp_draw.draw_landmarks(
            frame,
            result["hand_lm_obj"],
            self._mp_hands.HAND_CONNECTIONS,
            self._mp_style.get_default_hand_landmarks_style(),
            self._mp_style.get_default_hand_connections_style(),
        )
        return frame

    def get_bounding_box(self, landmarks_px: list, frame_shape: tuple) -> tuple:
        """
        Returns (x, y, w, h) bounding box around all landmarks.
        """
        xs = [p[0] for p in landmarks_px]
        ys = [p[1] for p in landmarks_px]
        pad = 20
        h_f, w_f = frame_shape[:2]
        x1 = max(0, min(xs) - pad)
        y1 = max(0, min(ys) - pad)
        x2 = min(w_f, max(xs) + pad)
        y2 = min(h_f, max(ys) + pad)
        return (x1, y1, x2 - x1, y2 - y1)

    def close(self):
        self.hands.close()
