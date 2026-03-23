# ============================================================
# MNCS — vision/gesture_detector.py
# Rule-based gesture classification from 21 landmarks
# ============================================================

import math
from config import LM
from utils.math_utils import euclidean, angle_between


# ── Gesture names (canonical) ─────────────────────────────────
class Gesture:
    NONE          = "NONE"
    INDEX_UP      = "INDEX_UP"       # [0,1,0,0,0] → move mouse
    TWO_FINGERS   = "TWO_FINGERS"    # [0,1,1,0,0] → scroll
    FIST          = "FIST"           # [0,0,0,0,0] → pause media
    OPEN_HAND     = "OPEN_HAND"      # [1,1,1,1,1] → play / launch chrome
    OK_SIGN       = "OK_SIGN"        # thumb+index circle → screenshot
    THUMB_UP      = "THUMB_UP"       # [1,0,0,0,0] + orientation
    THUMB_DOWN    = "THUMB_DOWN"     # thumb pointing down
    PINCH         = "PINCH"          # thumb+index close → volume
    THREE_FINGERS = "THREE_FINGERS"  # [0,1,1,1,0]
    FOUR_FINGERS  = "FOUR_FINGERS"   # [0,1,1,1,1]
    # ── NEW click gestures ────────────────────────────────────
    LEFT_CLICK    = "LEFT_CLICK"     # index+middle tips pinched → left click
    RIGHT_CLICK   = "RIGHT_CLICK"    # index+pinky up [0,1,0,0,1] → right click
    DOUBLE_CLICK  = "DOUBLE_CLICK"   # all fingers briefly curled then index pops up


class GestureDetector:
    """
    Classifies gestures from landmark pixel coordinates.

    Call detect(landmarks_px) → Gesture constant string.
    """

    # ─────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _fingers_up(lm: list) -> list:
        """
        Returns [thumb, index, middle, ring, pinky] as 0/1.
        Uses tip.y < pip.y rule for index→pinky.
        Thumb uses tip.x vs ip.x (right-hand mirrored camera).
        """
        fingers = []

        # Thumb: compare x coordinates (camera is mirrored)
        if lm[LM.THUMB_TIP][0] < lm[3][0]:   # lm[3] = THUMB_IP
            fingers.append(1)
        else:
            fingers.append(0)

        # Index → Pinky: tip.y < pip.y means finger is up
        tip_ids = [LM.INDEX_TIP, LM.MIDDLE_TIP, LM.RING_TIP, LM.PINKY_TIP]
        pip_ids = [LM.INDEX_PIP, LM.MIDDLE_PIP, LM.RING_PIP, LM.PINKY_PIP]
        for tip, pip in zip(tip_ids, pip_ids):
            fingers.append(1 if lm[tip][1] < lm[pip][1] else 0)

        return fingers  # length 5

    @staticmethod
    def _thumb_pointing_down(lm: list) -> bool:
        """True if thumb tip is significantly below wrist."""
        return lm[LM.THUMB_TIP][1] > lm[LM.WRIST][1] + 40

    @staticmethod
    def _is_ok_sign(lm: list) -> bool:
        """
        OK sign: thumb tip close to index tip,
        other fingers extended.
        """
        dist = euclidean(lm[LM.THUMB_TIP], lm[LM.INDEX_TIP])
        other_up = (
            lm[LM.MIDDLE_TIP][1] < lm[LM.MIDDLE_PIP][1] and
            lm[LM.RING_TIP][1]   < lm[LM.RING_PIP][1]   and
            lm[LM.PINKY_TIP][1]  < lm[LM.PINKY_PIP][1]
        )
        return dist < 40 and other_up

    @staticmethod
    def _is_left_click(lm: list, f: list) -> bool:
        """
        LEFT CLICK gesture:
        Index AND middle fingers are up, and their tips are
        pinched close together (< 40 px apart).
        Ring and pinky are folded. Thumb doesn't matter.

        Visual: hold up index + middle, then bring them together
        like a "snap" or "clip".
        """
        if not (f[1] and f[2]):          # index + middle must be up
            return False
        if f[3] or f[4]:                 # ring + pinky must be down
            return False
        dist = euclidean(lm[LM.INDEX_TIP], lm[LM.MIDDLE_TIP])
        return dist < 40

    @staticmethod
    def _is_right_click(lm: list, f: list) -> bool:
        """
        RIGHT CLICK gesture:
        Index finger UP + Pinky finger UP, middle + ring folded.
        Pattern: [*, 1, 0, 0, 1]  (rock / call-me sign)

        Visual: extend index and pinky like devil horns / hang-loose.
        """
        return (
            f[1] == 1 and    # index up
            f[2] == 0 and    # middle down
            f[3] == 0 and    # ring down
            f[4] == 1        # pinky up
        )

    @staticmethod
    def _is_double_click(lm: list, f: list) -> bool:
        """
        DOUBLE CLICK gesture:
        Index + Middle up and spread wide apart (> 80 px).
        This is the "victory / V" variant that is clearly
        wider than the normal TWO_FINGERS scroll pose.

        Visual: make a wide peace/V sign with index+middle spread out.
        """
        if not (f[1] and f[2]):
            return False
        if f[3] or f[4]:
            return False
        dist = euclidean(lm[LM.INDEX_TIP], lm[LM.MIDDLE_TIP])
        return dist > 80   # fingers spread wide = double-click

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def detect(self, landmarks_px: list) -> str:
        """
        Classify the gesture from 21 (x,y) pixel landmarks.
        Returns a Gesture.* constant string.

        Priority order (most specific first):
          click gestures → OK sign → thumb gestures → patterns
        """
        if not landmarks_px or len(landmarks_px) < 21:
            return Gesture.NONE

        lm = landmarks_px
        f  = self._fingers_up(lm)

        # ── 1. Click gestures (highest priority) ──────────────
        # Check double-click BEFORE left-click (wider spread > narrower pinch)
        if self._is_double_click(lm, f):
            return Gesture.DOUBLE_CLICK

        if self._is_left_click(lm, f):
            return Gesture.LEFT_CLICK

        if self._is_right_click(lm, f):
            return Gesture.RIGHT_CLICK

        # ── 2. OK sign ────────────────────────────────────────
        if self._is_ok_sign(lm):
            return Gesture.OK_SIGN

        # ── 3. Thumb-only gestures ────────────────────────────
        if f == [1, 0, 0, 0, 0]:
            if self._thumb_pointing_down(lm):
                return Gesture.THUMB_DOWN
            return Gesture.THUMB_UP

        # ── 4. Standard finger patterns ───────────────────────
        pattern_map = {
            (0, 1, 0, 0, 0): Gesture.INDEX_UP,
            (0, 1, 1, 0, 0): Gesture.TWO_FINGERS,
            (0, 1, 1, 1, 0): Gesture.THREE_FINGERS,
            (0, 1, 1, 1, 1): Gesture.FOUR_FINGERS,
            (1, 1, 1, 1, 1): Gesture.OPEN_HAND,
            (0, 0, 0, 0, 0): Gesture.FIST,
        }

        return pattern_map.get(tuple(f), Gesture.NONE)

    # ── Landmark accessors ────────────────────────────────────

    def pinch_distance(self, landmarks_px: list) -> float:
        """Pixel distance between thumb tip and index tip (volume control)."""
        if not landmarks_px or len(landmarks_px) < 21:
            return 0.0
        return euclidean(landmarks_px[LM.THUMB_TIP], landmarks_px[LM.INDEX_TIP])

    def index_tip(self, landmarks_px: list) -> tuple:
        """Returns (x, y) of index finger tip."""
        return landmarks_px[LM.INDEX_TIP]

    def middle_tip(self, landmarks_px: list) -> tuple:
        """Returns (x, y) of middle finger tip."""
        return landmarks_px[LM.MIDDLE_TIP]

    def wrist(self, landmarks_px: list) -> tuple:
        """Returns (x, y) of wrist."""
        return landmarks_px[LM.WRIST]

    def index_middle_midpoint(self, landmarks_px: list) -> tuple:
        """
        Midpoint between index and middle tips.
        Used to position the cursor during click gestures so the
        click lands where the user is pointing.
        """
        ix, iy = landmarks_px[LM.INDEX_TIP][:2]
        mx, my = landmarks_px[LM.MIDDLE_TIP][:2]
        return ((ix + mx) // 2, (iy + my) // 2)
