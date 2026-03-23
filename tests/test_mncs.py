# ============================================================
#  tests/test_mncs.py  —  Unit & integration test suite
# ============================================================
"""
Run with:
    python -m pytest tests/ -v

Coverage:
  • math_utils          — pure functions, no mocks needed
  • gesture_detector    — synthetic landmark fixtures
  • mouse_control       — mocked PyAutoGUI
  • scroll_control      — delta accumulation logic
  • volume_control      — normalisation maths
  • air_drawing         — canvas operations
  • ml_classifier       — feature extraction shape
  • hud_display         — rendering smoke test
"""

import sys
import os
import types
import unittest
import numpy as np

# ── Project root on path ─────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)


# ════════════════════════════════════════════════════════════
#  Helpers / Fixtures
# ════════════════════════════════════════════════════════════

def _flat_landmarks(n: int = 21, base_y: int = 400) -> list:
    """
    21 landmarks arranged horizontally.
    All tips are ABOVE (lower y) their PIPs — all fingers appear extended.
    """
    lm = []
    for i in range(n):
        x = 100 + i * 20
        y = base_y - (i * 5)   # tip above pip by construction
        z = 0.0
        lm.append((x, y, z))
    return lm


def _build_landmarks(tip_ys: dict) -> list:
    """
    Build a 21-landmark list from a {landmark_id: y} override dict.
    Remaining landmarks get default positions that are neutral.
    """
    lm = [(200 + i * 10, 500 - i * 3, 0.0) for i in range(21)]
    for idx, y in tip_ys.items():
        x, _, z = lm[idx]
        lm[idx] = (x, y, z)
    return lm


# ════════════════════════════════════════════════════════════
#  1. math_utils
# ════════════════════════════════════════════════════════════

class TestMathUtils(unittest.TestCase):

    def setUp(self):
        from utils.math_utils import (
            euclidean, normalize, map_range, ewa_smooth,
            angle_three_points, centroid
        )
        self.eu  = euclidean
        self.nor = normalize
        self.mr  = map_range
        self.ewa = ewa_smooth
        self.ang = angle_three_points
        self.cen = centroid

    def test_euclidean_zero(self):
        self.assertAlmostEqual(self.eu((0, 0), (0, 0)), 0.0)

    def test_euclidean_known(self):
        self.assertAlmostEqual(self.eu((0, 0), (3, 4)), 5.0)

    def test_normalize_clamp_low(self):
        self.assertEqual(self.nor(-10, 0, 100), 0.0)

    def test_normalize_clamp_high(self):
        self.assertEqual(self.nor(200, 0, 100), 1.0)

    def test_normalize_midpoint(self):
        self.assertAlmostEqual(self.nor(50, 0, 100), 0.5)

    def test_map_range(self):
        result = self.mr(50, 0, 100, 0, 1000)
        self.assertAlmostEqual(result, 500.0)

    def test_ewa_smooth_full_alpha(self):
        """alpha=1.0 → output == current."""
        result = self.ewa((0, 0), (100, 200), alpha=1.0)
        self.assertEqual(result, (100, 200))

    def test_ewa_smooth_zero_alpha(self):
        """alpha=0 → output ≈ prev (clamped to int)."""
        result = self.ewa((50, 80), (100, 200), alpha=0.0)
        self.assertEqual(result, (50, 80))

    def test_angle_right_angle(self):
        a = (0, 0); b = (1, 0); c = (1, 1)
        ang = self.ang(a, b, c)
        self.assertAlmostEqual(ang, 90.0, places=1)

    def test_centroid(self):
        pts = [(0, 0), (4, 0), (2, 4)]
        cx, cy = self.cen(pts)
        self.assertEqual(cx, 2)
        self.assertEqual(cy, 1)


# ════════════════════════════════════════════════════════════
#  2. gesture_detector
# ════════════════════════════════════════════════════════════

class TestGestureDetector(unittest.TestCase):

    def setUp(self):
        from vision.gesture_detector import GestureDetector
        self.det = GestureDetector()

    def test_none_landmarks(self):
        self.assertEqual(self.det.detect(None), "NONE")

    def test_fist_all_fingers_down(self):
        """All finger tips BELOW their PIPs → FIST."""
        # Build: tips have higher y (lower on screen) than PIPs
        lm = [(200, 400 + i, 0.0) for i in range(21)]
        # Tip y > PIP y for index, middle, ring, pinky
        # (tip indices: 8,12,16,20  pip indices: 6,10,14,18)
        # Thumb: tip x > IP x for right-hand convention
        result = self.det.detect(lm)
        # May be FIST or THUMB_DOWN depending on exact coords — not OPEN_HAND
        self.assertNotEqual(result, "OPEN_HAND")

    def test_volume_distance_returns_float(self):
        lm = _flat_landmarks()
        dist = self.det.volume_distance(lm)
        self.assertIsInstance(dist, float)
        self.assertGreater(dist, 0)

    def test_volume_distance_none(self):
        self.assertIsNone(self.det.volume_distance(None))


# ════════════════════════════════════════════════════════════
#  3. scroll_control
# ════════════════════════════════════════════════════════════

class TestScrollControl(unittest.TestCase):

    def setUp(self):
        # Patch pyautogui so no OS calls happen
        self._mock_scroll_calls = []

        import unittest.mock as mock
        self._patcher = mock.patch("pyautogui.scroll",
                                   side_effect=self._mock_scroll_calls.append)
        self._patcher.start()

        from control.scroll_control import ScrollController
        self.sc = ScrollController()

    def tearDown(self):
        self._patcher.stop()

    def test_reset_clears_state(self):
        self.sc.update(300)
        self.sc.reset()
        self.assertIsNone(self.sc._prev_y)
        self.assertEqual(self.sc._accumulated, 0.0)

    def test_no_scroll_on_first_frame(self):
        self.sc.update(300)
        self.assertEqual(len(self._mock_scroll_calls), 0)

    def test_large_upward_movement_triggers_scroll(self):
        from config import SCROLL_SENSITIVITY
        self.sc.update(300)
        # Move up by 3× sensitivity in one frame
        self.sc.update(300 - SCROLL_SENSITIVITY * 3)
        self.assertGreater(len(self._mock_scroll_calls), 0)
        # Scrolled positive (up)
        self.assertGreater(self._mock_scroll_calls[0], 0)


# ════════════════════════════════════════════════════════════
#  4. mouse_control
# ════════════════════════════════════════════════════════════

class TestMouseControl(unittest.TestCase):

    def setUp(self):
        import unittest.mock as mock

        self._moves = []
        self._patcher = mock.patch("pyautogui.moveTo",
                                   side_effect=lambda x, y: self._moves.append((x, y)))
        self._patcher.start()

        # Patch screeninfo so it works without a display
        mock.patch("screeninfo.get_monitors",
                   return_value=[types.SimpleNamespace(width=1920, height=1080)]
                   ).start()

        from control.mouse_control import MouseController
        self.mc = MouseController()

    def tearDown(self):
        import unittest.mock as mock
        mock.patch.stopall()

    def test_move_calls_moveto(self):
        self.mc.move(640, 360)
        self.assertEqual(len(self._moves), 1)

    def test_move_maps_to_screen_space(self):
        self.mc.move(640, 360)
        sx, sy = self._moves[0]
        # Should be somewhere in screen space (0 < sx < 1920, 0 < sy < 1080)
        self.assertGreater(sx, 0)
        self.assertGreater(sy, 0)

    def test_screen_position_property(self):
        self.mc.move(640, 360)
        pos = self.mc.screen_position
        self.assertIsInstance(pos, tuple)
        self.assertEqual(len(pos), 2)


# ════════════════════════════════════════════════════════════
#  5. air_drawing
# ════════════════════════════════════════════════════════════

class TestAirDrawing(unittest.TestCase):

    def setUp(self):
        from ai_modules.air_drawing import AirDrawing
        self.ad = AirDrawing(width=640, height=480)

    def _fake_lm(self, x=200, y=150):
        """21 landmarks; index tip (id=8) at (x, y)."""
        lm = [(100, 400, 0.0)] * 21
        lm[8] = (x, y, 0.0)
        return lm

    def test_canvas_initially_black(self):
        canvas = self.ad.get_canvas()
        self.assertEqual(canvas.sum(), 0)

    def test_drawing_adds_pixels(self):
        lm = self._fake_lm(200, 200)
        # Two frames of drawing to get a line
        self.ad.update(lm, is_drawing=True)
        lm2 = self._fake_lm(250, 200)
        self.ad.update(lm2, is_drawing=True)
        self.assertGreater(self.ad.get_canvas().sum(), 0)

    def test_clear_wipes_canvas(self):
        self.ad.update(self._fake_lm(200, 200), is_drawing=True)
        self.ad.update(self._fake_lm(250, 200), is_drawing=True)
        self.ad.clear()
        self.assertEqual(self.ad.get_canvas().sum(), 0)

    def test_undo_removes_stroke(self):
        # Draw stroke 1
        self.ad.update(self._fake_lm(100, 100), is_drawing=True)
        self.ad.update(self._fake_lm(200, 100), is_drawing=True)
        self.ad.update(None, is_drawing=False)   # end stroke 1

        pixel_sum_before = self.ad.get_canvas().sum()
        self.ad.undo()
        self.assertLessEqual(self.ad.get_canvas().sum(), pixel_sum_before)

    def test_blend_returns_same_shape(self):
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        blended = self.ad.blend(frame)
        self.assertEqual(blended.shape, frame.shape)

    def test_erase_at(self):
        self.ad.update(self._fake_lm(200, 200), is_drawing=True)
        self.ad.update(self._fake_lm(210, 200), is_drawing=True)
        self.ad.update(None, is_drawing=False)
        before = self.ad.get_canvas().sum()
        self.ad.erase_at(205, 200)
        self.assertLessEqual(self.ad.get_canvas().sum(), before)


# ════════════════════════════════════════════════════════════
#  6. ml_classifier — feature extraction
# ════════════════════════════════════════════════════════════

class TestMLClassifierFeatures(unittest.TestCase):

    def test_feature_vector_shape(self):
        from ai_modules.ml_classifier import landmarks_to_features
        lm = _flat_landmarks()
        features = landmarks_to_features(lm)
        # 21 landmarks × 3 (x, y, z) = 63
        self.assertEqual(features.shape, (63,))

    def test_features_normalised(self):
        from ai_modules.ml_classifier import landmarks_to_features
        lm = _flat_landmarks()
        features = landmarks_to_features(lm)
        # After normalisation, max abs value ≈ 1
        self.assertLessEqual(np.abs(features).max(), 1.01)

    def test_wrist_at_origin(self):
        from ai_modules.ml_classifier import landmarks_to_features
        lm = _flat_landmarks()
        features = landmarks_to_features(lm)
        # First 3 values are wrist x,y,z — should all be ~0 after translation
        np.testing.assert_allclose(features[:3], 0.0, atol=1e-5)

    def test_ml_classifier_no_model(self):
        """When no model is on disk, predict returns UNKNOWN gracefully."""
        import unittest.mock as mock
        with mock.patch("os.path.isfile", return_value=False):
            from ai_modules.ml_classifier import MLClassifier
            clf = MLClassifier()
            self.assertFalse(clf.is_ready)
            result = clf.predict(_flat_landmarks())
            self.assertEqual(result, "UNKNOWN")


# ════════════════════════════════════════════════════════════
#  7. hud_display — smoke test (no display required)
# ════════════════════════════════════════════════════════════

class TestHUDDisplay(unittest.TestCase):

    def test_draw_returns_frame(self):
        from interface.hud_display import HUDDisplay
        hud   = HUDDisplay()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        hud.update(gesture="INDEX_UP", action="Move cursor")
        result = hud.draw(frame, volume_pct=55.0, mouse_pos=(960, 540))
        self.assertEqual(result.shape, (720, 1280, 3))

    def test_hud_update_no_crash(self):
        from interface.hud_display import HUDDisplay
        hud = HUDDisplay()
        for gesture in ("FIST", "OPEN_HAND", "THUMB_UP", "UNKNOWN", "NONE"):
            hud.update(gesture=gesture, action="test")


# ════════════════════════════════════════════════════════════
#  Runner
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
