# ============================================================
# MNCS — utils/math_utils.py
# Math helpers used across all modules
# ============================================================

import math
import numpy as np


def angle_three_points(a, b, c) -> float:
    """Interior angle at vertex *b* formed by a-b-c (degrees)."""
    import numpy as np
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cosang, -1.0, 1.0)))


def centroid(points: list) -> tuple:
    """Return (x, y) centroid of a list of (x, y) points."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))


def euclidean(p1, p2) -> float:
    """Euclidean distance between two (x, y) points."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Clamp-normalize value to [0.0, 1.0]."""
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val + 1e-9)))


def map_range(value: float, in_min: float, in_max: float,
              out_min: float, out_max: float) -> float:
    """Linear map from one range to another."""
    return out_min + (value - in_min) / (in_max - in_min + 1e-9) * (out_max - out_min)


def ewa_smooth(prev: tuple, curr: tuple, alpha: float) -> tuple:
    """
    Exponential Weighted Average smoothing for (x, y) coordinates.
    alpha ∈ (0,1]: higher = more responsive, lower = smoother.
    """
    sx = alpha * curr[0] + (1 - alpha) * prev[0]
    sy = alpha * curr[1] + (1 - alpha) * prev[1]
    return (int(sx), int(sy))


def landmark_to_px(landmark, frame_w: int, frame_h: int) -> tuple:
    """Convert a MediaPipe normalized landmark to pixel coordinates."""
    return (int(landmark.x * frame_w), int(landmark.y * frame_h))


def midpoint(p1: tuple, p2: tuple) -> tuple:
    """Return integer midpoint of two (x, y) points."""
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)


def angle_between(a: tuple, b: tuple, c: tuple) -> float:
    """
    Angle (degrees) at point B formed by A-B-C.
    Useful for bend-angle detection in fingers.
    """
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))
