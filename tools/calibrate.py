# ============================================================
#  tools/calibrate.py  —  Per-user gesture calibration wizard
# ============================================================
"""
Walks the user through each gesture one-by-one, measures their
personal landmark geometry, then writes a calibration profile to
data/user_profile.json.  main.py automatically loads this profile
to adjust thresholds.

Usage:
    python tools/calibrate.py
"""

import sys
import os
import json
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision.hand_tracker import HandTracker
from utils.math_utils import euclidean
from config import CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
PROFILE_PATH = os.path.join(DATA_DIR, "user_profile.json")

# Landmark IDs
THUMB_TIP  = 4;  THUMB_IP  = 3
INDEX_TIP  = 8;  INDEX_PIP = 6
MIDDLE_TIP = 12; RING_TIP  = 16; PINKY_TIP  = 20
WRIST      = 0

STEPS = [
    {
        "id"      : "open_hand",
        "name"    : "OPEN HAND",
        "prompt"  : "Spread all five fingers wide and hold still",
        "measure" : "hand_span",
        "color"   : (0, 255, 150),
    },
    {
        "id"      : "pinch",
        "name"    : "PINCH",
        "prompt"  : "Touch your thumb tip to your index finger tip",
        "measure" : "pinch_dist",
        "color"   : (0, 200, 255),
    },
    {
        "id"      : "index_up",
        "name"    : "INDEX FINGER UP",
        "prompt"  : "Point your index finger straight up, others curled",
        "measure" : "index_reach",
        "color"   : (255, 200, 0),
    },
    {
        "id"      : "fist",
        "name"    : "FIST",
        "prompt"  : "Close your hand into a tight fist",
        "measure" : "fist_span",
        "color"   : (0, 80, 255),
    },
]

SAMPLE_FRAMES = 60   # number of frames to average per step


def measure_frame(landmarks: list, measure: str) -> float:
    """Extract the scalar measurement relevant to this calibration step."""
    if landmarks is None:
        return 0.0

    if measure == "hand_span":
        # Wrist → middle finger tip
        return euclidean(landmarks[WRIST][:2], landmarks[MIDDLE_TIP][:2])

    elif measure == "pinch_dist":
        return euclidean(landmarks[THUMB_TIP][:2], landmarks[INDEX_TIP][:2])

    elif measure == "index_reach":
        # Index tip to wrist
        return euclidean(landmarks[WRIST][:2], landmarks[INDEX_TIP][:2])

    elif measure == "fist_span":
        # Spread of fingertips when folded (should be small)
        tips = [landmarks[t][:2] for t in
                (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)]
        xs = [p[0] for p in tips]
        ys = [p[1] for p in tips]
        return euclidean((min(xs), min(ys)), (max(xs), max(ys)))

    return 0.0


def draw_countdown(frame, step: dict, countdown: int, samples: list,
                   step_idx: int, total: int):
    h, w = frame.shape[:2]

    # Background dim
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Step counter
    cv2.putText(frame, f"Step {step_idx+1}/{total}",
                (w - 120, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (100, 100, 100), 1, cv2.LINE_AA)

    # Gesture name
    cv2.putText(frame, step["name"],
                (18, 36), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                step["color"], 2, cv2.LINE_AA)

    # Prompt
    cv2.putText(frame, step["prompt"],
                (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)

    # Progress bar
    if samples:
        pct   = len(samples) / SAMPLE_FRAMES
        bar_w = int((w - 36) * pct)
        cv2.rectangle(frame, (18, h - 36), (18 + bar_w, h - 20),
                      step["color"], -1)
        cv2.rectangle(frame, (18, h - 36), (w - 18, h - 20),
                      step["color"], 1)
        mean_val = np.mean(samples)
        cv2.putText(frame, f"Measuring … {int(pct*100)}%  (avg {mean_val:.1f}px)",
                    (18, h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (160, 160, 160), 1, cv2.LINE_AA)
    else:
        # Countdown
        cv2.putText(frame,
                    f"Starting in {countdown} …  (SPACE to skip)",
                    (18, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (140, 140, 140), 1, cv2.LINE_AA)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    cap     = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    tracker = HandTracker()
    profile = {}

    print("\n=== MNCS Gesture Calibration Wizard ===")
    print("Follow on-screen instructions for each gesture.")
    print("Press SPACE to begin each step. Q to quit.\n")

    for step_idx, step in enumerate(STEPS):
        print(f"\n[{step_idx+1}/{len(STEPS)}] {step['name']}: {step['prompt']}")

        # ── Countdown phase ──────────────────────────────────
        countdown_start = None
        ready = False

        while not ready:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            lm, _ = tracker.process(frame)
            tracker.draw_landmarks(frame, lm)

            countdown = 3
            if countdown_start:
                elapsed = time.time() - countdown_start
                countdown = max(0, 3 - int(elapsed))
                if elapsed >= 3:
                    ready = True

            draw_countdown(frame, step, countdown, [], step_idx, len(STEPS))
            cv2.imshow("MNCS Calibration", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("Calibration cancelled.")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord(" ") and countdown_start is None:
                countdown_start = time.time()

        # ── Sampling phase ───────────────────────────────────
        samples = []
        while len(samples) < SAMPLE_FRAMES:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            lm, _ = tracker.process(frame)
            tracker.draw_landmarks(frame, lm)

            val = measure_frame(lm, step["measure"])
            if val > 0:
                samples.append(val)

            draw_countdown(frame, step, 0, samples, step_idx, len(STEPS))
            cv2.imshow("MNCS Calibration", frame)
            cv2.waitKey(1)

        mean_val = float(np.mean(samples))
        std_val  = float(np.std(samples))
        profile[step["id"]] = {
            "measure": step["measure"],
            "mean"   : round(mean_val, 2),
            "std"    : round(std_val,  2),
        }
        print(f"  ✓  {step['measure']:15s}  mean={mean_val:.1f}px  std={std_val:.1f}px")

    # ── Derive config overrides from profile ─────────────────
    overrides = {}

    if "pinch" in profile:
        pinch_mean = profile["pinch"]["mean"]
        # OK / pinch threshold = pinch mean + 1.5 σ
        overrides["PINCH_THRESHOLD"] = round(
            pinch_mean + 1.5 * profile["pinch"]["std"], 1
        )

    if "open_hand" in profile and "pinch" in profile:
        span   = profile["open_hand"]["mean"]
        pinch  = profile["pinch"]["mean"]
        overrides["VOLUME_MIN_DIST"] = round(pinch * 1.2, 1)
        overrides["VOLUME_MAX_DIST"] = round(span  * 0.85, 1)

    profile["config_overrides"] = overrides

    # ── Save ─────────────────────────────────────────────────
    with open(PROFILE_PATH, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"\n✓ Profile saved → {PROFILE_PATH}")
    print("\nDerived config overrides:")
    for k, v in overrides.items():
        print(f"   {k:25s} = {v}")

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
