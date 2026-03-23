# ============================================================
#  tools/dataset_recorder.py  —  Gesture dataset collector
# ============================================================
"""
Interactive tool for recording hand-landmark training data.

Usage:
    python tools/dataset_recorder.py

Workflow:
  1. Press a number key 0-8 to select gesture class
  2. Hold the gesture in front of the camera
  3. Press SPACE to start recording (500 samples)
  4. Recordings auto-save to data/gesture_dataset.csv
  5. Press 'q' to quit

CSV schema:
    label, lm0_x, lm0_y, lm0_z, lm1_x, ..., lm20_z  (63 features + 1 label)
"""

import sys
import os
import csv
import time
import cv2
import numpy as np

# Make sure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision.hand_tracker import HandTracker
from config import CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT

# ── Constants ────────────────────────────────────────────────
DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")
DATASET_PATH = os.path.join(DATA_DIR, "gesture_dataset.csv")
SAMPLES_PER_SESSION = 500   # samples captured per recording burst

GESTURE_CLASSES = {
    0: "INDEX_UP",
    1: "TWO_FINGERS",
    2: "FIST",
    3: "OPEN_HAND",
    4: "OK_SIGN",
    5: "THUMB_UP",
    6: "THUMB_DOWN",
    7: "PINCH",
    8: "UNKNOWN",
}

# CSV header
_HEADER = (
    ["label"]
    + [f"lm{i}_{ax}" for i in range(21) for ax in ("x", "y", "z")]
)


# ── Normalisation helper ─────────────────────────────────────

def normalise_landmarks(landmarks: list) -> list[float]:
    """
    Translate landmarks so the wrist is at the origin, then scale
    so the max absolute coordinate = 1.0.
    This makes features pose-invariant.
    """
    wrist = np.array(landmarks[0], dtype=float)
    pts   = np.array(landmarks, dtype=float)
    pts  -= wrist                          # translate
    scale = np.abs(pts).max() + 1e-6
    pts  /= scale                          # scale
    return pts.flatten().tolist()


# ── Main ─────────────────────────────────────────────────────

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Write header if file is new
    file_exists = os.path.isfile(DATASET_PATH)
    csvfile = open(DATASET_PATH, "a", newline="")
    writer  = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(_HEADER)

    cap     = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    tracker = HandTracker()

    current_class  = 0
    recording      = False
    record_count   = 0
    total_recorded = 0

    print("\n=== MNCS Dataset Recorder ===")
    print("Keys 0-8 → select gesture class")
    print("SPACE    → start/stop recording")
    print("q / ESC  → quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        landmarks, _ = tracker.process(frame)
        tracker.draw_landmarks(frame, landmarks)

        # ── Record sample ────────────────────────────────────
        if recording and landmarks:
            row = [GESTURE_CLASSES[current_class]] + normalise_landmarks(landmarks)
            writer.writerow(row)
            csvfile.flush()
            record_count  += 1
            total_recorded += 1

            if record_count >= SAMPLES_PER_SESSION:
                recording    = False
                record_count = 0
                print(f"  ✓ Session complete. Total samples: {total_recorded}")

        # ── HUD ──────────────────────────────────────────────
        h, w = frame.shape[:2]

        # Class selector
        for idx, name in GESTURE_CLASSES.items():
            y = 30 + idx * 26
            col = (0, 255, 150) if idx == current_class else (80, 80, 80)
            cv2.putText(frame, f"[{idx}] {name}", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

        # Recording indicator
        if recording:
            pct  = int(record_count / SAMPLES_PER_SESSION * 100)
            bar_w = int((w - 40) * pct / 100)
            cv2.rectangle(frame, (20, h - 40), (20 + bar_w, h - 20),
                          (0, 0, 220), -1)
            cv2.rectangle(frame, (20, h - 40), (w - 20, h - 20),
                          (0, 0, 220), 1)
            cv2.putText(frame, f"● REC  {pct}%  [{record_count}/{SAMPLES_PER_SESSION}]",
                        (25, h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (0, 80, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame,
                        f"READY  |  class: {GESTURE_CLASSES[current_class]}"
                        f"  |  total: {total_recorded}",
                        (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (160, 160, 160), 1, cv2.LINE_AA)

        cv2.imshow("MNCS Dataset Recorder", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            if not recording:
                recording    = True
                record_count = 0
                print(f"  ▶ Recording class [{current_class}] "
                      f"{GESTURE_CLASSES[current_class]} …")
            else:
                recording = False
                print(f"  ■ Stopped. Session samples: {record_count}")
        elif ord("0") <= key <= ord("8"):
            current_class = key - ord("0")
            recording     = False
            record_count  = 0
            print(f"  → Class: {GESTURE_CLASSES[current_class]}")

    csvfile.close()
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDataset saved → {DATASET_PATH}  ({total_recorded} total samples)")


if __name__ == "__main__":
    main()
