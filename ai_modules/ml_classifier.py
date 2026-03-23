# ============================================================
#  ai_modules/ml_classifier.py  —  Random Forest gesture classifier
# ============================================================
"""
Two-phase module:

TRAINING  (run once, or whenever you add data)
    python ai_modules/ml_classifier.py --train

INFERENCE (used automatically by gesture_detector if model exists)
    clf = MLClassifier()
    label = clf.predict(landmarks)   # returns gesture name string

The trained model is saved as:
    data/gesture_model.joblib
    data/gesture_label_encoder.joblib

If those files don't exist, the system silently falls back to the
rule-based detector in vision/gesture_detector.py.
"""

import os
import sys
import argparse
import logging
import numpy as np

log = logging.getLogger(__name__)

# Paths
_ROOT       = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR    = os.path.join(_ROOT, "data")
MODEL_PATH  = os.path.join(DATA_DIR, "gesture_model.joblib")
ENCODER_PATH = os.path.join(DATA_DIR, "gesture_label_encoder.joblib")
DATASET_PATH = os.path.join(DATA_DIR, "gesture_dataset.csv")


# ── Feature engineering ──────────────────────────────────────

def landmarks_to_features(landmarks: list) -> np.ndarray:
    """
    Convert raw landmark list → normalised 63-dim feature vector.

    Normalisation:
      • Translate wrist to origin
      • Scale by max absolute coordinate
      • Flatten (x, y, z) × 21 = 63 features
    """
    pts   = np.array([[lm[0], lm[1], lm[2]] for lm in landmarks],
                     dtype=np.float32)
    pts  -= pts[0]                          # wrist to origin
    scale = np.abs(pts).max() + 1e-6
    pts  /= scale
    return pts.flatten()


def csv_row_to_features(row: list) -> np.ndarray:
    """Parse a raw CSV row (all strings) into a float feature vector."""
    return np.array([float(v) for v in row], dtype=np.float32)


# ── Classifier ───────────────────────────────────────────────

class MLClassifier:
    """
    Loads the pre-trained Random Forest model and provides
    real-time gesture prediction.
    """

    def __init__(self):
        self._model   = None
        self._encoder = None
        self._ready   = False
        self._load()

    def _load(self):
        if not (os.path.isfile(MODEL_PATH) and os.path.isfile(ENCODER_PATH)):
            log.info("No ML model found — falling back to rule-based detector.")
            return
        try:
            import joblib
            self._model   = joblib.load(MODEL_PATH)
            self._encoder = joblib.load(ENCODER_PATH)
            self._ready   = True
            log.info("ML gesture model loaded ✓")
        except Exception as e:
            log.warning(f"Could not load ML model: {e}")

    @property
    def is_ready(self) -> bool:
        return self._ready

    def predict(self, landmarks) -> str:
        """
        Returns the predicted gesture string, or 'UNKNOWN' on failure.
        landmarks: raw list from HandTracker (21 items of (x_px, y_px, z))
        """
        if not self._ready or landmarks is None:
            return "UNKNOWN"
        try:
            features = landmarks_to_features(landmarks).reshape(1, -1)
            idx      = self._model.predict(features)[0]
            return self._encoder.inverse_transform([idx])[0]
        except Exception as e:
            log.debug(f"ML predict error: {e}")
            return "UNKNOWN"

    def predict_proba(self, landmarks) -> dict[str, float]:
        """
        Return {gesture_name: confidence} dict for all classes.
        Useful for debugging and confidence thresholding.
        """
        if not self._ready or landmarks is None:
            return {}
        try:
            features = landmarks_to_features(landmarks).reshape(1, -1)
            proba    = self._model.predict_proba(features)[0]
            classes  = self._encoder.inverse_transform(
                range(len(self._model.classes_))
            )
            return dict(zip(classes, proba.tolist()))
        except Exception:
            return {}


# ── Training pipeline ────────────────────────────────────────

def train(dataset_path: str = DATASET_PATH, verbose: bool = True):
    """
    Train a Random Forest classifier on the recorded dataset and
    save model + encoder to the data directory.

    Returns: (accuracy_on_test_set, classification_report_string)
    """
    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        import joblib
    except ImportError as e:
        print(f"Missing dependency: {e}\nRun: pip install scikit-learn joblib pandas")
        sys.exit(1)

    if not os.path.isfile(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        print("Run tools/dataset_recorder.py first.")
        sys.exit(1)

    # ── Load data ────────────────────────────────────────────
    df = pd.read_csv(dataset_path)
    print(f"\nLoaded dataset: {len(df)} samples, "
          f"{df['label'].nunique()} classes")
    print(df['label'].value_counts().to_string())

    X = df.drop("label", axis=1).values.astype(np.float32)
    y_raw = df["label"].values

    # Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    # ── Split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Model ────────────────────────────────────────────────
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )),
    ])

    print("\nTraining Random Forest …")
    clf.fit(X_train, y_train)

    # ── Evaluate ─────────────────────────────────────────────
    y_pred   = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    report   = classification_report(
        y_test, y_pred,
        target_names=encoder.classes_
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nTest accuracy : {accuracy*100:.1f}%")
    print("\nClassification report:")
    print(report)

    if verbose:
        print("Confusion matrix:")
        print(cm)

    # 5-fold cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"\n5-fold CV accuracy: {cv_scores.mean()*100:.1f}% "
          f"± {cv_scores.std()*100:.1f}%")

    # ── Save ─────────────────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)
    joblib.dump(clf,     MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"\nModel saved   → {MODEL_PATH}")
    print(f"Encoder saved → {ENCODER_PATH}")

    return accuracy, report


# ── CLI entry point ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MNCS ML Gesture Classifier"
    )
    parser.add_argument("--train", action="store_true",
                        help="Train the model from gesture_dataset.csv")
    parser.add_argument("--dataset", default=DATASET_PATH,
                        help="Path to dataset CSV")
    args = parser.parse_args()

    if args.train:
        train(dataset_path=args.dataset)
    else:
        parser.print_help()
