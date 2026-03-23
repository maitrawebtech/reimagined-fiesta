# ============================================================
# MAITRA NEURAL CONTROL SYSTEM (MNCS)
# config.py — Global Configuration
# ============================================================

# --- Camera ---
CAMERA_ID = 1
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30

# --- MediaPipe ---
MAX_HANDS = 1
DETECTION_CONFIDENCE = 0.75
TRACKING_CONFIDENCE = 0.75

# --- Screen ---
SCREEN_WIDTH = None   # None = auto-detect
SCREEN_HEIGHT = None  # None = auto-detect

# --- Cursor smoothing (EWA) ---
SMOOTHING_ALPHA = 0.25

# --- Gesture cooldown (seconds) ---
GESTURE_COOLDOWN = 0.8

# --- Scroll ---
SCROLL_SPEED = 30
SCROLL_DEADZONE = 0.03

# --- Volume ---
VOLUME_MIN_DIST = 20
VOLUME_MAX_DIST = 200

# --- Air Drawing ---
DRAWING_COLOR = (0, 255, 180)
DRAWING_THICKNESS = 4
DRAWING_HISTORY_LIMIT = 2000

# --- HUD ---
HUD_COLOR_PRIMARY = (0, 255, 180)
HUD_COLOR_WARNING = (0, 120, 255)
HUD_COLOR_DIM     = (100, 100, 100)
HUD_ALPHA         = 0.55

# --- Gesture Macros ---
import sys

MACRO_APPS = {
    "OPEN_HAND":  "start chrome" if sys.platform == "win32" else "google-chrome",
    "TWO_FINGERS": "code",
    "OK_SIGN":    "__SCREENSHOT__",
}

# --- Voice Commands ---
VOICE_COMMANDS = {
    "open chrome":     ("macro", "OPEN_HAND"),
    "open vscode":     ("macro", "TWO_FINGERS"),
    "close window":    ("key",   "alt+f4"),
    "take screenshot": ("macro", "OK_SIGN"),
    "start drawing":   ("mode",  "DRAW"),
    "stop drawing":    ("mode",  "NORMAL"),
    "scroll up":       ("scroll", 1),
    "scroll down":     ("scroll", -1),
}

# --- Modes ---
class Mode:
    NORMAL  = "NORMAL"
    DRAWING = "DRAWING"
    VOICE   = "VOICE"

# --- Landmark indices ---
class LM:
    WRIST      = 0
    THUMB_TIP  = 4
    INDEX_MCP  = 5
    INDEX_PIP  = 6
    INDEX_DIP  = 7
    INDEX_TIP  = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_TIP = 12
    RING_MCP   = 13
    RING_PIP   = 14
    RING_TIP   = 16
    PINKY_MCP  = 17
    PINKY_PIP  = 18
    PINKY_TIP  = 20
