# Maitra Neural Control System (MNCS)
### *"Control Your Computer Without Touching It."*

A production-grade, touchless human-computer interaction platform powered by computer vision, gesture recognition, air drawing, and voice commands.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the system
python main.py
```

---

## Architecture

```
Webcam (ID=1)
    ↓
OpenCV Frame Capture  [main.py]
    ↓
MediaPipe Hand Detection  [vision/hand_tracker.py]
    ↓
Landmark Extraction (21 points)
    ↓
Gesture Recognition Engine  [vision/gesture_detector.py]
    ↓
Command Interpreter  [main.py]
    ↙    ↓    ↘    ↘        ↘
Mouse  Scroll  Volume  Macros  Air Drawing
    ↓
HUD Overlay  [interface/hud_display.py]
    ↓
Display Output
```

---

## Gesture Reference

| Gesture       | Finger Pattern    | Action                |
|---------------|-------------------|-----------------------|
| Index Up      | `[0,1,0,0,0]`    | Move mouse cursor     |
| Two Fingers   | `[0,1,1,0,0]`    | Scroll page           |
| Fist          | `[0,0,0,0,0]`    | Pause media           |
| Open Hand     | `[1,1,1,1,1]`    | Play / Launch Chrome  |
| OK Sign       | thumb+index touch | Take screenshot       |
| Thumb Up      | `[1,0,0,0,0]`↑   | Volume increase       |
| Thumb Down    | `[1,0,0,0,0]`↓   | Volume decrease       |
| Three Fingers | `[0,1,1,1,0]`    | Launch VS Code        |

---

## Keyboard Controls (while running)

| Key  | Action                    |
|------|---------------------------|
| `Q`  | Quit                      |
| `D`  | Toggle Drawing Mode       |
| `C`  | Clear air canvas          |
| `V`  | Toggle Voice Mode display |
| `+`  | Increase cursor smoothing |
| `-`  | Decrease cursor smoothing |

---

## Voice Commands

Say these phrases aloud:

- `"open chrome"` → Launch Chrome browser
- `"open vscode"` → Launch VS Code
- `"close window"` → Alt+F4
- `"take screenshot"` → Save screenshot
- `"start drawing"` → Enter drawing mode
- `"stop drawing"` → Return to normal mode
- `"scroll up"` / `"scroll down"` → Scroll page

---

## Module Reference

```
maitra-neural-control-system/
│
├── main.py                    # Entry point & main loop
├── config.py                  # All tuneable parameters
├── requirements.txt
│
├── vision/
│   ├── hand_tracker.py        # MediaPipe Hands wrapper
│   └── gesture_detector.py    # Rule-based gesture classifier
│
├── control/
│   ├── mouse_control.py       # Virtual mouse with EWA smoothing
│   ├── scroll_control.py      # Scroll from wrist Y-delta
│   ├── volume_control.py      # PyCaw / amixer volume control
│   └── macros.py              # App launcher & screenshot engine
│
├── ai_modules/
│   └── air_drawing.py         # Canvas-based finger drawing
│
├── voice/
│   └── voice_commands.py      # Background voice listener
│
├── interface/
│   └── hud_display.py         # Jarvis-style HUD overlay
│
└── utils/
    └── math_utils.py          # EWA, normalize, euclidean, etc.
```

---

## Configuration (config.py)

Key parameters you may want to tune:

| Parameter            | Default | Description                            |
|----------------------|---------|----------------------------------------|
| `CAMERA_ID`          | `1`     | Webcam device ID                       |
| `SMOOTHING_ALPHA`    | `0.25`  | Cursor smoothing (0=max smooth, 1=raw) |
| `GESTURE_COOLDOWN`   | `0.8s`  | Delay between gesture triggers         |
| `DETECTION_CONFIDENCE` | `0.75` | MediaPipe detection threshold         |
| `VOLUME_MIN_DIST`    | `20px`  | Pinch distance = 0% volume             |
| `VOLUME_MAX_DIST`    | `200px` | Pinch distance = 100% volume           |

---

## Performance Notes

- Runs at ~25–30 FPS on a mid-range CPU with a 720p feed
- Increase `SMOOTHING_ALPHA` for more responsive (but jittery) cursor
- Set `MAX_HANDS = 2` in config.py for dual-hand support
- PyCaw (Windows volume) requires admin rights on some systems
- PyAudio requires PortAudio installed (`brew install portaudio` on macOS)

---

## Future Roadmap

- [ ] ML-based gesture classifier (Random Forest / CNN)
- [ ] Dual-hand support with role assignment
- [ ] Shape recognition from air drawings → generate HTML wireframes
- [ ] Gesture recording & custom macro binding UI
- [ ] WebSocket bridge for browser-based control
- [ ] Gaze tracking integration (MediaPipe FaceMesh)
- [ ] Low-latency mode via Numba JIT on landmark math

---

## 👨‍💻 Developer

**Ishan Maitra**  
Cloud & AI Developer of **Google Cloud**


