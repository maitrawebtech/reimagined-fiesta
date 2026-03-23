# ============================================================
# MAITRA NEURAL CONTROL SYSTEM (MNCS)
# main.py — Entry Point & Main Control Loop
#
# "Control Your Computer Without Touching It."
# ============================================================

import cv2
import time
import sys

# ── Config ────────────────────────────────────────────────────
from config import (
    CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS,
    GESTURE_COOLDOWN, Mode,
)

# ── Vision ────────────────────────────────────────────────────
from vision.hand_tracker     import HandTracker
from vision.gesture_detector import GestureDetector, Gesture

# ── Control ───────────────────────────────────────────────────
from control.mouse_control   import MouseController
from control.scroll_control  import ScrollController
from control.volume_control  import VolumeController
from control.macros          import MacroEngine

# ── AI Modules ────────────────────────────────────────────────
from ai_modules.air_drawing  import AirDrawing

# ── Voice ─────────────────────────────────────────────────────
from voice.voice_commands    import VoiceCommandListener

# ── Interface ─────────────────────────────────────────────────
from interface.hud_display   import HUDDisplay


# ─────────────────────────────────────────────────────────────
# CooldownTracker — per-gesture independent cooldowns
# ─────────────────────────────────────────────────────────────

class CooldownTracker:
    def __init__(self, cooldown: float = GESTURE_COOLDOWN):
        self._cd       = cooldown
        self._last: dict[str, float] = {}

    def ready(self, gesture: str, override_cd: float = None) -> bool:
        now = time.time()
        cd  = override_cd if override_cd is not None else self._cd
        if now - self._last.get(gesture, 0) >= cd:
            self._last[gesture] = now
            return True
        return False

    def reset(self, gesture: str):
        self._last[gesture] = 0.0


# ─────────────────────────────────────────────────────────────
# Click Flash — shows a visual burst on the frame at click point
# ─────────────────────────────────────────────────────────────

class ClickFlash:
    """Draws a brief animated ring at the click position."""

    def __init__(self):
        self._active    = False
        self._pos       = (0, 0)
        self._color     = (255, 255, 255)
        self._start     = 0.0
        self._duration  = 0.25   # seconds

    def trigger(self, pos: tuple, color: tuple = (0, 255, 180)):
        self._active   = True
        self._pos      = pos
        self._color    = color
        self._start    = time.time()

    def draw(self, frame):
        if not self._active:
            return
        elapsed = time.time() - self._start
        if elapsed > self._duration:
            self._active = False
            return

        # Expanding ring that fades out
        progress = elapsed / self._duration        # 0 → 1
        radius   = int(20 + progress * 35)         # grows 20 → 55
        alpha    = int(255 * (1.0 - progress))     # fades out
        # Draw with dimmed color to simulate transparency
        c = tuple(int(ch * (1.0 - progress * 0.7)) for ch in self._color)
        cv2.circle(frame, self._pos, radius, c, 2, cv2.LINE_AA)
        cv2.circle(frame, self._pos, 6, c, -1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
# DwellClicker — auto left-click when cursor stays still
# ─────────────────────────────────────────────────────────────

class DwellClicker:
    """
    Triggers a left click when the INDEX_UP cursor hasn't moved
    more than DWELL_RADIUS pixels for DWELL_TIME seconds.
    """
    DWELL_TIME   = 1.5   # seconds to hold still
    DWELL_RADIUS = 12    # pixel movement tolerance

    def __init__(self):
        self._anchor    = None
        self._start     = None
        self._fired     = False

    def update(self, screen_pos: tuple) -> bool:
        """
        Call every frame while INDEX_UP is active.
        Returns True exactly once when the dwell threshold is hit.
        """
        if self._anchor is None:
            self._anchor = screen_pos
            self._start  = time.time()
            self._fired  = False
            return False

        dist = ((screen_pos[0] - self._anchor[0]) ** 2 +
                (screen_pos[1] - self._anchor[1]) ** 2) ** 0.5

        if dist > self.DWELL_RADIUS:
            # Moved — reset timer
            self._anchor = screen_pos
            self._start  = time.time()
            self._fired  = False
            return False

        if not self._fired and (time.time() - self._start) >= self.DWELL_TIME:
            self._fired = True
            return True   # fire the click

        return False

    def reset(self):
        self._anchor = None
        self._start  = None
        self._fired  = False

    def progress(self) -> float:
        """0.0 → 1.0 fill for the dwell progress ring (for HUD)."""
        if self._anchor is None or self._start is None:
            return 0.0
        return min(1.0, (time.time() - self._start) / self.DWELL_TIME)


# ─────────────────────────────────────────────────────────────
# MNCS Main Application
# ─────────────────────────────────────────────────────────────

def draw_dwell_ring(frame, cam_tip: tuple, progress: float):
    """Draw a progress arc around the cursor tip for dwell-click feedback."""
    if progress <= 0:
        return
    cx, cy   = cam_tip
    radius   = 22
    angle    = int(360 * progress)
    color    = (0, int(200 * progress), int(255 * (1 - progress)))
    # Background circle
    cv2.circle(frame, (cx, cy), radius, (60, 60, 60), 2, cv2.LINE_AA)
    # Progress arc (approximated with ellipse)
    cv2.ellipse(frame, (cx, cy), (radius, radius),
                -90, 0, angle, color, 3, cv2.LINE_AA)


def main():
    # ── Initialise camera ─────────────────────────────────────
    print(f"[MNCS] Opening camera ID {CAMERA_ID}...")
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print(f"[MNCS] ERROR: Could not open camera {CAMERA_ID}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

    # ── Initialise modules ────────────────────────────────────
    tracker  = HandTracker()
    detector = GestureDetector()
    mouse    = MouseController()
    scroller = ScrollController()
    volume   = VolumeController()
    macros   = MacroEngine(cooldown=2.0)
    drawing  = AirDrawing(FRAME_WIDTH, FRAME_HEIGHT)
    voice    = VoiceCommandListener()
    hud      = HUDDisplay()
    cooldown = CooldownTracker()
    flash    = ClickFlash()
    dwell    = DwellClicker()

    # ── System state ──────────────────────────────────────────
    mode         = Mode.NORMAL
    current_gest = Gesture.NONE
    mouse_pos    = (0, 0)
    action_msg   = None
    prev_gesture = Gesture.NONE
    dwell_mode   = False    # toggled with 'w' key

    voice.start()

    print("[MNCS] System running. Press Q to quit.")
    print("[MNCS] Gestures:")
    print("         INDEX_UP      → Move cursor")
    print("         PINCH fingers → Left click  (index+middle tips together)")
    print("         DEVIL HORNS   → Right click (index+pinky up)")
    print("         V WIDE SPREAD → Double click")
    print("         TWO_FINGERS   → Scroll")
    print("         FIST          → Pause media")
    print("         OPEN_HAND     → Play / Chrome")
    print("         OK_SIGN       → Screenshot")
    print("         THUMB UP/DOWN → Volume +/-")
    print("[MNCS] Keys: D=draw  C=clear  V=voice  W=dwell-click  +/-=smoothing  Q=quit")

    # ─────────────────────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[MNCS] Frame capture failed.")
            break

        frame = cv2.flip(frame, 1)
        action_msg = None

        # ── Hand detection ────────────────────────────────────
        result        = tracker.process(frame)
        hand_detected = result is not None

        if hand_detected:
            tracker.draw_landmarks(frame, result)
            lm = result["landmarks_px"]

            # ── Gesture recognition ───────────────────────────
            current_gest = detector.detect(lm)

            if current_gest != prev_gesture:
                scroller.reset()
                # Reset dwell when gesture changes
                if current_gest != Gesture.INDEX_UP:
                    dwell.reset()

            prev_gesture = current_gest

            # ── Command dispatch ──────────────────────────────

            if mode == Mode.DRAWING:
                if current_gest == Gesture.INDEX_UP:
                    drawing.start()
                    tip = detector.index_tip(lm)
                    drawing.add_point(tip)
                else:
                    drawing.stop()
                if current_gest == Gesture.FIST and cooldown.ready("CLEAR"):
                    drawing.clear()
                    action_msg = "Canvas cleared"

            elif mode == Mode.NORMAL:

                # ── INDEX UP: Move cursor + optional dwell click ──
                if current_gest == Gesture.INDEX_UP:
                    tip       = detector.index_tip(lm)
                    mouse_pos = mouse.move(tip)
                    cv2.circle(frame, tip, 8, (0, 255, 180), 2)

                    if dwell_mode:
                        fired = dwell.update(mouse_pos)
                        prog  = dwell.progress()
                        draw_dwell_ring(frame, tip, prog)
                        if fired:
                            mouse.click("left")
                            flash.trigger(tip, color=(0, 255, 180))
                            action_msg = "🖱  Dwell Click"
                    else:
                        dwell.reset()

                # ── LEFT CLICK: index + middle pinched together ────
                elif current_gest == Gesture.LEFT_CLICK:
                    # Move cursor to midpoint of the two fingers first
                    mid = detector.index_middle_midpoint(lm)
                    mouse_pos = mouse.move(mid)
                    # Draw pinch indicator
                    ix, iy = detector.index_tip(lm)[:2]
                    mx, my = detector.middle_tip(lm)[:2]
                    cv2.line(frame, (ix, iy), (mx, my), (0, 255, 180), 2)
                    cv2.circle(frame, mid, 10, (0, 255, 180), -1)

                    if cooldown.ready("LEFT_CLICK", override_cd=0.5):
                        mouse.click("left")
                        flash.trigger(mid, color=(0, 255, 180))
                        action_msg = "🖱  Left Click"

                # ── RIGHT CLICK: index + pinky (devil horns) ──────
                elif current_gest == Gesture.RIGHT_CLICK:
                    tip = detector.index_tip(lm)
                    mouse_pos = mouse.move(tip)
                    # Draw indicator: red circle + "R"
                    cv2.circle(frame, tip, 12, (0, 80, 255), 2)
                    cv2.putText(frame, "R", (tip[0] + 14, tip[1] - 4),
                                cv2.FONT_HERSHEY_DUPLEX, 0.6,
                                (0, 80, 255), 1, cv2.LINE_AA)

                    if cooldown.ready("RIGHT_CLICK", override_cd=0.6):
                        mouse.right_click()
                        flash.trigger(tip, color=(0, 80, 255))
                        action_msg = "🖱  Right Click"

                # ── DOUBLE CLICK: wide V / peace sign ─────────────
                elif current_gest == Gesture.DOUBLE_CLICK:
                    mid = detector.index_middle_midpoint(lm)
                    mouse_pos = mouse.move(mid)
                    # Draw spread indicator
                    ix, iy = detector.index_tip(lm)[:2]
                    mx, my = detector.middle_tip(lm)[:2]
                    cv2.line(frame, (ix, iy), (mx, my), (0, 200, 255), 2)
                    cv2.circle(frame, mid, 10, (0, 200, 255), -1)
                    cv2.putText(frame, "2x", (mid[0] + 12, mid[1] - 4),
                                cv2.FONT_HERSHEY_DUPLEX, 0.55,
                                (0, 200, 255), 1, cv2.LINE_AA)

                    if cooldown.ready("DOUBLE_CLICK", override_cd=0.7):
                        mouse.double_click()
                        flash.trigger(mid, color=(0, 200, 255))
                        action_msg = "🖱  Double Click"

                # ── TWO FINGERS: Scroll ────────────────────────────
                elif current_gest == Gesture.TWO_FINGERS:
                    wrist = detector.wrist(lm)
                    amt   = scroller.update(wrist[1])
                    if amt != 0:
                        action_msg = f"Scroll {'↑ UP' if amt > 0 else '↓ DOWN'}"

                # ── FIST: Pause ────────────────────────────────────
                elif current_gest == Gesture.FIST:
                    if cooldown.ready("FIST"):
                        import pyautogui; pyautogui.press("space")
                        action_msg = "⏸  Pause"

                # ── OPEN HAND: Play / Chrome ───────────────────────
                elif current_gest == Gesture.OPEN_HAND:
                    if cooldown.ready("OPEN_HAND"):
                        msg = macros.trigger("OPEN_HAND")
                        action_msg = msg or "▶  Play"

                # ── OK SIGN: Screenshot ────────────────────────────
                elif current_gest == Gesture.OK_SIGN:
                    if cooldown.ready("OK_SIGN"):
                        msg = macros.trigger("OK_SIGN")
                        action_msg = msg or "📷  Screenshot"

                # ── THUMB UP / DOWN: Volume ────────────────────────
                elif current_gest == Gesture.THUMB_UP:
                    if cooldown.ready("THUMB_UP"):
                        volume.increase(0.05)
                        action_msg = f"🔊  Volume ↑ {volume.current_volume_pct}%"

                elif current_gest == Gesture.THUMB_DOWN:
                    if cooldown.ready("THUMB_DOWN"):
                        volume.decrease(0.05)
                        action_msg = f"🔉  Volume ↓ {volume.current_volume_pct}%"

                # ── THREE FINGERS: Launch VS Code ──────────────────
                elif current_gest == Gesture.THREE_FINGERS:
                    if cooldown.ready("THREE_FINGERS"):
                        msg = macros.trigger("TWO_FINGERS")
                        action_msg = msg or "App macro"

            # ── Pinch distance → volume (continuous, always active) ─
            pinch_dist = detector.pinch_distance(lm)
            if 15 < pinch_dist < 220:
                volume.set_from_distance(pinch_dist)

        else:
            current_gest = Gesture.NONE
            scroller.reset()
            drawing.stop()
            dwell.reset()

        # ── Voice polling ─────────────────────────────────────
        vc_action = voice.poll()
        if vc_action:
            kind = vc_action[0]
            if kind == "macro":
                action_msg = macros.trigger(vc_action[1])
            elif kind == "key":
                macros.run_key(vc_action[1])
                action_msg = f"Key: {vc_action[1]}"
            elif kind == "scroll":
                import pyautogui
                pyautogui.scroll(30 * vc_action[1])
                action_msg = "Voice: scroll"
            elif kind == "mode":
                mode = Mode.DRAWING if vc_action[1] == "DRAW" else Mode.NORMAL
                if mode == Mode.NORMAL:
                    drawing.stop()

        # ── Draw air canvas ───────────────────────────────────
        if mode == Mode.DRAWING:
            frame = drawing.composite(frame)

        # ── Draw click flash ──────────────────────────────────
        flash.draw(frame)

        # ── Dwell indicator badge ─────────────────────────────
        if dwell_mode:
            cv2.putText(frame, "[ DWELL CLICK ON ]",
                        (FRAME_WIDTH // 2 - 95, FRAME_HEIGHT - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (0, 255, 180), 1, cv2.LINE_AA)

        # ── HUD ───────────────────────────────────────────────
        frame = hud.render(
            frame,
            gesture       = current_gest,
            mode          = mode,
            volume_pct    = volume.current_volume_pct,
            mouse_pos     = mouse_pos,
            action_msg    = action_msg,
            hand_detected = hand_detected,
        )

        cv2.imshow("MNCS — Maitra Neural Control System", frame)

        # ── Keyboard shortcuts ────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("d"):
            mode = Mode.NORMAL if mode == Mode.DRAWING else Mode.DRAWING
            if mode == Mode.NORMAL:
                drawing.stop()
        elif key == ord("c"):
            drawing.clear()
        elif key == ord("v"):
            mode = Mode.VOICE if mode != Mode.VOICE else Mode.NORMAL
        elif key == ord("w"):
            dwell_mode = not dwell_mode
            dwell.reset()
            print(f"[MNCS] Dwell-click: {'ON' if dwell_mode else 'OFF'}")
        elif key == ord("+"):
            mouse.set_alpha(min(1.0, mouse._alpha + 0.05))
        elif key == ord("-"):
            mouse.set_alpha(max(0.01, mouse._alpha - 0.05))

    # ── Cleanup ───────────────────────────────────────────────
    print("[MNCS] Shutting down...")
    voice.stop()
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[MNCS] Goodbye.")


if __name__ == "__main__":
    main()
