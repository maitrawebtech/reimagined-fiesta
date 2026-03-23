# ============================================================
# MNCS — voice/voice_commands.py
# Background voice command listener using SpeechRecognition
# ============================================================

import threading
import queue
import time
from config import VOICE_COMMANDS

try:
    import speech_recognition as sr
    _SR_AVAILABLE = True
except ImportError:
    _SR_AVAILABLE = False


class VoiceCommandListener:
    """
    Runs speech recognition in a background daemon thread.
    Commands are placed in a queue for the main loop to poll.

    Usage:
        listener = VoiceCommandListener()
        listener.start()
        ...
        action = listener.poll()   # returns action tuple or None
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None
        self._available = _SR_AVAILABLE

    def start(self):
        if not self._available:
            print("[Voice] SpeechRecognition not installed — voice disabled.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("[Voice] Listener started.")

    def stop(self):
        self._running = False

    def poll(self) -> tuple | None:
        """
        Returns the next pending voice command action tuple,
        or None if no command is queued.
        """
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def _listen_loop(self):
        r = sr.Recognizer()
        r.energy_threshold   = 300
        r.dynamic_energy_threshold = True
        r.pause_threshold    = 0.6

        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=1.0)
            print("[Voice] Microphone calibrated. Listening...")

            while self._running:
                try:
                    audio = r.listen(source, timeout=2, phrase_time_limit=4)
                    text  = r.recognize_google(audio).lower().strip()
                    print(f"[Voice] Heard: '{text}'")
                    self._dispatch(text)
                except sr.WaitTimeoutError:
                    pass
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print(f"[Voice] API error: {e}")
                    time.sleep(2)
                except Exception as e:
                    print(f"[Voice] Error: {e}")

    def _dispatch(self, text: str):
        """Match text against VOICE_COMMANDS dict and queue action."""
        for phrase, action in VOICE_COMMANDS.items():
            if phrase in text:
                self._queue.put(action)
                print(f"[Voice] Matched '{phrase}' → {action}")
                return
        print(f"[Voice] No match for: '{text}'")
