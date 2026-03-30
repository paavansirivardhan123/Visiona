"""
SpeechEngine — thread-safe TTS with priority queue and stale-message dropping.
"""
import threading
import queue
import time
from core.config import Config

try:
    import pyttsx3
    _PYTTSX3 = True
except ImportError:
    _PYTTSX3 = False
    print("  [Speech] pyttsx3 not installed. Run: uv add pyttsx3")

_MAX_AGE = 2.0   # seconds — drop messages older than this


class SpeechEngine:

    def __init__(self):
        self._pq = queue.Queue()                  # priority messages
        self._nq = queue.Queue(maxsize=1)         # normal — latest only
        self.last_spoken_time = 0.0
        self.last_spoken_text = ""
        self._running = True

        if _PYTTSX3:
            try:
                self.engine = pyttsx3.init(driverName='sapi5')
                self.engine.setProperty('rate', Config.SPEECH_RATE)
                for v in self.engine.getProperty('voices'):
                    if 'female' in v.name.lower() or 'zira' in v.name.lower():
                        self.engine.setProperty('voice', v.id)
                        break
            except Exception as e:
                print(f"  [Speech] Init warning: {e}")
                self.engine = None
        else:
            self.engine = None

        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while self._running:
            item = None
            try:
                item = self._pq.get_nowait()
            except queue.Empty:
                try:
                    item = self._nq.get(timeout=0.1)
                except queue.Empty:
                    continue
            if item is None:
                break
            text, ts = item
            if time.time() - ts > _MAX_AGE:
                continue
            if self.engine:
                self.engine.say(text)
                self.engine.runAndWait()
                self.last_spoken_time = time.time()
            else:
                print(f"  [Speech] {text}")

    def speak(self, text: str, priority: bool = False):
        if not text:
            return
        now = time.time()
        cooldown_ok = (now - self.last_spoken_time) > Config.SPEECH_COOLDOWN
        is_new = text.strip().lower() != self.last_spoken_text.strip().lower()
        if priority:
            self._flush_normal()
            self._pq.put((text, now))
            self.last_spoken_text = text
        elif is_new and cooldown_ok:
            self._flush_normal()
            try:
                self._nq.put_nowait((text, now))
                self.last_spoken_text = text
            except queue.Full:
                pass

    def speak_all(self, messages: list, first_priority: bool = False):
        for i, msg in enumerate(messages[:Config.MAX_MESSAGES]):
            self.speak(msg, priority=(i == 0 and first_priority))

    def _flush_normal(self):
        while not self._nq.empty():
            try: self._nq.get_nowait()
            except queue.Empty: break

    def stop(self):
        self._running = False
        self._pq.put(None)
        self._nq.put(None)
