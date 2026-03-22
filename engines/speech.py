import pyttsx3
import threading
import queue
import time
import winsound
from core.config import Config

# Max age (seconds) a queued message can wait before being discarded as stale
_MAX_MESSAGE_AGE = 1.5

class SpeechEngine:
    """
    Thread-safe speech engine with:
    - Priority queue (urgent alerts jump the queue)
    - Stale message dropping (messages older than 1.5s are discarded)
    - Max queue depth of 1 for normal messages (always latest only)
    - Deduplication + cooldown
    - Stereo audio spatial cues (beeps for left/right hazards)
    """

    def __init__(self, rate: int = 185):
        # Each item is a tuple: (text, enqueue_time)
        self._priority_queue: queue.Queue = queue.Queue()
        self._normal_queue: queue.Queue = queue.Queue(maxsize=1)  # only keep latest
        self.last_spoken_time = 0.0
        self.last_spoken_text = ""
        self._lock = threading.Lock()

        try:
            self.engine = pyttsx3.init(driverName='sapi5')
            self.engine.setProperty('rate', rate)
            voices = self.engine.getProperty('voices')
            for v in voices:
                if 'female' in v.name.lower() or 'zira' in v.name.lower():
                    self.engine.setProperty('voice', v.id)
                    break
        except Exception as e:
            print(f"Speech init warning: {e}")
            self.engine = None

        self._running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self._running:
            item = None

            # Priority queue always wins
            try:
                item = self._priority_queue.get_nowait()
            except queue.Empty:
                try:
                    item = self._normal_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

            if item is None:
                break

            text, enqueue_time = item

            # Drop stale messages — video has moved on
            age = time.time() - enqueue_time
            if age > _MAX_MESSAGE_AGE:
                continue

            if self.engine:
                self.engine.say(text)
                self.engine.runAndWait()
                self.last_spoken_time = time.time()

    def speak(self, text: str, priority: bool = False):
        if not text:
            return
        now = time.time()
        cooldown_ok = (now - self.last_spoken_time) > Config.SPEECH_COOLDOWN
        is_new = text.strip().lower() != self.last_spoken_text.strip().lower()

        if priority:
            # Flush normal queue, push immediately
            self._flush_normal()
            self._priority_queue.put((text, now))
            self.last_spoken_text = text
        elif is_new and cooldown_ok:
            # Replace any waiting normal message with the latest one
            self._flush_normal()
            try:
                self._normal_queue.put_nowait((text, now))
                self.last_spoken_text = text
            except queue.Full:
                pass  # Already has a fresh message

    def _flush_normal(self):
        """Discard all pending normal messages."""
        while not self._normal_queue.empty():
            try:
                self._normal_queue.get_nowait()
            except queue.Empty:
                break

    def beep_spatial(self, position: str):
        """Play a directional audio cue for spatial awareness."""
        if not Config.AUDIO_CUES_ENABLED:
            return
        try:
            if "left" in position:
                threading.Thread(
                    target=winsound.Beep,
                    args=(Config.BEEP_FREQ_LEFT, Config.BEEP_DURATION_MS),
                    daemon=True
                ).start()
            elif "right" in position:
                threading.Thread(
                    target=winsound.Beep,
                    args=(Config.BEEP_FREQ_RIGHT, Config.BEEP_DURATION_MS),
                    daemon=True
                ).start()
        except Exception:
            pass

    def stop(self):
        self._running = False
        self._priority_queue.put(None)
        self._normal_queue.put(None)
