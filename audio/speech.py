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

_MAX_AGE = 5.0   # seconds — drop messages older than this


class SpeechEngine:

    def __init__(self):
        self._pq = queue.Queue()
        self._nq = queue.Queue(maxsize=1)
        self.last_spoken_time = 0.0
        self.last_spoken_text = ""
        self._running = True
        self._volume  = 1.0   # current volume (0.0 – 1.0)
        self._interrupt_requested = False

        if _PYTTSX3:
            try:
                self.engine = pyttsx3.init(driverName='sapi5')
                self.engine.setProperty('rate', Config.SPEECH_RATE)
                self.engine.setProperty('volume', self._volume)
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

    def interrupt(self):
        """Immediately stop current speech and clear queues."""
        # Note: We don't set _interrupt_requested here permanently, 
        # it's managed by duck/unduck.
        self._flush_all()
        if self.engine and self.engine.isBusy():
            try:
                self.engine.stop()
            except Exception:
                pass

    def duck(self):
        """Immediately silence all speech when mic is active."""
        self._interrupt_requested = True
        self._volume = 0.0 # Complete silence
        self.interrupt() 
        if self.engine:
            try:
                self.engine.setProperty('volume', self._volume)
            except Exception:
                pass

    def unduck(self):
        """Restore volume to 100% when mic is released."""
        self._interrupt_requested = False
        self._volume = 1.0
        if self.engine:
            try:
                self.engine.setProperty('volume', self._volume)
            except Exception:
                pass

    def _worker(self):
        while self._running:
            item = None
            try:
                # Priority queue first
                item = self._pq.get_nowait()
            except queue.Empty:
                try:
                    # Normal queue with timeout
                    item = self._nq.get(timeout=0.1)
                except queue.Empty:
                    continue
            
            if item is None:
                break
                
            text, ts, is_emergency = item

            # If we're currently ducked/interrupted, we skip messages 
            # UNLESS it's an emergency message.
            if self._interrupt_requested and not is_emergency:
                continue

            if time.time() - ts > _MAX_AGE:
                continue

            if self.engine:
                try:
                    # If it's an emergency, force volume to 100% even if ducked
                    target_vol = 1.0 if is_emergency else self._volume
                    self.engine.setProperty('volume', target_vol)
                    self.engine.say(text)
                    self.engine.runAndWait()
                    # Restore volume if it was an emergency bypass
                    if is_emergency:
                        self.engine.setProperty('volume', self._volume)
                    self.last_spoken_time = time.time()
                except Exception as e:
                    print(f"  [Speech] Playback error: {e}")
            else:
                print(f"  [Speech] {text}")

    def _flush_all(self):
        """Clear both priority and normal queues."""
        while not self._pq.empty():
            try: self._pq.get_nowait()
            except queue.Empty: break
        while not self._nq.empty():
            try: self._nq.get_nowait()
            except queue.Empty: break

    def speak(self, text: str, priority: bool = False, bypass_cooldown: bool = False, emergency: bool = False):
        if not text:
            return
        now = time.time()
        
        import re
        # 1. Anti-Spam: Strip numbers to extract the 'semantic meaning'
        semantic_base = re.sub(r'\d+', '', text.strip().lower())
        
        if not hasattr(self, '_semantic_history'):
            self._semantic_history = {}
            
        last_time = self._semantic_history.get(semantic_base, 0)
        
        # Warnings and high-priority alerts repeat faster than normal scenery
        is_warning = "warning" in semantic_base or "very close" in semantic_base or "found" in semantic_base or emergency
        required_cooldown = Config.SPEECH_COOLDOWN if is_warning else getattr(Config, 'SEMANTIC_COOLDOWN', 12.0)
        
        cooldown_ok = bypass_cooldown or emergency or (now - last_time) > required_cooldown

        if priority or emergency:
            if bypass_cooldown or emergency:
                # Add 'emergency' flag to the item so the worker knows to bypass ducking
                self._pq.put((text, now, emergency)) 
            else:
                self._flush_normal()
                self._pq.put((text, now, False))
            self._semantic_history[semantic_base] = now
            self.last_spoken_time = now
        elif cooldown_ok:
            self._flush_normal()
            try:
                self._nq.put_nowait((text, now, False))
                self._semantic_history[semantic_base] = now
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
