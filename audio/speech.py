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
        self._eq = queue.Queue()
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
        # Removed self.engine.stop() core bug: calling stop() on Windows SAPI5 permanently 
        # breaks the event loop, causing subsequent TTS requests to silently fail. 
        # The ducking logic (volume = 0.0) is sufficient to mute the output without breaking the engine.

    def duck(self):
        """Immediately silence all speech when mic is active."""
        self._interrupt_requested = True
        self._volume = 0.0 # Complete silence
        # REMOVED self.interrupt() so it doesn't flush queues, allowing them to seamlessly pause!
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
                # Emergency queue always processed first, regardless of ducking
                item = self._eq.get_nowait()
            except queue.Empty:
                if not self._interrupt_requested:
                    try:
                        # Priority queue
                        item = self._pq.get_nowait()
                    except queue.Empty:
                        try:
                            # Normal queue with timeout
                            item = self._nq.get(timeout=0.1)
                        except queue.Empty:
                            pass
                else:
                    time.sleep(0.1)
            
            if item is None:
                continue
                
            text, ts, is_emergency, scheduled_time = item

            # If user hold V for a long time, we don't want to drop the message unless it's very stale
            if time.time() - ts > 15.0:
                continue

            # Delay logic for scheduled messages (emergency messages bypass this)
            if not is_emergency and scheduled_time is not None:
                current_time = time.time()
                
                # Drop stale scheduled messages (scheduled_time > 15 seconds in the past)
                if current_time - scheduled_time > 15.0:
                    continue
                
                # If scheduled_time is in the future, sleep until then
                if scheduled_time > current_time:
                    delay = scheduled_time - current_time
                    time.sleep(delay)

            from datetime import datetime
            try:
                with open("conversation_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"SYSTEM SPEECH ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}): {text}\n")
            except Exception:
                pass

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
        """Clear all queues."""
        while not self._eq.empty():
            try: self._eq.get_nowait()
            except queue.Empty: break
        while not self._pq.empty():
            try: self._pq.get_nowait()
            except queue.Empty: break
        while not self._nq.empty():
            try: self._nq.get_nowait()
            except queue.Empty: break

    def speak(self, text: str, priority: bool = False, bypass_cooldown: bool = False, emergency: bool = False, scheduled_time: float = None):
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

        if emergency:
            # Emergency messages always use scheduled_time=None (immediate playback)
            self._eq.put((text, now, True, None))
            self._semantic_history[semantic_base] = now
            self.last_spoken_time = now
        elif priority:
            if bypass_cooldown:
                # Priority queue: use provided scheduled_time
                self._pq.put((text, now, False, scheduled_time)) 
            else:
                self._flush_normal()
                # Priority queue: use provided scheduled_time
                self._pq.put((text, now, False, scheduled_time))
            self._semantic_history[semantic_base] = now
            self.last_spoken_time = now
        elif cooldown_ok:
            self._flush_normal()
            try:
                # Normal queue: use provided scheduled_time
                self._nq.put_nowait((text, now, False, scheduled_time))
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
