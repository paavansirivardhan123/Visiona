"""
VoiceInputEngine — Dynamic Natural Language Capture with PTT.

Supports explicit start/stop recording for 'Hold-to-Talk' (V key).
"""
import threading
import time
import io
import wave
from typing import Callable, Optional

try:
    import speech_recognition as sr
    import pyaudio
    _READY = True
except ImportError:
    _READY = False

class VoiceInputEngine:

    def __init__(self, on_speech, on_listening=None):
        self.on_speech        = on_speech
        self.on_listening     = on_listening or (lambda _: None)

        self._cooldown  = 0.3
        self._last_cmd  = 0.0
        self._active    = False              # True while recording
        self._recording = False              # True while stream is active
        self._frames    = []
        self.ready      = _READY
        self._pa        = pyaudio.PyAudio() if _READY else None
        self._stream    = None
        self._lock      = threading.Lock()
        
        if _READY:
            self._start_background_stream()
            print(f"  [Voice] Mic ready. Low-latency PTT (V key) Active.")

    def _start_background_stream(self):
        """Pre-warms the microphone stream to eliminate activation lag."""
        try:
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024,
                start=True
            )
            self._recording = True
            threading.Thread(target=self._capture_loop, daemon=True).start()
        except Exception as e:
            print(f"  [Voice] Failed to pre-warm mic: {e}")
            self.ready = False

    def start_recording(self):
        """Instantly starts saving frames for processing."""
        if not self.ready or self._active:
            return
        
        with self._lock:
            self._active = True
            self._frames = []
        
        self.on_listening(True)
        print("  [Voice] 🎤 Listening (Zero-lag)...")

    def _capture_loop(self):
        """Continuously reads from the mic but only saves when _active is True."""
        while self._recording:
            try:
                data = self._stream.read(1024, exception_on_overflow=False)
                if self._active:
                    with self._lock:
                        self._frames.append(data)
            except Exception as e:
                if self._recording:
                    print(f"  [Voice] Capture loop error: {e}")
                break

    def stop_recording(self):
        """Stops capturing and processes the buffer."""
        if not self._active:
            return
        
        with self._lock:
            self._active = False
            captured_frames = self._frames.copy()
            self._frames = []
            
        self.on_listening(False)
        print("  [Voice] Processing speech...")
        
        if captured_frames:
            threading.Thread(target=self._process_buffer, args=(captured_frames,), daemon=True).start()

    def _process_buffer(self, frames):
        try:
            raw_data = b"".join(frames)
            with io.BytesIO() as wav_io:
                with wave.open(wav_io, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(16000)
                    wf.writeframes(raw_data)
                
                wav_io.seek(0)
                recognizer = sr.Recognizer()
                with sr.AudioFile(wav_io) as source:
                    audio = recognizer.record(source)
                    
            text = recognizer.recognize_google(audio).lower().strip()
            if text:
                print(f"  [Voice] Heard: \"{text}\"")
                self._dispatch(text)
        except sr.UnknownValueError:
            pass
        except Exception as e:
            print(f"  [Voice] Process Error: {e}")

    def stop(self):
        self._recording = False
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except: pass
        if self._pa:
            try:
                self._pa.terminate()
            except: pass

    def _dispatch(self, text):
        now = time.time()
        if now - self._last_cmd < self._cooldown:
            return
        self._last_cmd = now
        self.on_speech(text)

    def start(self): pass
    def trigger_ptt(self): pass



