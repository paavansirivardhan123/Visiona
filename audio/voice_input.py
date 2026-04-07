"""
VoiceInputEngine - Hold-to-Talk PTT voice capture.

Hold V -> start_recording()  (called from main.py on_press)
Release V -> stop_recording() (called from main.py on_release)

The mic stream is pre-warmed at startup so there is zero activation lag.
Audio is captured into a buffer while the key is held, then sent to
Google Speech-to-Text when released.
"""
import threading
import time
import io
import wave

try:
    import pyaudio
    import speech_recognition as sr
    _READY = True
except ImportError:
    _READY = False
    print("  [Voice] pyaudio or SpeechRecognition not installed.")
    print("  [Voice] Run: uv add pyaudio SpeechRecognition")


class VoiceInputEngine:

    def __init__(self, on_speech, on_listening=None):
        self.on_speech    = on_speech
        self.on_listening = on_listening or (lambda _: None)
        self.ready        = False

        self._lock      = threading.Lock()
        self._active    = False
        self._frames    = []
        self._stream    = None
        self._pa        = None
        self._capturing = False
        self._last_cmd  = 0.0

        if not _READY:
            return

        try:
            self._pa = pyaudio.PyAudio()
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024,
                start=True,
            )
            self._capturing = True
            threading.Thread(target=self._capture_loop, daemon=True).start()
            self.ready = True
            print("  [Voice] Mic pre-warmed. Hold V to speak, hold G for GPS.")
        except Exception as e:
            print(f"  [Voice] Mic init failed: {e}")
            print("  [Voice] Check that a microphone is connected and not in use.")

    def start_recording(self):
        """Start saving mic frames. Called when V/G key is pressed."""
        if not self.ready or self._active:
            return
        with self._lock:
            self._active = True
            self._frames = []
        self.on_listening(True)
        print("  [Voice] Listening...", end="\r")

    def stop_recording(self):
        """Stop saving frames and process the buffer. Called when V/G key is released."""
        if not self._active:
            return
        with self._lock:
            self._active = False
            captured = self._frames.copy()
            self._frames = []
        self.on_listening(False)

        if not captured:
            print("  [Voice] No audio captured.")
            return

        print("  [Voice] Processing...   ", end="\r")
        threading.Thread(target=self._process, args=(captured,), daemon=True).start()

    def stop(self):
        """Clean up mic resources on shutdown."""
        self._capturing = False
        try:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
            if self._pa:
                self._pa.terminate()
        except Exception:
            pass

    def _capture_loop(self):
        while self._capturing:
            try:
                data = self._stream.read(1024, exception_on_overflow=False)
                if self._active:
                    with self._lock:
                        self._frames.append(data)
            except Exception as e:
                if self._capturing:
                    print(f"\n  [Voice] Capture error: {e}")
                break

    def _process(self, frames):
        """Convert raw PCM frames to WAV, send to Google STT, dispatch result."""
        try:
            raw = b"".join(frames)

            wav_buf = io.BytesIO()
            with wave.open(wav_buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
                wf.setframerate(16000)
                wf.writeframes(raw)
            wav_buf.seek(0)  # rewind AFTER wave.open closes

            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_buf) as source:
                audio = recognizer.record(source)

            text = recognizer.recognize_google(audio).lower().strip()
            if text:
                print(f"  [Voice] Heard: \"{text}\"")
                now = time.time()
                if now - self._last_cmd >= 0.5:
                    self._last_cmd = now
                    self.on_speech(text)

        except sr.UnknownValueError:
            print("  [Voice] Could not understand - speak clearly and try again.")
        except sr.RequestError as e:
            print(f"  [Voice] Google STT error (check internet): {e}")
        except Exception as e:
            print(f"  [Voice] Processing error: {e}")

    # Legacy stubs
    def start(self): pass
    def trigger_ptt(self): pass