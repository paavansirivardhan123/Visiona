import threading
import time
from typing import Callable, Optional

try:
    import speech_recognition as sr
    _SR = True
except ImportError:
    _SR = False

VOICE_INTENTS = {
    "walk": "walk forward", "go forward": "walk forward",
    "move forward": "walk forward", "forward": "walk forward",
    "straight": "walk forward",
    "find bottle": "bottle", "bottle": "bottle",
    "find chair": "chair", "chair": "chair",
    "find person": "person", "person": "person",
    "find car": "car", "car": "car",
    "find door": "door", "door": "door",
    "find stairs": "stairs", "stairs": "stairs",
    "find laptop": "laptop", "laptop": "laptop",
    "find bag": "backpack", "backpack": "backpack",
    "find table": "dining table", "table": "dining table",
}

SCENE_PHRASES = {
    "what is around", "what around", "describe", "what do you see",
    "surroundings", "scene", "look around",
}

QUESTION_WORDS = {"where", "how", "is there", "can i", "should i", "help"}


class VoiceInputEngine:

    def __init__(self, on_intent, on_scene_request, on_question, on_listening=None):
        self.on_intent = on_intent
        self.on_scene_request = on_scene_request
        self.on_question = on_question
        self.on_listening = on_listening or (lambda _: None)
        self._running = False
        self._thread = None
        self._cooldown = 2.0
        self._last_cmd = 0.0
        self.mic = None
        self.recognizer = None
        self.ready = False
        self._setup()

    def _setup(self):
        if not _SR:
            print("  [Voice] speech_recognition not installed. Run: uv add SpeechRecognition pyaudio")
            return
        try:
            mic_list = sr.Microphone.list_microphone_names()
            if not mic_list:
                print("  [Voice] No microphones found.")
                return
            print(f"  [Voice] Found {len(mic_list)} microphone(s):")
            for i, n in enumerate(mic_list):
                print(f"           [{i}] {n}")
        except Exception as e:
            print(f"  [Voice] Could not list mics: {e}")
            return
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.7
        try:
            self.mic = sr.Microphone()
            print("  [Voice] Calibrating for ambient noise (1 sec)...")
            with self.mic as src:
                self.recognizer.adjust_for_ambient_noise(src, duration=1)
            print(f"  [Voice] Mic ready. Threshold: {int(self.recognizer.energy_threshold)}")
            self.ready = True
        except Exception as e:
            print(f"  [Voice] Mic error: {e}")

    def start(self):
        if not self.ready:
            print("  [Voice] Voice input unavailable.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("  [Voice] Listening... speak a command anytime.")
        print("  [Voice] Examples: find chair | what is around | where is the door")

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                self.on_listening(True)
                with self.mic as src:
                    audio = self.recognizer.listen(src, timeout=4, phrase_time_limit=6)
                self.on_listening(False)
                self._process(audio)
            except sr.WaitTimeoutError:
                self.on_listening(False)
            except OSError as e:
                self.on_listening(False)
                print(f"  [Voice] Mic error: {e}")
                time.sleep(1)
            except Exception:
                self.on_listening(False)
                time.sleep(0.5)

    def _process(self, audio):
        try:
            text = self.recognizer.recognize_google(audio).lower().strip()
            print(f"  [Voice] Heard: {text}")
        except sr.UnknownValueError:
            return
        except sr.RequestError as e:
            print(f"  [Voice] API error: {e}")
            return
        now = time.time()
        if now - self._last_cmd < self._cooldown:
            return
        self._last_cmd = now
        self._dispatch(text)

    def _dispatch(self, text):
        if any(p in text for p in SCENE_PHRASES):
            self.on_scene_request()
            return
        for phrase in sorted(VOICE_INTENTS, key=len, reverse=True):
            if phrase in text:
                self.on_intent(VOICE_INTENTS[phrase])
                return
        if any(text.startswith(w) for w in QUESTION_WORDS) or "?" in text:
            self.on_question(text)
            return
        if len(text.split()) >= 3:
            self.on_question(text)