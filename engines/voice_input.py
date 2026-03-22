import threading
import time
from typing import Callable, Optional

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

VOICE_INTENTS = {
    "walk":         "walk forward",
    "go forward":   "walk forward",
    "move forward": "walk forward",
    "forward":      "walk forward",
    "straight":     "walk forward",
    "find bottle":  "bottle",
    "bottle":       "bottle",
    "find chair":   "chair",
    "chair":        "chair",
    "find person":  "person",
    "person":       "person",
    "find car":     "car",
    "car":          "car",
    "find door":    "door",
    "door":         "door",
    "find stairs":  "stairs",
    "stairs":       "stairs",
    "find laptop":  "laptop",
    "laptop":       "laptop",
    "find bag":     "backpack",
    "backpack":     "backpack",
    "find table":   "dining table",
    "table":        "dining table",
}

SCENE_PHRASES = {
    "what's around", "what is around", "describe", "what do you see",
    "surroundings", "scene", "what's here", "what is here", "look around"
}

QUESTION_WORDS = {"where", "how", "is there", "can i", "should i", "help"}


class VoiceInputEngine:
    """
    Continuous background microphone listener.
    Prints clear status to console so you always know what's happening.
    """

    def __init__(
        self,
        on_intent: Callable[[str], None],
        on_scene_request: Callable[[], None],
        on_question: Callable[[str], None],
        on_listening: Optional[Callable[[bool], None]] = None,
    ):
        self.on_intent = on_intent
        self.on_scene_request = on_scene_request
        self.on_question = on_question
        self.on_listening = on_listening or (lambda _: None)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_command_time = 0.0
        self._command_cooldown = 2.0
        self.mic = None
        self.recognizer = None
        self.ready = False

        self._setup()

    def _setup(self):
        if not SR_AVAILABLE:
            print("\n  [Voice] ❌ speech_recognition not installed.")
            print("  [Voice]    Fix: pip install SpeechRecognition pyaudio\n")
            return

        # List available mics
        try:
            mic_list = sr.Microphone.list_microphone_names()
            if not mic_list:
                print("  [Voice] ❌ No microphones found on this system.")
                return
            print(f"\n  [Voice] Found {len(mic_list)} microphone(s):")
            for i, name in enumerate(mic_list):
                print(f"           [{i}] {name}")
        except Exception as e:
            print(f"  [Voice] ❌ Could not list microphones: {e}")
            return

        # Init recognizer
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.7

        # Init default mic
        try:
            self.mic = sr.Microphone()
            print("  [Voice] Calibrating microphone for ambient noise (1 sec)...")
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print(f"  [Voice] ✅ Mic ready. Energy threshold: {int(self.recognizer.energy_threshold)}")
            self.ready = True
        except Exception as e:
            print(f"  [Voice] ❌ Microphone error: {e}")
            print("  [Voice]    Make sure a microphone is connected and not in use.")
            self.mic = None

    def start(self):
        if not self.ready:
            print("  [Voice] ⚠️  Voice input not available — keyboard controls still work.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("  [Voice] 🎤 Listening... speak a command anytime.\n")
        print("  [Voice] Example commands:")
        print("           'find chair'       → search for a chair")
        print("           'what's around'    → describe the scene")
        print("           'walk forward'     → navigation mode")
        print("           'where is the door?' → ask the AI\n")

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------ #

    def _listen_loop(self):
        while self._running:
            try:
                self.on_listening(True)
                print("  [Voice] 🎤 Listening...", end="\r")
                with self.mic as source:
                    audio = self.recognizer.listen(source, timeout=4, phrase_time_limit=6)
                self.on_listening(False)
                print("  [Voice] 🔄 Processing speech...   ", end="\r")
                self._process_audio(audio)
            except sr.WaitTimeoutError:
                # No speech detected in timeout window — loop again silently
                self.on_listening(False)
            except OSError as e:
                self.on_listening(False)
                print(f"\n  [Voice] ❌ Mic read error: {e}")
                time.sleep(1)
            except Exception as e:
                self.on_listening(False)
                time.sleep(0.5)

    def _process_audio(self, audio):
        try:
            text = self.recognizer.recognize_google(audio).lower().strip()
            print(f"\n  [Voice] ✅ Heard: \"{text}\"")
        except sr.UnknownValueError:
            print("  [Voice] 🔇 Could not understand — speak clearly and try again.", end="\r")
            return
        except sr.RequestError as e:
            print(f"\n  [Voice] ❌ Google Speech API error: {e}")
            print("  [Voice]    Check your internet connection.")
            return

        now = time.time()
        if now - self._last_command_time < self._command_cooldown:
            print("  [Voice] ⏳ Cooldown — command ignored.")
            return
        self._last_command_time = now

        self._dispatch(text)

    def _dispatch(self, text: str):
        # 1. Scene request
        if any(phrase in text for phrase in SCENE_PHRASES):
            print("  [Voice] → Scene description requested")
            self.on_scene_request()
            return

        # 2. Known intents (longest match first)
        for phrase in sorted(VOICE_INTENTS.keys(), key=len, reverse=True):
            if phrase in text:
                intent = VOICE_INTENTS[phrase]
                print(f"  [Voice] → Intent: {intent}")
                self.on_intent(intent)
                return

        # 3. Question words
        if any(text.startswith(w) for w in QUESTION_WORDS) or "?" in text:
            print(f"  [Voice] → Question: {text}")
            self.on_question(text)
            return

        # 4. Fallback — any 3+ word phrase goes to LLM
        if len(text.split()) >= 3:
            print(f"  [Voice] → Free-form: {text}")
            self.on_question(text)
            return

        print(f"  [Voice] ❓ Not recognized: \"{text}\" — try 'find chair' or 'what's around'")
