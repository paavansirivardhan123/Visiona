import cv2
import time
from core.config import Config
from engines.vision import VisionSystem
from engines.speech import SpeechEngine
from engines.navigator import AINavigator
from engines.logger import SessionLogger
from engines.voice_input import VoiceInputEngine

# Keyboard fallback — still works for sighted operators/developers
INTENT_KEYS = {
    ord('1'): "walk forward",
    ord('2'): "bottle",
    ord('3'): "chair",
    ord('4'): "person",
    ord('5'): "car",
    ord('6'): "laptop",
    ord('7'): "backpack",
    ord('d'): "door",
    ord('s'): "stairs",
}

class NaVisionApp:
    """
    NaVision AI — Assistive Navigation System
    Real-time object detection + distance estimation + AI guidance.
    Supports both voice commands and keyboard input.
    """

    def __init__(self):
        print("\n  NaVision AI — Starting up...")
        self.vision  = VisionSystem()
        self.speech  = SpeechEngine()
        self.agent   = AINavigator()
        self.logger  = SessionLogger()
        self._frame_count = 0
        self._fps_time = time.time()
        self._fps = 0.0
        self._detections = []
        self._mic_active = False   # visual mic indicator

        # Video source
        self.cap = cv2.VideoCapture(Config.VIDEO_SOURCE)
        if not self.cap.isOpened():
            print(f"  Could not open '{Config.VIDEO_SOURCE}', falling back to webcam.")
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("No video source available.")

        # Voice input — runs in background thread
        self.voice = VoiceInputEngine(
            on_intent=self._on_voice_intent,
            on_scene_request=self._on_scene_request,
            on_question=self._on_voice_question,
            on_listening=self._on_listening,
        )
        self.voice.start()

        self.speech.speak("NaVision AI ready. You can speak to navigate.")
        print("  Keyboard controls: 1-7 D S = intent | H = scene | ESC = quit")
        print("  Voice commands: say 'find chair', 'what's around', or ask anything\n")

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            self._frame_count += 1
            frame = cv2.resize(frame, (Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT))

            if self._frame_count % Config.FRAME_SKIP == 0:
                self._detections = self.vision.detect(frame)
                self.agent.process_frame(self._detections, self._on_speech)
                self.logger.log_detection(self._detections)
                self._update_fps()

            self.vision.draw_overlay(frame, self._detections, self.agent)
            self._draw_fps(frame)
            self._draw_mic_indicator(frame)
            cv2.imshow(Config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            self._handle_key(key)

        self._shutdown()

    # ------------------------------------------------------------------ #
    #  Speech callback
    # ------------------------------------------------------------------ #

    def _on_speech(self, text: str, priority: bool = False):
        self.speech.speak(text, priority=priority)
        self.logger.log_instruction(text, source="agent", priority=priority)
        if self._detections:
            closest = self._detections[0]
            if closest.is_hazard and closest.distance_cm and closest.distance_cm < Config.DIST_NEAR:
                self.speech.beep_spatial(closest.position)

    # ------------------------------------------------------------------ #
    #  Voice input callbacks
    # ------------------------------------------------------------------ #

    def _on_voice_intent(self, intent: str):
        """User spoke a navigation intent."""
        print(f"  [Voice] Intent → {intent}")
        self.agent.set_intent(intent, self._on_speech)
        self.logger.log_instruction(f"Voice intent: {intent}", source="voice")

    def _on_scene_request(self):
        """User asked what's around them."""
        print("  [Voice] Scene request")
        scene = self.agent._describe_scene(self._detections)
        if not self._detections:
            self.speech.speak("The path ahead looks clear. No objects detected.")
        else:
            self.speech.speak(f"I can see: {scene}")
        self.logger.log_instruction("scene_request", source="voice")

    def _on_voice_question(self, question: str):
        """User asked a free-form question."""
        print(f"  [Voice] Question → {question}")
        self.speech.speak("Let me check.", priority=False)
        self.agent.answer_question(question, self._on_speech)
        self.logger.log_instruction(f"Voice question: {question}", source="voice")

    def _on_listening(self, active: bool):
        """Mic state changed — update HUD indicator."""
        self._mic_active = active

    # ------------------------------------------------------------------ #
    #  Keyboard fallback
    # ------------------------------------------------------------------ #

    def _handle_key(self, key: int):
        if key in INTENT_KEYS:
            intent = INTENT_KEYS[key]
            self.agent.set_intent(intent, self._on_speech)
            self.logger.log_instruction(f"Key intent: {intent}", source="keyboard")
        elif key == ord('h'):
            self._on_scene_request()

    # ------------------------------------------------------------------ #
    #  HUD helpers
    # ------------------------------------------------------------------ #

    def _update_fps(self):
        now = time.time()
        elapsed = now - self._fps_time
        if elapsed > 0:
            self._fps = round(1.0 / elapsed, 1)
        self._fps_time = now

    def _draw_fps(self, frame):
        cv2.putText(frame, f"{self._fps} FPS",
                    (Config.DISPLAY_WIDTH - 80, Config.DISPLAY_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)

    def _draw_mic_indicator(self, frame):
        """Small mic dot in corner — green when listening, grey when idle."""
        color = (0, 220, 80) if self._mic_active else (80, 80, 80)
        cx = Config.DISPLAY_WIDTH - 20
        cy = Config.DISPLAY_HEIGHT - 30
        cv2.circle(frame, (cx, cy), 7, color, -1, cv2.LINE_AA)
        cv2.putText(frame, "MIC", (cx - 22, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    #  Shutdown
    # ------------------------------------------------------------------ #

    def _shutdown(self):
        stats = self.logger.get_stats()
        print(f"\n  Session ended. Duration: {stats['session_duration_s']}s | "
              f"Events logged: {stats['total_events']}")
        self.voice.stop()
        self.cap.release()
        cv2.destroyAllWindows()
        self.speech.stop()
        self.agent.stop()


if __name__ == "__main__":
    app = NaVisionApp()
    app.run()
