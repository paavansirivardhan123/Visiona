"""
Visiona AI — Multi-camera assistive navigation system.

Architecture:
  4 CameraFeed threads → VisionSystem (YOLO + depth + track + speed + TTC)
  → DetectionPriorityQueue → grouping → SpeechEngine + AlertSystem
"""
import cv2
import time
import threading
from typing import Dict, List, Optional

from core.config import Config
from models.detection import Detection
from models.priority_queue import DetectionPriorityQueue
from engines.vision import VisionSystem
from engines.grouping import group_detections, build_speech_messages
from engines.alert import AlertSystem
from engines.speech import SpeechEngine
from engines.voice_input import VoiceInputEngine
from engines.logger import SessionLogger


class CameraFeed:
    """Manages a single camera/video source for one direction."""

    def __init__(self, direction: str, source):
        self.direction = direction
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame = None
        self.active = False
        self._lock = threading.Lock()

        if source is None:
            return

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print(f"  [Camera] Could not open {direction} source: {source}")
            self.cap = None
            return

        self.active = True
        threading.Thread(target=self._read_loop, daemon=True).start()
        print(f"  [Camera] {direction} feed active")

    def _read_loop(self):
        while self.active and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            with self._lock:
                self.frame = frame

    def get_frame(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.active = False
        if self.cap:
            self.cap.release()


class VisionaApp:

    def __init__(self):
        print("\n  Visiona AI — Starting up...\n")

        self.vision = VisionSystem()
        self.alert  = AlertSystem()
        self.speech = SpeechEngine()
        self.logger = SessionLogger()

        self.feeds: Dict[str, CameraFeed] = {
            d: CameraFeed(d, s) for d, s in Config.SOURCES.items()
        }
        active = [d for d, f in self.feeds.items() if f.active]
        if not active:
            raise RuntimeError("No active camera feeds. Check Config.SOURCES.")
        print(f"\n  Active feeds: {', '.join(active)}")

        self._frame_count  = 0
        self._all_dets: List[Detection] = []
        self._state        = "SCANNING"
        self._last_info    = ""
        self._mic_active   = False
        self._search_intent: Optional[str] = None

        self.voice = VoiceInputEngine(
            on_intent=self._on_intent,
            on_scene_request=self._on_scene,
            on_question=self._on_question,
            on_listening=lambda a: setattr(self, '_mic_active', a),
        )
        self.voice.start()

        self.speech.speak("Visiona AI ready.")
        print("\n  Controls: H = describe scene | ESC = quit")
        print("  Voice: 'find chair' | 'what is around' | ask anything\n")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        while True:
            self._frame_count += 1
            all_dets: List[Detection] = []
            processed = False

            for direction, feed in self.feeds.items():
                if not feed.active:
                    continue
                frame = feed.get_frame()
                if frame is None:
                    continue

                # Always resize for display
                display_frame = cv2.resize(frame, (Config.DISPLAY_W, Config.DISPLAY_H))

                # Run detection pipeline every FRAME_SKIP frames
                if self._frame_count % Config.FRAME_SKIP == 0:
                    dets = self.vision.detect(frame, direction)
                    all_dets.extend(dets)
                    self.logger.log_detections(dets, direction)
                    processed = True

                # Always draw and show — smooth display regardless of processing
                self.vision.draw_overlay(
                    display_frame,
                    [d for d in self._all_dets if d.direction == direction],
                    self._state, self._last_info,
                )
                self._draw_extras(display_frame, direction)
                cv2.imshow(f"Visiona — {direction}", display_frame)

            if processed and all_dets:
                self._all_dets = all_dets
                self._pipeline(all_dets)

            # Minimal wait — keeps display smooth
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('h'):
                self._on_scene()

        self._shutdown()

    # ------------------------------------------------------------------
    # Detection pipeline
    # ------------------------------------------------------------------

    def _pipeline(self, detections: List[Detection]):
        pq = DetectionPriorityQueue()
        pq.push_all(detections)
        sorted_dets = pq.drain()

        hp = [d for d in sorted_dets if d.is_high_priority]

        # State
        if hp:
            self._state = "ALERT"
        elif any(d.distance_m and d.distance_m < 3.0 for d in sorted_dets):
            self._state = "AVOIDING"
        else:
            self._state = "SCANNING"

        # Beep
        self.alert.process(hp)

        # Speech
        grouped  = group_detections(sorted_dets)
        messages = build_speech_messages(grouped, hp)
        if messages:
            self._last_info = " | ".join(messages)
            self.logger.log_speech(messages)
            self.speech.speak_all(messages, first_priority=bool(hp))

        # Target seeking
        if self._search_intent:
            self._seek(sorted_dets)

    def _seek(self, detections: List[Detection]):
        matches = [d for d in detections if self._search_intent in d.label.lower()]
        if not matches:
            return
        t = matches[0]
        dist = f"{t.distance_ft} feet" if t.distance_ft else "nearby"
        dir_s = {"FRONT": "in front", "LEFT": "on the left",
                 "RIGHT": "on the right", "BACK": "behind you"}.get(t.direction, "")
        self.speech.speak(f"Found {t.label}. {dist} {dir_s}.", priority=True)
        self._search_intent = None
        self._state = "GUIDING"

    # ------------------------------------------------------------------
    # Voice callbacks
    # ------------------------------------------------------------------

    def _on_intent(self, intent: str):
        print(f"  [Voice] Intent: {intent}")
        self._search_intent = intent
        self._state = "SEARCHING" if intent != "walk forward" else "SCANNING"
        self.speech.speak(f"Searching for {intent}.", priority=True)

    def _on_scene(self):
        print("  [Voice] Scene request")
        if not self._all_dets:
            self.speech.speak("Path looks clear. No objects detected nearby.")
            return
        grouped  = group_detections(self._all_dets)
        messages = build_speech_messages(grouped, [])
        self.speech.speak_all(messages)

    def _on_question(self, question: str):
        print(f"  [Voice] Question: {question}")
        self.speech.speak(
            f"I can see: {self._last_info}" if self._last_info else "Nothing nearby."
        )

    # ------------------------------------------------------------------
    # HUD extras
    # ------------------------------------------------------------------

    def _draw_extras(self, frame, direction: str):
        cv2.putText(frame, direction,
                    (Config.DISPLAY_W // 2 - 30, Config.DISPLAY_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2, cv2.LINE_AA)
        color = (0, 220, 80) if self._mic_active else (80, 80, 80)
        cv2.circle(frame, (Config.DISPLAY_W - 18, Config.DISPLAY_H - 28),
                   6, color, -1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self):
        stats = self.logger.get_stats()
        print(f"\n  Session ended — {stats['duration_s']}s | {stats['events']} events")
        self.voice.stop()
        for f in self.feeds.values():
            f.release()
        cv2.destroyAllWindows()
        self.speech.stop()


if __name__ == "__main__":
    app = VisionaApp()
    app.run()
