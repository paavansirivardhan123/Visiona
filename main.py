"""
Visiona AI — Multi-camera assistive navigation system.

Architecture:
  4 CameraFeed threads → VisionSystem (YOLO + depth + track + speed + TTC)
  → DetectionPriorityQueue → grouping → SpeechEngine + AlertSystem
"""
import cv2
import time
import threading
import numpy as np
from typing import Dict, List, Optional

from core.config import Config
from core.detection import Detection
from core.priority_queue import DetectionPriorityQueue
from perception.vision import VisionSystem
from kinematics.heatmap import group_detections, build_speech_messages
from audio.alert import AlertSystem
from audio.speech import SpeechEngine
from audio.voice_input import VoiceInputEngine
from core.logger import SessionLogger

# Deep AI / LangChain Brains (Phase 6)
from agents.orchestrator import AgentEngine
from core.memory import memory_bank


class CameraFeed:
    """Manages a single camera/video source for one direction synchronously."""

    def __init__(self, direction: str, source):
        self.direction = direction
        self.cap: Optional[cv2.VideoCapture] = None
        self.active = False

        if source is None:
            return

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print(f"  [Camera] Could not open {direction} source: {source}")
            self.cap = None
            return

        self.active = True
        print(f"  [Camera] {direction} feed active")

    def get_frame(self):
        if not self.active or not self.cap:
            return None
        ret, frame = self.cap.read()
        if not ret:
            # Loop video infinitely
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return None
        return frame

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
        self.agent_engine = AgentEngine(
            tts_callback=lambda x: self.speech.speak(x, priority=True),
            search_intent_callback=self._on_intent
        )

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
        self._running      = True
        self._latest_frames = {}
        self._curiosity_cooldown = 0.0
        
        self._ai_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._ai_thread.start()

        self.voice = VoiceInputEngine(
            on_intent=self._on_intent,
            on_scene_request=self._on_scene,
            on_question=self._on_question,
            on_listening=self._on_listening,   # ducking wired here
        )
        self.voice.start()

        self.speech.speak("Visiona AI ready.")
        print("\n  Controls: P = describe scene | V = push-to-talk | ESC = quit")
        print("  Voice: 'find chair' | 'what is around' | ask anything\n")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        cv2.namedWindow("Visiona AI - Unified HUD", cv2.WINDOW_NORMAL)
        while self._running:
            self._frame_count += 1
            frames_to_render = {}

            # 1. Read frames synchronously to avoid staggered playback
            raw_frames = {}
            for direction, feed in self.feeds.items():
                if feed.active:
                    frame = feed.get_frame()
                    if frame is not None:
                        raw_frames[direction] = cv2.resize(frame, (Config.DISPLAY_W, Config.DISPLAY_H))

            if not raw_frames:
                break
                
            # Hand over frames to background AI thread
            self._latest_frames = raw_frames

            # 2. Draw overlay (Runs instantly, buttery smooth video)
            for direction, frame in raw_frames.items():
                self.vision.draw_overlay(
                    frame,
                    [d for d in self._all_dets if d.direction == direction],
                    self._state, self._last_info,
                )
                self._draw_extras(frame, direction)
                frames_to_render[direction] = frame

            # 3. Build Unified Grid & Show
            grid_frame = self._build_grid(frames_to_render)
            cv2.imshow("Visiona AI - Unified HUD", grid_frame)

            # Wait ~30ms to throttle video playback appropriately
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break
            elif key == ord('p') or key == ord('P'):
                self._on_scene()
            elif key == ord('v') or key == ord('V'):
                self.voice.trigger_ptt()

        self._shutdown()

    def _processing_loop(self):
        """Background thread evaluating YOLO+Depth at ~5-10 FPS without stalling video playback."""
        while self._running:
            start_t = time.time()
            frames = dict(self._latest_frames)
            
            if not frames:
                time.sleep(0.01)
                continue
                
            all_dets = []
            for direction, frame in frames.items():
                # We skip FRAME_SKIP manually here if we want, but since it's a background 
                # thread, we can just process every snapshot.
                dets = self.vision.detect(frame, direction)
                all_dets.extend(dets)
                self.logger.log_detections(dets, direction)
                
            if all_dets:
                self._all_dets = all_dets
                self._pipeline(all_dets)
            else:
                self._all_dets = []
                
            # Keep AI polling sensible
            elapsed = time.time() - start_t
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed)

    def _build_grid(self, frames: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combines up to 4 camera frames into a single unified window.
        Uses 2x2 grid if >=3 cameras, 1x2 if 2, 1x1 if 1.
        Positions: TOP-LEFT=FRONT, BOTTOM-LEFT=LEFT, TOP-RIGHT=RIGHT, BOTTOM-RIGHT=BACK.
        """
        count = len(frames)
        if count == 0:
            return np.zeros((Config.DISPLAY_H, Config.DISPLAY_W, 3), dtype=np.uint8)

        blank = np.zeros((Config.DISPLAY_H, Config.DISPLAY_W, 3), dtype=np.uint8)
        cv2.putText(blank, "NO SIGNAL", (Config.DISPLAY_W//2 - 60, Config.DISPLAY_H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2, cv2.LINE_AA)

        if count == 1:
            return list(frames.values())[0]
        elif count == 2:
            keys = list(frames.keys())
            grid = np.hstack((frames[keys[0]], frames[keys[1]]))
            return cv2.resize(grid, (Config.DISPLAY_W * 2, Config.DISPLAY_H))
        else: # 3 or 4 cameras
            front = frames.get("FRONT", blank)
            left  = frames.get("LEFT", blank)
            right = frames.get("RIGHT", blank)
            back  = frames.get("BACK", blank)
            
            top    = np.hstack((front, right))
            bottom = np.hstack((left, back))
            grid   = np.vstack((top, bottom))
            
            # Resize 4-grid to something manageable for a laptop screen
            target_w = int(Config.DISPLAY_W * 1.5)
            target_h = int(Config.DISPLAY_H * 1.5)
            return cv2.resize(grid, (target_w, target_h))

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
            
            # 6.1 Vision-Augmented Memory Push
            memory_bank.add_detections(messages)
            
            # 6.3 Contextual Curiosity (Proactive Agent Interruption)
            import time
            if time.time() - self._curiosity_cooldown > 30.0:
                for d in hp:  # Check high priority nearby objects
                    if d.label in ("chair", "car", "stop sign", "bench"):
                        import threading
                        anchor_ctx = f"Important object nearby: {d.label} at {d.distance_m}m."
                        threading.Thread(
                            target=self.agent_engine.process_voice_command, 
                            args=(f"You proactively noticed a {d.label}. Politely ask the user in one sentence if they need help interacting with or avoiding it.", anchor_ctx)
                        ).start()
                        self._curiosity_cooldown = time.time()
                        break
                        
            self.speech.speak_all(messages, first_priority=bool(hp))

        # Target seeking
        if self._search_intent:
            self._seek(sorted_dets)

    def _seek(self, detections: List[Detection]):
        matches = [d for d in detections if self._search_intent in d.label.lower()]
        if not matches:
            return   # keep searching silently
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

    def _on_listening(self, active: bool):
        """Called when PTT mic opens/closes — duck/unduck speech volume."""
        self._mic_active = active
        if active:
            self.speech.duck()    # drop to 30% so user can think and speak
        else:
            self.speech.unduck()  # restore to 100% for response

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

        # Always speak directly first — works even without API key
        self.speech.speak_all(messages)

        # Also route to agent for richer description if available
        if self.agent_engine.llm_with_tools:
            import threading
            raw_scene = ", ".join(messages)
            threading.Thread(
                target=self.agent_engine.process_voice_command,
                args=("Describe this scene in one natural sentence.", raw_scene),
                daemon=True
            ).start()

    def _on_question(self, question: str):
        print(f"  [Voice] Question: {question}")
        raw_scene = self._last_info if self._last_info else "Nothing detected right now."

        if self.agent_engine.llm_with_tools:
            import threading
            threading.Thread(
                target=self.agent_engine.process_voice_command,
                args=(question, raw_scene),
                daemon=True
            ).start()
        else:
            # Fallback: answer directly from current scene
            self.speech.speak(
                f"I can see: {raw_scene}" if self._last_info else "Nothing nearby right now.",
                priority=False
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
        self._running = False
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
