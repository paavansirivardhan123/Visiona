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
from pynput import keyboard

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
        self.source = source

        if source is None:
            return

        self._open_capture(source)

    def _open_capture(self, source):
        try:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                print(f"  [Camera] Could not open {self.direction} source: {source}")
                # Fallback: if source is an index and failed, try 0 as last resort
                if isinstance(source, int) and source != 0:
                    print(f"  [Camera] Attempting fallback to device 0 for {self.direction}...")
                    self.cap = cv2.VideoCapture(0)
                
                if not self.cap or not self.cap.isOpened():
                    self.cap = None
                    return

            self.active = True
            print(f"  [Camera] {self.direction} feed active (Source: {source})")
        except Exception as e:
            print(f"  [Camera] Error opening {self.direction}: {e}")
            self.cap = None

    def get_frame(self):
        if not self.active or not self.cap:
            return None
        try:
            ret, frame = self.cap.read()
            if not ret:
                # Loop video infinitely if it's a file
                if isinstance(self.source, str) and not self.source.isdigit():
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                
                if not ret:
                    # If still fails, maybe camera disconnected
                    self.active = False
                    return None
            return frame
        except Exception as e:
            print(f"  [Camera] Runtime error reading {self.direction}: {e}")
            return None

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
            tts_callback=lambda x: self.speech.speak(x, priority=True, bypass_cooldown=True),
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
        self._mic_mode     = "LLM"           # "LLM" or "MAPS"
        self._search_intent: Optional[str] = None
        self._running      = True
        self._latest_frames = {}
        self._curiosity_cooldown = 0.0
        
        self._ai_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._ai_thread.start()

        self.voice = VoiceInputEngine(
            on_speech=self._on_speech,
            on_listening=self._on_listening,
        )
        # Start voice engine immediately (it just calibrates in PTT mode)
        self.voice.start()

        self.speech.speak("Visiona AI ready.")
        print("\n  [System] Visiona AI Initialized.")
        print("  [System] V (Hold) = General AI reasoning / needs.")
        print("  [System] G (Hold) = Dedicated GPS Navigation.")
        print("  [System] Controls: ESC = quit\n")

        # Keyboard listener for PTT
        self._keyboard_listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._keyboard_listener.start()

    def _on_press(self, key):
        try:
            if hasattr(key, 'char'):
                if key.char == 'v':
                    if not self._mic_active:
                        self._mic_mode = "LLM"
                        self.voice.start_recording()
                elif key.char == 'g':
                    if not self._mic_active:
                        self._mic_mode = "MAPS"
                        self.voice.start_recording()
        except AttributeError:
            pass

    def _on_release(self, key):
        try:
            if hasattr(key, 'char'):
                if key.char == 'v' and self._mic_mode == "LLM":
                    self.voice.stop_recording()
                elif key.char == 'g' and self._mic_mode == "MAPS":
                    self.voice.stop_recording()
        except AttributeError:
            pass

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
            self._latest_frames = raw_frames.copy()

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

        # 1. State & Threat Level
        if hp:
            self._state = "ALERT"
        elif any(d.distance_m and d.distance_m < 3.0 for d in sorted_dets):
            self._state = "AVOIDING"
        else:
            self._state = "SCANNING"

        # 2. Busy Road Detection (Vehicle Density)
        vehicles = [d for d in detections if d.label in ("car", "truck", "bus", "motorcycle")]
        if len(vehicles) >= 5:
            self.speech.speak("Busy road detected. Use caution while navigating.", priority=True)
            self._state = "CAUTION"

        # 2.5 Extreme Threat Detection (Emergency Verbal Bypass)
        # If a large/fast object is very close, speak it IMMEDIATELY even if user is talking
        for d in hp:
            if d.label in ("truck", "bus", "car") and d.distance_m and d.distance_m < 2.5:
                self.speech.speak(f"Emergency: {d.label} ahead!", emergency=True)
            elif d.label == "person" and d.distance_m and d.distance_m < 1.0:
                self.speech.speak("Person very close!", emergency=True)

        # 3. Beep (High Priority Alerts)
        self.alert.process(hp)

        # 4. Speech Summaries
        grouped  = group_detections(sorted_dets)
        messages = build_speech_messages(grouped, hp)
        
        # 5. Vision-Augmented Memory Push
        full_context = self._get_full_spatial_context()
        memory_bank.add_detections([full_context])

        # 6. Persistent Goal & Proactive Reasoning
        self._match_goals(sorted_dets)

        if messages:
            self._last_info = " | ".join(messages)
            self.logger.log_speech(messages)
            
            # 6.3 Contextual Curiosity
            import time
            if time.time() - self._curiosity_cooldown > 30.0:
                for d in hp:
                    if d.label in ("chair", "car", "stop sign", "bench"):
                        import threading
                        anchor_ctx = f"Important object nearby: {d.label} at {d.distance_m}m."
                        threading.Thread(
                            target=self.agent_engine.process_voice_command, 
                            args=(f"You proactively noticed a {d.label}. Politely ask the user in one sentence if they need help interacting with or avoiding it.", anchor_ctx),
                            daemon=True
                        ).start()
                        self._curiosity_cooldown = time.time()
                        break
                        
            self.speech.speak_all(messages, first_priority=bool(hp))

    def _match_goals(self, detections: List[Detection]):
        """Proactively checks if any detected objects match a persistent user goal."""
        from core.memory import goal_system
        candidates = goal_system.get_active_candidates()
        if not candidates:
            return

        # Track which goals we've announced recently to avoid spam
        if not hasattr(self, '_announced_goals'):
            self._announced_goals = {}
        
        import time
        now = time.time()

        for d in detections:
            label_lower = d.label.lower()
            if label_lower in [c.lower() for c in candidates]:
                # Check if we've announced this specific object recently
                announce_key = f"{label_lower}_{d.direction}"
                last_announce = self._announced_goals.get(announce_key, 0)
                
                # Only announce every 10 seconds for the same object type in same direction
                if now - last_announce < 10.0:
                    continue
                
                # We found a goal-related object!
                dist = f"{d.distance_ft} feet" if d.distance_ft else "nearby"
                dir_s = {"FRONT": "in front", "LEFT": "on the left", "RIGHT": "on the right", "BACK": "behind you"}.get(d.direction, "")
                msg = f"Found a {label_lower}, {dist} {dir_s}."
                
                # Speak it immediately with high priority
                self.speech.speak(msg, priority=True, bypass_cooldown=True)
                self._announced_goals[announce_key] = now
                break

        # Target seeking (for explicit search commands)
        if self._search_intent or (hasattr(self, '_search_intents') and self._search_intents):
            self._seek(detections)

    def _seek(self, detections: List[Detection]):
        """Check if any detected objects match active search intents."""
        if not hasattr(self, '_search_intents'):
            self._search_intents = []
            
        # Check all search intents, not just one
        all_intents = self._search_intents + ([self._search_intent] if self._search_intent else [])
        
        for intent in all_intents:
            if not intent:
                continue
            matches = [d for d in detections if intent.lower() in d.label.lower()]
            if matches:
                t = matches[0]
                dist = f"{t.distance_ft} feet" if t.distance_ft else "nearby"
                dir_s = {"FRONT": "in front", "LEFT": "on the left",
                         "RIGHT": "on the right", "BACK": "behind you"}.get(t.direction, "")
                self.speech.speak(f"Found {t.label}. {dist} {dir_s}.", priority=True, bypass_cooldown=True)
                
                # Remove this intent from the list
                if intent in self._search_intents:
                    self._search_intents.remove(intent)
                if intent == self._search_intent:
                    self._search_intent = None
                    
                self._state = "GUIDING"
                return  # Announce one at a time

    # ------------------------------------------------------------------
    # Voice callbacks
    # ------------------------------------------------------------------

    def _on_listening(self, active: bool):
        """Called when PTT or continuous mic opens/closes."""
        self._mic_active = active
        if active:
            self.speech.duck()    # stops and clears speech
            self.alert.pause()    # stops beep alerts
        else:
            # Immediately resume when key is released
            # Voice processing happens in background thread
            self.speech.unduck()
            self.alert.resume()

    def _on_speech(self, text: str):
        """Called when any speech is captured. Routes to LLM or Maps based on mode."""
        print(f"  [Voice] Received Speech ({self._mic_mode}): \"{text}\"")
        full_context = self._get_full_spatial_context()
        
        if self._mic_mode == "MAPS":
            # Force navigation intent for G key
            if self.agent_engine.llm_with_tools:
                import threading
                threading.Thread(
                    target=self.agent_engine.process_voice_command,
                    args=(f"I need walking directions to: {text}", full_context),
                    daemon=True
                ).start()
            return

        if self.agent_engine.llm_with_tools:
            import threading
            threading.Thread(
                target=self.agent_engine.process_voice_command,
                args=(text, full_context),
                daemon=True
            ).start()
        else:
            # Fallback for offline mode
            self.speech.speak(f"I heard you say: {text}. Reasoning engine is offline.", bypass_cooldown=True)

    def _on_intent(self, intent: str):
        """Called when agent triggers a search. Stores intent for matching in _seek()."""
        print(f"  [Voice] Tool Triggered Search: {intent}")
        
        # First check if the object is already visible in current detections
        if self._all_dets:
            for d in self._all_dets:
                if intent.lower() in d.label.lower():
                    # Found it immediately!
                    dist = f"{d.distance_ft} feet" if d.distance_ft else "nearby"
                    dir_s = {"FRONT": "in front", "LEFT": "on the left",
                             "RIGHT": "on the right", "BACK": "behind you"}.get(d.direction, "")
                    self.speech.speak(f"There's a {d.label} {dist} {dir_s}.", priority=True, bypass_cooldown=True)
                    return  # Don't add to search list since we found it
        
        # Not currently visible, add to search list
        if not hasattr(self, '_search_intents'):
            self._search_intents = []
        if intent not in self._search_intents:
            self._search_intents.append(intent)
        self._search_intent = intent  # Keep for backward compatibility
        self._state = "SEARCHING"


    def _get_full_spatial_context(self) -> str:
        """Returns a detailed list of all current detections for the AI reasoning engine."""
        if not self._all_dets:
            return "No objects currently detected in view."
        
        ctx_parts = []
        for d in self._all_dets:
            dist = f"{d.distance_m:.1f}m" if d.distance_m else "unknown distance"
            dir_s = {"FRONT": "in front", "LEFT": "on the left", 
                     "RIGHT": "on the right", "BACK": "behind you"}.get(d.direction, d.direction.lower())
            ctx_parts.append(f"{d.label} at {dist} {dir_s}")
        
        return " | ".join(ctx_parts)

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
        print("\n  [System] Shutting down Visiona AI...")
        stats = self.logger.get_stats()
        print(f"  [System] Session Statistics:")
        print(f"           - Duration: {stats['duration_s']}s")
        print(f"           - Total Events: {stats['events']}")
        print(f"           - Logs saved to: {self.logger._path if hasattr(self.logger, '_path') else 'N/A'}")
        
        self.voice.stop()
        for f in self.feeds.values():
            f.release()
        cv2.destroyAllWindows()
        self.speech.stop()
        print("  [System] Shutdown complete. Stay safe.")


if __name__ == "__main__":
    app = VisionaApp()
    app.run()
