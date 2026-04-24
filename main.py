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
from core.recognition import feature_db


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
            tts_callback=lambda x: self.speech.speak(x, priority=True, bypass_cooldown=True, emergency=True),
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
        self._mic_mode     = "LLM"           # "LLM", "MAPS", "REMEMBER"
        self._search_intent: Optional[str] = None
        self._running      = True
        self._latest_frames = {}
        self._curiosity_cooldown = 0.0

        self._capture_alias = None
        self._capture_base_class = None
        self._capture_count = 0
        import os
        os.makedirs("database", exist_ok=True)
        
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
                elif key.char.lower() == 'r':
                    if not self._mic_active:
                        self._mic_mode = "REMEMBER"
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
                elif key.char.lower() == 'r' and self._mic_mode == "REMEMBER":
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
                # Check for cached identity memory (Label Swapping)
                fh, fw = frame.shape[:2]
                for d in dets:
                    # we only try to identify things if they are reasonably large
                    x1, y1, x2, y2 = d.box
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(fw, x2), min(fh, y2)
                    w, h = x2 - x1, y2 - y1
                    if w > 40 and h > 40: # Minimum crop size for decent features
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            matched_alias = feature_db.match(crop)
                            if matched_alias:
                                d.base_label = d.label           # Preserve for threat fallback
                                d.label = matched_alias

                all_dets.extend(dets)
                self.logger.log_detections(dets, direction)

                # Capture logic
                if self._capture_count > 0 and self._capture_alias and self._capture_base_class and dets:
                    # Filter elements by base class
                    target_dets = [d for d in dets if d.label.lower() == self._capture_base_class.lower()]
                    
                    if target_dets:
                        # Pick the most threatening/prominent object among TARGET objects
                        best_det = max(target_dets, key=lambda x: x.threat_score)
                        x1, y1, x2, y2 = best_det.box
                        fh, fw = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(fw, x2), min(fh, y2)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            import os
                            folder = os.path.join("database", self._capture_alias)
                            os.makedirs(folder, exist_ok=True)
                            filepath = os.path.join(folder, f"img_{50 - self._capture_count}.jpg")
                            cv2.imwrite(filepath, crop)
                            self._capture_count -= 1
                            if self._capture_count == 0:
                                print(f"  [Capture] Finished capturing 50 frames for {self._capture_alias}")
                                self.speech.speak(f"Finished capturing images for {self._capture_alias}. Connecting memory bank.", bypass_cooldown=True, emergency=True)
                                # Load it into VectorDB Memory live
                                feature_db.load_alias(self._capture_alias)
                                self._capture_alias = None
                                self._capture_base_class = None
                
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

        # Filter for ambient alerts based on user behavior:
        # Non-front detections are silent EXCEPT large approaching vehicles
        def _is_ambient_reportable(det: Detection) -> bool:
            if det.direction == "FRONT":
                return True
            base_l = getattr(det, 'base_label', det.label).lower()
            return base_l in ("car", "truck", "bus", "motorcycle", "bicycle") and getattr(det, 'motion', None) == "approaching"
            
        def _is_threat(det: Detection) -> bool:
            base_l = getattr(det, 'base_label', det.label).lower()
            if (det.ttc_sec is not None and det.ttc_sec <= Config.TTC_WARN_THRESHOLD): return True
            if (det.threat_score > Config.THREAT_HIGH_THRESHOLD): return True
            if base_l in ("car", "truck", "bus", "motorcycle", "bicycle") and getattr(det, 'motion', None) == "approaching": return True
            return False

        if not hasattr(self, '_ambient_states'):
            self._ambient_states = {}
            
        import time
        now = time.time()
        
        raw_ambient_dets = [d for d in sorted_dets if _is_ambient_reportable(d)]
        ambient_dets = []
        allowed_labels_this_frame = set()
        
        for d in raw_ambient_dets:
            if _is_threat(d):
                ambient_dets.append(d)
                continue
                
            label_lower = d.label.lower()
            dir_str = d.direction
            
            if label_lower not in self._ambient_states:
                self._ambient_states[label_lower] = {}
                
            key = f"{label_lower}_{dir_str}"
            if key in allowed_labels_this_frame:
                # Already allowed one of these in this exact frame, let the others through to preserve group counting
                ambient_dets.append(d)
                continue
                
            if now - self._ambient_states[label_lower].get(dir_str, 0.0) >= 3.0:
                ambient_dets.append(d)
                self._ambient_states[label_lower][dir_str] = now
                allowed_labels_this_frame.add(key)
                
        ambient_hp = [d for d in ambient_dets if d.is_high_priority]

        # 2.5 Extreme Threat Detection (Emergency Verbal Bypass)
        # If a large/fast object is very close, speak it IMMEDIATELY even if user is talking
        for d in ambient_hp:
            base_l = getattr(d, 'base_label', d.label).lower()
            dir_s = {"FRONT": "ahead", "LEFT": "on the left", "RIGHT": "on the right", "BACK": "behind you"}.get(d.direction, "nearby")
            if base_l in ("truck", "bus", "car", "motorcycle", "bicycle") and d.distance_m and d.distance_m < 2.5:
                self.speech.speak(f"Emergency: {d.label} {dir_s}!", emergency=True)
            elif base_l == "person" and d.distance_m and d.distance_m < 1.0:
                self.speech.speak(f"{d.label} very close {dir_s}!", emergency=True)

        # 3. Beep (High Priority Alerts)
        self.alert.process(ambient_hp)

        # 4. Speech Summaries
        grouped  = group_detections(ambient_dets)
        messages = build_speech_messages(grouped, ambient_hp)
        
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
                for d in ambient_hp:
                    if d.label in ("chair", "car", "stop sign", "bench"):
                        import threading
                        steps = max(1, round(d.distance_m / Config.METERS_PER_STEP)) if d.distance_m else 2
                        anchor_ctx = f"Important object nearby: {d.label} at {steps} steps."
                        threading.Thread(
                            target=self.agent_engine.process_voice_command, 
                            args=(f"You proactively noticed a {d.label}. Politely ask the user in one sentence if they need help interacting with or avoiding it.", anchor_ctx),
                            daemon=True
                        ).start()
                        self._curiosity_cooldown = time.time()
                        break
                        
            self.speech.speak_all(messages, first_priority=bool(ambient_hp))

    def _evaluate_target(self, det: Detection, label: str) -> str:
        """Returns action: 'IGNORE' or 'NEW_CLOSER_DOUBLE'."""
        if not hasattr(self, '_target_states'):
            self._target_states = {}
            
        label = label.lower()
        import time
        now = time.time()
        
        d_val = det.distance_m if det.distance_m is not None else 999.0
        steps = max(1, round(d_val / Config.METERS_PER_STEP)) if d_val != 999.0 else 999
        direction = det.direction
        
        if label not in self._target_states:
            self._target_states[label] = {
                "global_min_steps": 999,
                "directions": {}
            }
            
        state = self._target_states[label]
        
        if direction not in state["directions"]:
            state["directions"][direction] = {
                "last_time": 0.0
            }
            
        dir_state = state["directions"][direction]
        
        # 1. 5-second gap for the same panel
        time_since_last = now - dir_state["last_time"]
        if time_since_last < 5.0:
            return "IGNORE"
            
        # 2. Shorter distance check (using physical steps to prevent noise spam)
        if steps < state.get("global_min_steps", 999):
            state["global_min_steps"] = steps
            dir_state["last_time"] = now
            return "NEW_CLOSER_DOUBLE"
            
        # 3. Same or greater distance -> DO NOT RECOMMEND
        return "IGNORE"

    def _trigger_contextual_arrival(self, label: str, dir_s: str):
        if self.agent_engine and self.agent_engine.llm:
            import threading
            from langchain_core.messages import HumanMessage
            
            def _ask_llm():
                prompt = (
                    f"The user is blind and has just arrived right next to a '{label}' ({dir_s}). "
                    "Write exactly ONE comforting, concise sentence telling them where it is, "
                    "suggesting what they can logically do with it, and asking if they want to set a new goal. "
                    "Do not use markdown."
                )
                try:
                    resp = self.agent_engine.llm.invoke([HumanMessage(content=prompt)])
                    if self.speech:
                        self.speech.speak(resp.content, priority=True, bypass_cooldown=True, emergency=True)
                except Exception as e:
                    print(f"  [Agent] Contextual fallback needed: {e}")
                    if self.speech:
                        self.speech.speak(f"You are next to the {label}. Do you want to set a new goal?", priority=True, bypass_cooldown=True, emergency=True)
            
            threading.Thread(target=_ask_llm, daemon=True).start()
        else:
            self.speech.speak(f"You have arrived at the {label}. Do you want to set a new goal?", priority=True, bypass_cooldown=True, emergency=True)

    def _match_goals(self, detections: List[Detection]):
        """Proactively checks if any detected objects match a persistent user goal."""
        from core.memory import goal_system
        candidates = goal_system.get_active_candidates()
        if not candidates:
            return

        # Enforce shortest distance precedence to prioritize side objects closer than front ones
        sorted_dets = sorted(detections, key=lambda x: x.distance_m if x.distance_m is not None else 999.0)

        for d in sorted_dets:
            label_lower = d.label.lower()
            if label_lower in [c.lower() for c in candidates]:
                is_very_near = d.distance_m is not None and d.distance_m < 0.5
                dir_s = {"FRONT": "directly in front of you", "LEFT": "right beside you on the left", "RIGHT": "right beside you on the right", "BACK": "right behind you"}.get(d.direction, "nearby")
                
                if is_very_near:
                    # Successfully reached the persistent goal. Complete it and announce contextually.
                    goal_system.complete_goal(label_lower)
                    self._trigger_contextual_arrival(label_lower, dir_s)
                    break
                    
                eval_cmd = self._evaluate_target(d, label_lower)
                if eval_cmd == "NEW_CLOSER_DOUBLE":
                    # We found a goal-related object!
                    if d.distance_m:
                        steps = max(1, round(d.distance_m / Config.METERS_PER_STEP))
                        dist = f"{steps} step{'s' if steps > 1 else ''}"
                    else:
                        dist = "nearby"
                    speak_dir = {"FRONT": "in front", "LEFT": "on the left", "RIGHT": "on the right", "BACK": "behind you"}.get(d.direction, "")
                    msg = f"Found a {label_lower}, {dist} {speak_dir}."
                    
                    # Speak it immediately with high priority
                    self.speech.speak(msg, priority=True, bypass_cooldown=True)
                    
                    # Trigger second announcement after 2 seconds
                    import threading
                    threading.Timer(2.0, self.speech.speak, args=(msg,), kwargs={"priority": True, "bypass_cooldown": True}).start()
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
                # Explicitly sort matches by minimum distance to ALWAYS select the closest object
                matches.sort(key=lambda x: x.distance_m if x.distance_m is not None else 999.0)
                t = matches[0]
                
                is_very_near = t.distance_m is not None and t.distance_m < 0.5
                dir_s = {"FRONT": "directly in front of you", "LEFT": "right beside you on the left",
                         "RIGHT": "right beside you on the right", "BACK": "right behind you"}.get(t.direction, "nearby")
                
                if is_very_near:
                    # Remove this intent from the list since we have successfully found it up close
                    if intent in self._search_intents:
                        self._search_intents.remove(intent)
                    if intent == self._search_intent:
                        self._search_intent = None
                    self._state = "GUIDING"
                    
                    self._trigger_contextual_arrival(t.label, dir_s)
                    return  # Stop searching after completion
                    
                eval_cmd = self._evaluate_target(t, intent)
                if eval_cmd == "NEW_CLOSER_DOUBLE":
                    if t.distance_m:
                        steps = max(1, round(t.distance_m / Config.METERS_PER_STEP))
                        dist = f"{steps} step{'s' if steps > 1 else ''}"
                    else:
                        dist = "nearby"
                    speak_dir = {"FRONT": "in front", "LEFT": "on the left",
                             "RIGHT": "on the right", "BACK": "behind you"}.get(t.direction, "")
                    msg = f"Found {t.label}. {dist} {speak_dir}."
                    self.speech.speak(msg, priority=True, bypass_cooldown=True)
                    
                    # Trigger second announcement after 2 seconds
                    import threading
                    threading.Timer(2.0, self.speech.speak, args=(msg,), kwargs={"priority": True, "bypass_cooldown": True}).start()
                    
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
            
        if self._mic_mode == "REMEMBER":
            import threading
            def _extract_and_capture():
                if self.agent_engine.llm:
                    data = self.agent_engine.extract_memory_label(text)
                    if data and isinstance(data, dict):
                        alias = data.get("alias")
                        base_class = data.get("base_class")
                        if alias and alias.lower() != "none" and alias.lower() != "unknown" and base_class:
                            self.speech.speak(f"Okay, I will memorize {alias} now. Keep looking at it.", bypass_cooldown=True, emergency=True)
                            import time
                            time.sleep(2) # Give user a sec to point the camera
                            self._capture_alias = alias
                            self._capture_base_class = base_class
                            self._capture_count = 50
                        else:
                            self.speech.speak("I didn't catch what you wanted me to remember.", bypass_cooldown=True, emergency=True)
                    else:
                        self.speech.speak("I didn't catch what you wanted me to remember.", bypass_cooldown=True, emergency=True)
                else:
                    self.speech.speak("The reasoning engine is offline.", bypass_cooldown=True, emergency=True)
            threading.Thread(target=_extract_and_capture, daemon=True).start()
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
            # Sort by distance so the absolute closest matching object is announced first
            sorted_dets = sorted(self._all_dets, key=lambda x: x.distance_m if x.distance_m is not None else 999.0)
            for d in sorted_dets:
                if intent.lower() in d.label.lower():
                    # Found it immediately!
                    if d.distance_m:
                        steps = max(1, round(d.distance_m / Config.METERS_PER_STEP))
                        dist = f"{steps} step{'s' if steps > 1 else ''}"
                    else:
                        dist = "nearby"
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
        # Get egomotion context
        motion_ctx = f"User State: {getattr(self.vision, 'user_state', 'Stationary')} ({getattr(self.vision, 'user_speed', 0.0):.1f} m/s)."
        
        if not self._all_dets:
            return f"{motion_ctx} No objects currently detected in view."
        
        ctx_parts = []
        # Sort by distance so the closest objects are always processed first
        sorted_dets = sorted(self._all_dets, key=lambda x: x.distance_m if x.distance_m is not None else 999.0)
        
        for d in sorted_dets:
            if d.distance_m:
                steps = max(1, round(d.distance_m / Config.METERS_PER_STEP))
                dist = f"{steps} steps"
            else:
                dist = "unknown distance"
            dir_s = {"FRONT": "in front", "LEFT": "on the left", 
                     "RIGHT": "on the right", "BACK": "behind you"}.get(d.direction, d.direction.lower())
            
            # Highlight true independent motion
            mot_s = f"({d.motion})" if getattr(d, 'motion', None) else ""
            ctx_parts.append(f"{d.label} at {dist} {dir_s} {mot_s}".strip())
        
        return f"{motion_ctx} Objects: " + " | ".join(ctx_parts)

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
