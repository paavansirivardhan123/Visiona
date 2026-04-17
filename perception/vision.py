"""
VisionSystem — YOLO + async depth pipeline for smooth video.

Architecture:
  - Main thread: YOLO on downscaled frame (fast, ~50ms)
  - Background thread: MiDaS depth (slow, ~300ms) — never blocks display
  - Depth results cached and reused until next background update
  - Display always runs at full speed
"""
import time
import threading
import cv2
import numpy as np
from typing import List, Optional
from ultralytics import YOLO
from core.config import Config
from core.detection import Detection
from perception.mono_depth import MonoDepth
from perception.tracker import ObjectTracker
from perception.egomotion import OpticalFlowEgomotion
from kinematics.speed import SpeedEstimator
from kinematics.ttc import TTCCalculator

# Downscale factor for YOLO — run on 1280px wide instead of 4K
_YOLO_WIDTH = 1280


class VisionSystem:

    STATE_COLORS = {
        "SCANNING": (0, 220, 80),
        "ALERT":    (0, 0, 255),
        "AVOIDING": (0, 140, 255),
        "GUIDING":  (255, 220, 0),
    }

    def __init__(self):
        print("  [Vision] Loading YOLO model...")
        self.model   = YOLO(Config.MODEL_PATH)
        self.depth   = MonoDepth()
        self.tracker = ObjectTracker()
        self.egomotion = OpticalFlowEgomotion()
        self.speed_e = SpeedEstimator()
        self.ttc_e   = TTCCalculator()
        self.user_state = "Stationary"
        self.user_speed = 0.0

        # Async depth state
        self._depth_lock    = threading.Lock()
        self._depth_running = False
        self._cached_depths: List[Optional[float]] = []
        self._cached_boxes:  List         = []
        self._cached_labels: List[str]    = []
        self._depth_frame:   Optional[np.ndarray] = None
        self._depth_ready    = False

        self._frame_count = 0
        self._last_detections: List[Detection] = []

        # Warm up MiDaS in background so it's ready quickly
        threading.Thread(target=self.depth._ensure_loaded, daemon=True).start()
        print(f"  [Vision] Ready: {Config.MODEL_PATH} (MiDaS loading in background...)")

    # ------------------------------------------------------------------
    # Detection pipeline
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray, direction: str) -> List[Detection]:
        """
        Fast path: YOLO on downscaled frame.
        Depth runs async in background — cached results used immediately.
        """
        self._frame_count += 1
        orig_h, orig_w = frame.shape[:2]

        # 1. Skip frames for efficiency if needed
        # (YOLO is fast, but we can still skip to save CPU for MiDaS)
        if self._frame_count % Config.FRAME_SKIP != 0:
            return self._last_detections

        # Downscale for YOLO (much faster than 4K)
        scale_factor = _YOLO_WIDTH / orig_w
        yolo_h = int(orig_h * scale_factor)
        yolo_frame = cv2.resize(frame, (_YOLO_WIDTH, yolo_h))

        # Scale factors back to display coords
        disp_sx = Config.DISPLAY_W / _YOLO_WIDTH
        disp_sy = Config.DISPLAY_H / yolo_h

        # 2. YOLO on downscaled frame
        raw = self.model(yolo_frame, imgsz=Config.IMG_SIZE, verbose=False, device='cpu') # Explicit CPU for low-end compatibility
        detections: List[Detection] = []
        yolo_boxes = []

        for r in raw:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < Config.CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self.model.names[int(box.cls[0])]
                # Scale to display coords
                dx1, dy1 = int(x1 * disp_sx), int(y1 * disp_sy)
                dx2, dy2 = int(x2 * disp_sx), int(y2 * disp_sy)
                detections.append(Detection(
                    label=label, direction=direction,
                    confidence=conf, box=(dx1, dy1, dx2, dy2),
                ))
                # Keep yolo-scale boxes for depth
                yolo_boxes.append((x1, y1, x2, y2))

        if not detections:
            self._last_detections = []
            return []

        # 2. Kick off async depth if not already running
        self._maybe_start_depth(yolo_frame, yolo_boxes,
                                [d.label for d in detections])

        # 3. Apply cached depth results
        with self._depth_lock:
            cached = list(self._cached_depths)

        if cached and len(cached) == len(detections):
            depths = cached
        else:
            # No depth yet — use heuristic fallback
            depths = self.depth._heuristic(
                yolo_boxes, [d.label for d in detections]
            )

        mode = "monocular"
        for det, dm in zip(detections, depths):
            det.distance_m = dm
            det.depth_mode = mode

        # 4. Filter by distance — keep objects within range OR with no depth yet
        detections = [
            d for d in detections
            if d.distance_m is None
            or d.distance_m <= Config.CONSIDER_MAX_M
        ]

        # 5. Track
        detections = self.tracker.update(detections, yolo_frame)

        # 5.5 Egomotion
        gray_frame = cv2.cvtColor(yolo_frame, cv2.COLOR_BGR2GRAY)
        ego_boxes = [(int(d.box[0]/disp_sx), int(d.box[1]/disp_sy), int(d.box[2]/disp_sx), int(d.box[3]/disp_sy)) for d in detections]
        self.user_speed, self.user_state = self.egomotion.update(gray_frame, ego_boxes, [d.label for d in detections])

        # 6. Speed + TTC
        tracks = self.tracker.active_tracks
        for det in detections:
            if det.track_id is None or det.track_id not in tracks:
                continue
            trk = tracks[det.track_id]
            spd, mot = self.speed_e.update(trk, self.user_speed, self.user_state)
            det.speed_mps = spd
            det.motion    = mot
            ttc = self.ttc_e.compute(det.distance_m, spd, mot, trk.prev_ttc)
            det.ttc_sec   = ttc
            trk.prev_ttc  = ttc

        self._last_detections = detections
        return detections

    def _maybe_start_depth(self, frame, boxes, labels):
        """Start async MiDaS depth if not already running."""
        with self._depth_lock:
            if self._depth_running:
                return
            self._depth_running = True

        def _run(depth_frame, depth_boxes, depth_labels):
            try:
                depths = self.depth.compute(
                    depth_frame, depth_boxes, depth_labels,
                    [None] * len(depth_boxes)
                )
                with self._depth_lock:
                    self._cached_depths = depths
                    self._depth_ready   = True
            except Exception as e:
                print(f"  [Vision] Depth error: {e}")
            finally:
                with self._depth_lock:
                    self._depth_running = False

        # Pass copies to thread to avoid closure issues or mutation
        import copy
        threading.Thread(
            target=_run, 
            args=(frame.copy(), copy.deepcopy(boxes), copy.deepcopy(labels)), 
            daemon=True
        ).start()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def draw_overlay(self, frame, detections: List[Detection],
                     state: str = "SCANNING", info: str = ""):
        self._draw_hud(frame, state, info, len(detections))
        self._draw_boxes(frame, detections)
        return frame

    def _draw_hud(self, frame, state, info, count):
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (Config.DISPLAY_W, Config.HUD_H),
                      (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)

        color = self.STATE_COLORS.get(state, (255, 255, 255))
        badge = f" {state} "
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (8, 8), (8 + tw + 4, 8 + th + 8), color, -1)
        cv2.putText(frame, badge, (10, 8 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, info[:80], (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        calib = "CAL" if self.depth.is_calibrated else "..."
        cv2.putText(frame, f"Obj:{count} {calib}",
                    (Config.DISPLAY_W - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, "Visiona AI",
                    (Config.DISPLAY_W - 100, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 70), 1, cv2.LINE_AA)

    def _draw_boxes(self, frame, detections: List[Detection]):
        for d in detections:
            if d.is_high_priority:
                color = (0, 0, 255)
            elif d.distance_m and d.distance_m < 2.0:
                color = (0, 140, 255)
            else:
                color = (0, 200, 80)

            x1, y1, x2, y2 = d.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            dist_s = f"{d.distance_m:.1f}m" if d.distance_m else "?"
            mot_s  = (f" {d.motion[:3]}"
                      if d.motion and d.motion != "stationary" else "")
            txt = f"{d.label} {dist_s}{mot_s}"
            (lw, lh), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 8),
                          (x1 + lw + 6, y1), color, -1)
            cv2.putText(frame, txt, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

            bar_w = int((x2 - x1) * d.confidence)
            cv2.rectangle(frame, (x1, y2),
                          (x1 + bar_w, y2 + 3), color, -1)

            if d.ttc_sec is not None and d.ttc_sec <= Config.TTC_WARN_THRESHOLD:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                r  = max(x2 - x1, y2 - y1) // 2 + 8
                cv2.circle(frame, (cx, cy), r, (0, 0, 255), 1, cv2.LINE_AA)
