import cv2
import numpy as np
from typing import List
from ultralytics import YOLO
from core.config import Config
from models.detection import Detection
from engines.depth import DepthEstimator

class VisionSystem:
    """
    Enhanced vision system with:
    - 5-zone spatial grid
    - Real distance estimation
    - Confidence scoring
    - Hazard/target classification
    - Rich HUD overlay
    """

    ZONE_COLORS = {
        "far-left":  (255, 200, 0),
        "left":      (0, 200, 255),
        "ahead":     (0, 80, 255),
        "right":     (0, 200, 255),
        "far-right": (255, 200, 0),
    }

    STATE_COLORS = {
        "SCANNING":  (0, 220, 80),
        "ALERT":     (0, 0, 255),
        "AVOIDING":  (0, 140, 255),
        "GUIDING":   (255, 220, 0),
        "SEARCHING": (200, 0, 255),
    }

    def __init__(self):
        self.model = YOLO("yolov8s.pt")
        self.depth = DepthEstimator()
        cv2.namedWindow(Config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(Config.WINDOW_NAME, Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT)

    def _get_zone(self, center_x: int) -> str:
        w = Config.DISPLAY_WIDTH
        if center_x < w * 0.15:   return "far-left"
        if center_x < w * 0.38:   return "left"
        if center_x < w * 0.62:   return "ahead"
        if center_x < w * 0.85:   return "right"
        return "far-right"

    def detect(self, frame) -> List[Detection]:
        results = self.model(frame, imgsz=Config.IMG_SIZE, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < Config.CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self.model.names[int(box.cls[0])]
                center_x = (x1 + x2) // 2
                zone = self._get_zone(center_x)
                area = (x2 - x1) * (y2 - y1)
                dist = self.depth.estimate(label, (x1, y1, x2, y2))

                detections.append(Detection(
                    label=label,
                    position=zone,
                    area=area,
                    box=(x1, y1, x2, y2),
                    confidence=conf,
                    distance_cm=dist,
                    is_hazard=label in Config.HAZARD_LABELS,
                    is_target=label in Config.TARGET_LABELS,
                ))
        # Sort by distance (closest first)
        detections.sort(key=lambda d: d.distance_cm or 9999)
        return detections

    def draw_overlay(self, frame, detections: List[Detection], agent):
        self._draw_zone_grid(frame)
        self._draw_hud(frame, agent)
        self._draw_detections(frame, detections)

    def _draw_zone_grid(self, frame):
        """Draw subtle vertical zone dividers."""
        w = Config.DISPLAY_WIDTH
        boundaries = [int(w * p) for p in [0.15, 0.38, 0.62, 0.85]]
        for x in boundaries:
            cv2.line(frame, (x, Config.HUD_HEIGHT), (x, Config.DISPLAY_HEIGHT),
                     (60, 60, 60), 1, cv2.LINE_AA)

    def _draw_hud(self, frame, agent):
        """Top HUD bar with state, reasoning, and stats."""
        # Semi-transparent dark bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (Config.DISPLAY_WIDTH, Config.HUD_HEIGHT), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        state_color = self.STATE_COLORS.get(agent.current_state, (255, 255, 255))

        # State badge
        badge_text = f" {agent.current_state} "
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (8, 8), (8 + tw + 4, 8 + th + 8), state_color, -1)
        cv2.putText(frame, badge_text, (10, 8 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        # Intent
        cv2.putText(frame, f"INTENT: {agent.intent.upper()}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # AI reasoning (truncated)
        reasoning = agent.reasoning[:72] + "..." if len(agent.reasoning) > 72 else agent.reasoning
        cv2.putText(frame, reasoning,
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        # Object count top-right
        count_text = f"Objects: {len(agent.last_detections)}"
        cv2.putText(frame, count_text,
                    (Config.DISPLAY_WIDTH - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        # Branding
        cv2.putText(frame, "NaVision AI",
                    (Config.DISPLAY_WIDTH - 130, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)

    def _draw_detections(self, frame, detections: List[Detection]):
        for d in detections:
            # Color by distance zone
            if d.distance_cm and d.distance_cm < Config.DIST_CRITICAL:
                color = (0, 0, 255)       # red — critical
            elif d.distance_cm and d.distance_cm < Config.DIST_NEAR:
                color = (0, 140, 255)     # orange — near
            elif d.is_hazard:
                color = (0, 200, 255)     # yellow — hazard but medium
            elif d.is_target:
                color = (0, 255, 120)     # green — target object
            else:
                color = (160, 160, 160)   # grey — neutral

            x1, y1, x2, y2 = d.box
            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Label chip
            dist_str = f"{int(d.distance_cm)}cm" if d.distance_cm else ""
            label_text = f"{d.label} {dist_str}"
            (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), color, -1)
            cv2.putText(frame, label_text, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            # Confidence bar (small, below label)
            bar_w = int((x2 - x1) * d.confidence)
            cv2.rectangle(frame, (x1, y2), (x1 + bar_w, y2 + 4), color, -1)

            # Hazard pulse ring
            if d.is_hazard and d.distance_cm and d.distance_cm < Config.DIST_NEAR:
                cx, cy = d.center_x, d.center_y
                radius = max((x2 - x1), (y2 - y1)) // 2 + 10
                cv2.circle(frame, (cx, cy), radius, (0, 0, 255), 1, cv2.LINE_AA)
