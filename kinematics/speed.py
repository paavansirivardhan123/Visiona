"""
SpeedEstimator — per-track speed and motion classification (spec §10, §11).

speed = (previous_distance - current_distance) / delta_time
Motion: approaching | moving_away | lateral | stationary
"""
from collections import deque
from typing import List, Dict, Tuple
import numpy as np
from core.detection import Detection
from core.config import Config


class SpeedEstimator:

    def __init__(self):
        self._speed_history: Dict[int, deque] = {}

    STATIC_CLASSES = {
        "bench", "chair", "potted plant", "door", "table", "sofa", "bed", 
        "dining table", "tv", "stop sign", "traffic light", "fire hydrant", 
        "sink", "refrigerator", "oven", "microwave", "toilet", "vase", 
        "clock", "book", "bottle", "wine glass", "cup", "fork", "knife", 
        "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", 
        "carrot", "hot dog", "pizza", "donut", "cake", "laptop", "mouse", 
        "remote", "keyboard", "cell phone", "scissors", "teddy bear", 
        "hair drier", "toothbrush", "backpack", "umbrella", "handbag", 
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
        "tennis racket", "parking meter", "couch"
    }

    def update(self, track, egomotion_speed_mps: float = 0.0, egomotion_state: str = "Stationary") -> Tuple[float, str]:
        """
        Returns (speed_mps, motion_class) for a track.
        Requires at least 2 history entries.
        """
        h = track.history
        if len(h) < 2:
            return 0.0, "stationary"

        # Semantic Clamp: Architecturally static objects cannot move.
        if track.label.lower() in self.STATIC_CLASSES:
            return 0.0, "stationary"

        d1, pos1, t1 = h[-2]
        d2, pos2, t2 = h[-1]
        dt = t2 - t1
        if dt <= 0 or d1 is None or d2 is None:
            return 0.0, "stationary"

        delta_d   = d2 - d1
        raw_speed = abs(delta_d) / dt
        smoothed  = self._smooth(track.track_id, raw_speed)

        # Convert to algebraic velocity: negative means approaching, positive means moving away
        relative_vel = delta_d / dt
        
        # When user walks forward, egomotion > 0. A stationary car appears to approach (relative_vel < 0)
        # Adding egomotion cancels it: true_vel = (-1) + (+1) = 0
        true_vel = relative_vel + egomotion_speed_mps

        lateral = abs(pos2[0] - pos1[0])
        is_panning = "Panning" in egomotion_state

        if abs(true_vel) < Config.SPEED_MIN_THRESHOLD:
            if lateral >= Config.LATERAL_THRESHOLD_PX and not is_panning:
                return 0.0, "lateral"
            return 0.0, "stationary"

        # Guard against minor depth fluctuations making objects appear to be jumping
        if abs(true_vel) < 0.5:
             if lateral >= Config.LATERAL_THRESHOLD_PX and not is_panning:
                 return 0.0, "lateral"
             return 0.0, "stationary"

        motion = "approaching" if true_vel < 0 else "moving_away"
        return round(abs(true_vel), 2), motion

    def _smooth(self, tid: int, raw: float) -> float:
        if tid not in self._speed_history:
            self._speed_history[tid] = deque(maxlen=Config.SPEED_SMOOTH_FRAMES)
        self._speed_history[tid].append(raw)
        return float(np.mean(self._speed_history[tid]))
