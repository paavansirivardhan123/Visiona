"""
SpeedEstimator — per-track speed and motion classification (spec §10, §11).

speed = (previous_distance - current_distance) / delta_time
Motion: approaching | moving_away | lateral | stationary
"""
from collections import deque
from typing import Dict, Tuple
import numpy as np
from core.config import Config


class SpeedEstimator:

    def __init__(self):
        self._speed_history: Dict[int, deque] = {}

    def update(self, track) -> Tuple[float, str]:
        """
        Returns (speed_mps, motion_class) for a track.
        Requires at least 2 history entries.
        """
        h = track.history
        if len(h) < 2:
            return 0.0, "stationary"

        d1, pos1, t1 = h[-2]
        d2, pos2, t2 = h[-1]
        dt = t2 - t1
        if dt <= 0 or d1 is None or d2 is None:
            return 0.0, "stationary"

        delta_d   = d2 - d1
        raw_speed = abs(delta_d) / dt
        smoothed  = self._smooth(track.track_id, raw_speed)

        if smoothed < Config.SPEED_MIN_THRESHOLD:
            lateral = abs(pos2[0] - pos1[0])
            if lateral >= Config.LATERAL_THRESHOLD_PX:
                return 0.0, "lateral"
            return 0.0, "stationary"

        motion = "approaching" if delta_d < 0 else "moving_away"
        return round(smoothed, 2), motion

    def _smooth(self, tid: int, raw: float) -> float:
        if tid not in self._speed_history:
            self._speed_history[tid] = deque(maxlen=Config.SPEED_SMOOTH_FRAMES)
        self._speed_history[tid].append(raw)
        return float(np.mean(self._speed_history[tid]))
