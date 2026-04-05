"""
AlertSystem — beep alerts scaled by proximity (spec §14).
Faster approach → faster beep rate via BEEP_LEVELS config.
"""
import threading
import time
import numpy as np
from typing import List
from core.detection import Detection
from core.config import Config

try:
    import winsound
    _WINSOUND = True
except ImportError:
    _WINSOUND = False


class AlertSystem:

    def __init__(self):
        self._last_beep = 0.0
        self._min_interval = 0.3   # seconds between beeps
        self._paused = False

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def process(self, detections: List[Detection]):
        if not _WINSOUND or not detections:
            return
            
        hp = [d for d in detections if d.is_high_priority and d.distance_m is not None]
        
        # If paused (user speaking), only allow high priority threats
        if self._paused and not hp:
            return
            
        if not hp:
            # Normal mode, no high priority detections
            return
            
        closest = min(hp, key=lambda d: d.distance_m)
        self._beep(closest.distance_m)

    def _beep(self, dist_m: float):
        now = time.time()
        if now - self._last_beep < self._min_interval:
            return
        self._last_beep = now
        freq, dur = self._params(dist_m)
        threading.Thread(target=winsound.Beep, args=(freq, dur), daemon=True).start()

    def _params(self, dist_m: float):
        for threshold, freq, dur in Config.BEEP_LEVELS:
            if dist_m <= threshold:
                return freq, dur
        return 800, 150
