"""
TTCCalculator — Time-To-Collision with EMA smoothing (spec §12).

TTC = distance / speed
Only computed when: motion == "approaching" AND speed > SPEED_MIN_THRESHOLD.
"""
import math
from typing import List, Optional
from core.detection import Detection
from core.config import Config


class TTCCalculator:

    def compute(
        self,
        distance_m: Optional[float],
        speed_mps: float,
        motion: Optional[str],
        prev_ttc: Optional[float],
    ) -> Optional[float]:
        if motion != "approaching":
            return None
        if speed_mps <= Config.SPEED_MIN_THRESHOLD:
            return None
        if distance_m is None or distance_m <= 0:
            return None

        raw = distance_m / speed_mps
        if raw <= 0 or not math.isfinite(raw):
            return None

        a = Config.TTC_SMOOTH_ALPHA
        smoothed = a * raw + (1.0 - a) * prev_ttc if prev_ttc is not None else raw
        return round(smoothed, 1)
