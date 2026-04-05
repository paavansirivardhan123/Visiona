from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class Detection:
    """
    Single detected object with full spatial + motion metadata.
    All new fields are Optional with None defaults for backward compatibility.
    """
    label:     str
    direction: str                        # FRONT / LEFT / RIGHT / BACK
    confidence: float
    box:       Tuple[int, int, int, int]  # x1, y1, x2, y2

    # Depth
    distance_m:       Optional[float] = None
    depth_mode:       str             = "unknown"   # "stereo" | "monocular" | "heuristic"

    # Tracking
    track_id: Optional[int] = None

    # Motion
    speed_mps: Optional[float] = None
    motion:    Optional[str]   = None   # approaching | moving_away | lateral | stationary
    ttc_sec:   Optional[float] = None

    # Priority (computed by priority queue)
    priority: int = 0

    @property
    def distance_ft(self) -> Optional[float]:
        if self.distance_m is None:
            return None
        return round(self.distance_m * 3.281, 1)
        
    @property
    def threat_score(self) -> float:
        from core.config import Config
        dist = self.distance_m if self.distance_m is not None else 10.0
        dist_factor = max(0.0, 10.0 - dist)
        
        weight = Config.OBJECT_PRIORITY.get(self.label.lower(), Config.DEFAULT_PRIORITY)
        score = dist_factor * weight
        
        if self.motion == "approaching" and self.speed_mps:
            score += (self.speed_mps * 20)  # Increased from 10 to 20
            
        if self.ttc_sec is not None and self.ttc_sec <= Config.TTC_WARN_THRESHOLD:
            score += 100.0  # Increased from 50 to 100 for high threat awareness
            
        return score

    @property
    def is_high_priority(self) -> bool:
        from core.config import Config
        return self.threat_score >= Config.THREAT_HIGH_THRESHOLD

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return (x2 - x1) * (y2 - y1)

    def to_record(self) -> dict:
        """Structured output record matching spec §19."""
        return {
            "object":      self.label,
            "direction":   self.direction,
            "mode":        self.depth_mode,
            "distance_m":  self.distance_m,
            "speed_mps":   self.speed_mps,
            "motion":      self.motion,
            "ttc_sec":     self.ttc_sec,
            "priority":    "high" if self.is_high_priority else (
                           "medium" if self.priority > 50 else "low"),
        }
