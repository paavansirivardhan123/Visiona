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
    is_high_priority: bool            = False

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
