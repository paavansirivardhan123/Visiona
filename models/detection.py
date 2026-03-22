from dataclasses import dataclass, field
from typing import Tuple, Optional

@dataclass
class Detection:
    """Enriched detection with distance, confidence, and tracking."""
    label: str
    position: str                           # spatial zone: far-left/left/ahead/right/far-right
    area: int                               # bounding box pixel area
    box: Tuple[int, int, int, int]          # x1, y1, x2, y2
    confidence: float = 0.0
    distance_cm: Optional[float] = None    # estimated real-world distance
    track_id: Optional[int] = None         # persistent tracking ID
    is_hazard: bool = False
    is_target: bool = False

    @property
    def distance_label(self) -> str:
        if self.distance_cm is None:
            return "unknown"
        if self.distance_cm < 80:
            return "critical"
        if self.distance_cm < 150:
            return "near"
        if self.distance_cm < 300:
            return "medium"
        return "far"

    @property
    def center_x(self) -> int:
        return (self.box[0] + self.box[2]) // 2

    @property
    def center_y(self) -> int:
        return (self.box[1] + self.box[3]) // 2
