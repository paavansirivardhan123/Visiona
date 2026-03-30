"""
DetectionPriorityQueue — max-heap priority queue for detections.

Priority factors (spec §13):
  1. Lowest TTC  → highest urgency
  2. Closest distance
  3. Object type importance (Config.OBJECT_PRIORITY)
"""
import heapq
from typing import List
from models.detection import Detection
from core.config import Config


def compute_priority(det: Detection) -> float:
    """
    Returns a numeric priority score. Higher = more urgent.

    TTC is the primary factor: a 2-second TTC scores much higher than a 10-second one.
    Distance is secondary. Object type is tertiary.
    """
    obj_score = Config.OBJECT_PRIORITY.get(det.label, Config.DEFAULT_PRIORITY)

    # Distance score: closer = higher (max 10m considered)
    if det.distance_m is not None and det.distance_m > 0:
        dist_score = max(0.0, (Config.CONSIDER_MAX_M - det.distance_m) * 10)
    else:
        dist_score = 0.0

    # TTC score: lower TTC = much higher priority
    if det.ttc_sec is not None and det.ttc_sec > 0:
        ttc_score = 1000.0 / det.ttc_sec   # e.g. TTC=2s → 500, TTC=10s → 100
    else:
        ttc_score = 0.0

    return ttc_score + dist_score + obj_score


class DetectionPriorityQueue:
    """Max-heap. Highest priority detection comes out first."""

    def __init__(self):
        self._heap: List = []
        self._counter = 0

    def push(self, det: Detection):
        score = compute_priority(det)
        det.priority = int(score)
        heapq.heappush(self._heap, (-score, self._counter, det))
        self._counter += 1

    def pop(self) -> Detection:
        _, _, det = heapq.heappop(self._heap)
        return det

    def push_all(self, detections: List[Detection]):
        for d in detections:
            self.push(d)

    def drain(self) -> List[Detection]:
        """Return all detections sorted by priority (highest first)."""
        result = []
        while self._heap:
            result.append(self.pop())
        return result

    def __len__(self):
        return len(self._heap)

    def is_empty(self):
        return len(self._heap) == 0
