"""
DepthEngine — monocular-only depth estimation using MiDaS.

No stereo. No auto-detection. Single camera, always monocular.
Falls back to bounding-box heuristic if MiDaS fails to load.
"""
from typing import List, Optional, Tuple
import numpy as np
from core.config import Config
from engines.mono_depth import MonoDepth

_REQUIRED = [
    "MIDAS_MODEL_TYPE", "MIDAS_DEFAULT_SCALE", "DEPTH_SMOOTH_FRAMES",
    "DEPTH_SPIKE_THRESHOLD", "KF_PROCESS_NOISE", "KF_MEASUREMENT_NOISE",
    "FOCAL_LENGTH_PX", "AVG_OBJECT_WIDTH_CM", "KNOWN_WIDTHS_CM",
]


class DepthEngine:
    """
    Monocular depth engine.
    mode is always 'monocular'.
    """

    mode: str = "monocular"

    def __init__(self):
        missing = [p for p in _REQUIRED if not hasattr(Config, p)]
        if missing:
            raise ValueError(f"DepthEngine: missing Config params: {', '.join(missing)}")
        self._mono = MonoDepth()

    def estimate_frame(self, frame: np.ndarray, detections) -> List[Optional[float]]:
        """Batch depth for all detections. Returns metres per detection."""
        if not detections:
            return []
        boxes     = [d.box for d in detections]
        labels    = [d.label for d in detections]
        track_ids = [getattr(d, "track_id", None) for d in detections]
        return self._mono.compute(frame, boxes, labels, track_ids)

    def estimate(self, label: str, box: Tuple, frame: np.ndarray) -> Optional[float]:
        r = self._mono.compute(frame, [box], [label], [None])
        return r[0] if r else None

    @property
    def is_calibrated(self) -> bool:
        return self._mono.is_calibrated

    @property
    def scale(self) -> float:
        return self._mono.scale
