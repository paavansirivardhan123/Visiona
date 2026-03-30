"""
MonoDepth — monocular depth via MiDaS (spec §5B-7B).

MiDaS outputs INVERSE depth (disparity-like): higher value = closer object.
Correct formula:  metric_depth_m = scale / midas_value

Calibration (spec §6B):
  scale = real_distance_m * D_ref
  where D_ref = median MiDaS value inside the reference object bounding box.

For a person at ~4m with MiDaS value ~314:
  scale = 4.0 * 314 = 1256  →  depth = 1256 / 314 = 4.0m  ✓

Falls back to bounding-box heuristic if MiDaS unavailable.
Runs depth inference every DEPTH_SKIP_FRAMES for performance.
"""
import logging
from collections import deque
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from core.config import Config

logger = logging.getLogger(__name__)

# Known real-world widths (m) — used to estimate real distance for calibration
_REF_WIDTHS_M = {
    "person": 0.50, "car": 1.80, "truck": 2.50, "bus": 2.50,
    "bicycle": 0.60, "motorcycle": 0.80,
}

# How many frames to skip between MiDaS inference calls (for performance)
_DEPTH_SKIP = 3   # run MiDaS every 3rd frame, cache in between


class MonoDepth:

    def __init__(self):
        self._model       = None
        self._transform   = None
        self._device      = None
        self._load_failed = False
        # scale: metric_depth = scale / midas_value
        # Default: calibrated for typical indoor/outdoor scene
        # Will be updated automatically from detected reference objects
        self._scale: float = Config.MIDAS_DEFAULT_SCALE
        self._calibrated   = False
        self._history: Dict[int, deque] = {}   # track_id → depth history
        self._cached_depth_map: Optional[np.ndarray] = None
        self._frame_skip_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        frame: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        labels: Optional[List[str]] = None,
        track_ids: Optional[List[Optional[int]]] = None,
    ) -> List[Optional[float]]:
        """Returns metric depth (m) per bounding box."""
        self._ensure_loaded()

        if self._load_failed or self._model is None:
            return self._heuristic(boxes, labels)

        # Run MiDaS every _DEPTH_SKIP frames, cache in between
        self._frame_skip_count += 1
        if self._frame_skip_count >= _DEPTH_SKIP or self._cached_depth_map is None:
            depth_map = self._run_midas(frame)
            if depth_map is not None:
                self._cached_depth_map = depth_map
            self._frame_skip_count = 0
        else:
            depth_map = self._cached_depth_map

        if depth_map is None:
            return self._heuristic(boxes, labels)

        # Update scale calibration from reference objects
        if labels and not self._calibrated:
            self._calibrate(depth_map, boxes, labels, frame.shape)

        results: List[Optional[float]] = []
        for i, box in enumerate(boxes):
            tid = track_ids[i] if track_ids else None
            midas_val = self._median_roi(depth_map, box)
            if midas_val is None or midas_val <= 0:
                results.append(None)
                continue
            # Correct formula: metric_depth = scale / midas_value
            metric = round(self._scale / midas_val, 2)
            metric = max(0.1, min(metric, Config.MAX_DISTANCE_M))
            metric = self._smooth(tid, metric)
            results.append(metric)
        return results

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def scale(self) -> float:
        return self._scale

    def _ensure_loaded(self):
        if self._model is not None or self._load_failed:
            return
        try:
            import torch
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            mt = Config.MIDAS_MODEL_TYPE
            print(f"  [Depth] Loading MiDaS '{mt}' on {self._device}...")
            self._model = torch.hub.load("intel-isl/MiDaS", mt, trust_repo=True)
            self._model.to(self._device)
            self._model.eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self._transform = (transforms.dpt_transform
                               if mt in ("DPT_Large", "DPT_Hybrid")
                               else transforms.small_transform)
            print(f"  [Depth] MiDaS ready. Scale={self._scale:.1f}")
        except Exception as exc:
            print(f"  [Depth] MiDaS failed: {exc}. Using heuristic fallback.")
            self._load_failed = True

    def _run_midas(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            import torch
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch = self._transform(rgb).to(self._device)
            with torch.no_grad():
                pred = self._model(batch)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=frame.shape[:2],
                    mode="bicubic", align_corners=False,
                ).squeeze()
            return pred.cpu().numpy().astype(np.float32)
        except Exception as exc:
            logger.warning(f"MiDaS inference: {exc}")
            return None

    # ------------------------------------------------------------------
    # Calibration: scale = real_distance * D_ref
    # ------------------------------------------------------------------

    def _calibrate(self, depth_map, boxes, labels, frame_shape):
        """
        Calibrate scale using first detected reference object.
        For a person at estimated real distance R with MiDaS value D:
          scale = R * D  →  metric = scale / D = R  ✓
        """
        orig_h, orig_w = frame_shape[:2]
        for box, label in zip(boxes, labels):
            real_w_m = _REF_WIDTHS_M.get(label)
            if real_w_m is None:
                continue
            x1, y1, x2, y2 = box
            px_w = max(x2 - x1, 1)
            # Estimate real distance using pinhole model as bootstrap
            # real_dist = (real_width_m * focal_px) / pixel_width
            # For 4K frame (3840px wide), focal ~= 2800px (approx 70% of width)
            focal_px = orig_w * 0.73
            real_dist = (real_w_m * focal_px) / px_w
            real_dist = max(0.5, min(real_dist, 15.0))

            d_ref = self._median_roi(depth_map, box)
            if d_ref is None or d_ref <= 0:
                continue

            # scale = real_distance * D_ref
            new_scale = real_dist * d_ref
            if new_scale <= 0:
                continue

            self._scale = new_scale
            self._calibrated = True
            print(f"  [Depth] Calibrated: scale={self._scale:.1f} "
                  f"(ref={label} real_dist={real_dist:.1f}m D_ref={d_ref:.1f})")
            break

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _median_roi(self, depth_map, box) -> Optional[float]:
        x1, y1, x2, y2 = box
        h, w = depth_map.shape[:2]
        x1c, x2c = max(0, x1), min(x2, w)
        y1c, y2c = max(0, y1), min(y2, h)
        if x2c <= x1c or y2c <= y1c:
            return None
        roi = depth_map[y1c:y2c, x1c:x2c]
        return float(np.median(roi)) if roi.size > 0 else None

    def _smooth(self, track_id: Optional[int], depth: float) -> float:
        if track_id is None:
            return depth
        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=Config.DEPTH_SMOOTH_FRAMES)
        self._history[track_id].append(depth)
        return round(float(np.mean(self._history[track_id])), 2)

    def _heuristic(self, boxes, labels) -> List[Optional[float]]:
        results = []
        for i, box in enumerate(boxes):
            label = labels[i] if labels else "unknown"
            x1, y1, x2, y2 = box
            px_w  = max(x2 - x1, 1)
            real_w = Config.KNOWN_WIDTHS_CM.get(label, Config.AVG_OBJECT_WIDTH_CM)
            results.append(round((real_w * Config.FOCAL_LENGTH_PX) / (px_w * 100.0), 2))
        return results
