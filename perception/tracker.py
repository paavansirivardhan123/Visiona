"""
ObjectTracker — ByteTrack-style IoU tracker (spec §9).

No external dependencies. Pure numpy IoU association.
Each track owns a KalmanFilter1D for depth noise reduction.
"""
import time
from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np
from core.config import Config
from kinematics.kalman import KalmanFilter1D
from core.detection import Detection

_IOU_THRESHOLD = 0.3


def _centre(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1)
    ub = (bx2 - bx1) * (by2 - by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


class Track:
    _id_counter = 0

    def __init__(self, det: Detection):
        Track._id_counter += 1
        self.track_id: int = Track._id_counter
        self.label: str    = det.label
        self.age: int      = 0
        self.kf = KalmanFilter1D(Config.KF_PROCESS_NOISE, Config.KF_MEASUREMENT_NOISE)
        # history: deque of (smoothed_distance_m, centre_xy, timestamp)
        self.history: deque = deque(maxlen=Config.TRACK_HISTORY_LEN)
        self.prev_ttc: Optional[float] = None
        self._ingest(det)

    def predict(self):
        self.kf.update(None)
        self.age += 1

    def update(self, det: Detection):
        self.age = 0
        self._ingest(det)

    def _ingest(self, det: Detection):
        smoothed = self.kf.update(det.distance_m) if det.distance_m is not None else self.kf.update(None)
        cx, cy = _centre(det.box)
        self.history.append((smoothed, (cx, cy), time.time()))

    @property
    def is_stale(self) -> bool:
        return self.age > Config.TRACK_MAX_AGE


class ObjectTracker:

    def __init__(self):
        backend = getattr(Config, "TRACKER_BACKEND", "bytetrack")
        if backend not in ("bytetrack", "deepsort"):
            raise ValueError(f"Unknown tracker backend: '{backend}'")
        self._tracks: Dict[int, Track] = {}
        Track._id_counter = 0

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Detection]:
        """Associate detections with tracks; attach track_id to each Detection."""
        for t in self._tracks.values():
            t.predict()

        if not detections:
            self._evict()
            return detections

        active = list(self._tracks.values())
        if active:
            matched, unmatched_d, _ = self._associate(detections, active)
        else:
            matched, unmatched_d = [], list(range(len(detections)))

        for di, ti in matched:
            active[ti].update(detections[di])
            detections[di].track_id = active[ti].track_id

        for di in unmatched_d:
            t = Track(detections[di])
            self._tracks[t.track_id] = t
            detections[di].track_id = t.track_id

        self._evict()
        return detections

    @property
    def active_tracks(self) -> Dict[int, Track]:
        return self._tracks

    def _associate(self, detections, tracks):
        n, m = len(detections), len(tracks)
        iou_mat = np.zeros((n, m), dtype=np.float32)
        for i, det in enumerate(detections):
            for j, trk in enumerate(tracks):
                if trk.history:
                    cx_t, cy_t = trk.history[-1][1]
                    bw = det.box[2] - det.box[0]
                    bh = det.box[3] - det.box[1]
                    trk_box = (int(cx_t - bw/2), int(cy_t - bh/2),
                               int(cx_t + bw/2), int(cy_t + bh/2))
                    iou_mat[i, j] = _iou(det.box, trk_box)

        matched, unmatched_d, unmatched_t = [], list(range(n)), list(range(m))
        while iou_mat.size:
            idx = np.argmax(iou_mat)
            i, j = divmod(int(idx), m)
            if iou_mat[i, j] < _IOU_THRESHOLD:
                break
            matched.append((i, j))
            iou_mat[i, :] = -1
            iou_mat[:, j] = -1
            if i in unmatched_d: unmatched_d.remove(i)
            if j in unmatched_t: unmatched_t.remove(j)
        return matched, unmatched_d, unmatched_t

    def _evict(self):
        for tid in [tid for tid, t in self._tracks.items() if t.is_stale]:
            del self._tracks[tid]
