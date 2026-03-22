import time
from collections import defaultdict
from typing import List, Dict, Optional
from models.detection import Detection
from core.config import Config

class SceneMemory:
    """
    Tracks objects across frames, maintains scene history,
    and provides context-aware summaries for the LLM.
    """

    def __init__(self):
        # track_id -> list of recent detections
        self._tracks: Dict[int, List[Detection]] = defaultdict(list)
        self._last_seen: Dict[int, float] = {}
        self._scene_history: List[str] = []   # last N scene descriptions
        self._max_history = 5
        self._next_id = 1
        self._label_id_map: Dict[str, int] = {}  # simple label→id for untracked

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Assign track IDs and prune stale tracks. Returns enriched detections."""
        now = time.time()
        seen_ids = set()

        for d in detections:
            # Simple label-based tracking (upgrade to IoU tracker if needed)
            if d.label not in self._label_id_map:
                self._label_id_map[d.label] = self._next_id
                self._next_id += 1
            tid = self._label_id_map[d.label]
            d.track_id = tid
            self._tracks[tid].append(d)
            self._last_seen[tid] = now
            seen_ids.add(tid)

            # Keep only last 10 frames per track
            if len(self._tracks[tid]) > 10:
                self._tracks[tid].pop(0)

        # Prune stale tracks (not seen for > N frames worth of time)
        stale = [tid for tid, t in self._last_seen.items()
                 if now - t > Config.TRACKING_MAX_AGE * 0.1]
        for tid in stale:
            self._tracks.pop(tid, None)
            self._last_seen.pop(tid, None)
            # Remove from label map
            self._label_id_map = {k: v for k, v in self._label_id_map.items() if v != tid}

        return detections

    def get_approaching(self) -> List[Detection]:
        """Returns objects that are getting closer (area increasing over frames)."""
        approaching = []
        for tid, history in self._tracks.items():
            if len(history) >= 3:
                areas = [d.area for d in history[-3:]]
                if areas[-1] > areas[0] * 1.15:  # 15% growth = approaching
                    approaching.append(history[-1])
        return approaching

    def add_scene_description(self, desc: str):
        self._scene_history.append(desc)
        if len(self._scene_history) > self._max_history:
            self._scene_history.pop(0)

    def get_context_summary(self) -> str:
        if not self._scene_history:
            return "No prior context."
        return " → ".join(self._scene_history[-3:])

    def object_count(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for history in self._tracks.values():
            if history:
                counts[history[-1].label] += 1
        return dict(counts)
