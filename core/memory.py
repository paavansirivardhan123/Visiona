"""
Rolling chronological memory buffer for YOLO detections.
Powers Phase 6.1: Vision-Augmented Memory.
"""
from collections import deque
import time
from typing import List, Dict

class VisionMemory:
    def __init__(self, max_minutes=15):
        self.capacity_seconds = max_minutes * 60
        # Store dicts: {"ts": time.time(), "description": "Cup on LEFT at 2m"}
        self._buffer = deque()

    def add_detections(self, formatted_detections: List[str]):
        """Append a snapshot of the current scene to memory."""
        now = time.time()
        for desc in formatted_detections:
            self._buffer.append({"ts": now, "description": desc})
        self._trim()

    def _trim(self):
        """Remove memories older than the capacity."""
        cutoff = time.time() - self.capacity_seconds
        while self._buffer and self._buffer[0]["ts"] < cutoff:
            self._buffer.popleft()

    def get_recent_history(self) -> str:
        """Returns the chronological string of recent events for LangChain."""
        if not self._buffer:
            return "No recent visual memory."
            
        history = []
        for item in self._buffer:
            age_min = int((time.time() - item["ts"]) / 60)
            if age_min == 0:
                time_str = "Just now"
            else:
                time_str = f"{age_min} mins ago"
            history.append(f"[{time_str}] {item['description']}")
            
        return "\n".join(history[-50:])  # Cap to last 50 events to avoid overwhelming LLM token limit

# Global memory instance
memory_bank = VisionMemory()
