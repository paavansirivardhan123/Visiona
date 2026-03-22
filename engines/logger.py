import os
import json
import time
from datetime import datetime
from core.config import Config

class SessionLogger:
    """
    Logs session events (detections, instructions, states) to JSON.
    Useful for analytics, debugging, and future ML training data.
    """

    def __init__(self):
        if not Config.LOG_ENABLED:
            self._enabled = False
            return
        self._enabled = True
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(Config.LOG_DIR, f"session_{ts}.jsonl")
        self._session_start = time.time()
        self._event_count = 0
        self._log({"event": "session_start", "timestamp": ts})

    def log_detection(self, detections):
        if not self._enabled: return
        self._log({
            "event": "detection",
            "objects": [
                {
                    "label": d.label,
                    "position": d.position,
                    "distance_cm": d.distance_cm,
                    "confidence": round(d.confidence, 3),
                    "is_hazard": d.is_hazard,
                }
                for d in detections
            ]
        })

    def log_instruction(self, text: str, source: str, priority: bool = False):
        if not self._enabled: return
        self._log({
            "event": "instruction",
            "text": text,
            "source": source,   # "safety" | "llm" | "intent"
            "priority": priority,
        })

    def log_state(self, state: str, reasoning: str):
        if not self._enabled: return
        self._log({"event": "state_change", "state": state, "reasoning": reasoning})

    def get_stats(self) -> dict:
        elapsed = round(time.time() - self._session_start, 1)
        return {"session_duration_s": elapsed, "total_events": self._event_count}

    def _log(self, data: dict):
        data["t"] = round(time.time() - self._session_start, 3)
        self._event_count += 1
        with open(self._path, "a") as f:
            f.write(json.dumps(data) + "\n")
