"""
SessionLogger — JSONL session telemetry.
Logs full Detection records matching spec §19 output format.
"""
import os
import json
import time
from datetime import datetime
from core.config import Config


class SessionLogger:

    def __init__(self):
        if not Config.LOG_ENABLED:
            self._enabled = False
            return
        self._enabled = True
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path  = os.path.join(Config.LOG_DIR, f"session_{ts}.jsonl")
        self._start = time.time()
        self._count = 0
        self._write({"event": "session_start", "timestamp": ts})

    def log_detections(self, detections, direction: str):
        if not self._enabled:
            return
        self._write({
            "event":     "detection",
            "direction": direction,
            "objects":   [d.to_record() for d in detections],
        })

    def log_speech(self, messages: list):
        if not self._enabled:
            return
        self._write({"event": "speech", "messages": messages})

    def get_stats(self) -> dict:
        return {
            "duration_s": round(time.time() - self._start, 1),
            "events":     self._count,
        }

    def _write(self, data: dict):
        data["t"] = round(time.time() - self._start, 3)
        self._count += 1
        with open(self._path, "a") as f:
            f.write(json.dumps(data) + "\n")
