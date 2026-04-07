"""
VisionMemory — Structured Spatial Memory for interactive assistance.

Stores historical detections and user-labeled 'custom objects'.
"""
from collections import deque
import time
import json
from typing import List, Dict, Optional

class VisionMemory:
    def __init__(self, max_minutes=15):
        self.capacity_seconds = max_minutes * 60
        # Store dicts: {"ts": time.time(), "description": "Cup on LEFT at 2m"}
        self._buffer = deque()
        # Custom labeled objects: {"label": {"alias": "my keys", "last_seen": {...}}}
        self._custom_objects: Dict[str, Dict] = {}

    def add_detections(self, formatted_detections: List[str]):
        """Append a snapshot of the current scene to memory."""
        now = time.time()
        for desc in formatted_detections:
            self._buffer.append({"ts": now, "description": desc})
        self._trim()

    def label_object(self, original_label: str, user_alias: str, current_spatial_data: str):
        """Associates a generic YOLO label with a user's personal alias."""
        self._custom_objects[user_alias.lower()] = {
            "original_label": original_label,
            "alias": user_alias,
            "last_seen_context": current_spatial_data,
            "timestamp": time.time()
        }
        print(f"  [Memory] Labeled {original_label} as '{user_alias}'")

    def find_custom_object(self, query: str) -> Optional[str]:
        """Searches for a user-defined object in memory."""
        query = query.lower()
        for alias, data in self._custom_objects.items():
            if alias in query:
                return f"Last saw your {alias} {data['last_seen_context']} ({self._format_age(data['timestamp'])})."
        return None

    def _trim(self):
        """Remove memories older than the capacity."""
        cutoff = time.time() - self.capacity_seconds
        while self._buffer and self._buffer[0]["ts"] < cutoff:
            self._buffer.popleft()

    def _format_age(self, ts: float) -> str:
        age_sec = time.time() - ts
        if age_sec < 60: return "just now"
        return f"{int(age_sec // 60)} mins ago"

    def get_recent_history(self) -> str:
        """Returns a summary of objects seen recently and their last known locations."""
        if not self._buffer and not self._custom_objects:
            return "No recent visual memory."
            
        summary = {}
        # Add custom labeled objects first
        for alias, data in self._custom_objects.items():
            summary[f"Personal {alias}"] = (f"{data['alias']} ({data['last_seen_context']})", data['timestamp'])

        # Process buffer to find last known location of every generic object
        for item in self._buffer:
            # The description is now a " | " separated list of "label at dist dir"
            parts = item["description"].split(" | ")
            for part in parts:
                if " at " in part:
                    label = part.split(" at ")[0].strip()
                    summary[label] = (part, item["ts"])
        
        if not summary:
            return "No distinct objects recognized in the last 15 minutes."
            
        history_lines = []
        for label, (info, ts) in summary.items():
            history_lines.append(f"{info} [{self._format_age(ts)}]")
            
        return " | ".join(history_lines)


class GoalSystem:
    """Tracks persistent user goals (e.g., 'water', 'rest') and candidate objects."""
    def __init__(self):
        # Format: {"water": {"candidates": ["bottle", "cup", "sink", "grocery store", "refrigerator"], "priority": "high", "active": True}}
        self._goals: Dict[str, Dict] = {}
        # Map common needs to YOLO classes
        self._goal_map = {
            "water": ["bottle", "cup", "wine glass", "sink", "refrigerator", "grocery store"],
            "rest": ["chair", "bench", "sofa", "bed", "dining table"],
            "sit": ["chair", "bench", "sofa", "bed"],
            "food": ["apple", "banana", "sandwich", "orange", "broccoli", "carrot", "pizza", "donut", "cake", "refrigerator", "dining table", "grocery store"],
            "exit": ["door", "stairs"],
            "wallet": ["handbag", "backpack"],
            "money": ["handbag", "backpack"],
        }

    def set_goal(self, user_intent: str) -> List[str]:
        """Sets a persistent goal and returns candidate objects to look for."""
        intent = user_intent.lower()
        candidates = []
        matched_goal = None
        
        for goal, objs in self._goal_map.items():
            if goal in intent:
                self._goals[goal] = {
                    "candidates": objs,
                    "priority": "high",
                    "active": True,
                    "timestamp": time.time()
                }
                candidates.extend(objs)
                matched_goal = goal
        
        if not matched_goal:
            # Generic fallback
            self._goals[intent] = {"candidates": [intent], "priority": "high", "active": True, "timestamp": time.time()}
            candidates = [intent]
            
        print(f"  [Goal] Active: '{matched_goal or intent}' -> {candidates}")
        return candidates

    def get_active_candidates(self) -> List[str]:
        """Returns all unique objects the system should currently be searching for."""
        all_objs = []
        for g in self._goals.values():
            if g["active"]:
                all_objs.extend(g["candidates"])
        unique_objs = list(set(all_objs))
        # Only log once when goals change, not every frame
        return unique_objs

    def complete_goal(self, goal_or_candidate: str):
        """Disables a goal if the user has reached it or confirmed it's done."""
        g_to_remove = []
        for g_name, g_data in self._goals.items():
            if g_name in goal_or_candidate or any(c in goal_or_candidate for c in g_data["candidates"]):
                g_to_remove.append(g_name)
        
        for g in g_to_remove:
            self._goals[g]["active"] = False
            print(f"  [Goal] Completed: {g}")

    def lower_priority(self, candidate: str):
        """Lowers priority of a specific goal candidate (e.g., user said 'no' to a grocery store)."""
        for g_data in self._goals.values():
            if candidate in g_data["candidates"]:
                g_data["priority"] = "low"
                print(f"  [Goal] Lowered priority for: {candidate}")

    def get_goal_summary(self) -> str:
        active = [f"{k} (looking for: {', '.join(v['candidates'][:3])}...)" for k, v in self._goals.items() if v['active']]
        if not active: return "No active goals."
        return "ACTIVE GOALS: " + " | ".join(active)

# Global instances
memory_bank = VisionMemory()
goal_system = GoalSystem()

