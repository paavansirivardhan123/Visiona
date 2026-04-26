"""
Grouping + speech message builder.

Spec rules:
  - count > GROUP_THRESHOLD (5) → "Group of <label>s <direction>"
  - count <= 5 → "<count> <label>s <direction>"
  - multiple object types → describe each separately
  - approaching + speed → include speed
  - TTC <= TTC_WARN_THRESHOLD → prepend "Warning:"
  - distance <= HIGH_PRIORITY_M → include "very close"
  - max Config.MAX_MESSAGES per cycle
  - direction order: FRONT → LEFT → RIGHT → BACK
"""
from collections import Counter
from typing import Dict, List
import time

from core.detection import Detection
from core.priority_queue import DetectionPriorityQueue
from core.config import Config


def group_detections(detections: List[Detection]) -> Dict[str, List[Detection]]:
    """Group detections by direction."""
    by_dir: Dict[str, List[Detection]] = {}
    for d in detections:
        by_dir.setdefault(d.direction, []).append(d)
    return by_dir

STATIC_LABELS = {
    "chair", "bench", "potted plant", "tree", "helmet", 
    "bed", "dining table", "toilet", "tv", "microwave", 
    "oven", "toaster", "sink", "refrigerator", "book", 
    "clock", "vase", "fire hydrant", "stop sign", 
    "parking meter", "traffic light"
}


_direction_penalties = {"FRONT": 0.0, "LEFT": 0.0, "RIGHT": 0.0, "BACK": 0.0}
_last_eval_time = 0.0

def build_speech_messages(
    grouped: Dict[str, List[Detection]],
    high_priority: List[Detection],
) -> List[str]:
    """
    Build up to MAX_MESSAGES spoken sentences.
    Uses Dynamic Threat Scoring with Penalty Decay to prevent sensory monopolization.
    """
    global _direction_penalties, _last_eval_time
    import time
    
    now = time.time()
    dt = now - _last_eval_time if _last_eval_time > 0 else 0.0
    _last_eval_time = now
    
    # 1. Decay all penalties
    for d in _direction_penalties:
        _direction_penalties[d] = max(0.0, _direction_penalties[d] - (Config.PENALTY_DECAY * dt))

    messages: List[str] = []

    # 2. Critical TTC Emergency Warnings first (Absolute Bypass)
    # This prevents the user from crashing regardless of anti-spam rules.
    emergency_handled = False
    for d in high_priority:
        if d.ttc_sec is not None and d.ttc_sec <= Config.TTC_WARN_THRESHOLD:
            msg = _build_msg(d, force_close=True)
            if msg not in messages:
                messages.append(msg)
                emergency_handled = True
            if len(messages) >= Config.MAX_MESSAGES:
                return messages

    # 3. Score the directions mathematically
    dir_scores = {}
    for direction in ("FRONT", "LEFT", "RIGHT", "BACK"):
        dets = grouped.get(direction, [])
        if not dets:
            continue
        max_threat = max((d.threat_score for d in dets), default=0.0)
        
        # Override penalty massively if threat is an Extreme Threat
        penalty = _direction_penalties[direction]
        if max_threat > Config.THREAT_HIGH_THRESHOLD: 
            penalty = 0.0  # Bypass penalty for severe threats
            
        dir_scores[direction] = max_threat - penalty

    # 4. Sort directions by penalized score
    sorted_dirs = sorted(dir_scores.keys(), key=lambda k: dir_scores[k], reverse=True)

    for direction in sorted_dirs:
        dets = grouped[direction]
        counts = Counter(d.label for d in dets)
        dir_phrases = []

        for label, count in counts.items():
            sample = max((d for d in dets if d.label == label), key=lambda x: x.threat_score)
            
            # Skip if we already warned about this exact sample in emergency phase
            if emergency_handled and sample.ttc_sec is not None and sample.ttc_sec <= Config.TTC_WARN_THRESHOLD:
                continue

            phrase = _count_phrase(label, count, direction, sample)
            dir_phrases.append(phrase)

        if dir_phrases:
            combined = ", ".join(dir_phrases)
            if combined not in messages:
                messages.append(combined)
                # Apply Speech Penalty because we decided to speak about this direction!
                _direction_penalties[direction] += Config.PENALTY_APPLY
                
            if len(messages) >= Config.MAX_MESSAGES:
                return messages

    return messages[:Config.MAX_MESSAGES]


def _count_phrase(label: str, count: int, direction: str, sample: Detection) -> str:
    """
    Build a count-based phrase per spec:
      count > 5  → "Group of people in front"
      count <= 5 → "3 people on the left"
      count == 1 → "a person in front"
    With motion awareness appended if applicable.
    """
    dir_str = _dir(direction)
    plural  = label + "s" if not label.endswith("s") else label

    if count > Config.GROUP_THRESHOLD:
        base = f"Group of {plural} {dir_str}"
    elif count == 1:
        base = f"a {label} {dir_str}"
    else:
        base = f"{count} {plural} {dir_str}"

    # Append distance
    if sample.distance_m is not None:
        steps = max(1, round(sample.distance_m / Config.METERS_PER_STEP))
        step_word = "step" if steps == 1 else "steps"
        base += f" at {steps} {step_word}"

    # Append motion info - ONLY for approaching objects or vehicles
    is_static = label.lower() in STATIC_LABELS
    base_l = label.lower()
    is_vehicle = base_l in ("car", "truck", "bus", "motorcycle", "bicycle")
    
    if not is_static:
        # Only mention motion for approaching objects or any vehicle motion
        if sample.motion == "approaching" and sample.speed_mps:
            speed_word = _get_speed_descriptor(sample.speed_mps)
            base += f", approaching {speed_word}"
        elif is_vehicle and sample.motion == "moving_away" and sample.speed_mps:
            # Only mention moving away for vehicles (safety relevant)
            speed_word = _get_speed_descriptor(sample.speed_mps)
            base += f", moving away {speed_word}"

    # Warning prefix for low TTC
    if sample.ttc_sec is not None and sample.ttc_sec <= Config.TTC_WARN_THRESHOLD:
        base = f"Warning: {base}"

    return base


def _build_msg(d: Detection, force_close: bool) -> str:
    """Build message for a single high-priority detection."""
    dir_str = _dir(d.direction)
    label   = d.label
    dist_s  = ""
    if d.distance_m:
        steps = max(1, round(d.distance_m / Config.METERS_PER_STEP))
        step_word = "step" if steps == 1 else "steps"
        dist_s = f"{steps} {step_word}"

    is_static = label.lower() in STATIC_LABELS
    if not is_static and d.motion == "approaching" and d.speed_mps:
        speed_word = _get_speed_descriptor(d.speed_mps)
        msg = f"{label.capitalize()} approaching {dir_str} {speed_word}"
    elif force_close or (d.distance_m and d.distance_m <= Config.HIGH_PRIORITY_M):
        msg = f"{label.capitalize()} very close {dir_str}"
        if dist_s:
            msg += f", {dist_s}"
    else:
        msg = f"{label.capitalize()} {dir_str}"
        if dist_s:
            msg += f" at {dist_s}"

    if d.ttc_sec is not None and d.ttc_sec <= Config.TTC_WARN_THRESHOLD:
        msg = f"Warning: {msg}"

    return msg


def _dir(direction: str) -> str:
    return {
        "FRONT": "in front",
        "LEFT":  "on the left",
        "RIGHT": "on the right",
        "BACK":  "behind you",
    }.get(direction, direction.lower())

def _get_speed_descriptor(speed_mps: float) -> str:
    """Production-level translation of raw speed to qualitative descriptors for user comfort."""
    if speed_mps < 0.8:
        return "slowly"
    elif speed_mps < 2.0:
        return "fast"
    else:
        return "very quickly"

