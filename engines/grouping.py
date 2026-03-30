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
from models.detection import Detection
from core.config import Config


def group_detections(detections: List[Detection]) -> Dict[str, List[Detection]]:
    """Group detections by direction."""
    by_dir: Dict[str, List[Detection]] = {}
    for d in detections:
        by_dir.setdefault(d.direction, []).append(d)
    return by_dir


def build_speech_messages(
    grouped: Dict[str, List[Detection]],
    high_priority: List[Detection],
) -> List[str]:
    """
    Build up to MAX_MESSAGES spoken sentences.

    Priority order:
      1. High-priority (very close) objects first
      2. Then by direction: FRONT → LEFT → RIGHT → BACK
    """
    messages: List[str] = []

    # 1. High-priority alerts first (≤ HIGH_PRIORITY_M)
    for d in high_priority:
        msg = _build_msg(d, force_close=True)
        if msg not in messages:
            messages.append(msg)
        if len(messages) >= Config.MAX_MESSAGES:
            return messages

    # 2. Group by direction and label, build natural sentences
    for direction in ("FRONT", "LEFT", "RIGHT", "BACK"):
        dets = grouped.get(direction, [])
        if not dets:
            continue

        # Count each label in this direction
        counts = Counter(d.label for d in dets)
        dir_phrases = []

        for label, count in counts.items():
            # Pick best sample (closest) for motion info
            sample = min(
                (d for d in dets if d.label == label),
                key=lambda d: d.distance_m or 9999
            )
            if sample.is_high_priority:
                continue  # already handled above

            phrase = _count_phrase(label, count, direction, sample)
            dir_phrases.append(phrase)

        if dir_phrases:
            # Combine all objects in this direction into one sentence
            combined = ", ".join(dir_phrases)
            if combined not in messages:
                messages.append(combined)
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
        base += f" at {round(sample.distance_m, 1)} meters"

    # Append motion info
    if sample.motion == "approaching" and sample.speed_mps:
        base += f", approaching at {sample.speed_mps} m/s"
    elif sample.motion == "moving_away":
        base += ", moving away"

    # Warning prefix for low TTC
    if sample.ttc_sec is not None and sample.ttc_sec <= Config.TTC_WARN_THRESHOLD:
        base = f"Warning: {base}"

    return base


def _build_msg(d: Detection, force_close: bool) -> str:
    """Build message for a single high-priority detection."""
    dir_str = _dir(d.direction)
    label   = d.label
    dist_s  = f"{round(d.distance_m, 1)} meters" if d.distance_m else ""

    if d.motion == "approaching" and d.speed_mps:
        msg = f"{label.capitalize()} approaching {dir_str} at {d.speed_mps} metres per second"
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
