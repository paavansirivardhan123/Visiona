from langchain_core.tools import tool
from core.memory import memory_bank, goal_system

@tool
def set_persistent_goal(user_need: str) -> str:
    """
    Sets a long-term goal (e.g., 'water', 'rest', 'food') and starts a proactive search for 
    a variety of candidate objects (e.g., bottle, sink, chair, grocery store).
    """
    candidates = goal_system.set_goal(user_need)
    # Also trigger the immediate search intent for the most likely first candidate
    if _search_intent_callback and candidates:
        _search_intent_callback(candidates[0])
    return f"Goal '{user_need}' is now active. I am looking for: {', '.join(candidates)}."

@tool
def mark_goal_completed(goal_or_object: str) -> str:
    """Marks a persistent goal as finished because the user reached it or changed their mind."""
    goal_system.complete_goal(goal_or_object)
    return f"Goal related to '{goal_or_object}' has been cleared."

@tool
def lower_goal_candidate_priority(candidate: str) -> str:
    """Lowers priority of a specific candidate (e.g., user said 'no' to a grocery store)."""
    goal_system.lower_priority(candidate)
    return f"I will stop suggesting '{candidate}' as frequently for current goals."
from services.google_maps import get_directions
from services.vector_db import search_remembered_object
import cv2
from typing import Any

_search_intent_callback: Any = None

def register_search_intent_callback(callback: Any):
    global _search_intent_callback
    _search_intent_callback = callback

@tool
def query_past_detections(question: str) -> str:
    """Useful for answering questions about objects seen in the recent past, like 'where did I leave my cup?'."""
    context = memory_bank.get_recent_history()
    return f"Here is the visual memory log:\n{context}\n\nAnswer the user's question: {question}"

@tool
def summarize_scene(scene_description: str) -> str:
    """Translate dense, robotic JSON spatial coordinates into one natural, comforting sentence for a blind user."""
    return f"Analyze this scene: {scene_description}"

@tool
def set_search_intent(object_types: list[str] | str) -> str:
    """Sets YOLO to seek specific semantic target objects like ['chair', 'sofa', 'bench']."""
    resolved_list = [object_types] if isinstance(object_types, str) else object_types
    resolved_target = resolved_list[0] if resolved_list else None
    
    if _search_intent_callback and resolved_target:
        print(f"  [Tool] Setting search intent: {resolved_target}")
        _search_intent_callback(resolved_target)
    else:
        print(f"  [Tool] Search intent callback unavailable or no target")

    return f"Search intent successfully set for: {', '.join(resolved_list)}."

@tool
def calculate_route(destination: str) -> str:
    """Use Google Maps API to get step-by-step walking directions to a destination."""
    return get_directions(destination)

@tool
def save_object_signature(label: str, alias: str, current_spatial_context: str) -> str:
    """
    Labels a generic object with a user's custom alias (e.g., label='bottle', alias='my water').
    Used when a user says 'This is my water'.
    """
    memory_bank.label_object(label, alias, current_spatial_context)
    return f"Successfully remembered that the {label} is '{alias}'."


@tool
def get_objects_near() -> str:
    """Checks the immediate spatial grouping logic when asking dense questions like 'what is on the table?'."""
    return "The objects overlapping in the scene currently are being queried..."
