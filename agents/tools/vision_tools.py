"""
Agent Tools — LangChain tool definitions for Visiona AI.

All tools are decorated with @tool and bound to the LLM in orchestrator.py.
Imports are ordered correctly: stdlib → third-party → local.
"""
import re
from typing import Any, List, Union

from langchain_core.tools import tool

from core.memory import memory_bank, goal_system
from services.google_maps import get_directions
from services.vector_db import search_remembered_object

# Callback registered by AgentEngine so tools can trigger YOLO search
_search_intent_callback: Any = None


def register_search_intent_callback(callback: Any):
    global _search_intent_callback
    _search_intent_callback = callback


# ---------------------------------------------------------------------- #
#  Memory Tools
# ---------------------------------------------------------------------- #

@tool
def query_past_detections(question: str) -> str:
    """Answer questions about objects seen recently, e.g. 'where did I leave my cup?'"""
    context = memory_bank.get_recent_history()
    if not context or context == "No recent visual memory.":
        return "I haven't seen anything notable recently."
    return f"Recent visual memory: {context}"


@tool
def save_object_signature(label: str, alias: str, current_spatial_context: str) -> str:
    """
    Label a generic YOLO object with a user's personal alias.
    Example: label='bottle', alias='my water bottle'.
    """
    memory_bank.label_object(label, alias, current_spatial_context)
    return f"Remembered: the {label} is now called '{alias}'."


# ---------------------------------------------------------------------- #
#  Scene Tools
# ---------------------------------------------------------------------- #

@tool
def summarize_scene(scene_description: str) -> str:
    """Convert raw spatial detection data into one natural sentence for a blind user."""
    return f"Scene summary requested for: {scene_description}"


@tool
def get_objects_near() -> str:
    """Check what objects are currently overlapping or very close together in the scene."""
    history = memory_bank.get_recent_history()
    return f"Current nearby objects: {history}"


# ---------------------------------------------------------------------- #
#  Search / Navigation Tools
# ---------------------------------------------------------------------- #

@tool
def set_search_intent(object_types: Union[List[str], str]) -> str:
    """
    Tell YOLO to actively search for specific objects.
    Example: set_search_intent(['chair', 'bench']) to find a seat.
    """
    if isinstance(object_types, str):
        targets = [object_types]
    else:
        targets = list(object_types)

    if _search_intent_callback and targets:
        print(f"  [Tool] Setting search intent: {targets[0]}")
        _search_intent_callback(targets[0])
    else:
        print("  [Tool] Search intent callback not registered or no targets provided.")

    return f"Now searching for: {', '.join(targets)}."


@tool
def calculate_route(destination: str) -> str:
    """Get step-by-step walking directions to a destination using Google Maps."""
    return get_directions(destination)


# ---------------------------------------------------------------------- #
#  Goal System Tools
# ---------------------------------------------------------------------- #

@tool
def set_persistent_goal(user_need: str) -> str:
    """
    Set a long-term navigation goal (e.g., 'water', 'rest', 'food', 'exit').
    The system will proactively alert the user when relevant objects are detected.
    """
    candidates = goal_system.set_goal(user_need)
    if _search_intent_callback and candidates:
        _search_intent_callback(candidates[0])
    return f"Goal '{user_need}' is active. Searching for: {', '.join(candidates)}."


@tool
def mark_goal_completed(goal_or_object: str) -> str:
    """Mark a persistent goal as done because the user reached it or no longer needs it."""
    goal_system.complete_goal(goal_or_object)
    return f"Goal '{goal_or_object}' has been cleared."


@tool
def lower_goal_candidate_priority(candidate: str) -> str:
    """
    Lower the priority of a specific candidate object for the current goal.
    Use when the user says 'no' to a suggested option (e.g., 'not that grocery store').
    """
    goal_system.lower_priority(candidate)
    return f"Will suggest '{candidate}' less frequently."
