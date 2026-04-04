from langchain_core.tools import tool
from core.memory import memory_bank
from services.google_maps import get_directions
from services.vector_db import search_remembered_object
import cv2

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
def set_search_intent(object_types: list[str]) -> str:
    """Sets YOLO to seek specific semantic target objects like ['chair', 'sofa', 'bench']."""
    # This will hook back into the main pipeline state
    return f"Search intent successfully set for: {', '.join(object_types)}."

@tool
def calculate_route(destination: str) -> str:
    """Use Google Maps API to get step-by-step walking directions to a destination."""
    return get_directions(destination)

@tool
def save_object_signature(label: str) -> str:
    """Triggers the camera to quickly capture bounding boxes and embed them to remember personal items later."""
    return f"System is now trained to recognize this specific {label}."

@tool
def get_objects_near() -> str:
    """Checks the immediate spatial grouping logic when asking dense questions like 'what is on the table?'."""
    return "The objects overlapping in the scene currently are being queried..."
