from core.config import Config
import requests

def get_directions(destination: str) -> str:
    """
    Dummy/placeholder implementation for Phase 6.6 GPS Node.
    Requires Google Maps Directions API key.
    """
    if not Config.MAPS_API_KEY:
        return f"Cannot navigate to {destination}. Google Maps API key is missing. Please add MAPS_API_KEY to your .env file."
        
    return f"Directions to {destination} requested from Google Maps. Turn-by-turn navigation started."
