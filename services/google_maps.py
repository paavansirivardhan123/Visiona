from core.config import Config
import requests

def get_directions(destination: str, origin: str = None) -> str:
    """
    Fetch shortest walking directions using Google Maps Directions API.
    Uses 'current location' if origin is None.
    """
    api_key = Config.MAPS_API_KEY
    if not api_key:
        return f"Cannot navigate to {destination}. Google Maps API key is missing."

    # Use live location from Config if available, else default to 'current location'
    if origin is None:
        origin = getattr(Config, "CURRENT_LOCATION", "current location")

    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": origin,
        "destination": destination,
        "mode": "walking",
        "alternatives": "true", # Fetch multiple routes to find the shortest
        "key": api_key
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if data["status"] == "OK":
            # Select the shortest route from alternatives
            routes = data["routes"]
            shortest_route = min(routes, key=lambda r: r["legs"][0]["distance"]["value"])
            
            leg = shortest_route["legs"][0]
            summary = (
                f"I've found the shortest path to {destination}. "
                f"It's about {leg['distance']['text']} and will take {leg['duration']['text']}.\n"
            )
            
            steps = []
            for step in leg["steps"]:
                import re
                instruction = re.sub('<[^<]+?>', '', step["html_instructions"])
                steps.append(f"- {instruction} ({step['distance']['text']})")
            
            return summary + "\n".join(steps[:5]) + "\n(I will keep monitoring your progress.)"
        else:
            return f"Google Maps Error: {data.get('error_message', data['status'])}"
    except Exception as e:
        return f"Navigation system error: {str(e)}"
