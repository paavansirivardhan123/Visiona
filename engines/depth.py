from core.config import Config

class DepthEstimator:
    """
    Estimates real-world distance using the pinhole camera model.
    Formula: distance = (real_width * focal_length) / pixel_width
    """

    KNOWN_WIDTHS_CM = {
        "person":     50,
        "car":        180,
        "truck":      250,
        "bus":        250,
        "bicycle":    60,
        "motorcycle": 80,
        "dog":        40,
        "cat":        25,
        "bottle":     8,
        "chair":      50,
        "cup":        8,
        "laptop":     35,
        "backpack":   35,
        "book":       20,
    }

    def estimate(self, label: str, box) -> float:
        """Returns estimated distance in cm."""
        x1, y1, x2, y2 = box
        pixel_width = max(x2 - x1, 1)
        real_width = self.KNOWN_WIDTHS_CM.get(label, Config.AVG_OBJECT_WIDTH_CM)
        distance = (real_width * Config.FOCAL_LENGTH_PX) / pixel_width
        return round(distance, 1)
