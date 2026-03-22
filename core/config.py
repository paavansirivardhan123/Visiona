class Config:
    """Centralized configuration for the navigation system."""

    # --- Video ---
    VIDEO_SOURCE = "sample-vid/sample.mp4"
    IMG_SIZE = 640
    CONF_THRESHOLD = 0.35
    FRAME_SKIP = 2                  # Process every 2nd frame (was 3)

    # --- Timing ---
    SPEECH_COOLDOWN = 2.0
    LLM_COOLDOWN = 3.5
    TRACKING_MAX_AGE = 10           # Frames before a tracked object is dropped

    # --- Distance Estimation (pinhole camera model) ---
    # Calibrate: measure real object width (cm) and known distance (cm) at known pixel width
    FOCAL_LENGTH_PX = 700           # Approximate for 640px wide frame, standard webcam
    AVG_PERSON_WIDTH_CM = 50
    AVG_OBJECT_WIDTH_CM = 30

    # --- Distance Zones (cm) ---
    DIST_CRITICAL = 80              # < 80cm  → STOP immediately
    DIST_NEAR = 150                 # < 150cm → slow down / avoid
    DIST_MEDIUM = 300               # < 300cm → heads-up warning

    # --- Legacy area thresholds (kept for fallback) ---
    AREA_VERY_CLOSE = 80000
    AREA_NEAR = 30000

    # --- Spatial Grid (5 zones across frame width) ---
    ZONES = ["far-left", "left", "ahead", "right", "far-right"]

    # --- UI ---
    WINDOW_NAME = "NaVision AI — Assistive Navigator"
    DISPLAY_WIDTH = 800
    DISPLAY_HEIGHT = 600
    HUD_HEIGHT = 100

    # --- Audio Spatial Cues ---
    AUDIO_CUES_ENABLED = True
    BEEP_FREQ_LEFT = 440            # Hz
    BEEP_FREQ_RIGHT = 880           # Hz
    BEEP_DURATION_MS = 120

    # --- Logging ---
    LOG_ENABLED = True
    LOG_DIR = "logs"

    # --- LLM ---
    LLM_MODEL = "llama3-70b-8192"
    LLM_TEMPERATURE = 0

    # --- Hazard labels (priority objects) ---
    HAZARD_LABELS = {"car", "truck", "bus", "motorcycle", "bicycle", "person", "dog", "stairs"}
    TARGET_LABELS = {"bottle", "chair", "cup", "laptop", "phone", "book", "backpack"}
