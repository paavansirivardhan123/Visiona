import os
from dotenv import load_dotenv
load_dotenv()


class Config:
    """Centralized configuration — single source of truth."""

    # ------------------------------------------------------------------ #
    #  Video Sources
    #  Use integer index (0,1,2,3) for live cameras.
    #  Set to None to disable a direction.
    # ------------------------------------------------------------------ #
    SOURCES = {
        "FRONT": "sample-vid/sample3.mp4",
        "LEFT":  None,
        "RIGHT": None,
        "BACK":  None,
    }

    # ------------------------------------------------------------------ #
    #  YOLO
    # ------------------------------------------------------------------ #
    MODEL_PATH     = "yolov8n.pt"
    IMG_SIZE       = 640
    CONF_THRESHOLD = 0.35
    FRAME_SKIP     = 2

    # ------------------------------------------------------------------ #
    #  Monocular Depth (Depth Anything V2 via HuggingFace Transformers)
    # ------------------------------------------------------------------ #
    DEPTH_MODEL_ID      = "depth-anything/Depth-Anything-V2-Small-hf"
    MIDAS_DEFAULT_SCALE = 2000.0   # scale / midas_val = metres; tuned for typical scenes
    DEPTH_SMOOTH_FRAMES = 5

    # ------------------------------------------------------------------ #
    #  Distance Filtering
    #  1 step ≈ 0.75 metres
    # ------------------------------------------------------------------ #
    METERS_PER_STEP = 0.75
    MAX_DISTANCE_M  = 10.0 * METERS_PER_STEP   # ~7.5 m
    CONSIDER_MAX_M  = 7.0  * METERS_PER_STEP   # ~5.25 m
    HIGH_PRIORITY_M = 1.5  * METERS_PER_STEP   # ~1.125 m

    # ------------------------------------------------------------------ #
    #  Heuristic fallback (used when depth model unavailable)
    # ------------------------------------------------------------------ #
    FOCAL_LENGTH_PX     = 700
    AVG_OBJECT_WIDTH_CM = 30
    KNOWN_WIDTHS_CM = {
        "person": 50, "car": 180, "truck": 250, "bus": 250,
        "bicycle": 60, "motorcycle": 80, "dog": 40, "cat": 25,
        "bottle": 8, "chair": 50, "cup": 8, "laptop": 35,
        "backpack": 35, "book": 20, "dining table": 80,
    }

    # ------------------------------------------------------------------ #
    #  API Keys (loaded from .env)
    # ------------------------------------------------------------------ #
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    MAPS_API_KEY = os.getenv("MAPS_API_KEY", "")

    # ------------------------------------------------------------------ #
    #  AI Reasoning (LangChain + Groq)
    # ------------------------------------------------------------------ #
    LLM_MODEL       = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE = 0.1
    LLM_COOLDOWN    = 5.0

    # ------------------------------------------------------------------ #
    #  Hardware
    # ------------------------------------------------------------------ #
    MIC_DEVICE_INDEX = None   # Set to int (e.g. 1) if default mic fails

    # ------------------------------------------------------------------ #
    #  Object Grouping
    # ------------------------------------------------------------------ #
    GROUP_THRESHOLD = 5

    # ------------------------------------------------------------------ #
    #  Threat Scoring & Penalty Heatmap
    # ------------------------------------------------------------------ #
    OBJECT_PRIORITY = {
        "person": 12, "child": 12,
        "car": 9, "truck": 9, "bus": 9, "motorcycle": 8, "bicycle": 7,
        "dog": 6, "cat": 5,
        "chair": 3, "dining table": 3, "bottle": 2, "cup": 2,
    }
    DEFAULT_PRIORITY      = 1
    THREAT_HIGH_THRESHOLD = 50.0
    PENALTY_APPLY         = 30.0
    PENALTY_DECAY         = 3.0

    # ------------------------------------------------------------------ #
    #  Alert beep levels  (distance_m, freq_hz, duration_ms)
    # ------------------------------------------------------------------ #
    BEEP_LEVELS = [
        (0.5,  2000, 80),
        (1.0,  1500, 100),
        (1.5,  1000, 120),
    ]

    # ------------------------------------------------------------------ #
    #  Speech
    # ------------------------------------------------------------------ #
    SPEECH_RATE       = 185
    SPEECH_COOLDOWN   = 2.5    # Warning / priority repeat delay (seconds)
    SEMANTIC_COOLDOWN = 5.0    # Normal object repeat delay (seconds)
    MAX_MESSAGES      = 3

    # ------------------------------------------------------------------ #
    #  UI
    # ------------------------------------------------------------------ #
    WINDOW_NAME = "Visiona AI"
    DISPLAY_W   = 640
    DISPLAY_H   = 480
    HUD_H       = 90

    # ------------------------------------------------------------------ #
    #  Logging
    # ------------------------------------------------------------------ #
    LOG_ENABLED = True
    LOG_DIR     = "logs"

    # ------------------------------------------------------------------ #
    #  Kalman Filter
    # ------------------------------------------------------------------ #
    DEPTH_SPIKE_THRESHOLD = 1.5
    KF_PROCESS_NOISE      = 0.01
    KF_MEASUREMENT_NOISE  = 0.1

    # ------------------------------------------------------------------ #
    #  Object Tracking
    # ------------------------------------------------------------------ #
    TRACK_HISTORY_LEN = 30
    TRACK_MAX_AGE     = 10
    TRACKER_BACKEND   = "bytetrack"

    # ------------------------------------------------------------------ #
    #  Speed Estimation
    # ------------------------------------------------------------------ #
    SPEED_MIN_THRESHOLD  = 0.2
    LATERAL_THRESHOLD_PX = 20
    SPEED_SMOOTH_FRAMES  = 5

    # ------------------------------------------------------------------ #
    #  Time-To-Collision
    # ------------------------------------------------------------------ #
    TTC_SMOOTH_ALPHA   = 0.3
    TTC_WARN_THRESHOLD = 3.0

    # ------------------------------------------------------------------ #
    #  Performance
    # ------------------------------------------------------------------ #
    FRAME_BUDGET_MS = 2000
