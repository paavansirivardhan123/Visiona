class Config:
    """Centralized configuration."""

    SOURCES = {
        "FRONT": "sample-vid/3Dir/vid1/FRONT.mp4",
        "LEFT":  "sample-vid/3Dir/vid1/LEFT.mp4",
        "RIGHT": "sample-vid/3Dir/vid1/RIGHT.mp4",
        "BACK":  None,
    }

    MODEL_PATH     = "yolo11n.pt"
    IMG_SIZE       = 640
    CONF_THRESHOLD = 0.35
    FRAME_SKIP     = 2

    METERS_PER_STEP = 0.75
    MAX_DISTANCE_M  = 10.0 * METERS_PER_STEP
    CONSIDER_MAX_M  = 7.0 * METERS_PER_STEP
    HIGH_PRIORITY_M = 3.0 * METERS_PER_STEP

    FOCAL_LENGTH_PX     = 700
    AVG_OBJECT_WIDTH_CM = 30
    KNOWN_WIDTHS_CM = {
        "person": 50, "car": 180, "truck": 250, "bus": 250,
        "bicycle": 60, "motorcycle": 80, "dog": 40, "cat": 25,
        "bottle": 8, "chair": 50, "cup": 8, "laptop": 35,
        "backpack": 35, "book": 20, "dining table": 80,
    }

    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
    MAPS_API_KEY  = os.getenv("MAPS_API_KEY", "")

    GROUP_THRESHOLD = 5

    OBJECT_PRIORITY = {
        "person": 10, "child": 10,
        "car": 9, "truck": 9, "bus": 9, "motorcycle": 8, "bicycle": 7,
        "dog": 6, "cat": 5,
        "chair": 3, "dining table": 3, "bottle": 2, "cup": 2,
    }
    DEFAULT_PRIORITY = 1
    
    THREAT_HIGH_THRESHOLD = 70.0
    PENALTY_APPLY         = 30.0
    PENALTY_DECAY         = 3.0

    BEEP_LEVELS = [
        (0.5,  2000, 80),
        (1.0,  1500, 100),
        (1.5,  1000, 120),
    ]

    SPEECH_RATE       = 185
    SPEECH_COOLDOWN   = 2.5
    SEMANTIC_COOLDOWN = 5.0
    MAX_MESSAGES      = 3

    WINDOW_NAME = "Visiona AI"
    DISPLAY_W   = 640
    DISPLAY_H   = 480
    HUD_H       = 90

    LOG_ENABLED = True
    LOG_DIR     = "logs"

    LLM_MODEL       = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE = 0
    LLM_COOLDOWN    = 5.0

    DEPTH_MODEL_ID      = "depth-anything/Depth-Anything-V2-Small-hf"
    MIDAS_DEFAULT_SCALE = 1.0
    DEPTH_SMOOTH_FRAMES = 5

    DEPTH_SPIKE_THRESHOLD = 1.5
    KF_PROCESS_NOISE      = 0.01
    KF_MEASUREMENT_NOISE  = 0.1

    TRACK_HISTORY_LEN = 30
    TRACK_MAX_AGE     = 10
    TRACKER_BACKEND   = "bytetrack"

    SPEED_MIN_THRESHOLD  = 0.2
    LATERAL_THRESHOLD_PX = 20
    SPEED_SMOOTH_FRAMES  = 5

    TTC_SMOOTH_ALPHA   = 0.3
    TTC_WARN_THRESHOLD = 3.0

    FRAME_BUDGET_MS = 2000
