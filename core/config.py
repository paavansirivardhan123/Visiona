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
    #  Distance Filtering (Step-Based for Mobile)
    #  1 step ≈ 2.5 feet ≈ 0.75 meters
    # ------------------------------------------------------------------ #
    METERS_PER_STEP = 0.75
    MAX_DISTANCE_M  = 10.0 * METERS_PER_STEP  # ~7.5m (10 steps)
    CONSIDER_MAX_M  = 7.0 * METERS_PER_STEP   # ~5.25m (7 steps)
    HIGH_PRIORITY_M = 3.0 * METERS_PER_STEP   # ~2.25m (3 steps)

    # ------------------------------------------------------------------ #
    #  Heuristic fallback (used when MiDaS unavailable)
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
    #  Environment variables (API Keys)
    # ------------------------------------------------------------------ #
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
    MAPS_API_KEY  = os.getenv("MAPS_API_KEY", "")

    # ------------------------------------------------------------------ #
    #  Hardware / Hardware Config
    # ------------------------------------------------------------------ #
    MIC_DEVICE_INDEX = None  # Change this to an integer (e.g., 1) if default mic fails

    # ------------------------------------------------------------------ #
    #  Object Grouping
    # ------------------------------------------------------------------ #
    GROUP_THRESHOLD = 5

    # ------------------------------------------------------------------ #
    #  Threat Scoring & Penalty Heatmap
    # ------------------------------------------------------------------ #
    OBJECT_PRIORITY = {
        "person": 10, "child": 10,
        "car": 9, "truck": 9, "bus": 9, "motorcycle": 8, "bicycle": 7,
        "dog": 6, "cat": 5,
        "chair": 3, "dining table": 3, "bottle": 2, "cup": 2,
    }
    DEFAULT_PRIORITY = 1
    
    THREAT_HIGH_THRESHOLD = 70.0  # Above this score is an Extreme Threat
    PENALTY_APPLY         = 30.0  # Points deducted from a direction when spoken
    PENALTY_DECAY         = 3.0   # Points recovered per second

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
    SPEECH_COOLDOWN   = 2.5   # Warning repeat delay
    SEMANTIC_COOLDOWN = 5.0   # Normal object repeat delay (reduced from 12s)
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
    #  LLM (optional)
    # ------------------------------------------------------------------ #
    LLM_MODEL       = "llama3-70b-8192"
    LLM_TEMPERATURE = 0
    LLM_COOLDOWN    = 5.0

    # ------------------------------------------------------------------ #
    #  Monocular Depth (Depth Anything v2)
    # ------------------------------------------------------------------ #
    DEPTH_MODEL_ID      = "depth-anything/Depth-Anything-V2-Small-hf"
    MIDAS_DEFAULT_SCALE = 1.0
    DEPTH_SMOOTH_FRAMES = 5

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
    FRAME_BUDGET_MS = 2000  # MiDaS on CPU needs up to 200ms, YOLO ~100ms
