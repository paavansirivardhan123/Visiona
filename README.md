<div align="center">

# 🦯 Visiona AI

### Real-time Assistive Vision System for Blind Users

*Four-camera spatial awareness · MiDaS depth estimation · Voice guidance · Priority alerts*

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8n-Ultralytics-purple?style=flat-square)
![MiDaS](https://img.shields.io/badge/MiDaS-Intel-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## What Is Visiona?

Visiona is a real-time object detection and spatial awareness system built for blind users. It processes up to four directional camera feeds simultaneously, detects nearby objects, estimates their distance using monocular depth estimation, and delivers clear spoken audio guidance — all without any keyboard interaction required.

The user simply listens. Visiona speaks.

---

## How It Works

```
4 Camera Feeds (FRONT / LEFT / RIGHT / BACK)
        │
        ▼
  YOLO Detection (1280px, ~50ms)
        │
        ▼
  MiDaS Depth Estimation (async background thread)
        │  scale = real_distance × D_ref
        │  depth = scale / midas_value
        ▼
  Distance Filter (only ≤ 1.7m described, ≤ 1.0m = HIGH PRIORITY)
        │
        ▼
  ByteTrack Object Tracking (persistent IDs across frames)
        │
        ▼
  Speed + Motion Classification (approaching / moving away / lateral)
        │
        ▼
  TTC — Time To Collision (distance / speed)
        │
        ▼
  Priority Queue (TTC → distance → object type)
        │
        ▼
  Object Grouping + Speech Messages
        │  "3 people in front at 1.4 meters"
        │  "Group of people on the left"
        │  "Warning: Person very close in front"
        ▼
  TTS Audio Output + Beep Alerts
```

---

## Project Structure

```
visiona/
├── main.py                    # App entry point, camera feeds, main loop
├── core/
│   └── config.py              # All tunable settings
├── engines/
│   ├── vision.py              # YOLO + async MiDaS pipeline
│   ├── depth.py               # Depth engine interface
│   ├── mono_depth.py          # MiDaS monocular depth + calibration
│   ├── tracker.py             # ByteTrack IoU object tracker
│   ├── speed.py               # Speed estimation + motion classification
│   ├── ttc.py                 # Time-To-Collision calculator
│   ├── kalman.py              # Per-track Kalman filter (noise reduction)
│   ├── grouping.py            # Object grouping + speech message builder
│   ├── alert.py               # Proximity beep alerts
│   ├── speech.py              # Priority TTS queue
│   ├── voice_input.py         # Background mic listener
│   └── logger.py              # JSONL session logging
├── models/
│   ├── detection.py           # Detection dataclass
│   └── priority_queue.py      # Max-heap priority queue (DSA)
├── sample-vid/                # Test video files
│   ├── front.mp4
│   ├── left.mp4
│   ├── right.mp4
│   └── back.mp4
├── yolov8n.pt                 # YOLO model weights
├── pyproject.toml
└── .env                       # API keys (optional)
```

---

## Quick Start

**1. Install dependencies**

```bash
uv sync
```

**2. Run**

```bash
uv run main.py
```

On first run, MiDaS downloads ~100MB of model weights (cached after that). The system starts speaking within a few seconds.

**3. Use live cameras instead of video files**

Edit `core/config.py`:

```python
SOURCES = {
    "FRONT": 0,   # webcam index
    "LEFT":  1,
    "RIGHT": 2,
    "BACK":  None,   # disabled
}
```

---

## Installation

All dependencies managed with `uv`.

```bash
# Install uv
pip install uv

# Install all project dependencies
uv sync

# Add a new package
uv add package-name
```

| Package | Version | Purpose |
|---|---|---|
| ultralytics | ≥8.3.0 | YOLOv8n object detection |
| opencv-python | 4.9.0.80 | Video capture + rendering |
| torch | 2.2.2 | Deep learning backend (YOLO + MiDaS) |
| torchvision | 0.17.2 | Image transforms |
| timm | ≥1.0.26 | MiDaS model backbone |
| pyttsx3 | 2.90 | Offline text-to-speech |
| SpeechRecognition | ≥3.10.0 | Voice command input |
| pyaudio | ≥0.2.13 | Microphone access |
| numpy | 1.26.4 | Numerical operations |

> Windows note: if `pyaudio` fails, run `pip install pipwin && pipwin install pyaudio`

---

## Voice Commands

Visiona listens continuously in the background. No button press needed.

| Say this | What happens |
|---|---|
| `"find chair"` | Searches for a chair, guides toward it |
| `"find person"` | Searches for a person |
| `"find door"` | Searches for a door |
| `"find stairs"` | Searches for stairs |
| `"walk forward"` | Returns to general navigation mode |
| `"what is around"` | Announces everything currently detected |
| `"describe"` | Same as above |
| `"where is the car?"` | Answers based on current scene |
| Any 3+ word question | Routed to scene description |

---

## Keyboard Controls

For developers and sighted operators.

| Key | Action |
|---|---|
| `H` | Announce current scene |
| `ESC` | Quit |

---

## Audio Output Examples

| Situation | What you hear |
|---|---|
| 6 people detected in front | "Group of people in front at 1.5 meters" |
| 3 people on the left | "3 people on the left at 1.2 meters" |
| Person within 1 meter | Beep + "Person very close in front, 0.8 meters" |
| Person walking toward you | "Person approaching in front at 1.2 m/s" |
| TTC under 3 seconds | "Warning: Person approaching in front at 1.4 m/s" |
| Object moving away | "a car in front at 1.6 meters, moving away" |

---

## Distance Zones

| Zone | Range | Behaviour |
|---|---|---|
| High Priority | ≤ 1.0 m | Beep alert + immediate voice warning |
| Describe | ≤ 1.7 m | Included in speech output |
| Ignore | > 3.0 m | Filtered out entirely |

---

## HUD Display

Each camera window shows:

- State badge — `SCANNING` / `ALERT` / `AVOIDING` / `GUIDING`
- Last spoken message
- Bounding boxes colored by distance:
  - 🔴 Red — high priority (≤ 1.0m)
  - 🟠 Orange — near (≤ 2.0m)
  -  Green — within range
- Confidence bar under each box
- TTC warning ring (red circle) when collision imminent
- Object count + calibration status
- Direction label (FRONT / LEFT / RIGHT / BACK)
- Mic indicator dot (green = listening)

---

## Configuration

All settings in `core/config.py`.

```python
# Distance thresholds
MAX_DISTANCE_M  = 3.0    # ignore beyond this
CONSIDER_MAX_M  = 1.7    # only describe within this
HIGH_PRIORITY_M = 1.0    # triggers beep + priority alert

# Performance
FRAME_SKIP      = 2      # process every Nth frame
FRAME_BUDGET_MS = 2000   # max ms before skipping depth

# Speech
SPEECH_COOLDOWN = 2.5    # seconds between announcements
MAX_MESSAGES    = 3      # max messages per cycle

# Depth
MIDAS_MODEL_TYPE = "MiDaS_small"   # lightweight, CPU-friendly

# Tracking
TRACKER_BACKEND  = "bytetrack"
TRACK_MAX_AGE    = 10              # frames before stale track removed

# TTC
TTC_WARN_THRESHOLD = 3.0           # seconds — prepend "Warning:"
```

---

## Depth Estimation

Visiona uses **MiDaS** (Intel) for monocular depth estimation — no stereo camera or LiDAR required.

MiDaS outputs inverse depth (higher value = closer object). The correct formula is:

```
metric_depth_m = scale / midas_value
```

Scale is auto-calibrated on first frame using detected reference objects:

```
scale = real_distance_m × D_ref
```

Where `D_ref` is the MiDaS value inside the object's bounding box and `real_distance_m` is estimated from the object's known real-world width using the pinhole camera model.

**Limitations:**
- Accuracy depends on calibration quality
- Lighting conditions affect MiDaS output
- Cannot guarantee centimetre-level precision
- First run requires internet to download model weights (~100MB, cached)

---

## Priority Queue (DSA)

Detections are sorted by a max-heap priority queue with three factors:

1. **TTC** — lower time-to-collision = highest urgency (`1000 / ttc_sec`)
2. **Distance** — closer objects score higher
3. **Object type** — `person > car > bicycle > dog > chair > bottle`

This ensures the most dangerous object is always announced first.

---

## Session Logs

Every session is saved to `logs/session_YYYYMMDD_HHMMSS.jsonl`.

```json
{"event": "detection", "direction": "FRONT", "objects": [
  {"object": "person", "direction": "FRONT", "mode": "monocular",
   "distance_m": 1.4, "speed_mps": 0.8, "motion": "approaching",
   "ttc_sec": 1.8, "priority": "high"}
], "t": 4.21}
{"event": "speech", "messages": ["Warning: Person approaching in front at 0.8 m/s"], "t": 4.22}
```

Useful for debugging, tuning thresholds, and future model training.

---

## Roadmap

- [x] YOLOv8n real-time detection (4 camera feeds)
- [x] MiDaS monocular depth estimation
- [x] Auto scale calibration from reference objects
- [x] ByteTrack object tracking
- [x] Speed + motion classification
- [x] Time-To-Collision calculation
- [x] Kalman filter noise reduction
- [x] Priority queue (TTC → distance → object type)
- [x] Object grouping ("Group of people", "3 chairs")
- [x] Priority TTS queue with stale-message dropping
- [x] Beep alerts scaled by proximity
- [x] Voice commands (hands-free)
- [x] Session logging (JSONL)
- [x] Async MiDaS (non-blocking display)
- [ ] GPS turn-by-turn navigation
- [ ] Landmark recognition
- [ ] Offline LLM fallback (Ollama)
- [ ] Mobile app wrapper
- [ ] Wearable camera support

---

## Author

Paavan Siri Vardhan Narava  
naravapaavansirivardhan@gmail.com

---

## License

MIT — free to use, modify, and build on.

---

<div align="center">
Built to make the world navigable for everyone.
</div>
