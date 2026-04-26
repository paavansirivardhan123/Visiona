<div align="center">

# 🦯 Visiona AI

### Intelligent Assistive Navigation for the Visually Impaired

*Real-time object detection · Monocular depth estimation · LLM reasoning · Voice interaction · GPS navigation*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8n-Ultralytics-8A2BE2?style=flat-square)](https://ultralytics.com)
![Depth Anything V2](https://img.shields.io/badge/Depth_Anything_V2-HuggingFace-FFD21E?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-Groq_LLaMA3-00A67E?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)

</div>

---

## Overview 🚀

Visiona AI is a real-time assistive navigation platform for blind and visually impaired users. It processes live camera feeds, detects and tracks surrounding objects, estimates distance from a single RGB camera, and delivers concise spoken guidance — entirely hands-free.

Users speak naturally; Visiona listens, reasons, and responds with context-aware audio.

No stereo cameras. No LiDAR. No custom sensors. Just a standard RGB camera and a modern laptop.

---

## Demo Output

```
[FRONT] person ahead   →  "Person ahead approaching"
[FRONT] 6 persons      →  "Group of persons ahead"
[ALERT] car at 0.8m    →  Beep + "Warning: car ahead coming fast"
[VOICE] "I need water" →  "Goal water is active. Searching for bottle, cup, sink..."
[GPS]   "Take me to the pharmacy" → "Shortest path: 400m, 5 minutes walking"
```

---

## Features

**🧭 Spatial Awareness**
- 🔍 Detects 80+ object classes in real time using YOLOv8n
- 📏 Estimates metric distance using Depth Anything V2 (monocular, no special hardware)
- 🧷 Tracks objects across frames with a ByteTrack IoU tracker
- ⚡ Computes per-object speed, motion direction, and time-to-collision

**🔊 Intelligent Audio Guidance**
- 🗣️ Speaks compact, natural descriptions: "2 persons ahead", "Group of cars left"
- 📌 Priority queue (max-heap) ensures the most dangerous object is announced first
- ⚠️ Dynamic threat scoring: distance × object weight × approach speed × TTC (static objects filtered from false alarms)
- ⏱️ Smart Cooldowns: Strict 7-second ambient cooldowns and 5-second goal repeat intervals
- 🎯 Step-Based Tracking: Tracks record distances in distinct "steps" to eliminate noise from sensor fluctuation
- 🌡️ Penalty heatmap prevents one direction from monopolizing audio
- 🔔 Proximity beep alerts with frequency scaling by distance

**🎙️ Voice Interaction (Push-To-Talk)**
- Hold `V` to speak a general command or question
- Hold `G` to request GPS navigation
- Hold `R` to engage "Remember Mode" — point the camera at any custom object/person to capture exact image data
- Zero-latency mic pre-warming — no activation delay
- Audio ducking: speech volume lowers while the mic is open

**🧠 LLM Reasoning (Groq LLaMA 3.3 70B)**
- Routes voice commands to the correct tool automatically
- Answers questions about the current scene and recent activity
- Sets persistent goals ("I need water") and proactively alerts when relevant objects appear
- Contextual Arrival: Generates comforting, environment-aware guidance when you reach your destination (<= 1 step)
- Confirmation system asks before executing ambiguous requests
- Graceful offline fallback when the API key is missing

**🗂️ Memory System**
- 15-minute rolling visual memory of detected objects
- Custom object labeling: "This is my water bottle" saved as an alias
- **Real-Time Object "Training"**: Mathematical MobileNetV3 vector embeddings immediately swap generic YOLO labels (e.g. "person") for your custom aliases (e.g. "Paavan") on-the-fly, requiring zero fine-tuning epochs and fully avoiding catastrophic forgetting
- Goal system tracks user needs and scans for matching items

**🧭 GPS Navigation**
- Google Maps walking directions via voice command
- Selects shortest route from multiple alternatives
- Returns step-by-step instructions as spoken audio

**📷 Multi-Camera Support**
- Up to 4 directional feeds: FRONT, LEFT, RIGHT, BACK
- Unified 2x2 grid HUD display
- Each feed processed independently with a shared YOLO model

---

## Tech Stack

| Layer | Technology |
|---|---|
| Object Detection | YOLOv8n (Ultralytics) |
| Depth Estimation | Depth Anything V2 Small (HuggingFace Transformers) |
| Object Tracking | ByteTrack (IoU-based, pure NumPy) |
| Noise Reduction | 1D Kalman Filter (per-track) |
| Egomotion Detection | Optical Flow (OpenCV) |
| LLM Reasoning | Groq API — LLaMA 3.3 70B Versatile |
| Agent Framework | LangChain Core (tool binding) |
| Text-to-Speech | pyttsx3 (SAPI5, offline) |
| Speech Recognition | SpeechRecognition + PyAudio |
| Keyboard Listener | pynput |
| GPS Navigation | Google Maps Directions API |
| Video Processing | OpenCV |
| Deep Learning | PyTorch |
| Custom Recognition | MobileNetV3 Vector Embeddings (torchvision / Cosine Similarity) |
| Package Management | uv (Python package installer) |

---

## Project Structure

```
VISIONA/
├── main.py                         Entry point — camera feeds, main loop, PTT
├── core/
│   ├── config.py                   All settings — single source of truth
│   ├── detection.py                Detection dataclass with threat_score property
│   ├── priority_queue.py           Max-heap priority queue
│   ├── memory.py                   VisionMemory (15-min rolling) + GoalSystem
├── perception/
│   ├── vision.py                   YOLO + async depth pipeline
│   ├── mono_depth.py               Depth Anything V2 inference + calibration
│   ├── tracker.py                  ByteTrack IoU tracker with Kalman per track
│   └── egomotion.py                Optical flow user motion detection
├── kinematics/
│   ├── heatmap.py                  Object grouping + speech message builder
│   ├── speed.py                    Speed estimation + motion classification
│   ├── ttc.py                      Time-To-Collision calculator (EMA smoothed)
│   └── kalman.py                   1D Kalman filter for depth noise reduction
├── audio/
│   ├── speech.py                   Priority TTS queue + ducking + emergency bypass
│   ├── voice_input.py              Zero-latency PTT mic capture
│   └── alert.py                    Proximity beep alerts (pauseable)
├── agents/
│   ├── orchestrator.py             LangChain agent engine + confirmation system
│   └── tools/
│       └── vision_tools.py         9 LangChain tools (memory, search, GPS, goals)
├── services/
│   ├── google_maps.py              Google Maps Directions API wrapper
│   └── vector_db.py                Object signature storage (placeholder)
├── sample-vid/                     Test video files
├── yolov8n.pt                      YOLO model weights
├── pyproject.toml                  Dependencies (managed with uv)
├── .env                            API keys (not committed)
└── .env.example                    Template for .env
```

---

## System Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1 — Input Capture                                    │
│                                                             │
│  • CameraFeed reads frames from up to 4 video sources       │
│  • Frames resized to 1280px wide for YOLO processing        │
│  • Each direction (FRONT / LEFT / RIGHT / BACK) runs        │
│    independently in its own thread                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2 — Object Detection          perception/vision.py   │
│                                                             │
│  • YOLOv8n detects 80+ object classes (~50ms on CPU)        │
│  • Bounding boxes scaled to display coordinates             │
│  • Depth Anything V2 runs in background thread (~300ms)     │
│  • Cached depth applied immediately; heuristic fallback     │
│    used until depth model warms up                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3 — Tracking + Kinematics     perception/tracker.py  │
│                                      perception/egomotion.py│
│                                      kinematics/            │
│  • ByteTrack assigns persistent IDs via IoU matching        │
│  • Optical Flow detects user motion (walking/stationary)    │
│  • Egomotion compensation: distinguishes true object motion │
│    from camera movement (e.g. user walking forward)         │
│  • FeatureDB extracts crop embeddings & hot-swaps custom    │
│    labels (e.g. "person" -> "Paavan") via Cosine Sim        │
│  • Kalman filter smooths depth per track, rejects spikes    │
│  • Speed = ΔDistance / ΔTime  (metres per second)           │
│  • Motion classification: approaching/receding/lateral      │
│    (compensated for user movement)                          │
│  • TTC  = Distance / Speed    (EMA smoothed)                │
│  • threat_score = distance × weight + speed + TTC bonus     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4 — Priority + Grouping       core/priority_queue.py │
│                                      kinematics/heatmap.py  │
│  • Max-heap sorts by: TTC → distance → object type          │
│  • Penalty heatmap prevents one direction dominating audio  │
│  • Groups objects: "Group of persons ahead" / "2 cars left" │
│  • Enforces 7-second ambient and 5-second goal cooldowns    │
│  • Emergency TTC bypass always speaks critical warnings     │
│    (Filters out static objects to prevent false alarms)     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 5 — Output                    audio/                 │
│                                                             │
│  • SpeechEngine: priority queue, ducking, emergency bypass  │
│  • AlertSystem: proximity beeps, frequency scales by dist   │
│  • HUD: state badge, bounding boxes, TTC warning rings      │
│  • SessionLogger: full JSONL telemetry saved per session    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 6 — Voice + LLM               audio/voice_input.py   │
│                                      agents/orchestrator.py │
│  • Hold V / G → PyAudio pre-warmed stream (zero latency)    │
│  • SpeechRecognition → Google STT → plain text              │
│  • AgentEngine routes to the right LangChain tool:          │
│      query_past_detections  →  15-min visual memory         │
│      set_search_intent      →  YOLO target seeking          │
│      set_persistent_goal    →  proactive object scanning    │
│      calculate_route        →  Google Maps directions       │
│      save_object_signature  →  custom object labeling       │
│  • Groq LLaMA 3.3 70B synthesizes final spoken response     │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/yourname/visiona.git
cd visiona
```

**2. Install dependencies**

```bash
pip install uv
uv sync
```

> All dependencies are pinned in `pyproject.toml`. Key packages:
> `ultralytics` · `transformers` · `torch` · `opencv-python` · `langchain-groq` · `pyttsx3` · `SpeechRecognition` · `pyaudio` · `pynput`

**3. Configure API keys**

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY="your_groq_api_key_here"
MAPS_API_KEY="your_google_maps_api_key_here"
```

- Free Groq key: [console.groq.com](https://console.groq.com)
- Google Maps key: [console.cloud.google.com](https://console.cloud.google.com) — enable Directions API
- Both are optional. The system runs without them using offline fallbacks.

**4. Run**

```bash
python main.py
```

> On first run, Depth Anything V2 downloads ~100MB of model weights (cached after that).

The default config uses `sample-vid/sample3.mp4`. To use a live webcam, set `SOURCES = {"FRONT": 0}` in `core/config.py`.

> Windows note: If `pyaudio` fails, run `pip install pipwin && pipwin install pyaudio`

---

## Usage

### Controls

| Key | Action |
|---|---|
| Hold `V` | Open mic for general AI command or question |
| Hold `G` | Open mic for GPS navigation request |
| Hold `R` | Open mic for Remember Mode to capture and learn custom objects |
| `ESC` | Quit |

### Voice Commands

| Say this | What happens |
|---|---|
| "find chair" | YOLO starts searching for a chair |
| "I need water" | Sets persistent goal, searches for bottle/cup/sink |
| "where is the exit?" | Searches for door/stairs |
| "what is around me?" | Describes current scene |
| "take me to the pharmacy" | Google Maps walking directions |
| "this is my water bottle" | Labels the nearest bottle in memory |
| "where did I leave my keys?" | Searches 15-minute visual memory |
| Any question | Routed to LLM with full scene context |

### Configuration

All settings in `core/config.py`:

```python
# Video sources — use integer index for live cameras
SOURCES = {
    "FRONT": 0,      # webcam
    "LEFT":  None,   # disabled
}

# Distance zones
CONSIDER_MAX_M  = 5.25    # only describe objects within this range
HIGH_PRIORITY_M = 1.125   # triggers beep + priority alert

# Speech
SEMANTIC_COOLDOWN = 5.0   # seconds between repeated normal announcements
SPEECH_COOLDOWN   = 2.5   # seconds between repeated warnings

# Depth model
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
```

---

## How It Works

### Depth Estimation

Depth Anything V2 outputs relative inverse depth — higher values mean closer objects. Converted to metric metres:

```
metric_depth_m = (scale / depth_value) x 2.0
```

Scale is auto-calibrated on first frame using detected reference objects via the pinhole camera model.

### Threat Scoring

```python
score = (10.0 - distance_m) x object_weight
      + speed_mps x 20       # if approaching
      + 100.0                 # if TTC <= 3.5 seconds
```

Objects above `THREAT_HIGH_THRESHOLD = 50.0` trigger beep alerts and priority speech.

### Priority Queue

Max-heap sorted by three factors:

1. TTC — `1000 / ttc_sec` (lower TTC = highest urgency)
2. Distance — `(CONSIDER_MAX_M - distance_m) x 10`
3. Object type — `OBJECT_PRIORITY` weight

### Penalty Heatmap

After speaking about a direction, a penalty of 30 points is applied to it. Penalties decay at 3 points/second. This prevents the system from repeatedly announcing the same direction and ensures all directions get coverage.

### LLM Agent

1. Voice command arrives via PTT
2. LLM decides which tool(s) to call
3. If intent is ambiguous, asks "Asking for [intent]?" before executing
4. Tools execute (memory lookup, YOLO search, GPS, goal setting)
5. LLM synthesizes a single spoken response from tool results

---

## Key Design Decisions

**Async depth pipeline** — Depth Anything V2 runs in a background thread. The main loop never waits for it. This keeps video smooth at 30+ FPS while depth updates at ~3 FPS.

**YOLO on 1280px, not 4K** — Running YOLO on a 4K frame produces zero detections because objects are too small after internal downscaling. Resizing to 1280px first gives YOLO enough detail while being 4x faster.

**Zero-latency PTT** — The microphone stream is pre-warmed at startup. When the user presses V, recording starts instantly with no activation delay.

**Emergency speech bypass** — Critical alerts (vehicle < 2.5m, person < 1.0m) bypass the ducking system and play at full volume even while the mic is open.

**Offline-first design** — Every component has a graceful fallback. No API key means LLM disabled with direct scene description used. Depth model failure means heuristic bounding-box estimation. No mic means keyboard controls still work.

---

## Session Logs

Every session saved to `logs/session_YYYYMMDD_HHMMSS.jsonl`:

```json
{"event": "detection", "direction": "FRONT", "objects": [
  {"object": "person", "direction": "FRONT", "mode": "monocular",
   "distance_m": 1.4, "speed_mps": 0.8, "motion": "approaching",
   "ttc_sec": 1.8, "priority": "high"}
], "t": 4.21}
{"event": "speech", "messages": ["Warning: person ahead coming fast"], "t": 4.22}
```

---

## Roadmap

- [x] YOLOv8n real-time detection (4 camera feeds)
- [x] Depth Anything V2 monocular depth estimation
- [x] Auto scale calibration from reference objects
- [x] ByteTrack object tracking with Kalman filter
- [x] Optical flow egomotion detection
- [x] Motion compensation (user movement vs object movement)
- [x] Speed + motion classification (approaching/receding/lateral)
- [x] Time-To-Collision calculation
- [x] Max-heap priority queue (TTC → distance → type)
- [x] Dynamic threat scoring + penalty heatmap
- [x] Priority TTS queue with ducking + emergency bypass
- [x] Zero-latency PTT voice input
- [x] LangChain agent with 9 tools (Groq LLaMA 3.3 70B)
- [x] 15-minute visual memory + custom object labeling
- [x] MobileNetV3 vector embeddings for custom object recognition
- [x] Real-time label swapping (zero fine-tuning, no catastrophic forgetting)
- [x] Persistent goal system (proactive scanning)
- [x] Google Maps GPS navigation (shortest path selection)
- [x] Session logging (JSONL telemetry)
- [x] Remember Mode (R key) - capture and learn custom objects/people
- [x] Contextual curiosity (proactive assistance suggestions)
- [x] Busy road detection (vehicle density warnings)
- [x] Emergency verbal bypass (critical alerts override mic input)
- [x] Contextual goal arrival notifications & LLM guidance
- [x] Step-based spatial tracking for noise elimination
- [ ] GPU acceleration (CUDA)

## Author

**Author:** Paavan Siri Vardhan Narava
**Email:** naravapaavansirivardhan@gmail.com
 
---

<div align="center">
Built to make the world navigable for everyone.
</div>
