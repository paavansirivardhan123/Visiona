<div align="center">

# 🦯 Visiona AI

### Real-time AI Navigation Assistant for the Visually Impaired

*See the world through AI — obstacle detection, distance estimation, voice guidance, and GPS navigation in one system.*

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square)
![LLaMA3](https://img.shields.io/badge/LLM-LLaMA3--70B-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## What Is Visiona?

Visiona is an AI-powered assistive navigation system that gives visually impaired users a real-time audio description of their surroundings. It combines computer vision, distance estimation, large language model reasoning, voice input, and GPS navigation into a single system that runs on a laptop or Raspberry Pi with a camera.

The user simply speaks — "find the chair", "what's around me?", "take me to the hospital" — and Visiona responds with clear, spoken guidance.

---

## Key Features

| Feature | Description |
|---|---|
| Real-time object detection | YOLOv8s detects 80+ object classes at 30+ FPS |
| Distance estimation | Pinhole camera model gives real cm distances per object |
| 5-zone spatial grid | Far-left / Left / Ahead / Right / Far-right awareness |
| AI scene reasoning | Groq LLaMA3-70B generates contextual spoken guidance |
| Voice input | Speak commands hands-free — no keyboard needed |
| Free-form Q&A | Ask anything: "is there a car nearby?" and get an answer |
| Approaching object detection | Warns when objects are moving toward the user |
| Priority speech queue | Critical alerts interrupt normal guidance instantly |
| Spatial audio beeps | Different tones for left vs right hazards |
| Object tracking | Tracks objects across frames for motion awareness |
| Session logging | Every session saved to JSONL for analytics and training |
| GPS navigation | Turn-by-turn route guidance from one location to another *(coming soon)* |

---

## How It Works

```
Camera Feed
    │
    ▼
VisionSystem (YOLOv8)
    │  detects objects, estimates distance, classifies zones
    ▼
AINavigator
    ├── Fast Path: safety rules  →  immediate spoken alert
    └── Slow Path: LLM reasoning →  contextual guidance
    │
    ▼
SpeechEngine (pyttsx3)
    │  priority queue, stale-message dropping, spatial beeps
    ▼
User hears guidance
    │
    ▼
VoiceInputEngine (SpeechRecognition)
    │  listens in background, parses commands
    └── on_intent / on_scene_request / on_question → back to AINavigator
```

---

## Project Structure

```
visiona/
├── main.py                  # App entry point
├── core/
│   └── config.py            # All tunable settings
├── engines/
│   ├── vision.py            # YOLOv8 detection + HUD rendering
│   ├── depth.py             # Pinhole camera distance estimation
│   ├── navigator.py         # Safety rules + LLM reasoning
│   ├── speech.py            # Priority TTS queue + spatial beeps
│   ├── voice_input.py       # Background mic listener + command parser
│   ├── memory.py            # Object tracking + scene history
│   └── logger.py            # Session telemetry (JSONL)
├── models/
│   └── detection.py         # Detection dataclass
├── logs/                    # Auto-created session logs
├── sample-vid/              # Test video files
├── yolov8s.pt               # YOLOv8 model weights
├── pyproject.toml
└── .env                     # API keys
```

---

## Quick Start

**1. Clone and install**

```bash
git clone https://github.com/yourname/visiona.git
cd visiona
uv sync
```

**2. Add your Groq API key**

```bash
# .env
GROQ_API_KEY="your_key_here"
```

Get a free key at [console.groq.com](https://console.groq.com) — the free tier is enough.

**3. Run**

```bash
uv run main.py
```

---

## Installation Details

All dependencies are managed with `uv`.

```bash
# Install uv (if not already installed)
pip install uv

# Install all project dependencies
uv sync

# Add a new package
uv add package-name
```

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| ultralytics | 8.2.0 | YOLOv8 object detection |
| opencv-python | 4.9.0.80 | Video capture + rendering |
| torch | 2.2.2 | Deep learning backend |
| pyttsx3 | 2.90 | Text-to-speech (offline) |
| langchain-groq | latest | LLM integration |
| SpeechRecognition | 3.10+ | Voice input |
| pyaudio | 0.2.13+ | Microphone access |
| python-dotenv | latest | API key management |
| numpy | 1.26.4 | Numerical operations |

> **Windows note:** If `pyaudio` fails to install, run:
> ```bash
> pip install pipwin
> pipwin install pyaudio
> ```

---

## Voice Commands

Visiona listens continuously in the background. Just speak naturally.

| Say this | What happens |
|---|---|
| `"find chair"` | Starts searching for a chair, guides you toward it |
| `"find person"` | Searches for a person |
| `"find door"` | Searches for a door |
| `"find stairs"` | Searches for stairs |
| `"walk forward"` / `"forward"` | Returns to general navigation mode |
| `"what's around"` | Announces everything currently detected |
| `"describe"` / `"what do you see"` | Same as above |
| `"where is the car?"` | Free-form AI answer based on current scene |
| `"should I turn left?"` | AI reasons about the scene and answers |
| Any question (3+ words) | Routed to LLM automatically |

The terminal shows exactly what was heard and what action was taken:

```
  [Voice] ✅ Heard: "find the chair"
  [Voice] → Intent: chair
```

---

## Keyboard Controls

For developers and sighted operators testing the system.

| Key | Action |
|---|---|
| `1` | Intent: walk forward |
| `2` | Find: bottle |
| `3` | Find: chair |
| `4` | Find: person |
| `5` | Find: car |
| `6` | Find: laptop |
| `7` | Find: backpack |
| `D` | Find: door |
| `S` | Find: stairs |
| `H` | Announce current scene |
| `ESC` | Quit |

---

## HUD Display

The live video window shows:

- State badge — `SCANNING` / `ALERT` / `AVOIDING` / `GUIDING` / `SEARCHING`
- Current intent and AI reasoning text
- Bounding boxes colored by distance:
  - 🔴 Red — critical (< 80cm)
  - 🟠 Orange — near (< 150cm)
  - 🟡 Yellow — hazard at medium range
  - 🟢 Green — target object found
  - ⚪ Grey — neutral object
- Confidence bar under each bounding box
- Hazard pulse ring for close threats
- 5-zone grid lines
- MIC indicator (green = listening, grey = idle)
- FPS counter

---

## Configuration

All settings are in `core/config.py`. Key values to tune:

```python
# Camera
FOCAL_LENGTH_PX = 700       # Calibrate for your camera for accurate distances

# Distance zones (cm)
DIST_CRITICAL = 80          # Triggers immediate STOP
DIST_NEAR = 150             # Triggers avoidance routing
DIST_MEDIUM = 300           # Heads-up warning

# Performance
FRAME_SKIP = 2              # Process every Nth frame (lower = more CPU)
CONF_THRESHOLD = 0.35       # Detection confidence cutoff

# Timing
SPEECH_COOLDOWN = 2.0       # Seconds between normal speech
LLM_COOLDOWN = 3.5          # Seconds between LLM calls

# Audio
BEEP_FREQ_LEFT = 440        # Hz — left hazard tone
BEEP_FREQ_RIGHT = 880       # Hz — right hazard tone
```

---

## GPS Navigation *(Planned)*

The next major feature is turn-by-turn GPS navigation — guiding the user from their current location to a destination entirely by voice.

**Planned flow:**

```
User says: "Take me to the nearest pharmacy"
    │
    ▼
GPS Engine gets current coordinates (device GPS / IP fallback)
    │
    ▼
Routing API (OpenRouteService / Google Maps) calculates walking route
    │
    ▼
Navigator combines GPS waypoints with live obstacle detection
    │  "In 20 meters, turn right onto Main Street"
    │  "Stop — person directly ahead"
    ▼
User arrives safely
```

**Planned voice commands:**

| Say this | Action |
|---|---|
| `"take me to the hospital"` | Route to nearest hospital |
| `"navigate to [address]"` | Route to specific address |
| `"where am I?"` | Announce current street/location |
| `"how far to destination?"` | Remaining distance |
| `"cancel navigation"` | Return to obstacle-avoidance mode |

**Planned engines to add:**

```
engines/
├── gps.py          # Location provider (device GPS / IP geolocation fallback)
└── routing.py      # Waypoint calculation + turn-by-turn instruction generator
```

**APIs being evaluated:**
- [OpenRouteService](https://openrouteservice.org/) — free, pedestrian routing
- [Nominatim](https://nominatim.org/) — free geocoding (address → coordinates)
- Google Maps Directions API — most accurate, paid

---

## Session Logs

Every session is saved to `logs/session_YYYYMMDD_HHMMSS.jsonl`.

Each line is a JSON event:

```json
{"event": "detection", "objects": [{"label": "person", "position": "ahead", "distance_cm": 120.5, "confidence": 0.87, "is_hazard": true}], "t": 4.21}
{"event": "instruction", "text": "Person ahead at 120 centimeters. Turn left.", "source": "agent", "priority": false, "t": 4.22}
{"event": "state_change", "state": "AVOIDING", "reasoning": "Avoiding person, turning left.", "t": 4.22}
```

This data is valuable for:
- Debugging and tuning thresholds
- Analyzing real-world usage patterns
- Future fine-tuning of the navigation model

---

## Roadmap

- [x] YOLOv8 real-time detection
- [x] Pinhole camera distance estimation
- [x] 5-zone spatial grid
- [x] LLM scene reasoning (Groq LLaMA3)
- [x] Priority speech queue with stale-message dropping
- [x] Voice input (hands-free commands + free-form Q&A)
- [x] Object tracking + approaching detection
- [x] Spatial audio beeps
- [x] Session logging
- [ ] GPS location provider
- [ ] Turn-by-turn pedestrian routing
- [ ] Landmark recognition ("you are near a bus stop")
- [ ] Mobile app (Android/iOS) wrapper
- [ ] Offline LLM fallback (Ollama)
- [ ] Wearable camera support (glasses mount)

---

## License

MIT — free to use, modify, and build on.

---

<div align="center">
Built to make the world navigable for everyone.
</div>
