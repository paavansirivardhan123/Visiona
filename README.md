# Visiona AI — Intelligent Assistive Navigation System

## Overview

Visiona is a real-time, multi-camera computer vision system designed to empower visually impaired users with intelligent spatial awareness. Combining YOLO object detection, depth estimation, motion tracking, and LangChain-powered AI reasoning, Visiona transforms raw visual data into conversational guidance delivered through text-to-speech.

The system processes synchronized multi-camera feeds, detects objects, calculates distance and collision risk, and communicates findings through intelligible audio alerts and natural language responses.

## Demo / Output

*Real-time video stream processing at 30+ FPS with synchronized 4-camera multi-view display grid, proximity-based audio alerts, and voice-interactive guidance system.*

## Features

- **Multi-Camera Processing**: Synchronous capture from up to 4 directional camera feeds (Front, Left, Right, Back) with dynamic grid-based UI
- **Fast Object Detection**: YOLO v8 Nano model running on downscaled frames (~50ms per frame)
- **Monocular Depth Estimation**: Depth-Anything-V2-Small model via HuggingFace Transformers with step-based distance calculation (1 step ≈ 0.75m)
- **Object Tracking**: ByteTrack-based persistent tracking across frames with motion classification (approaching, moving away, lateral, stationary)
- **Time-To-Collision (TTC)**: Real-time collision risk computation for approaching objects
- **Priority-Based Grouping**: Intelligent object grouping with threat scoring and semantic description generation
- **Multi-Level Alerts**: Proximity-based beeping and high-priority spoken warnings
- **Voice-Interactive Agent**: LangChain + Groq LLM-powered agent for scene queries, memory retrieval, and target seeking
- **Session Logging**: JSONL-formatted structured logging of all detections and events
- **Docker Containerization**: Cross-platform deployment with all system dependencies pre-configured
- **Asynchronous Architecture**: Non-blocking depth pipeline ensures smooth real-time performance

## Tech Stack

### Core Libraries
- **Python 3.11+**: Primary programming language
- **OpenCV 4.9.0**: Video capture and frame processing
- **YOLOv8 (Ultralytics)**: Object detection and classification
- **Transformers 5.5.0**: Depth-Anything-V2-Small depth estimation
- **PyTorch 2.4.0+**: Deep learning inference backend

### AI & Reasoning
- **LangChain**: Conversational AI framework
- **Groq API**: LLM backbone (Llama 3.3 70B Versatile)
- **Python-dotenv**: Environment variable management

### Audio
- **PyTTSX3 2.90**: Text-to-speech synthesis
- **SpeechRecognition 3.10+**: Voice input capture
- **PyAudio 0.2.13**: Audio device interface

### DevOps
- **Docker**: Full containerization with system dependencies
- **Docker Compose**: Multi-service orchestration

### Data & Logging
- **NumPy 1.26.4**: Numerical operations and array processing
- **TIMM 1.0.26**: Vision transformer utilities

## Project Structure

```
Visiona/
├── main.py                           # Application entry point and main loop
├── pyproject.toml                    # Project metadata and dependencies
├── requirements.txt                  # Pip dependency listing
├── yolov8n.pt                        # YOLO v8 Nano model weights
│
├── agents/
│   ├── orchestrator.py               # LangChain agent reasoning engine
│   └── tools/
│       └── vision_tools.py           # Vision query tools and intent handlers
│
├── core/
│   ├── config.py                     # Centralized configuration and constants
│   ├── detection.py                  # Detection dataclass with spatial metadata
│   ├── logger.py                     # JSONL session logging system
│   ├── memory.py                     # Agent memory bank for historical detections
│   └── priority_queue.py             # Max-heap priority queue for threat scoring
│
├── perception/
│   ├── vision.py                     # YOLO detection pipeline + async depth
│   ├── mono_depth.py                 # Monocular depth estimation (Depth-Anything)
│   └── tracker.py                    # Object tracking with motion analysis
│
├── kinematics/
│   ├── speed.py                      # Speed estimation from tracking
│   ├── ttc.py                        # Time-to-collision calculation
│   ├── kalman.py                     # Kalman filter for smoothing trajectories
│   └── heatmap.py                    # Threat heatmap and object grouping
│
├── audio/
│   ├── speech.py                     # Priority-based TTS engine
│   ├── alert.py                      # Proximity-based beep alerts
│   └── voice_input.py                # Background microphone listener
│
├── services/
│   ├── google_maps.py                # Google Maps integration (future)
│   └── vector_db.py                  # Vector database for semantic search
│
├── docker/
│   ├── Dockerfile                    # Linux environment with all dependencies
│   └── docker-compose.yml            # Multi-service orchestration
│
├── sample-vid/                       # Test video samples
└── logs/                             # Session logs (JSONL format)
```

## Installation

### Prerequisites
- Python 3.11 or higher
- pip or UV package manager
- (Optional) NVIDIA GPU with CUDA for faster inference

### Local Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/Visiona.git
   cd Visiona
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables** (Optional for LangChain Agent)
   ```bash
   cp .env.example .env
   # Edit .env and add:
   # GROQ_API_KEY=your_groq_api_key
   # MAPS_API_KEY=your_google_maps_api_key (optional)
   ```

### Docker Installation

1. **Build Docker Image**
   ```bash
   docker-compose -f docker/docker-compose.yml build
   ```

2. **Run Container**
   ```bash
   docker-compose -f docker/docker-compose.yml up
   ```

## Usage

### Running the Application

**Local Execution:**
```bash
python main.py
```

**Docker Execution:**
```bash
docker-compose -f docker/docker-compose.yml up
```

### Configuration

Edit `core/config.py` to customize:
- **Video sources**: `SOURCES` dictionary (file paths or camera indices 0, 1, 2, 3)
- **Detection thresholds**: `CONF_THRESHOLD`, `FRAME_SKIP`
- **Distance filtering**: `MAX_DISTANCE_M`, `CONSIDER_MAX_M`, `HIGH_PRIORITY_M`
- **Object priorities**: `OBJECT_PRIORITY` dictionary
- **Speech settings**: `SPEECH_RATE`, `SPEECH_COOLDOWN`, `SEMANTIC_COOLDOWN`
- **Alert beep levels**: `BEEP_LEVELS` tuples (distance_m, freq_hz, duration_ms)

### Voice Interaction

Once running, the system listens for voice commands:
- *"Where did I put my coffee cup?"* → Queries memory for past detections
- *"What is in front of me?"* → Summarizes current scene
- *"Find me a chair"* → Sets search intent and navigates to target
- Regular queries are answered based on real-time YOLO detections

### Output Format

**Session Logs** (`logs/session_YYYYMMDD_HHMMSS.jsonl`):
Each line contains a JSON detection record:
```json
{
  "timestamp": "2026-04-04T19:54:47.123",
  "direction": "FRONT",
  "label": "person",
  "confidence": 0.92,
  "distance_m": 2.3,
  "speed_mps": 0.5,
  "motion": "approaching",
  "ttc_sec": 4.6,
  "track_id": 5,
  "threat_score": 45.3
}
```

## How It Works

### Processing Pipeline

1. **Frame Capture**: Four camera feeds are synchronized and captured at the start of each main loop cycle
2. **YOLO Detection**: Frames are downscaled to 1280px width and processed by YOLO v8 Nano (~50ms)
3. **Async Depth**: Background thread runs Depth-Anything-V2 inference without blocking display (~300ms, results cached)
4. **Distance Calculation**: Raw depth values are scaled to real-world meters using reference calibration
5. **Tracking**: ByteTrack maintains persistent object identities across frames
6. **Motion Analysis**: Speed and motion direction calculated from tracking trajectories
7. **TTC Computation**: For approaching objects, time-to-collision computed as distance / speed
8. **Priority Scoring**: Objects ranked by TTC, distance, and threat category
9. **Grouping**: Nearby objects merged into semantic groups (e.g., "group of people")
10. **Speech Generation**: Groups converted to natural language descriptions
11. **Audio Output**: TTS renders descriptions with priority-based queue; proximity alerts beep
12. **Agent Query** (Optional): Voice commands routed to LangChain agent for reasoning

### Key Architecture Decisions

- **Asynchronous Depth**: Depth inference runs in background thread to prevent frame rate degradation
- **Frame Downscaling**: YOLO processes downscaled frames for speed (1280px vs. 4K)
- **Step-Based Distance**: Distances converted to "steps" (0.75m each) for user comprehension
- **Priority Queue**: Threat-score-based heap ensures critical alerts delivered first
- **Synthetic Depth Fallback**: Heuristic depth calculation (focal length + object width) as fallback
- **Kalman Smoothing**: Per-object Kalman filters reduce noise in tracking and depth estimates

### Agentic Features (Phase 6)

The integrated LangChain agent provides advanced capabilities:
- **Memory Retrieval**: "Where did I leave my..."
- **Scene Summarization**: Natural language descriptions of complex environments
- **Spatial Reasoning**: "What is on the table?"
- **Contextual Curiosity**: Proactive guidance for navigation anchors
- **Search Intent**: User-defined target seeking with active guidance
- **Route Planning**: Integration with Google Maps for turn-by-turn navigation (future)

## Future Improvements

- **Stereo Depth**: Add binocular depth from dual cameras for improved accuracy
- **3D Scene Reconstruction**: Build persistent 3D maps of environments
- **Multi-Modal Learning**: Combine depth, motion, and semantic context for stronger predictions
- **Adaptive Beep Patterns**: Spatial audio via directional beeping or bone-conduction audio
- **Turn-by-Turn Navigation**: Full GPS + visual navigation integration
- **Fine-Tuned LLM**: Domain-specific model trained on spatial reasoning tasks
- **Real-time Performance Optimization**: GPU acceleration and model quantization
- **Outdoor Robustness**: Adapt detection and depth for sunlight variations
- **User Feedback Loop**: Active learning from user corrections

## Author

Author: Paavan Siri Vardhan Narava
Email: naravapaavansirivardhan@gmail.com
