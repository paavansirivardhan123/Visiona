# Visiona AI: Phased Task & Progress Tracker

This document tracks the progress of the modular enhancements to the Visiona project. For each phase, it explains what was done, why it provides value, and the specific files involved.

---

## ✅ Phase 1: Dockerization

### What was done?
We containerized the entire Visiona application. We created the necessary scripts to build an isolated Linux environment that pre-installs all system-level dependencies for OpenCV routines, Python audio libraries (PyAudio), and Text-to-Speech engines (Pyttsx3). We then moved these files into a dedicated `docker/` folder to keep your root directory clean.

### How is it useful to you?
Visiona relies on complex C-libraries for audio streaming and computer vision (like `libasound2`, `portaudio`, `ffmpeg`, and `libgl1`). Installing these natively on Windows is notoriously buggy and can break your local setups.
Docker acts as a virtual "sandbox". With one command, anyone can run your application on Windows, Mac, or Linux without having to manually install system packages, debug `pipwin` microphone errors, or mess with environment variables. It ensures perfect reproducibility. 

### Files Created:
1. `docker/Dockerfile`: The recipe that sets up the Python 3.11 Linux sandbox and uses `uv` to install everything cleanly.
2. `docker/docker-compose.yml`: A convenience script that maps your local code, Windows cameras, and display sockets into the Docker sandbox so you don't have to write long commands to start the app.

---

## ✅ Phase 2: Upgrading Depth Estimation for Mobile Cameras

### What was done?
We successfully replaced the old MiDaS depth model with `Depth-Anything-V2-Small-hf` using the HuggingFace transformers pipeline. This new zero-shot model is extremely robust to varying mobile camera focal lengths. We also fundamentally changed the core logic: the engine now computes distances in "steps" (1 step = ~0.75m).
*   Objects beyond **10 steps** are completely ignored.
*   Objects within **7 steps** are described.
*   Objects within **3 steps** trigger high-priority proximity alerts.
To ensure this runs perfectly without lagging your video, the new transformers depth pipeline operates on an asynchronous background thread. The speech engine also dynamically reads out distance in steps!

### Files modified:
- `core/config.py`: Defined the step-to-meter conversion and depth thresholds.
- `engines/grouping.py`: Refactored speech algorithms to translate raw depth into human-readable steps.
- `engines/mono_depth.py`: Replaced TorchHub integrations with real Metric Depth transformers `pipeline`.

---

---

## ✅ Phase 3: Unified Grid UI & Video Synchronization

### What was done?
Instead of launching 4 separate randomized windows, we successfully rewrote the internal game-loop of the system to become a synchronous "Frame Master". This explicitly requests exactly one frame simultaneously from every active camera feed per cycle, eliminating playback lag and ensuring perfect chronologic alignment.
We then built a dynamic UI engine that maps your outputs into a visually pleasing unified display:
*   4 cameras = 2x2 grid `[Front, Right] / [Left, Back]`
*   3 cameras = 2x2 grid with one black `"NO SIGNAL"` placeholder keeping the grid aligned.
*   2 cameras = 1x2 split screen.
*   1 camera = Fullscreen.

### How is it useful to you?
It completely solves background lag because the main thread strictly dictates when feeds update. The user (you) gets a much cleaner, singular terminal window that automatically sorts itself instead of manually sizing 4 different `cv2` popups!

### Files modified:
- `main.py`: Refactored `CameraFeed` to pull synchronously. Built `_build_grid` array generator, isolated YOLO processing into a dedicated asynchronous thread for optimal 30 FPS playback.
- `implementation_plan.md`: Updated roadmap to reflect this architectural shift.

---

## ✅ Phase 3.5: Spatial Fairness Scheduler & Threat Assessment

### What was done?
Completely bypassed rigid queueing algorithms and implemented an autonomous vehicle-styled **Dynamic Threat Heatmap**. The `detection.py` engine now mathematically calculates an Extreme Threat Score (0 to 100+) by factoring in Object Weight, Proximity, and Approaching Speed.
The `grouping.py` engine maps these scores to camera directions and dynamically punishes "spoken" directions with a mathematical score penalty. This beautifully simulates natural attention-shifting (organic Round-Robin) without blinding the user to actual life-threatening obstacles.

### How is it useful to you?
Ensures a person standing closely on your left doesn't blind the system to a car approaching from the right. Intelligently filters speech to behave like a human companion rather than a machine, allowing extreme dangers to interrupt but forcing casual objects into a polite, fair rotation.

### Files modified:
- `models/detection.py`: Programmed `threat_score` calculation property.
- `engines/grouping.py`: Refactored to sort by penalized threat scores instead of arbitrary lists.
- `core/config.py`: Added configuration sliders for the penalty decayer.

---

## ⏳ Phase 4: Spatial (3D) Audio Implementation

### What will be done?
Implementing Head-Related Transfer Functions (HRTF) or 3D audio panning for the alerts.

### How is it useful to you?
Instead of a slow text-to-speech engine saying "Warning, person on your left", the user will physically hear the warning beep originating from their left ear. This drastically decreases reaction time and reduces the cognitive load of understanding spoken sentences.

### Files modified:
*Pending implementation...*

---

## ⏳ Phase 5: Edge Computation Optimizations

### What will be done?
Exporting YOLO and the depth models to TensorRT/ONNX and adding dynamic frame-rate-processing algorithms.

### How is it useful to you?
Running 4 camera streams is highly battery-intensive. This ensures the computer/wearable doesn't overheat and scales down the processing load aggressively when no threats are nearby.

### Files modified:
*Pending implementation...*

---

## ⏳ Phase 6: Agentic LLM Integration (LangChain + Groq)

### What will be done?
Integrating `langchain-groq` to give the Visiona engine a powerful reasoning core. We will be building 7 distinct Agentic workflows:

*   **[x] Phase 6.1: Vision-Augmented Memory:** `deque` buffer logging YOLO objects so the user can ask "Where did I leave my cup?".
*   **[x] Phase 6.2: Dense Scene Explorer:** Spatial logic agent allowing the user to interrogate complex groupings ("What's on the table?").
*   **[x] Phase 6.3: Contextual Curiosity Agent:** Proactive navigation that pauses reading to offer specific guidance (e.g. stairs).
*   **[x] Phase 6.4: Spatial Scene Summarization (The Storyteller):** JSON-to-Text dynamic room descriptions.
*   **[x] Phase 6.5: Conversational Seeking (The Retrieval Agent):** Allowing vague semantic targets (e.g. "Find me a seat").
*   **[x] Phase 6.6: GPS Turn-by-Turn Wayfinder:** Linking Google Maps API for macro-navigation combined with YOLO micro-navigation.
*   **[x] Phase 6.7: Personalized Object Embedding:** "Remember This" custom fine-tuning to recognize personal belongings.

### How is it useful to you?
Transforms Visiona from a rigid object-detector into a proactive, intelligent companion that scales perfectly without hallucinating hazards.

### Files modified:
*   [x] Re-architecting file structure to FAANG-grade Domain-Driven paths.
*   [x] Refactoring all Python imports across `main.py` and subsystems to match new geometry.
