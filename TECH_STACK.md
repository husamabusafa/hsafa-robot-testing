# Tech Stack — main.py (Minimal)

This document lists the technologies used by the **lean** version of `main.py`.  Heavy optional modules (face recognition, gestures, head-pose, voice identity) have been moved to `archive/` and are no longer imported at startup.

---

## 1. Robot Hardware

| Tech | Role |
|------|------|
| **Reachy Mini** (Pollen Robotics) | Embodied robot platform — head (yaw/pitch/roll), body yaw, antennas, speaker, microphone, camera. |

---

## 2. Core Python / System

| Library | Usage |
|---------|-------|
| `asyncio` | Async event loop for Gemini Live session. |
| `threading` | Background tracker and VAD threads. |
| `logging` | Structured runtime logging. |
| `signal` | Graceful shutdown (`SIGINT`). |
| `json`, `re`, `base64`, `math`, `time` | Data parsing, regex, image encoding, geometry, timestamps. |

---

## 3. Computer Vision & Tracking

| Library / Model | Version | Usage |
|-----------------|---------|-------|
| **OpenCV** (`opencv-python`) | `>=4.9` | Raw camera capture (AVFoundation on macOS), frame preprocessing, drawing overlays. |
| **Ultralytics YOLO** (`ultralytics`) | `>=8.3` | `YOLOv8n-Pose` — person detection + 17 COCO keypoints. |
| **ByteTrack** | bundled | Multi-object tracking by detection; assigns stable IDs across frames. |
| **Kalman filter** | bundled | Predicts bbox motion between frames for smooth tracking. |
| **MOG2** (OpenCV) | bundled | Background-subtraction fallback when YOLO misses the target. |

---

## 4. Voice & Audio

| Library / Model | Version | Usage |
|-----------------|---------|-------|
| **Silero VAD** (`silero-vad`) | `>=5.0` | Determines whether microphone audio contains human speech. |
| **GStreamer** | system | Reachy `MediaManager` uses GStreamer for device selection, channel duplication, and 24 kHz → 16 kHz resampling. |

---

## 5. AI / LLM APIs

| Service / SDK | Version | Usage |
|---------------|---------|-------|
| **Google GenAI** (`google-genai`) | `>=1.70` | Gemini Live API — bidirectional voice + vision streaming. |

---

## 6. Motion Control

| Module | Role |
|--------|------|
| `hsafa_robot.robot_control` | P-controller that maps normalized image error → head angles. Body yaw engages when head nears limit. |
| `hsafa_robot.animation` | Idle + talking head-motion overlays blended via cross-fade. |
| `scipy.spatial.transform.Rotation` | `>=1.13` — quaternion / Euler conversions for head-pose math. |

---

## 7. Active Modules (custom)

| Module | Layer | Role |
|--------|-------|------|
| `tracker` | L1 | CascadeTracker thread (YOLO + ByteTrack + Kalman + MOG2). |
| `audio_vad` | L1 | Silero VAD speech-detection thread. |
| `events` | L2 | `EventBus` — typed pub/sub for cross-module communication. |
| `world_state` | L2 | `WorldStateHolder` — canonical snapshot of who/where/what in the scene. |
| `gemini_live` | L3 | `GeminiLiveSession` — async WebSocket to Gemini Live. |
| `robot_control` | L0 | Head/body motion controller. |
| `animation` | L0 | Idle / talking animation overlays. |

---

## 8. Archived Modules (in `archive/hsafa_robot/`)

These modules were removed from the active stack to reduce dependencies and startup weight. They can be restored later if needed.

| Module | Why Archived | Heavy Dependencies |
|--------|-----------|-------------------|
| `face_db`, `face_recognizer` | Face enrollment / identification | `facenet-pytorch`, `Pillow`, `torch` |
| `focus`, `gaze_policy`, `perception` | Gaze scoring & focus management | `mediapipe` (via head_pose, gestures) |
| `gestures`, `head_pose`, `object_detector` | MediaPipe hands / face-mesh | `mediapipe` |
| `lip_motion` | Mouth optical-flow speaker detection | `mediapipe` (via face mesh) |
| `identity_graph` | Cross-modal face+voice linking | `facenet-pytorch`, `speechbrain` |
| `voice_embedder`, `voice_identity` | Speaker recognition by voice-print | `speechbrain`, `torchaudio` |
| `natural_gaze` | Saccades / idle drift / search | `mediapipe` |
| `esp_gyro_bridge`, `gyro_stabilizer`, `head_gyro` | Gyro-based stabilization | hardware-specific |
| `voice_recognizer` | Legacy voice recognition stub | — |

---

## 9. Configuration & Secrets

| File | Purpose |
|------|---------|
| `.env` | Runtime secrets (`GEMINI_API_KEY`). |
| `models/yolov8n-pose.pt` | YOLOv8-Pose weights (auto-downloaded on first run). |

---

## 10. Summarized Requirements

```text
reachy-mini==1.6.3
numpy>=2.0
scipy>=1.13
opencv-python>=4.9
ultralytics>=8.3
google-genai>=1.70
python-dotenv>=1.0
silero-vad>=5.0
```

(Implicit: `torch` pulled in by `silero-vad` only — much smaller than the full FaceNet + SpeechBrain stack.)
