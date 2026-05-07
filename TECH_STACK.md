# Tech Stack — main.py

This document lists every technology, library, model, and external service used by `main.py` (and the modules it imports from `hsafa_robot/`).

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
| `threading` | Background tracker, VAD, lip-motion, and gesture threads. |
| `pathlib` | Data directory paths (`data/faces`, `data/identity`). |
| `logging` | Structured runtime logging. |
| `signal` | Graceful shutdown (`SIGINT`). |
| `json`, `re`, `base64`, `math`, `time` | Data parsing, regex, image encoding, geometry, timestamps. |

---

## 3. Computer Vision & Tracking

| Library / Model | Version | Usage |
|-----------------|---------|-------|
| **OpenCV** (`opencv-python`) | `>=4.9` | Raw camera capture (AVFoundation on macOS), frame preprocessing, drawing overlays. |
| **Ultralytics YOLO** (`ultralytics`) | `>=8.3` | `YOLOv8n-Pose` — person detection + 17 COCO keypoints (nose, eyes, shoulders, etc.). |
| **ByteTrack** | bundled | Multi-object tracking by detection; assigns stable IDs across frames. |
| **Kalman filter** | bundled | Predicts bbox motion between frames for smooth tracking. |
| **MOG2** (OpenCV) | bundled | Background-subtraction fallback when YOLO misses the target. |
| **MediaPipe** (`mediapipe`) | `>=0.10` | Face-mesh landmarks for head-pose estimation (yaw/pitch/roll + `is_facing_camera`). Hand landmarks for gesture recognition (wave, point, thumbs-up, open-palm, fist). |

---

## 4. Face Recognition & Identity

| Library / Model | Version | Usage |
|-----------------|---------|-------|
| **facenet-pytorch** | `>=2.5` | `MTCNN` (face detection + alignment) → `InceptionResnetV1` (512-D embedding, VGGFace2 weights). |
| **PyTorch** (`torch`) | implicit | Runs FaceNet, Silero VAD, and SpeechBrain models on CPU. |
| **Pillow** (`PIL`) | `>=10.0` | Image format conversion for FaceNet pipeline. |
| **FaceDB** (`hsafa_robot.face_db`) | custom | SQLite-backed L2-normalized embedding store + cosine-nearest-neighbor identity search. |
| **IdentityGraph** (`hsafa_robot.identity_graph`) | custom | Links face names ↔ voice embeddings ↔ spatial history into a unified person record. |

---

## 5. Voice & Audio

| Library / Model | Version | Usage |
|-----------------|---------|-------|
| **Silero VAD** (`silero-vad`) | `>=5.0` | Determines whether microphone audio contains human speech (gates lip-motion false-positives). |
| **SpeechBrain** + **torchaudio** | `>=1.0` / `>=2.0` | `ECAPA-TDNN` speaker-embedding model for voice identity / voice-print enrollment. |
| **GStreamer** | system | Reachy `MediaManager` uses GStreamer for device selection, channel duplication, and 24 kHz → 16 kHz resampling. |

---

## 6. AI / LLM APIs

| Service / SDK | Model | Usage |
|---------------|-------|-------|
| **Google GenAI** (`google-genai`) | `>=1.70` | Gemini Live API — bidirectional voice + vision streaming. The robot hears, sees, and speaks through this session. |
| **OpenAI SDK** (`openai`) | `>=1.0` | Client for **OpenRouter** (`https://openrouter.ai/api/v1`). Calls `qwen/qwen3-vl-8b-instruct` for object-localization when the user says "look at the X". |

---

## 7. Motion Control

| Module | Role |
|--------|------|
| `hsafa_robot.robot_control` | P-controller that maps normalized image error → head angles (world-frame). Body yaw engages when head nears limit. |
| `hsafa_robot.animation` | Idle + talking head-motion overlays (breathing / nod) blended via cross-fade. |
| `scipy.spatial.transform.Rotation` | `>=1.13` — quaternion / Euler conversions for head-pose math. |

---

## 8. Perception & State Modules (custom)

| Module | Layer | Role |
|--------|-------|------|
| `tracker` | L1 | CascadeTracker thread (YOLO + ByteTrack + Kalman + MOG2). |
| `face_recognizer` | L1 | MTCNN + FaceNet enroll / identify pipeline. |
| `lip_motion` | L1 | Mouth-region optical-flow tracker (Lukas-Kanade) gated by VAD. |
| `audio_vad` | L1 | Silero VAD speech-detection thread. |
| `head_pose` | L1 | MediaPipe face-mesh → yaw/pitch/roll + `is_facing_camera`. |
| `gestures` | L1 | MediaPipe hand landmarks → gesture classification + pointing vector. |
| `object_detector` | L1 | YOLO object detection for held-item tagging. |
| `voice_embedder` | L1 | SpeechBrain ECAPA-TDNN voice-print extraction. |
| `perception` | L1/L2 | `HumanRegistry` — links face bboxes ↔ body bboxes ↔ keypoints. |
| `events` | L2 | `EventBus` — typed pub/sub for cross-module communication. |
| `world_state` | L2 | `WorldStateHolder` — canonical snapshot of who/where/what in the scene. |
| `gaze_policy` | L2 | Scoring engine that decides who the robot should look at (proximity, speaker, gesture, familiarity). |
| `focus` | L2 | `FocusManager` — drives `GazePolicy` scores into concrete head/body targets. |
| `identity_graph` | L2 | Persistent graph linking face ↔ voice ↔ name across sessions. |
| `voice_identity` | L2 | `VoiceIdentityWorker` — matches voice-prints to enrolled identities. |
| `gemini_live` | L3 | `GeminiLiveSession` — async WebSocket to Gemini Live; handles audio in, audio out, and vision frames. |

---

## 9. Configuration & Secrets

| File | Purpose |
|------|---------|
| `.env` | Runtime secrets (`GEMINI_API_KEY`, `OPENROUTER_API_KEY`). |
| `data/faces/` | SQLite + image cache for enrolled face embeddings. |
| `data/identity/` | JSON/graph store for `IdentityGraph`. |
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
openai>=1.0
python-dotenv>=1.0
facenet-pytorch>=2.5
Pillow>=10.0
mediapipe>=0.10
silero-vad>=5.0
speechbrain>=1.0
torchaudio>=2.0
```

(Implicit: `torch`, `torchvision`, `onnxruntime` where required by above packages.)
