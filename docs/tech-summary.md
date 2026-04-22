# Tech & AI Stack — Hsafa Robot (Reachy Mini)

A single-file summary of every technology, model, and AI subsystem used in
this repo, plus how Hsafa Core fits on top. Built from the actual code
under `hsafa_robot/` and `main.py`, and the design docs under `docs/`.

---

## 1. Hardware platform

- **Reachy Mini (wired / USB-C)** — a pure USB peripheral, not a networked
  device. Exposes three USB functions:
  - **Serial (CDC)** — motor bus for 9 motors (body yaw, 6-DOF Stewart
    platform for the head, 2 antennas).
  - **USB Audio (UAC)** — speaker + microphones.
  - **UVC Camera** — "Reachy Mini Camera" webcam.
- **Host** — macOS machine. A `reachy-mini-daemon` runs locally and
  exposes an HTTP/WebSocket API on `localhost:8000`; the Python SDK talks
  to that daemon.

---

## 2. Core runtime / language stack

- **Python 3.10+** (tested with 3.12) — single process, mostly threaded.
- **Reachy SDK** — `reachy-mini==1.6.3` (`ReachyMini`, `MediaManager`).
  Handles motor kinematics, min-jerk motion, audio pipelines (GStreamer
  under the hood).
- **NumPy ≥ 2.0**, **SciPy ≥ 1.13** — math, polyphase audio resampling
  (`scipy.signal.resample_poly`), rotations (`Rotation` from
  `scipy.spatial.transform`).
- **OpenCV (`opencv-python` ≥ 4.9)** — camera capture (AVFoundation on
  macOS), BGR frame handling, JPEG encoding, CLAHE auto-brightness,
  MOG2 background subtraction, drawing/overlay.
- **PyTorch** — backend for YOLO and FaceNet. Auto-selects CPU / MPS /
  CUDA via `pick_device()`.
- **python-dotenv** — loads `GEMINI_API_KEY` from `.env`.
- **Pillow** — image conversion for MTCNN.

---

## 3. Vision & AI models (what the robot actually "sees")

### 3.1 Person detection + pose — `tracker.py`

- **Model:** **YOLOv8n-Pose** (Ultralytics, COCO-17 keypoints).
  Weights auto-downloaded to `models/yolov8n-pose.pt`.
- **Library:** `ultralytics ≥ 8.3`.
- **Multi-object tracking:** **ByteTrack** (built into Ultralytics) —
  gives stable `track_id`s across frames so "Husam" stays the same
  person even through brief occlusions.
- **Inference tuning:** `imgsz=256`, `conf=0.35`, keypoint conf ≥ 0.5.

### 3.2 Fallback cascade (tier system)

Tracking degrades gracefully through four tiers as the primary signal
fades — this is what makes the head look stable in bad light:

| Tier | Signal | Source |
|------|--------|--------|
| 1 `face` | nose / eyes / ears keypoints | YOLOv8-Pose |
| 2 `body` | shoulders midpoint | YOLOv8-Pose |
| 3 `predicted` | Kalman extrapolation | custom Kalman filter |
| 4 `motion` | motion centroid | OpenCV **MOG2** background subtraction |

- **Kalman filter** — coasts the last known target for up to
  `PREDICT_COAST_S = 0.8 s` after YOLO misses.
- **MOG2** — last-resort fallback when even the Kalman guess expires.

### 3.3 Face detection + recognition — `face_recognizer.py`, `face_db.py`

- **Library:** `facenet-pytorch ≥ 2.5`.
- **Detection:** **MTCNN** — finds and aligns faces, crops to 160×160.
- **Embedding:** **InceptionResnetV1** pretrained on **VGGFace2** →
  512-D float32 embedding, L2-normalized.
- **Storage (`FaceDB`):** one `.npy` per person under `data/faces/
  <name>.npy`, shape `(N, 512)`. Cap of `MAX_EMBEDDINGS_PER_PERSON = 50`
  (oldest evicted).
- **Matching:** cosine similarity ≥ `0.6` → known; otherwise `unknown`.
- **Positioning:** each match is tagged `left` / `center` / `right`
  by bbox centroid (thirds of frame width).
- **Enrollment:** captures a burst of frames live when Gemini calls
  `enroll_face(name)`.

### 3.4 Active-speaker detection — `lip_motion.py`

- **Current (v1):** custom **mouth-region motion** tracker. Crops the
  lower-face region from each MTCNN bbox, computes frame-to-frame L1
  diff on a small grayscale patch, and takes a rolling max over
  ~1.5 s → per-face `is_speaking`.
- **Track continuity:** bbox IoU matching across frames gives each
  face a stable `track_id`; embedding lookup runs only every few
  seconds to resolve the name (cheap).
- **Planned (v2, see `docs/identity.md`):** swap in **TalkNet**
  (audio-visual cross-attention between the mouth crop and the mic)
  for a true `speaking_prob ∈ [0,1]` that fuses lips + audio.
- **Limitation today:** only works for visible speakers. Off-camera
  voices are the job of the future `voice_recognizer.py` (ECAPA-TDNN
  voice embedding planned).

### 3.5 Focus manager — `focus.py`

L2 glue that joins YOLO track IDs, face names, and lip motion into a
single "who should the robot look at *now*?" decision. Modes: `auto`
(largest/closest), `person` (named lock), `speaker` (active-speaker
follow). Exposed to Gemini as `focus_on_person`, `focus_on_speaker`,
`clear_focus`.

### 3.6 Motion / robot control — `robot_control.py`, `animation.py`

- **P-controller** on head yaw/pitch (gains `KP_YAW=0.6`, `KP_PITCH=0.4`).
- Body yaw extends the effective yaw range when the head saturates.
- Two overlay animations (`IdleAnimation`, `TalkingAnimation`) blend on
  top of the tracking pose; antennas follow the animation layer.
- Motion primitives planned per `docs/natural-gaze.md`: saccades +
  fixations, smooth pursuit, micro-saccades, search behavior, idle
  drift — all layered on this same controller.

---

## 4. Voice brain — Gemini Live — `gemini_live.py`

- **Service:** **Google Gemini Live API** (`google-genai ≥ 1.70`).
- **Default model:** `gemini-3.1-flash-live-preview` (override with
  `GEMINI_MODEL` env var or `--model`).
- **Voice:** `Puck` by default (`Charon`, `Kore`, `Fenrir`, `Aoede`
  available).
- **Streams:**
  - mic → Gemini: PCM16 mono at **16 kHz**.
  - camera JPEGs → Gemini: ~1 FPS (`--video-fps`).
  - Gemini audio → speaker: PCM16 **24 kHz** in, resampled (polyphase
    up=2/down=3) to float32 mono 16 kHz for Reachy's `MediaManager`.
- **Session resumption:** handles the ~10 min Gemini session rollover.
- **Function calling (tools):** the robot exposes these to Gemini
  (defined in `main.py::build_face_tools`):

| Tool | Purpose |
|------|---------|
| `enroll_face(name)` | Remember the face currently in view. |
| `identify_person()` | Name every visible face + position. |
| `find_person(name)` | Is this specific known person visible? |
| `list_known_people()` | Who do I already know? |
| `who_is_speaking()` | Which visible face is actively speaking. |
| `focus_on_person(name)` | Lock head/body onto a named person. |
| `focus_on_speaker()` | Follow whoever is speaking. |
| `clear_focus()` | Default: follow the closest person. |

- **System instruction** (`DEFAULT_SYSTEM_INSTRUCTION` in `main.py`) —
  persona = "Hsafa", small expressive desk robot, short warm replies,
  clear rules for when to invoke each tool.

---

## 5. Why the camera path is what it is

`main.py` opens the UVC camera **directly** via OpenCV/AVFoundation at
640×480 BGR (`CAP_AVFOUNDATION`) instead of going through the daemon's
WebRTC feed, because the WebRTC stream was too dark, laggy, and had
unsteady frame pacing that broke ByteTrack ID continuity. macOS allows
multiple processes to open the camera, so this coexists with the daemon.

Audio still goes through Reachy's `MediaManager` (GStreamer) — it
handles device selection, channel duplication, and device-rate
resampling.

---

## 6. The architecture (layered)

From `docs/architecture.md`, mapped to real files:

```
L4  Thinker        Hsafa Core (slow brain, remote server)
L3  Voice/persona  Gemini Live (gemini_live.py)
L2  Cognition      FocusManager (focus.py), future WorldState /
                   EventBus / IdentityGraph / GazePolicy
L1  Perception     CascadeTracker (tracker.py), FaceRecognizer
                   (face_recognizer.py), LipMotion (lip_motion.py),
                   VoiceRecognizer (stub)
L0  Hardware I/O   camera (OpenCV), mic/speaker (MediaManager),
                   motors (ReachyMini, robot_control.py, animation.py)
```

**Rule:** a layer only calls *down* or publishes events. L4 doesn't
poll motors; it asks L2. L1 doesn't know Gemini exists.

**Two canonical data shapes** (planned):
- `WorldState` — single snapshot written once per tick (humans,
  objects, active speaker, robot pose).
- `Event` — `{kind, ts, source, payload}` on a shared in-process
  `EventBus` for `person_detected`, `speech_heard`, `gaze_target_
  changed`, `say_this`, etc.

---

## 7. Identity graph (planned)

From `docs/identity.md` — the design for linking face + voice + name
into one `Identity`:

- `IdentityNode` (UUID + canonical name + aliases)
- `FaceSignal` and `VoiceSignal` linked to identity
- `cooccurs` edge when a recognized face is speaking + a clean
  voice sample lands in the same ≤2 s window → automatic
  cross-modal enrollment (the robot learns your voice by hearing
  you while it watches your mouth).
- **Voice embedder planned:** SpeechBrain **ECAPA-TDNN** on CPU,
  per finalized utterance.
- **Recognition fusion:** weighted sum over face cosine, voice
  cosine, recency prior, and a Hsafa social prior.
- **Correction without a tool:** saying "I'm not Kindom, I'm Husam"
  emits a `correction` event — the graph merges identities silently.

---

## 8. Natural gaze (planned, `docs/natural-gaze.md`)

Layer on top of the P-controller to stop feeling like a machine:

- **Saccades + fixations** (120–220 ms ballistic) instead of smooth
  chasing.
- **Micro-saccades** after ~700 ms of stillness.
- **Idle drift** — slow sinusoidal sweep when room is empty.
- **Search behavior** — "where did they go?" and "who just said
  that?" reduce to one `SearchIntent` event.
- **GazePolicy** — scoring function over candidates (speaking, known,
  just-appeared, curiosity, thinker prior), with hysteresis.
- **Thinker priors** — Hsafa publishes `gaze_prior` events with a
  TTL; the policy decides whether to comply.

---

## 9. Hsafa Core integration (the "slow brain")

From `docs/hsafa-integration.md`, `docs/unified-brain.md`, and
`hsafa-core.md`:

- **Hsafa Core** is a brain-as-a-service (Express HTTP + Prisma +
  pgvector, `streamText` from AI SDK v6). It hosts **Haseefs** —
  stateful identities with four pillars: **Profile**, **Config**,
  **Skills**, and **Memory**.
- **Memory (four types):** Episodic, Semantic, Social, Procedural.
- **Coordinator** enforces one run per Haseef with `AbortController`
  interrupts so new events preempt in-flight thinking.
- **SDK contract (`@hsafa/sdk`):** three methods — `registerTools`,
  `onToolCall`, `pushEvent` — plus `connect()` (SSE). Tool dispatch
  is unicast SSE with a POST-back for results.
- **Event routing:** by `haseefId` (direct) or `target` (profile
  match like `{robotId}`, `{phone}`, `{email}`).

### How this robot plugs in

One Haseef (`Atlas`) with three scopes all running inside this repo:

- **`body`** — `look_at`, `set_tracking_target`, `set_gaze_mode`,
  `wave`, `play_animation`, `get_position` + events `battery_low`,
  `bump_detected`.
- **`vision`** — `enroll_face`, `find_person`, `describe_scene`,
  `capture_image` + events `person_detected`, `person_left`,
  `gesture_detected`.
- **`voice`** — `speak`, `set_volume` + event `speech_heard`.

### Unified brain rule

Gemini Live = mouth + reflexes. Hsafa = memory + intention. One
shared `ConversationLog` (append-only JSONL). Hsafa authors the
voice's system prompt on each Gemini session rollover. Hsafa can
push `say_this` events into Gemini Live for thinker-initiated
speech (urgency-aware: `normal` queues, `high` interrupts, `idle`
drops if silent).

---

## 10. Dependency summary (`requirements.txt`)

```
reachy-mini==1.6.3         # robot SDK + daemon client
numpy>=2.0                 # math
scipy>=1.13                # rotations, audio resampling
opencv-python>=4.9         # camera, MOG2, drawing
ultralytics>=8.3           # YOLOv8-Pose + ByteTrack
google-genai>=1.70         # Gemini Live voice/vision
python-dotenv>=1.0         # .env loader
facenet-pytorch>=2.5       # MTCNN + InceptionResnetV1 (VGGFace2)
Pillow>=10.0               # image format glue
```

Implicit (via the above): **PyTorch**, **GStreamer** (via Reachy's
`MediaManager`).

---

## 11. What "Hsafa" means

**Hsafa** (حصيف, "the wise one") is the name of the whole system:
- Hsafa Core — the remote brain-as-a-service.
- A **Haseef** — one Hsafa-hosted mind, identified by a profile
  (e.g. `{robotId: "reachy-01"}` for this robot, whose Haseef is
  "Atlas").
- "Hsafa" is also the robot's persona name in Gemini's system
  prompt (`"You are Hsafa, a small expressive desk robot…"`), so
  the fast voice and the slow thinker present as one entity.

---

## 12. One-paragraph picture

A macOS Python process drives a USB-tethered Reachy Mini. OpenCV
pulls 640×480 frames from the built-in camera; a background
**YOLOv8n-Pose + ByteTrack** tracker locks onto a person, with
**Kalman + MOG2** fallback tiers for robustness. A second thread
runs **MTCNN + FaceNet (VGGFace2)** on demand to name faces, and a
third thread watches mouth-region motion to guess who's talking.
A focus manager turns "who to look at" intent into a `track_id`
that a smoothed P-controller drives into Reachy's 9 motors, with
idle / talking animations blended on top. **Gemini Live** handles
voice I/O (16 kHz in, 24 kHz out, ~1 FPS video), calls tools to
enroll / identify / focus, and speaks through Reachy's speaker.
The planned next step is **Hsafa Core**: a remote brain-as-a-service
whose Haseef "Atlas" owns persistent memory, biases gaze, authors
Gemini's system prompt, and speaks proactively via `say_this` — so
the fast voice and the slow thinker feel like one mind.
