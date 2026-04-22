# Tech Recommendations — Add / Simplify / Watch Out

Distilled from the external tech review of this project. Organized as
the reviewer suggested: **add these**, **simplify these**, **watch out
for these**. Priorities are concrete and ordered by impact per hour of
work.

Cross-refs: `tech-summary.md` (what we already have), `gaze-policy.md`
(where most of these signals plug in), `identity.md` (cross-modal
enrollment), `natural-gaze.md` (behavior).

---

## 1. Add — in priority order

### 1.1 Silero VAD (do this first)

- **Why:** today we have `lip_motion.py` for visible speakers, but
  **no** voice activity detection on the audio stream. Mouth motion
  alone false-fires on chewing, yawning, silent laughing.
- **Fusion rule:** `speaking = lip_motion_high AND audio_has_speech`.
  One AND kills the bulk of false positives at near-zero cost.
- **Model:** [Silero VAD](https://github.com/snakers4/silero-vad)
  (MIT, ONNX). ~1 ms per 30 ms chunk on a single CPU thread.
- **Where it lands:** publishes `audio_speech_active(bool)` on the
  `EventBus`; `LipMotionTracker.snapshot()` gates `is_speaking` on it.
  Feeds directly into the `is_speaking` term of
  `gaze-policy.md` §2.
- **Pkg:** `silero-vad` (or load the ONNX model directly).

### 1.2 Upgrade v2 speaker detection → TS-TalkNet or FabuLight-ASD

Our `identity.md` planned plain TalkNet. Skip straight to one of:

- **TS-TalkNet** (Target-Speaker TalkNet, INTERSPEECH 2023). Adds a
  pre-enrolled **ECAPA-TDNN** speaker embedding as a third cue
  alongside lip crop + audio. Beats vanilla TalkNet on
  AVA-ActiveSpeaker and ASW.
  - **Double duty:** the same ECAPA-TDNN embedding feeds the
    `IdentityGraph` voice side (`identity.md §3`), so one model
    handles speaker-recognition *and* active-speaker detection.
- **FabuLight-ASD** (late 2024). Light-ASD + body-pose skeletons →
  94.3 % mAP on WASD. Since we already run YOLOv8-Pose it's getting
  skeletons "for free" — natural fit for a desk robot where most of
  the upper body is visible.

Pick one after prototyping Silero VAD fusion. For our setup
**FabuLight-ASD is probably the best match** because YOLOv8-Pose is
already paying the skeleton cost.

### 1.3 Sound-source localization (DOA) — biggest humanoid upgrade

- **Why:** today the robot can only attend to things it already sees.
  Add DOA and it can turn toward off-camera voices, claps, door
  slams. This is what "natural gaze" actually requires.
- **Mic check needed:** confirm how many channels Reachy Mini's UAC
  interface exposes. **2 mics** is enough to get azimuth via
  **GCC-PHAT** (cross-correlation with phase transform). 4+ mics →
  SRP-PHAT / MVDR, more robust.
- **Effort:** GCC-PHAT is ~50–100 lines of NumPy, **no neural net**.
- **Integration:** publishes `sound_detected { azimuth, ts }` →
  `GazePolicy` consumes it as a **virtual candidate** for ~1.5 s
  (see `gaze-policy.md §5`). The head turns, YOLO picks up the
  person, they become a real candidate on the next tick.

### 1.4 Emotion recognition (two tiers)

- **Cheap on-robot tier:** small CNN (ResNet18 on AffectNet or FER+)
  on the MTCNN / MediaPipe face crop every N frames → 7 basic
  emotions or valence/arousal. One extra forward pass, crop already
  computed.
- **Better fused tier (slow brain):** models like **Emotion-LLaMA**
  (NeurIPS 2024) or the lighter **MemoCMT** fuse face + prosody +
  text on 5-second windows. Too heavy for on-robot; run on
  Hsafa Core and publish `emotional_state(person, valence, arousal)`.
- **Fits the architecture perfectly:** fast local reflex stays
  local, slow affect inference lives in Core. No changes to the L0
  layer.

### 1.5 Head-pose + gaze-direction estimation

- **Why:** we track head *position* but not *orientation*. Knowing
  "Husam is looking **at** me" vs "**away**" is a huge humanoid
  signal — it tells the robot whether to initiate or wait. Also
  enables **mutual-gaze** behavior: hold eye contact briefly, then
  break, which is a hallmark of natural social behavior.
- **Tool:** **MediaPipe Face Landmarker** — 468 3D landmarks + a
  head-pose transformation matrix for yaw / pitch / roll, on-device,
  real-time on Mac CPU.
- **Score term:** `is_being_addressed = (face_toward_camera AND
  is_speaking)` — already the top-weighted term in `gaze-policy.md`.

### 1.6 Gesture recognition

- **Tool:** **MediaPipe Hands** (or Holistic) — 21 hand landmarks
  per hand, ~30 FPS on CPU.
- **Deliverables:** wave, point, thumbs-up, open-palm-stop →
  `gesture_detected` events (already declared in the Atlas vision
  scope in `hsafa-integration.md` but unimplemented).
- **Bonus — pointing:** combined with the YOLO skeleton, compute
  "where is Husam pointing?" → the head follows the point vector.
  Reads as remarkably intelligent for very little code.

---

## 2. Simplify

### 2.1 MTCNN → MediaPipe Face Detector (BlazeFace)

- MTCNN is old and slow. **BlazeFace** is faster, gives 6 landmarks
  out of the box, comparable-or-better accuracy at far lower latency.
- **Drop-in:** replaces detection only. Keep **FaceNet
  (InceptionResnetV1 / VGGFace2)** for the 512-D embedding; that
  part is still good.
- **Bonus:** if we adopt **MediaPipe Face Landmarker** for head pose
  (§1.5), detection is free — one model replaces MTCNN *and* solves
  head-pose in one pass.

### 2.2 Consider ArcFace / InsightFace for embeddings

- FaceNet-VGGFace2 works, but **ArcFace (InsightFace)** embeddings
  are meaningfully more discriminative at the same 512-D size,
  especially for small-N enrollment (our case: a handful of people,
  ~50 embeddings each).
- **Not urgent.** Upgrade only if the 0.6 cosine threshold starts
  producing false matches between look-alike people.

### 2.3 Don't wait for TalkNet v2 — fuse today

Ship Silero VAD + existing lip-motion AND fusion **today**. Keep
TS-TalkNet / FabuLight-ASD as the v2 rebuild, not a blocker.

---

## 3. Watch out for (architectural hazards)

### 3.1 Build the EventBus + WorldState **before** adding more senses

We're currently wiring modules directly (`tracker → focus →
robot_control`). Every new capability above (VAD, DOA, emotion,
gesture) naturally wants to *publish* an event, not plug into a
specific consumer.

- **If we build the bus first:** each new sense ≈ ~100 lines.
- **If we add senses first:** painful refactor later.

This is already item #1–#2 in `architecture.md §4`. Promote it.

### 3.2 Keep Gemini out of reflexive decisions

We push ~1 FPS JPEGs to Gemini → Gemini sees a world ~1 s stale. If
Gemini is ever authoritative for reflexive decisions (who to look at,
whether to respond to a wave), the lag will be felt.

**Rule:** `FocusManager` owns gaze — full stop. Gemini owns speech
content. `focus_on_speaker` must **not** round-trip through Gemini —
it becomes a local prior (§1.2 of `gaze-policy.md`).

### 3.3 Give the slow brain real conversational initiative

`say_this` (from `unified-brain.md`) is a great pattern. Add a sibling:

- **`ask_this(question, ttl)`** — Hsafa pushes a question that
  requires a user response before Hsafa's next turn. Gives Core real
  conversational initiative without fighting Gemini for the mic.

### 3.4 Fatigue + hysteresis are non-optional

A pure-argmax focus picker twitches. `gaze-policy.md §3` already
specifies the `SWITCH_MARGIN` + `MIN_DWELL_MS` rules; don't ship the
scorer without them.

---

## 4. Concrete dependency additions

If/when we implement the above, add to `requirements.txt`:

```
silero-vad           # §1.1 — MIT, ONNX runtime, tiny
mediapipe            # §1.5, §1.6, §2.1 — face landmarks, hands, BlazeFace
speechbrain          # §1.2 — ECAPA-TDNN voice embeddings (TS-TalkNet + identity)
# GCC-PHAT DOA (§1.3) needs no package — ~50 lines of NumPy.
```

Everything else (TS-TalkNet, FabuLight-ASD, Emotion-LLaMA, MemoCMT,
ArcFace/InsightFace) is opt-in and can be deferred until a specific
symptom drives it.

---

## 5. Suggested execution order (smallest wins first)

1. **EventBus + WorldState skeleton** (`architecture.md §4` step 1–2).
2. **GazePolicy v0** — `gaze-policy.md §9` steps 1–3 (`is_speaking`,
   `is_known`, `proximity`, `is_new`, `fatigue`, decision log).
3. **Silero VAD fusion** — improves `is_speaking` instantly.
4. **MediaPipe Face Landmarker** — replaces MTCNN, adds head-pose →
   unlocks `is_being_addressed` (biggest felt upgrade for groups).
5. **Person-mode auto-fallback** (`gaze-policy.md §1.2`).
6. **DOA (GCC-PHAT)** once mic channels are confirmed → virtual
   sound candidate.
7. **MediaPipe Hands** → `gesture_detected` events.
8. **ECAPA-TDNN voice embeddings** → cross-modal enrollment
   (`identity.md`), unblocks TS-TalkNet if we go that route.
9. **Emotion (on-robot tier)**, then slow-brain fused tier later.
10. **Hsafa priors on the bus** — `gaze_prior`, `say_this`,
    `ask_this`.

Each step ships independently; none blocks the next.

---

## 6. What we are *not* adopting (and why)

- **Full TalkNet/TS-TalkNet integration before VAD fusion** — the
  VAD-fused lip-motion signal is 80 % of the win for 5 % of the
  effort.
- **On-robot Emotion-LLaMA** — too heavy. Keep big multimodal
  affect models server-side (Hsafa Core) as events, not per-frame.
- **Switching embedding backbone (ArcFace) preemptively** — only if
  false-match symptoms appear.
- **Dedicated audio beamformer** — GCC-PHAT is fine until we have
  ≥4 mics and a concrete mis-localization problem.
- **Deep-learned gaze policy** — hand-tuned scoring (`gaze-policy.md`)
  gets us 90 % there with full interpretability. Defer until we
  actually have behavior data to learn from.
