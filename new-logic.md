# Reachy Mini — Modular Visual Tracker Design

A design and implementation guide for a fast, modular object-tracking system for the Reachy Mini robot. The tracker is triggered by Gemini Live (voice agent) via a tool call, uses Qwen-VL for one-shot detection, classical computer-vision modules for fast frame-to-frame tracking, and a head-mounted gyro (over ESP/USB-C) for ego-motion compensation.

---

## 1. System overview

### 1.1 Actors

- **Gemini Live** — handles speech in/out. Calls a `start_tracking(target_description)` tool when the user says "look at the red mug."
- **Big Planner LLM** — separate "brain" for higher-level reasoning and task planning. Out of scope for this document.
- **Qwen-VL** — vision-language model. Called rarely. Given a frame and a text description, returns a bounding box.
- **Tracker Core** — the subject of this document. Composed of small modules that vote on object position.
- **Reachy Mini** — physical robot with neck servos, a camera in the head, and a gyro sensor on the head.

### 1.2 High-level flow

```
User speaks --> Gemini Live --> tool call: start_tracking("red mug")
                                        |
                                        v
                         Qwen-VL on current frame --> bbox
                                        |
                                        v
                      Tracker Core init(frame, bbox)
                                        |
                                        v   (per frame, ~30 FPS)
                       Tracker Core update(frame, gyro_samples)
                                        |
                                        v
                       Object position --> Neck servo target angles
                                        |
                                        v
                       PID / smoothing --> Reachy head moves
```

### 1.3 Why this architecture

- **Speech and vision are decoupled.** Gemini does not see the image; it just calls tools. The tracker doesn't talk; it just tracks.
- **Slow models run rarely.** Qwen-VL is used at init and on re-detection only. SAM2 is intentionally removed in favor of classical methods.
- **Fast loop is pure CPU classical CV.** Optical flow, color histograms, template matching, Kalman prediction. 100+ FPS achievable on a Raspberry Pi class machine.
- **Modular.** Every "logic" is a self-contained module. You can add, disable, or replace modules without touching the rest of the system.

---

## 2. Hardware setup

### 2.1 Camera
- Standard Reachy Mini head camera.
- Frames pulled at 30 FPS (or whatever the camera supports).
- Camera intrinsics matrix `K` must be calibrated once and stored. Required for gyro compensation math.

### 2.2 Gyro
- IMU mounted physically on Reachy's head, near the camera.
- Connected to an **ESP32** (or similar microcontroller).
- ESP32 connects to the host computer via **USB-C** as a serial device (CDC/ACM, e.g., `/dev/ttyACM0` on Linux).
- ESP32 firmware reads gyro at 200+ Hz and streams samples over serial as compact frames:
  ```
  <timestamp_us>,<gx>,<gy>,<gz>\n
  ```
  where `gx, gy, gz` are angular velocities in rad/s.
- Host-side reader runs in a background thread, parses lines, and pushes samples into a ring buffer indexed by timestamp.

### 2.3 Calibration data needed

| Quantity | How to obtain | Stored as |
|---|---|---|
| Camera intrinsics `K` (3x3) | OpenCV `calibrateCamera` with checkerboard | `calib/camera.npz` |
| Lens distortion coeffs | Same calibration step | `calib/camera.npz` |
| Gyro bias `b` (3,) | Average reading over ~2 s of stillness at startup | recomputed each session |
| Gyro -> camera extrinsic `R_gc` | Hand-eye calibration, or assume identity if axes aligned | `calib/gyro_cam.npz` |
| Time offset `dt_gyro_cam` | Estimated by cross-correlating motion peaks | `calib/timing.json` |

---

## 3. Tracker architecture

### 3.1 Core idea

The tracker is a `TrackerCore` object that holds shared state (current bbox, point cloud, appearance model, confidence, etc.) and a list of registered `TrackingModule`s. Each frame, the core runs each module, collects their outputs, and fuses them.

### 3.2 Module interface

Every module conforms to one shape:

```python
class TrackingModule:
    name: str

    def init(self, frame, bbox, state) -> None:
        """Called once at start_tracking time."""

    def update(self, frame, state) -> ModuleOutput:
        """Called each frame. Returns its estimate."""

    def on_reinit(self, frame, bbox, state) -> None:
        """Called when re-detection happens mid-track."""
```

`ModuleOutput` carries:
- An estimated bbox (or `None` if the module abstains).
- A confidence in `[0, 1]`.
- Optional metadata (e.g., point survival ratio, histogram distance).

### 3.3 Fusion

The core fuses module outputs with a confidence-weighted average for position and a logical aggregator for health:

- **Position fusion:** weighted mean of bbox centers, weights = module confidences.
- **Scale fusion:** median of module-reported scales.
- **Health fusion:** combine module confidences into one `0-1` score and a discrete state in `{HEALTHY, DEGRADED, LOST}`.

### 3.4 Module list (build in this order)

| # | Module | Role | Speed |
|---|---|---|---|
| 1 | `OpticalFlowModule` | Lucas-Kanade on Shi-Tomasi corners | very fast |
| 2 | `GyroEgoMotionModule` | Subtract head-induced pixel shift | very fast |
| 3 | `ConfidenceModule` | Aggregate health, decide state | trivial |
| 4 | `KalmanPredictorModule` | Predict next position; survive brief occlusion | trivial |
| 5 | `ColorHistModule` | HSV histogram match against init region | fast |
| 6 | `TemplateMatchModule` | NCC against saved patch | fast (small window) |
| 7 | `MemoryBankModule` | Diverse appearance snapshots for re-ID | medium (only on LOST) |
| 8 | `QwenRedetectModule` | Re-detect when LOST | slow (rare) |
| 9 | `ScaleRotationModule` | Estimate scale/rotation from point cloud | fast |
| 10 | `OcclusionModule` | Distinguish occlusion from drift | fast |

### 3.5 The two key tricks

**Median, not mean.** When fusing point cloud motion, use the median displacement vector. A handful of points latching onto the background is normal; the median ignores them.

**Forward-backward error filtering.** For each tracked point, run optical flow forward (frame N -> N+1), then backward (N+1 -> N). The endpoint should match the start within ~1 pixel. Drop points where it doesn't. This single check kills most drift.

---

## 4. Module details

### 4.1 OpticalFlowModule
- At `init`, run `cv2.goodFeaturesToTrack` inside the bbox. Keep ~30-50 corners. Drop corners within 10% of the bbox edge.
- Each `update`, run `cv2.calcOpticalFlowPyrLK` from previous frame to current frame.
- Apply forward-backward filter, drop bad points.
- bbox center = median of surviving points. bbox size = previous size scaled by median pairwise distance ratio.
- Confidence = `n_surviving / n_initial`.
- Replenish points (via Shi-Tomasi inside current bbox) when survivors drop below ~70%.

### 4.2 GyroEgoMotionModule
- This module is a **preprocessor**. It does not produce a bbox; it modifies the optical-flow displacement before fusion.
- Between frame N (timestamp `t_N`) and N+1 (`t_{N+1}`):
  1. Pull all gyro samples in `[t_N, t_{N+1}]` from the ring buffer.
  2. Subtract bias, integrate `omega * dt` to get rotation `dR_gyro` (small angle, so `dR ~= I + skew(omega) * dt`).
  3. Transform into camera frame: `dR_cam = R_gc * dR_gyro * R_gc^T`.
  4. Project to pixel shift via `K`: a point at pixel `p` shifts approximately by `K * dR_cam * K^-1 * p - p`. For small rotations and points near image center, this is roughly a pure translation `(f * omega_y * dt, -f * omega_x * dt)` plus a small roll term.
- Subtract this shift from each tracked point's observed displacement. The result is the object's true motion in the world.

### 4.3 ConfidenceModule
- Reads outputs from every other module.
- Maintains a moving average of OpticalFlow survival ratio, ColorHist similarity, and TemplateMatch score.
- Outputs:
  - `HEALTHY` if all signals above thresholds.
  - `DEGRADED` if one or two signals below threshold; triggers point replenishment and increased template-match frequency.
  - `LOST` if multiple signals below threshold for N consecutive frames; triggers `QwenRedetectModule`.

### 4.4 KalmanPredictorModule
- State: `[x, y, vx, vy, scale, dscale]`.
- Predict step every frame.
- Update step using the fused observation from visual modules.
- When `state == LOST`, the predictor coasts on its last velocity for up to ~1 second, giving the head something to point at while re-detection runs.

### 4.5 ColorHistModule
- At init, compute HSV histogram of bbox interior (drop saturated highlights and shadows).
- Each frame, compute histogram of current bbox region. Compare via Bhattacharyya distance or correlation.
- Confidence = similarity score normalized to `[0, 1]`.
- Cheap, catches "drifted onto background" failures that pure flow misses.

### 4.6 TemplateMatchModule
- At init, save a 32x32 (or 64x64) grayscale patch from the bbox center.
- Every N frames (e.g., every 5), run `cv2.matchTemplate` with `TM_CCOEFF_NORMED` in a search window around the predicted bbox center.
- Confidence = peak NCC score.
- If the peak is far from the optical-flow estimate, flag a discrepancy.

### 4.7 MemoryBankModule
- A ring buffer of ~8 "appearance snapshots". Each snapshot stores: small template patch, color histogram, and (optionally) a feature descriptor.
- A snapshot is added when the tracker is HEALTHY **and** the current appearance is sufficiently different from existing snapshots (so the bank stays diverse).
- When the tracker is LOST and Qwen returns multiple candidate boxes, score each candidate against all snapshots and pick the best match.
- This is what lets the system re-acquire the **same** object after long occlusions, instead of locking onto something new of similar color.

### 4.8 QwenRedetectModule
- Triggered only when state == LOST.
- Sends the original target description plus the last-known bbox region (expanded by some margin) to Qwen-VL.
- Receives candidate bbox(es). Hands them to MemoryBank for scoring.
- The winning candidate triggers `tracker.reinit(frame, bbox)`.

### 4.9 ScaleRotationModule
- From the surviving optical-flow point cloud, compute:
  - Scale = median pairwise distance now / median pairwise distance at init.
  - Rotation = median angular change of point pairs.
- Lets the bbox grow, shrink, and rotate with the object.

### 4.10 OcclusionModule
- Watches **how** points die. If many points die in a coherent spatial region (e.g., from one side of the bbox sweeping inward) rather than scattering randomly, the object is being occluded, not lost.
- During occlusion: do not call Qwen yet. Coast on Kalman. Increase tolerance for low optical-flow confidence.

---

## 5. Implementation plan — separate test scripts per feature

Each module gets its own standalone Python script. Each script can be run on a recorded video clip (or live camera) and visualizes what that module is doing. Only when a script works on its own do we wire it into `TrackerCore`.

### 5.1 Suggested project layout

```
reachy_tracker/
  README.md
  calib/                       # camera and gyro calibration files
  recordings/                  # short test clips (with synced gyro logs)
  hardware/
    gyro_serial.py             # ESP32 USB-C reader, gyro ring buffer
    camera.py                  # camera grabber with timestamps
  modules/
    base.py                    # TrackingModule abstract class, ModuleOutput
    optical_flow.py
    gyro_ego_motion.py
    confidence.py
    kalman_predictor.py
    color_hist.py
    template_match.py
    memory_bank.py
    qwen_redetect.py
    scale_rotation.py
    occlusion.py
  core/
    tracker_core.py            # TrackerCore: registers modules, fuses, exposes init/update
    fusion.py                  # weighted bbox fusion, health aggregator
  tests/                       # one script per feature, runnable standalone
    test_01_camera.py
    test_02_gyro_serial.py
    test_03_qwen_bbox.py
    test_04_optical_flow.py
    test_05_gyro_compensation.py
    test_06_forward_backward_filter.py
    test_07_color_hist.py
    test_08_template_match.py
    test_09_kalman_predictor.py
    test_10_memory_bank.py
    test_11_qwen_redetect.py
    test_12_full_pipeline.py
  app/
    gemini_tools.py            # tool definitions exposed to Gemini Live
    head_controller.py         # bbox -> neck servo angles, smoothing/PID
    main.py                    # ties everything together
```

### 5.2 What each test script proves

| Script | Goal | Pass criterion |
|---|---|---|
| `test_01_camera.py` | Camera frames arrive with timestamps. | Live preview at 30 FPS, timestamps monotonic. |
| `test_02_gyro_serial.py` | ESP32 gyro stream is parsed correctly. | Log shows ~200 Hz samples; turning the head produces expected sign and magnitude on each axis. |
| `test_03_qwen_bbox.py` | Qwen-VL returns a usable bbox for a text prompt. | Draw the bbox on the frame; visually correct on a few test images. |
| `test_04_optical_flow.py` | LK optical flow on Shi-Tomasi corners follows a hand-clicked bbox. | Bbox stays on a moving object in a recorded clip with the head still. |
| `test_05_gyro_compensation.py` | Gyro-derived pixel shift matches observed image shift on a static scene. | Point head at a static target, rotate head; predicted vs observed pixel shift agree within a few pixels. |
| `test_06_forward_backward_filter.py` | FB filter eliminates bad points. | Compared to flow without FB, far fewer outliers; tracker survives longer in test clip. |
| `test_07_color_hist.py` | HSV histogram detects drift onto background. | When tracker is manually pushed off-target, histogram score drops below threshold within a second. |
| `test_08_template_match.py` | Template NCC catches small drift. | NCC peak position vs flow position diverges when drift is induced. |
| `test_09_kalman_predictor.py` | Predictor coasts through brief occlusions. | In a clip where target hides for ~0.5 s, predicted bbox stays reasonable; reacquired smoothly on reappearance. |
| `test_10_memory_bank.py` | Bank picks the right candidate among distractors. | Two similar objects in scene; after occlusion, correct one re-acquired. |
| `test_11_qwen_redetect.py` | Re-detection triggers on LOST and re-inits cleanly. | Forced loss in test clip; Qwen called once; tracker resumes. |
| `test_12_full_pipeline.py` | End-to-end: voice trigger -> Qwen init -> tracking -> head move. | On hardware: say "look at the red mug"; Reachy looks at it and follows. |

### 5.3 Testing without the robot
- Each test script can run on a **prerecorded video** plus a **prerecorded gyro log** with timestamps, to iterate on algorithms without needing the robot powered on.
- The recording tool (`tests/record_session.py`, optional) saves synchronized camera frames and gyro samples to disk. This is the single most useful piece of infra to build early.

---

## 6. Gemini tool surface

Minimal toolset Gemini Live needs:

```python
def start_tracking(target_description: str) -> dict:
    """
    Begin tracking the described object with the head.
    Returns: {"status": "tracking" | "not_found", "bbox": [...] | None}
    """

def stop_tracking() -> dict:
    """Stop tracking and recenter the head."""

def get_tracking_status() -> dict:
    """Returns: {"state": "HEALTHY"|"DEGRADED"|"LOST", "target": str}"""
```

Gemini does not see frames. It just speaks and calls these tools. The big planner LLM lives above Gemini and can also call these tools (or richer ones) when planning multi-step tasks.

---

## 7. Head control loop

Outside the scope of the tracker itself, but for completeness:
- Tracker outputs the object bbox center in image coordinates and a confidence.
- Convert bbox center to a desired neck angle delta: small angle approximation gives `delta_yaw ~ atan2(cx - cx0, fx)`, `delta_pitch ~ atan2(cy0 - cy, fy)`.
- Feed through a low-pass filter or PID controller to get smooth servo commands.
- Clamp velocity so the head doesn't snap.
- When `state == LOST`, hold the last position (don't chase the Kalman prediction past ~0.5 s).

---

## 8. Suggested build order (recap)

1. Camera grab + gyro serial reader (test 01, 02). You now have synchronized inputs.
2. Recording tool. You now have repeatable test data.
3. Qwen bbox call (test 03). You can detect once.
4. Optical flow + FB filter (tests 04, 06). You can track on a still head.
5. Gyro compensation (test 05). You can track while the head moves.
6. Confidence + Kalman (test 09). You handle brief failures.
7. Color hist + template (tests 07, 08). You catch drift.
8. Memory bank + Qwen re-detect (tests 10, 11). You recover from long failures.
9. Full pipeline (test 12) wired into Gemini tools and head controller.

At every step the system is runnable end-to-end with whatever modules exist so far. Adding a module is always a strict improvement and never blocks anything else.