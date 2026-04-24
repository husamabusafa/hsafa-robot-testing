"""
Reachy Mini — track any object by text prompt AND move the head to follow it.

Pipeline:
  SAM 3.1 (open-vocabulary detect, every N seconds)
    → fast tracker (ViT/Nano/CSRT) every frame
    → TargetFilter (rejects junk, smooths jitter)
    → HeadController (P-controller with velocity limit, sends to Reachy @50Hz)
    → Reachy Mini head servos

What makes it smooth:
  • Control loop runs at fixed 50 Hz independent of vision FPS.
  • Pixel error → angular VELOCITY (not position) → natural deceleration.
  • Deadzone around center so head stops instead of hunting.
  • Per-axis velocity + acceleration caps = no jerky moves.
  • Soft joint limits (never slams into Reachy's mechanical stops).
  • On target-lost: head eases back to neutral (0°, 0°), doesn't freeze.
  • Camera motion awareness: because head moves slowly & we send VELOCITY
    commands, tracker sees a continuously-moving view and follows fluidly
    rather than getting a sudden jump.

Safety:
  • "Move Head" checkbox — can disable motion at any time.
  • Emergency stop button — freezes head where it is.
  • Disconnects cleanly on quit, returns head to neutral.

Requirements:
    pip install reachy_mini opencv-contrib-python torch torchvision ultralytics pillow
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time

import numpy as np
import cv2
import torch
from PIL import Image, ImageTk

from ultralytics.models.sam import SAM3VideoSemanticPredictor

# Reachy is optional — app runs in simulation mode if not connected
try:
    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose
    REACHY_AVAILABLE = True
except ImportError:
    REACHY_AVAILABLE = False
    print("[reachy] SDK not installed — simulation mode only")


# ======================================================================
# Config
# ======================================================================
DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available()
          else "cpu")

# --- SAM ---
SAM_WEIGHTS = os.getenv("SAM3_WEIGHTS", "checkpoints/sam3.pt")
SAM_IMGSZ   = int(os.getenv("SAM3_IMGSZ", 256))   # must be multiple of 16 (ViT patch size)
SAM_CONF    = float(os.getenv("SAM3_CONF", 0.30))

# --- Trackers (all optional) ---
VIT_ONNX  = os.getenv("VIT_TRACKER_ONNX", "checkpoints/vit_tracker.onnx")
NANO_BACK = os.getenv("NANO_BACK_ONNX",   "checkpoints/nanotrack_backbone.onnx")
NANO_HEAD = os.getenv("NANO_HEAD_ONNX",   "checkpoints/nanotrack_head.onnx")

# --- Camera ---
CAM_INDEX = int(os.getenv("CAM_INDEX", 0))
CAM_W     = int(os.getenv("CAM_W", 640))
CAM_H     = int(os.getenv("CAM_H", 480))

# --- Optics ---
CAM_FOV_H_DEG = float(os.getenv("FOV_H", 70.0))
CAM_FOV_V_DEG = float(os.getenv("FOV_V", 50.0))

# --- SAM refresh ---
SAM_REFRESH_SEC = float(os.getenv("SAM_REFRESH_SEC", 1.0))

# --- Head control ---
CONTROL_HZ       = 50.0           # control loop rate
DEAD_ZONE_DEG    = 2.0            # no movement within this angular error
MAX_YAW_DEG      = 45.0           # soft joint limit (check Reachy specs for yours)
MAX_PITCH_DEG    = 25.0
MAX_VEL_DEG_S    = 90.0           # max angular speed (degrees/sec)
MAX_ACCEL_DEG_S2 = 360.0          # max angular accel (degrees/sec^2) — smoothness
KP_YAW           = 2.5            # proportional gain (error° * KP = target velocity°/s)
KP_PITCH         = 2.5
NEUTRAL_YAW      = 0.0            # where head returns to when no target
NEUTRAL_PITCH    = 0.0
LOST_EASE_TIME   = 1.5            # seconds to ease back to neutral after losing target

print(f"[cfg] device={DEVICE}  reachy_sdk={REACHY_AVAILABLE}")


# ======================================================================
# SAM 3.1 — one-shot text detector
# ======================================================================
class _FakeDS:
    mode = "video"; bs = 1
    def __init__(self): self.frame = 0; self.frames = 10_000_000


class Sam3Detector:
    def __init__(self):
        if not os.path.isfile(SAM_WEIGHTS):
            raise FileNotFoundError(f"SAM weights not found: {SAM_WEIGHTS}")
        print(f"[sam3] loading {SAM_WEIGHTS} ...")
        overrides = dict(
            model=SAM_WEIGHTS, task="segment", mode="predict",
            device=DEVICE, half=(DEVICE in ("cuda", "mps")),
            imgsz=SAM_IMGSZ, conf=SAM_CONF,
            verbose=False, save=False, retina_masks=False,
        )
        self.p = SAM3VideoSemanticPredictor(overrides=overrides)
        self.p.setup_model(model=None, verbose=False)
        dummy = np.zeros((SAM_IMGSZ, SAM_IMGSZ, 3), np.uint8)
        self.p.setup_source(dummy)
        self.p.dataset = _FakeDS()
        self.p.run_callbacks("on_predict_start")
        try:
            self.p.batch = (["w"], [dummy], [""])
            im = self.p.preprocess([dummy])
            with torch.inference_mode():
                self.p.inference(im, text=["thing"])
        except Exception as e:
            print(f"[sam3] warmup err: {e}")
        self._reset()
        print("[sam3] ready")

    def _reset(self):
        self.p.inference_state = {}
        self.p.dataset = _FakeDS()
        self.p.run_callbacks("on_predict_start")

    def detect(self, frame_rgb, prompt):
        self._reset()
        self.p.dataset.frame = 1
        self.p.batch = (["cam"], [frame_rgb], [""])
        im = self.p.preprocess([frame_rgb])
        with torch.inference_mode():
            preds = self.p.inference(im, text=[prompt])
            results = self.p.postprocess(preds, im, [frame_rgb])
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None
        data = boxes.data.detach().cpu().numpy()
        best = int(np.argmax(data[:, 5]))
        x1, y1, x2, y2 = data[best, :4]
        return (float(x1), float(y1), float(x2), float(y2))


# ======================================================================
# Tracker factory
# ======================================================================
def _try_attr(obj, path):
    o = obj
    for p in path.split("."):
        o = getattr(o, p, None)
        if o is None: return None
    return o


def make_tracker():
    if os.path.isfile(VIT_ONNX):
        try:
            params = cv2.TrackerVit_Params(); params.net = VIT_ONNX
            return cv2.TrackerVit_create(params), "ViT"
        except Exception as e:
            print(f"[tracker] ViT init failed: {e}")
    if os.path.isfile(NANO_BACK) and os.path.isfile(NANO_HEAD):
        try:
            params = cv2.TrackerNano_Params()
            params.backbone = NANO_BACK; params.neckhead = NANO_HEAD
            return cv2.TrackerNano_create(params), "Nano"
        except Exception as e:
            print(f"[tracker] Nano init failed: {e}")
    for path in ("TrackerCSRT_create", "legacy.TrackerCSRT_create"):
        fn = _try_attr(cv2, path)
        if fn:
            try: return fn(), "CSRT"
            except Exception: pass
    for path in ("TrackerKCF_create", "legacy.TrackerKCF_create"):
        fn = _try_attr(cv2, path)
        if fn:
            try: return fn(), "KCF"
            except Exception: pass
    for path in ("legacy.TrackerMOSSE_create", "TrackerMOSSE_create"):
        fn = _try_attr(cv2, path)
        if fn:
            try: return fn(), "MOSSE"
            except Exception: pass
    raise RuntimeError("No OpenCV tracker. pip install opencv-contrib-python")


# ======================================================================
# Pixel → angle mapping
# ======================================================================
def pixel_to_angles(cx, cy, W, H):
    dx = (cx - W / 2) / (W / 2)
    dy = (cy - H / 2) / (H / 2)
    yaw   = -dx * (CAM_FOV_H_DEG / 2)
    pitch = dy * (CAM_FOV_V_DEG / 2)
    return yaw, pitch, dx, dy


# ======================================================================
# TargetFilter — clean tracker output before control
# ======================================================================
class TargetFilter:
    def __init__(self,
                 max_box_frac=0.85, min_box_frac=0.005,
                 max_jump_frac=0.25, max_scale_ratio=1.8,
                 smoothing=0.35, max_vel_frac=0.05, lost_after=5):
        self.max_box_frac = max_box_frac
        self.min_box_frac = min_box_frac
        self.max_jump_frac = max_jump_frac
        self.max_scale_ratio = max_scale_ratio
        self.alpha = smoothing
        self.max_vel_frac = max_vel_frac
        self.lost_after = lost_after
        self._smoothed = None
        self._raw_last = None
        self._reject_count = 0

    def reset(self):
        self._smoothed = None
        self._raw_last = None
        self._reject_count = 0

    def _reject(self, reason):
        self._reject_count += 1
        if self._reject_count >= self.lost_after:
            return (None, f"LOST:{reason}")
        if self._smoothed is None:
            return (None, f"reject:{reason}")
        scx, scy, sw, sh = self._smoothed
        return ((float(scx - sw / 2), float(scy - sh / 2),
                 float(sw), float(sh)), f"reject:{reason}")

    def update(self, raw_box, frame_w, frame_h):
        if raw_box is None:
            self._reject_count += 1
            if self._reject_count >= self.lost_after:
                return (None, "LOST")
            if self._smoothed is None:
                return (None, "no-detection")
            scx, scy, sw, sh = self._smoothed
            return ((float(scx - sw / 2), float(scy - sh / 2),
                     float(sw), float(sh)), "no-detection")

        x, y, w, h = raw_box
        cx, cy = x + w / 2, y + h / 2
        frame_diag = (frame_w ** 2 + frame_h ** 2) ** 0.5
        frame_area = frame_w * frame_h
        box_area = max(1.0, w * h)

        if not all(np.isfinite([x, y, w, h])): return self._reject("NaN")
        if w <= 2 or h <= 2: return self._reject("too-small")
        if box_area / frame_area > self.max_box_frac: return self._reject("box-explosion")
        if box_area / frame_area < self.min_box_frac: return self._reject("box-too-small")

        if self._raw_last is not None:
            px, py, pw, ph = self._raw_last
            pcx, pcy = px + pw / 2, py + ph / 2
            jump = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
            if jump > self.max_jump_frac * frame_diag: return self._reject("teleport")
            size_ratio = max(w / max(1, pw), pw / max(1, w),
                             h / max(1, ph), ph / max(1, h))
            if size_ratio > self.max_scale_ratio: return self._reject("scale-jump")

        self._reject_count = 0
        self._raw_last = (x, y, w, h)

        target = np.array([cx, cy, w, h], dtype=float)
        if self._smoothed is None:
            self._smoothed = target.copy()
        else:
            prev = self._smoothed[:2].copy()
            self._smoothed = self.alpha * target + (1 - self.alpha) * self._smoothed
            delta = self._smoothed[:2] - prev
            vmax = self.max_vel_frac * frame_diag
            mag = np.linalg.norm(delta)
            if mag > vmax and mag > 1e-6:
                self._smoothed[:2] = prev + delta * (vmax / mag)

        scx, scy, sw, sh = self._smoothed
        return ((float(scx - sw / 2), float(scy - sh / 2),
                 float(sw), float(sh)), "ok")


# ======================================================================
# HeadController — P-control with vel/accel limits + soft limits
# ======================================================================
class HeadController:
    """
    Smooth head controller. Runs at fixed CONTROL_HZ on its own thread.

    State:
      self.pos_yaw, self.pos_pitch    — current commanded pose (°)
      self.vel_yaw, self.vel_pitch    — current commanded velocity (°/s)
      self.target_yaw_err, target_pitch_err  — last measured pixel error → °

    Each tick:
      1. Read latest angular error from vision (set by set_error or clear_error)
      2. Compute desired velocity = KP * error (clamped to MAX_VEL)
      3. Accelerate current velocity toward desired (clamped to MAX_ACCEL)
      4. Integrate: pos += vel * dt (clamped to joint limits)
      5. Send mini.set_target(head=pose) — non-blocking, fresh pose each tick
    """

    def __init__(self, mini, on_status=None):
        self.mini = mini
        self.on_status = on_status or (lambda s: None)

        # Commanded state
        self.pos_yaw   = NEUTRAL_YAW
        self.pos_pitch = NEUTRAL_PITCH
        self.vel_yaw   = 0.0
        self.vel_pitch = 0.0

        # Target error from vision (degrees). None = no target.
        self._err_yaw   = None
        self._err_pitch = None
        self._err_lock  = threading.Lock()
        self._last_target_ts = 0.0

        # Control on/off
        self.active = False          # False = hold position, no servo updates
        self.enabled = True          # master switch (UI checkbox)
        self.estop = False           # emergency stop

        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def shutdown(self):
        self._stop = True
        try: self._thread.join(timeout=1.0)
        except Exception: pass
        # Return to neutral on exit
        if self.mini is not None:
            try:
                self.mini.goto_target(
                    head=create_head_pose(yaw=NEUTRAL_YAW, pitch=NEUTRAL_PITCH,
                                          roll=0.0, degrees=True),
                    duration=1.0, method="minjerk")
            except Exception as e:
                print(f"[head] neutral on exit failed: {e}")

    # ---- called from vision thread ----
    def set_error(self, yaw_err_deg, pitch_err_deg):
        """Vision tells us 'target is X° right, Y° up'. We handle the rest."""
        with self._err_lock:
            self._err_yaw = yaw_err_deg
            self._err_pitch = pitch_err_deg
            self._last_target_ts = time.time()

    def clear_error(self):
        with self._err_lock:
            self._err_yaw = None
            self._err_pitch = None

    def set_enabled(self, on: bool):
        self.enabled = on
        if not on:
            # Decelerate to stop (keeps position; next tick will zero velocity)
            self.vel_yaw = 0.0
            self.vel_pitch = 0.0

    def emergency_stop(self):
        self.estop = True
        self.vel_yaw = 0.0
        self.vel_pitch = 0.0

    def release_estop(self):
        self.estop = False

    # ---- internal ----
    def _compute_desired_vel(self, err_deg, kp):
        """Error → target velocity, with deadzone + saturation."""
        if err_deg is None:
            return 0.0
        if abs(err_deg) < DEAD_ZONE_DEG:
            return 0.0
        v = kp * err_deg
        return float(np.clip(v, -MAX_VEL_DEG_S, MAX_VEL_DEG_S))

    def _step_axis(self, pos, vel, desired_vel, dt, lim):
        """Accel-limited velocity tracking + position integration + soft limits."""
        # Limit acceleration
        dv_max = MAX_ACCEL_DEG_S2 * dt
        dv = desired_vel - vel
        dv = float(np.clip(dv, -dv_max, dv_max))
        vel = vel + dv

        # Integrate
        pos = pos + vel * dt

        # Soft joint limit: clamp position, zero velocity pushing into limit
        if pos > lim:
            pos = lim
            vel = min(0.0, vel)
        elif pos < -lim:
            pos = -lim
            vel = max(0.0, vel)
        return pos, vel

    def _loop(self):
        period = 1.0 / CONTROL_HZ
        next_t = time.time()
        last_tick = time.time()
        lost_since = None

        while not self._stop:
            now = time.time()
            dt = now - last_tick
            last_tick = now
            if dt > 0.2: dt = period  # large gap → treat as single tick

            # --- Determine desired velocity from error ---
            with self._err_lock:
                err_y = self._err_yaw
                err_p = self._err_pitch
                stale_age = now - self._last_target_ts

            if self.estop or not self.enabled or not self.active:
                desired_vy = 0.0
                desired_vp = 0.0
            elif err_y is None or stale_age > 0.5:
                # No target — ease back to neutral
                if lost_since is None:
                    lost_since = now
                ease_t = min(1.0, (now - lost_since) / LOST_EASE_TIME)
                # Proportional return to neutral, gentler than live tracking
                target_y = (NEUTRAL_YAW   - self.pos_yaw)   * ease_t
                target_p = (NEUTRAL_PITCH - self.pos_pitch) * ease_t
                desired_vy = self._compute_desired_vel(target_y, KP_YAW * 0.5)
                desired_vp = self._compute_desired_vel(target_p, KP_PITCH * 0.5)
            else:
                lost_since = None
                desired_vy = self._compute_desired_vel(err_y, KP_YAW)
                desired_vp = self._compute_desired_vel(err_p, KP_PITCH)

            # --- Step dynamics (accel-limited velocity + pos + soft limits) ---
            self.pos_yaw, self.vel_yaw = self._step_axis(
                self.pos_yaw, self.vel_yaw, desired_vy, dt, MAX_YAW_DEG)
            self.pos_pitch, self.vel_pitch = self._step_axis(
                self.pos_pitch, self.vel_pitch, desired_vp, dt, MAX_PITCH_DEG)

            # --- Send to Reachy ---
            if self.mini is not None and self.enabled and not self.estop:
                try:
                    pose = create_head_pose(
                        yaw=self.pos_yaw,
                        pitch=self.pos_pitch,
                        roll=0.0,
                        degrees=True,
                    )
                    # Use set_target (immediate, no interpolation) — we're already
                    # smooth because we compute pose at 50Hz with velocity/accel caps.
                    self.mini.set_target(head=pose)
                except Exception as e:
                    # Don't spam; show once
                    self.on_status(f"head send err: {e}")

            # --- Sleep to maintain rate ---
            next_t += period
            sleep_t = next_t - time.time()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                # We fell behind — resync (don't accumulate lag)
                next_t = time.time()


# ======================================================================
# App
# ======================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reachy Mini  •  follow any object by text")
        self.geometry("980x800")

        # --- UI state ---
        self.prompt        = tk.StringVar(value="person")
        self.status        = tk.StringVar(value="Ready.")
        self.disp_fps      = tk.StringVar(value="disp: —")
        self.trk_fps       = tk.StringVar(value="trk: —")
        self.sam_ms        = tk.StringVar(value="sam: —")
        self.trk_name      = tk.StringVar(value="tracker: —")
        self.filter_status = tk.StringVar(value="filter: —")
        self.angles        = tk.StringVar(value="—")
        self.head_state    = tk.StringVar(value="head: —")
        self.move_head     = tk.BooleanVar(value=False)   # start OFF for safety

        # --- internal ---
        self._stop = False
        self._want_detect = False
        self._tracking = False
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._bbox = None
        self._bbox_lock = threading.Lock()
        self._tracker = None
        self._sam_lock = threading.Lock()
        self._last_sam_ts = 0.0
        self._lost_count = 0
        self.filter = TargetFilter()

        # --- Reachy ---
        self.mini = None
        if REACHY_AVAILABLE:
            try:
                self.mini = ReachyMini()
                print("[reachy] connected")
            except Exception as e:
                print(f"[reachy] not connected: {e}")
                self.mini = None

        self.head = HeadController(self.mini, on_status=self._set_status_safe)
        self.head.start()

        self._build_ui()

        # --- camera ---
        self.cap = cv2.VideoCapture(CAM_INDEX)
        if not self.cap.isOpened():
            messagebox.showerror("Camera", f"Cannot open camera index {CAM_INDEX}")
            self.destroy(); return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        self.cap.set(cv2.CAP_PROP_FPS,          30)

        self.sam = None
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        threading.Thread(target=self._camera_loop,      daemon=True).start()
        threading.Thread(target=self._tracker_loop,     daemon=True).start()
        threading.Thread(target=self._sam_refresh_loop, daemon=True).start()
        self.after(33, self._draw_loop)
        self.after(100, self._head_status_loop)

    # ------------------------------------------------------------------
    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        top = ttk.Frame(self); top.pack(fill="x", **pad)
        ttk.Label(top, text="Track:").pack(side="left")
        e = ttk.Entry(top, textvariable=self.prompt, width=22)
        e.pack(side="left", padx=4)
        e.bind("<Return>", lambda _ev: self.on_detect())
        ttk.Button(top, text="Detect & Track", command=self.on_detect).pack(side="left", padx=4)
        ttk.Button(top, text="Stop",           command=self.on_stop  ).pack(side="left", padx=4)
        ttk.Checkbutton(top, text="Move Head", variable=self.move_head,
                        command=self._on_move_toggle).pack(side="left", padx=10)
        ttk.Button(top, text="E-STOP", command=self.on_estop).pack(side="left", padx=4)
        ttk.Button(top, text="Center head", command=self.on_center).pack(side="left", padx=4)

        right = ttk.Frame(top); right.pack(side="right")
        for var in (self.trk_name, self.sam_ms, self.trk_fps, self.disp_fps, self.filter_status):
            ttk.Label(right, textvariable=var, foreground="gray").pack(side="right", padx=6)

        self.canvas = tk.Label(self, bg="black")
        self.canvas.pack(fill="both", expand=True, **pad)

        mid = ttk.Frame(self); mid.pack(fill="x", **pad)
        ttk.Label(mid, textvariable=self.angles,
                  font=("Menlo", 14, "bold"), foreground="#00aa55").pack(side="left")
        ttk.Label(mid, textvariable=self.head_state,
                  font=("Menlo", 12), foreground="#0088cc").pack(side="right")

        bot = ttk.Frame(self); bot.pack(fill="x", **pad)
        ttk.Label(bot, textvariable=self.status, foreground="gray").pack(side="left")
        reachy_msg = "Reachy: connected" if self.mini else "Reachy: not connected (sim)"
        ttk.Label(bot, text=reachy_msg, foreground="#888").pack(side="right")

    # ------------------------------------------------------------------
    def _set_status_safe(self, msg):
        self.after(0, lambda m=msg: self.status.set(m))

    # ------------------------------------------------------------------
    # UI callbacks
    # ------------------------------------------------------------------
    def on_detect(self):
        p = self.prompt.get().strip()
        if not p:
            messagebox.showwarning("Missing", "Type what to track.")
            return
        self._want_detect = True
        self.status.set(f"Detecting '{p}' …")

    def on_stop(self):
        self._tracking = False
        self._want_detect = False
        self._tracker = None
        with self._bbox_lock:
            self._bbox = None
        self.filter.reset()
        self.head.clear_error()
        self.head.active = False
        self.trk_name.set("tracker: —")
        self.angles.set("—")
        self.status.set("Stopped.")

    def on_estop(self):
        if not self.head.estop:
            self.head.emergency_stop()
            self.status.set("⚠ E-STOP engaged. Click E-STOP again to release.")
        else:
            self.head.release_estop()
            self.status.set("E-STOP released.")

    def on_center(self):
        self.head.clear_error()
        self.head.active = False
        if self.mini is not None:
            try:
                self.mini.goto_target(
                    head=create_head_pose(yaw=NEUTRAL_YAW, pitch=NEUTRAL_PITCH,
                                          roll=0.0, degrees=True),
                    duration=0.6, method="minjerk")
                # Sync controller state to where the head actually is
                self.head.pos_yaw = NEUTRAL_YAW
                self.head.pos_pitch = NEUTRAL_PITCH
                self.head.vel_yaw = 0.0
                self.head.vel_pitch = 0.0
            except Exception as e:
                messagebox.showerror("Reachy", f"Center failed: {e}")

    def _on_move_toggle(self):
        self.head.set_enabled(self.move_head.get())
        self.status.set(f"Move head: {'ON' if self.move_head.get() else 'OFF'}")

    def on_close(self):
        self._stop = True
        try: self.cap.release()
        except Exception: pass
        try: self.head.shutdown()
        except Exception: pass
        try:
            if self.mini is not None:
                # ReachyMini is a context manager; this best-effort releases it
                if hasattr(self.mini, "__exit__"):
                    self.mini.__exit__(None, None, None)
        except Exception: pass
        self.destroy()

    # ------------------------------------------------------------------
    # Threads
    # ------------------------------------------------------------------
    def _camera_loop(self):
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005); continue
            with self._frame_lock:
                self._latest_frame = frame

    def _tracker_loop(self):
        fps_ema = 0.0
        last_t  = time.time()
        last_id = id(None)

        while not self._stop:
            if (not self._tracking) or self._tracker is None:
                time.sleep(0.01); continue

            with self._frame_lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()
                fid = id(self._latest_frame)
            if frame is None or fid == last_id:
                time.sleep(0.002); continue
            last_id = fid

            try:
                ok, box = self._tracker.update(frame)
            except Exception as e:
                print(f"[tracker] update error: {e}")
                ok = False; box = None

            now = time.time()
            dt = now - last_t; last_t = now
            if dt > 0:
                inst = 1.0 / dt
                fps_ema = 0.9 * fps_ema + 0.1 * inst if fps_ema else inst
                self.trk_fps.set(f"trk: {fps_ema:5.1f}")

            if ok and box is not None:
                x, y, w, h = box
                if w > 5 and h > 5 and w < frame.shape[1] and h < frame.shape[0]:
                    with self._bbox_lock:
                        self._bbox = (float(x), float(y), float(w), float(h))
                    self._lost_count = 0
                    continue

            self._lost_count += 1
            if self._lost_count >= 3:
                with self._bbox_lock:
                    self._bbox = None

    def _sam_refresh_loop(self):
        while not self._stop:
            do_full_detect = self._want_detect
            lost = (self._tracking and self._bbox is None)
            do_refresh = (self._tracking
                          and (time.time() - self._last_sam_ts) > SAM_REFRESH_SEC)

            if not (do_full_detect or do_refresh or lost):
                time.sleep(0.05); continue

            if self.sam is None:
                try:
                    self.sam = Sam3Detector()
                except Exception as e:
                    err = str(e)
                    self.after(0, lambda: messagebox.showerror("SAM", err))
                    self._want_detect = False
                    time.sleep(0.5); continue

            with self._frame_lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()
            if frame is None:
                time.sleep(0.05); continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prompt    = self.prompt.get().strip()

            t0 = time.time()
            with self._sam_lock:
                bbox = self.sam.detect(frame_rgb, prompt)
            sam_dt = (time.time() - t0) * 1000
            self.sam_ms.set(f"sam: {sam_dt:4.0f}ms")
            self._last_sam_ts = time.time()

            if bbox is None:
                if do_full_detect:
                    self._want_detect = False
                    self.after(0, lambda p=prompt: self.status.set(f"No '{p}' found"))
                continue

            x1, y1, x2, y2 = bbox
            w = max(10.0, x2 - x1); h = max(10.0, y2 - y1)
            new_box = (int(x1), int(y1), int(w), int(h))
            try:
                tracker, name = make_tracker()
                tracker.init(frame, new_box)
            except Exception as e:
                err = f"Tracker init failed: {e}"
                self.after(0, lambda: messagebox.showerror("Tracker", err))
                continue

            self._tracker = tracker
            self.filter.reset()
            self.trk_name.set(f"tracker: {name}")
            with self._bbox_lock:
                self._bbox = (float(new_box[0]), float(new_box[1]),
                              float(new_box[2]), float(new_box[3]))
            self._tracking    = True
            self._want_detect = False
            self._lost_count  = 0
            self.head.active = True
            self.after(0, lambda p=prompt, n=name: self.status.set(
                f"Tracking '{p}' with {n}"))

    # ------------------------------------------------------------------
    # Head status label (cheap, runs on UI thread)
    # ------------------------------------------------------------------
    def _head_status_loop(self):
        if self._stop: return
        estop = " ⚠E-STOP" if self.head.estop else ""
        en    = "ON" if self.head.enabled else "off"
        self.head_state.set(
            f"head: yaw {self.head.pos_yaw:+5.1f}° "
            f"pitch {self.head.pos_pitch:+5.1f}°  [{en}]{estop}"
        )
        self.after(100, self._head_status_loop)

    # ------------------------------------------------------------------
    # Display / vision → control bridge
    # ------------------------------------------------------------------
    def _draw_loop(self):
        if self._stop: return

        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()

        if frame is not None:
            H, W = frame.shape[:2]
            vis = frame

            cv2.drawMarker(vis, (W // 2, H // 2), (0, 255, 0),
                           cv2.MARKER_CROSS, 30, 2)
            # Deadzone ring (in pixels, approximate from angular)
            dz_px = int((DEAD_ZONE_DEG / (CAM_FOV_H_DEG / 2)) * (W / 2))
            cv2.circle(vis, (W // 2, H // 2), dz_px, (0, 200, 0), 1, cv2.LINE_AA)

            with self._bbox_lock:
                raw_bbox = self._bbox

            filtered_bbox, fstatus = self.filter.update(raw_bbox, W, H)
            self.filter_status.set(f"filter: {fstatus}")

            if fstatus.startswith("LOST"):
                with self._bbox_lock:
                    self._bbox = None
                self.head.clear_error()

            if raw_bbox is not None:
                rx, ry, rw, rh = [int(v) for v in raw_bbox]
                cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (120, 120, 120), 1)

            if filtered_bbox is not None and self._tracking:
                x, y, w, h = filtered_bbox
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                cx, cy = x + w / 2, y + h / 2

                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                cv2.arrowedLine(vis, (W // 2, H // 2), (int(cx), int(cy)),
                                (255, 200, 0), 2, tipLength=0.15)

                yaw_err, pitch_err, dx, dy = pixel_to_angles(cx, cy, W, H)

                # ⭐ HAND OFF TO HEAD CONTROLLER
                # Pass angular ERROR (not absolute target). The controller
                # handles deadzone, velocity, accel, and limits internally.
                self.head.set_error(yaw_err, pitch_err)

                in_dz = abs(yaw_err) < DEAD_ZONE_DEG and abs(pitch_err) < DEAD_ZONE_DEG
                self.angles.set(
                    f"err yaw {yaw_err:+5.1f}°  pitch {pitch_err:+5.1f}°   "
                    f"{'[DZ]' if in_dz else '→ moving head'}"
                )
            elif self._tracking:
                self.head.clear_error()
                self.angles.set("target lost — head easing to neutral")
            else:
                self.head.clear_error()
                self.angles.set("—")

            # FPS
            now = time.time()
            if not hasattr(self, "_last_d"):
                self._last_d = now; self._d_ema = 0.0
            dt = now - self._last_d; self._last_d = now
            if dt > 0:
                inst = 1.0 / dt
                self._d_ema = 0.9 * self._d_ema + 0.1 * inst if self._d_ema else inst
                self.disp_fps.set(f"disp: {self._d_ema:5.1f}")

            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            cw = self.canvas.winfo_width()  or W
            ch = self.canvas.winfo_height() or H
            img.thumbnail((cw, ch))
            tkimg = ImageTk.PhotoImage(img)
            self.canvas.configure(image=tkimg)
            self.canvas.image = tkimg

        self.after(33, self._draw_loop)


if __name__ == "__main__":
    App().mainloop()