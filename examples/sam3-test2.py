"""
Reachy Mini head follow — SAM 3.1 native + latency-aware world-frame control
================================================================================

WHY THIS VERSION (vs the previous one)
---------------------------------------

The previous version had a positive feedback loop:

    head moves up → SAM is 400ms behind → SAM still sees face "up" →
    controller moves head further up → SAM still says "up" → ...

The fix has three pieces:

1. SAM stamps each bbox with WHEN the camera grabbed the frame it
   processed (capture_ts).
2. The controller keeps a short history of head angles vs time.
3. When SAM returns, the controller uses the head angle that was
   commanded AT THE CAPTURE TIME, not the current angle. The world-
   frame target then stays stable across SAM's latency window.

Plus: critical damping (CONTROL_GAIN = 0.35). Don't correct 100% of
the error in one tick — correct only a fraction. Multiple SAM samples
will refine the target. This single change kills overshoot even if
everything else is wrong.

This is the standard solution for visual servoing with latent perception.

USAGE
-----

    python reachy_head_follow.py --concept person       # real robot
    python reachy_head_follow.py --no-reachy            # dry run
    python reachy_head_follow.py --sim                  # simulated daemon

KEY KNOBS (top of file)
-----------------------

    HORIZONTAL_FOV_DEG  must match your camera. Procedure in comment.
    CONTROL_GAIN        lower if head still oscillates. Default 0.35.
    DEADZONE_RAD        ignore errors below this. Default 2°.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk

BBox = tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Camera FOV — TUNE THIS EMPIRICALLY for your specific camera.
# ---------------------------------------------------------------------------
# Procedure to verify/fix:
#   1. Run with --no-reachy.
#   2. Lock on a stationary object, dead-center.
#   3. Manually rotate the camera by ~30°.
#   4. Commanded yaw on the stats line should change by ~30° in the
#      OPPOSITE direction. If too much: FOV too small, increase. If too
#      little: FOV too large, decrease. Tune in 5° increments.
HORIZONTAL_FOV_DEG = 66.0
HORIZONTAL_FOV_RAD = math.radians(HORIZONTAL_FOV_DEG)


# ---------------------------------------------------------------------------
# Control tuning — designed against latency, not for snappiness.
# ---------------------------------------------------------------------------
# Sign conventions (un-mirrored Reachy camera):
#   image +x = right, image +y = down
#   head +yaw = turn LEFT, head +pitch = look DOWN
YAW_SIGN = -1.0
PITCH_SIGN = +1.0

YAW_LIMIT_RAD = math.radians(60)
PITCH_LIMIT_RAD = math.radians(30)

# CONTROL_GAIN: fraction of the *measured world-frame error* we correct
# per SAM update. With SAM at 2.5 Hz and the head moving smoothly, ~0.35
# means it takes ~3 SAM samples to converge — plenty of time for the
# next SAM sample to confirm the correction worked. Lower = more
# resistant to overshoot, higher = snappier. IF YOU SEE OSCILLATION,
# DROP THIS FIRST. Try 0.2.
# Raised from 0.35 because SAM is running ~0.2 Hz, not 2.5 Hz — each
# sample has to do more work or the head never catches up.
CONTROL_GAIN = 0.6

# Don't chase errors smaller than this — prevents micro-jitter.
DEADZONE_RAD = math.radians(2.0)

# When no target, decay world target back to neutral. Slow on purpose.
RECENTER_DECAY_PER_TICK = 0.97

# Trust window for the last SAM bbox.
COAST_S = 1.0

# Control loop rate.
TICK_MS = 33

# Head-angle history retention (seconds).
HEAD_HISTORY_S = 8.0

# Command smoothing (30 Hz loop, sparse SAM updates).
# Without smoothing, _cmd snaps to _target the moment SAM produces a
# new sample, then holds still for 5 s — visibly blocky. Instead we
# ease toward the target every tick and cap the per-tick step so
# large jumps turn into a smooth ramp.
CMD_SMOOTH_ALPHA = 0.22      # ease-in to predicted target each tick
MAX_YAW_RATE_DPS = 160.0     # deg/s — generous cap; predictor does real work
MAX_PITCH_RATE_DPS = 100.0   # deg/s

# Predictive tracking: since SAM is ~0.2 Hz, we extrapolate target motion
# between detections using a simple linear-velocity model.  The head
# never sits still waiting for the next SAM frame.
PREDICTOR_MAX_AGE_S = 5.0    # trust prediction up to this long after last SAM
PREDICTOR_BLEND_S = 0.3      # seconds over which a new SAM sample eases in
PREDICTOR_MIN_DELTA_S = 0.05 # ignore samples closer than this (avoid div-by-zero)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()


# ---------------------------------------------------------------------------
# SAM 3.1 weights discovery
# ---------------------------------------------------------------------------
CANDIDATE_WEIGHTS = (
    "checkpoints/sam3.1_multiplex.pt",
    "checkpoints/sam3.pt",
)


def _pick_weights(weights_path: Optional[str]) -> str:
    if weights_path:
        return weights_path
    env = os.getenv("SAM3_WEIGHTS")
    if env:
        return env
    for p in CANDIDATE_WEIGHTS:
        if os.path.isfile(p):
            return p
    return CANDIDATE_WEIGHTS[0]


# ---------------------------------------------------------------------------
# Fake video dataset for Ultralytics' SAM 3 video predictor.
# ---------------------------------------------------------------------------
class _FakeVideoDataset:
    mode = "video"
    bs = 1

    def __init__(self, frame_cap: int = 1_000_000) -> None:
        self.frames = frame_cap
        self.frame = 0


# ---------------------------------------------------------------------------
# Follower state
# ---------------------------------------------------------------------------
class FollowState(Enum):
    IDLE = auto()
    LOCKING = auto()
    TRACKING = auto()
    LOST = auto()


# ---------------------------------------------------------------------------
# Sam3Follower — wraps SAM 3.1 native, includes capture timestamp on output.
# ---------------------------------------------------------------------------
class Sam3Follower:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        imgsz: Optional[int] = None,
        conf: float = 0.25,
        max_lost_frames: int = 8,
    ):
        weights = _pick_weights(weights_path)
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"SAM 3 weights not found at {weights!r}.\n"
                "Place under checkpoints/ or set SAM3_WEIGHTS=/path/to.pt"
            )
        imgsz = int(os.getenv("SAM3_IMGSZ", imgsz or 448))
        print(f"[sam3] device={DEVICE} weights={weights} imgsz={imgsz}")

        from ultralytics.models.sam import SAM3VideoSemanticPredictor

        overrides = dict(
            model=weights,
            task="segment",
            mode="predict",
            device=DEVICE,
            half=(DEVICE == "cuda"),  # MPS half is buggy for SAM3 → slower, not faster
            imgsz=imgsz,
            conf=conf,
            verbose=False,
            save=False,
        )
        self.predictor = SAM3VideoSemanticPredictor(overrides=overrides)
        self.predictor.setup_model(model=None, verbose=False)

        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self.predictor.setup_source(dummy)
        self.predictor.dataset = _FakeVideoDataset()
        print("[sam3] ready")

        self.max_lost_frames = int(max_lost_frames)
        self._inf_ctx: Optional[torch.inference_mode] = None

        # Shared state, guarded by _lock.
        self._lock = threading.Lock()
        self._state = FollowState.IDLE
        self._concept: Optional[str] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_ts: float = 0.0
        self._frame_seq: int = 0
        self._last_used_seq: int = -1
        self._current_bbox: Optional[BBox] = None
        self._current_obj_id: Optional[int] = None
        # Timestamp of the camera frame the current bbox was computed FROM.
        # The controller uses this to look up the head angle from that moment.
        self._current_capture_ts: float = 0.0
        self._last_update_ts: float = 0.0
        self._latencies: deque = deque(maxlen=30)
        self._empty_count: int = 0

        # Worker-thread state.
        self._pending_reset = False
        self._frame_index: int = 0

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ---- public API ---------------------------------------------------

    def start_following(self, concept: str) -> None:
        concept = (concept or "").strip()
        if not concept:
            return
        with self._lock:
            self._concept = concept
            self._state = FollowState.LOCKING
            self._current_bbox = None
            self._current_obj_id = None
            self._empty_count = 0
            self._pending_reset = True

    def stop_following(self) -> None:
        with self._lock:
            self._concept = None
            self._state = FollowState.IDLE
            self._current_bbox = None
            self._current_obj_id = None
            self._empty_count = 0
            self._pending_reset = True

    def push_frame(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr is None:
            return
        snap = frame_bgr.copy()
        ts = time.time()
        with self._lock:
            self._latest_frame = snap
            self._latest_frame_ts = ts
            self._frame_seq += 1

    def get_current_bbox(self) -> tuple[FollowState, Optional[BBox], float, float]:
        """Return (state, bbox, age_seconds, capture_ts).

        capture_ts is when the camera grabbed the frame this bbox came
        from — NOT now. The controller uses it to look up the head angle
        that was commanded at that moment, breaking the latency loop.
        """
        with self._lock:
            age = (
                time.time() - self._last_update_ts
                if self._last_update_ts
                else float("inf")
            )
            return (
                self._state,
                self._current_bbox,
                age,
                self._current_capture_ts,
            )

    def get_stats(self) -> dict:
        with self._lock:
            lat = list(self._latencies)
            return {
                "median_ms": (sorted(lat)[len(lat) // 2] if lat else 0.0),
                "obj_id": self._current_obj_id,
            }

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    # ---- worker thread ------------------------------------------------

    def _run(self) -> None:
        self._inf_ctx = torch.inference_mode()
        self._inf_ctx.__enter__()
        try:
            while not self._stop.is_set():
                self._tick()
        finally:
            try:
                self._inf_ctx.__exit__(None, None, None)
            except Exception:
                pass

    def _tick(self) -> None:
        with self._lock:
            frame = self._latest_frame
            frame_ts = self._latest_frame_ts
            concept = self._concept
            state = self._state
            seq = self._frame_seq
            do_reset = self._pending_reset
            self._pending_reset = False

        if do_reset:
            self._reset_predictor_state()

        if frame is None or state == FollowState.IDLE or concept is None:
            time.sleep(0.02)
            return
        if seq == self._last_used_seq:
            time.sleep(0.005)
            return
        self._last_used_seq = seq

        try:
            self._step_once(frame, frame_ts, concept)
        except Exception as e:
            print(f"[sam3] step error: {e}")
            with self._lock:
                self._state = FollowState.LOST
                self._pending_reset = True
            time.sleep(0.1)

    def _reset_predictor_state(self) -> None:
        self.predictor.inference_state = {}
        self.predictor.dataset = _FakeVideoDataset()
        self.predictor.run_callbacks("on_predict_start")
        self._frame_index = 0

    def _step_once(self, frame: np.ndarray, frame_ts: float, concept: str) -> None:
        t0 = time.time()
        self.predictor.dataset.frame = self._frame_index + 1
        self.predictor.batch = (["camera"], [frame], [""])
        im = self.predictor.preprocess([frame])
        if "text_ids" not in self.predictor.inference_state:
            preds = self.predictor.inference(im, text=[concept])
        else:
            preds = self.predictor.inference(im)
        results = self.predictor.postprocess(preds, im, [frame])
        latency_ms = (time.time() - t0) * 1000
        self._frame_index += 1

        dets = self._extract_dets(results[0])

        with self._lock:
            self._latencies.append(latency_ms)

            if not dets:
                self._empty_count += 1
                if (
                    self._state in (FollowState.LOCKING, FollowState.TRACKING)
                    and self._empty_count > self.max_lost_frames
                ):
                    self._state = FollowState.LOST
                    print(
                        f"[sam3] LOST '{concept}' "
                        f"(no detections for {self._empty_count} frames)"
                    )
                return

            self._empty_count = 0
            box, obj_id, _score = self._pick_best(
                dets, self._current_bbox, self._current_obj_id
            )

            if self._state == FollowState.LOCKING:
                print(f"[sam3] LOCKED on obj_id={obj_id} bbox={box}")
            elif self._state == FollowState.LOST:
                print(f"[sam3] RE-ACQUIRED obj_id={obj_id} bbox={box}")

            self._state = FollowState.TRACKING
            self._current_obj_id = obj_id
            self._current_bbox = box
            # Critical: stamp the bbox with WHEN the frame was grabbed.
            self._current_capture_ts = frame_ts
            self._last_update_ts = time.time()

    @staticmethod
    def _extract_dets(r) -> list[tuple[BBox, int, float]]:
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        data = boxes.data.detach().cpu().numpy()
        if data.shape[1] < 7:
            return []
        out = []
        for k in range(data.shape[0]):
            box = tuple(int(v) for v in data[k, :4])
            obj_id = int(data[k, 4])
            score = float(data[k, 5])
            out.append((box, obj_id, score))
        return out

    @staticmethod
    def _iou(a: BBox, b: BBox) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
            0, min(ay2, by2) - max(ay1, by1)
        )
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _pick_best(self, dets, last_box, locked_id):
        if locked_id is not None:
            for d in dets:
                if d[1] == locked_id:
                    return d
        if last_box is not None:
            best = max(dets, key=lambda d: self._iou(last_box, d[0]))
            if self._iou(last_box, best[0]) > 0:
                return best
        return max(dets, key=lambda d: d[2])


# ---------------------------------------------------------------------------
# HeadFollower — latency-aware world-frame controller.
#
# THE CRITICAL CHANGE FROM THE PREVIOUS VERSION:
#
# We keep a deque of (timestamp, yaw, pitch) showing where the head was
# commanded over the last 2 seconds. When SAM returns a bbox stamped
# with capture_ts, we look up the head angle that was COMMANDED AT
# capture_ts — not what the head is doing right now.
#
# This breaks the positive feedback loop. The world-frame target now
# reflects "where the face was, given where the head was looking when
# we took the picture", which stays stable as the head turns.
# ---------------------------------------------------------------------------
@dataclass
class ControlSnap:
    err_yaw_rad: float
    err_pitch_rad: float
    cmd_yaw: float
    cmd_pitch: float
    have_target: bool
    sam_lag_ms: float    # how stale this SAM sample was when we got it


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class HeadFollower:
    def __init__(self) -> None:
        self._target_yaw_world = 0.0
        self._target_pitch_world = 0.0
        self._cmd_yaw = 0.0
        self._cmd_pitch = 0.0
        # Ring buffer of (ts, cmd_yaw, cmd_pitch). Used to look up "what
        # were we commanding at time T?" so we pair past SAM bboxes with
        # the head angles that produced them.
        self._history: deque = deque(maxlen=int(HEAD_HISTORY_S / (TICK_MS / 1000)))
        # Last SAM sample we already integrated. Prevents re-correcting
        # the same error every tick at 30 Hz from a 2-3 Hz SAM.
        self._last_consumed_capture_ts: float = 0.0

        # Predictor: velocity estimate so we extrapolate between SAM frames.
        self._vel_yaw = 0.0
        self._vel_pitch = 0.0
        self._last_sam_yaw = 0.0
        self._last_sam_pitch = 0.0
        self._last_sam_ts = 0.0
        self._pred_valid = False
        self._last_tick_ts = 0.0

    def reset(self) -> None:
        self._target_yaw_world = 0.0
        self._target_pitch_world = 0.0
        self._cmd_yaw = 0.0
        self._cmd_pitch = 0.0
        self._history.clear()
        self._last_consumed_capture_ts = 0.0
        self._vel_yaw = 0.0
        self._vel_pitch = 0.0
        self._last_sam_yaw = 0.0
        self._last_sam_pitch = 0.0
        self._last_sam_ts = 0.0
        self._pred_valid = False
        self._last_tick_ts = 0.0

    def _record_history(self) -> None:
        self._history.append((time.time(), self._cmd_yaw, self._cmd_pitch))

    def _angle_at(self, ts: float) -> tuple[float, float]:
        """Return (yaw, pitch) commanded closest to timestamp ts."""
        if not self._history:
            return (self._cmd_yaw, self._cmd_pitch)
        best = self._history[0]
        best_dt = abs(best[0] - ts)
        for entry in self._history:
            dt = abs(entry[0] - ts)
            if dt < best_dt:
                best = entry
                best_dt = dt
        return (best[1], best[2])

    def step(
        self,
        bbox: Optional[BBox],
        frame_hw: tuple[int, int],
        have_target: bool,
        capture_ts: float,
    ) -> ControlSnap:
        """One control tick (called at TICK_MS rate from the camera loop).

        Parameters
        ----------
        bbox        : latest bbox in pixel coords (camera frame)
        frame_hw    : (H, W) of the camera frame
        have_target : True iff bbox is recent and follower is TRACKING
        capture_ts  : when the bbox's source frame was grabbed (epoch s)
        """
        H, W = frame_hw
        v_fov_rad = HORIZONTAL_FOV_RAD * (H / max(1, W))

        err_yaw_cam = 0.0
        err_pitch_cam = 0.0
        sam_lag_ms = 0.0

        now = time.time()
        dt_tick = now - self._last_tick_ts if self._last_tick_ts else (TICK_MS / 1000.0)
        self._last_tick_ts = now

        # ---------------------------------------------------------------
        # New SAM sample: fuse measurement and update velocity estimate.
        # ---------------------------------------------------------------
        new_sample = (
            have_target
            and bbox is not None
            and capture_ts > self._last_consumed_capture_ts + 1e-4
        )

        if new_sample:
            self._last_consumed_capture_ts = capture_ts
            sam_lag_ms = (now - capture_ts) * 1000.0

            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            err_x_px = cx - W / 2.0
            err_y_px = cy - H / 2.0

            err_yaw_cam = (err_x_px / W) * HORIZONTAL_FOV_RAD
            err_pitch_cam = (err_y_px / H) * v_fov_rad

            past_yaw, past_pitch = self._angle_at(capture_ts)
            meas_yaw = past_yaw + YAW_SIGN * err_yaw_cam
            meas_pitch = past_pitch + PITCH_SIGN * err_pitch_cam

            # Update predictor velocity from last SAM measurement to this one.
            if self._pred_valid:
                dt_sam = capture_ts - self._last_sam_ts
                if dt_sam > PREDICTOR_MIN_DELTA_S:
                    self._vel_yaw = (meas_yaw - self._last_sam_yaw) / dt_sam
                    self._vel_pitch = (meas_pitch - self._last_sam_pitch) / dt_sam

            self._last_sam_yaw = meas_yaw
            self._last_sam_pitch = meas_pitch
            self._last_sam_ts = capture_ts
            self._pred_valid = True

            # Fuse measurement into world target (partial correction).
            if (
                abs(err_yaw_cam) > DEADZONE_RAD
                or abs(err_pitch_cam) > DEADZONE_RAD
            ):
                self._target_yaw_world = (
                    (1 - CONTROL_GAIN) * self._target_yaw_world
                    + CONTROL_GAIN * meas_yaw
                )
                self._target_pitch_world = (
                    (1 - CONTROL_GAIN) * self._target_pitch_world
                    + CONTROL_GAIN * meas_pitch
                )

        # ---------------------------------------------------------------
        # Predict / coast between SAM frames so the head never freezes.
        # ---------------------------------------------------------------
        age_since_sam = now - self._last_sam_ts
        if self._pred_valid and age_since_sam < PREDICTOR_MAX_AGE_S:
            # Extrapolate target motion using estimated velocity.
            self._target_yaw_world += self._vel_yaw * dt_tick
            self._target_pitch_world += self._vel_pitch * dt_tick
        elif not have_target:
            self._target_yaw_world *= RECENTER_DECAY_PER_TICK
            self._target_pitch_world *= RECENTER_DECAY_PER_TICK
        # else: stale SAM but still coasting — hold predicted target.

        # Clamp to head workspace.
        self._target_yaw_world = _clamp(
            self._target_yaw_world, -YAW_LIMIT_RAD, YAW_LIMIT_RAD
        )
        self._target_pitch_world = _clamp(
            self._target_pitch_world, -PITCH_LIMIT_RAD, PITCH_LIMIT_RAD
        )

        # Smooth ease toward the (possibly extrapolated) target.
        max_dy = math.radians(MAX_YAW_RATE_DPS) * dt_tick
        max_dp = math.radians(MAX_PITCH_RATE_DPS) * dt_tick
        dy = (self._target_yaw_world - self._cmd_yaw) * CMD_SMOOTH_ALPHA
        dp = (self._target_pitch_world - self._cmd_pitch) * CMD_SMOOTH_ALPHA
        dy = _clamp(dy, -max_dy, max_dy)
        dp = _clamp(dp, -max_dp, max_dp)
        self._cmd_yaw += dy
        self._cmd_pitch += dp

        self._record_history()

        return ControlSnap(
            err_yaw_rad=err_yaw_cam,
            err_pitch_rad=err_pitch_cam,
            cmd_yaw=self._cmd_yaw,
            cmd_pitch=self._cmd_pitch,
            have_target=have_target,
            sam_lag_ms=sam_lag_ms,
        )


# ---------------------------------------------------------------------------
# Reachy gateway
# ---------------------------------------------------------------------------
class _NullReachy:
    def set_target(self, head=None, body_yaw=None, antennas=None):
        return

    def close(self):
        return


def _open_reachy(enable: bool, sim: bool):
    if not enable:
        return _NullReachy(), "disabled (--no-reachy)"
    try:
        from reachy_mini import ReachyMini  # type: ignore
    except ImportError as e:
        print(f"[reachy] reachy_mini not importable: {e}")
        return _NullReachy(), f"ImportError: {e}"
    try:
        if sim:
            rm = ReachyMini(spawn_daemon=True, use_sim=True, automatic_body_yaw=False)
            return rm, "simulated (in-process daemon)"
        rm = ReachyMini(automatic_body_yaw=False)
        return rm, f"connected to {getattr(rm, 'host', '?')}:{getattr(rm, 'port', '?')}"
    except Exception as e:
        print(f"[reachy] connect failed: {e}")
        return _NullReachy(), f"connect error: {e}"


def _head_pose_rpy(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    try:
        from reachy_mini.utils import create_head_pose  # type: ignore

        return create_head_pose(
            roll=roll_rad, pitch=pitch_rad, yaw=yaw_rad, degrees=False
        )
    except ImportError:
        from scipy.spatial.transform import Rotation as R  # type: ignore

        pose = np.eye(4)
        pose[:3, :3] = R.from_euler(
            "xyz", [roll_rad, pitch_rad, yaw_rad], degrees=False
        ).as_matrix()
        return pose


# ---------------------------------------------------------------------------
# Camera discovery
# ---------------------------------------------------------------------------
def list_cameras_macos() -> list[str]:
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPCameraDataType"], text=True, timeout=5
        )
    except Exception:
        return []
    names: list[str] = []
    for raw in out.splitlines():
        line = raw.strip()
        if not line.endswith(":"):
            continue
        name = line.rstrip(":").strip()
        if not name or name.lower() == "camera":
            continue
        names.append(name)
    return names


def discover_cameras(max_probe: int = 5) -> list[tuple[int, str]]:
    names = list_cameras_macos() if sys.platform == "darwin" else []
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    found: list[tuple[int, str]] = []
    for idx in range(max_probe):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            label = names[idx] if idx < len(names) else f"Camera {idx}"
            found.append((idx, f"{idx}: {label}"))
        cap.release()
    return found


def _pick_camera(cams, prefer_reachy: bool) -> int:
    if prefer_reachy:
        for idx, label in cams:
            if "reachy" in label.lower():
                return idx
    for idx, label in cams:
        if "reachy" not in label.lower():
            return idx
    return cams[0][0] if cams else 0


# ---------------------------------------------------------------------------
# Tk app
# ---------------------------------------------------------------------------
STATE_COLORS = {
    FollowState.IDLE: "gray",
    FollowState.LOCKING: "orange",
    FollowState.TRACKING: "lime green",
    FollowState.LOST: "red",
}


class App:
    CAM_W, CAM_H = 640, 480
    DISPLAY_MAX_W = 720
    CAP_BACKEND = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY

    def __init__(
        self,
        root: tk.Tk,
        follower: Sam3Follower,
        controller: HeadFollower,
        reachy,
        reachy_label: str,
        initial_concept: str,
        prefer_reachy_camera: bool,
    ):
        self.root = root
        self.follower = follower
        self.controller = controller
        self.reachy = reachy
        self.reachy_label = reachy_label
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._drive_enabled = tk.BooleanVar(value=True)
        self._last_state = FollowState.IDLE
        self._tick_counter = 0

        self.cameras = discover_cameras()
        print(f"[camera] detected: {self.cameras}")
        self.current_cam_index = _pick_camera(self.cameras, prefer_reachy_camera)

        self._build_ui(initial_concept)
        self.cap: Optional[cv2.VideoCapture] = None
        self._open_camera(self.current_cam_index)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(0, self._tick)

    def _open_camera(self, index: int) -> bool:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        cap = cv2.VideoCapture(index, self.CAP_BACKEND)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAM_H)
        # Keep only the newest frame — avoids cumulative lag building up
        # behind a slow GUI tick.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            self._set_status(f"ERROR: could not open camera {index}", "red")
            self.cap = None
            return False
        self.cap = cap
        self.current_cam_index = index
        return True

    def _build_ui(self, initial_concept: str) -> None:
        self.root.title("Reachy head follow — SAM 3.1 + latency-aware control")
        self.root.minsize(760, 680)

        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(top, text="Camera:").pack(side="left")
        cam_values = [label for _i, label in self.cameras] or ["(none detected)"]
        self.cam_combo = ttk.Combobox(top, values=cam_values, state="readonly", width=34)
        pre = next(
            (lbl for i, lbl in self.cameras if i == self.current_cam_index),
            cam_values[0],
        )
        self.cam_combo.set(pre)
        self.cam_combo.pack(side="left", padx=(6, 6))
        self.cam_combo.bind("<<ComboboxSelected>>", self._on_camera_change)

        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Label(controls, text="Concept:").pack(side="left")
        self.entry = ttk.Entry(controls, width=26)
        self.entry.pack(side="left", padx=(6, 6))
        self.entry.insert(0, initial_concept)
        self.entry.bind("<Return>", lambda _e: self._on_lock())

        ttk.Button(controls, text="Lock on", command=self._on_lock).pack(side="left")
        ttk.Button(controls, text="Release", command=self._on_release).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(controls, text="Home", command=self._on_home).pack(
            side="left", padx=(6, 0)
        )
        ttk.Checkbutton(
            controls, text="Drive Reachy", variable=self._drive_enabled
        ).pack(side="left", padx=(12, 0))

        self.status_label = ttk.Label(self.root, text="Idle.", foreground="black")
        self.status_label.pack(fill="x", padx=8, pady=(0, 4))

        self.stats_label = ttk.Label(
            self.root,
            text=f"Reachy: {self.reachy_label}",
            font=("TkFixedFont", 11),
        )
        self.stats_label.pack(fill="x", padx=8, pady=(0, 8))

        self.video_label = ttk.Label(self.root)
        self.video_label.pack(padx=8, pady=8)

    def _set_status(self, text: str, color: str = "black") -> None:
        self.status_label.config(text=text, foreground=color)

    def _on_camera_change(self, _event=None) -> None:
        label = self.cam_combo.get()
        for idx, lbl in self.cameras:
            if lbl == label:
                self._open_camera(idx)
                return

    def _on_lock(self) -> None:
        concept = self.entry.get().strip()
        if not concept:
            self._set_status("Type a concept first.", "orange")
            return
        self.controller.reset()
        self.follower.start_following(concept)
        self._set_status(f"Locking on '{concept}' …", "blue")

    def _on_release(self) -> None:
        self.follower.stop_following()
        self._set_status("Released — head will decay to neutral.", "black")

    def _on_home(self) -> None:
        self.controller.reset()
        try:
            self.reachy.set_target(head=_head_pose_rpy(0.0, 0.0, 0.0))
        except Exception as e:
            print(f"[reachy] home failed: {e}")

    def _tick(self) -> None:
        if self.cap is None:
            self.root.after(100, self._tick)
            return
        ok, frame = self.cap.read()
        if not ok:
            self.root.after(TICK_MS, self._tick)
            return

        H, W = frame.shape[:2]
        self.follower.push_frame(frame)
        f_state, bbox, age, capture_ts = self.follower.get_current_bbox()

        have_target = (
            bbox is not None
            and f_state == FollowState.TRACKING
            and age < COAST_S
        )
        snap = self.controller.step(bbox, (H, W), have_target, capture_ts)

        if self._drive_enabled.get():
            try:
                self.reachy.set_target(
                    head=_head_pose_rpy(0.0, snap.cmd_pitch, snap.cmd_yaw)
                )
            except Exception as e:
                print(f"[reachy] set_target failed: {e}")

        # Display — only every 2nd tick so ImageTk conversion doesn't
        # starve the control loop on macOS (Tk PhotoImage is slow).
        self._tick_counter += 1
        do_display = (self._tick_counter % 2) == 0
        if not do_display:
            if f_state != self._last_state:
                self._set_status(
                    f"State: {f_state.name}  age={age:.2f}s",
                    STATE_COLORS.get(f_state, "black"),
                )
                self._last_state = f_state
            self.root.after(TICK_MS, self._tick)
            return

        # Display
        if W > self.DISPLAY_MAX_W:
            scale = self.DISPLAY_MAX_W / float(W)
            disp_w = int(round(W * scale))
            disp_h = int(round(H * scale))
            display = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            disp_w, disp_h = W, H
            display = frame.copy() if bbox is not None else frame

        if bbox is not None and f_state in (FollowState.TRACKING, FollowState.LOST):
            if scale != 1.0 and display is frame:
                display = cv2.resize(
                    frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA
                )
            x1, y1, x2, y2 = bbox
            x1 = int(round(x1 * scale))
            y1 = int(round(y1 * scale))
            x2 = int(round(x2 * scale))
            y2 = int(round(y2 * scale))
            color = (0, 255, 0) if f_state == FollowState.TRACKING else (0, 0, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.drawMarker(display, (cx, cy), (0, 255, 255),
                           cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
            cv2.drawMarker(display, (disp_w // 2, disp_h // 2), (255, 255, 255),
                           cv2.MARKER_TILTED_CROSS, 12, 1, cv2.LINE_AA)
            cv2.line(display, (disp_w // 2, disp_h // 2), (cx, cy),
                     (0, 255, 255), 1, cv2.LINE_AA)

        if f_state != self._last_state:
            self._set_status(
                f"State: {f_state.name}  age={age:.2f}s",
                STATE_COLORS.get(f_state, "black"),
            )
            self._last_state = f_state

        st = self.follower.get_stats()
        med_ms = st.get("median_ms", 0.0)
        fps = (1000.0 / med_ms) if med_ms else 0.0
        self.stats_label.config(
            text=(
                f"Reachy: {self.reachy_label}  |  "
                f"SAM 3.1 {med_ms:4.0f} ms ({fps:4.1f} fps) "
                f"obj_id={st.get('obj_id')} lag={snap.sam_lag_ms:4.0f}ms  |  "
                f"err=({math.degrees(snap.err_yaw_rad):+5.1f}°, "
                f"{math.degrees(snap.err_pitch_rad):+5.1f}°)  "
                f"cmd yaw={math.degrees(snap.cmd_yaw):+5.1f}° "
                f"pitch={math.degrees(snap.cmd_pitch):+5.1f}°  "
                f"drive={'on' if self._drive_enabled.get() else 'off'}"
            )
        )

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.configure(image=self._photo)
        self.root.after(TICK_MS, self._tick)

    def _on_close(self) -> None:
        try:
            self.follower.stop_following()
            self.follower.close()
        except Exception:
            pass
        try:
            self.reachy.set_target(head=_head_pose_rpy(0.0, 0.0, 0.0))
            time.sleep(0.1)
        except Exception:
            pass
        try:
            if hasattr(self.reachy, "close"):
                self.reachy.close()
        except Exception:
            pass
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.root.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concept", default="person")
    ap.add_argument("--no-reachy", action="store_true")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--no-reachy-camera", action="store_true")
    args = ap.parse_args()

    print(f"[startup] device={DEVICE} fov={HORIZONTAL_FOV_DEG:.1f}° gain={CONTROL_GAIN}")

    follower = Sam3Follower()
    controller = HeadFollower()
    reachy, reachy_label = _open_reachy(enable=not args.no_reachy, sim=args.sim)
    print(f"[startup] reachy: {reachy_label}")

    root = tk.Tk()
    App(
        root,
        follower=follower,
        controller=controller,
        reachy=reachy,
        reachy_label=reachy_label,
        initial_concept=args.concept,
        prefer_reachy_camera=not args.no_reachy_camera,
    )
    root.mainloop()


if __name__ == "__main__":
    main()