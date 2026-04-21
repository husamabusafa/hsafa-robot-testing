"""tracker.py — YOLOv8-Pose + ByteTrack + Kalman + MOG2 cascade tracker.

Runs on a background thread. The main loop submits frames via `submit()` and
polls the latest result via `get()`. The tracker locks onto a single person
via ByteTrack's stable track IDs and falls back through tiers as the primary
signal fades:

    Tier 1 (face)      -> nose / eyes / ears keypoints
    Tier 2 (body)      -> shoulders midpoint, offset above
    Tier 3 (predicted) -> Kalman filter extrapolation
    Tier 4 (motion)    -> MOG2 background-subtraction centroid
"""
from __future__ import annotations

import math
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# --- Model weights ---------------------------------------------------------

POSE_MODEL_URL = (
    "https://github.com/ultralytics/assets/releases/download/"
    "v8.4.0/yolov8n-pose.pt"
)
POSE_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "yolov8n-pose.pt"
)

# --- Inference tuning ------------------------------------------------------

YOLO_IMGSZ = 256
YOLO_CONF = 0.35
KP_CONF = 0.5

# COCO-17 keypoint indices (YOLOv8-Pose).
KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE = 0, 1, 2
KP_LEFT_EAR, KP_RIGHT_EAR = 3, 4
KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER = 5, 6

# --- Cascade timings -------------------------------------------------------

PREDICT_COAST_S = 0.8     # how long to Kalman-coast after a miss
LOCK_TIMEOUT_S = 2.5      # release track-ID lock if unseen this long

# --- Tiers -----------------------------------------------------------------

TIER_FACE      = "face"
TIER_BODY      = "body"
TIER_PREDICTED = "predicted"
TIER_MOTION    = "motion"
TIER_NONE      = "none"

TIER_COLORS = {
    TIER_FACE:      (0, 255, 0),     # green
    TIER_BODY:      (0, 255, 255),   # yellow
    TIER_PREDICTED: (0, 165, 255),   # orange
    TIER_MOTION:    (0, 0, 255),     # red
}


# --- Helpers ---------------------------------------------------------------

def ensure_pose_model() -> str:
    """Download YOLOv8-Pose weights if missing (curl fallback for mac SSL)."""
    if POSE_MODEL_PATH.exists():
        return str(POSE_MODEL_PATH)
    POSE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model to {POSE_MODEL_PATH} ...")
    try:
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
        print("Done (urllib).")
        return str(POSE_MODEL_PATH)
    except Exception as e:
        print(f"urllib download failed ({e}); trying curl ...")
    try:
        subprocess.run(
            ["curl", "-fsSL", "-o", str(POSE_MODEL_PATH), POSE_MODEL_URL],
            check=True,
        )
        print("Done (curl).")
        return str(POSE_MODEL_PATH)
    except Exception as e:
        print(f"curl download failed: {e}", file=sys.stderr)
        print(f"Download it manually from:\n  {POSE_MODEL_URL}\n"
              f"and save as:\n  {POSE_MODEL_PATH}", file=sys.stderr)
        sys.exit(1)


def pick_device() -> str:
    """Prefer Apple Silicon GPU (MPS) if available, else CPU."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


# --- Result type -----------------------------------------------------------

@dataclass
class TrackResult:
    err_x: float                      # normalized horizontal error in [-1, 1]
    err_y: float                      # normalized vertical error in [-1, 1]
    bbox_px: tuple                    # (x1, y1, x2, y2, tx, ty)
    tier: str                         # TIER_FACE / BODY / PREDICTED / MOTION
    track_id: Optional[int]
    timestamp: float


# --- Keypoint -> face/head point ------------------------------------------

def _face_from_keypoints(
    kps: np.ndarray, kps_conf: np.ndarray,
) -> Optional[tuple]:
    """Best face/head center from one person's pose row.

    Returns ``(x, y, tier)`` or ``None``.
    """
    # Tier 1a: nose
    if kps_conf[KP_NOSE] >= KP_CONF:
        return (float(kps[KP_NOSE, 0]), float(kps[KP_NOSE, 1]), TIER_FACE)
    # Tier 1b: both eyes
    if kps_conf[KP_LEFT_EYE] >= KP_CONF and kps_conf[KP_RIGHT_EYE] >= KP_CONF:
        cx = (kps[KP_LEFT_EYE, 0] + kps[KP_RIGHT_EYE, 0]) / 2.0
        cy = (kps[KP_LEFT_EYE, 1] + kps[KP_RIGHT_EYE, 1]) / 2.0
        return (float(cx), float(cy), TIER_FACE)
    # Tier 1c: both ears
    if kps_conf[KP_LEFT_EAR] >= KP_CONF and kps_conf[KP_RIGHT_EAR] >= KP_CONF:
        cx = (kps[KP_LEFT_EAR, 0] + kps[KP_RIGHT_EAR, 0]) / 2.0
        cy = (kps[KP_LEFT_EAR, 1] + kps[KP_RIGHT_EAR, 1]) / 2.0
        return (float(cx), float(cy), TIER_FACE)
    # Tier 2: head estimated above shoulders
    if (kps_conf[KP_LEFT_SHOULDER] >= KP_CONF
            and kps_conf[KP_RIGHT_SHOULDER] >= KP_CONF):
        lsx, lsy = kps[KP_LEFT_SHOULDER]
        rsx, rsy = kps[KP_RIGHT_SHOULDER]
        mid_x = (lsx + rsx) / 2.0
        mid_y = (lsy + rsy) / 2.0
        width = math.hypot(lsx - rsx, lsy - rsy)
        head_y = mid_y - 0.75 * width
        return (float(mid_x), float(head_y), TIER_BODY)
    return None


def _make_kalman() -> cv2.KalmanFilter:
    """Constant-velocity Kalman filter on (x, y) pixel position."""
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], dtype=np.float32,
    )
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]], dtype=np.float32,
    )
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 5e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf


# --- Tracker ---------------------------------------------------------------

class CascadeTracker(threading.Thread):
    """Async 4-tier CV cascade for tracking one person on a webcam stream."""

    def __init__(self, model_path: str, device: str,
                 imgsz: int = YOLO_IMGSZ, conf: float = YOLO_CONF) -> None:
        super().__init__(daemon=True)
        self.detector = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf

        self.locked_id: Optional[int] = None
        self.last_lock_seen: float = 0.0

        self.kf = _make_kalman()
        self.kf_initialized = False
        self.kf_last_update: float = 0.0

        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=150, varThreshold=32, detectShadows=False,
        )

        self._new_frame_evt = threading.Event()
        self._lock = threading.Lock()
        self._pending_frame: Optional[np.ndarray] = None
        self._latest: Optional[TrackResult] = None
        self._stopped = False

        self.infer_count = 0
        self.infer_total_ms = 0.0
        self.tier_counts = {
            TIER_FACE: 0, TIER_BODY: 0,
            TIER_PREDICTED: 0, TIER_MOTION: 0, TIER_NONE: 0,
        }

    # ---- thread I/O ------------------------------------------------------
    def warmup(self, h: int, w: int) -> None:
        self.detector.predict(
            np.zeros((h, w, 3), dtype=np.uint8),
            imgsz=self.imgsz, conf=self.conf, device=self.device,
            verbose=False,
        )

    def submit(self, frame: np.ndarray) -> None:
        with self._lock:
            self._pending_frame = frame
        self._new_frame_evt.set()

    def get(self) -> Optional[TrackResult]:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stopped = True
        self._new_frame_evt.set()

    # ---- selection helpers ----------------------------------------------
    def _pick_locked_row(self, ids: np.ndarray, boxes: np.ndarray) -> Optional[int]:
        if self.locked_id is None:
            return None
        matches = np.flatnonzero(ids == self.locked_id)
        return int(matches[0]) if len(matches) else None

    def _acquire_lock(self, ids: np.ndarray, boxes: np.ndarray) -> int:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(np.argmax(areas))
        self.locked_id = int(ids[i]) if ids is not None and len(ids) > i else None
        return i

    # ---- Kalman ----------------------------------------------------------
    def _kalman_update(self, x: float, y: float, ts: float) -> None:
        if not self.kf_initialized:
            self.kf.statePost = np.array(
                [[x], [y], [0.0], [0.0]], dtype=np.float32,
            )
            self.kf_initialized = True
        else:
            self.kf.predict()
            self.kf.correct(np.array([[x], [y]], dtype=np.float32))
        self.kf_last_update = ts

    def _kalman_predict_only(self) -> Optional[tuple]:
        if not self.kf_initialized:
            return None
        s = self.kf.predict()
        return (float(s[0, 0]), float(s[1, 0]))

    # ---- Motion fallback -------------------------------------------------
    def _motion_fallback(self, frame: np.ndarray) -> Optional[tuple]:
        mask = self.bg_sub.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < (frame.shape[0] * frame.shape[1]) * 0.003:
            return None
        x, y, bw, bh = cv2.boundingRect(c)
        return (x + bw / 2.0, y + bh / 2.0, x, y, x + bw, y + bh)

    # ---- Run loop --------------------------------------------------------
    def run(self) -> None:
        while not self._stopped:
            self._new_frame_evt.wait()
            self._new_frame_evt.clear()
            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None
            if frame is None or self._stopped:
                continue
            h, w = frame.shape[:2]
            ts = time.time()

            t0 = time.perf_counter()
            results = self.detector.track(
                frame, imgsz=self.imgsz, conf=self.conf,
                device=self.device, persist=True,
                tracker="bytetrack.yaml", verbose=False,
            )
            self.infer_total_ms += (time.perf_counter() - t0) * 1000.0
            self.infer_count += 1

            tier = TIER_NONE
            tx = ty = 0.0
            x1 = y1 = x2 = y2 = 0
            used_id: Optional[int] = None

            r0 = results[0] if results else None
            boxes_obj = getattr(r0, "boxes", None) if r0 is not None else None

            if (boxes_obj is not None and boxes_obj.xyxy is not None
                    and len(boxes_obj.xyxy) > 0):
                boxes = boxes_obj.xyxy.cpu().numpy()
                ids = (boxes_obj.id.cpu().numpy().astype(int)
                       if boxes_obj.id is not None
                       else np.arange(len(boxes)))
                kps_xy = (r0.keypoints.xy.cpu().numpy()
                          if r0.keypoints is not None else None)
                kps_cf = (r0.keypoints.conf.cpu().numpy()
                          if (r0.keypoints is not None
                              and r0.keypoints.conf is not None)
                          else None)

                if (self.locked_id is not None
                        and (ts - self.last_lock_seen) > LOCK_TIMEOUT_S):
                    self.locked_id = None

                idx = self._pick_locked_row(ids, boxes)
                if idx is None:
                    idx = self._acquire_lock(ids, boxes)

                face_pt: Optional[tuple] = None
                if kps_xy is not None and kps_cf is not None and idx < len(kps_xy):
                    face_pt = _face_from_keypoints(kps_xy[idx], kps_cf[idx])

                if face_pt is not None:
                    tx, ty, tier = face_pt
                    self.last_lock_seen = ts
                    used_id = int(ids[idx]) if idx < len(ids) else None
                    x1, y1, x2, y2 = [int(v) for v in boxes[idx]]
                    self._kalman_update(tx, ty, ts)

            if tier == TIER_NONE:
                if (self.kf_initialized
                        and (ts - self.kf_last_update) < PREDICT_COAST_S):
                    pred = self._kalman_predict_only()
                    if pred is not None:
                        tx, ty = pred
                        tier = TIER_PREDICTED
                        side = 80
                        x1 = int(tx - side / 2)
                        y1 = int(ty - side / 2)
                        x2 = int(tx + side / 2)
                        y2 = int(ty + side / 2)
                else:
                    mot = self._motion_fallback(frame)
                    if mot is not None:
                        tx, ty, mx1, my1, mx2, my2 = mot
                        tier = TIER_MOTION
                        x1, y1, x2, y2 = int(mx1), int(my1), int(mx2), int(my2)

            self.tier_counts[tier] = self.tier_counts.get(tier, 0) + 1
            if tier != TIER_NONE:
                # Clamp target to frame bounds so err stays in [-1, 1].
                # TIER_BODY estimates head above shoulders (can be off-screen),
                # Kalman predict can extrapolate anywhere, and motion blobs
                # sometimes sit at the edge -- any of these would otherwise
                # produce err > 1 and drive the P-controller to its limits.
                tx_c = max(0.0, min(float(w - 1), tx))
                ty_c = max(0.0, min(float(h - 1), ty))
                err_x = (tx_c / w - 0.5) * 2.0
                err_y = (ty_c / h - 0.5) * 2.0
                res = TrackResult(
                    err_x=err_x, err_y=err_y,
                    bbox_px=(x1, y1, x2, y2, int(tx_c), int(ty_c)),
                    tier=tier, track_id=used_id, timestamp=ts,
                )
            else:
                res = None

            with self._lock:
                self._latest = res
