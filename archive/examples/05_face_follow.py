"""
05_face_follow.py — Make Reachy Mini track a person using a 4-tier cascade.

The detector is a cascade of complementary CV techniques so we keep tracking
even when any single one fails:

  Tier 1  YOLOv8-Pose + ByteTrack  →  nose / eye / ear keypoints  (face visible)
  Tier 2  YOLOv8-Pose + ByteTrack  →  shoulders -> estimated head (face gone,
                                                                  body visible)
  Tier 3  Kalman filter predict    →  extrapolate last known position over
                                      short detector dropouts
  Tier 4  MOG2 motion detection    →  aim at the largest moving blob when
                                      nothing else fires

ByteTrack provides a stable track ID so we stick with one person when others
are in frame.

Speed tricks on Apple Silicon:
  * Inference on MPS (Metal GPU) if available; falls back to CPU.
  * YOLO input size 256 px.

Install extras:
    pip install opencv-python ultralytics

Quit with  q  (in the preview window)  or  Ctrl-C.
"""
import argparse
import math
import signal
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
from scipy.spatial.transform import Rotation as R
from reachy_mini import ReachyMini
from ultralytics import YOLO


# --- Tuning ---------------------------------------------------------------
# Proportional gains: how aggressively we correct per frame given a
# normalized error in [-1, 1]. Larger = snappier but may overshoot/jitter.
KP_YAW = 0.6
KP_PITCH = 0.4

# If the robot moves the WRONG way on one axis, flip the sign here.
# Defaults assume: +yaw turns robot left; +pitch looks down;
# raw (un-mirrored) image: x grows to the right, y grows downward.
YAW_SIGN = -1.0   # face on image-right (+x)  -> robot should yaw right (-)
PITCH_SIGN = +1.0  # face below center (+y)   -> robot should pitch down (+)

DEADZONE = 0.03                       # ignore tiny offsets
YAW_LIMIT = math.radians(60)
PITCH_LIMIT = math.radians(30)
STEP_SCALE = 0.20                     # per-iteration step multiplier

# EMA smoothing factors in [0, 1]. Higher = snappier, lower = smoother.
ERR_ALPHA = 0.6     # smooths the detected face position (kills jitter)
CMD_ALPHA = 0.4     # smooths the angle actually sent to the robot

# How long to wait with no face before slowly recentering.
RECENTER_AFTER_S = 1.5
# Keep commanding target toward last-seen position for this long even
# while no face is currently detected (bridges brief detection dropouts).
COAST_S = 0.6

# --- Body rotation ---
# Body lags the head in yaw: the head swings first, then the body rotates
# so the head can return toward center and we gain more total yaw range.
BODY_ENGAGE_RAD = math.radians(12)   # body stays still below this head yaw
BODY_FOLLOW_FRAC = 1               # body takes over this fraction of head-beyond-threshold
BODY_ALPHA = 0.08                    # very slow smoothing -> body moves calmly
BODY_LIMIT = math.radians(90)        # max body yaw (Reachy Mini has full rotation)
# --------------------------------------------------------------------------

# YOLOv8-Pose nano weights (from the official ultralytics/assets release).
POSE_MODEL_URL = (
    "https://github.com/ultralytics/assets/releases/download/"
    "v8.4.0/yolov8n-pose.pt"
)
POSE_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models" / "yolov8n-pose.pt"
)
# Inference image size — lower is faster.
YOLO_IMGSZ = 256
# YOLO confidence threshold for person detections.
YOLO_CONF = 0.35
# Per-keypoint confidence threshold.
KP_CONF = 0.5

# COCO keypoint indices (YOLOv8-Pose uses the standard COCO-17 ordering).
KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE = 0, 1, 2
KP_LEFT_EAR, KP_RIGHT_EAR = 3, 4
KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER = 5, 6

# How long (seconds) to keep coasting on Kalman prediction after the locked
# track disappears before falling back to motion detection.
PREDICT_COAST_S = 0.8
# How long before we release the track-ID lock and let a new person be picked.
LOCK_TIMEOUT_S = 2.5

# Tier names — used for logging / overlay color coding.
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


def head_pose(roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    return M


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def ensure_pose_model() -> str:
    """Download the YOLOv8-Pose weights if missing (curl fallback for mac SSL)."""
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


@dataclass
class TrackResult:
    err_x: float        # normalized horizontal error in [-1, 1]
    err_y: float        # normalized vertical error in [-1, 1]
    bbox_px: tuple      # (x1, y1, x2, y2, tx, ty) -- target point is (tx, ty)
    tier: str           # TIER_FACE / TIER_BODY / TIER_PREDICTED / TIER_MOTION
    track_id: Optional[int]
    timestamp: float


def _face_from_keypoints(
    kps: np.ndarray, kps_conf: np.ndarray,
) -> Optional[tuple]:
    """Best face/head center from a pose row.

    Returns (x, y, tier) or None.
    Tier is TIER_FACE if we have head-level keypoints, else TIER_BODY if we
    could only estimate from shoulders.
    """
    # Tier 1a: nose
    if kps_conf[KP_NOSE] >= KP_CONF:
        return (float(kps[KP_NOSE, 0]), float(kps[KP_NOSE, 1]), TIER_FACE)
    # Tier 1b: average of both eyes
    if kps_conf[KP_LEFT_EYE] >= KP_CONF and kps_conf[KP_RIGHT_EYE] >= KP_CONF:
        cx = (kps[KP_LEFT_EYE, 0] + kps[KP_RIGHT_EYE, 0]) / 2.0
        cy = (kps[KP_LEFT_EYE, 1] + kps[KP_RIGHT_EYE, 1]) / 2.0
        return (float(cx), float(cy), TIER_FACE)
    # Tier 1c: midpoint of both ears
    if kps_conf[KP_LEFT_EAR] >= KP_CONF and kps_conf[KP_RIGHT_EAR] >= KP_CONF:
        cx = (kps[KP_LEFT_EAR, 0] + kps[KP_RIGHT_EAR, 0]) / 2.0
        cy = (kps[KP_LEFT_EAR, 1] + kps[KP_RIGHT_EAR, 1]) / 2.0
        return (float(cx), float(cy), TIER_FACE)
    # Tier 2: estimate head above shoulders.
    if (kps_conf[KP_LEFT_SHOULDER] >= KP_CONF
            and kps_conf[KP_RIGHT_SHOULDER] >= KP_CONF):
        lsx, lsy = kps[KP_LEFT_SHOULDER]
        rsx, rsy = kps[KP_RIGHT_SHOULDER]
        mid_x = (lsx + rsx) / 2.0
        mid_y = (lsy + rsy) / 2.0
        width = math.hypot(lsx - rsx, lsy - rsy)
        # Typical human head is ~0.75 * shoulder-width above the shoulder line.
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


class CascadeTracker(threading.Thread):
    """Async 4-tier CV cascade for tracking one person on a webcam stream.

    1. YOLOv8-Pose + ByteTrack -> face/head keypoints (TIER_FACE or TIER_BODY)
    2. Kalman filter predict during short dropouts                (TIER_PREDICTED)
    3. MOG2 background-subtraction motion blob as last resort     (TIER_MOTION)
    """
    def __init__(self, model_path: str, device: str,
                 imgsz: int, conf: float) -> None:
        super().__init__(daemon=True)
        self.detector = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf

        # ByteTrack-state / lock
        self.locked_id: Optional[int] = None
        self.last_lock_seen: float = 0.0

        # Kalman
        self.kf = _make_kalman()
        self.kf_initialized = False
        self.kf_last_update: float = 0.0

        # Motion fallback
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=150, varThreshold=32, detectShadows=False,
        )

        # Threading plumbing
        self._new_frame_evt = threading.Event()
        self._lock = threading.Lock()
        self._pending_frame: Optional[np.ndarray] = None
        self._latest: Optional[TrackResult] = None
        self._stopped = False

        # Stats
        self.infer_count = 0
        self.infer_total_ms = 0.0
        self.tier_counts = {
            TIER_FACE: 0, TIER_BODY: 0,
            TIER_PREDICTED: 0, TIER_MOTION: 0, TIER_NONE: 0,
        }

    # ---- thread I/O ----
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

    # ---- tracking logic ----
    def _pick_locked_row(self, ids: np.ndarray, boxes: np.ndarray) -> Optional[int]:
        """Return the index of the locked track ID, or None if not present."""
        if self.locked_id is None:
            return None
        matches = np.flatnonzero(ids == self.locked_id)
        return int(matches[0]) if len(matches) else None

    def _acquire_lock(self, ids: np.ndarray, boxes: np.ndarray) -> int:
        """Pick the largest person and lock onto its ID. Returns the index."""
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(np.argmax(areas))
        self.locked_id = int(ids[i]) if ids is not None and len(ids) > i else None
        return i

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

    def _motion_fallback(self, frame: np.ndarray) -> Optional[tuple]:
        """Biggest moving contour centroid (fg mask via MOG2)."""
        mask = self.bg_sub.apply(frame)
        # Light morphology to consolidate blobs.
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
            return None  # too small to be meaningful
        x, y, bw, bh = cv2.boundingRect(c)
        return (x + bw / 2.0, y + bh / 2.0, x, y, x + bw, y + bh)

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
            # track() runs ByteTrack on top of detection and assigns stable IDs.
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

                # Re-acquire the lock if it's been gone too long.
                if (self.locked_id is not None
                        and (ts - self.last_lock_seen) > LOCK_TIMEOUT_S):
                    self.locked_id = None

                idx = self._pick_locked_row(ids, boxes)
                if idx is None:
                    idx = self._acquire_lock(ids, boxes)

                # Extract face/head position from the chosen person.
                face_pt: Optional[tuple] = None
                if kps_xy is not None and kps_cf is not None and idx < len(kps_xy):
                    face_pt = _face_from_keypoints(kps_xy[idx], kps_cf[idx])

                if face_pt is not None:
                    tx, ty, tier = face_pt
                    self.last_lock_seen = ts
                    used_id = int(ids[idx]) if idx < len(ids) else None
                    x1, y1, x2, y2 = [int(v) for v in boxes[idx]]
                    self._kalman_update(tx, ty, ts)

            # If we couldn't get a face/body point this frame, try to predict
            # with Kalman — but only if the last real update was recent.
            if tier == TIER_NONE:
                if (self.kf_initialized
                        and (ts - self.kf_last_update) < PREDICT_COAST_S):
                    pred = self._kalman_predict_only()
                    if pred is not None:
                        tx, ty = pred
                        tier = TIER_PREDICTED
                        # Bbox: a little square around the predicted point.
                        side = 80
                        x1 = int(tx - side / 2)
                        y1 = int(ty - side / 2)
                        x2 = int(tx + side / 2)
                        y2 = int(ty + side / 2)
                else:
                    # Tier 4: motion fallback.
                    mot = self._motion_fallback(frame)
                    if mot is not None:
                        tx, ty, mx1, my1, mx2, my2 = mot
                        tier = TIER_MOTION
                        x1, y1, x2, y2 = int(mx1), int(my1), int(mx2), int(my2)

            # Build result
            self.tier_counts[tier] = self.tier_counts.get(tier, 0) + 1
            if tier != TIER_NONE:
                err_x = (tx / w - 0.5) * 2.0
                err_y = (ty / h - 0.5) * 2.0
                res = TrackResult(
                    err_x=err_x, err_y=err_y,
                    bbox_px=(x1, y1, x2, y2, int(tx), int(ty)),
                    tier=tier, track_id=used_id, timestamp=ts,
                )
            else:
                res = None

            with self._lock:
                self._latest = res


def open_camera(index: int) -> "cv2.VideoCapture | None":
    """Open a camera on macOS with the AVFoundation backend."""
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Try to grab one frame to confirm it really works.
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def list_cameras(max_index: int = 6) -> None:
    """Probe camera indices 0..max_index-1 and print what works."""
    print("Probing cameras (AVFoundation)...")
    found = 0
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print(f"  [{i}] (not available)")
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"  [{i}] OK  {w}x{h}")
            found += 1
        else:
            print(f"  [{i}] opened but no frame")
        cap.release()
    if found == 0:
        print("\nNo cameras produced frames. Most likely cause on macOS:")
        print("  Terminal lacks camera permission.")
        print("  Open System Settings -> Privacy & Security -> Camera")
        print("  and enable it for your terminal app (Terminal / iTerm / VSCode).")
        print("  Then fully QUIT and relaunch the terminal and try again.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0,
                        help="OpenCV camera index (default: 0). Use "
                             "--list-cameras to see which indices work.")
    parser.add_argument("--list-cameras", action="store_true",
                        help="Probe camera indices 0..5 and exit")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable the debug preview window")
    parser.add_argument("--no-body", action="store_true",
                        help="Do NOT auto-rotate the body along with the head")
    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    # --- Camera ---
    cap = open_camera(args.camera)
    if cap is None:
        print(f"Could not open camera index {args.camera}.", file=sys.stderr)
        print("Tips:", file=sys.stderr)
        print("  * Run  `python examples/05_face_follow.py --list-cameras`  "
              "to see which indices work.", file=sys.stderr)
        print("  * If no cameras are listed, grant camera permission to your "
              "terminal:", file=sys.stderr)
        print("      System Settings -> Privacy & Security -> Camera -> "
              "enable for Terminal / iTerm / VSCode,", file=sys.stderr)
        print("      then fully quit and relaunch the terminal.", file=sys.stderr)
        sys.exit(1)

    # Read actual frame size (we set it to 640x480 but cameras may override).
    ret, probe = cap.read()
    if not ret:
        print("Camera opened but returned no frame.", file=sys.stderr)
        sys.exit(1)
    frame_h, frame_w = probe.shape[:2]

    # --- Detector (4-tier cascade on a background thread) ---
    model_path = ensure_pose_model()
    device = pick_device()
    print(f"Loading YOLOv8-Pose on {device.upper()} (imgsz={YOLO_IMGSZ}) ...")
    tracker = CascadeTracker(model_path, device, YOLO_IMGSZ, YOLO_CONF)
    tracker.warmup(frame_h, frame_w)
    tracker.start()

    # --- Ctrl-C handling ---
    stop = {"flag": False}
    def _sigint(_sig, _frm):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _sigint)

    # --- Control state ---
    cmd_yaw = 0.0          # controller target (integrated from error)
    cmd_pitch = 0.0
    sent_yaw = 0.0         # EMA-smoothed angle actually sent to the robot
    sent_pitch = 0.0
    err_x_s = 0.0          # EMA-smoothed detected error
    err_y_s = 0.0
    body_yaw = 0.0         # current commanded body yaw (slow-lagging)
    last_seen = 0.0
    last_log = 0.0

    print("Opening Reachy ... (Ctrl-C or q to quit)")
    # We drive body_yaw manually, so disable SDK-level auto to avoid conflict.
    with ReachyMini(automatic_body_yaw=False) as reachy:
        reachy.goto_target(head=head_pose(), duration=0.8, body_yaw=0.0)
        time.sleep(0.3)

        last_det_ts = 0.0          # timestamp of the last detection we acted on
        current_tier = TIER_NONE
        current_track_id: Optional[int] = None
        while not stop["flag"]:
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]

            # Hand the frame to the async cascade tracker and read whatever
            # the most recent result is. Control runs at camera FPS.
            tracker.submit(frame)
            det = tracker.get()

            have_face = False
            err_x = err_y = 0.0
            bbox_px = None
            if det is not None and det.timestamp != last_det_ts:
                err_x = det.err_x
                err_y = det.err_y
                bbox_px = det.bbox_px
                current_tier = det.tier
                current_track_id = det.track_id
                have_face = True
                last_seen = det.timestamp
                last_det_ts = det.timestamp
            elif det is not None and (time.time() - det.timestamp) < COAST_S:
                # No fresh result but the last one is still recent:
                # keep reusing its error so the controller doesn't freeze.
                err_x = det.err_x
                err_y = det.err_y
                bbox_px = det.bbox_px
                current_tier = det.tier
                current_track_id = det.track_id
                have_face = True
            else:
                current_tier = TIER_NONE

            now = time.time()

            if have_face:
                # Smooth the measurement itself to kill per-frame jitter.
                err_x_s = (1 - ERR_ALPHA) * err_x_s + ERR_ALPHA * err_x
                err_y_s = (1 - ERR_ALPHA) * err_y_s + ERR_ALPHA * err_y

            # Keep driving toward the (smoothed) last-known error during
            # short detection dropouts; only start recentering after COAST_S.
            active = have_face or (now - last_seen) < COAST_S
            if active:
                if abs(err_x_s) > DEADZONE:
                    cmd_yaw += YAW_SIGN * KP_YAW * err_x_s * STEP_SCALE
                if abs(err_y_s) > DEADZONE:
                    cmd_pitch += PITCH_SIGN * KP_PITCH * err_y_s * STEP_SCALE
            elif now - last_seen > RECENTER_AFTER_S:
                # No face for a while: decay error + command back toward center.
                err_x_s *= 0.9
                err_y_s *= 0.9
                cmd_yaw *= 0.95
                cmd_pitch *= 0.95

            cmd_yaw = clamp(cmd_yaw, -YAW_LIMIT, YAW_LIMIT)
            cmd_pitch = clamp(cmd_pitch, -PITCH_LIMIT, PITCH_LIMIT)

            # EMA-smooth what we actually send to the robot.
            sent_yaw = (1 - CMD_ALPHA) * sent_yaw + CMD_ALPHA * cmd_yaw
            sent_pitch = (1 - CMD_ALPHA) * sent_pitch + CMD_ALPHA * cmd_pitch

            # Body rotation: engages when head yaw exceeds BODY_ENGAGE_RAD,
            # then takes over BODY_FOLLOW_FRAC of the excess. Heavily smoothed
            # so the body drifts calmly while the head stays snappy.
            if args.no_body:
                body_target = 0.0
            elif cmd_yaw > BODY_ENGAGE_RAD:
                body_target = (cmd_yaw - BODY_ENGAGE_RAD) * BODY_FOLLOW_FRAC
            elif cmd_yaw < -BODY_ENGAGE_RAD:
                body_target = (cmd_yaw + BODY_ENGAGE_RAD) * BODY_FOLLOW_FRAC
            else:
                body_target = 0.0
            body_yaw = (1 - BODY_ALPHA) * body_yaw + BODY_ALPHA * body_target
            body_yaw = float(clamp(body_yaw, -BODY_LIMIT, BODY_LIMIT))

            reachy.set_target(
                head=head_pose(pitch=sent_pitch, yaw=sent_yaw),
                body_yaw=body_yaw,
            )

            # Occasional console heartbeat.
            if now - last_log > 0.5:
                tid = f"#{current_track_id}" if current_track_id is not None else "--"
                print(
                    f"tier={current_tier:<9s} {tid:<4s} "
                    f"err=({err_x_s:+.2f},{err_y_s:+.2f})  "
                    f"yaw={math.degrees(sent_yaw):+6.1f}  "
                    f"pitch={math.degrees(sent_pitch):+6.1f}  "
                    f"body={math.degrees(body_yaw):+6.1f}"
                )
                last_log = now

            # --- Preview ---
            if not args.no_preview:
                view = cv2.flip(frame, 1)   # mirror only for display
                color = TIER_COLORS.get(current_tier, (200, 200, 200))
                if bbox_px is not None:
                    x1, y1, x2, y2, dx, dy = bbox_px
                    # Flip x coords for the mirrored view.
                    x1m, x2m = w - x2, w - x1
                    dxm = w - dx
                    cv2.rectangle(view, (x1m, y1), (x2m, y2), color, 2)
                    cv2.circle(view, (dxm, dy), 5, color, -1)
                cv2.line(view, (w // 2, 0), (w // 2, h), (80, 80, 80), 1)
                cv2.line(view, (0, h // 2), (w, h // 2), (80, 80, 80), 1)
                tid = f"#{current_track_id}" if current_track_id is not None else "--"
                cv2.putText(
                    view,
                    f"{current_tier} {tid}  yaw={math.degrees(sent_yaw):+.0f}  "
                    f"pitch={math.degrees(sent_pitch):+.0f}  "
                    f"body={math.degrees(body_yaw):+.0f}  (q to quit)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 2,
                )
                cv2.imshow("reachy cascade-follow", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        print("\nStopping, recentering ...")
        reachy.goto_target(head=head_pose(), duration=0.6, body_yaw=0.0)

    tracker.stop()
    tracker.join(timeout=1.0)
    if tracker.infer_count:
        avg_ms = tracker.infer_total_ms / tracker.infer_count
        total = sum(tracker.tier_counts.values())
        print(f"Detector: {tracker.infer_count} inferences, "
              f"avg {avg_ms:.1f} ms ({1000.0/avg_ms:.1f} FPS)")
        if total:
            parts = [f"{k}={v}({100*v/total:.0f}%)"
                     for k, v in tracker.tier_counts.items() if v]
            print("Tier usage: " + "  ".join(parts))
    cap.release()
    if not args.no_preview:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
