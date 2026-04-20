"""
05_face_follow.py — Make Reachy Mini look at your face using YOLOv8-face.

Pipeline:
  webcam frame  →  YOLOv8-face detection (lindevs weights)  →  face bbox
  center  →  EMA-smoothed error  →  P-controller on head yaw/pitch  →
  EMA-smoothed command  →  set_target()

Speed tricks on Apple Silicon:
  * Inference on MPS (Metal GPU) if available; falls back to CPU.
  * YOLO input size dropped to 320 px (from 640) — ~2-3x faster on CPU.

The wired Reachy Mini exposes its camera as a normal UVC webcam on your Mac
(the daemon is started with --no-media, so it does NOT proxy the camera).
We open it directly with OpenCV. Pass --camera N if index 0 is not it.

The face weights (~6 MB) are auto-downloaded on first run into ./models/.

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

# Face-specific YOLOv8 weights from lindevs/yolov8-face (Apache-2.0).
FACE_MODEL_URL = (
    "https://github.com/lindevs/yolov8-face/releases/download/"
    "1.0.1/yolov8n-face-lindevs.pt"
)
FACE_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models" / "yolov8n-face-lindevs.pt"
)
# Inference image size — lower is faster. 224 is plenty for one near face.
YOLO_IMGSZ = 224
# Confidence threshold.
YOLO_CONF = 0.4


def head_pose(roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    return M


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def ensure_face_model() -> str:
    """Download the YOLOv8-face weights if missing (curl fallback for mac SSL)."""
    if FACE_MODEL_PATH.exists():
        return str(FACE_MODEL_PATH)
    FACE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading face model to {FACE_MODEL_PATH} ...")
    try:
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
        print("Done (urllib).")
        return str(FACE_MODEL_PATH)
    except Exception as e:
        print(f"urllib download failed ({e}); trying curl ...")
    try:
        subprocess.run(
            ["curl", "-fsSL", "-o", str(FACE_MODEL_PATH), FACE_MODEL_URL],
            check=True,
        )
        print("Done (curl).")
        return str(FACE_MODEL_PATH)
    except Exception as e:
        print(f"curl download failed: {e}", file=sys.stderr)
        print(f"Download it manually from:\n  {FACE_MODEL_URL}\n"
              f"and save as:\n  {FACE_MODEL_PATH}", file=sys.stderr)
        sys.exit(1)


def pick_device() -> str:
    """Prefer Apple Silicon GPU (MPS) if available, else CPU."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


@dataclass
class Detection:
    err_x: float        # normalized horizontal error in [-1, 1]
    err_y: float        # normalized vertical error in [-1, 1]
    bbox_px: tuple      # (x1, y1, x2, y2, cx, cy) in pixels
    timestamp: float    # wall clock when this detection was produced


class AsyncFaceDetector(threading.Thread):
    """Runs YOLO inference on a background thread.

    The main loop pushes the newest camera frame via `submit()` and reads the
    most recent face detection via `get()`. This decouples the slow detector
    (10-25 Hz on M-series CPU/MPS) from the fast control loop (~30 Hz).
    """
    def __init__(self, model_path: str, device: str,
                 imgsz: int, conf: float) -> None:
        super().__init__(daemon=True)
        self.detector = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self._new_frame_evt = threading.Event()
        self._lock = threading.Lock()
        self._pending_frame: Optional[np.ndarray] = None
        self._latest: Optional[Detection] = None
        self._stopped = False
        # Stats
        self.infer_count = 0
        self.infer_total_ms = 0.0

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

    def get(self) -> Optional[Detection]:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stopped = True
        self._new_frame_evt.set()

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
            t0 = time.perf_counter()
            results = self.detector.predict(
                frame, imgsz=self.imgsz, conf=self.conf,
                device=self.device, verbose=False,
            )
            self.infer_total_ms += (time.perf_counter() - t0) * 1000.0
            self.infer_count += 1

            det: Optional[Detection] = None
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()  # (N, 4) x1,y1,x2,y2
                areas = ((boxes[:, 2] - boxes[:, 0]) *
                         (boxes[:, 3] - boxes[:, 1]))
                i = int(np.argmax(areas))
                x1, y1, x2, y2 = boxes[i]
                fx = (x1 + x2) / 2.0
                fy = (y1 + y2) / 2.0
                err_x = (fx / w - 0.5) * 2.0
                err_y = (fy / h - 0.5) * 2.0
                det = Detection(
                    err_x=err_x, err_y=err_y,
                    bbox_px=(int(x1), int(y1), int(x2), int(y2),
                             int(fx), int(fy)),
                    timestamp=time.time(),
                )
            with self._lock:
                self._latest = det


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

    # --- Detector (YOLOv8-face, async on a background thread) ---
    model_path = ensure_face_model()
    device = pick_device()
    print(f"Loading YOLOv8-face on {device.upper()} (imgsz={YOLO_IMGSZ}) ...")
    async_det = AsyncFaceDetector(model_path, device, YOLO_IMGSZ, YOLO_CONF)
    async_det.warmup(frame_h, frame_w)
    async_det.start()

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
        while not stop["flag"]:
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]

            # Hand the frame to the async detector and read whatever the
            # most recent detection is. This loop runs at camera FPS even
            # if the detector is slower.
            async_det.submit(frame)
            det = async_det.get()

            have_face = False
            err_x = err_y = 0.0
            bbox_px = None
            if det is not None and det.timestamp != last_det_ts:
                err_x = det.err_x
                err_y = det.err_y
                bbox_px = det.bbox_px
                have_face = True
                last_seen = det.timestamp
                last_det_ts = det.timestamp
            elif det is not None and (time.time() - det.timestamp) < COAST_S:
                # No fresh detection, but the last one is still recent:
                # keep reusing its error so the controller doesn't freeze.
                err_x = det.err_x
                err_y = det.err_y
                bbox_px = det.bbox_px
                have_face = True

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
                print(
                    f"face={'Y' if have_face else '-'}  "
                    f"err=({err_x_s:+.2f},{err_y_s:+.2f})  "
                    f"yaw={math.degrees(sent_yaw):+6.1f}  "
                    f"pitch={math.degrees(sent_pitch):+6.1f}  "
                    f"body={math.degrees(body_yaw):+6.1f}"
                )
                last_log = now

            # --- Preview ---
            if not args.no_preview:
                view = cv2.flip(frame, 1)   # mirror only for display
                if bbox_px is not None:
                    x1, y1, x2, y2, dx, dy = bbox_px
                    # Flip x coords for the mirrored view.
                    x1m, x2m = w - x2, w - x1
                    dxm = w - dx
                    cv2.rectangle(view, (x1m, y1), (x2m, y2), (0, 255, 0), 2)
                    cv2.circle(view, (dxm, dy), 4, (0, 255, 255), -1)
                cv2.line(view, (w // 2, 0), (w // 2, h), (80, 80, 80), 1)
                cv2.line(view, (0, h // 2), (w, h // 2), (80, 80, 80), 1)
                cv2.putText(
                    view,
                    f"yaw={math.degrees(sent_yaw):+.0f}  "
                    f"pitch={math.degrees(sent_pitch):+.0f}  "
                    f"face={'Y' if have_face else 'N'}  (q to quit)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2,
                )
                cv2.imshow("reachy face-follow", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        print("\nStopping, recentering ...")
        reachy.goto_target(head=head_pose(), duration=0.6, body_yaw=0.0)

    async_det.stop()
    async_det.join(timeout=1.0)
    if async_det.infer_count:
        avg_ms = async_det.infer_total_ms / async_det.infer_count
        print(f"Detector: {async_det.infer_count} inferences, "
              f"avg {avg_ms:.1f} ms ({1000.0/avg_ms:.1f} FPS)")
    cap.release()
    if not args.no_preview:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
