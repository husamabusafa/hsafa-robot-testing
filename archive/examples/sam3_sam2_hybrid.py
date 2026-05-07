"""
SAM 3.1 detection + KLT fast-tracking hybrid test
==================================================

How it works
------------
1. SAM 3.1 detects the object by text prompt (~2 Hz) in a background thread.
2. When SAM 3.1 finds a bbox, we seed KLT feature points inside it.
3. Optical flow tracks those features every frame at ~30 FPS.
4. Median of tracked points = object centre, updated in real time.
5. Visual overlay shows:
   - Green box = SAM 3.1 detection (slow but accurate)
   - Red  dot  = KLT tracked centre (fast, every frame)

NOTE: SAM 2.1 has a camera-predictor API (`build_sam2_camera_predictor`)
in a third-party CUDA-only fork. It cannot be installed on macOS/MPS.
KLT optical flow is used instead — it is extremely fast (1-2 ms/frame)
and gives smooth, responsive tracking between SAM 3.1 detections.

No robot control — pure camera visualization for testing speed/quality.
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
from enum import Enum, auto
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk

BBox = tuple[int, int, int, int]

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
# Sam3Follower — wraps SAM 3.1 native text-based detection.
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
            half=(DEVICE == "cuda"),
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
        self._current_capture_ts: float = 0.0
        self._last_update_ts: float = 0.0
        self._latencies: deque = deque(maxlen=30)
        self._empty_count: int = 0

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
# KLT tracker — tracks object features inside the SAM bbox at camera rate.
# ---------------------------------------------------------------------------
class KltTracker:
    """Track an object region with KLT optical flow.

    When SAM gives a bbox we seed features inside it.
    Every camera frame we forward-track those features.
    Median of tracked feature positions = object centre.
    This runs at ~30 Hz (1-2 ms/frame), much faster than SAM.
    """

    def __init__(self):
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_kp: Optional[np.ndarray] = None
        self.cx: float = 0.0
        self.cy: float = 0.0
        self.lost: bool = True

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_kp = None
        self.cx = 0.0
        self.cy = 0.0
        self.lost = True

    def init(self, gray: np.ndarray, bbox: BBox) -> None:
        x1, y1, x2, y2 = bbox
        margin = 4
        roi = gray[y1 + margin:y2 - margin, x1 + margin:x2 - margin]
        if roi.size == 0:
            self.lost = True
            return
        kp = cv2.goodFeaturesToTrack(roi, maxCorners=60, qualityLevel=0.01, minDistance=5)
        if kp is not None and len(kp) >= 5:
            kp[:, 0, 0] += x1 + margin
            kp[:, 0, 1] += y1 + margin
            self.prev_kp = kp
        self.cx = float((x1 + x2) / 2)
        self.cy = float((y1 + y2) / 2)
        self.prev_gray = gray.copy()
        self.lost = False

    def update(self, gray: np.ndarray) -> tuple[float, float, bool]:
        """Return (cx, cy, is_lost)."""
        if self.lost or self.prev_gray is None or self.prev_kp is None:
            return self.cx, self.cy, True

        next_kp, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_kp, None,
            winSize=(21, 21), maxLevel=3
        )
        if next_kp is None:
            return self.cx, self.cy, True

        mask = status.reshape(-1) == 1
        if mask.sum() < 5:
            return self.cx, self.cy, True

        good = next_kp[mask]
        self.cx = float(np.median(good[:, 0, 0]))
        self.cy = float(np.median(good[:, 0, 1]))

        self.prev_gray = gray.copy()
        self.prev_kp = good.reshape(-1, 1, 2)
        self.lost = False
        return self.cx, self.cy, False


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
    TICK_MS = 33

    def __init__(
        self,
        root: tk.Tk,
        follower: Sam3Follower,
        tracker: KltTracker,
        initial_concept: str,
        prefer_reachy_camera: bool,
    ):
        self.root = root
        self.follower = follower
        self.tracker = tracker
        self.have_target: bool = False
        self._photo: Optional[ImageTk.PhotoImage] = None
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
        self.root.title("SAM 3.1 detect + SAM 2.1 track — visual test")
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

        self.status_label = ttk.Label(self.root, text="Idle.", foreground="black")
        self.status_label.pack(fill="x", padx=8, pady=(0, 4))

        self.stats_label = ttk.Label(
            self.root,
            text="SAM 3.1: —  |  SAM 2.1: —",
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
                self.tracker.reset()
                self.have_target = False
                self._open_camera(idx)
                return

    def _on_lock(self) -> None:
        concept = self.entry.get().strip()
        if not concept:
            self._set_status("Type a concept first.", "orange")
            return
        self.tracker.reset()
        self.have_target = False
        self.follower.start_following(concept)
        self._set_status(f"Locking on '{concept}' …", "blue")

    def _on_release(self) -> None:
        self.follower.stop_following()
        self.tracker.reset()
        self.have_target = False
        self._set_status("Released.", "black")

    def _tick(self) -> None:
        if self.cap is None:
            self.root.after(100, self._tick)
            return
        ok, frame = self.cap.read()
        if not ok:
            self.root.after(self.TICK_MS, self._tick)
            return

        # Downscale huge camera frames so SAM 3.1 doesn't choke.
        if frame.shape[0] > 720:
            scale = 720 / frame.shape[0]
            new_w = int(frame.shape[1] * scale)
            frame = cv2.resize(frame, (new_w, 720), interpolation=cv2.INTER_AREA)

        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Feed frame to SAM 3.1 detector (background thread).
        self.follower.push_frame(frame)

        # Get SAM 3.1 state.
        f_state, sam3_bbox, age, _capture_ts = self.follower.get_current_bbox()

        # When SAM 3.1 has a fresh detection, seed KLT tracker.
        if sam3_bbox is not None and f_state == FollowState.TRACKING:
            self.tracker.init(gray, sam3_bbox)
            self.have_target = not self.tracker.lost

        # Run KLT tracking every frame (fast, ~1-2 ms).
        if self.have_target:
            _cx, _cy, lost = self.tracker.update(gray)
            self.have_target = not lost

        # --- Display --------------------------------------------------
        self._tick_counter += 1
        do_display = (self._tick_counter % 2) == 0
        if not do_display:
            if f_state != self._last_state:
                self._set_status(
                    f"State: {f_state.name}  age={age:.2f}s",
                    STATE_COLORS.get(f_state, "black"),
                )
                self._last_state = f_state
            self.root.after(self.TICK_MS, self._tick)
            return

        if W > self.DISPLAY_MAX_W:
            disp_scale = self.DISPLAY_MAX_W / float(W)
            disp_w = int(round(W * disp_scale))
            disp_h = int(round(H * disp_scale))
            display = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        else:
            disp_scale = 1.0
            disp_w, disp_h = W, H
            display = frame.copy()

        # SAM 3.1 detection = green box + label.
        if sam3_bbox is not None:
            x1, y1, x2, y2 = sam3_bbox
            x1 = int(round(x1 * disp_scale))
            y1 = int(round(y1 * disp_scale))
            x2 = int(round(x2 * disp_scale))
            y2 = int(round(y2 * disp_scale))
            color = (0, 255, 0) if f_state == FollowState.TRACKING else (0, 0, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            label = "SAM3.1"
            if f_state == FollowState.LOCKING:
                label += " LOCKING"
            elif f_state == FollowState.LOST:
                label += " LOST"
            cv2.putText(
                display, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
            )

        # KLT tracking = red dot + line to image centre.
        if self.have_target:
            tx = int(round(self.tracker.cx * disp_scale))
            ty = int(round(self.tracker.cy * disp_scale))
            cv2.circle(display, (tx, ty), 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.drawMarker(
                display, (disp_w // 2, disp_h // 2), (255, 255, 255),
                cv2.MARKER_TILTED_CROSS, 12, 1, cv2.LINE_AA,
            )
            cv2.line(
                display, (disp_w // 2, disp_h // 2), (tx, ty),
                (0, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                display, "KLT", (tx + 8, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
            )

        # State label update.
        if f_state != self._last_state:
            self._set_status(
                f"State: {f_state.name}  age={age:.2f}s",
                STATE_COLORS.get(f_state, "black"),
            )
            self._last_state = f_state

        # Stats line.
        st = self.follower.get_stats()
        med_ms = st.get("median_ms", 0.0)
        sam3_fps = (1000.0 / med_ms) if med_ms else 0.0
        klt_status = "TRACKING" if self.have_target else "idle"
        self.stats_label.config(
            text=(
                f"SAM 3.1: {med_ms:4.0f} ms ({sam3_fps:4.1f} fps) obj={st.get('obj_id')}  |  "
                f"KLT: {klt_status}"
            )
        )

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.configure(image=self._photo)
        self.root.after(self.TICK_MS, self._tick)

    def _on_close(self) -> None:
        try:
            self.follower.stop_following()
            self.follower.close()
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
    ap.add_argument("--no-reachy-camera", action="store_true")
    args = ap.parse_args()

    print(f"[startup] device={DEVICE}")

    follower = Sam3Follower()
    tracker = KltTracker()

    root = tk.Tk()
    App(
        root,
        follower=follower,
        tracker=tracker,
        initial_concept=args.concept,
        prefer_reachy_camera=not args.no_reachy_camera,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
