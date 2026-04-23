"""
SAM 3 / 3.1 + CSRT tracker — lock on a concept and follow ONE instance
======================================================================

Unlike `sam3-test.py` (which re-runs the full SAM 3 detector every iteration
and therefore has no stable instance identity), this module combines:

    SAM 3.1   — slow, open-vocabulary: "find me a 'red mug'".
    CSRT      — fast, per-frame: once locked, follow *that* bbox at ~15–30 fps.

The CSRT bbox won't be as tight as a fresh SAM 3 bbox, but for head-follow
applications you typically feed the bbox *centroid* into a PID controller,
and a loose box with a stable center is far better than a tight box that
swaps identities when another matching object enters the frame.

A small state machine decides which runs when:

    IDLE       nothing to do.
    LOCKING    user gave us a concept; run SAM 3 to acquire the first bbox.
    TRACKING   CSRT updates every frame. SAM 3 re-grounds every N seconds to
               correct drift and validate we're still on the right object.
    LOST       CSRT lost the target; run SAM 3 to re-acquire.

Interface (thread-safe — call from anywhere):

    follower = Sam3Follower(Sam3Segmenter())
    follower.start_following("person with red shirt")
    while running:
        follower.push_frame(camera_frame_bgr)           # camera thread
        state, bbox = follower.get_current_bbox()       # any thread
    follower.stop_following()
    follower.close()

Running this file directly opens a Tk demo that exercises the full pipeline
on your webcam. The bbox it draws is the *tracker's* bbox, not a fresh SAM 3
output, so updates are smooth at camera-frame rate.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk

# Type alias — used throughout the follower
BBox = tuple[int, int, int, int]  # (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Device selection (MPS on Mac, CUDA if available, else CPU)
# ---------------------------------------------------------------------------
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()
print(f"[startup] Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# SAM 3 / 3.1 detector wrapper (same class as sam3-test.py, copy kept local
# so this file stays self-contained)
# ---------------------------------------------------------------------------
@dataclass
class Sam3Detection:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    score: float
    label: str


class Sam3Segmenter:
    """Ultralytics SAM3SemanticPredictor wrapper. Loads SAM 3 or SAM 3.1."""

    CANDIDATE_WEIGHTS = (
        "checkpoints/sam3.1_multiplex.pt",
        "checkpoints/sam3.pt",
    )

    def __init__(self, weights_path: Optional[str] = None, conf: float = 0.25):
        if weights_path is None:
            env = os.getenv("SAM3_WEIGHTS")
            if env:
                weights_path = env
            else:
                weights_path = next(
                    (p for p in self.CANDIDATE_WEIGHTS if os.path.isfile(p)),
                    self.CANDIDATE_WEIGHTS[0],
                )
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"SAM 3 weights not found at {weights_path!r}. "
                "See examples/sam3-test.py for download instructions."
            )

        print(f"[sam3] Loading weights from {weights_path}...")
        from ultralytics.models.sam import SAM3SemanticPredictor

        imgsz = int(os.getenv("SAM3_IMGSZ", "448"))
        overrides = dict(
            conf=conf,
            task="segment",
            mode="predict",
            model=weights_path,
            half=(DEVICE in ("cuda", "mps")),
            imgsz=imgsz,
            device=DEVICE,
            verbose=False,
            save=False,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        self._set_image_accepts_ndarray: Optional[bool] = None
        self._latencies: deque = deque(maxlen=50)
        print(f"[sam3] Ready (imgsz={imgsz}, half={overrides['half']}).")
        self._warmup()

    def _warmup(self) -> None:
        print("[sam3] Warming up...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            self.segment(dummy, ["warmup"])
            print("[sam3] Warm.")
        except Exception as e:
            print(f"[sam3] Warmup skipped: {e}")

    def segment(self, frame_bgr: np.ndarray, concepts: list[str]) -> list[Sam3Detection]:
        if not concepts:
            return []
        t0 = time.time()
        self._set_image(frame_bgr)
        results = self.predictor(text=concepts)
        total_ms = (time.time() - t0) * 1000
        self._latencies.append(total_ms)
        return self._to_detections(results, frame_bgr.shape[:2], concepts)

    def _set_image(self, frame_bgr: np.ndarray) -> None:
        if self._set_image_accepts_ndarray is not False:
            try:
                self.predictor.set_image(frame_bgr)
                self._set_image_accepts_ndarray = True
                return
            except Exception:
                self._set_image_accepts_ndarray = False
        tmp_path = "/tmp/_sam3_frame.jpg"
        cv2.imwrite(tmp_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.predictor.set_image(tmp_path)

    @staticmethod
    def _to_detections(
        results, frame_hw: tuple[int, int], concepts: list[str]
    ) -> list[Sam3Detection]:
        out: list[Sam3Detection] = []
        if not results:
            return out
        r = results[0]
        masks = getattr(r, "masks", None)
        boxes = getattr(r, "boxes", None)
        if masks is None or boxes is None:
            return out
        mask_arr = masks.data.detach().cpu().numpy()
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else None
        cls = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else None
        names = getattr(r, "names", None) or {}
        H, W = frame_hw
        for i in range(len(mask_arr)):
            m = mask_arr[i]
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            mask_bool = m > 0.5
            x1, y1, x2, y2 = xyxy[i].astype(int).tolist()
            score = float(conf[i]) if conf is not None else 1.0
            if cls is not None and cls[i] in names:
                label = str(names[int(cls[i])])
            elif cls is not None and 0 <= int(cls[i]) < len(concepts):
                label = concepts[int(cls[i])]
            else:
                label = concepts[0] if concepts else "?"
            out.append(Sam3Detection(mask=mask_bool, bbox=(x1, y1, x2, y2), score=score, label=label))
        return out


# ---------------------------------------------------------------------------
# Tracker helpers
# ---------------------------------------------------------------------------
def _create_csrt():
    """Create a CSRT tracker across OpenCV API variants."""
    # OpenCV >= 4.5.2 new API
    factory = getattr(cv2, "TrackerCSRT", None)
    if factory is not None and hasattr(factory, "create"):
        return factory.create()
    # Legacy API (cv2 4.x)
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
        return legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    raise RuntimeError(
        "No CSRT tracker in this OpenCV build. "
        "Install with: pip install opencv-contrib-python"
    )


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _area(b: BBox) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _area_ratio(a: BBox, b: BBox) -> float:
    """Return area(b) / area(a); `inf` if `a` has zero area (degenerate)."""
    area_a = _area(a)
    area_b = _area(b)
    if area_a <= 0:
        return float("inf")
    return area_b / area_a


def _is_same_target(a: BBox, b: BBox, max_center_shift_ratio: float = 1.5) -> bool:
    """Heuristic: same physical object drifted/moved, not a different instance.

    A low IoU between the CSRT bbox and a fresh SAM 3 bbox is ambiguous: it
    could mean drift (same target, tracker slipped) OR identity swap (different
    instance entirely). Check the center-to-center distance relative to box
    size: a truly different instance is typically more than ~1.5 box-widths
    away, while drift keeps the centers close.
    """
    cxa = (a[0] + a[2]) / 2
    cya = (a[1] + a[3]) / 2
    cxb = (b[0] + b[2]) / 2
    cyb = (b[1] + b[3]) / 2
    w = max(a[2] - a[0], 1)
    h = max(a[3] - a[1], 1)
    dist = ((cxa - cxb) ** 2 + (cya - cyb) ** 2) ** 0.5
    return dist < max_center_shift_ratio * max(w, h)


def _shrink_bbox(bbox: BBox, factor: float = 0.9) -> BBox:
    """Shrink around the center. SAM 3 boxes are usually ~10% looser than the
    actual object; passing the shrunk box to CSRT keeps the tracker's
    appearance model focused on foreground pixels and slows drift.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * factor
    h = (y2 - y1) * factor
    return (
        int(round(cx - w / 2)),
        int(round(cy - h / 2)),
        int(round(cx + w / 2)),
        int(round(cy + h / 2)),
    )


# ---------------------------------------------------------------------------
# Follower: concept -> stable bbox across frames
# ---------------------------------------------------------------------------
class FollowState(Enum):
    IDLE = auto()
    LOCKING = auto()    # no lock yet; SAM 3 searches for the concept
    TRACKING = auto()   # CSRT is following an instance
    LOST = auto()       # CSRT lost; SAM 3 re-acquires


class Sam3Follower:
    """Track one instance of a text concept across frames.

    Runs its own background thread. Call `push_frame` from the camera loop
    and `get_current_bbox` from anywhere. Thread-safe.

    Parameters
    ----------
    segmenter:
        A loaded `Sam3Segmenter`. Shared across followers is fine as long as
        only one follower uses it at a time (the predictor is not reentrant).
    reground_every_s:
        Seconds between periodic SAM 3 re-grounds while TRACKING. Lower =
        more correct, slower. Default 2.0 seconds is a good middle ground.
    iou_accept:
        Minimum IoU between the CSRT bbox and a fresh SAM 3 detection for us
        to snap the tracker to the SAM 3 bbox. Prevents re-latching onto a
        different instance. Default 0.3.
    """

    def __init__(
        self,
        segmenter: Sam3Segmenter,
        reground_every_s: float = 0.8,
        iou_accept: float = 0.3,
        center_shift_ratio: float = 1.5,
        init_shrink: float = 0.9,
        tracker_factory=None,
        tracker_name: str = "CSRT",
        # When re-grounding on small, semantically-ambiguous concepts
        # (e.g. "glasses" that sometimes match just the frames, sometimes
        # the whole face region), the fresh SAM bbox can be an order of
        # magnitude bigger/smaller than the tracker bbox while still being
        # centered on the same target. The center-proximity check alone
        # snaps through that scale jump and produces a jittery "box grows,
        # box shrinks" pattern. This guard rejects the snap when the area
        # ratio is outside [1/max_area_ratio, max_area_ratio]. 3.0 leaves
        # room for genuine scale changes (walking toward the camera) while
        # catching glasses <-> face -class jumps.
        max_area_ratio: float = 3.0,
    ):
        self.segmenter = segmenter
        self.reground_every_s = reground_every_s
        self.iou_accept = iou_accept
        # If the fresh SAM 3 bbox has low IoU with the tracker bbox but its
        # center is within `center_shift_ratio * max(w, h)` of the tracker
        # center, we treat it as the same target drifting and snap anyway.
        # This fixes the common drift-rejection bug where low IoU meant
        # "keep the bad box."
        self.center_shift_ratio = center_shift_ratio
        self.max_area_ratio = max(1.0, float(max_area_ratio))
        # Shrink SAM 3 boxes by this factor before initializing the tracker so
        # the tracker doesn't learn background as part of its appearance model.
        self.init_shrink = init_shrink
        # Per-frame tracker. Default = CSRT, but any OpenCV tracker with the
        # standard init/update API works (ViT, Nano, KCF, etc.). `sam3_vit_tracker.py`
        # swaps this for a cv2.TrackerVit factory.
        self._tracker_factory = tracker_factory or _create_csrt
        self.tracker_name = tracker_name

        # Guarded by _lock
        self._lock = threading.Lock()
        self._state = FollowState.IDLE
        self._concept: Optional[str] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_seq: int = 0
        self._last_used_seq: int = -1
        self._current_bbox: Optional[tuple[int, int, int, int]] = None
        self._current_score: float = 0.0
        self._last_update_ts: float = 0.0
        self._last_reground_ts: float = 0.0

        # Tracker is only touched from the worker thread
        self._tracker = None

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # -- public API -----------------------------------------------------

    def start_following(self, concept: str) -> None:
        """Begin acquiring + tracking the given concept."""
        concept = (concept or "").strip()
        if not concept:
            return
        with self._lock:
            self._concept = concept
            self._state = FollowState.LOCKING
            self._current_bbox = None
            self._current_score = 0.0
            self._last_reground_ts = 0.0
        # Tracker reset happens on the worker thread (not thread-safe to reuse)

    def stop_following(self) -> None:
        """Release the lock and return to IDLE."""
        with self._lock:
            self._concept = None
            self._state = FollowState.IDLE
            self._current_bbox = None
            self._current_score = 0.0

    def push_frame(self, frame_bgr: np.ndarray) -> None:
        """Give the follower the freshest camera frame (copies internally)."""
        if frame_bgr is None:
            return
        snap = frame_bgr.copy()
        with self._lock:
            self._latest_frame = snap
            self._frame_seq += 1

    def get_current_bbox(self) -> tuple[FollowState, Optional[tuple[int, int, int, int]], float]:
        """Return (state, bbox, age_seconds). bbox is None unless TRACKING/LOST-with-last."""
        with self._lock:
            age = time.time() - self._last_update_ts if self._last_update_ts else float("inf")
            return (self._state, self._current_bbox, age)

    def get_stats(self) -> dict:
        """Uniform stats accessor so consumers (e.g. the Reachy demo UI) can
        pull SAM-3 latency and tracker name from any of the three follower
        variants (`Sam3Follower` / `Sam3NativeFollower` / ViT wrapper).
        """
        lat = list(self.segmenter._latencies)  # deque of recent segment() ms
        return {
            "median_ms": (sorted(lat)[len(lat) // 2] if lat else 0.0),
            "mean_ms": (sum(lat) / len(lat) if lat else 0.0),
            "tracker_name": self.tracker_name,
            "score": self._current_score,
        }

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    # -- worker loop ----------------------------------------------------

    def _grab_frame(self) -> tuple[Optional[np.ndarray], Optional[str], FollowState, int]:
        with self._lock:
            return (
                self._latest_frame,
                self._concept,
                self._state,
                self._frame_seq,
            )

    def _set_state(self, state: FollowState) -> None:
        with self._lock:
            self._state = state

    def _set_bbox(self, bbox: Optional[tuple[int, int, int, int]], score: float) -> None:
        with self._lock:
            self._current_bbox = bbox
            self._current_score = score
            self._last_update_ts = time.time()

    def _run(self) -> None:
        while not self._stop.is_set():
            frame, concept, state, seq = self._grab_frame()
            if frame is None or state == FollowState.IDLE or concept is None:
                time.sleep(0.03)
                continue
            if seq == self._last_used_seq and state == FollowState.TRACKING:
                # Nothing new; avoid burning CPU on stale frames.
                time.sleep(0.005)
                continue
            self._last_used_seq = seq

            try:
                if state == FollowState.LOCKING:
                    self._handle_locking(frame, concept)
                elif state == FollowState.TRACKING:
                    self._handle_tracking(frame, concept)
                elif state == FollowState.LOST:
                    self._handle_lost(frame, concept)
            except Exception as e:
                print(f"[follower] worker error: {e}")
                self._set_state(FollowState.LOST)
                time.sleep(0.1)

    # -- per-state handlers --------------------------------------------

    def _acquire(
        self,
        frame: np.ndarray,
        concept: str,
        current_bbox: Optional[BBox] = None,
    ) -> Optional[Sam3Detection]:
        """Run SAM 3; pick the detection that best matches `current_bbox` if
        given (re-ground mode), otherwise the highest-scoring one."""
        dets = self.segmenter.segment(frame, [concept])
        if not dets:
            return None
        if current_bbox is None:
            return max(dets, key=lambda d: d.score)

        # Re-ground: rank detections by (same_target_bool, iou, score).
        # This picks the detection that's most plausibly our current target.
        def rank(d: Sam3Detection) -> tuple:
            iou = _iou(current_bbox, d.bbox)
            same = _is_same_target(current_bbox, d.bbox, self.center_shift_ratio)
            return (int(same), iou, d.score)

        return max(dets, key=rank)

    def _init_tracker(self, frame: np.ndarray, bbox: BBox) -> bool:
        """Initialize the per-frame tracker on a shrunk version of `bbox`.
        Returns False when the shrunk box is too small to track reliably.
        """
        x1, y1, x2, y2 = _shrink_bbox(bbox, self.init_shrink)
        w, h = x2 - x1, y2 - y1
        if w < 10 or h < 10:
            print(f"[follower] refusing degenerate bbox {bbox} (shrunk={w}x{h})")
            self._tracker = None
            return False
        self._tracker = self._tracker_factory()
        self._tracker.init(frame, (int(x1), int(y1), int(w), int(h)))
        return True

    def _handle_locking(self, frame: np.ndarray, concept: str) -> None:
        print(f"[follower] LOCKING on '{concept}' ...")
        best = self._acquire(frame, concept)
        if best is None:
            # Stay in LOCKING; next frame we'll try again. Avoid hammering.
            time.sleep(0.05)
            return
        if not self._init_tracker(frame, best.bbox):
            time.sleep(0.05)
            return
        self._set_bbox(best.bbox, best.score)
        self._last_reground_ts = time.time()
        self._set_state(FollowState.TRACKING)
        print(f"[follower] LOCKED on {best.label} at {best.bbox} (score={best.score:.2f})")

    def _handle_tracking(self, frame: np.ndarray, concept: str) -> None:
        if self._tracker is None:
            self._set_state(FollowState.LOCKING)
            return
        ok, rect = self._tracker.update(frame)
        if not ok:
            print(f"[follower] {self.tracker_name} lost target -> LOST")
            self._tracker = None
            self._set_state(FollowState.LOST)
            return
        x, y, w, h = rect
        tracker_bbox = (int(x), int(y), int(x + w), int(y + h))
        self._set_bbox(tracker_bbox, self._current_score)

        # Periodic re-ground: blocks CSRT for ~400 ms every N seconds, but
        # correcting drift matters more than a brief freeze for a head-follow
        # loop running on a 1–2 Hz planner.
        now = time.time()
        if now - self._last_reground_ts >= self.reground_every_s:
            # Use biased acquire: bias towards the detection that best matches
            # the current CSRT bbox. Prevents swapping to a different instance.
            fresh = self._acquire(frame, concept, current_bbox=tracker_bbox)
            self._last_reground_ts = time.time()
            if fresh is None:
                print(f"[follower] reground: SAM 3 found nothing; keep {self.tracker_name} bbox")
                return
            overlap = _iou(tracker_bbox, fresh.bbox)
            same = _is_same_target(tracker_bbox, fresh.bbox, self.center_shift_ratio)
            area_ratio = _area_ratio(tracker_bbox, fresh.bbox)
            scale_ok = (1.0 / self.max_area_ratio) <= area_ratio <= self.max_area_ratio

            if (overlap >= self.iou_accept or same) and scale_ok:
                # Same target AND same scale — snap to fresh SAM box to
                # cancel drift. Covers both "normal" drift (low IoU but
                # close centers) and clean re-grounds (high IoU).
                if self._init_tracker(frame, fresh.bbox):
                    self._set_bbox(fresh.bbox, fresh.score)
                    reason = "iou" if overlap >= self.iou_accept else "same-target"
                    print(
                        f"[follower] reground snap ({reason}, "
                        f"iou={overlap:.2f}, area_ratio={area_ratio:.2f}, "
                        f"score={fresh.score:.2f})"
                    )
            elif (overlap >= self.iou_accept or same) and not scale_ok:
                # Same *region* but a very different *scale*. This is the
                # classic glasses <-> face ambiguity: SAM flipped which
                # semantic level it matched. Refuse to snap so the tracker
                # box stays stable; we'll get another chance next reground.
                print(
                    f"[follower] reground: scale mismatch "
                    f"(area_ratio={area_ratio:.2f}, iou={overlap:.2f}); "
                    f"keep {self.tracker_name} bbox"
                )
            else:
                # Far away — likely a different instance; don't swap identity.
                print(
                    f"[follower] reground: different instance "
                    f"(iou={overlap:.2f}); keep {self.tracker_name} bbox"
                )

    def _handle_lost(self, frame: np.ndarray, concept: str) -> None:
        best = self._acquire(frame, concept)
        if best is None:
            time.sleep(0.1)
            return
        if not self._init_tracker(frame, best.bbox):
            time.sleep(0.1)
            return
        self._set_bbox(best.bbox, best.score)
        self._last_reground_ts = time.time()
        self._set_state(FollowState.TRACKING)
        print(f"[follower] RE-ACQUIRED {best.label} at {best.bbox}")


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


def pick_default_camera(cams: list[tuple[int, str]]) -> int:
    for idx, label in cams:
        if "reachy" not in label.lower():
            return idx
    return cams[0][0] if cams else 0


# ---------------------------------------------------------------------------
# Tk demo
# ---------------------------------------------------------------------------
@dataclass
class AppState:
    status: str = "Idle. Type a concept and press Lock on."
    last_bbox: Optional[tuple[int, int, int, int]] = None
    last_state: FollowState = FollowState.IDLE


class App:
    CAM_W, CAM_H = 640, 480
    CAP_BACKEND = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    STATE_COLORS = {
        FollowState.IDLE: "gray",
        FollowState.LOCKING: "orange",
        FollowState.TRACKING: "lime green",
        FollowState.LOST: "red",
    }

    def __init__(self, root: tk.Tk, follower: Sam3Follower):
        self.root = root
        self.follower = follower
        self.state = AppState()

        self.cameras = discover_cameras()
        print(f"[camera] Detected: {self.cameras}")
        self.current_cam_index = pick_default_camera(self.cameras)

        self._build_ui()

        self.cap: Optional[cv2.VideoCapture] = None
        self._open_camera(self.current_cam_index)

        self.root.after(0, self._tick)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _open_camera(self, index: int) -> bool:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        cap = cv2.VideoCapture(index, self.CAP_BACKEND)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAM_H)
        if not cap.isOpened():
            self._set_status(f"ERROR: Could not open camera {index}.", "red")
            self.cap = None
            return False
        self.cap = cap
        self.current_cam_index = index
        return True

    def _build_ui(self):
        self.root.title("SAM 3 + CSRT follower")
        self.root.minsize(720, 640)

        cam_row = ttk.Frame(self.root)
        cam_row.pack(fill="x", side="top", padx=8, pady=(8, 4))
        ttk.Label(cam_row, text="Camera:").pack(side="left")
        cam_values = [label for _i, label in self.cameras] or ["(none detected)"]
        self.cam_combo = ttk.Combobox(cam_row, values=cam_values, state="readonly", width=36)
        self.cam_combo.set(cam_values[0])
        self.cam_combo.pack(side="left", padx=(6, 6))
        self.cam_combo.bind("<<ComboboxSelected>>", self._on_camera_change)

        controls = ttk.Frame(self.root)
        controls.pack(fill="x", side="top", padx=8, pady=(0, 4))
        ttk.Label(controls, text="Concept:").pack(side="left")
        self.entry = ttk.Entry(controls, width=30)
        self.entry.pack(side="left", padx=(6, 6))
        self.entry.insert(0, "person")
        self.entry.bind("<Return>", lambda _e: self._on_lock())

        self.lock_btn = ttk.Button(controls, text="Lock on", command=self._on_lock)
        self.lock_btn.pack(side="left")
        self.release_btn = ttk.Button(controls, text="Release", command=self._on_release)
        self.release_btn.pack(side="left", padx=(6, 0))

        self.status_label = ttk.Label(self.root, text=self.state.status, foreground="black")
        self.status_label.pack(fill="x", side="top", padx=8, pady=(0, 8))

        self.video_label = ttk.Label(self.root)
        self.video_label.pack(side="top", padx=8, pady=8)

    def _on_camera_change(self, _event=None):
        label = self.cam_combo.get()
        for idx, lbl in self.cameras:
            if lbl == label:
                self._open_camera(idx)
                return

    def _set_status(self, text: str, color: str = "black"):
        self.state.status = text
        self.status_label.config(text=text, foreground=color)

    def _on_lock(self):
        concept = self.entry.get().strip()
        if not concept:
            self._set_status("Type a concept first.", "orange")
            return
        self.follower.start_following(concept)
        self._set_status(f"Locking on '{concept}' ...", "blue")

    def _on_release(self):
        self.follower.stop_following()
        self._set_status("Released.", "black")

    def _tick(self):
        if self.cap is None:
            self.root.after(100, self._tick)
            return
        ok, frame = self.cap.read()
        if not ok:
            self.root.after(33, self._tick)
            return

        # Feed the tracker with the newest frame.
        self.follower.push_frame(frame)

        # Query the follower (non-blocking).
        f_state, bbox, age = self.follower.get_current_bbox()

        display = frame
        if bbox is not None and f_state in (FollowState.TRACKING, FollowState.LOST):
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if f_state is FollowState.TRACKING else (0, 0, 255)
            display = frame.copy()
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            # Centroid dot — this is the point a head-follow controller should
            # use. Even when the CSRT box is loose, the centroid usually sits
            # on the target, which is what the robot actually cares about.
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(display, (cx, cy), 6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(display, (cx, cy), 4, (0, 255, 255), -1, cv2.LINE_AA)
            tag = f"{f_state.name}  age={age:.2f}s"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(display, tag, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Only update status when state transitions (avoids flicker).
        if f_state != self.state.last_state:
            self._set_status(
                f"State: {f_state.name}",
                self.STATE_COLORS.get(f_state, "black"),
            )
            self.state.last_state = f_state

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.configure(image=self._photo)
        self.root.after(33, self._tick)

    def _on_close(self):
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
def main():
    print("[startup] Initializing SAM 3 segmenter + follower.")
    try:
        segmenter = Sam3Segmenter()
    except (FileNotFoundError, ImportError) as e:
        print(f"\n[startup] {e}\n")
        sys.exit(1)

    follower = Sam3Follower(segmenter)

    root = tk.Tk()
    App(root, follower)
    root.mainloop()


if __name__ == "__main__":
    main()
