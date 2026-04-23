"""
SAM 3.1 NATIVE tracker — detector + tracker shipped by the model itself
========================================================================

Tracks one instance of a text concept using the SAM 3.1 native video
predictor (detector + memory-based tracker in a single model).  No
CSRT/ViT fallback.

Per the SAM 3 design, the detector and the tracker share one Perception
Encoder backbone and are supposed to run together. The Ultralytics class
that exposes this is `SAM3VideoSemanticPredictor`. It was written to be
driven by Ultralytics' `stream_inference` pipeline with a file-backed
video, which does not play well with a live webcam (mode='stream',
`.frame` missing, inference_mode not wrapped around manual calls).

Instead of fighting that layer, we drive the predictor directly, one
frame at a time, sharing a single `torch.inference_mode()` context
across the session:

        predictor.dataset.frame = i + 1
        predictor.batch = (["camera"], [frame], [""])
        im = predictor.preprocess([frame])
        preds = predictor.inference(im, text=[concept])  # first frame
        preds = predictor.inference(im)                   # subsequent frames
        results = predictor.postprocess(preds, im, [frame])

`results[0].boxes.data` is a (N, 7) tensor:
    [x1, y1, x2, y2, obj_id, score, cls]

Identity is model-owned — the same `obj_id` persists across frames even
when the person briefly occludes. There is no CSRT drift to correct.

Performance note (Apple M2 Max, MPS, imgsz=448):
    * first frame:  ~1.2 s  (one-time backbone warm-up)
    * steady-state: ~0.4–1.5 s / frame depending on whether the detector
      also fires this step. You should expect 1–3 FPS from SAM alone.

Interface (thread-safe):

    follower = Sam3NativeFollower()
    follower.start_following("person with red shirt")
    while running:
        follower.push_frame(camera_frame_bgr)        # camera thread
        state, bbox, age = follower.get_current_bbox()
    follower.stop_following()
    follower.close()
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections import deque
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np
import torch

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
print(f"[startup] Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Minimal fake dataset — the Ultralytics SAM 3 predictor internals expect a
# video-mode dataset object with `.frame` and `.frames`. A live webcam
# dataset has neither shape, so we replace it with this tiny object after
# setup_source initializes everything else.
# ---------------------------------------------------------------------------
class _FakeVideoDataset:
    mode = "video"
    bs = 1

    def __init__(self, frame_cap: int = 1_000_000) -> None:
        self.frames = frame_cap
        self.frame = 0


# ---------------------------------------------------------------------------
# SAM 3 native follower — owns the predictor and a single long-running
# inference_mode context.
# ---------------------------------------------------------------------------
class FollowState(Enum):
    IDLE = auto()
    LOCKING = auto()    # started; no obj_id picked yet
    TRACKING = auto()   # tracking a specific obj_id
    LOST = auto()       # tracked obj_id disappeared; try to re-acquire


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


class Sam3NativeFollower:
    """Native SAM 3 follower: one worker thread drives the predictor."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        imgsz: Optional[int] = None,
        conf: float = 0.25,
        # Only knob that matters: how many consecutive frames with ZERO
        # detections of the concept before we declare LOST. At ~2 FPS this
        # is seconds. The object is not lost just because it moved fast;
        # it is lost when SAM stops seeing "person" at all.
        max_lost_frames: int = 8,
        debug: bool = False,
    ):
        weights = _pick_weights(weights_path)
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"SAM 3 weights not found at {weights!r}. "
                "Download from https://github.com/facebookresearch/sam3 "
                "and place in checkpoints/."
            )
        imgsz = int(os.getenv("SAM3_IMGSZ", imgsz or 448))
        print(f"[native] Loading {weights} (imgsz={imgsz}, device={DEVICE})...")
        from ultralytics.models.sam import SAM3VideoSemanticPredictor

        overrides = dict(
            model=weights,
            task="segment",
            mode="predict",
            device=DEVICE,
            half=(DEVICE in ("cuda", "mps")),
            imgsz=imgsz,
            conf=conf,
            verbose=False,
            save=False,
        )
        self.predictor = SAM3VideoSemanticPredictor(overrides=overrides)
        self.predictor.setup_model(model=None, verbose=False)

        # Kick setup_source with a 1-frame dummy so `self.imgsz`, tracker
        # backbone feature sizes, and the memory encoder are all initialized
        # correctly. We then swap the dataset for our video-mode fake.
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self.predictor.setup_source(dummy)
        self.predictor.dataset = _FakeVideoDataset()
        print(f"[native] Ready.")

        # One persistent inference_mode context — created in the worker so
        # the whole session stays under inference_mode. Tensors created by
        # `add_prompt` would otherwise leak into normal autograd and crash
        # in `layer_norm` ("Inference tensors cannot be saved for backward").
        self._inf_ctx: Optional[torch.inference_mode] = None

        # Configuration
        self.max_lost_frames = int(max_lost_frames)
        self.debug = bool(debug)

        # Guarded by _lock
        self._lock = threading.Lock()
        self._state = FollowState.IDLE
        self._concept: Optional[str] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_seq: int = 0
        self._last_used_seq: int = -1
        self._current_bbox: Optional[BBox] = None
        self._current_mask: Optional[np.ndarray] = None
        self._current_score: float = 0.0
        self._current_obj_id: Optional[int] = None
        self._last_update_ts: float = 0.0
        self._latencies: deque = deque(maxlen=30)
        self._empty_count: int = 0      # consecutive frames with zero detections
        self._drift_events: int = 0     # internal counter; not exposed in stats

        # Worker-thread-owned state
        self._pending_reset = False    # set on start/stop to re-init
        self._frame_index: int = 0     # 0-based SAM 3 frame counter

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # -- public API -----------------------------------------------------

    def start_following(self, concept: str) -> None:
        concept = (concept or "").strip()
        if not concept:
            return
        with self._lock:
            self._concept = concept
            self._state = FollowState.LOCKING
            self._current_bbox = None
            self._current_mask = None
            self._current_score = 0.0
            self._current_obj_id = None
            self._empty_count = 0
            self._drift_events = 0
            self._pending_reset = True

    def stop_following(self) -> None:
        with self._lock:
            self._concept = None
            self._state = FollowState.IDLE
            self._current_bbox = None
            self._current_mask = None
            self._current_score = 0.0
            self._current_obj_id = None
            self._empty_count = 0
            self._pending_reset = True

    def push_frame(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr is None:
            return
        snap = frame_bgr.copy()
        with self._lock:
            self._latest_frame = snap
            self._frame_seq += 1

    def get_current_bbox(self) -> tuple[FollowState, Optional[BBox], float]:
        with self._lock:
            age = (
                time.time() - self._last_update_ts
                if self._last_update_ts
                else float("inf")
            )
            return (self._state, self._current_bbox, age)

    def get_current_mask(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._current_mask is None else self._current_mask.copy()

    def get_stats(self) -> dict:
        with self._lock:
            lat = list(self._latencies)
            return {
                "median_ms": (sorted(lat)[len(lat) // 2] if lat else 0.0),
                "mean_ms": (sum(lat) / len(lat) if lat else 0.0),
                "obj_id": self._current_obj_id,
            }

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    # -- worker thread --------------------------------------------------

    def _run(self) -> None:
        # One long-lived inference_mode context for the whole worker life.
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
            self._step_once(frame, concept, state)
        except Exception as e:
            print(f"[native] step error: {e}")
            with self._lock:
                self._state = FollowState.LOST
                self._pending_reset = True
            time.sleep(0.1)

    def _reset_predictor_state(self) -> None:
        """Clear SAM 3 inference_state so the next frame re-prompts with text."""
        self.predictor.inference_state = {}
        self.predictor.dataset = _FakeVideoDataset()
        self.predictor.run_callbacks("on_predict_start")
        self._frame_index = 0

    def _step_once(self, frame: np.ndarray, concept: str, state: FollowState) -> None:
        """One SAM 3 step. Policy:

        * If SAM returns zero detections, increment an empty-counter; go
          LOST only after `max_lost_frames` consecutive empties.
        * Otherwise, pick the best-matching detection (see `_pick_best`)
          and keep TRACKING. An obj_id change is normal on MPS — counted
          as drift, not logged.
        """
        t0 = time.time()
        self.predictor.dataset.frame = self._frame_index + 1  # 1-based
        self.predictor.batch = (["camera"], [frame], [""])
        im = self.predictor.preprocess([frame])
        if "text_ids" not in self.predictor.inference_state:
            preds = self.predictor.inference(im, text=[concept])
        else:
            preds = self.predictor.inference(im)
        results = self.predictor.postprocess(preds, im, [frame])
        latency_ms = (time.time() - t0) * 1000

        dets = self._extract_dets(results[0])
        self._frame_index += 1

        if self.debug:
            print(
                f"[native] f={self._frame_index} lat={latency_ms:.0f}ms "
                f"state={state.name} ids={[d[1] for d in dets]} "
                f"locked={self._current_obj_id}"
            )

        with self._lock:
            self._latencies.append(latency_ms)

            # ---- No detections this frame -----------------------------
            if not dets:
                self._empty_count += 1
                if (
                    self._state in (FollowState.LOCKING, FollowState.TRACKING)
                    and self._empty_count > self.max_lost_frames
                ):
                    self._state = FollowState.LOST
                    print(
                        f"[native] LOST '{concept}' "
                        f"(no detections for {self._empty_count} frames)"
                    )
                return

            # ---- At least one detection; pick and track ---------------
            self._empty_count = 0
            box, obj_id, score, mask = self._pick_best(
                dets, self._current_bbox, self._current_obj_id
            )

            prev_state = self._state
            if prev_state == FollowState.LOCKING:
                print(
                    f"[native] LOCKED on obj_id={obj_id} "
                    f"score={score:.2f} bbox={box}"
                )
            elif prev_state == FollowState.LOST:
                print(
                    f"[native] RE-ACQUIRED obj_id={obj_id} "
                    f"score={score:.2f} bbox={box}"
                )
            self._state = FollowState.TRACKING

            if (
                self._current_obj_id is not None
                and obj_id != self._current_obj_id
            ):
                self._drift_events += 1

            self._current_obj_id = obj_id
            self._current_bbox = box
            self._current_mask = mask
            self._current_score = score
            self._last_update_ts = time.time()

    # -- detection helpers ---------------------------------------------

    @staticmethod
    def _mask_for_index(masks, k: int) -> Optional[np.ndarray]:
        if masks is None:
            return None
        try:
            return masks.data[k].detach().cpu().numpy().astype(bool)
        except Exception:
            return None

    def _extract_dets(self, r) -> list[tuple[BBox, int, float, Optional[np.ndarray]]]:
        boxes = getattr(r, "boxes", None)
        masks = getattr(r, "masks", None)
        if boxes is None or len(boxes) == 0:
            return []
        data = boxes.data.detach().cpu().numpy()
        if data.shape[1] < 7:
            return []
        out: list[tuple[BBox, int, float, Optional[np.ndarray]]] = []
        for k in range(data.shape[0]):
            box = tuple(int(v) for v in data[k, :4])
            obj_id = int(data[k, 4])
            score = float(data[k, 5])
            out.append((box, obj_id, score, self._mask_for_index(masks, k)))
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

    @staticmethod
    def _center(b: BBox) -> tuple[float, float]:
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    def _pick_best(self, dets, last_box: Optional[BBox], locked_id: Optional[int]):
        """Pick ONE detection. Priority:
        1. Same obj_id as currently locked (native tracking held up).
        2. Highest IoU with last bbox (overlap, typical slow motion).
        3. Nearest centroid to last bbox (handles fast motion — we still
           pick *something* instead of flipping to LOST).
        4. Highest score (first frame, or no prior bbox).
        """
        if locked_id is not None:
            for d in dets:
                if d[1] == locked_id:
                    return d
        if last_box is not None:
            best_iou_det = max(dets, key=lambda d: self._iou(last_box, d[0]))
            if self._iou(last_box, best_iou_det[0]) > 0:
                return best_iou_det
            cx, cy = self._center(last_box)
            return min(
                dets,
                key=lambda d: (
                    (self._center(d[0])[0] - cx) ** 2
                    + (self._center(d[0])[1] - cy) ** 2
                ),
            )
        return max(dets, key=lambda d: d[2])


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


