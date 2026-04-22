"""face_recognizer.py - MTCNN + FaceNet wrapper for enroll / identify.

Wraps ``facenet-pytorch``:

* :class:`MTCNN` finds the largest face in a frame, aligns and crops
  it to 160x160.
* :class:`InceptionResnetV1` (VGGFace2 weights) turns that crop into
  a 512-D embedding.

Both models are loaded lazily on first use so importing this module
doesn't immediately hit disk or network. Inference runs on CPU by
default -- we only call it on demand (enroll / identify), so a GPU is
not worth the portability cost here.

Embeddings are L2-normalized before being handed to :class:`FaceDB`.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .face_db import FaceDB, canonicalize_name

log = logging.getLogger(__name__)


FrameGetter = Callable[[], Optional[np.ndarray]]  # returns BGR uint8 frame or None


# ---- Public result type ---------------------------------------------------

# Horizontal position thresholds (fraction of frame width). A face is
# "center" when its centroid is in the middle ~33% of the image.
_POS_LEFT = 1.0 / 3.0
_POS_RIGHT = 2.0 / 3.0


def _bbox_position(bbox: Tuple[int, int, int, int], frame_w: int) -> str:
    """Classify a bbox's horizontal position as left / center / right."""
    x1, _y1, x2, _y2 = bbox
    if frame_w <= 0:
        return "center"
    cx = 0.5 * (x1 + x2) / float(frame_w)
    if cx < _POS_LEFT:
        return "left"
    if cx > _POS_RIGHT:
        return "right"
    return "center"


@dataclass
class FaceMatch:
    """One detected face, with optional identification + framing info.

    ``name`` is ``None`` when no known person scored above the DB's
    threshold; ``similarity`` is still the best score found.
    """
    name: Optional[str]
    similarity: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coords
    position: str                    # "left" | "center" | "right"
    det_prob: float                  # MTCNN face-detection confidence
    frame_w: int
    frame_h: int

    @property
    def bbox_area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def is_known(self) -> bool:
        return self.name is not None

    def to_dict(self) -> dict:
        """JSON-safe projection for sending to Gemini Live."""
        return {
            "name": self.name or "unknown",
            "similarity": round(self.similarity, 3),
            "position": self.position,
            "bbox": list(self.bbox),
            "bbox_area_px": self.bbox_area,
        }


# ---- Weight pre-download --------------------------------------------------
# facenet-pytorch's default downloader fails on macOS system Python because of
# the classic "unable to get local issuer certificate" SSL error. We mirror
# the approach used for the YOLO weights in ``tracker.py``: pre-download via
# urllib, falling back to ``curl`` if SSL fails, into the exact cache path
# that facenet-pytorch's ``load_weights`` will look at.

_VGGFACE2_URL = (
    "https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/"
    "20180402-114759-vggface2.pt"
)


def _torch_checkpoints_dir() -> Path:
    home = os.environ.get("TORCH_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "torch",
    )
    return Path(home) / "checkpoints"


def ensure_vggface2_weights() -> Path:
    """Ensure the VGGFace2 checkpoint is cached before InceptionResnetV1 loads.

    Returns the local path. Safe to call multiple times.
    """
    dst = _torch_checkpoints_dir() / os.path.basename(_VGGFACE2_URL)
    if dst.exists() and dst.stat().st_size > 1024 * 1024:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading VGGFace2 weights to %s ...", dst)
    try:
        urllib.request.urlretrieve(_VGGFACE2_URL, dst)
        log.info("VGGFace2 weights: download OK (urllib).")
        return dst
    except Exception as e:
        log.warning("urllib download failed (%s); trying curl ...", e)
    try:
        subprocess.run(
            ["curl", "-fsSL", "-o", str(dst), _VGGFACE2_URL],
            check=True,
        )
        log.info("VGGFace2 weights: download OK (curl).")
        return dst
    except Exception as e:
        print(
            f"Could not download VGGFace2 weights: {e}\n"
            f"Download manually from:\n  {_VGGFACE2_URL}\n"
            f"and save as:\n  {dst}\n",
            file=sys.stderr,
        )
        raise


# ---- Enrollment parameters ------------------------------------------------
# How many usable face embeddings to collect per enroll() call. Small enough
# to keep the robot responsive (~2-4 s), large enough that averaging across
# slight angle variation makes matching robust.
DEFAULT_NUM_SAMPLES = 5
# Maximum time to spend trying to collect those samples.
DEFAULT_ENROLL_TIMEOUT_S = 8.0
# Minimum gap between accepted samples so we capture pose variety, not 5
# near-duplicate frames.
SAMPLE_MIN_GAP_S = 0.25


class FaceRecognizer:
    """Face embedding + enrollment + identification, backed by a :class:`FaceDB`."""

    def __init__(
        self,
        db: FaceDB,
        *,
        device: str = "cpu",
        image_size: int = 160,
        mtcnn_margin: int = 20,
        detection_min_confidence: float = 0.90,
    ) -> None:
        self.db = db
        self.device = torch.device(device)
        self.image_size = image_size
        self.mtcnn_margin = mtcnn_margin
        self.detection_min_confidence = detection_min_confidence

        # Guards concurrent calls from the Gemini tool dispatcher. The
        # MTCNN / resnet are not designed for multi-threaded inference.
        self._lock = threading.Lock()
        self._mtcnn = None  # lazy
        self._resnet = None  # lazy
        self._loaded = False

    # ---- model loading -------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        # Import lazily so ``pip install facenet-pytorch`` is only required
        # for users who actually turn on face recognition.
        from facenet_pytorch import MTCNN, InceptionResnetV1

        # Pre-download the VGGFace2 checkpoint into the facenet-pytorch
        # cache path BEFORE instantiating the model, so its built-in
        # downloader (which fails on macOS system Python SSL) never runs.
        ensure_vggface2_weights()

        log.info(
            "FaceRecognizer: loading MTCNN + InceptionResnetV1 on %s ...",
            self.device,
        )
        self._mtcnn = MTCNN(
            image_size=self.image_size,
            margin=self.mtcnn_margin,
            keep_all=True,              # detect and return ALL faces
            post_process=True,          # normalize to FaceNet's expected range
            device=self.device,
        )
        self._resnet = (
            InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        )
        self._loaded = True
        log.info("FaceRecognizer: ready.")

    # ---- core embedding ------------------------------------------------
    def _detect_and_embed_all(
        self, frame_bgr: np.ndarray,
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        """Detect every face in the frame and embed it.

        Returns a list of ``(embedding, bbox_xyxy, detect_prob)`` tuples,
        one per face that passes ``detection_min_confidence``. Bboxes
        and embeddings are guaranteed to be in the same order (we ask
        MTCNN's ``extract`` to align them explicitly).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        self._ensure_loaded()

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        with torch.no_grad():
            # Step 1: bounding boxes only (no alignment yet).
            boxes, probs = self._mtcnn.detect(pil)  # type: ignore[union-attr]
            if boxes is None or len(boxes) == 0:
                return []

            # Step 2: filter by confidence and re-stack so ``extract``
            # aligns the same boxes we want to return.
            kept: List[Tuple[np.ndarray, float]] = []
            for b, p in zip(boxes, probs):
                if b is None or p is None:
                    continue
                if float(p) < self.detection_min_confidence:
                    continue
                kept.append((np.asarray(b, dtype=np.float32), float(p)))
            if not kept:
                return []

            kept_boxes = np.stack([b for b, _ in kept], axis=0)

            # Step 3: aligned 160x160 crops, batched by ``extract``.
            faces = self._mtcnn.extract(pil, kept_boxes, save_path=None)  # type: ignore[union-attr]
            if faces is None:
                return []
            if faces.dim() == 3:
                faces = faces.unsqueeze(0)

            embs = (
                self._resnet(faces.to(self.device)).detach().cpu().numpy()
            )

        out: List[Tuple[np.ndarray, Tuple[int, int, int, int], float]] = []
        for emb, (bbox, prob) in zip(embs, kept):
            n = float(np.linalg.norm(emb))
            if n <= 1e-8:
                continue
            emb_norm = (emb / n).astype(np.float32)
            x1, y1, x2, y2 = (int(v) for v in bbox.tolist())
            out.append((emb_norm, (x1, y1, x2, y2), prob))
        return out

    # ---- public API ----------------------------------------------------
    def detect_faces(
        self, frame_bgr: np.ndarray,
    ) -> List[Tuple[int, int, int, int]]:
        """MTCNN-only detection (no embedding). Returns bbox list.

        Cheap enough to call several times a second -- used by the
        lip-motion tracker to keep per-face motion scores warm without
        paying the InceptionResnetV1 cost every tick.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        self._ensure_loaded()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        with self._lock, torch.no_grad():
            boxes, probs = self._mtcnn.detect(pil)  # type: ignore[union-attr]
        if boxes is None:
            return []
        out: List[Tuple[int, int, int, int]] = []
        for b, p in zip(boxes, probs):
            if b is None or p is None:
                continue
            if float(p) < self.detection_min_confidence:
                continue
            x1, y1, x2, y2 = (int(v) for v in b.tolist())
            out.append((x1, y1, x2, y2))
        return out

    def identify_all_in_frame(
        self, frame_bgr: np.ndarray,
    ) -> List["FaceMatch"]:
        """Frame-direct variant of :meth:`identify_all` (no callback)."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        with self._lock:
            detections = self._detect_and_embed_all(frame_bgr)
        if not detections:
            return []
        h, w = frame_bgr.shape[:2]
        matches: List[FaceMatch] = []
        for emb, bbox, prob in detections:
            name, sim = self.db.identify(emb)
            matches.append(
                FaceMatch(
                    name=name,
                    similarity=float(sim),
                    bbox=bbox,
                    position=_bbox_position(bbox, w),
                    det_prob=float(prob),
                    frame_w=w,
                    frame_h=h,
                )
            )
        matches.sort(key=lambda m: m.bbox_area, reverse=True)
        return matches

    def identify_all(self, get_frame: FrameGetter) -> List["FaceMatch"]:
        """Detect every visible face and match each against the DB.

        Returns a list of :class:`FaceMatch` sorted by bbox area
        (largest / closest first). Empty list when no face is visible.
        Unknown faces are returned with ``name=None`` and
        ``similarity`` equal to the best score found.
        """
        frame = get_frame()
        if frame is None:
            return []
        return self.identify_all_in_frame(frame)

    def identify(
        self, get_frame: FrameGetter,
    ) -> Tuple[Optional[str], float, bool]:
        """Backward-compatible single-answer wrapper around :meth:`identify_all`.

        Returns ``(name, similarity, face_found)`` using the *highest-
        confidence recognized* match if any, otherwise the largest
        visible face (reported as ``name=None``).
        """
        matches = self.identify_all(get_frame)
        if not matches:
            return None, 0.0, False

        named = [m for m in matches if m.name is not None]
        if named:
            best = max(named, key=lambda m: m.similarity)
            return best.name, best.similarity, True
        # No recognized match, but we did see at least one face.
        largest = matches[0]
        return None, largest.similarity, True

    def find(
        self, get_frame: FrameGetter, name: str,
    ) -> Optional["FaceMatch"]:
        """Return the :class:`FaceMatch` for ``name`` if currently visible."""
        canonical = canonicalize_name(name)
        if not canonical:
            return None
        for m in self.identify_all(get_frame):
            if m.name == canonical:
                return m
        return None

    def enroll(
        self,
        name: str,
        get_frame: FrameGetter,
        *,
        num_samples: int = DEFAULT_NUM_SAMPLES,
        timeout_s: float = DEFAULT_ENROLL_TIMEOUT_S,
    ) -> int:
        """Capture ~``num_samples`` embeddings and save them under ``name``.

        Blocks until either ``num_samples`` usable embeddings have been
        collected OR ``timeout_s`` seconds elapse. Returns the number of
        *new* embeddings actually saved (0 means enrollment failed --
        e.g. no face was visible).
        """
        canonical = canonicalize_name(name)
        if not canonical:
            raise ValueError("name cannot be empty")

        collected: list[np.ndarray] = []
        last_accepted_ts = 0.0
        deadline = time.time() + timeout_s

        while len(collected) < num_samples and time.time() < deadline:
            now = time.time()
            # Enforce a small gap between samples so we don't save 5 copies
            # of a single instant.
            if now - last_accepted_ts < SAMPLE_MIN_GAP_S:
                time.sleep(SAMPLE_MIN_GAP_S - (now - last_accepted_ts))
                continue

            frame = get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            with self._lock:
                detections = self._detect_and_embed_all(frame)
            if not detections:
                time.sleep(0.05)
                continue

            # Enroll the LARGEST face in frame. When a user says "I am X"
            # they are almost always the closest / most prominent face;
            # picking by bbox area protects us from accidentally learning
            # a bystander's face in the background.
            def _area(d):
                x1, y1, x2, y2 = d[1]
                return max(0, x2 - x1) * max(0, y2 - y1)
            emb, bbox, _prob = max(detections, key=_area)

            collected.append(emb)
            last_accepted_ts = time.time()
            log.info("FaceRecognizer: captured %d/%d for '%s'",
                     len(collected), num_samples, canonical)

        if not collected:
            log.warning("FaceRecognizer: enroll('%s') failed - no face captured",
                        canonical)
            return 0

        arr = np.stack(collected, axis=0).astype(np.float32)
        total = self.db.add(canonical, arr)
        log.info(
            "FaceRecognizer: enrolled '%s' (+%d samples, %d total in DB)",
            canonical, len(collected), total,
        )
        return len(collected)
