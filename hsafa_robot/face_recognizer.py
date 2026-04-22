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
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .face_db import FaceDB, canonicalize_name

log = logging.getLogger(__name__)


FrameGetter = Callable[[], Optional[np.ndarray]]  # returns BGR uint8 frame or None


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
            keep_all=False,             # largest face only
            post_process=True,          # normalize to FaceNet's expected range
            device=self.device,
        )
        self._resnet = (
            InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        )
        self._loaded = True
        log.info("FaceRecognizer: ready.")

    # ---- core embedding ------------------------------------------------
    def _embed_bgr(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """BGR uint8 frame -> L2-normalized 512-D float32 embedding, or None."""
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        self._ensure_loaded()

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        with torch.no_grad():
            # MTCNN may return (tensor, prob) when return_prob=True; we
            # just want the aligned face tensor.
            face_tensor, prob = self._mtcnn(pil, return_prob=True)  # type: ignore[misc]
            if face_tensor is None:
                return None
            if prob is not None and float(prob) < self.detection_min_confidence:
                return None
            face_tensor = face_tensor.to(self.device).unsqueeze(0)
            emb = self._resnet(face_tensor)[0].detach().cpu().numpy()

        n = float(np.linalg.norm(emb))
        if n <= 1e-8:
            return None
        return (emb / n).astype(np.float32)

    # ---- public API ----------------------------------------------------
    def identify(
        self, get_frame: FrameGetter,
    ) -> Tuple[Optional[str], float, bool]:
        """Embed the current frame and look it up in the DB.

        Returns ``(name, similarity, face_found)``. ``name`` is ``None``
        when no face is visible OR when the best match is below the
        DB's threshold.
        """
        frame = get_frame()
        if frame is None:
            return None, 0.0, False

        with self._lock:
            emb = self._embed_bgr(frame)
        if emb is None:
            return None, 0.0, False

        name, sim = self.db.identify(emb)
        return name, float(sim), True

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
                emb = self._embed_bgr(frame)
            if emb is None:
                time.sleep(0.05)
                continue

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
