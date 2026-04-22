"""face_db.py - Persistent on-disk store of face embeddings.

One ``.npy`` file per person under ``data/faces/<name>.npy``, shape
``(N, D)`` of L2-normalized float32 embeddings. Simple, inspectable,
easy to delete a person by removing a file.

Matching is cosine similarity (equivalent to a dot product since the
stored and query embeddings are both L2-normalized). The DB returns
the best-matching name only when the similarity exceeds
``match_threshold``; otherwise it reports ``unknown``.
"""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# Keep the DB compact: once a person has this many embeddings, new ones
# push out the oldest. 50 * 512 * 4 bytes = ~100 KB per person.
MAX_EMBEDDINGS_PER_PERSON = 50

# Cosine-similarity threshold for a confident match. FaceNet (VGGFace2)
# same-person pairs typically score 0.7+, cross-person < 0.5.
DEFAULT_MATCH_THRESHOLD = 0.6


_NAME_RE = re.compile(r"[^a-z0-9_-]+")


def canonicalize_name(name: str) -> str:
    """Normalize a free-form name to a safe, consistent storage key.

    'Husam Abusafa' -> 'husam_abusafa'. Keeps lookups case-insensitive
    and keeps the filename safe.
    """
    n = name.strip().lower().replace(" ", "_")
    n = _NAME_RE.sub("", n)
    return n


class FaceDB:
    """Thread-safe in-memory map of ``name -> (N, D)`` embeddings, mirrored to disk."""

    def __init__(
        self,
        dir_path: Path,
        *,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> None:
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.threshold = float(match_threshold)
        self._data: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self.load()

    # ---- persistence ---------------------------------------------------
    def load(self) -> None:
        with self._lock:
            self._data.clear()
            for f in sorted(self.dir.glob("*.npy")):
                name = f.stem
                try:
                    arr = np.load(f)
                except Exception as e:
                    log.warning("FaceDB: failed to load %s: %s", f, e)
                    continue
                if arr.ndim != 2 or arr.shape[0] == 0:
                    log.warning("FaceDB: skipping malformed %s (shape=%s)",
                                f, arr.shape)
                    continue
                self._data[name] = arr.astype(np.float32, copy=False)
            log.info("FaceDB: loaded %d people from %s: %s",
                     len(self._data), self.dir,
                     ", ".join(self._data.keys()) or "(empty)")

    def _save(self, name: str) -> None:
        arr = self._data[name]
        np.save(self.dir / f"{name}.npy", arr)

    # ---- public API ----------------------------------------------------
    def add(self, name: str, new_embs: np.ndarray) -> int:
        """Append new embeddings for ``name`` and persist. Returns total count."""
        key = canonicalize_name(name)
        if not key:
            raise ValueError("name cannot be empty")
        if new_embs.ndim != 2 or new_embs.shape[0] == 0:
            raise ValueError(
                f"new_embs must be (N, D) with N>=1, got {new_embs.shape}"
            )
        with self._lock:
            existing = self._data.get(key)
            combined = (
                np.concatenate([existing, new_embs], axis=0)
                if existing is not None else new_embs
            ).astype(np.float32, copy=False)
            if combined.shape[0] > MAX_EMBEDDINGS_PER_PERSON:
                combined = combined[-MAX_EMBEDDINGS_PER_PERSON:]
            self._data[key] = combined
            self._save(key)
            return combined.shape[0]

    def remove(self, name: str) -> bool:
        key = canonicalize_name(name)
        with self._lock:
            if key not in self._data:
                return False
            self._data.pop(key, None)
            try:
                (self.dir / f"{key}.npy").unlink(missing_ok=True)
            except Exception as e:
                log.warning("FaceDB: failed to delete %s: %s", key, e)
            return True

    def identify(self, emb: np.ndarray) -> Tuple[Optional[str], float]:
        """Return ``(name, similarity)`` or ``(None, best_similarity)`` if below threshold.

        ``emb`` must be a 1-D, L2-normalized float32 array with the same
        dimensionality as the stored embeddings.
        """
        if emb.ndim != 1:
            raise ValueError(f"emb must be 1-D, got shape {emb.shape}")
        with self._lock:
            if not self._data:
                return None, 0.0
            best_name: Optional[str] = None
            best_sim = -1.0
            for name, mat in self._data.items():
                # Cosine similarity (both sides are L2-normalized).
                sims = mat @ emb
                s = float(sims.max())
                if s > best_sim:
                    best_sim = s
                    best_name = name
            if best_sim >= self.threshold:
                return best_name, best_sim
            return None, best_sim

    def list_names(self) -> List[str]:
        with self._lock:
            return sorted(self._data.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
