"""identity_graph.py - Link face + voice + name + history into one 'person'.

``FaceDB`` stores face embeddings keyed by name. The
:class:`IdentityGraph` is the layer above that -- it treats each
person as an :class:`Identity` node to which *evidence* (face, voice,
name alias) is attached. This is what lets the robot recognize you
by voice when it can't see you, and learn your voice automatically
the first time you talk in front of the camera (the co-occurrence
enrollment described in ``docs/identity.md`` \u00a73).

Design:

* :class:`Identity` owns a stable UUID, a canonical name, aliases,
  and references to ``FaceDB`` / voice embedding files on disk.
* The graph itself persists as ``data/identity/graph.json``; face
  embeddings still live where :class:`FaceDB` puts them
  (``data/faces/<name>.npy``) and voice embeddings will follow the
  same pattern under ``data/voices/<name>.npy`` once the voice
  embedder lands.
* Bootstrap migration: at load time, every name in :class:`FaceDB`
  that doesn't already have an :class:`Identity` gets one
  auto-created. Existing deployments pick this up transparently on
  first run.

This module is deliberately *not* wired into the gaze policy or the
Gemini tool surface yet -- its shape is stable enough for the voice
embedder, the cross-modal enrollment thread, and (later) a Hsafa
bridge to all consume without a rewrite.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .face_db import FaceDB, canonicalize_name

log = logging.getLogger(__name__)


# ---- Data classes ---------------------------------------------------------

@dataclass
class Identity:
    """One person. Stable across face/voice evidence changes."""
    id: str
    canonical_name: str
    aliases: List[str] = field(default_factory=list)
    created_ts: float = field(default_factory=time.time)
    last_seen_ts: float = 0.0
    interaction_count: int = 0

    # Evidence references (counts only; raw embeddings live in their
    # own files so this JSON stays small).
    face_count: int = 0
    voice_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Identity":
        return cls(**{
            k: d.get(k, getattr(cls(id="", canonical_name=""), k))
            for k in cls.__dataclass_fields__
        })


@dataclass
class CoOccurrenceSample:
    """One clean (face, voice) pair, cached until we have enough to commit."""
    identity_id: str
    ts: float
    voice_embedding: np.ndarray   # (D,), L2-normalized float32


# ---- The graph -----------------------------------------------------------

class IdentityGraph:
    """Persistent face+voice+name graph stored under ``data/identity/``.

    Thread-safe. ``face_db`` remains the writer for per-name face
    embeddings; this class just links identities to the names
    ``FaceDB`` knows.
    """

    GRAPH_FILENAME = "graph.json"

    def __init__(
        self,
        root_dir: Path,
        face_db: FaceDB,
    ) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._graph_path = self.root / self.GRAPH_FILENAME
        self._voices_dir = self.root / "voices"
        self._voices_dir.mkdir(exist_ok=True)
        self._history_dir = self.root / "history"
        self._history_dir.mkdir(exist_ok=True)

        self._face_db = face_db
        self._lock = threading.RLock()

        # Canonical name -> Identity
        self._by_name: Dict[str, Identity] = {}
        # uuid -> Identity
        self._by_id: Dict[str, Identity] = {}

        # Pending voice samples waiting for enough co-occurrences
        # before being committed as permanent voice embeddings.
        self._pending_voice: Dict[str, List[CoOccurrenceSample]] = {}

        self._load()
        self._migrate_facedb()

    # ---- load / save --------------------------------------------------
    def _load(self) -> None:
        if not self._graph_path.exists():
            log.info("IdentityGraph: no existing graph; starting fresh.")
            return
        try:
            raw = json.loads(self._graph_path.read_text())
        except Exception as e:
            log.warning("IdentityGraph: could not read %s: %s", self._graph_path, e)
            return
        for d in raw.get("identities", []):
            ident = Identity.from_dict(d)
            self._by_id[ident.id] = ident
            self._by_name[ident.canonical_name] = ident
        log.info(
            "IdentityGraph: loaded %d identities: %s",
            len(self._by_id),
            ", ".join(i.canonical_name for i in self._by_id.values()) or "(none)",
        )

    def _save_locked(self) -> None:
        raw = {
            "version": 1,
            "saved_at": time.time(),
            "identities": [i.to_dict() for i in self._by_id.values()],
        }
        tmp = self._graph_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(raw, indent=2, sort_keys=True))
        tmp.replace(self._graph_path)

    def _migrate_facedb(self) -> None:
        """Create :class:`Identity` nodes for every FaceDB name without one."""
        existing_names = set(self._by_name.keys())
        to_add = []
        for name in self._face_db.list_names():
            canonical = canonicalize_name(name)
            if canonical in existing_names:
                continue
            to_add.append(canonical)
        if not to_add:
            return
        with self._lock:
            for name in to_add:
                ident = Identity(
                    id=str(uuid.uuid4()),
                    canonical_name=name,
                    face_count=self._face_count_for(name),
                )
                self._by_id[ident.id] = ident
                self._by_name[name] = ident
            self._save_locked()
        log.info("IdentityGraph: migrated %d FaceDB names into the graph", len(to_add))

    def _face_count_for(self, name: str) -> int:
        """Count how many face embeddings ``FaceDB`` has stored for ``name``."""
        # FaceDB doesn't expose this directly; cheap fallback: read
        # the npy shape. Keeps this module independent of FaceDB's
        # internals.
        path = self._face_db.dir / f"{name}.npy"
        if not path.exists():
            return 0
        try:
            arr = np.load(path)
            return int(arr.shape[0]) if arr.ndim == 2 else 0
        except Exception:
            return 0

    # ---- identity lookup / create --------------------------------
    def get_or_create_by_name(self, name: str) -> Identity:
        canonical = canonicalize_name(name)
        if not canonical:
            raise ValueError("name cannot be empty")
        with self._lock:
            ident = self._by_name.get(canonical)
            if ident is not None:
                return ident
            ident = Identity(id=str(uuid.uuid4()), canonical_name=canonical)
            self._by_id[ident.id] = ident
            self._by_name[canonical] = ident
            self._save_locked()
            return ident

    def get_by_name(self, name: str) -> Optional[Identity]:
        canonical = canonicalize_name(name)
        if not canonical:
            return None
        with self._lock:
            return self._by_name.get(canonical)

    def get_by_id(self, identity_id: str) -> Optional[Identity]:
        with self._lock:
            return self._by_id.get(identity_id)

    def list_identities(self) -> List[Identity]:
        with self._lock:
            return list(self._by_id.values())

    # ---- lifecycle updates --------------------------------------
    def note_seen(self, name: str, now: Optional[float] = None) -> None:
        """Stamp ``last_seen_ts`` / increment interaction count for a name."""
        now = now if now is not None else time.time()
        with self._lock:
            ident = self._by_name.get(canonicalize_name(name))
            if ident is None:
                return
            ident.last_seen_ts = now
            ident.interaction_count += 1
            self._save_locked()

    def record_face_enrollment(self, name: str, new_face_count: int) -> None:
        with self._lock:
            ident = self.get_or_create_by_name(name)
            ident.face_count = max(ident.face_count, int(new_face_count))
            self._append_history(ident.id, {
                "ts": time.time(), "kind": "face_enrolled",
                "face_count": ident.face_count,
            })
            self._save_locked()

    # ---- aliases / corrections ------------------------------------
    def add_alias(self, canonical_name: str, alias: str) -> bool:
        alias_norm = canonicalize_name(alias)
        if not alias_norm:
            return False
        with self._lock:
            ident = self._by_name.get(canonicalize_name(canonical_name))
            if ident is None:
                return False
            if alias_norm in ident.aliases:
                return True
            ident.aliases.append(alias_norm)
            self._save_locked()
            return True

    def apply_correction(
        self, wrong_name: str, correct_name: str,
    ) -> Optional[Identity]:
        """Rename / merge identities after a user correction.

        Intended as the backing action for the "I'm not Kindom, I'm
        Husam" correction flow. See ``docs/identity.md`` \u00a75. If
        ``correct_name`` exists, the two identities merge and the
        wrong alias is retired; otherwise ``wrong_name``'s identity
        is simply relabeled.
        """
        wrong_c = canonicalize_name(wrong_name)
        right_c = canonicalize_name(correct_name)
        if not wrong_c or not right_c or wrong_c == right_c:
            return None

        with self._lock:
            wrong_id = self._by_name.get(wrong_c)
            right_id = self._by_name.get(right_c)
            # Rewire the FaceDB bank via its rename() so on-disk
            # filenames match the new canonical name.
            try:
                self._face_db.rename(wrong_c, right_c)
            except Exception as e:
                log.warning("IdentityGraph: facedb rename failed: %s", e)

            if right_id is None and wrong_id is not None:
                # Plain rename.
                wrong_id.canonical_name = right_c
                wrong_id.aliases = [
                    a for a in wrong_id.aliases if a != wrong_c
                ]
                if wrong_c not in wrong_id.aliases:
                    wrong_id.aliases.append(wrong_c)
                del self._by_name[wrong_c]
                self._by_name[right_c] = wrong_id
                self._save_locked()
                return wrong_id

            if wrong_id is not None and right_id is not None:
                # Merge wrong_id INTO right_id.
                right_id.aliases = list({*right_id.aliases, *wrong_id.aliases, wrong_c})
                right_id.face_count += wrong_id.face_count
                right_id.voice_count += wrong_id.voice_count
                right_id.interaction_count += wrong_id.interaction_count
                del self._by_id[wrong_id.id]
                del self._by_name[wrong_c]
                self._save_locked()
                return right_id

            if wrong_id is None and right_id is not None:
                # Wrong name never existed -- treat as alias bind.
                self.add_alias(right_c, wrong_c)
                return right_id

        return None

    # ---- Cross-modal voice enrollment ----------------------------
    def stash_voice_sample(
        self, name: str, embedding: np.ndarray,
        *, min_samples: int = 5,
    ) -> int:
        """Record a fresh voice embedding while a face-matched speaker spoke.

        Returns the number of pending samples for that identity. Once
        at least ``min_samples`` have accumulated, call
        :meth:`commit_pending_voices` to write a consolidated
        ``voices/<name>.npy`` and mark the identity as voice-enrolled.
        """
        canonical = canonicalize_name(name)
        if not canonical:
            raise ValueError("name cannot be empty")
        ident = self.get_or_create_by_name(canonical)
        emb = np.ascontiguousarray(embedding, dtype=np.float32)
        n = float(np.linalg.norm(emb))
        if n <= 1e-8:
            return len(self._pending_voice.get(ident.id, []))
        emb = (emb / n).astype(np.float32)
        with self._lock:
            bucket = self._pending_voice.setdefault(ident.id, [])
            bucket.append(CoOccurrenceSample(
                identity_id=ident.id, ts=time.time(), voice_embedding=emb,
            ))
            return len(bucket)

    def identify_voice(
        self,
        embedding: np.ndarray,
        *,
        threshold: float = 0.70,
    ) -> Optional[Tuple[str, float]]:
        """Return ``(name, similarity)`` for the best-matching voice bank.

        Compares ``embedding`` (assumed L2-normalized) against every
        ``voices/<name>.npy`` using cosine similarity -- implemented
        as a simple dot product, mirroring :class:`FaceDB`. Only
        matches scoring at or above ``threshold`` are returned.

        Thread-safe: we cache an in-memory bank and hot-reload it
        when the matching ``.npy`` on disk grows.
        """
        emb = np.ascontiguousarray(embedding, dtype=np.float32)
        n = float(np.linalg.norm(emb))
        if n <= 1e-8:
            return None
        emb = (emb / n).astype(np.float32)

        best_name: Optional[str] = None
        best_sim: float = -1.0
        with self._lock:
            for path in sorted(self._voices_dir.glob("*.npy")):
                try:
                    bank = np.load(path)
                except Exception as e:
                    log.warning("identify_voice: could not load %s: %s", path, e)
                    continue
                if bank.ndim != 2 or bank.shape[1] != emb.shape[0]:
                    continue
                # Each row is already L2-normalized at stash time.
                sims = bank @ emb
                sim = float(sims.max())
                if sim > best_sim:
                    best_sim = sim
                    best_name = path.stem
        if best_name is None or best_sim < threshold:
            return None
        return best_name, best_sim

    def commit_pending_voices(self, min_samples: int = 5) -> List[str]:
        """Flush buckets with enough samples to disk. Returns committed names."""
        committed: List[str] = []
        with self._lock:
            for ident_id, bucket in list(self._pending_voice.items()):
                if len(bucket) < min_samples:
                    continue
                ident = self._by_id.get(ident_id)
                if ident is None:
                    continue
                embs = np.stack(
                    [s.voice_embedding for s in bucket], axis=0,
                ).astype(np.float32)
                path = self._voices_dir / f"{ident.canonical_name}.npy"
                # Append to existing bank, cap at 50 just like faces.
                if path.exists():
                    try:
                        prior = np.load(path)
                        embs = np.concatenate([prior, embs], axis=0)
                    except Exception:
                        pass
                if embs.shape[0] > 50:
                    embs = embs[-50:]
                np.save(path, embs)
                ident.voice_count = int(embs.shape[0])
                self._append_history(ident.id, {
                    "ts": time.time(),
                    "kind": "voice_enrolled",
                    "voice_count": ident.voice_count,
                })
                committed.append(ident.canonical_name)
                del self._pending_voice[ident_id]
            if committed:
                self._save_locked()
        return committed

    # ---- History (append-only JSONL) -----------------------------
    def _append_history(self, identity_id: str, entry: dict) -> None:
        path = self._history_dir / f"{identity_id}.jsonl"
        try:
            with path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:  # pragma: no cover
            log.warning("IdentityGraph: history append failed: %s", e)
