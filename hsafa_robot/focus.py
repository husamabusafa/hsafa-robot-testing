"""focus.py - Decide which person the robot should look at.

Glues together the three independent subsystems that each know about
*some* aspect of a person:

* :class:`CascadeTracker` - runs YOLOv8-Pose @ ~30 Hz and assigns each
  visible body a *bytetrack* ``track_id``. This is what the head/body
  servos actually follow.
* :class:`FaceRecognizer` - identifies a face bbox by name (via the
  :class:`FaceDB`).
* :class:`LipMotionTracker` - knows which visible face's mouth is
  currently moving.

The focus manager owns a small piece of state -- the user's *intent*
("follow Husam" / "follow whoever is speaking" / "default: largest
person") -- and each tick maps that intent into a concrete
``bytetrack_id`` and pushes it into :meth:`CascadeTracker.set_locked_id`.

Intent is mutated asynchronously by Gemini tool calls; the main loop
polls :meth:`update` every frame to apply it.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .face_db import canonicalize_name
from .face_recognizer import FaceMatch
from .lip_motion import LipMotionTracker, SpeakerCandidate
from .tracker import CascadeTracker

log = logging.getLogger(__name__)


Bbox = Tuple[int, int, int, int]


# ---- Focus modes ----------------------------------------------------------

MODE_AUTO = "auto"          # default: let the tracker pick the largest body
MODE_PERSON = "person"      # lock to a specific known name
MODE_SPEAKER = "speaker"    # follow whoever is currently speaking
VALID_MODES = (MODE_AUTO, MODE_PERSON, MODE_SPEAKER)


# How long a cached name <-> yolo_track_id mapping is considered valid.
# Short so a person swapping seats isn't followed by their old name.
_MAPPING_TTL_S = 2.0

# Minimum overlap-over-face-bbox-area required to say "this face sits
# inside this body track". Face bbox is small vs body bbox, so we don't
# care about IoU; we care whether the face is inside the body.
_MIN_FACE_IN_BODY = 0.5


# ---- Small geometry helper ------------------------------------------------

def _face_in_body_score(face_bbox: Bbox, body_bbox: Bbox) -> float:
    """Return fraction of ``face_bbox``'s area covered by ``body_bbox``.

    1.0 = face fully inside body bbox (normal case); 0.0 = disjoint.
    Much more useful than IoU here because face bboxes are tiny
    compared to body bboxes.
    """
    fx1, fy1, fx2, fy2 = face_bbox
    bx1, by1, bx2, by2 = body_bbox
    ix1, iy1 = max(fx1, bx1), max(fy1, by1)
    ix2, iy2 = min(fx2, bx2), min(fy2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    face_area = max(1, (fx2 - fx1) * (fy2 - fy1))
    return inter / face_area


def _match_face_to_yolo(
    face_bbox: Bbox,
    yolo_tracks: List[Tuple[int, Bbox]],
) -> Optional[int]:
    """Return the ``track_id`` of the YOLO body bbox that best contains ``face_bbox``."""
    best_id, best_score = None, 0.0
    for tid, body_bbox in yolo_tracks:
        score = _face_in_body_score(face_bbox, body_bbox)
        if score > best_score:
            best_score = score
            best_id = tid
    if best_id is None or best_score < _MIN_FACE_IN_BODY:
        return None
    return best_id


# ---- Snapshot of the current focus decision -------------------------------

@dataclass
class FocusSnapshot:
    """What the focus manager decided on the most recent ``update`` call."""
    mode: str                     # "auto" / "person" / "speaker"
    target_name: Optional[str]    # the canonical name we're trying to follow
    locked_id: Optional[int]      # bytetrack id currently pushed to the tracker
    focused_name: Optional[str]   # best-known name for the locked track
    # Full name <-> yolo_track_id mapping last produced. Useful for the
    # preview overlay (label every YOLO box, not just the focused one).
    name_by_track: Dict[int, str]

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "target_name": self.target_name,
            "focused_name": self.focused_name,
            "locked_id": self.locked_id,
        }


# ---- The manager ----------------------------------------------------------

class FocusManager:
    """Holds focus intent + produces per-tick decisions."""

    def __init__(
        self,
        tracker: CascadeTracker,
        lip_tracker: Optional[LipMotionTracker],
    ) -> None:
        self._tracker = tracker
        self._lip_tracker = lip_tracker
        self._lock = threading.Lock()

        self._mode = MODE_AUTO
        self._target_name: Optional[str] = None

        # Rolling name <-> yolo_track_id cache. Updated every tick from
        # the latest lip-motion snapshot (which already knows face
        # bboxes + names). Tuple of ``(track_id, ts)`` per name.
        self._name_to_yolo: Dict[str, Tuple[int, float]] = {}

        # Last-assigned locked id we pushed, to avoid chatty calls.
        self._last_pushed_id: Optional[int] = None

    # ---- Intent API (called from async Gemini tool handlers) ----------
    def set_mode_auto(self) -> None:
        with self._lock:
            self._mode = MODE_AUTO
            self._target_name = None
            log.info("FocusManager: mode=auto")
        # Immediately release the lock so the tracker goes back to
        # picking the largest body.
        self._tracker.set_locked_id(None)
        self._last_pushed_id = None

    def set_mode_person(self, name: str) -> bool:
        canonical = canonicalize_name(name)
        if not canonical:
            return False
        with self._lock:
            self._mode = MODE_PERSON
            self._target_name = canonical
            log.info("FocusManager: mode=person name=%s", canonical)
        return True

    def set_mode_speaker(self) -> None:
        with self._lock:
            self._mode = MODE_SPEAKER
            self._target_name = None
            log.info("FocusManager: mode=speaker")

    # ---- Per-frame update (called from the main control loop) ---------
    def update(
        self,
        yolo_tracks: List[Tuple[int, Bbox]],
        now: Optional[float] = None,
    ) -> FocusSnapshot:
        """Compute the desired ``locked_id`` and push it to the tracker.

        ``yolo_tracks`` comes from :meth:`CascadeTracker.get_all_tracks`;
        face data comes from the lip-motion snapshot cached inside this
        class. This is intentionally a read-mostly call so it's cheap
        to invoke every frame.
        """
        now = now if now is not None else time.time()
        with self._lock:
            mode = self._mode
            target_name = self._target_name

        name_by_track = self._refresh_name_mapping(yolo_tracks, now)

        desired: Optional[int] = None
        focused_name: Optional[str] = None

        if mode == MODE_PERSON and target_name:
            tid = self._lookup_name(target_name, now)
            if tid is not None:
                desired = tid
                focused_name = target_name
        elif mode == MODE_SPEAKER and self._lip_tracker is not None:
            speaker = self._lip_tracker.current_speaker()
            if speaker is not None:
                tid = _match_face_to_yolo(speaker.bbox, yolo_tracks)
                if tid is not None:
                    desired = tid
                    focused_name = speaker.name

        # MODE_AUTO (or a failed person/speaker lookup): do nothing --
        # the tracker keeps its built-in largest-body behavior.

        if desired != self._last_pushed_id:
            self._tracker.set_locked_id(desired)
            self._last_pushed_id = desired

        # Even in AUTO mode, fill in the name for whatever the tracker
        # *did* end up locking onto, so the preview overlay can show
        # "Husam" above his box.
        if focused_name is None:
            current = self._tracker.locked_id
            if current is not None and current in name_by_track:
                focused_name = name_by_track[current]

        return FocusSnapshot(
            mode=mode,
            target_name=target_name,
            locked_id=desired if desired is not None else self._tracker.locked_id,
            focused_name=focused_name,
            name_by_track=name_by_track,
        )

    # ---- helpers -------------------------------------------------------
    def _refresh_name_mapping(
        self,
        yolo_tracks: List[Tuple[int, Bbox]],
        now: float,
    ) -> Dict[int, str]:
        """Refresh the name <-> yolo_track_id cache from the lip snapshot."""
        out: Dict[int, str] = {}
        if self._lip_tracker is None or not yolo_tracks:
            return out

        snap = self._lip_tracker.snapshot()
        for cand in snap:
            if cand.name is None:
                continue
            tid = _match_face_to_yolo(cand.bbox, yolo_tracks)
            if tid is None:
                continue
            out[tid] = cand.name
            self._name_to_yolo[cand.name] = (tid, now)

        # Purge stale name -> id entries.
        cutoff = now - _MAPPING_TTL_S
        stale = [n for n, (_tid, ts) in self._name_to_yolo.items() if ts < cutoff]
        for n in stale:
            del self._name_to_yolo[n]

        return out

    def _lookup_name(self, name: str, now: float) -> Optional[int]:
        """Return the YOLO track_id currently bound to ``name``, or None."""
        entry = self._name_to_yolo.get(name)
        if entry is None:
            return None
        tid, ts = entry
        if (now - ts) > _MAPPING_TTL_S:
            return None
        return tid

    # ---- Pure-face helpers (used by the tool layer) -------------------
    def try_focus_by_face_match(
        self,
        face: FaceMatch,
        yolo_tracks: List[Tuple[int, Bbox]],
    ) -> Optional[int]:
        """Given a freshly-computed face bbox + yolo tracks, set the lock now.

        Used by the ``focus_on_person`` tool handler so the lock engages
        synchronously while the tool response is being sent, rather
        than waiting for the next ``update`` tick.
        """
        tid = _match_face_to_yolo(face.bbox, yolo_tracks)
        if tid is None:
            return None
        self._tracker.set_locked_id(tid)
        self._last_pushed_id = tid
        self._name_to_yolo[face.name or ""] = (tid, time.time())
        return tid
