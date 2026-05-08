"""perception.py - Build the WorldState from vision snapshots.

Composes the three independent visual senses into one
:class:`WorldState` list of humans:

* YOLO + ByteTrack bodies       -> stable ``track_id`` + body bbox
* Face recognizer / lip-motion  -> face bbox + name + speech state
* Head-pose module (optional)   -> per-face orientation signals
* Gesture module (optional)     -> per-body gesture list

This is the only place that maps "face bbox -> yolo track_id". If
that mapping moves, it moves here, not in three places.

The registry layer also gives each track a stable ``first_seen_ts``
across frames, which powers the gaze policy's recency bonus and the
``person_detected`` / ``person_left`` events emitted by the wiring
layer.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .world_state import Bbox, HumanView

log = logging.getLogger(__name__)


# ---- Position + distance helpers -----------------------------------------

_POS_LEFT = 1.0 / 3.0
_POS_RIGHT = 2.0 / 3.0


def _direction_from_bbox(bbox: Bbox, frame_w: int) -> str:
    x1, _y1, x2, _y2 = bbox
    if frame_w <= 0:
        return "center"
    cx = 0.5 * (x1 + x2) / float(frame_w)
    if cx < _POS_LEFT:
        return "left"
    if cx > _POS_RIGHT:
        return "right"
    return "center"


def _distance_from_proximity(prox: float) -> str:
    """Discretize a proximity ratio [0, 1] into near/mid/far buckets."""
    if prox > 0.18:
        return "near"
    if prox > 0.06:
        return "mid"
    return "far"


def _face_in_body_score(face: Bbox, body: Bbox) -> float:
    fx1, fy1, fx2, fy2 = face
    bx1, by1, bx2, by2 = body
    ix1, iy1 = max(fx1, bx1), max(fy1, by1)
    ix2, iy2 = min(fx2, bx2), min(fy2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    face_area = max(1, (fx2 - fx1) * (fy2 - fy1))
    return inter / face_area


# ---- Registry (per-track memory) -----------------------------------------

@dataclass
class _TrackMemo:
    first_seen: float
    last_seen: float
    # Sticky name: once a track has been identified, keep the name
    # even when the lip-motion tracker's 2.5 s identify cycle hasn't
    # re-confirmed it yet. Name is cleared only on track death.
    name: Optional[str] = None
    # Sticky head-pose / gestures so a single miss doesn't flicker
    # the WorldState between "facing" and "not facing".
    head_yaw_deg: Optional[float] = None
    head_pitch_deg: Optional[float] = None
    head_roll_deg: Optional[float] = None
    is_facing_camera: bool = False
    head_pose_last_update: float = 0.0
    active_gestures: List[str] = field(default_factory=list)
    gestures_last_update: float = 0.0
    emotion: Optional[str] = None
    emotion_last_update: float = 0.0


HEAD_POSE_TTL_S = 1.5
GESTURE_TTL_S = 1.5
EMOTION_TTL_S = 3.0
TRACK_FORGET_AFTER_S = 10.0


class HumanRegistry:
    """Per-track memory that survives across ticks.

    The YOLO tracker's ``get_all_tracks`` only returns *current* boxes;
    we need ``first_seen`` / sticky name / sticky head-pose to compose
    a useful :class:`HumanView`. This class is the home for those.

    Thread-safe so the head-pose / gesture threads can write into it
    without fighting the main-loop reader.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._memos: Dict[int, _TrackMemo] = {}

    # ---- writers ------------------------------------------------------
    def touch(self, track_id: int, now: float) -> _TrackMemo:
        with self._lock:
            memo = self._memos.get(track_id)
            if memo is None:
                memo = _TrackMemo(first_seen=now, last_seen=now)
                self._memos[track_id] = memo
            else:
                memo.last_seen = now
            return memo

    def remember_name(self, track_id: int, name: Optional[str]) -> None:
        if name is None:
            return
        with self._lock:
            memo = self._memos.get(track_id)
            if memo is None:
                return
            memo.name = name

    def set_head_pose(
        self,
        track_id: int,
        *,
        yaw_deg: Optional[float],
        pitch_deg: Optional[float],
        roll_deg: Optional[float],
        is_facing_camera: bool,
        now: Optional[float] = None,
    ) -> None:
        now = now if now is not None else time.monotonic()
        with self._lock:
            memo = self._memos.get(track_id)
            if memo is None:
                return
            memo.head_yaw_deg = yaw_deg
            memo.head_pitch_deg = pitch_deg
            memo.head_roll_deg = roll_deg
            memo.is_facing_camera = bool(is_facing_camera)
            memo.head_pose_last_update = now

    def set_gestures(
        self,
        track_id: int,
        gestures: List[str],
        now: Optional[float] = None,
    ) -> None:
        now = now if now is not None else time.monotonic()
        with self._lock:
            memo = self._memos.get(track_id)
            if memo is None:
                return
            memo.active_gestures = list(gestures)
            memo.gestures_last_update = now

    def set_emotion(
        self,
        track_id: int,
        emotion: Optional[str],
        now: Optional[float] = None,
    ) -> None:
        now = now if now is not None else time.monotonic()
        with self._lock:
            memo = self._memos.get(track_id)
            if memo is None:
                return
            memo.emotion = emotion
            memo.emotion_last_update = now

    def prune(self, alive_tids: Iterable[int], now: float) -> List[int]:
        """Forget tracks we haven't seen in ``TRACK_FORGET_AFTER_S``.

        Returns the list of track_ids that were just evicted -- useful
        for emitting ``person_left`` events.
        """
        alive = set(alive_tids)
        evicted: List[int] = []
        with self._lock:
            for tid in list(self._memos):
                memo = self._memos[tid]
                if tid in alive:
                    continue
                if (now - memo.last_seen) > TRACK_FORGET_AFTER_S:
                    evicted.append(tid)
                    del self._memos[tid]
        return evicted

    # ---- readers ------------------------------------------------------
    def get(self, track_id: int) -> Optional[_TrackMemo]:
        with self._lock:
            return self._memos.get(track_id)

    def alive_track_ids(self) -> List[int]:
        with self._lock:
            return list(self._memos.keys())


# ---- Builder --------------------------------------------------------------

YoloTrack = Tuple[int, Bbox]
SpeechSnap = Tuple[Bbox, Optional[str], bool, float]  # (face_bbox, name, is_speaking, prob)


def build_humans(
    *,
    yolo_tracks: List[YoloTrack],
    speech_snaps: List[SpeechSnap],
    frame_w: int,
    frame_h: int,
    registry: HumanRegistry,
    now: Optional[float] = None,
) -> Tuple[List[HumanView], Dict[int, int]]:
    """Combine body tracks + per-face speech info into :class:`HumanView` s.

    Returns ``(humans, face_to_track)`` where ``face_to_track[i]`` is
    the ``track_id`` that face index ``i`` of ``speech_snaps`` was
    attributed to (``-1`` if unattributable; useful for overlays).
    """
    now = now if now is not None else time.monotonic()
    frame_area = max(1.0, float(frame_w * frame_h))

    # Step 1: bring every live body into the registry.
    for tid, _bbox in yolo_tracks:
        registry.touch(tid, now)

    # Step 2: for every face, find the best YOLO body.
    face_to_track: Dict[int, int] = {}
    face_info_by_tid: Dict[int, SpeechSnap] = {}
    for i, (face_bbox, name, is_speaking, prob) in enumerate(speech_snaps):
        best_tid, best_score = -1, 0.0
        for tid, body_bbox in yolo_tracks:
            s = _face_in_body_score(face_bbox, body_bbox)
            if s > best_score:
                best_score = s
                best_tid = tid
        if best_score >= 0.5 and best_tid != -1:
            face_to_track[i] = best_tid
            # If multiple faces land on the same track, keep the one
            # with higher speaking_prob so the dominant one wins.
            existing = face_info_by_tid.get(best_tid)
            if existing is None or prob > existing[3]:
                face_info_by_tid[best_tid] = (face_bbox, name, is_speaking, prob)
            # Remember the name even if cheap-detect tick didn't confirm.
            if name is not None:
                registry.remember_name(best_tid, name)
        else:
            face_to_track[i] = -1

    # Step 3: assemble HumanViews.
    humans: List[HumanView] = []
    for tid, bbox in yolo_tracks:
        memo = registry.get(tid)
        if memo is None:  # shouldn't happen; touch() ran above
            continue
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        proximity = ((x2 - x1) * (y2 - y1)) / frame_area
        direction = _direction_from_bbox(bbox, frame_w)
        distance = _distance_from_proximity(proximity)

        name: Optional[str] = memo.name
        is_speaking = False
        speaking_prob = 0.0
        face_info = face_info_by_tid.get(tid)
        if face_info is not None:
            _fbox, fname, f_speaking, f_prob = face_info
            if fname is not None:
                name = fname
            is_speaking = bool(f_speaking)
            speaking_prob = float(f_prob)

        # Apply head-pose only if it was updated recently.
        head_yaw_deg = head_pitch_deg = head_roll_deg = None
        is_facing = False
        if (now - memo.head_pose_last_update) < HEAD_POSE_TTL_S:
            head_yaw_deg = memo.head_yaw_deg
            head_pitch_deg = memo.head_pitch_deg
            head_roll_deg = memo.head_roll_deg
            is_facing = memo.is_facing_camera

        gestures: List[str] = []
        if (now - memo.gestures_last_update) < GESTURE_TTL_S:
            gestures = list(memo.active_gestures)

        emotion = None
        if memo.emotion and (now - memo.emotion_last_update) < EMOTION_TTL_S:
            emotion = memo.emotion

        humans.append(HumanView(
            track_id=tid,
            bbox=bbox,
            center_px=(cx, cy),
            direction=direction,
            distance_est=distance,
            proximity=proximity,
            name=name,
            is_speaking=is_speaking,
            speaking_prob=speaking_prob,
            head_yaw_deg=head_yaw_deg,
            head_pitch_deg=head_pitch_deg,
            head_roll_deg=head_roll_deg,
            is_facing_camera=is_facing,
            active_gestures=gestures,
            emotion=emotion,
            first_seen=memo.first_seen,
            last_seen=memo.last_seen,
        ))

    return humans, face_to_track
