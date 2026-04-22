"""lip_motion.py - Visual speaker detection via mouth-region motion.

Background thread that:

1. Pulls the latest camera frame a few times a second.
2. Asks :class:`FaceRecognizer` for the current face bounding boxes
   (MTCNN only, no embedding -- cheap).
3. Crops the mouth region from each bbox, resizes to a small grayscale
   patch, and scores frame-to-frame L1 difference. Rolling max over a
   ~1.5 s window is the per-face "is this person's mouth moving"
   signal.
4. Matches detections across frames by bbox IoU so each face gets a
   stable ``track_id``.
5. Periodically (every few seconds) runs the full embedding + DB
   lookup so each track picks up the person's name without paying the
   embedding cost on every tick.

A snapshot of the state is read on demand by the ``who_is_speaking``
Gemini tool. The rolling window (rather than "motion right now") is
critical because by the time the user finishes asking "who is
talking?", they've stopped -- but they were moving their mouth a
moment ago, which is the right answer.

Limitations (honest):

* Requires the speaker to be visible. Off-camera voices cannot be
  attributed (voice enrollment, when added later, will cover this).
* Static-pose false positives (chewing, smiling) are possible. Gating
  on "the mic is currently hearing speech" would reduce these; left as
  a follow-up because the current Gemini session doesn't expose a
  per-chunk speech flag to the tracker.
* MTCNN misses heavily-occluded / very small / back-of-head faces.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .face_recognizer import FaceRecognizer

log = logging.getLogger(__name__)


Bbox = Tuple[int, int, int, int]  # (x1, y1, x2, y2), pixel coords
FrameGetter = Callable[[], Optional[np.ndarray]]


# ---- Tunables -------------------------------------------------------------

# How often the background loop wakes up. 5 Hz gives us enough temporal
# resolution for lip motion (human speech syllables are ~5-8 Hz) while
# leaving plenty of headroom for MTCNN + bookkeeping.
DEFAULT_POLL_HZ = 5.0

# Full recognition (embedding + DB lookup) period. Only needed to keep
# track_id -> name fresh; names rarely change once assigned.
DEFAULT_IDENTIFY_PERIOD_S = 2.5

# How long motion history we keep for the "recently speaking" signal.
# Long enough to survive the latency between the user finishing their
# sentence and Gemini asking for the tool call.
MOTION_WINDOW_S = 1.5

# Track lifetimes.
TRACK_MAX_AGE_S = 1.0   # prune tracks we haven't seen this long
MIN_IOU_FOR_MATCH = 0.2

# Mouth-patch size (grayscale). Small and fixed so the L1 diff across
# frames is comparable regardless of how close the person is.
MOUTH_PATCH_W = 40
MOUTH_PATCH_H = 20

# Mouth region within the face bbox, as fractions (y_top, y_bot, x_left,
# x_right). Tuned for frontal MTCNN crops: the mouth sits in the lower
# third, horizontally centered.
MOUTH_REGION = (0.62, 0.92, 0.22, 0.78)

# Speaking threshold: min average L1 mouth-patch diff per pixel for a
# track to count as "speaking". Measured empirically on the current
# hardware at 640x480 -- quiet/still mouths sit ~1-3, speaking ~8-25.
SPEAKING_MOTION_THRESHOLD = 6.0


# ---- Data types -----------------------------------------------------------

@dataclass
class SpeakerCandidate:
    """Snapshot of one face's state, served to the Gemini tool layer."""
    track_id: int
    name: Optional[str]            # None = unknown (or not yet identified)
    position: str                  # "left" | "center" | "right"
    bbox: Bbox
    motion_score: float            # peak motion in the recent window
    is_speaking: bool              # motion_score >= threshold
    last_seen_age_s: float         # 0 when detected this tick
    frame_w: int
    frame_h: int

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "name": self.name or "unknown",
            "position": self.position,
            "motion": round(self.motion_score, 2),
            "is_speaking": self.is_speaking,
            "last_seen_age_s": round(self.last_seen_age_s, 2),
        }


@dataclass
class _Track:
    """Internal per-face state carried across ticks."""
    track_id: int
    bbox: Bbox
    name: Optional[str] = None
    last_seen_ts: float = 0.0
    last_patch: Optional[np.ndarray] = None  # last grayscale mouth patch
    motion_history: Deque[Tuple[float, float]] = field(default_factory=deque)
    frame_w: int = 0
    frame_h: int = 0

    def record_motion(self, ts: float, score: float) -> None:
        self.motion_history.append((ts, score))
        cutoff = ts - MOTION_WINDOW_S
        while self.motion_history and self.motion_history[0][0] < cutoff:
            self.motion_history.popleft()

    def peak_motion(self, now: float) -> float:
        cutoff = now - MOTION_WINDOW_S
        peak = 0.0
        for ts, score in self.motion_history:
            if ts < cutoff:
                continue
            if score > peak:
                peak = score
        return peak


# ---- Geometry helpers -----------------------------------------------------

def _bbox_iou(a: Bbox, b: Bbox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_position(bbox: Bbox, frame_w: int) -> str:
    x1, _y1, x2, _y2 = bbox
    if frame_w <= 0:
        return "center"
    cx = 0.5 * (x1 + x2) / float(frame_w)
    if cx < 1.0 / 3.0:
        return "left"
    if cx > 2.0 / 3.0:
        return "right"
    return "center"


def _mouth_patch(frame_bgr: np.ndarray, bbox: Bbox) -> Optional[np.ndarray]:
    """Crop the mouth region from ``frame_bgr`` and return a fixed-size gray patch."""
    x1, y1, x2, y2 = bbox
    fh, fw = frame_bgr.shape[:2]
    x1 = max(0, min(fw - 1, x1))
    x2 = max(0, min(fw, x2))
    y1 = max(0, min(fh - 1, y1))
    y2 = max(0, min(fh, y2))
    bw, bh = x2 - x1, y2 - y1
    if bw < 20 or bh < 20:
        return None
    ry1, ry2, rx1, rx2 = MOUTH_REGION
    mx1 = int(x1 + rx1 * bw)
    mx2 = int(x1 + rx2 * bw)
    my1 = int(y1 + ry1 * bh)
    my2 = int(y1 + ry2 * bh)
    if mx2 <= mx1 or my2 <= my1:
        return None
    crop = frame_bgr[my1:my2, mx1:mx2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (MOUTH_PATCH_W, MOUTH_PATCH_H), interpolation=cv2.INTER_AREA)


def _patch_motion(prev: np.ndarray, curr: np.ndarray) -> float:
    """Mean absolute pixel difference between two mouth patches."""
    # ``cv2.absdiff`` is faster than numpy for uint8 and saturates instead
    # of rolling over.
    diff = cv2.absdiff(prev, curr)
    return float(diff.mean())


# ---- The tracker ----------------------------------------------------------

class LipMotionTracker(threading.Thread):
    """Background thread exposing a :class:`SpeakerCandidate` list."""

    def __init__(
        self,
        recognizer: FaceRecognizer,
        get_frame: FrameGetter,
        *,
        poll_hz: float = DEFAULT_POLL_HZ,
        identify_period_s: float = DEFAULT_IDENTIFY_PERIOD_S,
        name: str = "lip-motion",
    ) -> None:
        super().__init__(daemon=True, name=name)
        self._recognizer = recognizer
        self._get_frame = get_frame
        self._period = 1.0 / max(poll_hz, 0.5)
        self._identify_period_s = identify_period_s
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._tracks: Dict[int, _Track] = {}
        self._next_track_id = 1
        self._last_identify_ts = 0.0
        self._frame_w = 0
        self._frame_h = 0

    # ---- lifecycle -----------------------------------------------------
    def stop(self) -> None:
        # NOTE: cannot name this attribute ``_stop`` -- that shadows
        # ``threading.Thread._stop``, which the stdlib calls internally
        # during ``join`` and produces a very confusing
        # 'Event object is not callable' crash on shutdown.
        self._stop_event.set()

    def run(self) -> None:
        log.info("LipMotionTracker: started (period=%.2fs)", self._period)
        try:
            while not self._stop_event.is_set():
                tick_start = time.time()
                try:
                    self._tick(tick_start)
                except Exception as e:  # pragma: no cover - defensive
                    log.exception("LipMotionTracker tick error: %s", e)
                # Pace ourselves.
                remaining = self._period - (time.time() - tick_start)
                if remaining > 0:
                    self._stop_event.wait(remaining)
        finally:
            log.info("LipMotionTracker: stopped")

    # ---- one tick ------------------------------------------------------
    def _tick(self, now: float) -> None:
        frame = self._get_frame()
        if frame is None:
            return

        h, w = frame.shape[:2]
        self._frame_w, self._frame_h = w, h

        # Decide: cheap detect-only, or full identify-with-names?
        do_identify = (now - self._last_identify_ts) >= self._identify_period_s

        if do_identify:
            matches = self._recognizer.identify_all_in_frame(frame)
            self._last_identify_ts = now
            observations = [
                (m.bbox, m.name) for m in matches
            ]
        else:
            bboxes = self._recognizer.detect_faces(frame)
            observations = [(b, None) for b in bboxes]

        with self._lock:
            self._associate_and_update(observations, frame, now, w, h, do_identify)

    def _associate_and_update(
        self,
        observations: List[Tuple[Bbox, Optional[str]]],
        frame: np.ndarray,
        now: float,
        frame_w: int,
        frame_h: int,
        from_identify: bool,
    ) -> None:
        # Greedy IoU matching. For a handful of faces this is fine and
        # avoids a Hungarian-solver dependency.
        assigned_tracks: set[int] = set()
        assigned_obs: set[int] = set()

        track_items = list(self._tracks.items())
        for obs_idx, (obs_bbox, obs_name) in enumerate(observations):
            best_tid, best_iou = -1, 0.0
            for tid, track in track_items:
                if tid in assigned_tracks:
                    continue
                iou = _bbox_iou(track.bbox, obs_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid != -1 and best_iou >= MIN_IOU_FOR_MATCH:
                self._update_track(
                    self._tracks[best_tid], obs_bbox, obs_name, frame,
                    now, frame_w, frame_h, from_identify,
                )
                assigned_tracks.add(best_tid)
                assigned_obs.add(obs_idx)

        # New tracks for unmatched observations.
        for obs_idx, (obs_bbox, obs_name) in enumerate(observations):
            if obs_idx in assigned_obs:
                continue
            tid = self._next_track_id
            self._next_track_id += 1
            track = _Track(
                track_id=tid,
                bbox=obs_bbox,
                name=obs_name,
                last_seen_ts=now,
                last_patch=_mouth_patch(frame, obs_bbox),
                frame_w=frame_w,
                frame_h=frame_h,
            )
            self._tracks[tid] = track

        # Prune stale tracks.
        stale = [
            tid for tid, t in self._tracks.items()
            if (now - t.last_seen_ts) > TRACK_MAX_AGE_S
        ]
        for tid in stale:
            del self._tracks[tid]

    def _update_track(
        self,
        track: _Track,
        bbox: Bbox,
        name: Optional[str],
        frame: np.ndarray,
        now: float,
        frame_w: int,
        frame_h: int,
        from_identify: bool,
    ) -> None:
        track.bbox = bbox
        track.last_seen_ts = now
        track.frame_w = frame_w
        track.frame_h = frame_h

        # Only overwrite the name when we actually ran identification --
        # a cheap detect-only tick has no name info.
        if from_identify:
            track.name = name

        patch = _mouth_patch(frame, bbox)
        if patch is not None:
            if track.last_patch is not None and track.last_patch.shape == patch.shape:
                motion = _patch_motion(track.last_patch, patch)
                track.record_motion(now, motion)
            track.last_patch = patch

    # ---- read-only snapshot -------------------------------------------
    def snapshot(self) -> List[SpeakerCandidate]:
        with self._lock:
            now = time.time()
            out: List[SpeakerCandidate] = []
            for track in self._tracks.values():
                peak = track.peak_motion(now)
                out.append(
                    SpeakerCandidate(
                        track_id=track.track_id,
                        name=track.name,
                        position=_bbox_position(track.bbox, track.frame_w),
                        bbox=track.bbox,
                        motion_score=peak,
                        is_speaking=peak >= SPEAKING_MOTION_THRESHOLD,
                        last_seen_age_s=now - track.last_seen_ts,
                        frame_w=track.frame_w,
                        frame_h=track.frame_h,
                    )
                )
            # Highest motion first so ``who_is_speaking`` is a simple [0].
            out.sort(key=lambda c: c.motion_score, reverse=True)
            return out

    def current_speaker(self) -> Optional[SpeakerCandidate]:
        """Convenience: highest-motion candidate iff it crosses the threshold."""
        snap = self.snapshot()
        if not snap:
            return None
        top = snap[0]
        return top if top.is_speaking else None
