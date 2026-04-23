"""gestures.py - Hand gesture recognition via MediaPipe Hands.

Detects a small vocabulary of humanoid-useful gestures:

* ``wave``         - open palm moving side-to-side above shoulders
* ``point``        - index extended, other fingers curled
* ``thumbs_up``    - thumb extended, other fingers curled
* ``open_palm``    - all five fingers extended (stop / "wait" cue)
* ``fist``         - all fingers curled

Each detection is:

1. Stamped onto the nearest YOLO body track via
   :class:`HumanRegistry.set_gestures`, so the gaze policy (and
   the Gemini system prompt) can see it on the :class:`HumanView`.
2. Emitted as a ``gesture_detected`` event on the :class:`EventBus`
   so other subscribers (say, a future Hsafa bridge) can react.

When a ``point`` is detected we also compute which visible body
the pointing finger is aimed at (by extending the
``INDEX_PIP -> INDEX_TIP`` vector and finding the first body bbox
the ray enters) and expose it as :meth:`GestureTracker.get_point_hint`.
That hint is the key signal for disambiguating
``enroll_face("this is my friend Ahmad")`` when multiple people
are visible.

Optional dependency: ``mediapipe``. When missing, :attr:`enabled`
stays False and the tracker thread never starts -- the rest of the
robot keeps working unaffected.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from .events import EVT_GESTURE_DETECTED, EventBus
from .perception import HumanRegistry, _face_in_body_score

log = logging.getLogger(__name__)


FrameGetter = Callable[[], Optional[np.ndarray]]
Bbox = Tuple[int, int, int, int]
YoloTrackSource = Callable[[], List[Tuple[int, Bbox]]]


# ---- Tunables -------------------------------------------------------------

POLL_HZ = 8.0                       # ~8 Hz -- enough for wave detection
WAVE_WINDOW_S = 1.5                 # rolling window for side-to-side motion
WAVE_MIN_SWINGS = 3                 # direction changes in window -> wave
WAVE_MIN_AMPLITUDE_FRAC = 0.04      # peak-to-peak x as fraction of frame width
GESTURE_REFIRE_COOLDOWN_S = 1.5     # re-emit same gesture at most every this


# Hand-landmark indices (MediaPipe Hands)
LM_WRIST = 0
LM_THUMB_TIP = 4
LM_INDEX_TIP = 8
LM_INDEX_PIP = 6
LM_MIDDLE_TIP = 12
LM_MIDDLE_PIP = 10
LM_RING_TIP = 16
LM_RING_PIP = 14
LM_PINKY_TIP = 20
LM_PINKY_PIP = 18


@dataclass
class _HandTrack:
    """Short history of a hand's wrist position for wave detection."""
    history: Deque[Tuple[float, float]] = field(default_factory=deque)
    last_wave_emit: float = 0.0
    last_static_emit: Dict[str, float] = field(default_factory=dict)


@dataclass
class PointHint:
    """Most recent pointing gesture and who it targets.

    ``pointer_tid`` is the YOLO body whose hand is doing the point
    (may be ``None`` if we couldn't associate the hand with a body).
    ``pointed_at_tid`` is the body the finger's ray hits first
    (``None`` means it points off-frame or only at the pointer's
    own body). ``pointed_at_bbox`` is cached for callers that need
    to match a face to the target without re-querying YOLO.
    """
    ts: float
    pointer_tid: Optional[int]
    pointed_at_tid: Optional[int]
    pointed_at_bbox: Optional[Bbox]
    tip_px: Tuple[float, float]             # pixel coords
    direction: Tuple[float, float]          # unit vector in pixel frame

    def is_fresh(self, max_age_s: float = 1.0, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.monotonic()
        return (now - self.ts) <= max_age_s


class GestureTracker:
    """Background thread stamping gestures on to body tracks + emitting events."""

    def __init__(
        self,
        *,
        get_frame: FrameGetter,
        yolo_tracks: YoloTrackSource,
        registry: HumanRegistry,
        bus: Optional[EventBus] = None,
        max_hands: int = 4,
        poll_hz: float = POLL_HZ,
    ) -> None:
        self._get_frame = get_frame
        self._yolo_tracks = yolo_tracks
        self._registry = registry
        self._bus = bus
        self._max_hands = int(max_hands)
        self._period = 1.0 / max(poll_hz, 0.5)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.enabled = False
        self._hands_mod = None
        # Keyed by "left"/"right" label as given by MediaPipe; a second
        # hand of the same label in the frame is treated as a new track.
        # Good enough for one-person-at-a-time waving.
        self._tracks: Dict[str, _HandTrack] = {}
        # Most recent pointing hint (published for the enroll-face
        # disambiguator in main.py).
        self._last_point: Optional[PointHint] = None
        self._point_lock = threading.Lock()

    # ---- public accessors ----------------------------------------
    def get_point_hint(self, max_age_s: float = 1.0) -> Optional[PointHint]:
        """Return the most recent ``point`` gesture hint if still fresh."""
        with self._point_lock:
            hint = self._last_point
        if hint is None or not hint.is_fresh(max_age_s):
            return None
        return hint

    # ---- lifecycle ------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="gestures", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # ---- worker ---------------------------------------------------
    def _load_model(self) -> bool:
        try:
            import mediapipe as mp   # type: ignore
            self._hands_mod = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self._max_hands,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.55,
            )
            self.enabled = True
            log.info("GestureTracker: MediaPipe Hands loaded (max=%d)",
                     self._max_hands)
            return True
        except Exception as e:
            log.warning(
                "GestureTracker: could not load MediaPipe Hands (%s). "
                "Install mediapipe to enable gesture detection.", e,
            )
            return False

    def _run(self) -> None:
        if not self._load_model():
            return
        import cv2
        while not self._stop.is_set():
            tick_start = time.monotonic()
            frame = self._get_frame()
            if frame is None:
                time.sleep(self._period)
                continue
            try:
                self._process_frame(frame, cv2)
            except Exception as e:  # pragma: no cover
                log.warning("GestureTracker tick error: %s", e)
            remaining = self._period - (time.monotonic() - tick_start)
            if remaining > 0:
                self._stop.wait(remaining)

    def _process_frame(self, frame: np.ndarray, cv2) -> None:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands_mod.process(rgb)
        if not result.multi_hand_landmarks:
            # Still decay wave history so old swings don't linger.
            now = time.monotonic()
            for t in self._tracks.values():
                cutoff = now - WAVE_WINDOW_S
                while t.history and t.history[0][0] < cutoff:
                    t.history.popleft()
            return

        now = time.monotonic()
        yolo = self._yolo_tracks() or []
        # Group gestures per body track so we can stamp the registry
        # in one call.
        gestures_by_tid: Dict[int, List[str]] = {}

        handedness = result.multi_handedness or []
        for i, hand_lm in enumerate(result.multi_hand_landmarks):
            label = "right"
            if i < len(handedness):
                cat = handedness[i].classification[0]
                label = (cat.label or "right").lower()

            track = self._tracks.setdefault(label, _HandTrack())
            static_gestures = _classify_static(hand_lm.landmark)

            # Wrist bbox + center to match a YOLO body.
            xs = [lm.x * w for lm in hand_lm.landmark]
            ys = [lm.y * h for lm in hand_lm.landmark]
            hand_bbox = (
                int(min(xs)), int(min(ys)),
                int(max(xs)), int(max(ys)),
            )
            best_tid = _match_bbox_to_body(hand_bbox, yolo)

            # Wave detection: record wrist x, then count direction changes.
            wrist = hand_lm.landmark[LM_WRIST]
            track.history.append((now, wrist.x))
            cutoff = now - WAVE_WINDOW_S
            while track.history and track.history[0][0] < cutoff:
                track.history.popleft()
            dynamic: List[str] = []
            if _is_wave(track.history, w):
                if now - track.last_wave_emit >= GESTURE_REFIRE_COOLDOWN_S:
                    dynamic.append("wave")
                    track.last_wave_emit = now

            combined = list(static_gestures) + dynamic
            if not combined:
                continue

            if best_tid is not None:
                gestures_by_tid.setdefault(best_tid, []).extend(combined)

            # Compute + stash the pointing hint so enroll_face can
            # disambiguate. Runs on every ``point`` detection; the
            # latest one wins.
            if "point" in static_gestures:
                hint = _compute_point_hint(
                    hand_lm.landmark, w, h, yolo,
                    pointer_tid=best_tid, now=now,
                )
                if hint is not None:
                    with self._point_lock:
                        self._last_point = hint

            # Emit events for first-occurrence gestures (per cooldown).
            for g in combined:
                last = track.last_static_emit.get(g, 0.0)
                if now - last < GESTURE_REFIRE_COOLDOWN_S:
                    continue
                track.last_static_emit[g] = now
                if self._bus is not None:
                    self._bus.publish(
                        EVT_GESTURE_DETECTED,
                        source="vision",
                        gesture=g,
                        hand=label,
                        track_id=best_tid,
                    )

        # Stamp into registry.
        for tid, gests in gestures_by_tid.items():
            # Dedup preserving order.
            seen = set()
            uniq = []
            for g in gests:
                if g in seen:
                    continue
                seen.add(g)
                uniq.append(g)
            self._registry.set_gestures(tid, uniq, now=now)


# ---- classification helpers ------------------------------------------

def _finger_extended(tip, pip, wrist) -> bool:
    """True if the finger is 'extended' (tip is further from wrist than pip)."""
    tip_d = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
    pip_d = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
    return tip_d > pip_d * 1.05


def _classify_static(lms) -> List[str]:
    """Return zero-or-more static gesture labels from one hand's landmarks."""
    wrist = lms[LM_WRIST]
    index = _finger_extended(lms[LM_INDEX_TIP], lms[LM_INDEX_PIP], wrist)
    middle = _finger_extended(lms[LM_MIDDLE_TIP], lms[LM_MIDDLE_PIP], wrist)
    ring = _finger_extended(lms[LM_RING_TIP], lms[LM_RING_PIP], wrist)
    pinky = _finger_extended(lms[LM_PINKY_TIP], lms[LM_PINKY_PIP], wrist)

    # Thumb: use horizontal distance since thumbs don't fold parallel
    # to the other fingers. Extended if the tip is above the MCP (y
    # smaller => higher in image, since camera coords).
    thumb_tip = lms[LM_THUMB_TIP]
    thumb_mcp = lms[2]   # THUMB_MCP
    thumb_extended = thumb_tip.y < thumb_mcp.y - 0.03

    labels: List[str] = []
    if index and not middle and not ring and not pinky:
        labels.append("point")
    if index and middle and ring and pinky:
        labels.append("open_palm")
    if thumb_extended and not index and not middle and not ring and not pinky:
        labels.append("thumbs_up")
    if not index and not middle and not ring and not pinky and not thumb_extended:
        labels.append("fist")
    return labels


def _is_wave(history: Deque[Tuple[float, float]], frame_w: int) -> bool:
    """Return True if wrist x in ``history`` shows >=N direction changes + amplitude."""
    if len(history) < 6:
        return False
    xs = [x for _, x in history]
    x_min = min(xs)
    x_max = max(xs)
    amp = x_max - x_min
    if amp < WAVE_MIN_AMPLITUDE_FRAC:
        return False
    swings = 0
    prev_sign = 0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        if abs(dx) < 0.005:
            continue
        sign = 1 if dx > 0 else -1
        if sign != prev_sign and prev_sign != 0:
            swings += 1
        prev_sign = sign
    return swings >= WAVE_MIN_SWINGS


def _match_bbox_to_body(
    hand_bbox: Bbox, yolo: List[Tuple[int, Bbox]],
) -> Optional[int]:
    best_tid, best_score = -1, 0.0
    for tid, body in yolo:
        s = _face_in_body_score(hand_bbox, body)   # same overlap-over-a math
        if s > best_score:
            best_score = s
            best_tid = tid
    if best_score < 0.3 or best_tid == -1:
        return None
    return best_tid


def _ray_bbox_entry_t(
    ox: float, oy: float,
    dx: float, dy: float,
    bbox: Bbox,
) -> Optional[float]:
    """Parametric t at which ray (o + t*d) enters ``bbox`` (t >= 0), or None.

    Standard 2D slab intersection test. If the ray never enters the
    box in the forward direction, returns ``None``. Used to find the
    first body a pointing finger is aimed at.
    """
    x1, y1, x2, y2 = bbox
    t_min = 0.0
    t_max = float("inf")
    for o, d, lo, hi in ((ox, dx, x1, x2), (oy, dy, y1, y2)):
        if abs(d) < 1e-6:
            if o < lo or o > hi:
                return None
            continue
        t1 = (lo - o) / d
        t2 = (hi - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return None
    return t_min if t_min >= 0.0 else None


def _compute_point_hint(
    lms,
    frame_w: int,
    frame_h: int,
    yolo: List[Tuple[int, Bbox]],
    *,
    pointer_tid: Optional[int],
    now: float,
) -> Optional[PointHint]:
    """Trace a ray from index_PIP -> index_TIP and find the first body it hits.

    Bodies associated with ``pointer_tid`` are ignored -- you can
    only point AT someone else, not yourself.
    """
    tip = lms[LM_INDEX_TIP]
    pip = lms[LM_INDEX_PIP]
    tip_x, tip_y = tip.x * frame_w, tip.y * frame_h
    pip_x, pip_y = pip.x * frame_w, pip.y * frame_h
    dx = tip_x - pip_x
    dy = tip_y - pip_y
    norm = math.hypot(dx, dy)
    if norm < 1e-3:
        return None
    dx /= norm
    dy /= norm

    best_t: Optional[float] = None
    best_tid: Optional[int] = None
    best_bbox: Optional[Bbox] = None
    for tid, body in yolo:
        if pointer_tid is not None and tid == pointer_tid:
            continue
        t = _ray_bbox_entry_t(tip_x, tip_y, dx, dy, body)
        if t is None:
            continue
        if best_t is None or t < best_t:
            best_t = t
            best_tid = tid
            best_bbox = body

    return PointHint(
        ts=now,
        pointer_tid=pointer_tid,
        pointed_at_tid=best_tid,
        pointed_at_bbox=best_bbox,
        tip_px=(tip_x, tip_y),
        direction=(dx, dy),
    )
