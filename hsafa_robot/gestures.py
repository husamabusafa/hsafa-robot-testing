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
