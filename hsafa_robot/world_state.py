"""world_state.py - Single source of truth for what the robot perceives.

Every sense (YOLO tracker, face recognizer, lip-motion, VAD, head-
pose, gestures, ...) writes into one shared :class:`WorldState`.
Every brain (focus policy, Gemini context, future Hsafa bridge)
reads from it under a lock.

Design invariants (see ``docs/architecture.md`` and
``docs/gaze-modes.md``):

* **One writer per field.** Two modules never own the same field.
  Vision owns ``humans[].bbox``; lip-motion owns ``is_speaking``;
  head-pose owns ``head_yaw``; focus owns ``current_target``.
* **Read-under-a-lock, copy out.** Readers call :meth:`snapshot` to
  get an immutable-ish copy. They must not mutate the returned
  object; treat it as a message.
* **No direct method calls between writers.** They don't even import
  each other. They all import this module.
* **Room for growth.** New senses add fields; don't reshape existing
  ones. ``objects``, ``robot``, and ``environment`` are already here
  as empty dicts so we can fill them in without a migration.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple


Bbox = Tuple[int, int, int, int]


# ---- Per-person view ------------------------------------------------------

@dataclass
class HumanView:
    """Everything we currently believe about one visible person.

    Fields populated over time by different writers -- a brand-new
    detection only has ``track_id`` and ``bbox``; face / speech /
    head-pose come in later.
    """
    track_id: int
    bbox: Bbox
    center_px: Tuple[int, int]

    # Positional class from frame width: "left" / "center" / "right"
    direction: str = "center"

    # Distance proxy, bucketized from bbox area: "near" / "mid" / "far".
    distance_est: str = "mid"

    # Fractional bbox area in [0, 1] (bbox_area / frame_area). This is
    # the actual number the GazePolicy uses for the proximity term.
    proximity: float = 0.0

    # Identity
    name: Optional[str] = None          # resolved by face recognizer
    identity_id: Optional[str] = None   # stable IdentityGraph UUID

    # Speech state
    is_speaking: bool = False           # lip-motion (gated on VAD if fused)
    speaking_prob: float = 0.0          # continuous [0, 1] signal

    # Head orientation (populated by head-pose module)
    head_yaw_deg: Optional[float] = None       # + = looking left in frame
    head_pitch_deg: Optional[float] = None
    head_roll_deg: Optional[float] = None
    is_facing_camera: bool = False             # |yaw| < ~20 deg, |pitch| < ~20

    # Gestures observed in the last ~1 s (e.g. ["wave", "point"]).
    active_gestures: List[str] = field(default_factory=list)

    # Emotion hint (populated by emotion module, optional).
    # E.g. "happy", "neutral", "surprised", "sad", ...
    emotion: Optional[str] = None

    # Timestamps (monotonic seconds)
    first_seen: float = 0.0
    last_seen: float = 0.0

    # ---- helpers ----
    def age_s(self, now: Optional[float] = None) -> float:
        now = now if now is not None else time.monotonic()
        return max(0.0, now - self.first_seen)

    def seen_recency_s(self, now: Optional[float] = None) -> float:
        now = now if now is not None else time.monotonic()
        return max(0.0, now - self.last_seen)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "name": self.name,
            "direction": self.direction,
            "distance_est": self.distance_est,
            "is_speaking": self.is_speaking,
            "speaking_prob": round(self.speaking_prob, 3),
            "head_yaw_deg": self.head_yaw_deg,
            "is_facing_camera": self.is_facing_camera,
            "active_gestures": list(self.active_gestures),
            "emotion": self.emotion,
            "first_seen_s_ago": round(time.monotonic() - self.first_seen, 2),
        }


# ---- Robot self-view ------------------------------------------------------

@dataclass
class RobotView:
    """What the robot knows about itself."""
    head_yaw_deg: float = 0.0
    head_pitch_deg: float = 0.0
    head_roll_deg: float = 0.0
    body_yaw_deg: float = 0.0
    is_speaking: bool = False           # robot's mouth is talking
    # Current focus decision. Written by FocusManager.
    current_target_track_id: Optional[int] = None
    current_target_name: Optional[str] = None
    gaze_mode: str = "normal"           # "normal" / "person"
    gaze_state: str = "idle"            # "engaged" / "scanning" / "idle" / "searching"


# ---- Environment (reserved) -----------------------------------------------

@dataclass
class EnvView:
    """Room-level signals. Filled in by DOA / ambient modules later."""
    audio_speech_active: bool = False   # Silero VAD fires
    doa_azimuth_deg: Optional[float] = None   # sound direction (2+ mics)
    noise_level: float = 0.0            # linear RMS of room audio
    lighting: str = "normal"
    # Most recent voice-identification result; ``None`` means either
    # no one has spoken yet or the last speaker wasn't recognized.
    last_heard_voice_name: Optional[str] = None
    last_heard_voice_similarity: float = 0.0
    last_heard_voice_ts: float = 0.0


# ---- The whole state ------------------------------------------------------

@dataclass
class WorldState:
    """One snapshot of what the robot perceives, owned by one holder."""
    humans: List[HumanView] = field(default_factory=list)
    objects: List[Dict[str, Any]] = field(default_factory=list)   # reserved
    robot: RobotView = field(default_factory=RobotView)
    env: EnvView = field(default_factory=EnvView)
    last_update: float = 0.0

    # ---- helpers ----
    def find_by_track(self, track_id: int) -> Optional[HumanView]:
        for h in self.humans:
            if h.track_id == track_id:
                return h
        return None

    def find_by_name(self, name: str) -> Optional[HumanView]:
        if not name:
            return None
        for h in self.humans:
            if h.name == name:
                return h
        return None

    def active_speaker(self) -> Optional[HumanView]:
        best: Optional[HumanView] = None
        best_p = 0.0
        for h in self.humans:
            if h.is_speaking and h.speaking_prob > best_p:
                best_p = h.speaking_prob
                best = h
        if best is not None:
            return best
        # Fallback: highest speaking_prob even if below threshold.
        for h in self.humans:
            if h.speaking_prob > best_p:
                best_p = h.speaking_prob
                best = h
        return best

    def brief_text(self) -> str:
        """Compact one-line summary suitable for Gemini context injection.

        Example:
            humans: husam (left, speaking, facing), unknown (right);
            target: husam; state: engaged
        """
        if not self.humans:
            crowd = "nobody"
        else:
            parts = []
            for h in self.humans:
                bits = [h.name or "unknown", h.direction]
                if h.is_speaking:
                    bits.append("speaking")
                if h.is_facing_camera:
                    bits.append("facing")
                if h.active_gestures:
                    bits.append("+".join(h.active_gestures))
                parts.append(" ".join(bits))
            crowd = ", ".join(parts)
        tgt = self.robot.current_target_name or (
            f"#{self.robot.current_target_track_id}"
            if self.robot.current_target_track_id is not None else "(none)"
        )
        return (
            f"humans: {crowd}; target: {tgt}; "
            f"state: {self.robot.gaze_state}; mode: {self.robot.gaze_mode}"
        )


# ---- Thread-safe holder ---------------------------------------------------

class WorldStateHolder:
    """Thread-safe container for the single :class:`WorldState` instance.

    Readers call :meth:`snapshot` to get a copy they can use without a
    lock; writers use :meth:`update` with a callback that mutates the
    state under the lock. Small surface, tight invariant.
    """

    def __init__(self, initial: Optional[WorldState] = None) -> None:
        self._lock = threading.RLock()
        self._state = initial or WorldState()

    def snapshot(self) -> WorldState:
        """Return a shallow copy of the current state.

        ``humans`` is re-listed with fresh :class:`HumanView` copies so
        readers can mutate their own copy without corrupting the
        canonical one. Nested ``RobotView`` / ``EnvView`` are
        re-created via :func:`dataclasses.replace`.
        """
        with self._lock:
            s = self._state
            return WorldState(
                humans=[replace(h) for h in s.humans],
                objects=list(s.objects),
                robot=replace(s.robot),
                env=replace(s.env),
                last_update=s.last_update,
            )

    def update(self, mutator):
        """Run ``mutator(state)`` under the lock and stamp ``last_update``.

        ``mutator`` receives the live state (NOT a copy) and is free to
        mutate it in place.
        """
        with self._lock:
            mutator(self._state)
            self._state.last_update = time.monotonic()

    # ---- targeted writers (the common patterns; avoid ad-hoc locking) ---

    def replace_humans(self, humans: List[HumanView]) -> None:
        """Replace the whole humans list atomically."""
        with self._lock:
            self._state.humans = list(humans)
            self._state.last_update = time.monotonic()

    def set_human_speech(
        self,
        track_id: int,
        *,
        is_speaking: bool,
        speaking_prob: float,
    ) -> None:
        with self._lock:
            for h in self._state.humans:
                if h.track_id == track_id:
                    h.is_speaking = bool(is_speaking)
                    h.speaking_prob = float(speaking_prob)
                    break

    def set_human_name(self, track_id: int, name: Optional[str]) -> None:
        with self._lock:
            for h in self._state.humans:
                if h.track_id == track_id:
                    h.name = name
                    break

    def set_human_head_pose(
        self,
        track_id: int,
        *,
        yaw_deg: Optional[float],
        pitch_deg: Optional[float],
        roll_deg: Optional[float],
        is_facing_camera: bool,
    ) -> None:
        with self._lock:
            for h in self._state.humans:
                if h.track_id == track_id:
                    h.head_yaw_deg = yaw_deg
                    h.head_pitch_deg = pitch_deg
                    h.head_roll_deg = roll_deg
                    h.is_facing_camera = bool(is_facing_camera)
                    break

    def set_human_gestures(self, track_id: int, gestures: List[str]) -> None:
        with self._lock:
            for h in self._state.humans:
                if h.track_id == track_id:
                    h.active_gestures = list(gestures)
                    break

    def set_audio_speech_active(self, active: bool) -> None:
        with self._lock:
            self._state.env.audio_speech_active = bool(active)
            self._state.last_update = time.monotonic()

    def set_last_heard_voice(
        self,
        name: Optional[str],
        similarity: float,
        *,
        now: Optional[float] = None,
    ) -> None:
        """Record the most recent speaker-ID result on ``EnvView``."""
        now = now if now is not None else time.monotonic()
        with self._lock:
            self._state.env.last_heard_voice_name = name
            self._state.env.last_heard_voice_similarity = float(similarity)
            self._state.env.last_heard_voice_ts = now
            self._state.last_update = now

    def set_robot_target(
        self,
        track_id: Optional[int],
        name: Optional[str],
        *,
        gaze_mode: Optional[str] = None,
        gaze_state: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._state.robot.current_target_track_id = track_id
            self._state.robot.current_target_name = name
            if gaze_mode is not None:
                self._state.robot.gaze_mode = gaze_mode
            if gaze_state is not None:
                self._state.robot.gaze_state = gaze_state
            self._state.last_update = time.monotonic()

    def set_robot_pose(
        self,
        *,
        head_yaw_deg: float,
        head_pitch_deg: float,
        head_roll_deg: float,
        body_yaw_deg: float,
        is_speaking: bool,
    ) -> None:
        with self._lock:
            r = self._state.robot
            r.head_yaw_deg = float(head_yaw_deg)
            r.head_pitch_deg = float(head_pitch_deg)
            r.head_roll_deg = float(head_roll_deg)
            r.body_yaw_deg = float(body_yaw_deg)
            r.is_speaking = bool(is_speaking)
            self._state.last_update = time.monotonic()
