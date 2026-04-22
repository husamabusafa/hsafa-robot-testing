"""robot_control.py - Stateful control loop that drives Reachy Mini.

Responsibilities:

* Feed camera frames to :class:`CascadeTracker` and read the latest track.
* Run a smoothed P-controller on head yaw/pitch to keep the tracked person
  centered in the image.
* Rotate the body to extend the effective yaw range when the head is at the
  edge of its workspace.
* Overlay one of two animations (idle or talking) on top of the tracking
  pose and drive the antennas.
* Send the combined command to ``reachy.set_target`` every tick.

Designed so a future Hsafa Core client can inject higher-level pose overrides
(look-at targets, gestures) without fighting the voice/animation layer.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from .animation import IdleAnimation, TalkingAnimation, blend_offsets
from .natural_gaze import GazeCommand, MotionProfile, NaturalGaze
from .tracker import CascadeTracker, TIER_COLORS, TIER_NONE, TrackResult

log = logging.getLogger(__name__)


# --- Control tuning --------------------------------------------------------

# P-controller gains (radians of command per unit of normalized error).
KP_YAW = 0.6
KP_PITCH = 0.4
STEP_SCALE = 0.2

# Flip signs if the robot turns the wrong way. Defaults match the working
# ``examples/05_face_follow.py`` (raw, un-mirrored image: +x = right, +y =
# down; +yaw turns left, +pitch looks down).
YAW_SIGN = -1.0
PITCH_SIGN = +1.0

# Hardware workspace of the head (radians).
YAW_LIMIT = math.radians(60)
PITCH_LIMIT = math.radians(30)

# Small tolerance around center where we don't bother correcting.
DEADZONE = 0.03

# EMA smoothing factors in [0, 1]. Higher = snappier, lower = smoother.
ERR_ALPHA = 0.6
CMD_ALPHA = 0.4

# Dropout handling
COAST_S = 0.6             # keep using last-known error for this long on miss
RECENTER_AFTER_S = 1.5    # after this long with no face, decay toward center

# Body rotation takes over when head yaw exceeds BODY_ENGAGE_RAD.
BODY_ENGAGE_RAD = math.radians(12)
BODY_FOLLOW_FRAC = 1.0
BODY_ALPHA = 0.08
BODY_LIMIT = math.radians(90)

# Crossfade time between idle and talking animations.
ANIM_CROSSFADE_S = 0.35


# --- Pose helpers ----------------------------------------------------------

def head_pose(
    roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
) -> np.ndarray:
    """Build a 4x4 head pose matrix from roll/pitch/yaw (radians)."""
    M = np.eye(4)
    M[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    return M


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# --- Snapshot --------------------------------------------------------------

@dataclass
class ControlSnapshot:
    """A read-only view of the controller's last tick for logging/UI."""
    tier: str
    track_id: Optional[int]
    have_face: bool
    err_x: float
    err_y: float
    sent_yaw: float
    sent_pitch: float
    body_yaw: float
    antennas: tuple
    talking: bool


# --- Controller ------------------------------------------------------------

class RobotController:
    """Drives Reachy Mini given a tracker and an optional speech state.

    Parameters
    ----------
    reachy:
        An open ``ReachyMini`` context (we do NOT manage the connection).
    tracker:
        A running :class:`CascadeTracker`.
    is_talking_fn:
        Callable returning ``True`` when the robot is currently speaking.
        Typically ``gemini.is_speaking.is_set``.
    no_body:
        If True, never command body yaw.
    """

    def __init__(
        self,
        reachy,
        tracker: CascadeTracker,
        is_talking_fn=lambda: False,
        *,
        no_body: bool = False,
    ) -> None:
        self.reachy = reachy
        self.tracker = tracker
        self.is_talking_fn = is_talking_fn
        self.no_body = no_body

        # Animations
        self._idle = IdleAnimation()
        self._talking = TalkingAnimation()
        self._anim_blend = 0.0          # 0 = idle, 1 = talking
        self._target_blend = 0.0

        # Natural-gaze motion planner (saccades / micro-saccades /
        # search / idle drift). All overrides are applied ON TOP of
        # the P-controller output; the planner itself doesn't move
        # motors.
        self._natural_gaze = NaturalGaze()
        self._last_locked_id: Optional[int] = None
        self._last_humans_seen_ts = time.time()

        # Face-tracking state
        self._cmd_yaw = 0.0
        self._cmd_pitch = 0.0
        self._sent_yaw = 0.0
        self._sent_pitch = 0.0
        self._err_x_s = 0.0
        self._err_y_s = 0.0
        self._body_yaw = 0.0
        self._last_seen = 0.0
        self._last_det_ts = 0.0
        self._last_tick = time.time()

        # Public snapshot for logging / preview
        self.snapshot = ControlSnapshot(
            tier=TIER_NONE, track_id=None, have_face=False,
            err_x=0.0, err_y=0.0,
            sent_yaw=0.0, sent_pitch=0.0, body_yaw=0.0,
            antennas=(0.0, 0.0), talking=False,
        )

    # ---- NaturalGaze accessors (for main loop / focus events) ----------
    @property
    def natural_gaze(self) -> NaturalGaze:
        return self._natural_gaze

    def notify_target_switched(
        self, new_yaw_rad: float, new_pitch_rad: float,
    ) -> None:
        """Let the gaze planner know the gaze target just switched.

        Pass the approximate yaw/pitch the head will be aimed at next
        so the planner can decide saccade vs. smooth pursuit.
        """
        self._natural_gaze.notify_target_changed(new_yaw_rad, new_pitch_rad)

    def notify_person_lost(self, last_known_yaw_rad: float) -> None:
        self._natural_gaze.notify_person_lost(last_known_yaw_rad=last_known_yaw_rad)

    def notify_voice_unseen(self, guess_yaw_rad: Optional[float] = None) -> None:
        self._natural_gaze.notify_voice_unseen(guess_yaw_rad=guess_yaw_rad)

    def cue_listener_glance(self, other_person_yaw_rad: float) -> None:
        self._natural_gaze.cue_listener_glance(other_person_yaw_rad)

    def mark_humans_seen(self) -> None:
        """Called by the main loop while at least one human is visible."""
        self._last_humans_seen_ts = time.time()

    # ---- per-frame tick -------------------------------------------------
    def tick(self, frame) -> ControlSnapshot:
        """Advance the controller by one frame and command the robot."""
        now = time.time()
        dt = max(1e-3, now - self._last_tick)
        self._last_tick = now

        # ---- 1. Submit frame & read latest detection --------------------
        self.tracker.submit(frame)
        det: Optional[TrackResult] = self.tracker.get()

        have_face = False
        err_x = err_y = 0.0
        current_tier = TIER_NONE
        current_id: Optional[int] = None

        if det is not None and det.timestamp != self._last_det_ts:
            err_x, err_y = det.err_x, det.err_y
            current_tier = det.tier
            current_id = det.track_id
            have_face = True
            self._last_seen = det.timestamp
            self._last_det_ts = det.timestamp
        elif det is not None and (now - det.timestamp) < COAST_S:
            err_x, err_y = det.err_x, det.err_y
            current_tier = det.tier
            current_id = det.track_id
            have_face = True

        # ---- 1b. Ask the natural-gaze planner for its preference -------
        # Planner might want to inject a saccade boost, an idle drift
        # override, a search sweep, or just pass through.
        no_humans_s = max(0.0, now - self._last_humans_seen_ts)
        gcmd: GazeCommand = self._natural_gaze.tick(
            have_target=have_face,
            current_yaw=self._cmd_yaw,
            current_pitch=self._cmd_pitch,
            no_humans_s=no_humans_s,
        )

        # ---- 2. Error smoothing & P-controller --------------------------
        if have_face:
            self._err_x_s = (1 - ERR_ALPHA) * self._err_x_s + ERR_ALPHA * err_x
            self._err_y_s = (1 - ERR_ALPHA) * self._err_y_s + ERR_ALPHA * err_y

        # Track whether the controller is actively moving (for micro-
        # saccade bookkeeping).
        was_moving = (
            abs(self._err_x_s) > DEADZONE or abs(self._err_y_s) > DEADZONE
        )

        # Gain scale from the planner -- saccades multiply KP for a
        # ballistic snap; idle drift uses gain 1.0 because it overrides
        # the target directly.
        gain = max(0.2, gcmd.gain_scale)

        active = have_face or (now - self._last_seen) < COAST_S
        if gcmd.override_yaw is not None or gcmd.override_pitch is not None:
            # Absolute override path (idle drift / search / listener glance)
            # -- ignore P-controller entirely for the overridden axis.
            if gcmd.override_yaw is not None:
                target_yaw = gcmd.override_yaw
                self._cmd_yaw += gain * KP_YAW * (target_yaw - self._cmd_yaw) * STEP_SCALE
            if gcmd.override_pitch is not None:
                target_pitch = gcmd.override_pitch
                self._cmd_pitch += (
                    gain * KP_PITCH * (target_pitch - self._cmd_pitch) * STEP_SCALE
                )
        elif active:
            if abs(self._err_x_s) > DEADZONE:
                self._cmd_yaw += (
                    YAW_SIGN * KP_YAW * self._err_x_s * STEP_SCALE * gain
                )
            if abs(self._err_y_s) > DEADZONE:
                self._cmd_pitch += (
                    PITCH_SIGN * KP_PITCH * self._err_y_s * STEP_SCALE * gain
                )
        elif now - self._last_seen > RECENTER_AFTER_S:
            self._err_x_s *= 0.9
            self._err_y_s *= 0.9
            self._cmd_yaw *= 0.95
            self._cmd_pitch *= 0.95

        # Bookkeeping for the planner so micro-saccades know when to
        # fire.
        if was_moving:
            self._natural_gaze.note_motion_active()
        else:
            self._natural_gaze.note_motion_settled(now)

        self._cmd_yaw = clamp(self._cmd_yaw, -YAW_LIMIT, YAW_LIMIT)
        self._cmd_pitch = clamp(self._cmd_pitch, -PITCH_LIMIT, PITCH_LIMIT)

        self._sent_yaw = (
            (1 - CMD_ALPHA) * self._sent_yaw + CMD_ALPHA * self._cmd_yaw
        )
        self._sent_pitch = (
            (1 - CMD_ALPHA) * self._sent_pitch + CMD_ALPHA * self._cmd_pitch
        )

        # ---- 3. Body yaw ------------------------------------------------
        if self.no_body:
            body_target = 0.0
        elif self._cmd_yaw > BODY_ENGAGE_RAD:
            body_target = (self._cmd_yaw - BODY_ENGAGE_RAD) * BODY_FOLLOW_FRAC
        elif self._cmd_yaw < -BODY_ENGAGE_RAD:
            body_target = (self._cmd_yaw + BODY_ENGAGE_RAD) * BODY_FOLLOW_FRAC
        else:
            body_target = 0.0
        self._body_yaw = (1 - BODY_ALPHA) * self._body_yaw + BODY_ALPHA * body_target
        self._body_yaw = float(clamp(self._body_yaw, -BODY_LIMIT, BODY_LIMIT))

        # ---- 4. Animation overlay ---------------------------------------
        talking = bool(self.is_talking_fn())
        self._target_blend = 1.0 if talking else 0.0
        # Exponential crossfade so transitions never snap.
        blend_step = dt / ANIM_CROSSFADE_S
        if self._anim_blend < self._target_blend:
            self._anim_blend = min(self._target_blend, self._anim_blend + blend_step)
        elif self._anim_blend > self._target_blend:
            self._anim_blend = max(self._target_blend, self._anim_blend - blend_step)

        idle_off = self._idle.offsets(now)
        talk_off = self._talking.offsets(now)
        off = blend_offsets(idle_off, talk_off, self._anim_blend)

        # ---- 5. Compose & send ------------------------------------------
        head_roll = off["roll"]
        head_pitch = self._sent_pitch + off["pitch"] + gcmd.offset_pitch
        head_yaw = self._sent_yaw + off["yaw"] + gcmd.offset_yaw

        # Clamp final commanded angles to stay inside the head workspace.
        head_pitch = clamp(head_pitch, -PITCH_LIMIT, PITCH_LIMIT)
        head_yaw = clamp(head_yaw, -YAW_LIMIT, YAW_LIMIT)

        right_ant, left_ant = off["antennas"]
        antennas = [float(right_ant), float(left_ant)]

        try:
            self.reachy.set_target(
                head=head_pose(roll=head_roll, pitch=head_pitch, yaw=head_yaw),
                body_yaw=self._body_yaw,
                antennas=antennas,
            )
        except Exception as e:
            # Don't let a transient set_target hiccup kill the loop.
            log.warning("set_target failed: %s", e)

        # ---- 6. Publish snapshot ----------------------------------------
        self.snapshot = ControlSnapshot(
            tier=current_tier,
            track_id=current_id,
            have_face=have_face,
            err_x=self._err_x_s,
            err_y=self._err_y_s,
            sent_yaw=self._sent_yaw,
            sent_pitch=self._sent_pitch,
            body_yaw=self._body_yaw,
            antennas=tuple(antennas),
            talking=talking,
        )
        return self.snapshot
