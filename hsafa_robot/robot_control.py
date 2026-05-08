"""robot_control.py - Simple, natural face-tracking controller for Reachy Mini.

Pipeline (per frame):

1. Submit frame to :class:`CascadeTracker` and read latest detection.
2. Convert normalized image error -> target head angle in world frame
   (the camera moves with the head, so each tick we ask "where is the
   target relative to my CURRENT head pose?" and steer toward it).
3. Slew the head smoothly toward that target with a critically-damped
   first-order filter -> looks natural, no overshoot, no oscillation.
4. Engage body yaw only when the head is near its yaw limit so we
   don't crane the neck.
5. Overlay idle / talking animation and send the combined target.

Design goals:
* Predictable, "natural" motion -- no saccades, no idle drift, no
  search sweeps. Just: see face -> turn toward face -> hold.
* No gyro feedback. Removed by request; can be re-added later.
* Manual override (Gemini ``set_head_angle`` / ``look_straight``)
  takes priority and freezes face-follow until ``clear_manual``.
* Face-follow is **ON by default** so the robot tracks people right
  out of the box.
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
from .tracker import CascadeTracker, TIER_NONE, TrackResult

log = logging.getLogger(__name__)


# --- Geometry / tuning -----------------------------------------------------

# Camera field of view (radians). The default AVFoundation/Reachy lens is
# roughly 60deg horizontal x 45deg vertical. Half-FOV maps a normalized
# image error of 1.0 to "target sits at the edge of the frame".
HALF_HFOV = math.radians(30.0)
HALF_VFOV = math.radians(22.5)

# Sign conventions (see examples/05_face_follow.py):
#   image: +err_x = target on RIGHT, +err_y = target BELOW center
#   robot: +yaw   = head turns LEFT, +pitch = head looks DOWN
# So target right (err_x > 0) needs negative yaw, target below
# (err_y > 0) needs positive pitch.
YAW_SIGN = -1.0
PITCH_SIGN = +1.0

# Head workspace (radians).
YAW_LIMIT = math.radians(60)
PITCH_LIMIT = math.radians(30)

# How fast we slew the head toward the target each frame. ``ALPHA`` is
# the per-tick fraction of the remaining error we close. Lower = smoother
# / less overshoot, higher = snappier.
ALPHA_YAW = 0.07
ALPHA_PITCH = 0.07

# Fraction of the angular offset we *aim for* on each fresh detection.
# Kept well below 1.0 because the camera rides on the head: chasing 100%
# of the measured offset causes the head to overshoot and oscillate.
LEAD_GAIN = 0.70

# When a fresh detection implies a *large* angular jump, boost the slew
# rate for that one tick so the head snaps over, then settles.
SACCADE_JUMP_RAD = math.radians(10)
ALPHA_SACCADE = 0.25

# How long a *temporary* manual gaze (look_at / look_left / look_right /
# look_up / look_down / set_head_angle) holds before automatically
# returning to face-follow. ``disable_face_follow`` is the only call
# that locks the head indefinitely.
AUTO_RESUME_S = 2.5

# Dead-zone on the normalized image error. Larger zone prevents the
# head from hunting back and forth when the face is already centered.
DEADZONE = 0.08

# Exponential-moving-average factor for raw tracker error. Rejects
# keypoint jitter from YOLOv8-Pose so the target doesn't dart around.
ERR_SMOOTH = 0.5

# Hard per-frame rate limit on commanded head motion. Stops integrator
# wind-up from crossing over a moving or suddenly-stopped face.
MAX_YAW_DELTA = math.radians(5.0)
MAX_PITCH_DELTA = math.radians(4.0)

# How long after the last detection we still trust the cached error
# before we declare the face lost.
COAST_S = 0.5

# After this long with no face, slowly recenter so the head doesn't sit
# locked at a stale angle.
RECENTER_AFTER_S = 2.0
RECENTER_DECAY = 0.97  # per-tick multiplicative decay toward 0

# Body yaw kicks in once the head crosses this yaw threshold, then
# follows 1:1 so the head can keep pointing at off-axis targets without
# hitting its limit.
BODY_ENGAGE_RAD = math.radians(15)
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


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# --- Snapshot --------------------------------------------------------------

@dataclass
class ControlSnapshot:
    """Read-only view of the controller's last tick (logging / preview)."""
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
    # Kept for backwards-compat with code that still reads gyro fields;
    # always zero now that gyro is disabled.
    gyro_yaw_rate: float = 0.0
    gyro_pitch_rate: float = 0.0
    gyro_heading: float = 0.0
    gyro_compensated: bool = False


# --- Controller ------------------------------------------------------------

class RobotController:
    """Drives Reachy Mini given a tracker and an optional speech state."""

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
        self._anim_blend = 0.0
        self._target_blend = 0.0

        # Head state (radians, head frame, before animation overlay).
        self._sent_yaw = 0.0
        self._sent_pitch = 0.0
        self._body_yaw = 0.0

        # World-frame target the head is currently slewing toward.
        # Only updated when a fresh detection arrives so stale ticks
        # don't accumulate overshoot.
        self._target_yaw = 0.0
        self._target_pitch = 0.0

        # Last detection for fresh-vs-stale comparison + coasting.
        self._last_err_x = 0.0
        self._last_err_y = 0.0
        self._last_det_ts = 0.0
        self._last_seen = 0.0
        self._last_tick = time.time()

        # Smoothed tracker errors (EMA) to reject keypoint jitter.
        self._smooth_err_x = 0.0
        self._smooth_err_y = 0.0

        # Manual override -- face follow is ON by default. ``_manual_until``
        # is a unix timestamp; if non-zero, we automatically return to
        # face-follow once we pass it. ``disable_face_follow`` sets it to
        # +inf for an indefinite hold.
        self._manual_override = False
        self._manual_yaw = 0.0
        self._manual_pitch = 0.0
        self._manual_body_yaw = 0.0
        self._manual_until = 0.0

        self.snapshot = ControlSnapshot(
            tier=TIER_NONE, track_id=None, have_face=False,
            err_x=0.0, err_y=0.0,
            sent_yaw=0.0, sent_pitch=0.0, body_yaw=0.0,
            antennas=(0.0, 0.0), talking=False,
        )

    # ---- Manual override ------------------------------------------------
    def set_manual_target(
        self,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        body_yaw_deg: float = 0.0,
        *,
        hold_s: Optional[float] = AUTO_RESUME_S,
    ) -> None:
        """Aim head/body at fixed angles, suppressing face follow.

        ``hold_s`` is how long the manual pose is held before face
        follow automatically resumes. The default mimics a human
        glance: look at the thing for ~2.5 s, then return to whoever
        you were talking to. Pass ``hold_s=None`` for an indefinite
        lock (used by ``disable_face_follow``).
        """
        self._manual_override = True
        self._manual_yaw = math.radians(float(yaw_deg))
        self._manual_pitch = math.radians(float(pitch_deg))
        self._manual_body_yaw = math.radians(float(body_yaw_deg))
        if hold_s is None:
            self._manual_until = float("inf")
        else:
            self._manual_until = time.time() + float(hold_s)

    def clear_manual(self) -> None:
        """Return to automatic face-tracking mode."""
        self._manual_override = False
        self._manual_until = 0.0

    @property
    def is_manual(self) -> bool:
        return self._manual_override

    def mark_humans_seen(self) -> None:
        """No-op kept for API compatibility with the previous controller."""

    # ---- Backwards-compat no-op hooks -----------------------------------
    # The old NaturalGaze planner had a bunch of "I lost the person",
    # "I switched targets", "I heard a voice off-camera" entry points.
    # Those behaviors caused the unnatural saccades / drift the user
    # reported, so they're gone. Callers in main.py still invoke them
    # via the event bus, hence the no-op stubs.
    def notify_target_switched(self, *_args, **_kwargs) -> None:  # noqa: D401
        return None

    def notify_person_lost(self, *_args, **_kwargs) -> None:
        return None

    def notify_voice_unseen(self, *_args, **_kwargs) -> None:
        return None

    def cue_listener_glance(self, *_args, **_kwargs) -> None:
        return None

    # ---- Per-frame tick -------------------------------------------------
    def tick(self, frame) -> ControlSnapshot:
        """Advance the controller by one frame and command the robot."""
        now = time.time()
        dt = max(1e-3, now - self._last_tick)
        self._last_tick = now

        # ---- 1. Latest detection ---------------------------------------
        self.tracker.submit(frame)
        det: Optional[TrackResult] = self.tracker.get()

        have_face = False
        err_x = err_y = 0.0
        tier = TIER_NONE
        tid: Optional[int] = None
        fresh = False

        if det is not None:
            if det.timestamp != self._last_det_ts:
                # New measurement from the tracker -- one we haven't
                # acted on yet.
                fresh = True
                self._last_err_x = det.err_x
                self._last_err_y = det.err_y
                self._last_det_ts = det.timestamp
                self._last_seen = det.timestamp
            if (now - det.timestamp) < COAST_S:
                err_x = self._last_err_x
                err_y = self._last_err_y
                tier = det.tier
                tid = det.track_id
                have_face = True

        # ---- 2. Decide world-frame target ------------------------------
        # Only re-aim on a *fresh* detection. Between detections we keep
        # slewing toward the previously-computed target so stale-error
        # ticks can't push the head past it (this is what was causing
        # the head to swing wildly back and forth).

        # Auto-resume: temporary manual gazes (look_at, look_left, ...)
        # release themselves after AUTO_RESUME_S so the robot naturally
        # returns its eyes to whoever it's talking to, like a human.
        if self._manual_override and now >= self._manual_until:
            self._manual_override = False

        # Saccade detection: a *big* implied jump on a fresh detection
        # gets a one-shot speed boost so the head snaps over quickly,
        # then settles smoothly. This is what makes new-target switches
        # look responsive.
        alpha_yaw = ALPHA_YAW
        alpha_pitch = ALPHA_PITCH

        if self._manual_override:
            self._target_yaw = self._manual_yaw
            self._target_pitch = self._manual_pitch
            target_body = self._manual_body_yaw
            # Snappy slew toward a freshly-set manual target too.
            if abs(self._manual_yaw - self._sent_yaw) > SACCADE_JUMP_RAD:
                alpha_yaw = ALPHA_SACCADE
            if abs(self._manual_pitch - self._sent_pitch) > SACCADE_JUMP_RAD:
                alpha_pitch = ALPHA_SACCADE
        elif fresh and have_face:
            # Smooth raw tracker error to reject keypoint / detection jitter.
            self._smooth_err_x += ERR_SMOOTH * (err_x - self._smooth_err_x)
            self._smooth_err_y += ERR_SMOOTH * (err_y - self._smooth_err_y)

            sx = self._smooth_err_x
            sy = self._smooth_err_y
            ex = sx if abs(sx) > DEADZONE else 0.0
            ey = sy if abs(sy) > DEADZONE else 0.0

            # Camera is on the head, so err already includes the robot's
            # own motion. ``LEAD_GAIN < 1`` keeps us slightly under-shooting
            # the measured offset so we don't blow past the target while
            # the head is still moving.
            new_target_yaw = _clamp(
                self._sent_yaw + YAW_SIGN * ex * HALF_HFOV * LEAD_GAIN,
                -YAW_LIMIT, YAW_LIMIT,
            )
            new_target_pitch = _clamp(
                self._sent_pitch + PITCH_SIGN * ey * HALF_VFOV * LEAD_GAIN,
                -PITCH_LIMIT, PITCH_LIMIT,
            )
            # If the new target is far from where we are right now,
            # treat this as a saccade and slew faster on this tick.
            if abs(new_target_yaw - self._sent_yaw) > SACCADE_JUMP_RAD:
                alpha_yaw = ALPHA_SACCADE
            if abs(new_target_pitch - self._sent_pitch) > SACCADE_JUMP_RAD:
                alpha_pitch = ALPHA_SACCADE
            self._target_yaw = new_target_yaw
            self._target_pitch = new_target_pitch
            target_body = self._compute_body_target(self._target_yaw)
        elif have_face:
            # Stale tick: keep slewing toward the target we already had.
            target_body = self._compute_body_target(self._target_yaw)
        else:
            # No face: hold for a grace period, then drift back to centre.
            if (now - self._last_seen) > RECENTER_AFTER_S:
                self._target_yaw *= RECENTER_DECAY
                self._target_pitch *= RECENTER_DECAY
            target_body = self._compute_body_target(self._target_yaw)

        # ---- 3. Smooth slew -------------------------------------------
        # Low-pass step toward the target, then rate-limit the delta so
        # integrator wind-up can't blow past a moving or stopped face.
        prev_yaw = self._sent_yaw
        prev_pitch = self._sent_pitch

        self._sent_yaw += alpha_yaw * (self._target_yaw - self._sent_yaw)
        self._sent_pitch += alpha_pitch * (self._target_pitch - self._sent_pitch)

        dy = self._sent_yaw - prev_yaw
        dp = self._sent_pitch - prev_pitch
        self._sent_yaw = prev_yaw + _clamp(dy, -MAX_YAW_DELTA, MAX_YAW_DELTA)
        self._sent_pitch = prev_pitch + _clamp(dp, -MAX_PITCH_DELTA, MAX_PITCH_DELTA)

        self._sent_yaw = _clamp(self._sent_yaw, -YAW_LIMIT, YAW_LIMIT)
        self._sent_pitch = _clamp(self._sent_pitch, -PITCH_LIMIT, PITCH_LIMIT)

        # Body yaw with its own (slower) low-pass.
        self._body_yaw += BODY_ALPHA * (target_body - self._body_yaw)
        self._body_yaw = float(_clamp(self._body_yaw, -BODY_LIMIT, BODY_LIMIT))

        # ---- 4. Animation overlay --------------------------------------
        talking = bool(self.is_talking_fn())
        self._target_blend = 1.0 if talking else 0.0
        blend_step = dt / ANIM_CROSSFADE_S
        if self._anim_blend < self._target_blend:
            self._anim_blend = min(self._target_blend, self._anim_blend + blend_step)
        elif self._anim_blend > self._target_blend:
            self._anim_blend = max(self._target_blend, self._anim_blend - blend_step)

        idle_off = self._idle.offsets(now)
        talk_off = self._talking.offsets(now)
        off = blend_offsets(idle_off, talk_off, self._anim_blend)

        head_roll = off["roll"]
        head_pitch = _clamp(
            self._sent_pitch + off["pitch"], -PITCH_LIMIT, PITCH_LIMIT,
        )
        head_yaw = _clamp(
            self._sent_yaw + off["yaw"], -YAW_LIMIT, YAW_LIMIT,
        )

        right_ant, left_ant = off["antennas"]
        antennas = [float(right_ant), float(left_ant)]

        # ---- 5. Send ---------------------------------------------------
        try:
            self.reachy.set_target(
                head=head_pose(roll=head_roll, pitch=head_pitch, yaw=head_yaw),
                body_yaw=self._body_yaw,
                antennas=antennas,
            )
        except Exception as e:
            log.warning("set_target failed: %s", e)

        # ---- 6. Snapshot ----------------------------------------------
        self.snapshot = ControlSnapshot(
            tier=tier,
            track_id=tid,
            have_face=have_face,
            err_x=err_x,
            err_y=err_y,
            sent_yaw=self._sent_yaw,
            sent_pitch=self._sent_pitch,
            body_yaw=self._body_yaw,
            antennas=tuple(antennas),
            talking=talking,
        )
        return self.snapshot

    # ---- Helpers --------------------------------------------------------
    def _compute_body_target(self, head_yaw: float) -> float:
        if self.no_body:
            return 0.0
        if head_yaw > BODY_ENGAGE_RAD:
            return head_yaw - BODY_ENGAGE_RAD
        if head_yaw < -BODY_ENGAGE_RAD:
            return head_yaw + BODY_ENGAGE_RAD
        return 0.0
