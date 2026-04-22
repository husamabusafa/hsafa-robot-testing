"""natural_gaze.py - Humanoid motion primitives on top of the P-controller.

``RobotController`` gives us a smooth pursuit controller. This module
layers the behaviors that make the head *feel* alive:

1. **Saccades.** When the target jumps > threshold, snap the head to
   it in ~150 ms with a ballistic profile, then hold.
2. **Micro-saccades.** After the head has been still for > 700 ms,
   inject tiny pseudo-random twitches. Nearly invisible individually,
   they're what make a fixation look alive instead of frozen.
3. **Idle drift.** When no humans are visible for > 3 s, sweep slowly
   (sinusoidal yaw/pitch) with random phase so it's not a metronome.
4. **Search.** On ``person_lost`` / ``voice_unseen``, do a 2-3
   fixation directed sweep (last known side first), then return to
   idle.

Implementation: this class generates yaw/pitch *overrides* (absolute
angles) plus a ``motion_profile`` hint that ``RobotController`` can
use to switch gain/time-constant. If no override is produced, the
controller falls through to its existing smooth-pursuit behavior.

See ``docs/natural-gaze.md`` for the design rationale.
"""
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

log = logging.getLogger(__name__)


# ---- Tuning --------------------------------------------------------------

# Saccade
SACCADE_MIN_JUMP_DEG = 8.0            # below this, use smooth pursuit
SACCADE_DURATION_S = 0.17             # ballistic duration (120-220 ms target)

# Micro-saccades
MICRO_STILL_TRIGGER_S = 0.7           # need this long still before twitching
MICRO_INTERVAL_S = (0.5, 1.8)         # random gap between twitches
MICRO_AMPLITUDE_DEG = 0.7             # peak yaw/pitch offset

# Idle drift
IDLE_AFTER_S = 3.0                    # no-humans time before drift kicks in
IDLE_YAW_AMP_DEG = 15.0
IDLE_PITCH_AMP_DEG = 5.0
IDLE_PERIOD_S = 6.0

# Search behavior
SEARCH_HOLD_S = 0.5                   # keep looking at last-known spot
SEARCH_SWEEP_YAW_DEG = 35.0           # first sweep target
SEARCH_FIXATION_S = 0.8               # hold at each sweep point
SEARCH_TOTAL_S = 3.0                  # total duration before giving up
SEARCH_COOLDOWN_S = 4.0               # min gap between two searches

# Listener-check glances (multi-person rooms)
LISTENER_GLANCE_PROBABILITY = 0.002   # per tick; ~0.2 Hz at 100 Hz ctrl loop
LISTENER_GLANCE_FIXATION_S = 0.4
LISTENER_GLANCE_YAW_DEG = 8.0


class MotionProfile(Enum):
    """Hint for how the P-controller should behave this tick."""
    PURSUIT = "pursuit"       # normal smooth pursuit (default)
    SACCADE = "saccade"       # fast ballistic
    IDLE_DRIFT = "idle_drift" # absolute override, no tracking
    SEARCH = "search"         # absolute override, directed sweep
    HOLD = "hold"             # freeze; small micro-saccades allowed


@dataclass
class GazeCommand:
    """Output of :meth:`NaturalGaze.tick` that ``RobotController`` consumes."""
    profile: MotionProfile = MotionProfile.PURSUIT

    # Absolute yaw/pitch overrides (radians). When both are None the
    # P-controller continues with its own error-driven command.
    override_yaw: Optional[float] = None
    override_pitch: Optional[float] = None

    # Additive offsets (radians). Used by micro-saccades so the
    # baseline tracking pose still shows through.
    offset_yaw: float = 0.0
    offset_pitch: float = 0.0

    # Gain multiplier applied to the P-controller's KP_{YAW,PITCH} for
    # this tick. Saccades use > 1 to snap; idle uses 1.
    gain_scale: float = 1.0

    reason: str = ""


# ---- Internal state -------------------------------------------------------

@dataclass
class _SearchIntent:
    reason: str
    started_at: float
    last_known_yaw: float   # radians -- direction to look first
    phase: str = "hold"     # "hold" -> "sweep_a" -> "sweep_b" -> "done"
    phase_started_at: float = 0.0


@dataclass
class _Micro:
    next_twitch_at: float = 0.0
    target_offset: Tuple[float, float] = (0.0, 0.0)
    target_until: float = 0.0


# ---- The planner ----------------------------------------------------------

class NaturalGaze:
    """Generates :class:`GazeCommand` s from focus state + history.

    Usage (per tick)::

        cmd = natural_gaze.tick(
            have_target=have_face,
            target_err=(err_x, err_y),    # from tracker (normalized [-1, 1])
            current_yaw=rc._cmd_yaw,
            current_pitch=rc._cmd_pitch,
            no_humans_s=no_humans_age,
            state_hint=focus_snapshot.state,
        )
        rc.apply_gaze_command(cmd)

    Thread-safety: single-writer (the main control loop). Don't call
    from multiple threads.
    """

    def __init__(self) -> None:
        self._last_target_yaw: Optional[float] = None
        self._last_target_pitch: Optional[float] = None
        self._last_target_change: float = 0.0

        self._saccade_until: float = 0.0
        self._saccade_gain: float = 1.0

        self._still_since: float = time.monotonic()
        self._micro = _Micro()

        self._idle_phase = random.uniform(0.0, 2.0 * math.pi)
        self._idle_pitch_phase = random.uniform(0.0, 2.0 * math.pi)

        self._search: Optional[_SearchIntent] = None
        # Don't restart a search until this clock. Prevents the
        # "thousand searches per second" failure mode caused by
        # upstream trackers re-IDing the same person repeatedly.
        self._search_cooldown_until: float = 0.0

        self._listener_glance_until: float = 0.0
        self._listener_glance_yaw: float = 0.0

    # ---- triggers --------------------------------------------------
    def notify_target_changed(
        self, new_yaw_rad: float, new_pitch_rad: float,
        *, now: Optional[float] = None,
    ) -> None:
        """Tell the planner the gaze target just switched.

        Based on the angular jump, either starts a saccade or leaves
        the smooth pursuit path alone.
        """
        now = now if now is not None else time.monotonic()
        if self._last_target_yaw is None:
            self._last_target_yaw = new_yaw_rad
            self._last_target_pitch = new_pitch_rad
            self._last_target_change = now
            return

        dyaw = new_yaw_rad - self._last_target_yaw
        dpitch = new_pitch_rad - self._last_target_pitch
        jump_deg = math.degrees(math.hypot(dyaw, dpitch))

        self._last_target_yaw = new_yaw_rad
        self._last_target_pitch = new_pitch_rad
        self._last_target_change = now

        if jump_deg >= SACCADE_MIN_JUMP_DEG:
            self._saccade_until = now + SACCADE_DURATION_S
            # Scale gain so the ballistic reaches target in ~duration.
            # Concretely: make the P-controller ~3-5x snappier for the
            # saccade window. Real human saccades peak ~5x tracking
            # speed (see natural-gaze.md §3).
            self._saccade_gain = 4.0
            # Saccades punch through micro-saccade stillness.
            self._still_since = now + 10.0  # defer the next micro
            log.debug("saccade: jump=%.1f deg, gain=%.1f", jump_deg, self._saccade_gain)

    def notify_person_lost(
        self,
        *,
        last_known_yaw_rad: float,
        now: Optional[float] = None,
    ) -> None:
        """Trigger a directed search where the person was last seen.

        Idempotent: if a search is already running or the cooldown is
        active, this is a no-op. Without this guard, ByteTrack re-IDs
        flood this method and the head ends up oscillating between
        every transient ``last_known_yaw`` the upstream trackers pick.
        """
        now = now if now is not None else time.monotonic()
        if self._search is not None:
            return
        if now < self._search_cooldown_until:
            return
        self._search = _SearchIntent(
            reason="person_lost",
            started_at=now,
            last_known_yaw=last_known_yaw_rad,
            phase="hold",
            phase_started_at=now,
        )
        log.info(
            "NaturalGaze: starting search (last_yaw=%+.1f deg)",
            math.degrees(last_known_yaw_rad),
        )

    def notify_voice_unseen(
        self,
        *,
        guess_yaw_rad: Optional[float] = None,
        now: Optional[float] = None,
    ) -> None:
        """Trigger a "who said that?" sweep when we hear but don't see."""
        now = now if now is not None else time.monotonic()
        if self._search is not None:
            return
        if now < self._search_cooldown_until:
            return
        # Without DOA, guess the opposite side of the room to whatever
        # we're looking at. With DOA, caller passes guess_yaw.
        seed_yaw = guess_yaw_rad if guess_yaw_rad is not None else (
            -math.copysign(math.radians(SEARCH_SWEEP_YAW_DEG),
                           self._last_target_yaw or 0.0)
            or math.radians(-SEARCH_SWEEP_YAW_DEG)
        )
        self._search = _SearchIntent(
            reason="voice_unseen",
            started_at=now,
            last_known_yaw=seed_yaw,
            phase="hold",
            phase_started_at=now,
        )

    def abort_search(self) -> None:
        self._search = None

    def cue_listener_glance(
        self, other_person_yaw_rad: float,
        *, now: Optional[float] = None,
    ) -> None:
        """Explicit "glance at the other person" hint.

        Called by a multi-person heuristic elsewhere -- this class
        doesn't know about ``WorldState``, just positions.
        """
        now = now if now is not None else time.monotonic()
        self._listener_glance_until = now + LISTENER_GLANCE_FIXATION_S
        self._listener_glance_yaw = other_person_yaw_rad

    # ---- per-tick main loop ---------------------------------------
    def tick(
        self,
        *,
        have_target: bool,
        current_yaw: float,
        current_pitch: float,
        no_humans_s: float,
        state_hint: str = "engaged",
        now: Optional[float] = None,
    ) -> GazeCommand:
        """Return the command the controller should apply this tick.

        Parameters
        ----------
        have_target:
            True if the tracker is currently locked on something.
        current_yaw / current_pitch:
            The P-controller's latest commanded angles (radians).
        no_humans_s:
            Seconds since anything was visible. 0 while anyone is
            present; grows when the room empties.
        state_hint:
            ``"engaged"`` / ``"scanning"`` / ``"idle"`` / ``"searching"``
            from the :class:`GazePolicy`.
        """
        now = now if now is not None else time.monotonic()

        # 1. Active search. Abort instantly if the person came back
        # into view -- otherwise the head keeps sweeping while the
        # user is already standing right in front of the camera.
        if self._search is not None:
            if have_target:
                log.info(
                    "NaturalGaze: target reacquired, aborting %s search",
                    self._search.reason,
                )
                self._search = None
                self._search_cooldown_until = now + SEARCH_COOLDOWN_S
                # Fall through to pursuit.
            else:
                return self._tick_search(now)

        # 2. Idle drift when the room is empty.
        if not have_target and no_humans_s >= IDLE_AFTER_S:
            return self._tick_idle(now)

        # 3. Listener-check glance (explicitly cued).
        if now < self._listener_glance_until:
            return GazeCommand(
                profile=MotionProfile.SACCADE,
                override_yaw=self._listener_glance_yaw,
                override_pitch=current_pitch,
                gain_scale=4.0,
                reason="listener_glance",
            )

        # 4. Saccade window: boost gain for fast snap.
        if now < self._saccade_until:
            return GazeCommand(
                profile=MotionProfile.SACCADE,
                gain_scale=self._saccade_gain,
                reason="saccade",
            )

        # 5. Micro-saccades when stationary + engaged.
        micro_off = self._maybe_micro(now, still=True)
        if micro_off is not None:
            return GazeCommand(
                profile=MotionProfile.HOLD,
                offset_yaw=micro_off[0],
                offset_pitch=micro_off[1],
                reason="micro_saccade",
            )

        # 6. Default: normal smooth pursuit.
        return GazeCommand(profile=MotionProfile.PURSUIT, reason="pursuit")

    # ---- Internal helpers ----------------------------------------
    def _tick_search(self, now: float) -> GazeCommand:
        assert self._search is not None
        s = self._search
        elapsed = now - s.started_at

        if elapsed >= SEARCH_TOTAL_S:
            self._search = None
            self._search_cooldown_until = now + SEARCH_COOLDOWN_S
            return GazeCommand(profile=MotionProfile.PURSUIT, reason="search_done")

        phase_dur = now - s.phase_started_at
        # Phase machine: hold (on last known) -> sweep_a (+dir) -> sweep_b (-dir)
        if s.phase == "hold" and phase_dur >= SEARCH_HOLD_S:
            s.phase = "sweep_a"
            s.phase_started_at = now
        elif s.phase == "sweep_a" and phase_dur >= SEARCH_FIXATION_S:
            s.phase = "sweep_b"
            s.phase_started_at = now
        elif s.phase == "sweep_b" and phase_dur >= SEARCH_FIXATION_S:
            self._search = None
            self._search_cooldown_until = now + SEARCH_COOLDOWN_S
            return GazeCommand(profile=MotionProfile.PURSUIT, reason="search_done")

        target_yaw = s.last_known_yaw
        if s.phase == "sweep_a":
            target_yaw = s.last_known_yaw + math.radians(SEARCH_SWEEP_YAW_DEG) * (
                1.0 if s.last_known_yaw >= 0 else -1.0
            )
        elif s.phase == "sweep_b":
            target_yaw = -s.last_known_yaw

        return GazeCommand(
            profile=MotionProfile.SEARCH,
            override_yaw=target_yaw,
            override_pitch=0.0,
            gain_scale=3.0,
            reason=f"search:{s.reason}:{s.phase}",
        )

    def _tick_idle(self, now: float) -> GazeCommand:
        # Slow sinusoidal sweep with pseudo-random phase so it's not a
        # metronome. Amplitudes from the docs.
        w = 2.0 * math.pi / IDLE_PERIOD_S
        yaw = math.radians(IDLE_YAW_AMP_DEG) * math.sin(w * now + self._idle_phase)
        pitch = (
            math.radians(IDLE_PITCH_AMP_DEG)
            * math.sin(0.7 * w * now + self._idle_pitch_phase)
        )
        return GazeCommand(
            profile=MotionProfile.IDLE_DRIFT,
            override_yaw=yaw,
            override_pitch=pitch,
            gain_scale=0.5,       # soft, slow motion
            reason="idle_drift",
        )

    def _maybe_micro(
        self, now: float, still: bool,
    ) -> Optional[Tuple[float, float]]:
        """Return a tiny (dyaw, dpitch) offset for a micro-saccade, or None."""
        if not still:
            return None
        # Let the micro-saccade clock start only after we've been still
        # long enough.
        if now - self._still_since < MICRO_STILL_TRIGGER_S:
            return None

        m = self._micro
        # Finishing an in-progress micro-saccade?
        if now < m.target_until:
            return m.target_offset

        if now < m.next_twitch_at:
            return None

        # Pick a new micro-saccade target.
        amp = math.radians(MICRO_AMPLITUDE_DEG)
        dyaw = random.uniform(-amp, amp)
        dpitch = random.uniform(-amp * 0.5, amp * 0.5)
        duration = random.uniform(0.04, 0.08)   # 40-80 ms burst
        m.target_offset = (dyaw, dpitch)
        m.target_until = now + duration
        m.next_twitch_at = now + random.uniform(*MICRO_INTERVAL_S)
        return m.target_offset

    # ---- RobotController bookkeeping hooks -----------------------
    def note_motion_settled(self, now: Optional[float] = None) -> None:
        """Called by RobotController when |error| has been under deadzone."""
        now = now if now is not None else time.monotonic()
        if self._still_since > now:
            # Earlier saccades parked this in the future; pull it back.
            self._still_since = now

    def note_motion_active(self) -> None:
        """Called when the P-controller is actively moving."""
        self._still_since = time.monotonic() + 1e9  # "not still"
