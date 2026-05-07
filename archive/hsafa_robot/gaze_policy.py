"""gaze_policy.py - Scoring-based focus engine.

Given the current :class:`WorldState` (+ any transient priors), pick
which person the robot should look at *right now*. Pure function --
no threads, no motor I/O, no lip/face access. Feed it a snapshot,
get back a :class:`GazePick`.

Design: see ``docs/gaze-policy.md``. Two modes only:

* ``NORMAL`` - scoring engine picks every tick.
* ``PERSON(name)`` - lock onto ``name``. If visible, that's the
  target. If not, silently fall through to ``NORMAL`` for this tick;
  the lock stays armed until :meth:`set_mode_normal` is called.

Everything else (speaker-follow, newcomer glance, idle drift) falls
out of the score terms, not a separate mode.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .world_state import HumanView, WorldState

log = logging.getLogger(__name__)


# ---- Mode ----------------------------------------------------------------

MODE_NORMAL = "normal"
MODE_PERSON = "person"


# ---- Default weights (tune from here, see docs/gaze-policy.md §2.1) -----

@dataclass
class Weights:
    addressed: float = 10.0       # is_being_addressed: facing camera + speaking
    speaking: float = 6.0         # lip motion (gated on VAD if fused)
    new: float = 4.0              # just appeared, decays to 0 over NEW_DECAY_S
    new_decay_s: float = 5.0
    known: float = 2.0            # named in FaceDB / IdentityGraph
    proximity: float = 2.0        # normalized bbox area
    centrality: float = 1.0       # tiebreaker on distance to image center
    thinker_prior: float = 1.0    # scale on prior.weight
    # Fatigue: after a grace period the locked track gets a growing
    # penalty so the eye breaks contact naturally.
    fatigue: float = 3.0
    fatigue_grace_s: float = 4.0
    fatigue_ramp_s: float = 3.0   # time from +0 to +fatigue after grace


DEFAULT_WEIGHTS = Weights()


# ---- Hysteresis -----------------------------------------------------------

# Challenger must beat the current target's score by this amount to
# "steal" focus. Stops the two-person flicker.
SWITCH_MARGIN = 2.0

# After committing to a target, stay at least this long before any
# non-addressed challenger can switch us away.
MIN_DWELL_S = 0.8

# Direct address (is_being_addressed) bypasses dwell entirely.
# (see docs/gaze-policy.md §3)

# When the scored winner has basically zero score AND nobody is
# interesting (no speaker, no newcomer), we flip internal state to
# SCANNING so the motion layer can inject listener-check glances.
ENGAGED_MIN_SCORE = 4.0


# ---- Priors ---------------------------------------------------------------

@dataclass
class GazePrior:
    """Soft nudge from Gemini / Hsafa. Auto-expires.

    At most one of (``track_id``, ``name``, ``azimuth_deg``) is set.
    A prior with ``name`` boosts every human whose ``name`` matches.
    A prior with ``azimuth_deg`` boosts virtual sound candidates (not
    yet wired; placeholder for DOA support).
    """
    weight: float
    ttl_s: float
    reason: str = ""
    track_id: Optional[int] = None
    name: Optional[str] = None
    azimuth_deg: Optional[float] = None
    _born: float = field(default_factory=time.monotonic)

    def alive(self, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.monotonic()
        return (now - self._born) < self.ttl_s


# ---- Candidate scoring output --------------------------------------------

@dataclass
class Candidate:
    track_id: int
    name: Optional[str]
    score: float
    parts: Dict[str, float]
    addressed: bool
    speaking: bool

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "name": self.name,
            "score": round(self.score, 3),
            "parts": {k: round(v, 3) for k, v in self.parts.items()},
            "addressed": self.addressed,
            "speaking": self.speaking,
        }


@dataclass
class GazePick:
    """What the policy decided this tick."""
    mode: str                       # "normal" / "person"
    state: str                      # "engaged" / "scanning" / "idle" / "searching"
    target: Optional[Candidate]     # None = no human in frame or no lock
    # Name we were trying to reach under MODE_PERSON, even if not visible.
    target_name: Optional[str] = None
    # Debugging: every scored candidate this tick + live priors.
    candidates: List[Candidate] = field(default_factory=list)
    priors_active: int = 0

    @property
    def track_id(self) -> Optional[int]:
        return self.target.track_id if self.target else None

    @property
    def name(self) -> Optional[str]:
        return self.target.name if self.target else None


# ---- Helpers -------------------------------------------------------------

def _centrality(h: HumanView, frame_w: Optional[int], frame_h: Optional[int]) -> float:
    """Return a [0, 1] "how centered in frame" score (1 = perfectly centered)."""
    cx, cy = h.center_px
    if not frame_w or not frame_h:
        return 0.5
    nx = (cx / float(frame_w)) - 0.5   # [-0.5, 0.5]
    ny = (cy / float(frame_h)) - 0.5
    d = math.hypot(nx, ny)             # [0, ~0.7]
    # Map 0 -> 1, ~0.5 -> 0. Clip to avoid negatives.
    return max(0.0, 1.0 - 2.0 * d)


def _recency_bonus(h: HumanView, now: float, decay_s: float) -> float:
    """1.0 for just-appeared, 0 after ``decay_s`` (linear)."""
    age = max(0.0, now - h.first_seen)
    if age >= decay_s:
        return 0.0
    return 1.0 - (age / decay_s)


# ---- The policy ----------------------------------------------------------

class GazePolicy:
    """Pure scoring engine + a small bit of hysteresis state.

    Not threaded: call :meth:`tick` whenever a WorldState snapshot
    lands. State kept between ticks is just "who we last committed to
    and when" so we can enforce dwell + fatigue.
    """

    def __init__(
        self,
        weights: Weights = DEFAULT_WEIGHTS,
        *,
        frame_w: Optional[int] = None,
        frame_h: Optional[int] = None,
    ) -> None:
        self.w = weights
        self.frame_w = frame_w
        self.frame_h = frame_h

        self._lock = threading.Lock()
        self._mode = MODE_NORMAL
        self._target_name: Optional[str] = None

        self._priors: List[GazePrior] = []

        # Hysteresis state
        self._current_tid: Optional[int] = None
        self._committed_at: float = 0.0

    # ---- mode API -----------------------------------------------------
    def set_mode_normal(self) -> None:
        with self._lock:
            self._mode = MODE_NORMAL
            self._target_name = None

    def set_mode_person(self, name: str) -> None:
        with self._lock:
            self._mode = MODE_PERSON
            self._target_name = name

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    @property
    def target_name(self) -> Optional[str]:
        with self._lock:
            return self._target_name

    # ---- priors -------------------------------------------------------
    def add_prior(self, prior: GazePrior) -> None:
        with self._lock:
            self._priors.append(prior)

    def clear_priors(self) -> None:
        with self._lock:
            self._priors.clear()

    def _live_priors(self, now: float) -> List[GazePrior]:
        with self._lock:
            # Drop expired priors inline so ``_priors`` doesn't grow
            # without bound.
            self._priors = [p for p in self._priors if p.alive(now)]
            return list(self._priors)

    # ---- set frame dims (for centrality denominator) -----------------
    def set_frame_size(self, w: int, h: int) -> None:
        self.frame_w = int(w)
        self.frame_h = int(h)

    # ---- main tick ----------------------------------------------------
    def tick(
        self,
        world: WorldState,
        *,
        now: Optional[float] = None,
    ) -> GazePick:
        now = now if now is not None else time.monotonic()
        with self._lock:
            mode = self._mode
            target_name = self._target_name

        priors = self._live_priors(now)

        if not world.humans:
            # Nobody visible. Drop lock; the motion layer handles idle.
            state = "searching" if self._just_lost(now) else "idle"
            self._current_tid = None
            return GazePick(
                mode=mode, state=state, target=None,
                target_name=target_name, candidates=[],
                priors_active=len(priors),
            )

        # 1. Person mode short-circuit: if target is visible, lock on.
        if mode == MODE_PERSON and target_name:
            for h in world.humans:
                if h.name == target_name:
                    cand = self._score_one(h, world, priors, now)
                    self._commit(cand.track_id, now)
                    return GazePick(
                        mode=mode, state="engaged", target=cand,
                        target_name=target_name,
                        candidates=[cand], priors_active=len(priors),
                    )
            # Target not visible: fall through to scoring.

        # 2. Score all visible humans.
        cands = [self._score_one(h, world, priors, now) for h in world.humans]
        cands.sort(key=lambda c: c.score, reverse=True)
        best = cands[0]

        # 3. Apply hysteresis.
        pick = self._apply_hysteresis(cands, best, now)

        # 4. State machine: engaged vs scanning.
        if pick.score >= ENGAGED_MIN_SCORE or pick.addressed or pick.speaking:
            state = "engaged"
        else:
            state = "scanning"

        return GazePick(
            mode=mode, state=state, target=pick,
            target_name=target_name if mode == MODE_PERSON else None,
            candidates=cands, priors_active=len(priors),
        )

    # ---- scoring ------------------------------------------------------
    def _score_one(
        self,
        h: HumanView,
        world: WorldState,
        priors: List[GazePrior],
        now: float,
    ) -> Candidate:
        w = self.w
        parts: Dict[str, float] = {}

        addressed = bool(h.is_facing_camera and h.is_speaking)
        speaking = bool(h.is_speaking) or (h.speaking_prob >= 0.5)

        parts["addressed"] = w.addressed if addressed else 0.0

        # speaking term uses probability if we have it, else boolean.
        spk_score = (
            h.speaking_prob if h.speaking_prob > 0 else (1.0 if h.is_speaking else 0.0)
        )
        parts["speaking"] = w.speaking * min(1.0, spk_score)

        parts["new"] = w.new * _recency_bonus(h, now, w.new_decay_s)
        parts["known"] = w.known if h.name else 0.0
        parts["proximity"] = w.proximity * max(0.0, min(1.0, h.proximity))
        parts["centrality"] = (
            w.centrality * _centrality(h, self.frame_w, self.frame_h)
        )

        # Thinker priors
        prior_total = 0.0
        for p in priors:
            match = False
            if p.track_id is not None and p.track_id == h.track_id:
                match = True
            elif p.name is not None and p.name == h.name:
                match = True
            if match:
                prior_total += p.weight
        parts["thinker_prior"] = w.thinker_prior * prior_total

        # Fatigue: penalty grows on the locked track.
        fatigue = 0.0
        if self._current_tid == h.track_id and self._committed_at > 0:
            held = now - self._committed_at
            if held > w.fatigue_grace_s:
                frac = min(1.0, (held - w.fatigue_grace_s) / max(1e-3, w.fatigue_ramp_s))
                fatigue = -w.fatigue * frac
        parts["fatigue"] = fatigue

        score = sum(parts.values())
        return Candidate(
            track_id=h.track_id,
            name=h.name,
            score=score,
            parts=parts,
            addressed=addressed,
            speaking=speaking,
        )

    # ---- hysteresis ---------------------------------------------------
    def _apply_hysteresis(
        self,
        cands: List[Candidate],
        best: Candidate,
        now: float,
    ) -> Candidate:
        """Pick a winner subject to SWITCH_MARGIN / MIN_DWELL_S."""
        if self._current_tid is None:
            self._commit(best.track_id, now)
            return best

        # Is the current track still in the list?
        current = next(
            (c for c in cands if c.track_id == self._current_tid), None,
        )
        if current is None:
            self._commit(best.track_id, now)
            return best

        if best.track_id == current.track_id:
            return current

        dwell_ok = (now - self._committed_at) >= MIN_DWELL_S
        steal_ok = best.score >= (current.score + SWITCH_MARGIN)
        if steal_ok and (dwell_ok or best.addressed):
            self._commit(best.track_id, now)
            return best
        return current

    def _commit(self, tid: Optional[int], now: float) -> None:
        if tid != self._current_tid:
            self._current_tid = tid
            self._committed_at = now

    def _just_lost(self, now: float) -> bool:
        return self._current_tid is not None and (now - self._committed_at) < 2.0
