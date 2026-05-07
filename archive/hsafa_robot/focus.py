"""focus.py - Decide which person the robot should look at.

Thin glue layer. The scoring lives in :class:`GazePolicy`; this class:

* Holds the current user-facing mode (``normal`` or ``person(name)``)
  and any soft :class:`GazePrior` s (e.g. from ``focus_on_speaker``).
* Every tick, builds a :class:`WorldState` from the vision snapshots,
  asks the policy for a :class:`GazePick`, and pushes that pick into
  :meth:`CascadeTracker.set_locked_id`.
* Publishes a ``gaze_target_changed`` event when the chosen track
  actually changes, so other subsystems (context injector, logs,
  future Hsafa bridge) can react without polling.

Backward-compat: ``main.py`` still calls
``set_mode_auto`` / ``set_mode_person`` / ``set_mode_speaker`` and the
classic ``focus_on_person`` / ``focus_on_speaker`` / ``clear_focus``
tools keep working. Internally those now just toggle the mode + add
or clear the speaker prior.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .events import (
    EVT_GAZE_TARGET_CHANGED,
    EVT_NAME_RESOLVED,
    EVT_PERSON_DETECTED,
    EVT_PERSON_LEFT,
    EventBus,
)
from .face_db import canonicalize_name
from .face_recognizer import FaceMatch
from .gaze_policy import (
    MODE_NORMAL,
    MODE_PERSON,
    Candidate,
    DEFAULT_WEIGHTS,
    GazePick,
    GazePolicy,
    GazePrior,
)
from .lip_motion import LipMotionTracker
from .perception import HumanRegistry, build_humans
from .tracker import CascadeTracker
from .world_state import WorldStateHolder

log = logging.getLogger(__name__)


Bbox = Tuple[int, int, int, int]


# Legacy mode constants kept for backward compat with any caller that
# still reads ``FocusSnapshot.mode``. Internally the policy only knows
# ``normal`` and ``person``; "auto" and "speaker" are presentation-layer
# strings that main.py / the preview overlay already use.
MODE_AUTO = "auto"
MODE_SPEAKER_LEGACY = "speaker"
VALID_MODES = (MODE_AUTO, MODE_PERSON, MODE_SPEAKER_LEGACY, MODE_NORMAL)


# Transient speaker-prior: when a caller says "focus on speaker",
# we bias the scorer toward the current speaker for this many
# seconds by attaching this prior to whoever is_speaking right now.
_SPEAKER_PRIOR_WEIGHT = 3.0
_SPEAKER_PRIOR_TTL_S = 10.0


# ---- Public snapshot ------------------------------------------------------

@dataclass
class FocusSnapshot:
    """What the focus manager decided on the most recent ``update`` call."""
    mode: str                       # presentation-layer string
    target_name: Optional[str]      # canonical name requested via MODE_PERSON
    locked_id: Optional[int]        # bytetrack id currently pushed to tracker
    focused_name: Optional[str]     # best-known name for the locked track
    state: str                      # "engaged" / "scanning" / "idle" / "searching"
    name_by_track: Dict[int, str]   # for preview overlays
    top_candidates: List[Candidate]

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "target_name": self.target_name,
            "focused_name": self.focused_name,
            "locked_id": self.locked_id,
            "state": self.state,
        }


# ---- Matcher util used by try_focus_by_face_match ------------------------

def _face_in_body_score(face: Bbox, body: Bbox) -> float:
    fx1, fy1, fx2, fy2 = face
    bx1, by1, bx2, by2 = body
    ix1, iy1 = max(fx1, bx1), max(fy1, by1)
    ix2, iy2 = min(fx2, bx2), min(fy2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    face_area = max(1, (fx2 - fx1) * (fy2 - fy1))
    return inter / face_area


def _match_face_to_yolo(
    face_bbox: Bbox,
    yolo_tracks: List[Tuple[int, Bbox]],
    *,
    min_overlap: float = 0.5,
) -> Optional[int]:
    best_id, best_score = None, 0.0
    for tid, body in yolo_tracks:
        s = _face_in_body_score(face_bbox, body)
        if s > best_score:
            best_score = s
            best_id = tid
    if best_id is None or best_score < min_overlap:
        return None
    return best_id


# ---- The manager ----------------------------------------------------------

class FocusManager:
    """Owns the current focus intent + drives the GazePolicy each tick."""

    def __init__(
        self,
        tracker: CascadeTracker,
        lip_tracker: Optional[LipMotionTracker],
        *,
        world: Optional[WorldStateHolder] = None,
        bus: Optional[EventBus] = None,
        frame_w: Optional[int] = None,
        frame_h: Optional[int] = None,
    ) -> None:
        self._tracker = tracker
        self._lip_tracker = lip_tracker
        self._world = world
        self._bus = bus

        self._policy = GazePolicy(
            DEFAULT_WEIGHTS, frame_w=frame_w, frame_h=frame_h,
        )
        self._registry = HumanRegistry()

        # Presentation-layer mode string (what tools / UI show) -- tracked
        # separately from the policy's real mode so "speaker" can show up
        # even though the policy treats it as NORMAL+prior.
        self._display_mode = MODE_AUTO

        # Sticky name-resolution cache so the tool layer's
        # ``try_focus_by_face_match`` keeps working synchronously.
        self._name_to_track: Dict[str, Tuple[int, float]] = {}
        self._mapping_ttl_s = 2.0

        self._lock = threading.Lock()
        self._last_pushed_id: Optional[int] = None
        self._last_target_name: Optional[str] = None

        # Remember alive tracks between updates so we can emit
        # person_detected / person_left events on transitions.
        self._known_tids: set[int] = set()
        # Pending-departure debounce: YOLO/ByteTrack re-assigns track
        # IDs on brief occlusions, which used to fire person_left +
        # retrigger a gaze search every single frame the ID flipped.
        # Only actually fire person_left once a tid has been missing
        # for this long, and only if the person's *name* isn't still
        # visible on another live track (same person, new ID).
        self._pending_departures: Dict[int, float] = {}
        self._last_names_seen: Dict[str, float] = {}
        self._departure_debounce_s = 2.0

    # ---- accessors -----------------------------------------------------
    @property
    def registry(self) -> HumanRegistry:
        return self._registry

    @property
    def policy(self) -> GazePolicy:
        return self._policy

    # ---- intent API (backward-compatible wrappers) --------------------
    def set_mode_auto(self) -> None:
        """Legacy alias for ``set_mode_normal()``."""
        self.set_mode_normal()

    def set_mode_normal(self) -> None:
        with self._lock:
            self._display_mode = MODE_AUTO
        self._policy.set_mode_normal()
        self._policy.clear_priors()
        self._tracker.set_locked_id(None)
        self._last_pushed_id = None
        log.info("FocusManager: mode=normal (priors cleared)")

    def set_mode_person(self, name: str) -> bool:
        canonical = canonicalize_name(name)
        if not canonical:
            return False
        with self._lock:
            self._display_mode = MODE_PERSON
        self._policy.set_mode_person(canonical)
        log.info("FocusManager: mode=person name=%s", canonical)
        return True

    def set_mode_speaker(self) -> None:
        """Legacy tool: stay in NORMAL but inject a speaker prior."""
        with self._lock:
            self._display_mode = MODE_SPEAKER_LEGACY
        self._policy.set_mode_normal()
        # Attach a TTL'd prior on whoever is_speaking at score time.
        # We can't know the track_id yet, so use ``name=None +
        # track_id=None`` and filter in _apply_speaker_prior below.
        self._policy.add_prior(GazePrior(
            weight=_SPEAKER_PRIOR_WEIGHT,
            ttl_s=_SPEAKER_PRIOR_TTL_S,
            reason="focus_on_speaker",
        ))
        log.info("FocusManager: mode=speaker (prior weight=%.1f ttl=%.1fs)",
                 _SPEAKER_PRIOR_WEIGHT, _SPEAKER_PRIOR_TTL_S)

    def add_prior(self, prior: GazePrior) -> None:
        self._policy.add_prior(prior)

    def clear_priors(self) -> None:
        self._policy.clear_priors()

    # ---- per-frame update ---------------------------------------------
    def update(
        self,
        yolo_tracks: List[Tuple[int, Bbox]],
        *,
        frame_w: Optional[int] = None,
        frame_h: Optional[int] = None,
        now: Optional[float] = None,
    ) -> FocusSnapshot:
        """Compute the desired locked_id and push it to the tracker."""
        now = now if now is not None else time.monotonic()

        if frame_w is not None and frame_h is not None:
            self._policy.set_frame_size(frame_w, frame_h)

        fw = frame_w or self._policy.frame_w or 640
        fh = frame_h or self._policy.frame_h or 480

        # ---- 1. Pull fresh speech snap -------------------------------
        speech_snaps = []
        if self._lip_tracker is not None:
            for cand in self._lip_tracker.snapshot():
                # Use motion_score / threshold as a rough speaking_prob.
                # Real prob comes from VAD fusion when available.
                prob = min(1.0, cand.motion_score / 10.0)
                speech_snaps.append(
                    (cand.bbox, cand.name, cand.is_speaking, prob),
                )

        # ---- 2. Build HumanViews --------------------------------------
        humans, _face_to_track = build_humans(
            yolo_tracks=yolo_tracks,
            speech_snaps=speech_snaps,
            frame_w=fw,
            frame_h=fh,
            registry=self._registry,
            now=now,
        )

        # Name mapping cache for the tool layer.
        name_by_track: Dict[int, str] = {}
        for h in humans:
            if h.name:
                name_by_track[h.track_id] = h.name
                self._name_to_track[h.name] = (h.track_id, now)

        # ---- 3. Publish WorldState + events ---------------------------
        if self._world is not None:
            self._world.replace_humans(humans)

        self._emit_presence_events(humans, now)

        # ---- 4. Policy decision ---------------------------------------
        # Transient per-tick "speaker prior": while a focus_on_speaker
        # prior is live, boost whoever is_speaking right now. We
        # implement this by adding an ephemeral same-tick prior with
        # ttl 0.01 s so it doesn't leak.
        self._refresh_speaker_prior(humans, now)

        # GazePolicy consumes a WorldState; build a lightweight one if
        # we don't have a holder.
        if self._world is not None:
            world_snap = self._world.snapshot()
        else:
            from .world_state import WorldState
            world_snap = WorldState(humans=humans, last_update=now)
        pick = self._policy.tick(world_snap, now=now)

        # ---- 5. Apply to tracker --------------------------------------
        desired_tid = pick.track_id
        focused_name = pick.name
        if pick.mode == MODE_PERSON and pick.target is None:
            # Person-mode target not visible. Don't push a lock; the
            # tracker keeps its largest-body behaviour so the robot
            # stays alive while we wait for the target to come back.
            desired_tid = None
            focused_name = None

        if desired_tid != self._last_pushed_id:
            self._tracker.set_locked_id(desired_tid)
            self._last_pushed_id = desired_tid
            if self._bus is not None:
                self._bus.publish(
                    EVT_GAZE_TARGET_CHANGED,
                    source="focus",
                    track_id=desired_tid,
                    name=focused_name,
                    state=pick.state,
                )

        # ---- 6. Update WorldState's robot view ------------------------
        if self._world is not None:
            self._world.set_robot_target(
                desired_tid, focused_name,
                gaze_mode=pick.mode,
                gaze_state=pick.state,
            )

        # ---- 7. Map display-mode string -------------------------------
        display_mode = self._display_mode
        if pick.mode == MODE_PERSON:
            display_mode = MODE_PERSON

        return FocusSnapshot(
            mode=display_mode,
            target_name=pick.target_name,
            locked_id=desired_tid,
            focused_name=focused_name,
            state=pick.state,
            name_by_track=name_by_track,
            top_candidates=pick.candidates[:3],
        )

    # ---- internals ----------------------------------------------------
    def _refresh_speaker_prior(self, humans, now: float) -> None:
        """If a speaker-prior is live, attach a per-tick prior to the active speaker."""
        # We look at our ``_policy._priors`` (read via a safe helper).
        # The persistent prior has reason=="focus_on_speaker"; we mirror
        # it onto the currently speaking human's name so the scorer
        # actually boosts someone this tick. The ephemeral prior has a
        # tiny TTL and will be pruned automatically.
        alive_speaker_prior = any(
            p.reason == "focus_on_speaker" and p.alive(now)
            for p in self._policy._live_priors(now)  # noqa: SLF001 (internal ok)
        )
        if not alive_speaker_prior:
            return
        # Pick the most probable speaker this tick.
        best = None
        best_p = 0.0
        for h in humans:
            if h.speaking_prob > best_p or h.is_speaking:
                best_p = max(best_p, h.speaking_prob, 1.0 if h.is_speaking else 0.0)
                best = h
        if best is None:
            return
        # Ephemeral prior = small TTL so it only counts this tick.
        self._policy.add_prior(GazePrior(
            weight=_SPEAKER_PRIOR_WEIGHT,
            ttl_s=0.01,
            reason="tick_speaker",
            track_id=best.track_id,
        ))

    def _emit_presence_events(self, humans, now: float) -> None:
        if self._bus is None:
            return
        alive = {h.track_id for h in humans}
        # Names currently visible on any live track.
        alive_names = {h.name for h in humans if h.name}

        # Arrivals: new track_ids. Suppress if a track with the same
        # *name* was seen very recently -- that's the same person
        # re-identified after a brief ByteTrack hiccup, not a new
        # arrival.
        arrivals = alive - self._known_tids
        for tid in arrivals:
            h = next((x for x in humans if x.track_id == tid), None)
            if h is None:
                continue
            last_name_seen = (
                self._last_names_seen.get(h.name) if h.name else None
            )
            is_reappearance = (
                last_name_seen is not None
                and (now - last_name_seen) <= self._departure_debounce_s
            )
            if is_reappearance:
                # Clear any pending departure for the old tid of the
                # same person -- the re-ID already covered it.
                for pending_tid in [
                    t for t, _ in self._pending_departures.items()
                    if self._registry.get(t)
                    and self._registry.get(t).name == h.name
                ]:
                    self._pending_departures.pop(pending_tid, None)
                continue
            self._bus.publish(
                EVT_PERSON_DETECTED,
                source="vision",
                track_id=tid,
                name=h.name,
                direction=h.direction,
                distance_est=h.distance_est,
            )

        # Departures: tids we stopped seeing. Debounce: record when we
        # first missed them, only actually fire person_left after the
        # grace period AND only if their name isn't still visible on
        # some other track.
        departures = self._known_tids - alive
        for tid in departures:
            if tid not in self._pending_departures:
                self._pending_departures[tid] = now

        # Process pending departures: fire if expired and the person's
        # name isn't still in the room.
        fired = []
        for tid, missed_since in list(self._pending_departures.items()):
            if tid in alive:
                # They came back under the same id. Cancel.
                fired.append(tid)
                continue
            age = now - missed_since
            if age < self._departure_debounce_s:
                continue
            memo = self._registry.get(tid)
            name = memo.name if memo is not None else None
            if name and name in alive_names:
                # Same person is visible on a different track; swallow.
                fired.append(tid)
                continue
            self._bus.publish(
                EVT_PERSON_LEFT, source="vision", track_id=tid, name=name,
            )
            fired.append(tid)
        for tid in fired:
            self._pending_departures.pop(tid, None)

        # Name resolutions: tid gained a name it didn't have before.
        for h in humans:
            memo = self._registry.get(h.track_id)
            if memo is None:
                continue
            if h.name:
                self._last_names_seen[h.name] = now
            # Fire once per (tid, name) on first appearance.
            key = f"__named_fired_{h.track_id}_{h.name}"
            if h.name and not getattr(memo, key, False):
                setattr(memo, key, True)
                self._bus.publish(
                    EVT_NAME_RESOLVED, source="vision",
                    track_id=h.track_id, name=h.name,
                )

        self._known_tids = alive

        # Evict long-gone tracks from the registry so we don't leak.
        self._registry.prune(alive, now)

    # ---- Public helpers for the tool layer ---------------------------
    def try_focus_by_face_match(
        self,
        face: FaceMatch,
        yolo_tracks: List[Tuple[int, Bbox]],
    ) -> Optional[int]:
        """Synchronous "focus_on_person" shortcut: lock now, don't wait a tick."""
        tid = _match_face_to_yolo(face.bbox, yolo_tracks)
        if tid is None:
            return None
        self._tracker.set_locked_id(tid)
        self._last_pushed_id = tid
        if face.name:
            self._name_to_track[face.name] = (tid, time.monotonic())
            # Seed the registry so the next update keeps the name sticky
            # even before the lip-motion tracker runs its identify cycle.
            memo = self._registry.touch(tid, time.monotonic())
            memo.name = face.name
        return tid

    def find_track_for_name(self, name: str) -> Optional[int]:
        """Return the YOLO track_id currently bound to ``name``, or None."""
        entry = self._name_to_track.get(canonicalize_name(name))
        if entry is None:
            return None
        tid, ts = entry
        if (time.monotonic() - ts) > self._mapping_ttl_s:
            return None
        return tid
