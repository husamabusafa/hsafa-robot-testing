"""events.py - In-process pub/sub for the robot's nervous system.

Every subsystem publishes :class:`Event` instances to one shared
:class:`EventBus` and subscribes to the kinds it cares about. No
module calls another module directly; they all talk through this.

See ``docs/architecture.md`` for the full rationale. This is the
piece that makes "add a new sense" or "add a new skill" a single
file + one subscription, never a rewrite.

Design notes:

* **Thread-safe.** Subscribers are registered / removed under a lock.
  Publishing is lock-free after a shallow copy of the subscriber list
  so a slow subscriber can't block a fast publisher.
* **Fire-and-forget.** Subscribers are called synchronously on the
  publishing thread. Long-running subscribers should hand off to
  their own queue/thread internally. This is intentional -- keeping
  the bus itself O(n) and stateless keeps it impossible to deadlock.
* **Kind is a string.** No Enum. New event kinds are added by writing
  the string; no central registry to edit. Keep kinds kebab-cased.
* **No wildcards.** A subscriber picks exactly the kinds it wants, or
  passes ``"*"`` to get everything (used by the
  :class:`ConversationLog` kind of consumer).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


WILDCARD = "*"


# ---- Canonical event kinds (add new ones freely; this is just docs) ------

# Vision
EVT_PERSON_DETECTED = "person_detected"
EVT_PERSON_LEFT = "person_left"
EVT_FACE_ENROLLED = "face_enrolled"
EVT_NAME_RESOLVED = "name_resolved"        # track_id -> name link appeared
EVT_GESTURE_DETECTED = "gesture_detected"  # wave / point / thumbs-up / open-palm

# Audio / voice
EVT_AUDIO_SPEECH_ACTIVE = "audio_speech_active"   # bool, from VAD
EVT_SPEECH_HEARD = "speech_heard"                 # finalized utterance text
EVT_USER_SAID = "user_said"                       # from Gemini turn boundary
EVT_ROBOT_SAID = "robot_said"                     # from Gemini turn boundary
EVT_VOICE_IDENTIFIED = "voice_identified"         # speaker-id from ECAPA utterance
EVT_VOICE_ENROLLED = "voice_enrolled"             # cross-modal co-occurrence commit

# Focus / gaze
EVT_GAZE_TARGET_CHANGED = "gaze_target_changed"
EVT_GAZE_PRIOR = "gaze_prior"               # thinker-pushed soft nudge
EVT_PERSON_LOST = "person_lost"             # someone we were tracking vanished
EVT_VOICE_UNSEEN = "voice_unseen"           # heard voice, saw no face

# Brain / proactive
EVT_SAY_THIS = "say_this"                   # thinker-initiated speech
EVT_CORRECTION = "correction"               # "I'm not Kindom, I'm Husam"


# ---- Event + handler types -----------------------------------------------

Handler = Callable[["Event"], None]


@dataclass
class Event:
    """One thing that happened. Immutable once published."""
    kind: str
    source: str = "?"                     # "vision" / "voice" / "focus" / "brain"
    ts: float = field(default_factory=time.monotonic)
    payload: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Event(kind={self.kind!r}, source={self.source!r}, "
            f"payload_keys={list(self.payload)})"
        )


# ---- The bus --------------------------------------------------------------

class EventBus:
    """Thread-safe pub/sub with exact-kind + wildcard subscribers.

    A :class:`EventBus` is process-local. Cross-process routing (e.g.
    to Hsafa Core) is a separate bridge subscriber that forwards
    events out. Keep this class ignorant of any transport.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # kind -> list[handler]. ``"*"`` lives here too.
        self._subs: Dict[str, List[Handler]] = {}
        self._published = 0

    # ---- subscription -------------------------------------------------
    def subscribe(self, kind: str, handler: Handler) -> Callable[[], None]:
        """Register a handler for one event kind (or ``"*"`` for all).

        Returns an unsubscribe callable so modules can be wired up with
        nothing more than a with-block or a ``try/finally`` stanza.
        """
        if not kind:
            raise ValueError("kind must be a non-empty string")
        with self._lock:
            self._subs.setdefault(kind, []).append(handler)

        def _unsub() -> None:
            with self._lock:
                lst = self._subs.get(kind)
                if not lst:
                    return
                try:
                    lst.remove(handler)
                except ValueError:
                    pass
                if not lst:
                    self._subs.pop(kind, None)

        return _unsub

    # ---- publishing ---------------------------------------------------
    def publish(
        self,
        kind: str,
        *,
        source: str = "?",
        **payload: Any,
    ) -> Event:
        """Build an :class:`Event` and deliver it synchronously.

        Returns the published event so callers can log / inspect it.
        Handler exceptions are caught and logged -- a buggy subscriber
        must never break unrelated ones.
        """
        evt = Event(kind=kind, source=source, payload=dict(payload))
        self._deliver(evt)
        return evt

    def publish_event(self, evt: Event) -> None:
        """Publish a pre-built :class:`Event` (handy for replay / bridges)."""
        self._deliver(evt)

    def _deliver(self, evt: Event) -> None:
        with self._lock:
            exact = list(self._subs.get(evt.kind, ()))
            wild = list(self._subs.get(WILDCARD, ()))
        self._published += 1
        for h in exact + wild:
            try:
                h(evt)
            except Exception:  # pragma: no cover - defensive
                log.exception(
                    "EventBus subscriber for %r raised; continuing", evt.kind,
                )

    # ---- diagnostics --------------------------------------------------
    @property
    def published_count(self) -> int:
        return self._published

    def subscriber_count(self, kind: Optional[str] = None) -> int:
        with self._lock:
            if kind is None:
                return sum(len(v) for v in self._subs.values())
            return len(self._subs.get(kind, ()))
