# Gaze Policy — Scoring-Based Focus

How Hsafa decides **who to look at**, with only two user-facing modes:

- **`person`** — lock onto a specific named person. If they leave the
  frame, automatically fall back to `normal` scoring until they return.
- **`normal`** (default) — scoring-based attention: every visible
  person is scored each tick, highest score wins, with hysteresis.

That's it. No `auto` vs `speaker` vs `largest` split. Everything else
(speaker-follow, newcomer-glance, idle drift) falls out of the scoring
function, exactly like a human's attention works.

This doc replaces the mode list in `gaze-modes.md` and makes the
`GazePolicy` sketched in `natural-gaze.md` concrete. It's scoped at
implementation level so it's a drop-in plan for `focus.py`.

---

## 1. The two modes

### 1.1 `normal` (default)

Scoring engine picks the target every tick. See §2 and §3.

### 1.2 `person(name)`

Lock onto `name`. Each tick:

1. Look up `name` in the current `WorldState`.
2. **If visible** → target = that person's `track_id`. Done.
3. **If NOT visible** → silently fall back to the `normal` scoring
   engine for this tick (and emit `person_lost(name)` once so Hsafa /
   search behavior can react). The lock stays armed — the instant
   they reappear in `WorldState`, we re-lock.

No third "speaker" mode is needed: to "follow whoever speaks" you
just stay in `normal`, because `is_speaking` is the biggest score
term.

### 1.3 Tool surface (unchanged for Gemini)

Gemini's existing tools keep working, now as thin wrappers:

| Tool | Effect |
|------|--------|
| `focus_on_person(name)` | `mode = person(name)` |
| `focus_on_speaker()` | `mode = normal` + adds a transient **speaker prior** boosting `is_speaking` for ~10 s (so the user-visible "follow the speaker" behavior is even stronger) |
| `clear_focus()` | `mode = normal`, clears priors |

Under the hood there are only two modes; the speaker tool is sugar.

---

## 2. The scoring function

Every tick (target ~10 Hz), for each person in `WorldState.humans`:

```
score(p) =
    w_addressed * is_being_addressed(p)    # looking at us + speaking
  + w_speaking  * is_speaking(p)
  + w_new       * recency_bonus(p)         # decays over ~5 s
  + w_named     * is_known(p)
  + w_proximity * closeness(p)             # bbox area, normalized
  + w_center    * centrality(p)            # tiebreaker
  + w_prior     * thinker_prior(p)         # Hsafa / Gemini nudges, TTL
  - w_fatigue   * stare_duration(p)        # kick in after ~4 s
```

### 2.1 Default weights (tune from these)

| Signal | Weight | Notes |
|---|---|---|
| `is_being_addressed` | **10** | Face toward camera + speaking. Strongest. |
| `is_speaking` | **6** | Lips moving, ideally fused with audio VAD. |
| `is_new` (just appeared) | **4 → 0 over 5 s** | Natural orienting response. |
| `is_known` | **2** | Slight pull toward named faces. |
| `proximity` | **2** | Normalized bbox area. |
| `thinker_prior` | **+3 / TTL** | Soft nudges, e.g. from Hsafa. |
| `centrality` | **1** | Tiebreaker only. |
| `fatigue` | **−3 after 4 s** | Break the stare. |

### 2.2 Signal sources (all present or already planned)

| Signal | Source |
|---|---|
| `is_speaking` | `LipMotionTracker` today, fused with **Silero VAD** next, **TS-TalkNet** or **FabuLight-ASD** later (see `tech-recommendations.md`). |
| `is_being_addressed` | Head-pose yaw ≈ 0° wrt camera (MediaPipe Face Landmarker) **AND** `is_speaking`. |
| `is_new` | `first_seen_ts` per track, decaying bonus for 5 s. |
| `is_known` | `FaceRecognizer` match (`FaceMatch.name is not None`). |
| `proximity` | YOLO bbox area / frame area. |
| `centrality` | Distance of bbox center from image center, inverted. |
| `thinker_prior` | `gaze_prior` events on the `EventBus` (Hsafa or Gemini tools). |
| `fatigue` | `time.monotonic() - locked_since` while `track_id == current`. |

Off-camera voices become a **virtual candidate** pushed into the scorer
for ~1.5 s when DOA fires (see §5).

---

## 3. Hysteresis — stop the twitching

Pure argmax jitters between two people whenever scores are close. Two
rules fix it:

1. **Switch margin.** The candidate must beat the current target by
   ≥ `SWITCH_MARGIN = 2.0` to steal focus.
2. **Minimum dwell.** After committing to a target, stay at least
   `MIN_DWELL_MS = 800` before any switch is allowed, **unless** the
   challenger is currently `is_being_addressed` (which bypasses dwell —
   direct address always wins instantly).

Pseudocode:

```python
def tick(world: WorldState) -> Optional[int]:
    cands = [score_one(p, world) for p in world.humans]
    if not cands:
        self._idle()
        return None
    best = max(cands, key=lambda c: c.score)

    if self.current is None:
        return self._commit(best)

    if best.track_id == self.current.track_id:
        self._refresh_dwell()
        return self.current.track_id

    # Different candidate is on top.
    steal_allowed = (
        best.score >= self.current.score + SWITCH_MARGIN
        and (self._dwell_elapsed() or best.addressed)
    )
    if steal_allowed:
        return self._commit(best)

    return self.current.track_id
```

---

## 4. Internal states (for animation / idle / search)

Three states, inferred from the scored output, not user-selectable:

- **`ENGAGED`** — we have a locked target with non-trivial score
  (speaking or addressed). Head tracks them smoothly. Fatigue clock
  accumulates slowly.
- **`SCANNING`** — visible humans exist but nobody is speaking /
  addressing. Pick the top-scored person, hold 1.5–3 s, saccade to
  next best. This is the "attentive but not creepy" mode.
- **`IDLE`** — no humans visible for > 3 s. Slow sinusoidal drift
  (from `natural-gaze.md §5`), occasional micro-saccades, one saccade
  toward the last-known exit point of a recently-lost person.

`mode=person(name)` with the target visible forces `ENGAGED` on that
track. With the target *not* visible, fall through to the state the
scorer would pick (usually `SCANNING` or `IDLE`) — so the robot still
looks alive while waiting for Husam to walk back in.

---

## 5. Off-camera voices — virtual candidate

When we add DOA (GCC-PHAT on 2 mics, see `tech-recommendations.md`):

1. `sound_detected { azimuth, ts }` event arrives.
2. If no visible `WorldState.humans` centroid matches that azimuth
   within a tolerance, push a **virtual candidate**:
   ```
   VirtualCandidate {
     kind: "sound_source",
     azimuth,
     score_override = w_speaking,   # same as is_speaking
     ttl = 1.5 s,
   }
   ```
3. The policy treats it like a real candidate → head turns → YOLO
   picks up the person on arrival → they become a real candidate on
   the next tick, the virtual one expires.

This is how the robot "hears" a voice behind it and looks.

---

## 6. Thinker / Gemini priors

External brains never set the target. They publish priors:

```
GazePrior {
  track_id | name | azimuth | None   # who
  weight:    float                    # added to score
  ttl_s:     float                    # auto-expires
  reason:    str                      # for logs
}
```

Examples:

- Gemini's `focus_on_speaker()` → prior on whoever is currently
  `is_speaking`, weight +3, ttl 10 s. Doesn't lock — a newcomer
  speaking louder / being addressed can still preempt.
- Hsafa "Husam is probably here, ~09:00" → prior on `name=husam`,
  weight +1, ttl 60 s.
- Hsafa "unread message from Ahmad" → prior on door azimuth,
  weight +2, ttl 30 s.

Only `mode=person(name)` is a real lock. Everything else is soft.

---

## 7. Debugging — log every decision

Every tick, append a JSONL line:

```json
{
  "ts": 1713770012.341,
  "state": "ENGAGED",
  "mode": "normal",
  "locked_id": 17,
  "candidates": [
    {"id": 17, "name": "husam", "score": 12.3,
     "parts": {"addressed": 10, "speaking": 6, "fatigue": -3, "...": "..."}},
    {"id": 23, "name": null,   "score": 4.8, "parts": {...}}
  ],
  "priors": [{"name": "husam", "weight": 3, "ttl_s": 7.1}]
}
```

When the robot does something weird, replay the log to see exactly
which term won. This is maybe 10 lines of code and saves hours of
tuning guesswork.

---

## 8. Mapping onto the current code

### `hsafa_robot/focus.py`

Replace today's `set_mode_auto / set_mode_person / set_mode_speaker`
with:

```python
class GazeMode(Enum):
    NORMAL = "normal"
    PERSON = "person"   # locked to self.target_name

class FocusManager:
    def set_mode_normal(self): ...
    def set_mode_person(self, name: str): ...
    def add_prior(self, prior: GazePrior): ...

    # Internal: the scoring engine.
    self._policy = GazePolicy(weights=DEFAULT_WEIGHTS)
```

`set_mode_speaker()` stays on the Gemini tool surface but internally
does `set_mode_normal()` + `add_prior(speaker_prior)` so the public
API doesn't break.

### New `hsafa_robot/gaze_policy.py`

Pure function layer:

```python
@dataclass
class Candidate:
    track_id: int
    name: Optional[str]
    score: float
    parts: dict[str, float]
    addressed: bool

class GazePolicy:
    def __init__(self, weights: Weights): ...
    def score(self, world: WorldState,
              priors: list[GazePrior]) -> list[Candidate]: ...
    def pick(self, cands: list[Candidate],
             current: Optional[Candidate]) -> Optional[Candidate]: ...
```

No threading, no motor I/O — the policy is a pure
`(WorldState + priors + memory) → track_id` function. Easy to unit-test.

### `CascadeTracker` additions

Expose per-track `first_seen_ts`, bbox area, bbox center (most already
there via `get_all_tracks()`). That plus head-pose from MediaPipe is
all the scorer needs.

---

## 9. Build order (smallest wins first)

1. Introduce `GazePolicy` + `Candidate` + default weights; have
   `FocusManager` call it in **NORMAL** mode using only `is_speaking`
   + `is_known` + `proximity`. Ship. Robot already feels less
   deterministic.
2. Add `is_new` (recency bonus) and `fatigue`. The "glance at the
   newcomer, then return" behavior appears.
3. Add decision JSONL log (§7).
4. Rewire `mode=person` with the auto-fallback to NORMAL.
5. Fold `focus_on_speaker` into a prior.
6. Add head-pose signal via MediaPipe → `is_being_addressed`. Biggest
   felt upgrade for group conversation.
7. Add Silero VAD → fuse with lip motion for a real `speaking_prob`.
8. Add DOA + virtual sound candidate (§5) once stereo mic access is
   confirmed.
9. Subscribe `GazePrior` events from Hsafa Core (`gaze_prior` on the
   `EventBus`).

Each step is independently shippable; none requires the next.

---

## 10. Why this shape

- **One real mode.** The scorer is the engine. `person(name)` is
  just "hard override when visible, fall through when not." No
  branching hell.
- **Soft priors from both brains.** Gemini and Hsafa nudge, they
  don't command. Feels like a mind, not a remote-controlled head.
- **Every new sense plugs in as a signal.** DOA, emotion, gesture,
  gaze direction — each adds one term to `score()`, no rewrites.
- **Deterministic + debuggable.** Scores are numbers, weights are
  numbers, logs are JSONL. You can replay any decision.
