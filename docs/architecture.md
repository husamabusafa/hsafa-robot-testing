# Architecture — The Foundation

How all the services connect. The goal is a spine strong enough that
adding a new sense, a new skill, or a new brain is *one file and one
subscription*, never a rewrite.

---

## 1. The layers (from reflex to reflection)

```
┌──────────────────────────────────────────────────────────────┐
│  L4  Thinker            Hsafa Core (slow brain, server)      │
│      - intentions, memory, social context, planning          │
├──────────────────────────────────────────────────────────────┤
│  L3  Voice / persona    Gemini Live (fast brain, on-device)  │
│      - listens, talks, reflexive function calls              │
├──────────────────────────────────────────────────────────────┤
│  L2  Cognition glue     FocusManager, IdentityGraph,         │
│                         GazePolicy, EventBus                 │
├──────────────────────────────────────────────────────────────┤
│  L1  Perception         CascadeTracker, FaceRecognizer,      │
│                         LipMotion, VoiceRecognizer (future)  │
├──────────────────────────────────────────────────────────────┤
│  L0  Hardware I/O       camera, mic, speaker, motors         │
└──────────────────────────────────────────────────────────────┘
```

**Rule:** a layer may only call *down* or publish events. It never
reaches up. L4 doesn't poll motors; it asks L2 to aim. L1 doesn't
know Gemini exists; it writes to the `WorldState` and emits events.

---

## 2. The two contracts everything shares

Only two data shapes are allowed to cross layer boundaries:

### 2a. `WorldState` (read-mostly snapshot)

See `gaze-modes.md` for the current fields. The invariant:
**one writer per field, many readers under a lock.** New senses add
fields; no field is owned by two modules.

### 2b. `Event` (push, fire-and-forget)

```
Event {
  kind: "person_detected" | "person_left" | "speech_heard"
      | "face_enrolled" | "gaze_target_changed" | "say_this"
      | ...
  ts: float                  # monotonic seconds
  source: "vision" | "voice" | "focus" | "brain"
  payload: { ... }           # kind-specific
}
```

Every subsystem publishes to one `EventBus` and subscribes to the
kinds it cares about. No direct method calls between L1/L2/L3/L4.

This is what makes "add a feature" cheap. Example: adding gesture
recognition =

1. Write a `GestureTracker` in L1 that publishes `gesture_detected`.
2. Subscribe `FocusManager` to decide if it changes gaze.
3. Subscribe `GeminiLive` context injector to mention it in the prompt.
4. Subscribe `HsafaBridge` to remember it.

Zero existing files change.

---

## 3. Current code mapped to the layers

| File | Layer | Notes |
|---|---|---|
| `hsafa_robot/tracker.py` | L1 | YOLO + ByteTrack. Owns body track_ids. |
| `hsafa_robot/face_recognizer.py` | L1 | Faces → names. |
| `hsafa_robot/lip_motion.py` | L1 | Who's mouth is moving. |
| `hsafa_robot/voice_recognizer.py` | L1 (stub) | Will become voice-embedding → name. |
| `hsafa_robot/focus.py` | L2 | Reads tracker+lip snap, writes target. Already the seed of L2. |
| `hsafa_robot/robot_control.py` | L0/L1 | P-controller pushing motors. |
| `hsafa_robot/animation.py` | L0 | Idle / talk animation overlays. |
| `hsafa_robot/gemini_live.py` | L3 | Voice brain, owns the tool surface. |
| `main.py` | app entry | Wires everything. Should shrink over time. |
| *(future)* `world_state.py` | L2 | Single source of truth snapshot. |
| *(future)* `event_bus.py` | L2 | In-process pub/sub. |
| *(future)* `identity_graph.py` | L2 | Face ↔ voice ↔ name links. |
| *(future)* `gaze_policy.py` | L2 | Natural gaze behavior. |
| *(future)* `hsafa_bridge.py` | L4 boundary | Pushes events to Hsafa, receives tool calls. |

---

## 4. Refactor targets (in priority order)

These are *small* steps; each one leaves the robot in a working state.

1. **Extract `WorldState`.** Pull today's scattered snapshots
   (`tracker.get_all_tracks`, `lip_tracker.snapshot`, face recognizer
   results) into one object built once per tick.
2. **Extract `EventBus`.** Even if it only has 3 subscribers at
   first, the second a tool handler needs to talk to the main loop
   the bus pays for itself. Today the tool handler mutates
   `FocusManager` directly — that's the first pattern to break.
3. **Move tool declarations out of `main.py`.** The Gemini function
   schemas live next to the handler that fulfils them, in a
   `tools/` package, grouped by skill (`tools/face.py`,
   `tools/gaze.py`, `tools/memory.py`). `main.py` just aggregates.
4. **Normalize handler errors.** Every tool returns the same envelope:
   `{ok, data?, reason?, hint?}`. Gemini gets predictable feedback.
5. **Introduce `IdentityGraph`.** Replace `FaceDB`-only lookups with
   a store that knows face *and* voice embeddings link to the same
   identity. See `identity.md`.
6. **Introduce `GazePolicy`.** `FocusManager` becomes a thin
   "apply the gaze policy's pick" layer. The policy itself is
   pluggable (rule-based now, learned later). See `natural-gaze.md`.
7. **Plug `HsafaBridge`.** Publish events, accept tool calls, inject
   social snapshot into Gemini's system prompt at session start.
   See `unified-brain.md` and `hsafa-integration.md`.

---

## 5. What we deliberately keep simple

- **In-process only.** No IPC, no ZMQ, no Redis. All of L0–L3 runs
  in one Python process. If a thread is enough, don't reach for a
  subprocess. L4 is the only thing that goes over the network.
- **Threads, not asyncio everywhere.** Vision threads stay classic
  `threading.Thread`; only Gemini's side is async. The event bus
  hands off via a thread-safe queue if a subscriber lives in asyncio.
- **No dependency injection framework.** `main.py` is the composition
  root. Anyone who wants a dep gets it passed in the constructor.
- **No config files until we need them.** CLI flags + env vars are
  fine. Migrate to a real config only once we have >1 deployment.

---

## 6. Failure modes this design prevents

- *"The face recognizer says Husam but the tracker is pointing at
  Ahmad."* → Impossible once both write into the same `WorldState`
  and the gaze policy reads the joined view.
- *"Who calls `set_locked_id`?"* → Only `FocusManager`. Every other
  module asks the focus layer, no one writes motors directly.
- *"Adding barge-in broke gesture detection."* → Can't happen; they
  don't share any wire, they share the bus.
- *"The Gemini handler forgot to release the focus lock on error."*
  → Handlers emit an `intent` event; the focus manager owns the
  state machine and handles all transitions, including errors.
