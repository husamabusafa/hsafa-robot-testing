# Gaze Modes & World State — Simple Design

How Reachy decides **who to look at**, and how Gemini Live (or Hsafa later)
switches between modes. Plain-English design doc, no code yet.

---

## 1. The World State

Everything starts with a single shared object that describes
*what the robot currently perceives in the room*. One source of truth,
updated by the vision thread, read by everyone else.

### Shape (today: humans only)

```
WorldState {
  humans: [
    {
      track_id: 17,               # from ByteTrack, stable across frames
      name: "husam" | null,       # set once face recognizer matches
      bbox: (x1, y1, x2, y2),
      center_px: (x, y),
      direction: "left" | "center" | "right",
      distance_est: "near" | "mid" | "far",
      is_speaking: false,         # updated by lip-motion / mic module
      last_seen: 1713770000.12,
      first_seen: 1713769980.03,
    },
    ...
  ],
  active_speaker_track_id: 17 | null,
  last_update: 1713770000.15,
}
```

### Room for growth (filled in later)

```
WorldState {
  humans: [...],
  objects: [                       # future — cat, pen, cup via YOLO-World etc.
    { class: "cat", track_id: 3, direction: "left" },
  ],
  robot: {                         # future — self-awareness
    head_yaw: 12.3,
    head_pitch: -5.0,
    posture: "standing",
    battery: 0.87,
    current_target: { kind: "human", track_id: 17 },
  },
  environment: {                   # future
    noise_level: 0.21,
    lighting: "normal",
  },
}
```

**Design rule:** anything a future brain might want to reason about
gets a slot here. Hsafa reads this snapshot to decide what to do;
Gemini Live can get a compressed text version injected into its
system prompt every few seconds.

### Who owns it

- The vision thread (this repo's `CascadeTracker` + face recognizer)
  is the only writer.
- Everyone else reads a copy under a lock.
- A diffing layer pushes Hsafa events when important fields change
  (new person, person gone, active speaker switched, name resolved).

---

## 2. Gaze Modes

Instead of one hard-coded "look at the biggest person" rule, the
tracker runs in one of several **modes**. A single function
`set_gaze_mode(mode, args)` is the only way to change it.
Gemini Live (and later Hsafa) calls it via a tool.

### Mode: `largest` (default, current behavior)
- Look at whichever human has the biggest bounding box.
- The dumb-but-reasonable default when the robot has no other info.

### Mode: `person`
- Args: `name` (e.g. `"husam"`).
- Lock onto the track whose `name` matches.
- If that person isn't currently visible, fall back to `largest`
  **and** emit a `target_lost` event so the brain can react.
- When they reappear, re-lock automatically.

### Mode: `speaker`
- Lock onto whichever human has `is_speaking == true`.
- If no one is speaking, hold the last speaker for ~2 seconds
  (natural, non-jittery), then fall back to `largest`.
- Source of truth for `is_speaking`:
  - v1: mouth-region variance over the last ~500 ms (CPU-cheap).
  - v2: mic-array direction of arrival.
  - v3: voice embedding → matches name.

### Mode: `free` / `idle`
- No lock. Head does a slow natural sweep.
- Used when the robot is alone or during long silences.

### Mode: `manual`
- Args: `yaw`, `pitch`.
- Hard override — the brain is pointing the head itself.
- Releases automatically after a few seconds of no further commands.

### Mode: `policy` (future)
- No fixed target. A small scoring function picks every few ticks:
  - speaking? +big
  - recognized? +medium
  - just appeared? +small
  - stared too long? −penalty
- Highest score wins, with hysteresis to avoid jitter.
- This is where the slow brain (Hsafa) lives later.

---

## 3. How Gemini Live Switches Modes

Expose one tool to Gemini Live via function calling:

```
set_gaze_mode(mode: "largest" | "person" | "speaker" | "free" | "manual",
              name?: string,
              yaw?: number, pitch?: number)
```

Examples of natural triggers:

| User says | Gemini calls |
|---|---|
| "Look at me" (and you're Husam) | `set_gaze_mode("person", name="husam")` |
| "Pay attention to whoever is talking" | `set_gaze_mode("speaker")` |
| "Stop staring, just chill" | `set_gaze_mode("free")` |
| "Look to the left" | `set_gaze_mode("manual", yaw=-25, pitch=0)` |
| "Go back to normal" | `set_gaze_mode("largest")` |

Gemini Live also receives a compact summary of the `WorldState`
every few seconds as a system-message update, so it knows who
is actually in the room and can pick a valid name.

Example inject (every ~3 s while people are present):

```
[world] humans: husam (left, speaking), ahmad (right);
        current_gaze: person=husam
```

This is enough for Gemini to say things like
*"Ahmad, did you want to ask something?"* and call
`set_gaze_mode("person", name="ahmad")` in the same turn.

---

## 4. The Control Flow

```
┌──────────────────────┐
│ Camera + CascadeTracker│  writes →  WorldState
│ + Face Recognizer     │
│ + Lip-motion speaker  │
└──────────┬────────────┘
           │ reads
           ▼
┌───────────────────────┐     ┌────────────────────────┐
│ Gaze Controller        │◄───│ set_gaze_mode(...)      │
│ (picks target from     │     │  ← Gemini Live tool    │
│  WorldState given the  │     │  ← Hsafa tool (later)   │
│  current mode)         │     └────────────────────────┘
└──────────┬────────────┘
           │ target pixel / angle
           ▼
┌──────────────────────┐
│ Head P-controller     │  (existing motor code)
└──────────────────────┘
```

The gaze controller is the *only* thing that turns "mode + WorldState"
into "which track to point at right now." Everything upstream just
observes; everything downstream just moves motors.

---

## 5. Why This Shape

- **World State first.** If every module reads the same snapshot,
  we never get the "face recognizer disagrees with tracker"
  class of bug. Adding a new sense = adding a new field.
- **Modes, not if-statements.** Gaze logic grows *huge* if you
  branch inside the tracker. Modes keep it flat and testable.
- **One tool, one surface.** Gemini Live only ever calls
  `set_gaze_mode`. It never picks track IDs or pixel coordinates —
  that keeps the brain's job small and the robot's job robust.
- **Ready for Hsafa.** When we plug Hsafa in, it just calls the
  same `set_gaze_mode` tool. No rewiring.

---

## 6. Minimal First Build (suggested order)

1. Add `WorldState` dataclass + a lock-protected holder.
2. Make `CascadeTracker` populate `WorldState.humans` each tick
   (it already has bboxes + track IDs; just expose them).
3. Implement `set_gaze_mode("largest" | "person" | "manual")`
   and route head control through it.
4. Add lip-motion `is_speaking` per track → unlock `"speaker"` mode.
5. Register `set_gaze_mode` as a Gemini Live tool.
6. Add face recognizer → fills `humans[i].name`.
7. Later: push `WorldState` diffs as Hsafa events;
   add `"policy"` mode driven by Hsafa.
