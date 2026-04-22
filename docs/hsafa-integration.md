# Hsafa Integration — Simple Plan

This is the plain-English plan for plugging this robot into Hsafa Core.
No code yet — just the shape of the system and what lives where.

## The Two Brains

We keep what we already have and add Hsafa on top. Two brains, one robot:

- **Fast brain — Gemini Live** (already built)
  - Real-time voice in/out, sees the JPEG stream, handles the live chat.
  - Sub-second latency. Forgets everything when the session ends.
  - Good at: chitchat, reflexes, "say hi back."
- **Slow brain — Hsafa Core** (new)
  - Lives on a server. Thinks in 1–3 s bursts.
  - Has persistent memory across every restart.
  - Good at: "who is this person, what do I know about them, what should I do."

Gemini Live stays the voice. Hsafa becomes the *self* that remembers and decides.

## The Haseef

One Haseef (call it **Atlas**) represents the robot.

```json
{
  "name": "Atlas",
  "profile": { "robotId": "reachy-01" },
  "skills": ["body", "vision", "voice"]
}
```

Profile is also the routing table — any event tagged `target.robotId = "reachy-01"`
lands on Atlas.

## Three Scopes, One Process

All three Hsafa scopes run inside the same robot process (this repo).
They're just three SDK clients sharing the existing `CascadeTracker`,
camera, mics, and motors.

### `body` scope
- **Tools:** `look_at`, `set_tracking_target`, `set_gaze_mode`,
  `wave`, `play_animation`, `get_position`.
- **Events:** `battery_low`, `bump_detected`.

### `vision` scope
- **Tools:** `enroll_face(name)`, `find_person(name)`, `describe_scene`,
  `capture_image`.
- **Events:**
  - `person_detected` → `{ name, track_id, direction, distance }`
  - `person_left` → `{ name, track_id }`
  - `gesture_detected` → `{ gesture, track_id }`

### `voice` scope
- **Tools:** `speak(text)` (only for when Hsafa wants to talk *without*
  going through Gemini Live), `set_volume`.
- **Events:**
  - `speech_heard` → `{ text, speaker, track_id }`
    (pushed when Gemini Live's STT or the speaker-ID module emits a
    finalized utterance worth remembering).

## Who Handles What

| Situation | Goes to |
|---|---|
| User says "hi" and robot replies | Gemini Live (fast) |
| User waves, robot waves back | Gemini Live (fast) |
| Face recognizer sees Husam | Hsafa (push `person_detected`) |
| "Remember I'm allergic to cats" | Hsafa (semantic memory) |
| "Who was I talking to yesterday?" | Hsafa (episodic + social memory) |
| "Enroll my face as Husam" | Hsafa decides → calls `enroll_face` tool on vision scope |
| Deciding who to stare at in a crowd | Hsafa (slow policy) → `set_gaze_mode` |
| Snapping the head to a sudden movement | Fast loop in this repo (no Hsafa) |

## The Enrollment Flow (the thing you asked about)

You sit in front of the robot. You say *"I am Husam, remember me."*

1. Gemini Live hears it, speaks back naturally, and emits a
   `speech_heard` event to Hsafa.
2. Hsafa's invoker runs: "A new person is introducing themselves."
3. Hsafa calls the `enroll_face("Husam")` tool on the vision scope.
4. Vision scope (this repo) captures ~10 frames, computes embeddings,
   saves them to disk under `husam`.
5. Hsafa also writes a social memory: *Husam, first met [date],
   introduced himself verbally.*
6. Next time you walk in, the face recognizer fires
   → vision pushes `person_detected: Husam`
   → Hsafa loads everything it knows about you
   → injects "You are now looking at Husam. Last chat was about X."
      into Gemini Live's context at the next session boundary.
7. Gemini Live greets you by name with the full backstory loaded.

No separate enrollment script. The robot learns people the way humans do:
by meeting them.

## Memory Split

- **Gemini Live session context** — only the current conversation.
  Session resumption keeps it alive across Gemini's 10-minute reconnects,
  but not across restarts.
- **Hsafa social memory** — per-person. "Husam prefers English,
  is learning robotics, said last Tuesday that…"
- **Hsafa episodic memory** — per run. "On April 22 at 10am, Husam
  asked about face recognition and I enrolled him."
- **Face embedding store** — on the robot's disk. Maps face vectors → names.
  This is the *only* thing that must live on the robot itself; everything
  else can live in Core.

## Wiring Plan (high level)

1. Add a thin `hsafa_robot/hsafa_client.py` that wraps the SDK's 3 methods
   and spins up the three scopes.
2. Add a `WorldState` object (see `gaze-modes.md`) that the vision thread
   updates and the Hsafa bridge reads.
3. On every `WorldState` change, diff and push relevant events
   (`person_detected`, `person_left`, `speech_heard`).
4. Expose `set_gaze_mode`, `set_tracking_target`, `enroll_face` as tools
   Hsafa can call back into the robot.
5. On Gemini Live session start, query Hsafa for the current social
   snapshot and inject it into the system prompt.

## Why This Is Worth Doing Early

Even before any fancy features, Hsafa gives us:

- **Persistent identity.** The robot stops being a stranger every reboot.
- **Cross-channel continuity.** Same Atlas on WhatsApp, in a chat,
  and in the room.
- **A clean debug surface.** Every run is a stored record with
  memory snapshots and tool calls — way easier than reading Gemini logs.

Start small: memory + enrollment. Move slow decisions (gaze policy,
proactive behavior) into Hsafa once the basics work.
