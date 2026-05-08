# Plan: Haseef + Gemini Live as One Entity

> Based on `docs/unified-brain.md` and `docs/hsafa-integration.md`. This is the implementation plan for making Haseef (Hsafa Core) and Gemini Live act as a single mind.

---

## 1. Philosophy: One Mind, Two Surfaces

- **Gemini Live** = the *voice surface*. Fast, real-time, no persistent memory. Only job: listen, talk, and use a minimal tool set for immediate physical actions.
- **Haseef (Hsafa Core)** = the *self / memory*. Slow (1-3s per turn), persistent across reboots. Decides who to look at, remembers people, plans proactive speech, controls body expressions.

**The Golden Rule:** Neither brain holds private state the other cannot see. Haseef's memory is the single source of truth; Gemini gets a read-only `voice_projection` refreshed on every session boundary.

---

## 2. What Changes in `main.py`

### Replaced
| Current | Replacement |
|---|---|
| `DEFAULT_SYSTEM_INSTRUCTION` (giant inline string) | `persona.md` loaded once, injected into both Gemini and Haseef prompts |
| Gemini owns all tool definitions (`build_test_tools`, `build_face_tools`) | **Only** voice-relevant tools stay with Gemini; all physical / memory tools move to Haseef |
| Gemini directly calls `set_head_angle`, `look_at`, `focus_on_person` | Gemini fires async `queue_thinker_task()` or Haseef pushes instructions via `inject_speech()` |
| `make_tool_handler()` in `main.py` (250+ lines) | Split: Gemini handler (thin) + Haseef handler (in `hsafa_robot/hsafa_bridge.py`) |

### Kept
- `GeminiLiveSession` class (with added `inject_client_message()`)
- Camera → tracker → `WorldState` pipeline
- `RobotController`, `FocusManager`, all L0-L2 reflexes
- `EventBus` (already the glue)

---

## 3. Haseef's Skills & Tools (Complete List)

Haseef runs as **one agent** with **three skill scopes** registered from the robot process via `hsafa-sdk`.

### Skill: `robot_body`
> Controls physical embodiment. Gemini NEVER touches motors directly.

| Tool | Description | Example Trigger |
|---|---|---|
| `move_head(yaw_deg, pitch_deg, hold_s?)` | Set head angle. Auto-resume face-follow after `hold_s`. | Haseef wants to search for someone |
| `look_at(description)` | Use CV to locate object, align head, return bbox. | User: "look at the cup" → Haseef decides → calls tool |
| `set_gaze_mode(mode, name?)` | `normal` / `person(name)` / `speaker` / `idle` | Haseef policy: "focus on Husam, he's the host" |
| `play_expression(name, intensity?)` | `happy`, `curious`, `surprised`, `sleepy`, `confused` | Haseef wants to react non-verbally |
| `wave()` | Quick antenna / hand wave | Greeting someone after absence |
| `get_body_state()` | Returns current yaw/pitch/gaze_mode/speaking_flag | Haseef planning its next move |
| `enable_face_follow()` | Resume default face tracking | After a deliberate look-around |
| `disable_face_follow()` | Freeze head at current angle | During a precise inspection |

### Skill: `robot_vision`
> Already partially built in `hsafa_voice_vision.py`. Expanded.

| Tool | Description | Notes |
|---|---|---|
| `enroll_face(name, who?, position?)` | Remember a face | Already works in `hsafa_voice_vision.py` |
| `identify_person()` | List all visible people with positions | Returns `left/center/right` + names |
| `find_person(name)` | Check if a known person is visible now | Returns position or `not_visible` |
| `who_is_speaking()` | Name of person whose mouth is moving | Uses `LipMotionTracker` |
| `describe_scene()` | Compact world snapshot | Who's visible, speaking, gestures, robot pose |
| `detect_gestures()` | List current gestures per person | Wave, point, thumbs_up, etc. |
| `capture_image()` | Return current camera frame as base64 | For Haseef to "look" when not in a voice turn |

### Skill: `robot_voice`
> The bridge between Haseef and Gemini Live.

| Tool | Description | Who Calls |
|---|---|---|
| `say_this(text, urgency?)` | **INTERNAL** — Haseef injects a message into Gemini Live. Gemini paraphrases it in its own voice. | Haseef only |
| `queue_thinker_task(task, what_i_told_user?)` | **Gemini tool** — FAST non-blocking. Queues a task for Haseef; returns immediately. Haseef answers later via `say_this()`. | Gemini only |
| `remember_fact(text, category?)` | Store a semantic memory in Haseef | Gemini calls this when user says "remember that I'm allergic to cats" |

### Future Skills (Phase 2+)
- `web_search(query)` — Haseef runs search, later injects result via `say_this`
- `schedule_reminder(when, text)` — Haseef plans, later fires `say_this`
- `send_message(channel, recipient, text)` — WhatsApp/Discord via Hsafa

---

## 4. Gemini Live's Tool Set

**No duplicate tools.** Every capability lives in exactly one place.

| Tool | Owner | Gemini Function Decl? | Why |
|---|---|---|---|
| `move_head` | Haseef (`robot_body`) | ❌ | Haseef decides contextually |
| `look_at` | Haseef (`robot_body`) | ❌ | Needs persistent memory |
| `set_gaze_mode` | Haseef (`robot_body`) | ❌ | Haseef knows who is host |
| `enroll_face` | Haseef (`robot_vision`) | ❌ | Writes to social memory |
| `identify_person` | Haseef (`robot_vision`) | ❌ | Needs face DB + social context |
| `describe_scene` | Haseef (`robot_vision`) | ❌ | Full world state |
| `say_this` | Haseef (`robot_voice`) | ❌ | Internal bridge only |
| `web_search` | Haseef (future) | ❌ | Slow, stateful |
| `queue_thinker_task` | **Gemini** | ✅ | The **only** bridge upward |
| `remember_fact` | **Gemini** | ✅ | Fast semantic store |
| `get_current_time` | **Gemini** | ✅ | Instant, no context |
| `get_robot_status` | **Gemini** | ✅ | Diagnostic |
| `ping` | **Gemini** | ✅ | Health check |

### How Gemini Knows What Haseef Can Do

Gemini's system instruction includes a **capabilities catalog** — a plain-text list of every Haseef tool with descriptions. This is **not** a function declaration; Gemini cannot *call* these tools directly. It knows they exist so it can:

1. **Say "ok" or "I can't" honestly.** When the user asks "look at the cup," Gemini sees `look_at` in its capability catalog and knows Haseef can handle it.
2. **Queue the right task.** Gemini calls `queue_thinker_task(task="User wants to look at the cup. They said: 'look at the red cup on the desk'")` — natural language, not tool schema.
3. **Haseef decides the actual tool.** Haseef receives the natural-language task, interprets it in full context, and calls the real `look_at(description="red cup on the desk")` tool.

**Example:**
```
User: "Look at the cup."

Gemini (reads capability catalog):
  → "look_at(description) — Haseef can do this"

Gemini (calls its OWN tool):
  → queue_thinker_task(
       task="User asked to look at the cup. Description: 'the cup'",
       what_i_told_user="Sure, looking."
     )
  → returns instantly

Gemini (speaks): "Sure, looking."

[Haseef receives the task]
→ interprets: user wants visual attention on a cup
→ calls real tool: look_at(description="the cup")
→ head moves, camera captures, result comes back

[Haseef may or may not inject follow-up via say_this]
```

**No blocking.** `queue_thinker_task` returns in <10ms. The audio stream never stalls.

---

## 5. The Two Speech Triggers

```
User speaks ──► Gemini Live hears ──► Gemini replies (direct, fast)
                    │
                    └──► Haseef observes (async, no blocking)

Haseef decides ──► calls `say_this(text, urgency)`
                         │
                         └──► queued → injected into Gemini Live
                              └──► Gemini paraphrases & speaks
```

### Trigger 1: User → Gemini (Direct Path)
- **No Haseef involvement in the realtime loop.**
- Gemini sees the user, hears them, replies instantly.
- After each `turn_complete`, we push two events to Haseef:
  - `user_said` — what the user spoke (from STT transcript)
  - `robot_said` — what Gemini replied (captured from the turn)
- Haseef updates its memory asynchronously. No blocking.

### Trigger 2: Haseef → Gemini (Injection Path)
- Haseef calls `say_this(text, urgency)` via its own skill tool.
- `GeminiLiveSession` has an internal queue with urgency rules:
  - `urgency="normal"` → queue until Gemini is idle (not speaking, not hearing user)
  - `urgency="high"` → interrupt current Gemini turn, barge in
  - `urgency="idle"` → drop if user is speaking or if >30s silence
- Injection is done via `session.send_client_content()` with a system framing prefix:
  > *(inner thought) You just remembered this — share it briefly and naturally: ...*
- Gemini paraphrases. **Never** hardcode exact spoken strings from Haseef.

### Trigger 3: Gemini → Haseef (Async Handoff Path)
When Gemini needs help (facts, memory, identity), it **cannot block.**

```
User: "What's the capital of Botswana?"

Gemini: "Hmm, I'm not sure off the top of my head. Let me check."
     → calls `queue_thinker_task(
           task="What is the capital of Botswana?",
           what_i_told_user="Hmm, I'm not sure. Let me check."
         )`
     → returns IMMEDIATELY: {"status": "queued"}

Gemini continues: (turn ends, user can speak again)

[Haseef receives the event]
→ Haseef reads: user asked about Botswana, Gemini already said "let me check"
→ Haseef runs: searches memory / web / whatever
→ Haseef calls `say_this("The capital of Botswana is Gaborone.", urgency="normal")`

[Gemini receives injection]
→ Gemini speaks: "The capital of Botswana is Gaborone, by the way."
```

**Key:** `what_i_told_user` is passed by Gemini itself. Haseef knows exactly what Gemini already said, so it won't repeat "let me check" — it jumps straight to the answer.

---

## 6. Memory Sharing (No Contradictions)

### Shared Artifacts

| Artifact | Owner | Format | Updated When |
|---|---|---|---|
| Hsafa semantic memory (`sdk.memory.set`) | Both write, Haseef reads natively | Key/value in Core | Every fact, preference, relationship |
| Hsafa episodic memory (`sdk.memory.episodes`) | Both write via events | Run summaries in Core | After every Haseef run; Gemini pushes summaries after turns |
| `docs/persona.md` | Human (we edit it) | Markdown | Rarely — this is the character bible |
| `voice_projection` (in-memory) | Haseef generates, Gemini reads | ≤1k tokens | Every Gemini session start / every 10-min rollover |

### The Voice Projection

On every Gemini Live session start (and every ~10-minute rollover), Haseef generates a compact prompt fragment injected into Gemini's system instruction:

```
CURRENT CONTEXT (from Hsafa)
- You are looking at: Husam (center), Ahmad (right)
- Husam: roboticist, prefers English, last chatted 3 days ago about project X
- Ahmad: first meeting today
- Current intention: idle / observing / greeting
- Recent events: [last 3-5 from memory]
- Agreements: none / "remind Husam about meeting at 5pm"
- Pending tasks: [any queued thinker tasks not yet answered]
```

This is the **only** way Gemini knows facts about people. It does not hallucinate memories.

### How Gemini Asks Haseef for Help (Non-Blocking)

Gemini **never blocks** waiting for Haseef. When it needs a fact, it has two options:

1. **The Voice Projection already has it** → use it directly.
2. **Not in the projection** → call `queue_thinker_task(task, what_i_told_user)`:
   - Returns **instantly**: `{"status": "queued", "task_id": "..."}`
   - Gemini should say something natural like "Let me think about that" or "I'm not sure, let me check"
   - `what_i_told_user` tells Haseef what Gemini already said, so Haseef doesn't repeat it
   - Haseef processes the task asynchronously and injects the answer via `say_this()` on the next available turn

This means the user hears a natural conversational filler immediately, then the real answer arrives 2-5 seconds later. No awkward silence. No broken audio stream.

---

## 7. Communication Flows (Examples)

### Example A: User greets robot
```
User: "Hi Hsafa!"

[Direct path — no Haseef blocking]
Gemini Live: "Hey! Good to see you."

[Async — after turn_complete]
→ push `user_said` event to Haseef: "Hi Hsafa!"
→ push `robot_said` event to Haseef: "Hey! Good to see you."
→ Haseef updates episodic memory: "casual greeting, neutral tone"
```

### Example B: User asks "who am I?"
```
User: "Who am I?"

Gemini Live: (checks voice projection)
→ projection says: "Husam is visible, center frame"
→ replies directly: "You're Husam! We talked about your robot project last time."

[Async — after turn_complete]
→ push `user_said` event to Haseef: "Who am I?"
→ push `robot_said` event to Haseef: "You're Husam! We talked about your robot project last time."
→ Haseef records: "Husam asked for identity confirmation"
```

### Example C: Haseef initiates speech
```
[Face recognizer detects Husam after 2 days absence]
→ EventBus: `person_detected(name="Husam", after_days=2)`

Haseef receives event:
→ loads social memory: "Husam, roboticist, project X"
→ decides: warm greeting appropriate
→ calls `say_this("Husam is back after 2 days. Greet him warmly and ask about project X.", urgency="normal")`

Gemini Live queue:
→ waits for current turn to finish (if any)
→ injects: "(inner thought) Husam just came back after 2 days..."

Gemini Live speaks: "Husam! Hey, it's been a couple of days. How's the robot coming along?"
```

### Example D: Enrollment
```
User: "Remember me, I'm Husam."

Gemini Live: "Got it!" (fast, no need to wait)
→ calls `remember_fact("User identifies as Husam")` (Gemini's own fast tool)

[Async — after turn_complete]
→ push `user_said` event to Haseef: "Remember me, I'm Husam."
→ push `robot_said` event to Haseef: "Got it!"

Haseef (async, next run):
→ sees enrollment intent from `user_said` event
→ calls `enroll_face(name="Husam")` (vision tool)
→ writes social memory: "Husam, first met [date], self-introduced"

[Next session]
Voice projection includes: "Husam (known, roboticist)"
Gemini greets by name naturally.
```

---

## 8. Implementation Phases

### Phase 0: Foundation (this plan)
- Create `docs/persona.md` (single source of truth for character)
- Confirm `hsafa_voice_vision.py` SDK connection works (✓ already done)

### Phase 1: The Bridge
- Create `hsafa_robot/hsafa_bridge.py`:
  - One `HsafaSDK` client connecting 3 skills (`robot_body`, `robot_vision`, `robot_voice`)
  - Tool handlers that call into existing robot modules
  - Event publishers that read `WorldState` and push to Haseef
- Create `hsafa_robot/voice_projection.py`:
  - Queries Haseef memory → generates ≤1k token prompt fragment
- Add `GeminiLiveSession.inject_client_message(text, interrupt_ok=False)`
- Add `GeminiLiveSession` internal speech queue with urgency rules

### Phase 2: Migrate Tools
- Remove physical tools from Gemini's `build_test_tools()` / `build_face_tools()`
- Add `queue_thinker_task()` as the **only** Gemini tool that talks upward
- Register all body/vision/voice tools with Haseef via `hsafa_bridge.py`
- Update `DEFAULT_SYSTEM_INSTRUCTION` to reference `persona.md` and explain the `queue_thinker_task` bridge

### Phase 3: Shared Memory
- Wire `user_said` / `robot_said` events into Haseef memory via `sdk.memory.set`
- Every `turn_complete` → push `robot_said` event to Haseef
- Every user speech finalization → push `user_said` event to Haseef
- Every tool call (both Gemini and Haseef) → push `tool_called` event to Haseef memory
- On Gemini session start, query Haseef for voice projection and inject it

### Phase 4: Proactive Haseef
- Haseef `say_this` events start firing:
  - Person detected after absence → greeting prompt
  - Scheduled reminder → reminder prompt
  - Background search completes → result prompt
- Urgency queue in `GeminiLiveSession` handles turn-taking

### Phase 5: Future Powers
- `web_search` skill for Haseef
- `scheduler` skill for Haseef
- Cross-channel memory (WhatsApp, chat, in-room all share Atlas)

---

## 9. Files to Create / Modify

### New Files
| File | Purpose |
|---|---|
| `docs/persona.md` | Single character bible |
| `hsafa_robot/voice_projection.py` | Haseef memory → Gemini prompt fragment |
| `hsafa_robot/hsafa_bridge.py` | SDK client, tool handlers, event publishers |
| `hsafa_robot/speech_queue.py` | Urgency-aware queue for `say_this` injection |

### Modified Files
| File | Change |
|---|---|
| `main.py` | Remove most tool defs from Gemini; add Haseef bridge init; inject voice projection on session start |
| `hsafa_robot/gemini_live.py` | Add `inject_client_message()`, speech queue consumer, `queue_thinker_task` handler |
| `hsafa_voice_vision.py` | Absorbed into `hsafa_bridge.py` or kept as the `robot_vision` skill entry point |

---

## 10. Summary: Who Does What

| Task | Gemini Live | Haseef |
|---|---|---|
| Listen to user | ✅ Native | Observes via `user_said` / `robot_said` events |
| Reply to user chitchat | ✅ Native | Can inject via `say_this` |
| Move head / body | ❌ Removed | ✅ `robot_body` tools |
| Face enroll / identify | ❌ Removed | ✅ `robot_vision` tools |
| Remember facts | ❌ Removed | ✅ Semantic memory |
| Know person's history | ❌ Removed | ✅ Social + episodic memory |
| Decide who to look at | ❌ Removed | ✅ Gaze policy + `set_gaze_mode` |
| Proactive speech | ❌ Removed | ✅ `say_this` injection |
| Reactive reflexes (glance at wave) | ✅ Fast loop in `main.py` | Haseef is too slow; keep L2 reflexes |
| Ask for help when stuck | ✅ `queue_thinker_task()` (non-blocking) | Receives and answers via `say_this()` |

**Result:** Two brains, one voice, one body, one memory. The user experiences a single consistent entity.
