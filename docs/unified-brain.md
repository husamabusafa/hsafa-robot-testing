# Unified Brain — One Entity Across Voice and Thinker

Reachy has two LLMs in its loop:

- **Gemini Live** — the fast voice. Sub-second, no persistent memory,
  knows how to *talk*.
- **Hsafa Haseef** — the slow thinker. Seconds per turn, persistent
  memory, knows what to *mean*.

The failure mode we must prevent: the voice saying something the
thinker didn't endorse, or the thinker planning around a conversation
the voice already walked past. Users will feel the seam instantly —
"wait, you just said the opposite."

This doc is how we make both LLMs feel like one mind.

---

## 1. The core idea: one context, two surfaces

The voice LLM and the thinker never hold private world-views.
There is **one canonical context** (owned by Hsafa) and two
*projections* of it:

- **Voice projection** = a compact system prompt regenerated
  on every Gemini Live session boundary (every ~10 min or on
  meaningful world changes). Includes: who is present, their social
  memory summary, recent shared history, current intention.
- **Thinker projection** = the full memory store, with embeddings,
  episodes, relationships, goals.

The thinker *authors* the voice's prompt. The voice *reports* what
happened back into the thinker's log. Neither holds state the other
can't see.

---

## 2. The shared conversation log

Everything said (by user or robot), every tool call, every
significant world event, goes into a single append-only log:

```
ConversationLog
  - [ts] user says "…"        (with speaker_identity if known)
  - [ts] robot says "…"       (via Gemini Live)
  - [ts] event: person_detected husam
  - [ts] tool_call: focus_on_person(husam) -> ok
  - [ts] thinker_note: "Husam seems tired; soften tone."
  - [ts] thinker_injection -> voice: "say: 'you look tired, everything ok?'"
```

**Writer:** the EventBus (see `architecture.md`) appends events;
Gemini Live appends utterances on `turn_complete` and `user_speech_end`.

**Readers:**

- Hsafa reads it to think.
- The voice prompt builder reads the tail to rebuild the system
  prompt when a Gemini session rolls over.

This is the one file that, if lost, makes the robot amnesiac. It's
also the only data structure both brains can fully trust — because
they both write to it.

---

## 3. No contradiction rule

Whenever the voice is about to say something it can't back up, we'd
rather it *not* say it. Concretely:

- Facts about a known person ("Husam is a roboticist", "we talked
  about X yesterday") must come from Hsafa's memory. The voice
  prompt includes the allowed facts; we lean on Gemini's system
  prompt adherence.
- The voice does **not** invent plans ("I'll remind you tomorrow").
  If the user asks for something actionable, the voice calls the
  `remember(...)` or `schedule(...)` tool which Hsafa owns. The
  voice only confirms actions the thinker actually committed to.
- On Gemini session rollover, the new session's prompt includes a
  one-line "what we agreed to" summary from Hsafa so promises
  survive the reconnect.

In practice the tooling enforces this: any "commitment" verb in the
voice has to be backed by a tool call whose return value Hsafa
produced.

---

## 4. Thinker-initiated speech (the hard part)

Today the voice only talks when the user talks. We want Hsafa to be
able to say: *"I just remembered something — tell the user."*

Design:

- New tool on the voice side, *from the thinker's perspective*:
  `say(text, urgency, interrupt_ok)`. It's not exposed to the LLM —
  it's an internal method Hsafa can call into Gemini Live's session.
- Implementation: Hsafa pushes a `say_this` event to the bus; the
  `GeminiLive` wrapper, when safe (no active user turn, or urgency
  allows interrupt), calls
  `session.send_client_content(Content(parts=[Part(text="...")]))`
  with an instruction framing like *"(system) You just thought of
  this, share it briefly and naturally: …"*.
- The voice paraphrases it in its own voice — we never hardcode
  exact strings from Hsafa. That keeps the persona unified.

Examples of natural triggers:

- Hsafa runs a web search in the background; when it completes it
  pushes `say_this("The answer is 42, mention it casually.")`.
- Face recognizer fires `person_detected(Husam)` after a long
  absence; Hsafa, looking at history, pushes `say_this("Greet
  Husam warmly — last chat was 3 days ago about project X.")`.
- A scheduled reminder fires → `say_this("Gentle reminder: the
  meeting starts in 5 minutes.")`.

The voice decides *how* to say it. Hsafa decides *whether and what*.

---

## 5. Turn-taking protocol between the two

To avoid Hsafa stepping on the user:

```
gemini_busy = is_speaking OR user_speaking
say_this with urgency="normal" -> queue until not gemini_busy
say_this with urgency="high"   -> interrupt_ok, send immediately
say_this with urgency="idle"   -> drop if not gemini_busy and > 30s silence
```

A 150 ms debounce prevents the "thinker pushes, then finishes the
thought, then pushes again" stutter — collapse bursts into one.

---

## 6. How the voice asks the thinker for help

The reverse direction — voice needing the thinker — today would be
a Gemini tool call. We keep that, but we add a *passive* channel:
every finalized user utterance emits `speech_heard` and the thinker
is always subscribed. So by the time the voice needs something, the
thinker often already has a relevant memory pre-fetched and injected
on the next turn boundary.

Concretely we give Gemini one smart tool, `ask_thinker(question)`,
that blocks up to ~1.5 s for a Hsafa completion. Use it for
"you don't know — ask me" moments. Don't use it for trivia (that's
what the voice model's own knowledge is for).

---

## 7. Session rollover — the moment of truth

Gemini Live sessions die every ~10 min. We already reconnect with
`session_resumption`, which preserves Gemini's *internal* context.
But that's not enough for a shared mind, because:

- The resumption handle is opaque and we can't inspect it.
- The thinker's latest decisions between old session and new one
  must be injected.

So on every rollover:

1. Freeze a fresh "voice projection" from Hsafa (≤ 1k tokens).
2. Start the new Gemini session with that projection *plus* the
   resumption handle. The handle keeps continuity; the fresh
   projection keeps truth.
3. Post-connect, replay any `say_this` events that were queued
   during the gap.

---

## 8. The persona layer

"One entity" also means one personality across both brains.
We define it in one place:

```
persona.md  (loaded by both brains)
  - Name: Hsafa
  - Tone: warm, curious, brief, playful-but-not-silly
  - Values: honesty, gentle humor, remembers what matters
  - Never: promises without a backing action, pretends to sense
    things it can't, narrates its internal state
```

Gemini Live's system prompt includes it. Hsafa's system prompt
includes it. If we tune the persona, we tune it once.

---

## 9. Minimal first build

1. Create `ConversationLog` as an append-only JSONL; have Gemini
   Live wrap its `_receive_task` / `_mic_task` to emit
   `robot_said` / `user_said` events.
2. Add a `voice_projection.py` that turns Hsafa's social + episodic
   memory into a ≤ 1k-token summary, rebuilt on session rollover.
3. Add `GeminiLive.inject_client_message(text, interrupt_ok)` —
   this is what Hsafa calls for `say_this`.
4. Add urgency-aware queue in front of it (§5).
5. Ship `ask_thinker(question)` as a tool the voice LLM can call.
6. Ship shared `persona.md` included in both system prompts.

After (1) and (3) we already have a robot that can:

- Remember what the user said across reboots.
- Speak up when the thinker has something to say.

That alone changes what the robot *is* far more than any individual
feature.
