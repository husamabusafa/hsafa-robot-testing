# Hsafa Robot — Design Docs

Short plain-English design notes. No code lives here — this is where we
decide the *shape* of the system before writing any of it.

Read in this order:

1. **`architecture.md`** — the foundation: processes, threads, WorldState,
   event bus. How every subsystem plugs into the same spine.
2. **`identity.md`** — how the robot knows *who* a person is across face,
   voice, and (later) gait / outfit. The multi-modal identity graph.
3. **`natural-gaze.md`** — how the head moves like a real being: saccades,
   curiosity, search when someone disappears, hysteresis. Upgrades
   `gaze-modes.md` from mechanical modes to believable behavior.
4. **`gaze-modes.md`** — existing: mode machine (`largest` / `person` /
   `speaker` / `free` / `manual` / `policy`) + `WorldState` shape.
   **Superseded for the mode surface by `gaze-policy.md`** — kept for
   the `WorldState` shape.
5. **`gaze-policy.md`** — the real focus design: two modes
   (`normal` scoring, `person(name)` with auto-fallback to scoring)
   + weights + hysteresis + virtual candidates for off-camera sounds.
6. **`unified-brain.md`** — how Gemini Live (fast, voice) and Hsafa
   (slow, thinker) feel like one entity: shared context, no
   contradictions, thinker-initiated speech through the voice.
7. **`hsafa-integration.md`** — existing: the concrete wiring plan for
   the Hsafa SDK (Haseef, scopes, events, tools).
8. **`tech-summary.md`** — snapshot of every tech / model / AI
   subsystem actually in this repo today.
9. **`tech-recommendations.md`** — add / simplify / watch-out list
   for pushing the robot toward humanoid: Silero VAD, DOA via
   GCC-PHAT, MediaPipe head-pose + hands, TS-TalkNet / FabuLight-ASD,
   emotion, and the execution order.

## Guiding principles

- **One WorldState.** Every sense writes into one snapshot; every brain
  reads from it. No module talks to another module directly.
- **Human-like first.** If a tool or behavior is a raw admin action
  (rename a row, reset a flag), it doesn't belong on the robot.
  Humans don't "rename_person" — they correct themselves in speech
  and remember.
- **Two brains, one self.** Gemini Live is the mouth + reflexes.
  Hsafa is the memory + intention. They share context so the robot
  never contradicts itself.
- **Everything is an event.** Vision, voice, motors, brain all emit
  and subscribe to events. Adding a sense or a skill = adding a
  publisher or subscriber, never a new private wire.
- **Gaze is behavior, not arithmetic.** A mathematically perfect
  P-controller on the largest bbox is uncanny. Real eyes glance,
  drift, get curious, and look for what disappeared.
