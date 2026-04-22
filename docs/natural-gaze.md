# Natural Gaze — Making Reachy's Head Feel Alive

`gaze-modes.md` nails the *mechanism* (modes + WorldState). This doc
is about the **behavior** layered on top so the head stops feeling
like a P-controller on a bounding-box centroid and starts feeling
like a curious animal.

The target: someone walks into the room and, at a glance, can't tell
whether a human or the robot is deciding where to look.

---

## 1. What makes a gaze look "alive"

Three things separate believable gaze from robotic gaze:

1. **Saccades + fixations, not smooth chasing.** Eyes don't
   continuously follow a moving target; they snap between fixations
   with ballistic motion and hold. Our current P-controller smoothly
   drifts — that's what reads as uncanny. We need fast saccades
   (100–200 ms) and ~0.5–2 s holds.
2. **Attention drift.** Even when a person is speaking, the gaze
   occasionally checks the environment — a hand, a movement in the
   periphery, a pet. Then returns. Staring *without* drift reads as
   threatening.
3. **Proactive search.** If the person you were looking at
   disappears, you look where they went, not where they were. And
   after a couple of seconds of absence you actively scan.

Everything in this doc serves those three.

---

## 2. The model: a gaze policy, not a target

Instead of "pick target, point head at target," the head is driven
by a **policy that picks a gaze target every 200–400 ms** and a
motion planner that executes human-like saccades.

```
GazePolicy.tick(WorldState, memory) -> GazeTarget
GazeTarget = {kind, ref, hold_ms, saccade_speed}
   kind: "person" | "face" | "object" | "point" | "search" | "idle_drift"
```

The policy is a scoring function. Every candidate gets a score; the
winner becomes the next fixation.

### Candidates

- **Each visible person** (with bonus if speaking — ideally via
  TalkNet's audio-visual `speaking_prob`, see `identity.md` §3 — if
  newly arrived, if named-and-known).
- **The last known location** of a person who just disappeared.
- **A "curiosity" target**: a region of high motion or a newly
  detected object.
- **An idle drift target**: a slow natural sweep when no one is in
  the room.
- **A thinker-pushed target**: when Hsafa says "look left, I think
  Ahmad is there," that's a candidate with a time-limited prior.

### Score terms (first pass, all tunable)

| Term | Sign | Notes |
|---|---|---|
| is_currently_speaking | + big | Strongest signal. |
| is_known_person | + med | Named faces beat strangers. |
| just_appeared | + small | Natural orienting response. |
| already_staring_at_them | − | Hysteresis *and* attention decay. |
| stare_duration_s | − growing | After ~3 s we feel the urge to glance away. |
| peripheral_motion_magnitude | + small | The curiosity term. |
| thinker_prior(kind, ttl) | + | Hsafa nudges. |
| last_lost_location_recency | + decaying | Keeps "they just walked out" salient. |

The trick is *hysteresis*: whoever we're looking at gets a short
lock-in bonus (1–1.5 s) so we don't flicker every time the scores
are close. After the lock-in decays, scores compete fairly again.

---

## 3. Saccadic motion planner

Two motion primitives, not one:

- **Saccade** — target angle reached in 120–220 ms with a velocity
  profile peaking ~5× the current tracking speed. Antennas don't
  move during a saccade (human neck-only movement feels right).
- **Smooth pursuit** — once locked on a moving person, we track
  them with the existing P-controller but damped harder so we don't
  overshoot.

Switching between them:

```
on gaze_target_changed:
    if angular_distance(current, target) > 8°:   saccade
    else:                                         smooth pursuit
```

Add **micro-saccades** when the head has been still > 700 ms: tiny
±0.5° twitches. They're almost invisible individually but they're
what makes a fixation look *alive* instead of frozen.

---

## 4. Search behavior — "where did they go?" / "who just said that?"

When we have evidence a person exists but we can't *see* them, the
brainstem-level response is **not** to go back to "largest body."
It's to actively look for them. Two triggers, same behavior.

### 4a. Trigger: visible person disappeared

1. Keep looking at their last known bbox for ~500 ms (maybe they'll
   come back).
2. Saccade to the edge of the frame they *left through* (extrapolate
   from their last velocity).
3. If still nothing, do a slow scan: 3 fixations across the frame
   (~1 s each).
4. Optional Hsafa turn: publish `person_lost(name)`; if Hsafa has a
   better guess ("they usually walk to the kitchen"), push a
   thinker prior for that direction.
5. After the scan, fall back to `idle_drift`.

### 4b. Trigger: heard a voice, see no face

We hear speech (Gemini-Live VAD fires `user_speech_start`, or
TalkNet reports audio energy with no face `speaking_prob > 0.5`,
or the voice embedding is a known identity) but no face in
`WorldState` is currently speaking. Someone is here but out of view.

1. **Prefer the direction of arrival if we have it.** With a
   mic array on Reachy we can estimate the DoA of the utterance;
   saccade straight to that yaw (±20° uncertainty cone).
2. **Mono mic (today): turn toward the wider side of the room.**
   Pick the side whose frame edge has had less recent motion — the
   sound probably came from the unobserved side. If tied, go *left
   first* (humans have a mild left bias; avoids looking mechanical).
3. **Scan sweep**: saccade to yaw −45°, hold ~800 ms, then yaw
   +45°, hold ~800 ms. A human-paced "who said that?" sweep, not
   a radar scan.
4. **If the voice is a known identity** (see `identity.md` §4 —
   voice embedding recognized Husam): bias the search with a
   thinker prior and add *"Husam?"* as a candidate the voice LLM
   can vocalize (the `say_this` path — if they don't appear in
   ~2 s, softly call their name).
5. If the scan fails, fall back to `idle_drift` but keep one
   extra saccade budget for the next 3 s — so if they speak *again*
   we immediately recheck their guessed direction instead of
   re-scanning from scratch.

### Shared shape

Both triggers reduce to:

```
SearchIntent {
  reason: "person_lost" | "voice_unseen"
  prior_direction: yaw | None       # DoA / last velocity
  target_identity: name | None      # for voice-known case
  ttl_s: 3.0
}
```

…pushed as an event the `GazePolicy` consumes as a high-priority
candidate. One code path, two senses feeding it.

This is the one piece that will most dramatically change how "smart"
the robot feels, for maybe 80 lines of code.

---

## 5. Idle behavior — the room is empty

When no humans are present, the head shouldn't freeze mid-center.
Humans idly look around. Do:

- Slow sinusoidal sweep, yaw amplitude ±15°, pitch ±5°, period ~6 s,
  with random-phase noise so it's not a metronome.
- Occasional "something caught my eye" saccade to a peripheral
  motion spike (the curiosity term, same as §2).
- Antennas cycle through a calm idle animation (already built).

The instant a person appears, `is_known_person` or `just_appeared`
dominates and we orient. This transition should be a *saccade*, not
a smooth pan — the head *notices*, it doesn't drift onto you.

---

## 6. Attention depth — multi-person rooms

When 2+ people are in view, the policy should feel like a person
listening to a conversation:

- Gaze follows the **current speaker** (§2 strongest term).
- ~20% of the time it glances at the **non-speaker** — the
  listener-check that humans do to read reactions. Short fixation
  (~400 ms), then back.
- If someone nods or waves (gesture hook), that's a large instant
  score bump — the gaze snaps there briefly.

This is the cheapest layer of social intelligence we can add, and
it's the one that makes group interaction feel natural.

---

## 7. Thinker-driven gaze (Hsafa's role)

The slow brain doesn't pick fixations; it *biases* the policy.
Examples of Hsafa priors:

- "Ahmad just came online in a WhatsApp chat — probably here;
  check the doorway." → `prior(point=(door_x, door_y), ttl=5s)`.
- "Husam asked me to watch his coffee." → `prior(object='cup', +)`.
- "It's 09:00, Husam usually arrives now." → `prior(name='husam', +)`.

Concretely: Hsafa publishes `gaze_prior` events to the bus with a
TTL; the policy adds them as candidates while live. No hard
overrides, no tool calls to move the head — the policy *chooses*
whether to comply, exactly like a human would with a hunch.

The `set_gaze_mode("person", name=...)` tool still exists for
explicit "look at me" commands. Priors are for soft suggestions.

---

## 8. Minimal first build (order matters)

1. Introduce `GazePolicy` + `GazeTarget` shape; refactor
   `FocusManager` to *apply* the policy's pick instead of making
   the decision. The tool handlers keep the same external surface.
2. Add saccade vs. smooth-pursuit split in `robot_control.py`.
3. Add person-lost search (§4). Biggest felt upgrade.
4. Add idle drift (§5). Second biggest felt upgrade.
5. Add listener-check glances (§6).
6. Wire `gaze_prior` events from Hsafa (§7).

Each step is independently valuable; none requires the next.

---

## 9. What we are *not* doing yet

- No deep-learned gaze policy. Hand-tuned scoring gets us 90% there.
- No eye/pupil simulation — Reachy has antennas but no eyes. We
  express attention with head + antenna posture.
- No multi-camera or environmental awareness — one front camera is
  enough for everything above.
- No micro-expression layer (head tilt when curious, dip when
  apologetic). Nice-to-have; keep it for the "v2 of animation" pass.
