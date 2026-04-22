# Identity — Linking Face, Voice, and Name

Humans don't remember people as "face row #47." They link a voice to
a face to a name to a history, and any of those signals alone is
enough to recognize the person. That's the target.

This doc is the design for the `IdentityGraph` — the thing that turns
our separate face / voice / name stores into one coherent *identity*.

---

## 1. Why it matters

Today we have three disconnected facts about a person:

- **Face** — a 512-d embedding + a name label in `FaceDB`.
- **Voice** — not yet used; Gemini Live hears speech but we throw away
  who said it.
- **Lip-sync speaker** — knows which *visible face* is talking, but not
  who is talking when the speaker is off-camera.

With no linking layer, the robot can't:

- Recognize Husam by voice alone when he's behind it.
- Learn a new voice *just by hearing Husam talk while the camera sees
  him* (automatic cross-modal enrollment — the human way).
- Say "I hear Husam but I don't see him — where are you?"

All of these fall out of one well-designed graph.

---

## 2. The graph

```
         Identity("husam")
         ├── names:   ["husam"]   (canonical + aliases: "hossam")
         ├── faces:   [emb_f1, emb_f2, ...]     (FaceDB entries)
         ├── voices:  [emb_v1, emb_v2, ...]     (voice embeddings)
         ├── memory:  Hsafa social-memory pointer
         └── history: [first_seen, last_seen, #interactions, ...]
```

An `Identity` is a *person*. Signals (face, voice, gait later) are
*evidence* linked to it. Recognition returns an `Identity`, not a face
row or a voice row.

### Node types

- `IdentityNode` — the person. Stable UUID + canonical name.
- `FaceSignal` — `{embedding, source_frame_ts, quality, identity_id}`.
- `VoiceSignal` — `{embedding, utterance_id, duration_s, snr, identity_id}`.
- `AliasNode` — spelling variants + preferred name.

### Edges

- `face -> identity`, `voice -> identity`, `alias -> identity`.
- `cooccurs(face_signal, voice_signal, Δt < 2s, speaker=true)` —
  *the glue*. When a recognized face is speaking *and* we capture a
  clean voice sample in the same window, we emit a co-occurrence
  edge. Enough co-occurrences and the voice cluster gets attached to
  that identity automatically.

---

## 3. Cross-modal enrollment (the human part)

The robot learns your voice the same way a friend does: by hearing you
while they can see your mouth move.

**Trigger:** every time the active-speaker detector says "face F is
speaking" and the mic has ≥ 1.5 s of audio above an SNR threshold
attributable to that utterance (Gemini-Live's VAD turn boundaries are
perfect for this).

**Active-speaker detector — use TalkNet.** Our current `LipMotion`
just looks at mouth-region pixel variance; it false-fires on chewing,
laughing, yawning, and anyone with a moving hand near their face.
**TalkNet** (Tao et al., 2021 — audio-visual speaker detection with
cross-attention between the mouth crop and the mic) gives a per-face
`speaking_prob ∈ [0, 1]` that fuses both modalities, so it only
fires when mouth motion and audio *agree*. It's small enough to run
at ~10 Hz on CPU for 1–3 faces with MTCNN crops we already compute.

Plugging it in is clean: `TalkNetTracker` becomes the v2 of
`LipMotionTracker` with the same `snapshot()` shape (face bbox +
`is_speaking`), so `FocusManager` / gaze `speaker` mode / the
`who_is_speaking` tool inherit it for free. The new, useful field it
adds is `speaking_prob` — the co-occurrence enroller in §3 should
require ≥ 0.8 sustained for 1 s before stashing a voice sample, so
we never attach a noisy "probably speaking" snippet to the wrong
identity.

This one swap also upgrades the speaker-mode gaze from "whoever's
lips are moving" to "whoever is actually producing the audio we hear,"
which is the intuition humans use.

**Pipeline:**

1. Snapshot `(face_embedding, voice_embedding, ts)`.
2. If the face resolves to `Identity(X)`, stash the voice sample
   under `pending_voice_samples[X]`.
3. When `pending_voice_samples[X]` has ≥ N clean samples (say 5),
   cluster them; if the intra-cluster variance is low, commit them
   as voice signals for `X`. Emit `voice_learned(X)`.
4. From that moment on `VoiceRecognizer` can answer
   "who is speaking?" without needing the camera.

**No tool call, no user interaction.** The robot just listens and links.
That's the part that feels alive.

---

## 4. Recognition fusion

When asked "who is this?", the answer is a weighted sum over
available evidence, not a single-modality vote:

```
score(identity) =
      w_face   * cos_sim(current_face,  closest_face_emb_of(id))
    + w_voice  * cos_sim(current_voice, closest_voice_emb_of(id))
    + w_prior  * recency_prior(id)     # people seen recently are more likely
    + w_social * hsafa_prior(id)       # Hsafa's context bias (expecting X)
```

- Face-only available → only `w_face` contributes; behaves like today.
- Voice-only available (off-camera) → `w_voice` carries it.
- Both → scores compound, ambiguity collapses fast.
- Hsafa can push a soft prior ("you just said goodbye to Husam 2 s
  ago; the new voice is probably not him") — this is how the slow
  brain helps the fast senses.

Thresholds and weights are just numbers; the architectural win is
that *every* recognition path goes through one function.

---

## 5. Wrong-name correction without a "rename" tool

Real humans don't have an admin command for this. When someone says
"actually I'm not Kindom, I'm Husam," what happens is:

- Gemini Live hears the correction and remembers for the session.
- Gemini emits a `correction` event (just plain text + the canonical
  face at that moment): *"face currently in frame is Husam, not
  Kindom."*
- The `IdentityGraph` applies it:
  - If `husam` doesn't exist → create it, move the `kindom` face
    signals over, retire `kindom`.
  - If `husam` already exists → merge the two identity nodes, union
    all signals, keep the stored Hsafa memory from both.
- Gemini in the next turn just says "got it, Husam" — no tool, no
  JSON, no confirmation ceremony.

The user never sees that anything was "renamed." They just feel
understood.

---

## 6. Storage layout (on-device)

```
data/
  identity/
    graph.json                   # nodes + edges, stable UUIDs
    faces/<identity_id>/*.npy    # face embeddings
    voices/<identity_id>/*.npy   # voice embeddings
    history/<identity_id>.jsonl  # append-only: seen, spoke, enrolled
```

Append-only history files are what Hsafa reads to rebuild episodic
memory after a restart. They're the ground truth for "what has this
robot actually observed."

---

## 7. Privacy & safety notes (for later, not now)

- Embeddings are irreversible but still biometric; we'll eventually
  need a "forget me" flow that wipes *all* signals + history for an
  identity.
- Voice enrollment must be pausable; add a mute-switch gesture / word
  that puts identity-learning to sleep.
- Never send raw embeddings off-device. Hsafa only receives names +
  events, never vectors.

---

## 8. Minimal first build

1. Introduce `Identity` dataclass + `graph.json` on disk; migrate
   `FaceDB` entries into it (one identity per existing name).
2. Add `VoiceEmbedder` (e.g. SpeechBrain ECAPA-TDNN on CPU, runs per
   finalized utterance — cheap).
3. Implement the co-occurrence enrollment pipeline (§3).
4. Replace `recognizer.identify_all` with
   `identity.recognize_all(face=..., voice=...)` using fusion (§4).
5. Route the `correction` event (§5) through the handler that today
   is `rename_person` — but without the tool surface.

After that, "who's speaking off-camera" and "greet Husam by voice
before seeing him" are both one-line features.
