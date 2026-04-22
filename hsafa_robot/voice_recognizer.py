"""voice_recognizer.py - [SCAFFOLD] Speaker identification by voice.

Status: **NOT IMPLEMENTED**. This module defines the intended public
interface so the rest of the codebase (tools layer, system prompt,
world-state merge) can be wired against a stable shape while the
actual model integration lands in a follow-up.

Why this exists: face recognition tells us *who is in the room* when
we can see them, and lip-motion tells us *who among visible people is
speaking right now*. Neither helps when the speaker is off-camera,
on a phone call, or wearing a mask. A voice-embedding model covers
that gap symmetrically with :mod:`face_recognizer`:

    enroll_face(name)   -> remember the face you see now
    enroll_voice(name)  -> remember the voice you hear now
    identify_person()   -> who is visible?
    identify_speaker()  -> whose voice is the mic hearing?

Planned stack
-------------

* Model: `Resemblyzer <https://github.com/resemble-ai/Resemblyzer>`_
  (GE2E, 256-D embeddings, ~30 MB, pure PyTorch, CPU-friendly).
  Alternative: `speechbrain` ECAPA-TDNN for higher accuracy at the
  cost of model size.
* Storage: mirror :class:`hsafa_robot.face_db.FaceDB` exactly -- one
  ``.npy`` per person under ``data/voices/<name>.npy``, L2-normalized
  256-D embeddings.
* Input: the same 16 kHz mono PCM the Gemini session already produces
  via :meth:`reachy.media.get_audio_sample`. Enrollment buffers ~3-5
  seconds of mic audio, filters out silent chunks, feeds into the
  encoder. Identification uses a rolling ~1-2 s buffer.
* Concurrency: runs in a worker thread just like
  :class:`LipMotionTracker`, exposing a snapshot the tools layer can
  poll.

Cross-linking with face + lip-motion
------------------------------------

Once both exist, the right thing is a single ``active_speaker`` fusion
step: if lip-motion says "Husam is moving his mouth" AND the voice
matches Husam's embedding, confidence is high. If only one side
agrees, confidence is medium. Disagreements either side should be
logged -- they often indicate a bystander speaking or occlusion.
"""
from __future__ import annotations


class VoiceRecognizer:
    """Placeholder. See module docstring for the planned interface."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "VoiceRecognizer is a scaffold. Implement with Resemblyzer "
            "(or speechbrain) when voice identification is turned on."
        )
