"""hsafa_robot — Reachy Mini runtime services.

Layer map (see docs/architecture.md):

L0  I/O        : reachy_mini (external), OpenCV, GStreamer (via MediaManager)
L1  Perception : tracker, face_recognizer, lip_motion, audio_vad,
                 head_pose, gestures, voice_embedder
L2  Cognition  : events (EventBus), world_state (WorldStateHolder),
                 perception (WorldState builder), gaze_policy
                 (scoring engine), focus (FocusManager driver),
                 natural_gaze (saccades / idle drift / search),
                 identity_graph (face+voice+name link),
                 voice_identity (speaker ID + cross-modal enrollment)
L3  Voice      : gemini_live
L4  Thinker    : (reserved for future Hsafa Core bridge)

Control / motion:
  - robot_control : P-controller + NaturalGaze overrides + animations
  - animation     : idle + talking overlay animations
"""

__all__ = [
    # L1 perception
    "tracker",
    "face_db",
    "face_recognizer",
    "lip_motion",
    "audio_vad",
    "head_pose",
    "gestures",
    "voice_embedder",
    # L2 cognition
    "events",
    "world_state",
    "perception",
    "gaze_policy",
    "focus",
    "natural_gaze",
    "identity_graph",
    "voice_identity",
    # L3 voice
    "gemini_live",
    # L0 motion
    "robot_control",
    "animation",
]
