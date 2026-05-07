"""hsafa_robot — Reachy Mini runtime services (minimal).

L0  I/O        : reachy_mini (external), OpenCV, GStreamer
L1  Perception : tracker (YOLOv8-Pose + ByteTrack), audio_vad (Silero)
L2  State      : events (EventBus), world_state (WorldStateHolder)
L3  Voice      : gemini_live

Control / motion:
  - robot_control : P-controller + animations
  - animation     : idle + talking overlay animations
"""

__all__ = [
    # L1 perception
    "tracker",
    "audio_vad",
    # L2 state
    "events",
    "world_state",
    # L3 voice
    "gemini_live",
    # L0 motion
    "robot_control",
    "animation",
]
