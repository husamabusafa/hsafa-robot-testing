"""hsafa_robot — Reachy Mini runtime services.

Subpackages:
  - tracker        : YOLOv8-Pose + ByteTrack + Kalman + motion cascade
  - animation      : Overlay animations (idle / talking)
  - gemini_live    : Gemini Live API voice+vision session
  - robot_control  : Combines tracking + animations into robot commands
"""

__all__ = [
    "tracker",
    "animation",
    "gemini_live",
    "robot_control",
]
