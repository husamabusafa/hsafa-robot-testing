"""hsafa_voice_vision.py — Minimal camera + robot controller.

Exports:
    Camera          — OpenCV camera wrapper
    RobotController — Thin wrapper around ReachyMini for head + emotion control
"""
from __future__ import annotations

import base64
import logging
import math
from typing import List, Optional

import cv2
import numpy as np

from hsafa_robot.robot_control import head_pose

log = logging.getLogger("robot_controller")

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
JPEG_QUALITY = 80


class Camera:
    """OpenCV camera wrapper."""

    def __init__(self, index: int = 0, width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT):
        self.index = index
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._latest: Optional[np.ndarray] = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.index, getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY))
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            log.warning("Could not open camera index %s", self.index)
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        ok, frame = self._cap.read()
        if not ok:
            self._cap.release()
            self._cap = None
            return False
        self._latest = frame
        log.info("Camera opened at %sx%s", self.width, self.height)
        return True

    def grab(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if ok:
            self._latest = frame
        return self._latest

    def get_jpeg(self, quality: int = JPEG_QUALITY, mirror: bool = True) -> Optional[bytes]:
        frame = self.grab()
        if frame is None:
            return None
        if mirror:
            frame = cv2.flip(frame, 1)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buf.tobytes() if ok else None

    def get_base64_jpeg(self, quality: int = JPEG_QUALITY, mirror: bool = True) -> Optional[str]:
        jpeg = self.get_jpeg(quality, mirror)
        return base64.b64encode(jpeg).decode("ascii") if jpeg else None

    def close(self):
        if self._cap:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Robot Controller
# ---------------------------------------------------------------------------
class RobotController:
    """Minimal wrapper around ReachyMini.

    - Head movement via goto_target (smooth) or set_target (instant).
    - Emotions via the official RecordedMoves library + play_move().
    """

    def __init__(self, reachy) -> None:
        self.reachy = reachy
        self._emotions = None  # lazy-loaded RecordedMoves

    def _load_emotions(self):
        if self._emotions is None:
            from reachy_mini.motion.recorded_move import RecordedMoves
            self._emotions = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
        return self._emotions

    def move_head(self, yaw_deg: float, pitch_deg: float, duration: float = 0.3) -> None:
        """Smoothly move the head to a yaw/pitch angle (degrees)."""
        self.reachy.goto_target(
            head=head_pose(
                roll=0.0,
                pitch=math.radians(pitch_deg),
                yaw=math.radians(yaw_deg),
            ),
            duration=duration,
        )
        log.info("Head moved to yaw=%.1f pitch=%.1f (dur=%.2fs)", yaw_deg, pitch_deg, duration)

    def center_head(self, duration: float = 0.5) -> None:
        self.move_head(0, 0, duration=duration)

    def show_expression(self, name: str) -> bool:
        """Play a recorded emotion clip (motion + sound)."""
        try:
            moves = self._load_emotions()
            move = moves.get(name)
            self.reachy.play_move(move, initial_goto_duration=0.5, sound=True)
            log.info("Played emotion '%s' (%.2fs)", name, move.duration)
            return True
        except Exception as e:
            log.warning("Expression '%s' failed: %s", name, e)
            return False

    def list_expressions(self) -> List[str]:
        try:
            return self._load_emotions().list_moves()
        except Exception:
            return []

    def cancel_expression(self) -> None:
        self.reachy.cancel_move()
