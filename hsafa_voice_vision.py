"""hsafa_voice_vision.py — Minimal camera + robot controller.

Exports:
    Camera          — OpenCV camera wrapper
    RobotController — Thin wrapper around ReachyMini for head + emotion control
"""
from __future__ import annotations

import base64
import collections
import logging
import math
import threading
import time
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

    Priority (highest first): expression > speaking > idle
    """

    def __init__(self, reachy) -> None:
        self.reachy = reachy
        self._emotions = None  # lazy-loaded RecordedMoves
        self._anim_state = "idle"  # "idle", "speaking", "expression"
        self._last_audio_time = 0.0
        self._speech_amp = 0.0
        self._stop_idle = threading.Event()
        self._idle_thread: Optional[threading.Thread] = None

    # ---- speaking detection ------------------------------------------------
    def notify_audio(self, samples) -> None:
        """Call from speaker_sink whenever Gemini audio is pushed.

        samples is a numpy ndarray (float32 @ 16 kHz) from gemini_live.
        """
        try:
            if samples is not None and len(samples) > 0:
                self._last_audio_time = time.time()
                raw = float(np.abs(samples).mean())
                scaled = min(raw * 20.0, 1.0)
                self._speech_amp = 0.7 * self._speech_amp + 0.3 * scaled
                if self._anim_state not in ("expression",):
                    self._anim_state = "speaking"
        except Exception as e:
            log.error("notify_audio failed: %s", e)

    # ---- idle / animation loop ---------------------------------------------
    def start_idle(self) -> None:
        """Start the background animation thread."""
        self._stop_idle.clear()
        self._idle_thread = threading.Thread(target=self._idle_loop, daemon=True, name="idle")
        self._idle_thread.start()
        log.info("Animation loop started.")

    def stop_idle(self) -> None:
        """Stop the background animation thread."""
        self._stop_idle.set()
        if self._idle_thread:
            self._idle_thread.join(timeout=1.0)
        log.info("Animation loop stopped.")

    def _idle_loop(self) -> None:
        """Continuous loop: idle sway, audio-reactive speaking, or sleep during expression."""
        import random
        import time

        from reachy_mini.utils import create_head_pose

        t0 = time.time()
        next_drift = 0.0
        pitch_off = 0.0
        yaw_off = 0.0

        # Speaking emphasis state
        next_emphasis = 0.0
        emphasis_yaw = 0.0
        emphasis_decay = 0.0

        # Sound direction tracking
        self._doa_history = collections.deque(maxlen=5)
        self._doa_yaw_target = 0.0
        self._doa_yaw_current = 0.0
        self._last_doa_poll = 0.0

        while not self._stop_idle.is_set():
            state = self._anim_state

            # Expression has full control — just sleep
            if state == "expression":
                time.sleep(0.05)
                continue

            # Speaking timeout → idle after 1.5 s silence (accounts for audio buffer)
            if state == "speaking" and time.time() - self._last_audio_time > 1.5:
                self._anim_state = "idle"
                state = "idle"
                self._speech_amp = 0.0
                emphasis_decay = 0.0

            # Poll sound direction at ~10 Hz
            now = time.time()
            if now - self._last_doa_poll > 0.1:
                self._last_doa_poll = now
                try:
                    doa, is_speech = self.reachy.media.get_DoA()
                    if is_speech:
                        self._doa_history.append(doa)
                        n = len(self._doa_history)
                        sin_mean = sum(math.sin(a) for a in self._doa_history) / n
                        cos_mean = sum(math.cos(a) for a in self._doa_history) / n
                        smoothed = math.atan2(sin_mean, cos_mean)
                        # Deadzone around front/back ambiguity (π/2 ± 0.2 rad)
                        if abs(smoothed - math.pi / 2) < 0.2:
                            self._doa_yaw_target = 0.0
                        else:
                            self._doa_yaw_target = 60.0 - (smoothed / math.pi) * 120.0
                    else:
                        self._doa_history.clear()
                        self._doa_yaw_target = 0.0
                except Exception:
                    pass

            # Smooth toward target (or back to zero when speech stops)
            self._doa_yaw_current += (self._doa_yaw_target - self._doa_yaw_current) * 0.15

            t = time.time() - t0

            if state == "speaking":
                amp = self._speech_amp

                # 1. Audio-reactive bob (z + pitch at 5.5 Hz)
                phase = 2 * math.pi * t * 5.5
                bob_z = amp * 6.0 * math.sin(phase)
                bob_pitch = amp * 6.0 * math.sin(phase)

                # 2. Slow ambient drift
                drift_yaw = 3.0 * math.sin(2 * math.pi * t / 6.1)
                drift_roll = 1.5 * math.sin(2 * math.pi * t / 8.7)

                # 3. Occasional emphasis tilts
                if t > next_emphasis and amp > 0.1:
                    emphasis_yaw = random.uniform(-10.0, 10.0)
                    emphasis_decay = 1.0
                    next_emphasis = t + random.uniform(2.0, 5.0)
                if emphasis_decay > 0.01:
                    emphasis_decay *= 0.95  # fade ~1 s

                pose = create_head_pose(
                    z=bob_z,
                    pitch=bob_pitch,
                    yaw=drift_yaw + emphasis_yaw * emphasis_decay + self._doa_yaw_current,
                    roll=drift_roll,
                    degrees=True,
                    mm=True,
                )
            else:
                # Idle: subtle roll sway + micro-glances
                roll = 1.5 * math.sin(2 * math.pi * t / 7.3)
                if t > next_drift:
                    pitch_off = random.uniform(-3.0, 3.0)
                    yaw_off = random.uniform(-5.0, 5.0)
                    next_drift = t + random.uniform(3.0, 8.0)

                pose = create_head_pose(
                    roll=roll,
                    pitch=pitch_off,
                    yaw=yaw_off + self._doa_yaw_current,
                    degrees=True,
                    mm=True,
                )
            self.reachy.set_target_head_pose(pose)

            # Antennas always sway (same for idle & speaking)
            ant_r = 0.3 * math.sin(2 * math.pi * t / 5.7)
            ant_l = 0.3 * math.sin(2 * math.pi * t / 6.3 + 1.0)
            self.reachy.set_target_antenna_joint_positions([ant_r, ant_l])

            time.sleep(0.02)  # ~50 Hz

    # ---- emotions ----------------------------------------------------------
    def _load_emotions(self):
        if self._emotions is None:
            from reachy_mini.motion.recorded_move import RecordedMoves
            self._emotions = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
        return self._emotions

    def show_expression(self, name: str) -> bool:
        """Play a recorded emotion clip (motion + sound).

        Pauses the animation loop, plays the clip, then smoothly returns to neutral.
        """
        try:
            from reachy_mini.utils import create_head_pose

            self._anim_state = "expression"
            moves = self._load_emotions()
            move = moves.get(name)
            self.reachy.play_move(move, initial_goto_duration=0.5, sound=True)
            log.info("Played emotion '%s' (%.2fs)", name, move.duration)

            # Smoothly return to neutral before resuming idle/speaking
            self.reachy.goto_target(
                head=create_head_pose(roll=0, pitch=0, yaw=0, degrees=True, mm=True),
                duration=1.5,
            )
            self._anim_state = "idle"
            return True
        except Exception as e:
            self._anim_state = "idle"
            log.warning("Expression '%s' failed: %s", name, e)
            return False

    def list_expressions(self) -> List[str]:
        try:
            return self._load_emotions().list_moves()
        except Exception:
            return []

    def cancel_expression(self) -> None:
        self.reachy.cancel_move()

    # ---- head movement -----------------------------------------------------
    def move_head(self, yaw_deg: float, pitch_deg: float, duration: float = 0.3) -> None:
        """Smoothly move the head to a yaw/pitch angle (degrees)."""
        self._anim_state = "expression"
        self.reachy.goto_target(
            head=head_pose(
                roll=0.0,
                pitch=math.radians(pitch_deg),
                yaw=math.radians(yaw_deg),
            ),
            duration=duration,
        )
        self._anim_state = "idle"
        log.info("Head moved to yaw=%.1f pitch=%.1f (dur=%.2fs)", yaw_deg, pitch_deg, duration)

    def center_head(self, duration: float = 0.5) -> None:
        self.move_head(0, 0, duration=duration)
