"""hsafa_voice_vision.py — Voice + Vision bridge for HSAFA Core.

Records your voice via microphone, transcribes via ElevenLabs STT,
captures a camera image, and pushes both to a Haseef as an event.
The Haseef can call the `move_head` tool to search for objects;
after moving, a fresh image is captured and returned.

Usage:
    ./.venv/bin/python hsafa_voice_vision.py

Env:
    HSAFA_CORE_URL   (default: https://core.hsafa.com)
    HSAFA_CORE_KEY   (default: the prod key below)
    ELEVENLABS_KEY   (default: key below)
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
import random
import signal
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import httpx
import numpy as np

from hsafa_sdk import HsafaSDK, SdkOptions

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CORE_URL = os.environ.get("HSAFA_CORE_URL", "https://core.hsafa.com")
CORE_KEY = os.environ.get(
    "HSAFA_CORE_KEY",
    "sk_prod_7f2e8d9c4b3a6f1e0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0",
)
ELEVENLABS_KEY = os.environ.get(
    "ELEVENLABS_KEY",
    "sk_0b4ccdbf366979b39f6368f0c659d273454abfa3d6876768",
)
SKILL_NAME = os.environ.get("SKILL_NAME", "robot_vision")
HASEEF_NAME = os.environ.get("HASEEF_NAME", "RobotVision")
HASEEF_ID = os.environ.get("HASEEF_ID", "8aa60ad7-cb23-4e44-a26c-e8b7c8332d11")

# Robot
from hsafa_robot.emotion_player import EmotionClipPlayer

ROBOT_AVAILABLE = False
try:
    from reachy_mini import ReachyMini
    from hsafa_robot.animation import (
        IdleAnimation,
        TalkingAnimation,
        blend_offsets,
    )
    from hsafa_robot.robot_control import head_pose
    ROBOT_AVAILABLE = True
except ImportError:
    ReachyMini = None  # type: ignore
    head_pose = None  # type: ignore

# Camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
JPEG_QUALITY = 80

# Audio
RECORD_SECONDS = 3
SAMPLE_RATE = 16000
CHANNELS = 1

# Head movement settle time (seconds) — matches main.py
SETTLE_S = 1.2

# ---------------------------------------------------------------------------
# ElevenLabs STT
# ---------------------------------------------------------------------------
class ElevenLabsSTT:
    """Async ElevenLabs speech-to-text client."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.elevenlabs.io/v1"

    async def transcribe(self, wav_bytes: bytes) -> str:
        """Send WAV audio to ElevenLabs STT, return transcript."""
        async with httpx.AsyncClient(timeout=60) as client:
            files = {
                "file": ("audio.wav", io.BytesIO(wav_bytes), "audio/wav"),
            }
            data = {
                "model_id": "scribe_v1",
            }
            headers = {
                "xi-api-key": self.api_key,
            }
            response = await client.post(
                f"{self.base_url}/speech-to-text",
                headers=headers,
                data=data,
                files=files,
            )
            response.raise_for_status()
            result = response.json()
            text = result.get("text", "")
            # ElevenLabs returns segments; grab full text
            if not text and "language" in result:
                # older response format
                text = result.get("text", "")
            return text.strip()


# ---------------------------------------------------------------------------
# Audio Recorder
# ---------------------------------------------------------------------------
class AudioRecorder:
    """Record microphone audio to WAV bytes."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS):
        self.sample_rate = sample_rate
        self.channels = channels

    def record(self, duration: float = RECORD_SECONDS) -> bytes:
        """Record audio for `duration` seconds, return WAV bytes."""
        # Try sounddevice first (best cross-platform)
        try:
            import sounddevice as sd
            import numpy as np
            frames = int(self.sample_rate * duration)
            recording = sd.rec(
                frames,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.int16,
            )
            sd.wait()
            return self._to_wav(recording)
        except ImportError:
            pass

        # Fallback: sox rec
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            import subprocess
            subprocess.run(
                [
                    "rec", "-q",
                    "-r", str(self.sample_rate),
                    "-c", str(self.channels),
                    "-b", "16",
                    "-e", "signed",
                    tmp_path,
                    "trim", "0", str(duration),
                ],
                check=True,
                capture_output=True,
            )
            with open(tmp_path, "rb") as f:
                return f.read()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # Fallback: ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            import subprocess
            subprocess.run(
                [
                    "ffmpeg", "-y", "-f", "avfoundation",
                    "-i", ":default",
                    "-t", str(duration),
                    "-ar", str(self.sample_rate),
                    "-ac", str(self.channels),
                    "-sample_fmt", "s16",
                    tmp_path,
                ],
                check=True,
                capture_output=True,
            )
            with open(tmp_path, "rb") as f:
                return f.read()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        raise RuntimeError(
            "No audio recorder available. Install sounddevice: "
            "pip install sounddevice numpy"
        )

    def _to_wav(self, samples: np.ndarray) -> bytes:
        """Convert numpy int16 array to WAV bytes."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(samples.tobytes())
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
class Camera:
    """OpenCV camera wrapper with frame buffer."""

    def __init__(self, index: int = 0, width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT):
        self.index = index
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._latest: Optional[np.ndarray] = None

    def open(self) -> bool:
        # macOS AVFoundation backend
        self._cap = cv2.VideoCapture(self.index, getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY))
        if not self._cap.isOpened():
            # Fallback
            self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            print(f"[camera] Could not open camera index {self.index}")
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        ok, frame = self._cap.read()
        if not ok:
            print("[camera] Opened but first read failed")
            self._cap.release()
            self._cap = None
            return False
        self._latest = frame
        print(f"[camera] Opened at {self.width}x{self.height}")
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
# Robot connection
# ---------------------------------------------------------------------------
class RobotHead:
    """Direct head control via ReachyMini."""

    def __init__(self):
        self.reachy = None

    def connect(self) -> bool:
        if not ROBOT_AVAILABLE or ReachyMini is None:
            print("[robot] reachy_mini not available — head moves will be simulated")
            return False
        try:
            self.reachy = ReachyMini(automatic_body_yaw=False)
            print("[robot] Reachy connected.")
            return True
        except Exception as e:
            print(f"[robot] Failed to connect: {e}")
            return False

    def move(
        self, yaw_deg: float, pitch_deg: float, body_yaw_deg: float = 0.0
    ) -> bool:
        if self.reachy is None:
            print(
                f"[robot] SIMULATE move(yaw={yaw_deg}, pitch={pitch_deg}, "
                f"body_yaw={body_yaw_deg})"
            )
            return True
        try:
            import math
            self.reachy.set_target(
                head=head_pose(
                    roll=0.0, pitch=math.radians(pitch_deg), yaw=math.radians(yaw_deg)
                ),
                body_yaw=math.radians(body_yaw_deg),
            )
            print(
                f"[robot] Head moved to yaw={yaw_deg}, pitch={pitch_deg}, "
                f"body_yaw={body_yaw_deg}"
            )
            return True
        except Exception as e:
            print(f"[robot] set_target failed: {e}")
            return False

    def move_smooth(
        self,
        yaw_deg: float,
        pitch_deg: float,
        duration: float = 0.3,
        body_yaw_deg: float = 0.0,
    ) -> bool:
        """Smooth interpolated head + body move."""
        if self.reachy is None:
            print(
                f"[robot] SIMULATE move_smooth(yaw={yaw_deg}, pitch={pitch_deg}, "
                f"body_yaw={body_yaw_deg})"
            )
            return True
        try:
            import math
            try:
                from reachy_mini.utils.interpolation import InterpolationTechnique
                method = InterpolationTechnique.MIN_JERK
            except Exception:
                method = "minimum_jerk"
            self.reachy.goto_target(
                head=head_pose(
                    roll=0.0, pitch=math.radians(pitch_deg), yaw=math.radians(yaw_deg)
                ),
                duration=duration,
                method=method,
                body_yaw=math.radians(body_yaw_deg),
            )
            print(
                f"[robot] Head moved smoothly to yaw={yaw_deg}, pitch={pitch_deg}, "
                f"body_yaw={body_yaw_deg} (dur={duration}s)"
            )
            return True
        except Exception as e:
            print(f"[robot] goto_target failed: {e}")
            return False

    def center(self) -> bool:
        return self.move(0, 0)

    def disconnect(self):
        if self.reachy is not None:
            try:
                self.reachy.close()
            except Exception:
                pass
            self.reachy = None


# ---------------------------------------------------------------------------
# Animation Controller
# ---------------------------------------------------------------------------
class AnimationController:
    """Manages idle, talking, tool-call, and expression animations.

    Priority (highest first): expression > tool_call > talking > idle
    """

    EXPRESSIONS: Dict[str, Dict[str, float]] = {
        "neutral":  {"yaw": 0, "pitch": 0},
        "happy":    {"yaw": 0, "pitch": -25},
        "sad":      {"yaw": 0, "pitch": 25},
        "angry":    {"yaw": 0, "pitch": 10},
        "surprised": {"yaw": 0, "pitch": -30},
        "love":     {"yaw": 10, "pitch": -15},
        "tired":    {"yaw": 0, "pitch": 30},
        "confused": {"yaw": 20, "pitch": 0},
        "excited":  {"yaw": 0, "pitch": -35},
    }

    def __init__(self, head: RobotHead) -> None:
        self.head = head
        self._state = "idle"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="anim-ctrl")

        # Timers
        self._next_idle_move = 0.0
        self._next_talk_bob = 0.0
        self._expr_end = 0.0
        self._expr_return_end = 0.0
        self._expr_name = "neutral"
        self._tool_end = 0.0

        # Audio-based talking detection
        self._last_audio_time = 0.0

        # Deduplication
        self._last_yaw = 0.0
        self._last_pitch = 0.0

        # Emotion clip player (HF dataset)
        self._emotion_player = EmotionClipPlayer(head)

    # ---- public API --------------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def notify_audio(self, samples: Optional[bytes]) -> None:
        """Call from speaker_sink whenever audio samples are pushed."""
        if samples and len(samples) > 0:
            self._last_audio_time = time.time()

    def set_tool_call(self, duration: float = 0.8) -> None:
        """Brief 'thinking' animation while a tool is handled."""
        with self._lock:
            self._state = "tool_call"
            self._tool_end = time.time() + duration

    def show_expression(self, name: str, duration: float = 2.0) -> None:
        """Play a facial expression.

        Tries the HuggingFace emotion clip first; falls back to a static pose
        if no clip is available or the player is not ready.
        """
        # Stop any running clip first
        self._emotion_player.stop()

        with self._lock:
            self._state = "expression"
            self._expr_name = name
            now = time.time()
            self._expr_end = now + duration
            # After expression, spend 1.5s returning to neutral before idle
            self._expr_return_end = self._expr_end + 1.5

        # Try animated clip first
        if self._emotion_player.play(name, duration=duration):
            return

        # Fallback: static pose
        expr = self.EXPRESSIONS.get(name, self.EXPRESSIONS["neutral"])
        self._send_smooth(expr["yaw"], expr["pitch"], 0.5)

    def list_expressions(self) -> List[str]:
        return list(self.EXPRESSIONS.keys())

    # ---- internal loop ------------------------------------------------------
    def _run_loop(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            state: str

            with self._lock:
                state = self._state

                # Auto-expire transient states
                if state == "tool_call" and now > self._tool_end:
                    self._state = "idle"
                    state = "idle"
                elif state == "expression":
                    if now > self._expr_return_end:
                        self._state = "idle"
                        state = "idle"
                        self._next_idle_move = now + 2.0  # give idle a break
                    elif now > self._expr_end:
                        state = "returning"

                # Audio-driven talking promotion (lowest priority)
                is_talking = (now - self._last_audio_time) < 0.3
                if state not in ("expression", "tool_call"):
                    if is_talking and state != "talking":
                        self._state = "talking"
                        state = "talking"
                        self._next_talk_bob = 0.0
                    elif not is_talking and state == "talking":
                        self._state = "idle"
                        state = "idle"
                        self._next_idle_move = 0.0

            if state == "idle":
                self._do_idle(now)
            elif state == "talking":
                self._do_talking(now)
            elif state == "tool_call":
                self._do_tool_call(now)
            elif state == "expression":
                self._do_expression(now)
            elif state == "returning":
                self._do_returning(now)

            time.sleep(0.1)

    # ---- animation primitives -----------------------------------------------
    def _send_smooth(
        self, yaw: float, pitch: float, duration: float, body_yaw: float = 0.0
    ) -> None:
        """Only send if pose changed significantly to avoid spamming."""
        if (
            abs(yaw - self._last_yaw) > 1.0
            or abs(pitch - self._last_pitch) > 1.0
        ):
            self.head.move_smooth(yaw, pitch, duration, body_yaw_deg=body_yaw)
            self._last_yaw = yaw
            self._last_pitch = pitch

    def _do_idle(self, now: float) -> None:
        if now > self._next_idle_move:
            yaw = random.uniform(-8, 8)
            pitch = random.uniform(-5, 5)
            self._send_smooth(yaw, pitch, 2.0)
            self._next_idle_move = now + random.uniform(4, 7)

    def _do_talking(self, now: float) -> None:
        if now > self._next_talk_bob:
            base = -2.0 if self._last_pitch > 0 else 2.0
            pitch = base + random.uniform(-1.5, 1.5)
            self._send_smooth(0, pitch, 0.4)
            self._next_talk_bob = now + 0.7

    def _do_tool_call(self, now: float) -> None:
        # Brief attentive look-up
        self._send_smooth(0, -5, 0.3)
        time.sleep(0.5)
        with self._lock:
            if self._state == "tool_call":
                self._state = "idle"
                self._next_idle_move = time.time() + 1.0

    def _do_expression(self, now: float) -> None:
        # If a clip is playing, it handles the motion in its own thread.
        # Otherwise re-send the static pose so idle doesn't override it.
        if not self._emotion_player._current_name:
            expr = self.EXPRESSIONS.get(self._expr_name, self.EXPRESSIONS["neutral"])
            self._send_smooth(expr["yaw"], expr["pitch"], 0.5)

    def _do_returning(self, now: float) -> None:
        """Smoothly return to neutral after an expression."""
        self._send_smooth(0, 0, 1.0)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
class VoiceVisionApp:
    """Record voice → STT → capture image → push to Haseef → handle move_head."""

    def __init__(
        self,
        core_url: str = CORE_URL,
        core_key: str = CORE_KEY,
        eleven_key: str = ELEVENLABS_KEY,
        skill: str = SKILL_NAME,
        haseef_name: str = HASEEF_NAME,
    ):
        self.core_url = core_url
        self.core_key = core_key
        self.eleven_key = eleven_key
        self.skill = skill
        self.haseef_name = haseef_name
        self.haseef_id: Optional[str] = None

        self.sdk: Optional[HsafaSDK] = None
        self.stt = ElevenLabsSTT(eleven_key)
        self.recorder = AudioRecorder()
        self.camera = Camera()
        self.head = RobotHead()

        self._tool_calls: List[Dict[str, Any]] = []
        self._runs_completed = []
        self._running = True

    # --- HSAFA lifecycle -------------------------------------------------

    async def setup(self):
        self.sdk = HsafaSDK(
            SdkOptions(core_url=self.core_url, api_key=self.core_key, skill=self.skill)
        )

        # Register tools
        await self.sdk.register_tools([
            {
                "name": "move_head",
                "description": (
                    "Move the robot's head to a specific yaw and pitch angle in degrees. "
                    "After moving, the robot captures a fresh camera image and returns it. "
                    "Use this to search for objects or look in a specific direction. "
                    "yaw=0 looks straight ahead; positive yaw turns left, negative turns right. "
                    "pitch=0 is level; positive looks down, negative looks up. "
                    "Range: yaw -60..+60, pitch -30..+30."
                ),
                "input": {
                    "yaw_deg": "number",
                    "pitch_deg": "number",
                },
            },
            {
                "name": "get_current_time",
                "description": "Get the current date and time.",
                "input": {},
            },
        ])
        print(f"[{self.skill}] Registered tools.")

        # Use hardcoded haseef ID
        self.haseef_id = HASEEF_ID
        print(f"[OK] Using haseef id={self.haseef_id}")

        # Verify it exists and has the skill
        try:
            h = await self.sdk.haseef.get(self.haseef_id)
            skills = h.get("skills") or []
            if self.skill not in skills:
                print(f"[INFO] Attaching skill '{self.skill}' to haseef...")
                await self.sdk.haseef.add_skill(self.haseef_id, self.skill)
            print(f"[OK] Haseef '{h.get('name')}' ready with skills: {skills}")
        except Exception as e:
            print(f"[WARN] Could not verify haseef: {e}")

        # Connect to robot
        self.head.connect()

        # Tool handlers
        self.sdk.on_tool_call("move_head", self._handle_move_head)
        self.sdk.on_tool_call("get_current_time", self._handle_get_time)

        self.sdk.on("run.started", lambda e: print(f"  [event] run started"))
        self.sdk.on("run.completed", lambda e: self._runs_completed.append(e))
        self.sdk.on("tool.error", lambda e: print(f"  [event] tool.error: {e}"))

    async def _handle_move_head(self, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        yaw = float(args.get("yaw_deg", 0))
        pitch = float(args.get("pitch_deg", 0))
        print(f"  [TOOL] move_head(yaw={yaw}, pitch={pitch})")

        # Clamp
        yaw = max(-60, min(60, yaw))
        pitch = max(-30, min(30, pitch))

        # Move immediately
        self.head.move(yaw, pitch)

        # Wait for head to settle, then capture image and return it
        # in the tool result so the same run continues (no new event).
        await asyncio.sleep(SETTLE_S)
        jpeg_b64 = self.camera.get_base64_jpeg()
        if jpeg_b64:
            print(f"  [TOOL] Fresh image captured after move (yaw={yaw}, pitch={pitch}).")
        else:
            print("  [TOOL] Camera capture failed after move.")

        return {
            "ok": True,
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "image_base64": jpeg_b64,
            "note": (
                f"Head moved to yaw={yaw}, pitch={pitch}. "
                + ("Fresh image attached." if jpeg_b64 else "Camera capture failed.")
            ),
        }

    async def _handle_get_time(self, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        return {"iso": now.isoformat(), "human": now.strftime("%Y-%m-%d %H:%M:%S UTC")}

    # --- Main loop -------------------------------------------------------

    async def run(self):
        # Open camera
        if not self.camera.open():
            print("[FATAL] Cannot open camera.")
            return

        # Start SSE listener in background
        print("[app] Connecting SSE stream...")
        listen_task = asyncio.create_task(self.sdk.connect())
        await asyncio.sleep(1)

        try:
            while self._running:
                print("\n" + "=" * 50)
                print("Press Enter to record voice (or 'q' + Enter to quit)")
                try:
                    user_input = await asyncio.to_thread(input)
                except EOFError:
                    break
                if user_input.strip().lower() == "q":
                    break

                # 1. Record audio
                print(f"[voice] Recording {RECORD_SECONDS}s...")
                try:
                    wav_bytes = await asyncio.to_thread(self.recorder.record, RECORD_SECONDS)
                    print(f"[voice] Recorded {len(wav_bytes)} bytes.")
                except Exception as e:
                    print(f"[voice] Record failed: {e}")
                    continue

                # 2. STT
                print("[stt] Sending to ElevenLabs...")
                try:
                    text = await self.stt.transcribe(wav_bytes)
                    print(f"[stt] Transcript: '{text}'")
                except Exception as e:
                    print(f"[stt] Failed: {e}")
                    continue
                if not text:
                    print("[stt] No speech detected.")
                    continue

                # 3. Capture image
                print("[camera] Capturing image...")
                jpeg_b64 = self.camera.get_base64_jpeg()
                if jpeg_b64:
                    print(f"[camera] Captured {len(jpeg_b64)//1024}KB JPEG.")
                else:
                    print("[camera] Capture failed!")
                    continue

                # 4. Push event to haseef (fire-and-forget)
                print("[hsafa] Pushing voice+vision event...")
                try:
                    await self.sdk.push_event({
                        "type": "user_message",
                        "data": {
                            "text": text,
                            "image_base64": jpeg_b64,
                        },
                        "haseefId": self.haseef_id,
                    })
                    print("[hsafa] Event pushed. Haseef will respond asynchronously.")
                except Exception as e:
                    print(f"[hsafa] Push failed: {type(e).__name__}: {e}")
                    continue

        finally:
            self._running = False
            self.camera.close()
            self.head.disconnect()
            if self.sdk:
                await self.sdk.disconnect()
            listen_task.cancel()
            try:
                await listen_task
            except asyncio.CancelledError:
                pass
            print("[app] Shutdown complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main():
    app = VoiceVisionApp()
    await app.setup()
    await app.run()


if __name__ == "__main__":
    # Graceful shutdown on Ctrl-C
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _sigint():
        print("\n[signal] Caught SIGINT, shutting down...")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _sigint)

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
    except asyncio.CancelledError:
        pass
