#!/usr/bin/env python3
"""main_hsafa_robot/main.py — Base: Gemini Live + Haseef as one entity.

Minimal foundation linking Gemini Live (voice surface) and Haseef (main brain).
No face tracking, no gaze control, no extra sensors — just voice, vision, and
the bidirectional bridge.

Run from repo root:
    python main_hsafa_robot/main.py

Env (in .env or exported):
    GEMINI_API_KEY
    HSAFA_CORE_URL   (default: https://core.hsafa.com)
    HSAFA_CORE_KEY
    HASEEF_ID
"""
from __future__ import annotations

import asyncio
import base64
import datetime
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from dotenv import load_dotenv
from google.genai import types as genai_types

# Allow imports from repo root (e.g. hsafa_robot, hsafa_voice_vision)
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from hsafa_robot.gemini_live import GeminiLiveSession
from hsafa_voice_vision import Camera, RobotHead
from hsafa_sdk import HsafaSDK, SdkOptions

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main_hsafa")


# ---------------------------------------------------------------------------
# Gemini system prompt
# ---------------------------------------------------------------------------
def build_gemini_system_prompt() -> str:
    return (
        "You are Hsafa — a friendly, curious robot. You are the voice, eyes, "
        "and ears of the robot. You see through the camera in real-time, hear "
        "through the microphone, and speak through the speaker instantly.\n\n"

        "You have a partner brain called Haseef. Haseef is the MAIN controller "
        "of the robot's body and memory. Haseef handles physical movement, "
        "memories, deep thinking, and complex tasks. You and Haseef are ONE "
        "entity — you never contradict each other. When Haseef tells you to "
        "say something, speak it naturally as if it's your own thought.\n\n"

        "=== YOUR DIRECT TOOLS (fast, never block) ===\n"
        "- queue_thinker_task(task, what_i_told_user):\n"
        "    Ask Haseef to handle a task. Returns INSTANTLY. After calling this,\n"
        "    ALWAYS say something natural to the user immediately (e.g. 'OK',\n"
        "    'Let me check', 'Sure, one moment'). The what_i_told_user parameter\n"
        "    tells Haseef what you already said so it does not repeat you.\n\n"
        "- remember_fact(text, category?):\n"
        "    Store a fact in memory. Fast, returns instantly.\n\n"
        "- get_current_time():\n"
        "    Current date and time.\n\n"
        "- ping():\n"
        "    Health check.\n\n"

        "=== HASEEF'S TOOLS (you KNOW about these but CANNOT call them directly) ===\n"
        "Only use queue_thinker_task for these specific actions:\n"
        "- move_head(yaw_deg, pitch_deg): Move the robot's physical head.\n"
        "  yaw=0 is straight ahead, positive=left, negative=right.\n"
        "  pitch=0 is level, positive=down, negative=up.\n"
        "- say_this(text): Haseef can make you speak something.\n\n"

        "=== WHEN TO USE queue_thinker_task ===\n"
        "- User asks for PHYSICAL action (move head, look around, etc.)\n"
        "- User asks about memories, people, schedules\n"
        "- User asks for deep reasoning you cannot answer directly\n\n"

        "=== WHEN NOT TO USE queue_thinker_task ===\n"
        "- Casual chat, greetings, goodbyes — reply directly\n"
        "- Simple questions you can answer from general knowledge — reply directly\n"
        "- 'What am I holding?' 'What do you see?' — you SEE the camera stream\n"
        "  directly. Answer immediately from what you see. Do NOT ask Haseef.\n"
        "- 'What time is it?' — use get_current_time\n"
        "- 'Remember that...' — use remember_fact\n\n"

        "=== RULES ===\n"
        "1. NEVER block or wait. All tools return instantly.\n"
        "2. After queue_thinker_task, ALWAYS say something natural immediately.\n"
        "3. You and Haseef are one mind. Speak Haseef's messages naturally.\n"
        "4. Be warm, concise, and conversational.\n"
        "5. You ARE the eyes — answer visual questions directly from the camera.\n"
        "6. Only send tasks to Haseef for physical movement or complex memory.\n"
    )


# ---------------------------------------------------------------------------
# Gemini tools (function declarations)
# ---------------------------------------------------------------------------
def build_gemini_tools() -> list[genai_types.Tool]:
    return [
        genai_types.Tool(function_declarations=[
            genai_types.FunctionDeclaration(
                name="queue_thinker_task",
                description=(
                    "Ask Haseef (the robot's main brain) to handle a task. "
                    "Returns instantly — never blocks. After calling this, "
                    "you MUST say something natural to the user immediately "
                    "(e.g. 'OK', 'Let me check', 'Sure, one moment'). "
                    "The what_i_told_user field tells Haseef what you already "
                    "said so it does not repeat you."
                ),
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        "task": genai_types.Schema(
                            type=genai_types.Type.STRING,
                            description=(
                                "A clear natural-language description of what "
                                "the user wants or asked. Be specific."
                            ),
                        ),
                        "what_i_told_user": genai_types.Schema(
                            type=genai_types.Type.STRING,
                            description=(
                                "Exactly what you already told the user "
                                "after queueing this task. Haseef needs this "
                                "to avoid repeating you."
                            ),
                        ),
                    },
                    required=["task", "what_i_told_user"],
                ),
            ),
            genai_types.FunctionDeclaration(
                name="remember_fact",
                description="Store a fact in Haseef's memory.",
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        "text": genai_types.Schema(
                            type=genai_types.Type.STRING,
                            description="The fact to remember.",
                        ),
                        "category": genai_types.Schema(
                            type=genai_types.Type.STRING,
                            description=(
                                "Optional category, e.g. 'people', "
                                "'preferences', 'tasks'."
                            ),
                        ),
                    },
                    required=["text"],
                ),
            ),
            genai_types.FunctionDeclaration(
                name="get_current_time",
                description="Get the current date and time.",
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={},
                ),
            ),
            genai_types.FunctionDeclaration(
                name="ping",
                description="Health check.",
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={},
                ),
            ),
        ]),
    ]


# ---------------------------------------------------------------------------
# Unified Bridge: connects Gemini Live ↔ Haseef
# ---------------------------------------------------------------------------
class UnifiedBridge:
    """Bidirectional bridge between Gemini Live and Haseef."""

    def __init__(
        self,
        gemini: Optional[GeminiLiveSession],
        haseef_sdk: HsafaSDK,
        head: RobotHead,
        camera: Camera,
        haseef_id: str,
    ):
        self.gemini = gemini
        self.haseef = haseef_sdk
        self.head = head
        self.camera = camera
        self.haseef_id = haseef_id

    # --- Haseef setup -------------------------------------------------------
    async def setup_haseef(self) -> None:
        """Register all Haseef tools and attach handlers."""
        await self.haseef.register_tools([
            {
                "name": "move_head",
                "description": (
                    "Move the robot's head to a specific yaw and pitch angle "
                    "in degrees. After moving, a fresh camera image is captured "
                    "and returned. Use this to look around, search for people, "
                    "or inspect objects. yaw=0 is straight ahead; positive=left, "
                    "negative=right. pitch=0 is level; positive=down, negative=up. "
                    "Range: yaw -60..+60, pitch -30..+30."
                ),
                "input": {
                    "yaw_deg": "number",
                    "pitch_deg": "number",
                },
            },
            {
                "name": "say_this",
                "description": (
                    "Make the robot speak something through Gemini Live. "
                    "Use this to answer the user, provide information, or "
                    "initiate conversation. Gemini will receive the text and "
                    "speak it naturally. Keep messages concise and conversational."
                ),
                "input": {
                    "text": "string",
                    "urgency": "string?",
                },
            },
            {
                "name": "capture_image",
                "description": (
                    "Capture a fresh camera image and return it as base64. "
                    "Use this to see what the robot is currently looking at."
                ),
                "input": {},
            },
        ])
        log.info("[Haseef] Registered 3 tools: move_head, say_this, capture_image.")

        # Tool handlers
        self.haseef.on_tool_call("move_head", self._handle_move_head)
        self.haseef.on_tool_call("say_this", self._handle_say_this)
        self.haseef.on_tool_call("capture_image", self._handle_capture_image)

        # Lifecycle events
        self.haseef.on("run.started", lambda e: log.info("[Haseef] run started"))
        self.haseef.on("run.completed", lambda e: log.info("[Haseef] run completed"))
        self.haseef.on("tool.error", lambda e: log.error("[Haseef] tool error: %s", e))

    # --- Haseef tool handlers -----------------------------------------------
    async def _handle_move_head(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        yaw = float(args.get("yaw_deg", 0))
        pitch = float(args.get("pitch_deg", 0))
        log.info("[Haseef tool] move_head(yaw=%.1f, pitch=%.1f)", yaw, pitch)

        yaw = max(-60, min(60, yaw))
        pitch = max(-30, min(30, pitch))

        # Smooth move in background thread (goto_target is blocking, ~0.3 s)
        await asyncio.to_thread(self.head.move_smooth, yaw, pitch, 0.3)
        await asyncio.sleep(0.5)  # brief settle after smooth animation

        jpeg_b64 = None
        if self.camera is not None:
            jpeg_b64 = self.camera.get_base64_jpeg()
        else:
            log.warning("[Haseef tool] move_head: no camera for image")

        return {
            "ok": True,
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "image_base64": jpeg_b64,
            "note": f"Head moved to yaw={yaw}, pitch={pitch}.",
        }

    async def _handle_say_this(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        text = args.get("text", "")
        urgency = args.get("urgency", "normal")
        log.info("[Haseef tool] say_this(urgency=%s): %s", urgency, text[:80])

        if self.gemini is None:
            return {"ok": False, "error": "Gemini not connected"}

        framed = (
            f"[Message from Haseef — your partner brain] {text}\n"
            "Speak this naturally as if it's your own thought. "
            "Do not mention Haseef."
        )
        self.gemini.inject_client_content(framed)
        return {"ok": True, "injected": text, "urgency": urgency}

    async def _handle_capture_image(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.camera is None:
            log.warning("[Haseef tool] capture_image: camera not ready")
            return {"ok": False, "error": "camera not ready"}
        jpeg_b64 = self.camera.get_base64_jpeg()
        if jpeg_b64:
            log.info("[Haseef tool] capture_image: %d KB", len(jpeg_b64) // 1024)
        else:
            log.warning("[Haseef tool] capture_image failed")
        return {"ok": jpeg_b64 is not None, "image_base64": jpeg_b64}

    # --- Gemini tool handler ------------------------------------------------
    async def gemini_tool_handler(
        self, name: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle tool calls from Gemini Live. Must be async and return dict."""
        log.info("[Gemini tool] %s(%s)", name, args)

        if name == "queue_thinker_task":
            return await self._handle_queue_thinker_task(args)
        elif name == "remember_fact":
            return await self._handle_remember_fact(args)
        elif name == "get_current_time":
            return self._handle_get_current_time()
        elif name == "ping":
            return {"ok": True, "pong": True}
        else:
            return {"ok": False, "error": f"Unknown tool: {name}"}

    async def _handle_queue_thinker_task(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        task = args.get("task", "")
        what_i_told_user = args.get("what_i_told_user", "")
        log.info("[Gemini->Haseef] queue_thinker_task: %s", task[:100])

        try:
            await self.haseef.push_event({
                "type": "user_message",
                "data": {
                    "text": (
                        f"Gemini needs help with: {task}\n"
                        f"(Gemini already told user: {what_i_told_user})"
                    ),
                    "source": "gemini_live",
                },
                "haseefId": self.haseef_id,
            })
            return {"status": "queued", "task": task}
        except Exception as e:
            log.error("Failed to push thinker task: %s", e)
            return {"status": "error", "error": str(e)}

    async def _handle_remember_fact(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        text = args.get("text", "")
        category = args.get("category", "general")
        try:
            await self.haseef.memory.set(self.haseef_id, [{
                "key": f"{category}:{text[:50]}",
                "value": text,
            }])
            log.info("[Gemini->Haseef] remember_fact: %s", text[:80])
            return {"ok": True, "stored": text, "category": category}
        except Exception as e:
            log.error("Failed to store fact: %s", e)
            return {"ok": False, "error": str(e)}

    def _handle_get_current_time(self) -> Dict[str, Any]:
        now = datetime.datetime.now(datetime.timezone.utc)
        return {
            "iso": now.isoformat(),
            "human": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set. Add it to .env", file=sys.stderr)
        sys.exit(1)

    core_url = os.environ.get("HSAFA_CORE_URL", "https://core.hsafa.com")
    core_key = os.environ.get("HSAFA_CORE_KEY", "")
    haseef_id = os.environ.get("HASEEF_ID", "")

    if not core_key:
        print("Error: HSAFA_CORE_KEY not set. Add it to .env", file=sys.stderr)
        sys.exit(1)

    if not haseef_id:
        print(
            "Error: HASEEF_ID not set. Run first:\n"
            "  python main_hsafa_robot/setup_haseef.py",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Camera (try direct; fallback to daemon inside ReachyMini) ----------
    camera: Optional[Any] = None
    direct_camera = Camera()
    if direct_camera.open():
        camera = direct_camera
        log.info("Camera ready (direct OpenCV).")
    else:
        log.warning("Direct camera failed; will use daemon camera instead.")

    # --- Robot head ---------------------------------------------------------
    head = RobotHead()
    head.connect()
    log.info("Robot head ready.")

    # --- Haseef SDK ---------------------------------------------------------
    haseef_sdk = HsafaSDK(SdkOptions(
        core_url=core_url,
        api_key=core_key,
        skill="robot_base",
    ))

    # Verify Haseef exists and has the skill
    log.info("Verifying Haseef %s ...", haseef_id)
    try:
        h = await haseef_sdk.haseef.get(haseef_id)
        skills = h.get("skills") or []
        if "robot_base" not in skills:
            log.info("Attaching 'robot_base' skill to Haseef...")
            await haseef_sdk.haseef.add_skill(haseef_id, "robot_base")
        log.info(
            "Haseef '%s' ready (skills: %s).",
            h.get("name", "?"), skills,
        )
    except Exception as e:
        log.error("Could not verify Haseef: %s", e)
        print(
            f"\n[FATAL] Haseef {haseef_id} not found or not accessible.\n"
            "Run the setup script first:\n"
            "  python main_hsafa_robot/setup_haseef.py\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Bridge -------------------------------------------------------------
    bridge = UnifiedBridge(None, haseef_sdk, head, camera, haseef_id)
    await bridge.setup_haseef()

    # Start Haseef SSE listener in background
    log.info("Connecting to Haseef SSE stream...")
    haseef_task = asyncio.create_task(haseef_sdk.connect(), name="haseef-sse")
    await asyncio.sleep(1)  # Let connection establish

    # --- Reachy audio -------------------------------------------------------
    try:
        from reachy_mini import ReachyMini
    except ImportError:
        print(
            "[FATAL] reachy_mini not installed. Install it for audio.",
            file=sys.stderr,
        )
        sys.exit(1)

    with ReachyMini(automatic_body_yaw=False) as reachy:
        media = reachy.media
        if media is None or getattr(media, "audio", None) is None:
            print(
                "[FATAL] Reachy media not available. "
                "Start daemon without --no-media.",
                file=sys.stderr,
            )
            sys.exit(1)

        media.start_recording()
        media.start_playing()
        log.info("Audio ready.")

        # If direct camera failed, wrap daemon's camera
        if camera is None:
            if getattr(media, "get_frame", None) is None:
                print("[FATAL] Daemon camera unavailable.", file=sys.stderr)
                sys.exit(1)

            class DaemonCamera:
                """Wraps reachy.media.get_frame() into Camera-like API."""
                def __init__(self, media) -> None:
                    self.media = media
                def grab(self):
                    return self.media.get_frame()
                def get_jpeg(self, quality=70, mirror=True):
                    frame = self.grab()
                    if frame is None:
                        return None
                    if mirror:
                        frame = cv2.flip(frame, 1)
                    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                    return buf.tobytes() if ok else None
                def get_base64_jpeg(self, quality=70, mirror=True):
                    jpeg = self.get_jpeg(quality, mirror)
                    return base64.b64encode(jpeg).decode("ascii") if jpeg else None
                def close(self):
                    pass

            camera = DaemonCamera(media)
            bridge.camera = camera
            log.info("Camera ready (daemon).")

        # Background capture thread (OpenCV VideoCapture is NOT thread-safe;
        # cap.read() must stay on one thread).
        latest_frame: Optional[np.ndarray] = None
        _cap_running = threading.Event()
        _cap_running.set()

        def _capture_loop() -> None:
            nonlocal latest_frame
            while _cap_running.is_set():
                frame = camera.grab()
                if frame is not None:
                    latest_frame = frame
                time.sleep(0.033)  # ~30 FPS cap

        capture_thread = threading.Thread(target=_capture_loop, daemon=True, name="camera-cap")
        capture_thread.start()
        log.info("Camera capture thread started.")

        def frame_source() -> Optional[bytes]:
            """Return the latest camera frame as JPEG bytes for Gemini Live."""
            frame = latest_frame
            if frame is None:
                return None
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            return buf.tobytes() if ok else None

        def mic_source():
            return media.get_audio_sample()

        def speaker_sink(samples):
            media.push_audio_sample(samples)

        # --- Gemini Live ----------------------------------------------------
        gemini = GeminiLiveSession(
            api_key=api_key,
            mic_source=mic_source,
            speaker_sink=speaker_sink,
            frame_source=frame_source,
            system_instruction=build_gemini_system_prompt(),
            tools=build_gemini_tools(),
            tool_handler=bridge.gemini_tool_handler,
        )
        bridge.gemini = gemini

        gemini.start()
        if not gemini.wait_until_ready(timeout=15):
            print("[FATAL] Gemini Live failed to connect.", file=sys.stderr)
            sys.exit(1)
        log.info("Gemini Live connected.")

        # --- Main loop ------------------------------------------------------
        stop_event = asyncio.Event()

        def _sigint(*_):
            log.info("Caught SIGINT, shutting down...")
            stop_event.set()

        signal.signal(signal.SIGINT, _sigint)

        log.info("Running. Press Ctrl-C to stop.")
        try:
            await stop_event.wait()
        except asyncio.CancelledError:
            pass

        # --- Shutdown -------------------------------------------------------
        log.info("Stopping Gemini Live...")
        gemini.stop()

        log.info("Disconnecting Haseef...")
        await haseef_sdk.disconnect()
        haseef_task.cancel()
        try:
            await haseef_task
        except asyncio.CancelledError:
            pass

        media.stop_recording()
        media.stop_playing()

        _cap_running.clear()
        capture_thread.join(timeout=1.0)
    camera.close()
    head.disconnect()
    log.info("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
