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
import httpx
import numpy as np

from dotenv import load_dotenv
from google.genai import types as genai_types

# Allow imports from repo root (e.g. hsafa_robot, hsafa_voice_vision)
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from hsafa_robot.gemini_live import GeminiLiveSession
from hsafa_voice_vision import Camera, RobotController
from hsafa_sdk import HsafaSDK, SdkOptions
from main_hsafa_robot.scheduler_skill import SchedulerSkill

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
        "- look_around(yaw_deg, pitch_deg): Move the robot's head and get a fresh\n"
        "  camera image. Haseef uses this to SEE.\n"
        "  yaw=0 is straight ahead, positive=left, negative=right.\n"
        "  pitch=0 is level, positive=down, negative=up.\n"
        "- set_head_pose(yaw_deg, pitch_deg): Move the robot's physical head\n"
        "  WITHOUT getting an image. Use for simple positioning.\n"
        "  yaw=0 is straight ahead, positive=left, negative=right.\n"
        "  pitch=0 is level, positive=down, negative=up.\n"
        "- say_this(text): Haseef can make you speak something.\n"
        "- show_expression(emotion): Show an animated emotion clip (motion + sound).\n"
        "  Valid: amazed, angry, anxiety, attentive, boredom, calming, cheerful, come,\n"
        "  confused, contempt, curious, dance, disgusted, displeased, downcast, dying, electric,\n"
        "  enthusiastic, exhausted, fear, frustrated, furious, go_away, grateful, happy, helpful,\n"
        "  impatient, indifferent, inquiring, irritated, laughing, lonely, lost, love, neutral,\n"
        "  no, oops, proud, rage, relief, reprimand, resigned, sad, scared, serenity, shy,\n"
        "  sleep, success, surprised, thoughtful, tired, uncertain, uncomfortable, understanding,\n"
        "  welcoming, yes.\n"
        "- create_schedule(description, type, scheduled_at?, cron_expression?, timezone?):\n"
        "  Haseef can schedule tasks to run later (one-time or recurring cron).\n"
        "  When the schedule fires, Haseef receives a schedule.triggered event.\n"
        "- list_schedules(): List all active schedules.\n"
        "- cancel_schedule(schedule_id): Cancel an active schedule.\n\n"

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
        gemini: Optional[Any],
        haseef_sdk: Any,
        robot: RobotController,
        camera: Any,
        haseef_id: str,
        main_loop: Optional[asyncio.AbstractEventLoop] = None,
        scheduler: Optional[SchedulerSkill] = None,
    ) -> None:
        self.gemini = gemini
        self.haseef_sdk = haseef_sdk
        self.robot = robot
        self.camera = camera
        self.haseef_id = haseef_id
        self._main_loop = main_loop
        self._say_lock = asyncio.Lock()
        self._pending_says: list[str] = []
        self.scheduler = scheduler

    # --- Haseef setup -------------------------------------------------------
    async def setup_haseef(self) -> None:
        """Register all Haseef tools and attach handlers."""
        await self.haseef_sdk.register_tools([
            {
                "name": "create_schedule",
                "description": (
                    "Create a schedule so Haseef handles a task later. "
                    "Use 'one_time' with a scheduled_at epoch timestamp, or "
                    "'recurring' with a cron_expression."
                ),
                "input": {
                    "description": "string",
                    "type": "string",
                    "scheduled_at": "number?",
                    "cron_expression": "string?",
                    "timezone": "string?",
                },
            },
            {
                "name": "list_schedules",
                "description": "List all active schedules.",
                "input": {},
            },
            {
                "name": "cancel_schedule",
                "description": "Cancel an active schedule by its id.",
                "input": {
                    "schedule_id": "string",
                },
            },
            {
                "name": "look_around",
                "description": (
                    "Move the robot's head to a specific yaw and pitch angle "
                    "in degrees, then capture and return a fresh camera image. "
                    "Use this when you need to SEE something: look around, search "
                    "for people, inspect objects, or verify what's in front of you. "
                    "yaw=0 is straight ahead; positive=left, negative=right. "
                    "pitch=0 is level; positive=down, negative=up. "
                    "Range: yaw -60..+60, pitch -30..+30."
                ),
                "input": {
                    "yaw_deg": "number",
                    "pitch_deg": "number",
                },
            },
            {
                "name": "set_head_pose",
                "description": (
                    "Move the robot's head to a specific yaw and pitch angle "
                    "in degrees. No image is returned. Use this for simple "
                    "physical positioning: face forward, look left/right, nod, "
                    "or adjust posture when you do NOT need to see the result. "
                    "yaw=0 is straight ahead; positive=left, negative=right. "
                    "pitch=0 is level; positive=down, negative=up. "
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
                "description": "Capture a fresh camera image and return it as base64 JPEG. Quality is optional (1-100, default 70).",
                "input": {"quality": "integer?"},
            },
            {
                "name": "show_expression",
                "description": (
                    "Show an animated emotion clip with head motion and sound. "
                    "Valid names: amazed, angry, anxiety, attentive, boredom, calming, cheerful, come, "
                    "confused, contempt, curious, dance, disgusted, displeased, downcast, dying, electric, "
                    "enthusiastic, exhausted, fear, frustrated, furious, go_away, grateful, happy, helpful, "
                    "impatient, indifferent, inquiring, irritated, laughing, lonely, lost, love, neutral, "
                    "no, oops, proud, rage, relief, reprimand, resigned, sad, scared, serenity, shy, "
                    "sleep, success, surprised, thoughtful, tired, uncertain, uncomfortable, understanding, "
                    "welcoming, yes."
                ),
                "input": {
                    "emotion": "string",
                },
            },
        ])
        log.info("[Haseef] Registered 8 tools: create_schedule, list_schedules, cancel_schedule, look_around, set_head_pose, say_this, capture_image, show_expression.")

        # Tool handlers
        self.haseef_sdk.on_tool_call("create_schedule", self._handle_create_schedule)
        self.haseef_sdk.on_tool_call("list_schedules", self._handle_list_schedules)
        self.haseef_sdk.on_tool_call("cancel_schedule", self._handle_cancel_schedule)
        self.haseef_sdk.on_tool_call("look_around", self._handle_look_around)
        self.haseef_sdk.on_tool_call("set_head_pose", self._handle_set_head_pose)
        self.haseef_sdk.on_tool_call("say_this", self._handle_say_this)
        self.haseef_sdk.on_tool_call("capture_image", self._handle_capture_image)
        self.haseef_sdk.on_tool_call("show_expression", self._handle_show_expression)

        # Lifecycle events
        self.haseef_sdk.on("run.started", lambda e: log.info("[Haseef] run started"))
        self.haseef_sdk.on("run.completed", lambda e: log.info("[Haseef] run completed: %s", e))
        self.haseef_sdk.on("tool.error", lambda e: log.error("[Haseef] tool error: %s", e))
        self.haseef_sdk.on("tool.call", lambda e: log.info("[Haseef] tool.call: %s", e))
        self.haseef_sdk.on("tool.result", lambda e: log.info("[Haseef] tool.result: %s", e))
        self.haseef_sdk.on("tool.input.start", lambda e: log.info("[Haseef] tool.input.start: %s", e))
        self.haseef_sdk.on("tool.input.delta", lambda e: log.info("[Haseef] tool.input.delta: %s", e))
        self.haseef_sdk.on("thought", lambda e: log.info("[Haseef] thought: %s", e))

    # --- Haseef tool handlers -----------------------------------------------
    async def _push_image_event(self, jpeg_b64: str, note: str = "") -> None:
        """Push an event with the image as an attachment so Haseef's LLM can see it."""
        if not jpeg_b64:
            return
        try:
            await self._run_sdk_on_main(self.haseef_sdk.push_event({
                "type": "user_message",
                "data": {"text": note or "Robot vision update."},
                "attachments": [
                    {
                        "type": "image",
                        "mimeType": "image/jpeg",
                        "base64": jpeg_b64,
                    }
                ],
                "haseefId": self.haseef_id,
            }))
            log.info("[ImageEvent] Pushed image to Haseef (%d KB)", len(jpeg_b64) // 1024)
        except Exception as e:
            log.error("[ImageEvent] Failed to push image: %s", e)

    async def _push_schedule_event(self, schedule) -> None:
        """Push a schedule.triggered event to Haseef so it can react."""
        try:
            await self._run_sdk_on_main(self.haseef_sdk.push_event({
                "type": "schedule.triggered",
                "data": {
                    "scheduleId": schedule.id,
                    "description": schedule.description,
                    "type": schedule.type,
                    "cronExpression": schedule.cron_expression,
                    "timezone": schedule.timezone,
                    "lastRunAt": schedule.last_run_at,
                    "formattedContext": self._build_schedule_context(schedule),
                },
                "haseefId": self.haseef_id,
            }))
            log.info("[ScheduleEvent] Pushed '%s' to Haseef", schedule.description)
        except Exception as e:
            log.error("[ScheduleEvent] Failed to push: %s", e)

    def _build_schedule_context(self, schedule) -> str:
        lines = [
            "[SCHEDULED TASK TRIGGERED]",
            f"Description: {schedule.description}",
            f"Type: {schedule.type}",
        ]
        if schedule.cron_expression:
            lines.append(f"Cron: {schedule.cron_expression}")
        if schedule.timezone:
            lines.append(f"Timezone: {schedule.timezone}")
        lines.append("\nThis scheduled task has fired. Please carry out the described action.")
        return "\n".join(lines)

    # --- Scheduler handlers (Haseef tools) ---------------------------------
    async def _handle_create_schedule(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        description = args.get("description", "")
        type_ = args.get("type", "one_time")
        scheduled_at = args.get("scheduled_at")
        cron = args.get("cron_expression")
        timezone = args.get("timezone", "UTC")

        if self.scheduler is None:
            return {"ok": False, "error": "Scheduler not available"}

        try:
            sid = self.scheduler.add_schedule(
                description=description,
                type=type_,
                scheduled_at=scheduled_at,
                cron_expression=cron,
                timezone=timezone,
            )
            return {
                "ok": True,
                "schedule_id": sid,
                "type": type_,
                "next_run_at": scheduled_at,
            }
        except Exception as exc:
            log.error("[Haseef tool] create_schedule failed: %s", exc)
            return {"ok": False, "error": str(exc)}

    async def _handle_list_schedules(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.scheduler is None:
            return {"ok": False, "error": "Scheduler not available"}
        return {"ok": True, "schedules": self.scheduler.list_schedules()}

    async def _handle_cancel_schedule(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        sid = args.get("schedule_id", "")
        if self.scheduler is None:
            return {"ok": False, "error": "Scheduler not available"}
        ok = self.scheduler.cancel_schedule(sid)
        return {"ok": ok, "schedule_id": sid}

    async def _handle_show_expression(self, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        name = args.get("emotion", "neutral")
        valid = self.robot.list_expressions()
        if name not in valid:
            return {"ok": False, "error": f"Unknown emotion '{name}'. Valid: {valid}"}
        # play_move blocks; run in thread so Haseef gets the result promptly
        await asyncio.to_thread(self.robot.show_expression, name)
        log.info("[Haseef tool] show_expression: %s", name)
        return {"ok": True, "emotion": name}

    async def _handle_look_around(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        yaw = float(args.get("yaw_deg", 0))
        pitch = float(args.get("pitch_deg", 0))
        log.info("[Haseef tool] look_around(yaw=%.1f, pitch=%.1f)", yaw, pitch)

        yaw = max(-60, min(60, yaw))
        pitch = max(-30, min(30, pitch))

        await asyncio.to_thread(self.robot.move_head, yaw, pitch, 0.3)
        await asyncio.sleep(0.5)

        jpeg_b64 = self.camera.get_base64_jpeg() if self.camera else None
        if jpeg_b64:
            await self._push_image_event(
                jpeg_b64,
                note=f"Head moved to yaw={yaw}, pitch={pitch}. Here is what I see.",
            )
        return {
            "ok": True,
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "image_base64": jpeg_b64,
            "note": f"Head moved to yaw={yaw}, pitch={pitch}.",
        }

    async def _handle_set_head_pose(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        yaw = float(args.get("yaw_deg", 0))
        pitch = float(args.get("pitch_deg", 0))
        log.info("[Haseef tool] set_head_pose(yaw=%.1f, pitch=%.1f)", yaw, pitch)

        yaw = max(-60, min(60, yaw))
        pitch = max(-30, min(30, pitch))

        await asyncio.to_thread(self.robot.move_head, yaw, pitch, 0.3)
        return {
            "ok": True,
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "note": f"Head pose set to yaw={yaw}, pitch={pitch}.",
        }

    async def _handle_say_this(
        self, args: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        text = args.get("text", "")
        urgency = args.get("urgency", "normal")
        log.info("[Haseef tool] say_this(urgency=%s): %s", urgency, text[:80])

        gemini = self.gemini
        if gemini is None:
            return {"ok": False, "error": "Gemini Live not connected"}

        framed = (
            f"[Message from Haseef — your partner brain] {text}\n"
            "Speak this naturally as if it's your own thought. "
            "Do not mention Haseef."
        )

        async with self._say_lock:
            if gemini.is_speaking.is_set():
                self._pending_says.append(framed)
                log.info("[Haseef tool] say_this queued (Gemini speaking)")
                return {"ok": True, "status": "queued", "text": text, "urgency": urgency}

        gemini.inject_client_content(framed)
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
            await self._push_image_event(
                jpeg_b64,
                note="Here is what I see right now.",
            )
        else:
            log.warning("[Haseef tool] capture_image failed")
        return {"ok": jpeg_b64 is not None, "image_base64": jpeg_b64}

    # --- Gemini tool handler ------------------------------------------------
    async def _run_sdk_on_main(self, coro):
        """Run an SDK coroutine on the main event loop from Gemini's thread."""
        if self._main_loop is None or self._main_loop.is_closed():
            raise RuntimeError("Main event loop not available for SDK call")
        future = asyncio.run_coroutine_threadsafe(coro, self._main_loop)
        return await asyncio.wrap_future(future)

    async def gemini_tool_handler(
        self, name: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle tool calls from Gemini Live. Must be async and return dict."""
        log.info("[Gemini tool] %s%s", name, args)

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

        last_err = None
        for attempt in range(1, 4):
            try:
                await self._run_sdk_on_main(self.haseef_sdk.push_event({
                    "type": "user_message",
                    "data": {
                        "text": (
                            f"Gemini needs help with: {task}\n"
                            f"(Gemini already told user: {what_i_told_user})"
                        ),
                        "source": "gemini_live",
                    },
                    "haseefId": self.haseef_id,
                }))
                if attempt > 1:
                    log.info("[Gemini->Haseef] push_event succeeded on attempt %d", attempt)
                return {"status": "queued", "task": task}
            except httpx.TimeoutException as e:
                last_err = e
                log.warning(
                    "[Gemini->Haseef] push_event timeout (attempt %d/3): %s",
                    attempt, type(e).__name__,
                )
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                last_err = e
                log.error(
                    "[Gemini->Haseef] push_event failed (attempt %d/3): %r",
                    attempt, e,
                )
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt)

        import traceback
        log.error("[Gemini->Haseef] push_event failed after 3 attempts.")
        log.error("Traceback: %s", traceback.format_exc())
        return {
            "status": "error",
            "error": f"{type(last_err).__name__}: {last_err}",
        }

    async def _handle_remember_fact(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        text = args.get("text", "")
        category = args.get("category", "general")
        try:
            await self._run_sdk_on_main(self.haseef_sdk.memory.set(self.haseef_id, [{
                "key": f"{category}:{text[:50]}",
                "value": text,
            }]))
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

    # --- Robot controller ---------------------------------------------------
    robot = None

    # --- Haseef SDK ---------------------------------------------------------
    haseef_sdk = HsafaSDK(SdkOptions(
        core_url=core_url,
        api_key=core_key,
        skill="robot_base",
    ))
    # Patch default 5 s timeout — server can be slow.
    # We monkey-patch _request rather than replacing _client to avoid
    # httpx asyncio event-loop binding issues.
    _sdk_timeout = httpx.Timeout(30.0, connect=10.0)
    _orig_request = haseef_sdk._request

    async def _request_with_timeout(self, method, path, body=None):
        url = f"{self.core_url}{path}"
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        response = await self._client.request(
            method, url, headers=headers, json=body, timeout=_sdk_timeout
        )
        if not response.is_success:
            raise Exception(
                f"{method} {path} failed ({response.status_code}): {response.text}"
            )
        if response.status_code == 204 or not response.content:
            return None
        if "application/json" in response.headers.get("content-type", ""):
            return response.json()
        return None

    haseef_sdk._request = _request_with_timeout.__get__(haseef_sdk, HsafaSDK)

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
        cfg = h.get("configJson") or {}
        log.info("Haseef configJson: %s", cfg)
    except Exception as e:
        log.error("Could not verify Haseef: %s", e)
        print(
            f"\n[FATAL] Haseef {haseef_id} not found or not accessible.\n"
            "Run the setup script first:\n"
            "  python main_hsafa_robot/setup_haseef.py\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Scheduler ------------------------------------------------------------
    main_loop = asyncio.get_running_loop()

    def on_schedule_trigger(schedule):
        if main_loop and not main_loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                bridge._push_schedule_event(schedule), main_loop
            )

    scheduler = SchedulerSkill(on_trigger=on_schedule_trigger)
    scheduler.start(poll_interval=30.0)
    log.info("Scheduler ready.")

    # --- Bridge -------------------------------------------------------------
    bridge = UnifiedBridge(
        None, haseef_sdk, robot, camera, haseef_id,
        main_loop=main_loop,
        scheduler=scheduler,
    )
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

        # Create robot controller now that we have the ReachyMini instance
        robot = RobotController(reachy)
        robot.start_idle()
        bridge.robot = robot
        log.info("Robot controller ready.")

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
            if robot:
                robot.notify_audio(samples)
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

        async def _drain_say_queue():
            while not stop_event.is_set():
                await asyncio.sleep(0.15)
                if gemini.is_speaking.is_set():
                    continue
                async with bridge._say_lock:
                    if bridge._pending_says:
                        text = bridge._pending_says.pop(0)
                        gemini.inject_client_content(text)
                        log.info("[SayDrain] injected queued text: %s", text[:80])

        say_drain_task = asyncio.create_task(_drain_say_queue(), name="say-drain")

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
        say_drain_task.cancel()
        try:
            await say_drain_task
        except asyncio.CancelledError:
            pass

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

        if robot:
            robot.stop_idle()

        if scheduler:
            scheduler.stop()
    camera.close()
    log.info("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
