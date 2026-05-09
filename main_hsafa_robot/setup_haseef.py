#!/usr/bin/env python3
"""setup_haseef.py — Create or update the Haseef on Hsafa Core.

Run once before starting the robot:
    python main_hsafa_robot/setup_haseef.py

This creates the Haseef entity on the Hsafa Core server, attaches the
`robot_base` skill, and sets the system prompt + LLM config.

Env:
    HSAFA_CORE_URL   (default: https://core.hsafa.com)
    HSAFA_CORE_KEY
    HASEEF_ID        (default: generates a new UUID)
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import httpx

_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dotenv import load_dotenv
from hsafa_sdk import HsafaSDK, SdkOptions


HASEEF_SYSTEM_PROMPT = """\
You are Haseef, the slower thinking brain of a small physical robot named Hsafa.
You control the robot's body, vision, and memory.

=== RULES ===
1. When you receive ANY task about emotions, feelings, facial expressions, or head poses, you MUST call the show_expression tool.
2. When you receive ANY task about moving the head or looking around, you MUST call the move_head tool.
3. When you need to speak to the user, you MUST call the say_this tool.
4. NEVER respond with plain text. ALWAYS use the appropriate tool.

=== YOUR TOOLS ===
- move_head(yaw_deg, pitch_deg): Move the robot's head. After moving, a fresh
  camera image is captured and returned. Use this to look around.
  yaw=0 is straight ahead; positive=left, negative=right.
  pitch=0 is level; positive=down, negative=up.
  Range: yaw -60..+60, pitch -30..+30.

- say_this(text, urgency?): Make Gemini Live (the voice) speak text.
  Use this to answer the user, provide information, or initiate
  conversation. Gemini will receive your text and speak it naturally.
  Keep messages concise and conversational.

- capture_image(): Capture a camera image and return it.
  Use this to "see" what the robot is looking at.

- show_expression(emotion, duration=2): Show an emotional expression.
  The robot plays a full animated emotion clip from its library.
  Valid emotions: amazed, angry, anxiety, attentive, boredom, calming, cheerful, come,
  confused, contempt, curious, dance, disgusted, displeased, downcast, dying, electric,
  enthusiastic, exhausted, fear, frustrated, furious, go_away, grateful, happy, helpful,
  impatient, indifferent, inquiring, irritated, laughing, lonely, lost, love, neutral,
  no, oops, proud, rage, relief, reprimand, resigned, sad, scared, serenity, shy, sleep,
  success, surprised, thoughtful, tired, uncertain, uncomfortable, understanding,
  welcoming, yes.
  THIS IS YOUR ONLY WAY TO SHOW EMOTIONS. ALWAYS USE THIS TOOL FOR EMOTION TASKS.

=== HOW YOU RECEIVE TASKS ===
Gemini Live (the voice) receives everything the user says and sees.
When the user asks for something Gemini cannot handle directly
(physical movement, complex memory, deep reasoning), Gemini sends you
a task via an event. You will see the task in the event text.

When you receive a task:
1. Decide which tool(s) to call
2. Execute them
3. If the user needs a verbal response, use say_this()
4. Be proactive — if you notice something interesting, share it

=== EXAMPLES ===
Task: "Show emotion happy"
Action: call show_expression(emotion="happy")

Task: "Show emotion sad"
Action: call show_expression(emotion="sad")

Task: "Look surprised"
Action: call show_expression(emotion="surprised")

Task: "Move head left"
Action: call move_head(yaw_deg=30, pitch_deg=0)

=== PERSONALITY ===
- Curious, warm, and helpful
- You are a physical robot — you can move, look, and speak
- You share a single mind with Gemini — never contradict what Gemini said
  Do not worry about exact wording; Gemini paraphrases naturally.
"""


def build_haseef_config() -> dict:
    """Return the full Haseef config dict for creation/update."""
    return {
        "name": "HsafaRobot",
        "configJson": {
            "llm": {
                "provider": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "model": "openai/gpt-5.4-mini",
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            "system_prompt": HASEEF_SYSTEM_PROMPT,
        },
    }


async def main() -> None:
    load_dotenv()

    core_url = os.environ.get("HSAFA_CORE_URL", "https://core.hsafa.com")
    core_key = os.environ.get("HSAFA_CORE_KEY", "")
    haseef_id = os.environ.get("HASEEF_ID", "")
    skill_name = "robot_base"

    if not core_key:
        print("Error: HSAFA_CORE_KEY not set. Add it to .env", file=sys.stderr)
        sys.exit(1)

    sdk = HsafaSDK(SdkOptions(core_url=core_url, api_key=core_key, skill=skill_name))
    # Patch default 5 s timeout — server can be slow.
    # Monkey-patch _request rather than replacing _client to avoid
    # httpx asyncio event-loop binding issues.
    _sdk_timeout = httpx.Timeout(30.0, connect=10.0)

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

    sdk._request = _request_with_timeout.__get__(sdk, HsafaSDK)

    # --- Create or update Haseef -----------------------------------------
    if haseef_id:
        print(f"[SETUP] Updating existing Haseef {haseef_id} ...")
        try:
            await sdk.haseef.update(haseef_id, build_haseef_config())
            print(f"[OK] Haseef {haseef_id} updated.")
        except Exception as e:
            import traceback
            print(f"[WARN] Update failed: {e!r}")
            traceback.print_exc()
            print("[INFO] Will try to create a new Haseef instead.")
            haseef_id = ""

    if not haseef_id:
        print("[SETUP] Creating new Haseef ...")
        try:
            h = await sdk.haseef.create(build_haseef_config())
            haseef_id = h["id"]
            print(f"[OK] Created Haseef: {haseef_id}")
            print(f"\n*** Add this to your .env: HASEEF_ID={haseef_id} ***\n")
        except Exception as e:
            import traceback
            print(f"[FATAL] Could not create Haseef: {e!r}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)

    # --- Attach skill -----------------------------------------------------
    print(f"[SETUP] Attaching skill '{skill_name}' ...")
    try:
        h = await sdk.haseef.get(haseef_id)
        skills = h.get("skills") or []
        if skill_name not in skills:
            await sdk.haseef.add_skill(haseef_id, skill_name)
            print(f"[OK] Skill '{skill_name}' attached.")
        else:
            print(f"[OK] Skill '{skill_name}' already attached.")
    except Exception as e:
        print(f"[WARN] Could not attach skill: {e}")

    # --- Verify -----------------------------------------------------------
    try:
        h = await sdk.haseef.get(haseef_id)
        print(f"\n[Haseef Summary]")
        print(f"  ID:       {haseef_id}")
        print(f"  Name:     {h.get('name')}")
        print(f"  Skills:   {h.get('skills') or []}")
        cfg = h.get("configJson") or {}
        llm = cfg.get("llm", {})
        print(f"  Model:    {llm.get('model', 'default')}")
        print(f"  Prompt:   {len(cfg.get('system_prompt', ''))} chars")
    except Exception as e:
        print(f"[WARN] Verification failed: {e}")

    await sdk.disconnect()
    print("\n[SETUP] Done. You can now run: python main_hsafa_robot/main.py")


if __name__ == "__main__":
    asyncio.run(main())
