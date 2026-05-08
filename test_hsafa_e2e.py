"""test_hsafa_e2e.py — Full end-to-end test: push event, listen for tool calls.

Starts the skill service, pushes a user_message event, and waits up to 30s
for the haseef to process and call tools via SSE.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import os
import subprocess
import sys
import time

from hsafa_sdk import HsafaSDK, SdkOptions

CORE_URL = "https://core.hsafa.com"
API_KEY = "sk_prod_7f2e8d9c4b3a6f1e0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0"
SKILL = "general_tester"
HASEEF_ID = "b8f0ead5-036c-4a0b-8afb-e56314acdb9f"
SKILL_LOG = "/tmp/skill_test_e2e.log"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()


async def main():
    # ── 1. Start skill service in background ───────────────────────
    log("=== 1. Starting skill service ===")
    env = os.environ.copy()
    env["HSAFA_CORE_URL"] = CORE_URL
    env["HSAFA_CORE_KEY"] = API_KEY

    with open(SKILL_LOG, "w") as f:
        proc = subprocess.Popen(
            [sys.executable, "-u", "test_hsafa_skill.py"],
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd="/Users/Husam/Dev/hsafa-robot",
            env=env,
        )
    log(f"Skill PID: {proc.pid}")

    await asyncio.sleep(3)
    log("Skill log so far:")
    with open(SKILL_LOG) as f:
        print(f.read())

    # ── 2. Push event ────────────────────────────────────────────────
    log("=== 2. Pushing event to haseef ===")
    sdk = HsafaSDK(SdkOptions(core_url=CORE_URL, api_key=API_KEY, skill=SKILL))

    try:
        await sdk.push_event({
            "type": "user_message",
            "data": {"text": "What time is it right now?"},
            "haseefId": HASEEF_ID,
        })
        log("Event pushed successfully.")
    except Exception as e:
        log(f"Push failed: {e}")
        proc.terminate()
        return

    # ── 3. Register handlers and listen on SSE ───────────────────────
    log("=== 3. Listening on SSE for tool calls (max 30s) ===")

    tool_calls = []
    runs_started = []
    runs_completed = []

    async def handle_echo(args, ctx):
        log(f"TOOL CALL: echo({json.dumps(args)})")
        tool_calls.append({"tool": "echo", "args": args})
        return {"echo": args.get("message", "")}

    async def handle_time(args, ctx):
        log(f"TOOL CALL: get_current_time()")
        tool_calls.append({"tool": "get_current_time", "args": args})
        now = datetime.datetime.now(datetime.timezone.utc)
        return {"iso": now.isoformat(), "human": now.strftime("%Y-%m-%d %H:%M:%S UTC")}

    async def handle_calc(args, ctx):
        log(f"TOOL CALL: calculate({json.dumps(args)})")
        tool_calls.append({"tool": "calculate", "args": args})
        return {"expression": args.get("expression", ""), "result": "mock"}

    async def handle_fact(args, ctx):
        log(f"TOOL CALL: get_random_fact()")
        tool_calls.append({"tool": "get_random_fact", "args": args})
        return {"fact": "Honey never spoils."}

    sdk.on_tool_call("echo", handle_echo)
    sdk.on_tool_call("get_current_time", handle_time)
    sdk.on_tool_call("calculate", handle_calc)
    sdk.on_tool_call("get_random_fact", handle_fact)

    sdk.on("run.started", lambda e: (log(f"EVENT: run.started"), runs_started.append(e)))
    sdk.on("run.completed", lambda e: (log(f"EVENT: run.completed in {e.get('durationMs')}ms"), runs_completed.append(e)))
    sdk.on("tool.error", lambda e: log(f"EVENT: tool.error {e.get('toolName')}: {e.get('error')}"))

    # Connect with a generous timeout
    try:
        await asyncio.wait_for(sdk.connect(), timeout=30)
    except asyncio.TimeoutError:
        log("SSE listen timed out (30s)")
    except Exception as e:
        log(f"SSE error: {e}")
    finally:
        await sdk.disconnect()

    # ── 4. Summary ───────────────────────────────────────────────────
    log("=== 4. Summary ===")
    log(f"Runs started:   {len(runs_started)}")
    log(f"Runs completed: {len(runs_completed)}")
    log(f"Tool calls:     {len(tool_calls)}")
    for tc in tool_calls:
        log(f"  - {tc['tool']}: {tc['args']}")

    # ── 5. Check runs via API ────────────────────────────────────────
    log("=== 5. Checking runs via API ===")
    try:
        runs = await sdk.runs.list(haseef_id=HASEEF_ID, limit=5)
        log(f"Found {len(runs)} run(s)")
        for r in runs:
            log(f"  Run {r.get('id')}: status={r.get('status')} duration={r.get('durationMs')}ms")
    except Exception as e:
        log(f"Run check failed: {e}")

    # ── 6. Cleanup ───────────────────────────────────────────────────
    log("=== 6. Cleanup ===")
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log("Skill service stopped.")


if __name__ == "__main__":
    asyncio.run(main())
