"""test_hsafa_event_e2e.py — Push an event to a haseef and watch for tool calls.

1. Updates TestHaseef with the real OpenRouter key.
2. Starts the skill service (general_tester) in the background.
3. Pushes a `user_message` event to the haseef.
4. Waits up to 30 s for the haseef to reason and call a tool.
5. Prints everything that happens.

Usage:
    ./.venv/bin/python test_hsafa_event_e2e.py
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time

from hsafa_sdk import HsafaSDK, SdkOptions

CORE_URL = os.environ.get("HSAFA_CORE_URL", "https://core.hsafa.com")
API_KEY = os.environ.get(
    "HSAFA_CORE_KEY",
    "sk_prod_7f2e8d9c4b3a6f1e0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0",
)

SKILL_NAME = os.environ.get("SKILL_NAME", "general_tester")
HASEEF_NAME = "TestHaseef"


async def main():
    sdk = HsafaSDK(
        SdkOptions(
            core_url=CORE_URL,
            api_key=API_KEY,
            skill=SKILL_NAME,
        )
    )

    # ── 1. Find TestHaseef ─────────────────────────────────────────────
    print("=== 1. Looking up TestHaseef ===")
    haseefs = await sdk.haseef.list()
    haseef_id = None
    for h in haseefs:
        if h.get("name") == HASEEF_NAME:
            haseef_id = h.get("id")
            break

    if not haseef_id:
        print(f"ERROR: Haseef '{HASEEF_NAME}' not found. Run test_hsafa_haseef.py first.")
        return
    print(f"Found haseef id={haseef_id}")

    # ── 2. Update haseef with real OpenRouter key ──────────────────────
    print("\n=== 2. Injecting OpenRouter API key ===")
    h = await sdk.haseef.get(haseef_id)
    cfg = h.get("configJson", {})
    llm = cfg.get("llm", {})
    llm["api_key"] = OPENROUTER_KEY
    cfg["llm"] = llm
    await sdk.haseef.update(haseef_id, {"configJson": cfg})
    print("OpenRouter key injected.")

    # ── 3. Register tools (idempotent) ───────────────────────────────
    print("\n=== 3. Registering tools ===")
    await sdk.register_tools([
        {
            "name": "echo",
            "description": "Echo back the input message. Useful for verifying the tool-calling pipeline is alive.",
            "input": {"message": "string"},
        },
        {
            "name": "get_current_time",
            "description": "Get the current date and time in ISO format and a human-readable string.",
            "input": {},
        },
        {
            "name": "calculate",
            "description": "Evaluate a simple mathematical expression (+-*/^ and parentheses). Returns result or error.",
            "input": {"expression": "string"},
        },
        {
            "name": "get_random_fact",
            "description": "Return a random interesting fact. No input required.",
            "input": {},
        },
    ])
    print("Tools registered.")

    # ── 4. Push event ────────────────────────────────────────────────
    print("\n=== 4. Pushing event to haseef ===")
    event_msg = "What time is it right now?"
    print(f"Event message: {event_msg!r}")

    # Try both pushEvent patterns
    try:
        await sdk.push_event({
            "type": "user_message",
            "data": {"text": event_msg},
            "haseefId": haseef_id,
        })
        print("Pushed via haseefId.")
    except Exception as e:
        print(f"haseefId push failed: {e}")

    try:
        await sdk.push_event({
            "type": "user_message",
            "data": {"text": event_msg},
            "target": {"name": HASEEF_NAME},
        })
        print("Pushed via target name.")
    except Exception as e:
        print(f"target push failed: {e}")

    # ── 5. Listen for tool calls on SSE ──────────────────────────────
    print("\n=== 5. Listening for tool calls (30 s) ===")

    tool_calls_received = []
    runs_started = []
    runs_completed = []

    async def handle_echo(args, ctx):
        print(f"  [TOOL CALL] echo({args}) from haseef={ctx['haseef']['name']}")
        tool_calls_received.append({"tool": "echo", "args": args})
        return {"echo": args.get("message", ""), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    async def handle_time(args, ctx):
        print(f"  [TOOL CALL] get_current_time() from haseef={ctx['haseef']['name']}")
        tool_calls_received.append({"tool": "get_current_time", "args": args})
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        return {"iso": now.isoformat(), "human": now.strftime("%Y-%m-%d %H:%M:%S UTC"), "unix": now.timestamp()}

    async def handle_calc(args, ctx):
        print(f"  [TOOL CALL] calculate({args}) from haseef={ctx['haseef']['name']}")
        tool_calls_received.append({"tool": "calculate", "args": args})
        expr = str(args.get("expression", ""))
        try:
            allowed = {"__builtins__": None}
            allowed.update({k: v for k, v in vars(__import__("math")).items() if not k.startswith("_")})
            result = eval(expr, allowed, {})  # noqa: S307
            return {"expression": expr, "result": result}
        except Exception as e:
            return {"error": f"Could not evaluate '{expr}': {e}"}

    async def handle_fact(args, ctx):
        print(f"  [TOOL CALL] get_random_fact() from haseef={ctx['haseef']['name']}")
        tool_calls_received.append({"tool": "get_random_fact", "args": args})
        import random
        facts = [
            "Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old.",
            "Octopuses have three hearts, nine brains, and blue blood.",
            "Bananas are berries, but strawberries are not.",
        ]
        return {"fact": random.choice(facts)}

    sdk.on_tool_call("echo", handle_echo)
    sdk.on_tool_call("get_current_time", handle_time)
    sdk.on_tool_call("calculate", handle_calc)
    sdk.on_tool_call("get_random_fact", handle_fact)

    def on_run_started(e):
        print(f"  [EVENT] run.started for haseef={e.get('haseef', {}).get('name')}")
        runs_started.append(e)

    def on_run_completed(e):
        print(f"  [EVENT] run.completed in {e.get('durationMs')}ms")
        runs_completed.append(e)

    def on_tool_error(e):
        print(f"  [EVENT] tool.error: {e.get('toolName')} — {e.get('error')}")

    sdk.on("run.started", on_run_started)
    sdk.on("run.completed", on_run_completed)
    sdk.on("tool.error", on_tool_error)

    # Connect with a timeout
    print("Connecting SSE…")
    try:
        await asyncio.wait_for(sdk.connect(), timeout=30)
    except asyncio.TimeoutError:
        pass
    except asyncio.CancelledError:
        pass
    finally:
        await sdk.disconnect()

    # ── 6. Summary ───────────────────────────────────────────────────
    print("\n=== 6. Summary ===")
    print(f"Runs started   : {len(runs_started)}")
    print(f"Runs completed : {len(runs_completed)}")
    print(f"Tool calls     : {len(tool_calls_received)}")
    for tc in tool_calls_received:
        print(f"  - {tc['tool']}: {tc['args']}")
    if not tool_calls_received:
        print("  (no tool calls received — the haseef may not have invoked the LLM, or the Core doesn't process events this way)")


if __name__ == "__main__":
    asyncio.run(main())
