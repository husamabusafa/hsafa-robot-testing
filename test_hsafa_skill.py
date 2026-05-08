"""test_hsafa_skill.py — General test skill for HSAFA Core.

Registers general-purpose tools (not robot-related) so a haseef can call them.
Tools: echo, get_current_time, calculate, get_random_fact.

Usage:
    ./.venv/bin/python test_hsafa_skill.py

Env:
    HSAFA_CORE_URL   (default: https://core.hsafa.com)
    HSAFA_CORE_KEY   (default: the prod key below)
    SKILL_NAME       (default: general_tester)
"""
from __future__ import annotations

import asyncio
import datetime
import json
import os
import random
import signal
import sys

from hsafa_sdk import HsafaSDK, SdkOptions

CORE_URL = os.environ.get("HSAFA_CORE_URL", "https://core.hsafa.com")
API_KEY = os.environ.get(
    "HSAFA_CORE_KEY",
    "sk_prod_7f2e8d9c4b3a6f1e0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0",
)
SKILL_NAME = os.environ.get("SKILL_NAME", "general_tester")

# ── Sample facts for the demo tool ─────────────────────────────────────
FACTS = [
    "Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still edible.",
    "Octopuses have three hearts, nine brains, and blue blood.",
    "Bananas are berries, but strawberries are not.",
    "A day on Venus is longer than a year on Venus.",
    "Wombat poop is cube-shaped.",
    "Sharks have been around longer than trees.",
    "The Eiffel Tower can grow taller in the summer due to heat expansion.",
    "A cloud can weigh more than a million pounds.",
]


async def main():
    sdk = HsafaSDK(
        SdkOptions(
            core_url=CORE_URL,
            api_key=API_KEY,
            skill=SKILL_NAME,
        )
    )

    # ── 1. Register tools ──────────────────────────────────────────────
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
    print(f"[{SKILL_NAME}] Registered 4 general tools with Core at {CORE_URL}")

    # ── 2. Handlers ────────────────────────────────────────────────────
    async def handle_echo(args, ctx):
        msg = args.get("message", "")
        print(f"  [echo] haseef={ctx['haseef']['name']} msg={msg!r}")
        return {"echo": msg, "timestamp": datetime.datetime.utcnow().isoformat()}

    async def handle_time(args, ctx):
        now = datetime.datetime.now(datetime.timezone.utc)
        print(f"  [time] haseef={ctx['haseef']['name']}")
        return {
            "iso": now.isoformat(),
            "human": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "unix": now.timestamp(),
        }

    async def handle_calc(args, ctx):
        expr = str(args.get("expression", ""))
        print(f"  [calc] haseef={ctx['haseef']['name']} expr={expr!r}")
        try:
            # Safe eval: only allow basic math
            allowed = {"__builtins__": None}
            allowed.update({k: v for k, v in vars(__import__("math")).items() if not k.startswith("_")})
            result = eval(expr, allowed, {})  # noqa: S307
            return {"expression": expr, "result": result}
        except Exception as e:
            return {"error": f"Could not evaluate '{expr}': {e}"}

    async def handle_fact(args, ctx):
        fact = random.choice(FACTS)
        print(f"  [fact] haseef={ctx['haseef']['name']}")
        return {"fact": fact}

    sdk.on_tool_call("echo", handle_echo)
    sdk.on_tool_call("get_current_time", handle_time)
    sdk.on_tool_call("calculate", handle_calc)
    sdk.on_tool_call("get_random_fact", handle_fact)

    # ── 3. Lifecycle listeners ─────────────────────────────────────────
    def on_run_started(e):
        print(f"  [event] run started for haseef={e.get('haseef', {}).get('name')}")

    def on_run_completed(e):
        dur = e.get("durationMs")
        print(f"  [event] run completed in {dur}ms")

    def on_tool_error(e):
        print(f"  [event] tool error: {e.get('toolName')} — {e.get('error')}")

    sdk.on("run.started", on_run_started)
    sdk.on("run.completed", on_run_completed)
    sdk.on("tool.error", on_tool_error)

    # ── 4. Connect SSE (blocks) ────────────────────────────────────────
    print(f"[{SKILL_NAME}] Connecting SSE stream… Press Ctrl-C to stop.")
    try:
        await sdk.connect()
    except asyncio.CancelledError:
        pass
    finally:
        await sdk.disconnect()
        print(f"[{SKILL_NAME}] Disconnected.")


if __name__ == "__main__":
    # Graceful shutdown
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: loop.create_task(asyncio.sleep(0)) or sys.exit(0))
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
