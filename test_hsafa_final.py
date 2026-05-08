"""Final clean test — one skill service, push event, wait for result."""
import asyncio
import datetime
import os
import sys
import time

from hsafa_sdk import HsafaSDK, SdkOptions

CORE_URL = "https://core.hsafa.com"
API_KEY = "sk_prod_7f2e8d9c4b3a6f1e0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0"
SKILL = "general_tester"
HASEEF_ID = "b8f0ead5-036c-4a0b-8afb-e56314acdb9f"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()


async def main():
    sdk = HsafaSDK(SdkOptions(core_url=CORE_URL, api_key=API_KEY, skill=SKILL))

    # -- Register tools --
    log("Registering tools...")
    await sdk.register_tools([
        {"name": "echo", "description": "Echo a message", "input": {"message": "string"}},
        {"name": "get_current_time", "description": "Get current UTC time", "input": {}},
        {"name": "calculate", "description": "Evaluate math expression", "input": {"expression": "string"}},
        {"name": "get_random_fact", "description": "Return a random fact", "input": {}},
    ])
    log("Tools registered.")

    # -- Track everything --
    tool_calls = []
    runs_started = []
    runs_completed = []

    async def on_echo(args, ctx):
        log(f"TOOL: echo({args.get('message','')!r})")
        tool_calls.append("echo")
        return {"echo": args.get("message", "")}

    async def on_time(args, ctx):
        log(f"TOOL: get_current_time()")
        tool_calls.append("get_current_time")
        now = datetime.datetime.now(datetime.timezone.utc)
        return {"iso": now.isoformat(), "human": now.strftime("%Y-%m-%d %H:%M:%S UTC")}

    async def on_calc(args, ctx):
        log(f"TOOL: calculate({args.get('expression','')!r})")
        tool_calls.append("calculate")
        return {"expression": args.get("expression", ""), "result": "evaluated"}

    async def on_fact(args, ctx):
        log(f"TOOL: get_random_fact()")
        tool_calls.append("get_random_fact")
        return {"fact": "Honey never spoils."}

    sdk.on_tool_call("echo", on_echo)
    sdk.on_tool_call("get_current_time", on_time)
    sdk.on_tool_call("calculate", on_calc)
    sdk.on_tool_call("get_random_fact", on_fact)

    sdk.on("run.started", lambda e: (log("EVENT: run.started"), runs_started.append(e)))
    sdk.on("run.completed", lambda e: (log(f"EVENT: run.completed in {e.get('durationMs')}ms"), runs_completed.append(e)))
    sdk.on("tool.error", lambda e: log(f"EVENT: tool.error {e.get('toolName')}: {e.get('error')}"))

    # -- Start listening in background --
    log("Connecting SSE stream...")
    listen_task = asyncio.create_task(sdk.connect())
    await asyncio.sleep(2)

    # -- Push event --
    log("Pushing event to haseef...")
    try:
        await sdk.push_event({
            "type": "user_message",
            "data": {"text": "Calculate 42 times 7 and tell me the current time too"},
            "haseefId": HASEEF_ID,
        })
        log("Event pushed.")
    except Exception as e:
        log(f"Push failed: {e}")
        listen_task.cancel()
        return

    # -- Wait for run to complete --
    log("Waiting for run to complete (max 20s)...")
    for _ in range(20):
        if runs_completed:
            break
        await asyncio.sleep(1)

    # -- Summary --
    log("=== Summary ===")
    log(f"Runs started:   {len(runs_started)}")
    log(f"Runs completed: {len(runs_completed)}")
    log(f"Tool calls:     {len(tool_calls)} - {tool_calls}")

    # -- Cleanup --
    log("Disconnecting...")
    await sdk.disconnect()
    try:
        listen_task.cancel()
        await listen_task
    except asyncio.CancelledError:
        pass
    log("Done.")


asyncio.run(main())
