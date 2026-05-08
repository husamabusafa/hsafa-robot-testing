"""Simple test: push event and listen for tool calls."""
import asyncio
import os
import time
from hsafa_sdk import HsafaSDK, SdkOptions

CORE_URL = "https://core.hsafa.com"
API_KEY = "sk_prod_7f2e8d9c4b3a6f1e0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0"
SKILL = "general_tester"

sdk = HsafaSDK(SdkOptions(core_url=CORE_URL, api_key=API_KEY, skill=SKILL))

tool_calls = []
runs_started = []
runs_completed = []

async def echo(args, ctx):
    print(f"  -> TOOL: echo({args})")
    tool_calls.append("echo")
    return {"echo": args.get("message", "")}

async def get_time(args, ctx):
    print(f"  -> TOOL: get_current_time()")
    tool_calls.append("get_current_time")
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    return {"iso": now.isoformat()}

sdk.on_tool_call("echo", echo)
sdk.on_tool_call("get_current_time", get_time)

sdk.on("run.started", lambda e: (print(f"  -> EVENT: run.started"), runs_started.append(e)))
sdk.on("run.completed", lambda e: (print(f"  -> EVENT: run.completed in {e.get('durationMs')}ms"), runs_completed.append(e)))
sdk.on("tool.error", lambda e: print(f"  -> EVENT: tool.error {e.get('toolName')}: {e.get('error')}"))

async def main():
    # Register tools
    print("Registering tools...")
    await sdk.register_tools([
        {"name": "echo", "description": "Echo a message", "input": {"message": "string"}},
        {"name": "get_current_time", "description": "Get current time", "input": {}},
    ])
    print("Tools registered.")

    # Push event
    print("\nPushing event to haseef...")
    try:
        await sdk.push_event({
            "type": "user_message",
            "data": {"text": "What time is it?"},
            "target": {"name": "TestHaseef"},
        })
        print("Event pushed.")
    except Exception as e:
        print(f"Push failed: {e}")

    # Listen on SSE for 20 seconds
    print("\nListening on SSE for 20s...")
    try:
        await asyncio.wait_for(sdk.connect(), timeout=20)
    except asyncio.TimeoutError:
        print("SSE timeout (expected)")
    finally:
        await sdk.disconnect()

    print(f"\nResults:")
    print(f"  Runs started:   {len(runs_started)}")
    print(f"  Runs completed: {len(runs_completed)}")
    print(f"  Tool calls:     {len(tool_calls)} ({tool_calls})")

asyncio.run(main())
