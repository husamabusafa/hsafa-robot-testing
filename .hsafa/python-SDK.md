# hsafa-sdk

Python SDK for **Hsafa Core v7**. Connect any service to a Haseef brain — register tools, handle tool calls over SSE, push events, and read/write the haseef's memory and profile.

## Install

```bash
pip install hsafa-sdk
```

Or from source:

```bash
cd sdks/python
pip install -e .
```

## Quick Start

```python
import asyncio
import os
from hsafa_sdk import HsafaSDK, SdkOptions

async def main():
    sdk = HsafaSDK(SdkOptions(
        core_url="http://localhost:3001",
        api_key=os.environ.get("HSAFA_CORE_KEY", "test-key"),
        skill="weather",
    ))

    # 1. Register tools
    await sdk.register_tools([{
        "name": "get_weather",
        "description": "Get current weather for a city",
        "input": {"city": "string", "units": "string?"},
    }])

    # 2. Handle tool calls
    async def handle_weather(args, ctx):
        print(f"{ctx['haseef']['name']} wants weather for {args.get('city')}")
        return {"temperature": 72, "conditions": "sunny", "city": args.get("city")}

    sdk.on_tool_call("get_weather", handle_weather)

    # 3. Listen to lifecycle events
    def on_run_started(e):
        print(f"Run started for {e['haseef']['name']}")

    sdk.on("run.started", on_run_started)

    # 4. Connect the SSE stream (blocks until disconnect)
    # For background usage alongside a web server:
    #     asyncio.create_task(sdk.connect())
    try:
        await sdk.connect()
    except KeyboardInterrupt:
        await sdk.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

## Namespaces

| Namespace | Methods |
|-----------|---------|
| `sdk.haseef` | `list()`, `get(id)`, `create(data)`, `update(id, patch)`, `delete(id)`, `get_profile(id)`, `update_profile(id, patch)`, `add_skill(id, name)`, `remove_skill(id, name)`, `status(id)` |
| `sdk.memory` | `list(id)`, `search(id, query, limit)`, `set(id, memories)`, `delete(id, keys)`, `episodes(id, limit)`, `search_episodes(id, query, limit)`, `social(id)`, `procedural(id)`, `stats(id)` |
| `sdk.runs` | `list(haseef_id?, status?, limit?)`, `get(run_id)` |

## License

MIT
