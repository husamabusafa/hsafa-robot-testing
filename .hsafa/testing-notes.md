# Hsafa Testing Notes

> Practical findings from testing Hsafa Core v7 (May 2026). Read this before writing new skills or tests.

---

## 1. Server Status

**URL:** `https://core.hsafa.com`

| Feature | Status | Notes |
|---------|--------|-------|
| Haseef CRUD | ✅ Working | Create, read, update, delete |
| Profile CRUD | ✅ Working | `get_profile`, `update_profile` |
| Memory (semantic) | ✅ Working | `list`, `set`, `search` |
| Runs API | ✅ Working | `list`, `get` — but runs auto-purge quickly |
| Skill registration | ✅ Working | `register_tools` via `PUT /skills/{name}/tools` |
| SSE stream | ✅ Working | `GET /skills/{name}/actions/stream` returns `text/event-stream` |
| Event push | ✅ Working | `POST /api/v7/events` triggers runs |
| Run execution | ✅ Working | LLM processes message → calls tools → returns results |
| Auth | ✅ Working | Uses `x-api-key` header with Core `SECRET_KEY` |

### Previous Issues (Now Fixed)
- ~~`500 Server auth not configured`~~ — Fixed
- ~~`503 no available server`~~ — Intermittent, retry works
- ~~Stale SSE connections stealing tool calls~~ — User-side issue, see §5

---

## 2. SDK Auth Pattern

**Single key only.** Hsafa v7 uses one Core `SECRET_KEY`. There are **no per-skill keys**.

```python
from hsafa_sdk import HsafaSDK, SdkOptions

sdk = HsafaSDK(SdkOptions(
    core_url="https://core.hsafa.com",
    api_key="sk_prod_...",  # Core's SECRET_KEY
    skill="my_skill_name",
))
```

Header sent: `x-api-key: <key>`

---

## 3. LLM Configuration

**The Core handles the OpenRouter key.** Do **NOT** put `api_key` in the haseef's `configJson.llm`. The Core server already has it configured.

```python
# ✅ Correct — no api_key in haseef config
await sdk.haseef.update(haseef_id, {
    "configJson": {
        "llm": {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "model": "openai/gpt-5.4-mini",  # or gpt-4o-mini, etc.
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        "system_prompt": "You are a helpful assistant...",
    }
})
```

**Models tested:**
- `openai/gpt-5.4-mini` — ✅ Supports tool calling
- `openai/gpt-4o-mini` — ✅ Supports tool calling

---

## 4. Tool Registration & SSE

```python
await sdk.register_tools([
    {
        "name": "get_current_time",
        "description": "Get current UTC time",
        "input": {},  # or inputSchema for complex schemas
    },
])

sdk.on_tool_call("get_current_time", async (args, ctx) => {
    return {"iso": datetime.utcnow().isoformat()}
})

await sdk.connect()  # Blocks on SSE stream
```

**Important:** `connect()` is **blocking**. Run it in a background task if you need to do other things:
```python
asyncio.create_task(sdk.connect())
```

---

## 5. The #1 Pitfall: Stale SSE Connections

**If multiple processes connect to the same skill name, the Core sends tool calls to the OLDEST connection.** This means stale skill service processes from previous tests will "steal" tool calls.

### Symptoms
- Event pushes succeed (HTTP 200)
- Run starts and completes
- **But your new process receives 0 tool calls**

### Fix
Always kill old skill processes before testing:

```bash
# macOS/Linux
pkill -f "test_hsafa_skill.py"
pkill -f "my_skill_service.py"

# Or use a unique skill name per test:
SKILL_NAME="test_$(date +%s)" python my_test.py
```

### Best Practice
Use the **same single process** to both register tools AND listen on SSE:
```python
# ✅ One process does everything
await sdk.register_tools([...])
sdk.on_tool_call("foo", handler)
await sdk.connect()  # This process receives all tool calls
```

---

## 6. Event Push → Run Flow

```python
# Push an event to trigger a haseef run
await sdk.push_event({
    "type": "user_message",
    "data": {"text": "What time is it?"},
    "haseefId": "<haseef-uuid>",
})
```

**What happens:**
1. HTTP 200 response with `{"triggered": true, "runId": "..."}`
2. Core queues a run for the haseef
3. LLM reads the message + available tools + system prompt
4. LLM decides which tools to call
5. Core dispatches tool calls over SSE to the skill service
6. Skill service executes handler and returns result
7. LLM gets results and formulates final response
8. Core emits `run.completed` event

**Latency:** ~2–5 seconds for simple tool calls.

---

## 7. Haseef System Prompt Matters

The LLM only calls tools if the **system prompt explicitly tells it to**.

```python
"system_prompt": (
    "You are a helpful assistant. You have access to tools: "
    "get_current_time, calculate, echo, get_random_fact. "
    "ALWAYS use a tool when the user asks for time, math, or facts."
)
```

Without explicit instructions, the LLM may answer directly without calling tools.

---

## 8. Testing Checklist

Before running a new test:

1. [ ] Kill all old skill processes (`pkill -f my_skill.py`)
2. [ ] Use a unique skill name OR verify no other connections
3. [ ] Ensure haseef has `skills: ["your_skill_name"]` attached
4. [ ] Ensure haseef LLM config has a model that supports tools
5. [ ] System prompt explicitly mentions available tools
6. [ ] Push event with clear trigger words ("what time", "calculate", etc.)
7. [ ] Wait 5–10 seconds for run completion
8. [ ] Check both tool calls AND run events

---

## 9. Quick Test Script Template

```python
import asyncio
from hsafa_sdk import HsafaSDK, SdkOptions

CORE_URL = "https://core.hsafa.com"
API_KEY = "sk_prod_..."
SKILL = "general_tester"
HASEEF_ID = "<your-haseef-id>"

async def main():
    sdk = HsafaSDK(SdkOptions(core_url=CORE_URL, api_key=API_KEY, skill=SKILL))

    # Register
    await sdk.register_tools([{
        "name": "ping",
        "description": "Reply with pong",
        "input": {},
    }])

    # Handle
    sdk.on_tool_call("ping", lambda args, ctx: {"pong": True})

    # Listen
    sdk.on("run.completed", lambda e: print(f"Run done in {e.get('durationMs')}ms"))

    # Connect + push
    task = asyncio.create_task(sdk.connect())
    await asyncio.sleep(1)

    await sdk.push_event({
        "type": "user_message",
        "data": {"text": "Ping"},
        "haseefId": HASEEF_ID,
    })

    await asyncio.sleep(5)
    await sdk.disconnect()
    task.cancel()

asyncio.run(main())
```

---

## 10. Mock Core (Local Testing)

For offline development, use `test_hsafa_mock_core.py` in this repo. It simulates:
- All CRUD endpoints
- SSE stream with tool dispatch
- Memory read/write
- Event push → run trigger → tool call loop

```bash
# Terminal 1: mock core
./.venv/bin/python test_hsafa_mock_core.py

# Terminal 2: skill + test
HSAFA_CORE_URL=http://localhost:3456 ./.venv/bin/python my_test.py
```

---

## Files in This Repo

| File | Purpose |
|------|---------|
| `test_hsafa_discovery.py` | Probe Core endpoints for connectivity |
| `test_hsafa_skill.py` | Standalone skill service with 4 general tools |
| `test_hsafa_haseef.py` | Haseef CRUD + memory + config tests |
| `test_hsafa_final.py` | **Recommended:** One-process e2e test (register + listen + push + verify) |
| `test_hsafa_mock_core.py` | Local mock server for offline testing |
| `update_haseef_model.py` | Update haseef LLM model config |
