"""test_hsafa_mock_core.py — Minimal mock HSAFA Core v7 for local testing.

Spins up a tiny HTTP+SSE server that mimics enough of Core's API so the
SDK skill service and haseef tests can run end-to-end locally.

Endpoints:
    GET  /api/v7/haseefs
    POST /api/v7/haseefs
    GET  /api/v7/haseefs/<id>
    PATCH /api/v7/haseefs/<id>
    DELETE /api/v7/haseefs/<id>
    GET  /api/v7/haseefs/<id>/profile
    PATCH /api/v7/haseefs/<id>/profile
    GET  /api/v7/haseefs/<id>/status
    GET  /api/v7/memory/<id>/semantic
    POST /api/v7/memory/<id>/semantic
    GET  /api/v7/memory/<id>/semantic/search
    GET  /api/v7/runs
    GET  /api/v7/runs/<id>
    PUT  /api/v7/skills/<name>/tools
    GET  /api/v7/skills/<name>/actions/stream  (SSE)
    POST /api/v7/actions/<id>/result
    POST /api/v7/events
    POST /api/v7/haseefs/<id>/chat  (triggers a mock run)

Usage:
    ./.venv/bin/python test_hsafa_mock_core.py

Then in another terminal:
    HSAFA_CORE_URL=http://localhost:3456 ./.venv/bin/python test_hsafa_skill.py
    HSAFA_CORE_URL=http://localhost:3456 ./.venv/bin/python test_hsafa_haseef.py
"""
from __future__ import annotations

import asyncio
import json
import random
import os
import time
import uuid
from typing import Any, Dict, List

from aiohttp import web

HASEEFS: Dict[str, Dict[str, Any]] = {}
MEMORIES: Dict[str, List[Dict[str, Any]]] = {}
RUNS: Dict[str, Dict[str, Any]] = {}
SKILL_TOOLS: Dict[str, List[Dict[str, Any]]] = {}
ACTION_QUEUES: Dict[str, asyncio.Queue] = {}

API_KEY = "sk_prod_7f2e8d9c4b3a6f1e0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0"


def _ok(data):
    return web.json_response(data)


def _check_auth(request: web.Request):
    key = request.headers.get("x-api-key", "")
    if key != API_KEY:
        raise web.HTTPUnauthorized(text=json.dumps({"error": "Invalid API key"}))


# ── Haseefs ────────────────────────────────────────────────────────────
async def list_haseefs(request: web.Request):
    _check_auth(request)
    return _ok({"haseefs": list(HASEEFS.values())})


async def create_haseef(request: web.Request):
    _check_auth(request)
    body = await request.json()
    hid = str(uuid.uuid4())
    haseef = {
        "id": hid,
        "name": body.get("name", "Unnamed"),
        "description": body.get("description", ""),
        "configJson": body.get("configJson", {}),
        "profileJson": body.get("profileJson", {}),
        "skills": body.get("skills", []),
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    HASEEFS[hid] = haseef
    MEMORIES[hid] = []
    return _ok({"haseef": haseef})


async def get_haseef(request: web.Request):
    _check_auth(request)
    hid = request.match_info["id"]
    h = HASEEFS.get(hid)
    if not h:
        raise web.HTTPNotFound()
    return _ok({"haseef": h})


async def patch_haseef(request: web.Request):
    _check_auth(request)
    hid = request.match_info["id"]
    h = HASEEFS.get(hid)
    if not h:
        raise web.HTTPNotFound()
    body = await request.json()
    h.update(body)
    h["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return _ok({"haseef": h})


async def delete_haseef(request: web.Request):
    _check_auth(request)
    hid = request.match_info["id"]
    HASEEFS.pop(hid, None)
    return web.Response(status=204)


async def get_profile(request: web.Request):
    _check_auth(request)
    hid = request.match_info["id"]
    h = HASEEFS.get(hid)
    if not h:
        raise web.HTTPNotFound()
    return _ok({"profile": h.get("profileJson", {})})


async def patch_profile(request: web.Request):
    _check_auth(request)
    hid = request.match_info["id"]
    h = HASEEFS.get(hid)
    if not h:
        raise web.HTTPNotFound()
    body = await request.json()
    current = h.get("profileJson", {})
    current.update(body)
    h["profileJson"] = current
    return _ok({"profile": current})


async def get_status(request: web.Request):
    _check_auth(request)
    hid = request.match_info["id"]
    if hid not in HASEEFS:
        raise web.HTTPNotFound()
    return _ok({"online": True, "lastSeen": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


# ── Memory ───────────────────────────────────────────────────────────────
async def list_memory(request: web.Request):
    _check_auth(request)
    hid = request.match_info["id"]
    return _ok({"memories": MEMORIES.get(hid, [])})


async def set_memory(request: web.Request):
    _check_auth(request)
    hid = request.match_info["id"]
    body = await request.json()
    for m in body.get("memories", []):
        entry = {
            "id": str(uuid.uuid4()),
            "haseefId": hid,
            "key": m.get("key"),
            "value": m.get("value"),
            "importance": m.get("importance", 3),
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        MEMORIES.setdefault(hid, []).append(entry)
    return _ok({"ok": True})


async def search_memory(request: web.Request):
    _check_auth(request)
    hid = request.match_info["id"]
    q = request.query.get("q", "").lower()
    limit = int(request.query.get("limit", 20))
    results = [
        m for m in MEMORIES.get(hid, [])
        if q in m.get("key", "").lower() or q in m.get("value", "").lower()
    ][:limit]
    return _ok({"results": results})


# ── Runs ─────────────────────────────────────────────────────────────────
async def list_runs(request: web.Request):
    _check_auth(request)
    hid = request.query.get("haseefId")
    limit = int(request.query.get("limit", 20))
    runs = [r for r in RUNS.values() if (not hid or r.get("haseefId") == hid)]
    runs = sorted(runs, key=lambda x: x.get("startedAt", ""), reverse=True)[:limit]
    return _ok({"runs": runs})


async def get_run(request: web.Request):
    _check_auth(request)
    rid = request.match_info["id"]
    r = RUNS.get(rid)
    if not r:
        raise web.HTTPNotFound()
    return _ok({"run": r})


# ── Skills / SSE ─────────────────────────────────────────────────────────
async def register_tools(request: web.Request):
    _check_auth(request)
    skill = request.match_info["name"]
    body = await request.json()
    SKILL_TOOLS[skill] = body.get("tools", [])
    ACTION_QUEUES.setdefault(skill, asyncio.Queue())
    print(f"[mock] Skill '{skill}' registered {len(body.get('tools', []))} tool(s)")
    return _ok({"ok": True})


async def sse_stream(request: web.Request):
    _check_auth(request)
    skill = request.match_info["name"]
    queue = ACTION_QUEUES.setdefault(skill, asyncio.Queue())

    async def stream():
        print(f"[mock] SSE client connected for skill='{skill}'")
        try:
            while True:
                msg = await queue.get()
                yield f"data: {json.dumps(msg)}\n\n".encode("utf-8")
        except asyncio.CancelledError:
            print(f"[mock] SSE client disconnected for skill='{skill}'")
            raise

    return web.Response(
        status=200,
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
        body=stream(),
    )


async def post_result(request: web.Request):
    _check_auth(request)
    aid = request.match_info["id"]
    body = await request.json()
    print(f"[mock] Action {aid} result: {json.dumps(body, default=str)[:200]}")
    return _ok({"ok": True})


async def push_event(request: web.Request):
    _check_auth(request)
    body = await request.json()
    print(f"[mock] Event pushed: {json.dumps(body, default=str)[:200]}")
    return _ok({"ok": True})


# ── Chat / run trigger ───────────────────────────────────────────────────
async def haseef_chat(request: web.Request):
    _check_auth(request)
    hid = request.match_info["id"]
    body = await request.json()
    msg = body.get("message", "")
    h = HASEEFS.get(hid)
    if not h:
        raise web.HTTPNotFound()

    run_id = str(uuid.uuid4())
    start_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Create run record
    run = {
        "id": run_id,
        "haseefId": hid,
        "status": "running",
        "triggerSkill": body.get("skill"),
        "triggerType": "chat",
        "startedAt": start_ts,
        "completedAt": None,
        "durationMs": None,
        "summary": None,
        "tokensUsed": None,
        "toolCallCount": 0,
    }
    RUNS[run_id] = run

    # Broadcast run.started to all skill SSE queues
    for skill, queue in ACTION_QUEUES.items():
        await queue.put({
            "type": "run.started",
            "data": {"runId": run_id, "haseef": {"name": h.get("name"), "id": hid}},
        })

    # Simulate haseef choosing a tool based on message keywords
    chosen_tool = None
    tool_args = {}
    lower_msg = msg.lower()
    if "time" in lower_msg:
        chosen_tool = "get_current_time"
    elif any(w in lower_msg for w in ["calc", "calculate", "math", "+", "-", "*", "/"]):
        chosen_tool = "calculate"
        tool_args = {"expression": "2 + 2"}
    elif "fact" in lower_msg:
        chosen_tool = "get_random_fact"
    elif "echo" in lower_msg or "hello" in lower_msg:
        chosen_tool = "echo"
        tool_args = {"message": msg}

    if chosen_tool:
        # Find which skill has this tool
        target_skill = None
        for skill, tools in SKILL_TOOLS.items():
            if any(t.get("name") == chosen_tool for t in tools):
                target_skill = skill
                break

        if target_skill:
            action_id = str(uuid.uuid4())
            queue = ACTION_QUEUES.get(target_skill)
            if queue:
                run["toolCallCount"] = 1
                await queue.put({
                    "type": "action",
                    "actionId": action_id,
                    "toolName": chosen_tool,
                    "args": tool_args,
                    "haseef": {"id": hid, "name": h.get("name"), "profile": h.get("profileJson", {})},
                })
                print(f"[mock] Dispatched tool '{chosen_tool}' to skill='{target_skill}' for haseef='{h.get('name')}'")
        else:
            print(f"[mock] No skill registered tool '{chosen_tool}'")
    else:
        print(f"[mock] No matching tool for message: {msg!r}")

    # Complete run after a short delay
    await asyncio.sleep(0.5)
    run["status"] = "completed"
    run["completedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run["durationMs"] = random.randint(300, 1200)
    run["summary"] = f"Processed: {msg[:50]}"

    for skill, queue in ACTION_QUEUES.items():
        await queue.put({
            "type": "run.completed",
            "data": {
                "runId": run_id,
                "haseef": {"name": h.get("name"), "id": hid},
                "summary": run["summary"],
                "durationMs": run["durationMs"],
            },
        })

    return _ok({"run": run})


# ── App setup ────────────────────────────────────────────────────────────
def make_app():
    app = web.Application()
    app.router.add_get("/api/v7/haseefs", list_haseefs)
    app.router.add_post("/api/v7/haseefs", create_haseef)
    app.router.add_get("/api/v7/haseefs/{id}", get_haseef)
    app.router.add_patch("/api/v7/haseefs/{id}", patch_haseef)
    app.router.add_delete("/api/v7/haseefs/{id}", delete_haseef)
    app.router.add_get("/api/v7/haseefs/{id}/profile", get_profile)
    app.router.add_patch("/api/v7/haseefs/{id}/profile", patch_profile)
    app.router.add_get("/api/v7/haseefs/{id}/status", get_status)
    app.router.add_get("/api/v7/memory/{id}/semantic", list_memory)
    app.router.add_post("/api/v7/memory/{id}/semantic", set_memory)
    app.router.add_get("/api/v7/memory/{id}/semantic/search", search_memory)
    app.router.add_get("/api/v7/runs", list_runs)
    app.router.add_get("/api/v7/runs/{id}", get_run)
    app.router.add_put("/api/v7/skills/{name}/tools", register_tools)
    app.router.add_get("/api/v7/skills/{name}/actions/stream", sse_stream)
    app.router.add_post("/api/v7/actions/{id}/result", post_result)
    app.router.add_post("/api/v7/events", push_event)
    app.router.add_post("/api/v7/haseefs/{id}/chat", haseef_chat)
    return app


if __name__ == "__main__":
    port = int(os.environ.get("MOCK_PORT", "3456"))
    print(f"[mock] HSAFA Core mock starting on http://localhost:{port}")
    print(f"[mock] API key: {API_KEY[:20]}...")
    web.run_app(make_app(), host="localhost", port=port)
