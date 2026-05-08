"""test_hsafa_haseef.py — Create and test a haseef with OpenRouter model.

Creates a haseef configured to use openai/gpt-5.4-mini via OpenRouter,
attaches the general_tester skill, and verifies basic CRUD + tool readiness.

Usage:
    ./.venv/bin/python test_hsafa_haseef.py

Env:
    HSAFA_CORE_URL       (default: https://core.hsafa.com)
    HSAFA_CORE_KEY       (default: the prod key below)
    OPENROUTER_API_KEY   (for the haseef's LLM provider)
    SKILL_NAME           (default: general_tester)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

import httpx
from hsafa_sdk import HsafaSDK, SdkOptions

CORE_URL = os.environ.get("HSAFA_CORE_URL", "https://core.hsafa.com")
API_KEY = os.environ.get(
    "HSAFA_CORE_KEY",
    "sk_prod_7f2e8d9c4b3a6f1e0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0",
)
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
SKILL_NAME = os.environ.get("SKILL_NAME", "general_tester")
HASEEF_NAME = "TestHaseef"


def _model_config():
    """Build haseef config for OpenRouter + openai/gpt-5.4-mini."""
    return {
        "llm": {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "model": "openai/gpt-5.4-mini",
            "api_key": OPENROUTER_KEY or None,
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        "system_prompt": (
            "You are a helpful test assistant. You can use tools to get the time, "
            "calculate math, echo messages, and fetch random facts. "
            "Always prefer calling a tool when the user asks for something specific."
        ),
    }


async def main():
    sdk = HsafaSDK(
        SdkOptions(
            core_url=CORE_URL,
            api_key=API_KEY,
            skill=SKILL_NAME,
        )
    )

    print(f"\n=== HSAFA General Test: Haseef + Skill ===")
    print(f"Core URL : {CORE_URL}")
    print(f"Skill    : {SKILL_NAME}")
    print(f"Model    : openai/gpt-5.4-mini (OpenRouter)")

    # ── 1. Connectivity / list existing ────────────────────────────────
    print("\n--- 1. List existing haseefs ---")
    try:
        haseefs = await sdk.haseef.list()
        print(f"Existing haseefs: {len(haseefs)}")
        for h in haseefs:
            print(f"  - {h.get('name')} (id={h.get('id')}, skills={h.get('skills')})")
    except Exception as e:
        print(f"ERROR listing haseefs: {e}")
        haseefs = []

    # ── 2. Create (or update) test haseef ────────────────────────────
    print(f"\n--- 2. Create / update haseef '{HASEEF_NAME}' ---")
    haseef_id = None
    for h in haseefs:
        if h.get("name") == HASEEF_NAME:
            haseef_id = h.get("id")
            break

    payload = {
        "name": HASEEF_NAME,
        "description": "General-purpose test haseef using openai/gpt-5.4-mini via OpenRouter.",
        "configJson": _model_config(),
        "profileJson": {"language": "en", "test": True},
        "skills": [SKILL_NAME],
    }

    try:
        if haseef_id:
            print(f"Updating existing haseef id={haseef_id}")
            haseef = await sdk.haseef.update(haseef_id, payload)
        else:
            print("Creating new haseef…")
            haseef = await sdk.haseef.create(payload)
            haseef_id = haseef.get("id")
        print(f"Haseef ready: id={haseef_id}")
        print(json.dumps(haseef, indent=2, default=str))
    except Exception as e:
        print(f"ERROR creating/updating haseef: {e}")
        haseef_id = None

    if not haseef_id:
        print("\nCannot continue without a haseef. Exiting.")
        return

    # ── 3. Verify get + status ───────────────────────────────────────
    print(f"\n--- 3. Read back haseef + status ---")
    try:
        h = await sdk.haseef.get(haseef_id)
        print(f"Name: {h.get('name')}")
        print(f"Skills: {h.get('skills')}")
        print(f"Config keys: {list(h.get('configJson', {}).keys())}")
        status = await sdk.haseef.status(haseef_id)
        print(f"Status: {status}")
    except Exception as e:
        print(f"ERROR reading haseef: {e}")

    # ── 4. Profile CRUD ──────────────────────────────────────────────
    print(f"\n--- 4. Profile CRUD ---")
    try:
        profile = await sdk.haseef.get_profile(haseef_id)
        print(f"Original profile: {profile}")
        new_profile = {**(profile or {}), "mood": "curious", "test_run": True}
        updated = await sdk.haseef.update_profile(haseef_id, new_profile)
        print(f"Updated profile: {updated}")
    except Exception as e:
        print(f"ERROR profile CRUD: {e}")

    # ── 5. Memory write / read ───────────────────────────────────────
    print(f"\n--- 5. Memory write + read ---")
    try:
        await sdk.memory.set(haseef_id, [
            {"key": "test_greeting", "value": "Hello from general test!", "importance": 5},
            {"key": "test_number", "value": "42", "importance": 3},
        ])
        mems = await sdk.memory.list(haseef_id)
        print(f"Semantic memories: {len(mems)}")
        for m in mems:
            print(f"  - {m.get('key')}: {m.get('value')}")
        search = await sdk.memory.search(haseef_id, "greeting", 5)
        print(f"Search 'greeting': {len(search)} result(s)")
    except Exception as e:
        print(f"ERROR memory test: {e}")

    # ── 6. Skill attach verification ─────────────────────────────────
    print(f"\n--- 6. Skill attach check ---")
    try:
        h = await sdk.haseef.get(haseef_id)
        skills = h.get("skills") or []
        if SKILL_NAME in skills:
            print(f"OK: '{SKILL_NAME}' is attached.")
        else:
            print(f"ATTACHING '{SKILL_NAME}'…")
            await sdk.haseef.add_skill(haseef_id, SKILL_NAME)
            print("Attached.")
    except Exception as e:
        print(f"ERROR skill attach: {e}")

    # ── 7. Runs list ─────────────────────────────────────────────────
    print(f"\n--- 7. Recent runs ---")
    try:
        runs = await sdk.runs.list(haseef_id=haseef_id, limit=5)
        print(f"Found {len(runs)} run(s)")
        for r in runs:
            print(f"  - run {r.get('id')}: status={r.get('status')} duration={r.get('durationMs')}ms")
    except Exception as e:
        print(f"ERROR listing runs: {e}")

    # ── 8. Direct chat trigger (if Core exposes /chat or /runs/create) ─
    print(f"\n--- 8. Attempt direct run trigger ---")
    try:
        # Some Hsafa cores expose a /haseefs/{id}/runs or /chat endpoint.
        # We'll try a couple of common paths and print what we learn.
        async with httpx.AsyncClient(timeout=30) as client:
            headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
            for path in [
                f"/api/v7/haseefs/{haseef_id}/runs",
                f"/api/v7/haseefs/{haseef_id}/chat",
                "/api/v7/runs",
            ]:
                url = f"{CORE_URL}{path}"
                body = {"message": "What time is it? Please use a tool.", "haseefId": haseef_id}
                try:
                    r = await client.post(url, headers=headers, json=body)
                    print(f"  POST {path} -> {r.status_code}: {r.text[:200]}")
                except Exception as e2:
                    print(f"  POST {path} -> error: {type(e2).__name__}")
    except Exception as e:
        print(f"ERROR run trigger: {e}")

    print("\n=== Test complete ===")
    await sdk.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
