#!/usr/bin/env python3
"""diagnose_hsafa_server.py — Check Hsafa Core server connectivity."""
import asyncio
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("HSAFA_CORE_KEY", "")
CORE_URL = os.environ.get("HSAFA_CORE_URL", "https://core.hsafa.com")
HASEEF_ID = os.environ.get("HASEEF_ID", "")

if not API_KEY:
    print("Error: HSAFA_CORE_KEY not set", file=sys.stderr)
    sys.exit(1)

HEADERS = {"x-api-key": API_KEY}


async def probe(url: str, method: str = "GET", body=None, timeout: float = 15.0):
    client = httpx.AsyncClient(timeout=timeout)
    try:
        if method == "GET":
            r = await client.get(url, headers=HEADERS)
        else:
            r = await client.post(url, headers=HEADERS, json=body)
        print(f"  {method} {url}")
        print(f"    status: {r.status_code}")
        print(f"    body:   {r.text[:200]}")
        return True
    except httpx.TimeoutException as e:
        print(f"  {method} {url}")
        print(f"    TIMEOUT after {timeout}s ({type(e).__name__})")
        return False
    except Exception as e:
        print(f"  {method} {url}")
        print(f"    ERROR: {type(e).__name__}: {e}")
        return False
    finally:
        await client.aclose()


async def main():
    print(f"Probing {CORE_URL} ...")
    print(f"API key prefix: {API_KEY[:10]}...")
    print()

    ok = True
    ok &= await probe(f"{CORE_URL}/api/v7/haseefs")
    if HASEEF_ID:
        ok &= await probe(f"{CORE_URL}/api/v7/haseefs/{HASEEF_ID}")
    ok &= await probe(
        f"{CORE_URL}/api/v7/events",
        method="POST",
        body={"type": "ping", "skill": "robot_base", "data": {}},
    )

    print()
    if ok:
        print("[OK] All probes succeeded.")
    else:
        print("[FAIL] Some probes timed out or errored.")
        print("The Hsafa Core server may be down or very slow.")


if __name__ == "__main__":
    asyncio.run(main())
