"""Discover HSAFA Core API — test connectivity and list endpoints."""
import asyncio
import json
import os
import sys

import httpx

CORE_URL = "https://core.hsafa.com"
API_KEY = "sk_prod_7f2e8d9c4b3a6f1e0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0"
API_BASE = "/api/v7"


async def request(method: str, path: str, body=None):
    url = f"{CORE_URL}{API_BASE}{path}"
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient() as client:
        response = await client.request(method, url, headers=headers, json=body, timeout=30)
        print(f"\n>>> {method} {path} — Status: {response.status_code}")
        try:
            data = response.json()
            print(json.dumps(data, indent=2, default=str))
            return data
        except Exception:
            print(response.text[:500])
            return None


async def main():
    # 1. Health / info
    await request("GET", "/")

    # 2. List existing haseefs
    await request("GET", "/haseefs")

    # 3. List existing skills
    await request("GET", "/skills")

    # 4. List existing runs
    await request("GET", "/runs")


if __name__ == "__main__":
    asyncio.run(main())
