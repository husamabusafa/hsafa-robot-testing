#!/usr/bin/env python3
"""test_push_event.py — Quick test of push_event to Haseef."""
import asyncio
import os
import sys
from pathlib import Path

_repo_root = Path(__file__).parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dotenv import load_dotenv
from hsafa_sdk import HsafaSDK, SdkOptions

load_dotenv()

CORE_URL = os.environ.get("HSAFA_CORE_URL", "https://core.hsafa.com")
CORE_KEY = os.environ.get("HSAFA_CORE_KEY", "")
HASEEF_ID = os.environ.get("HASEEF_ID", "")

if not CORE_KEY or not HASEEF_ID:
    print("Missing env vars", file=sys.stderr)
    sys.exit(1)


async def main():
    sdk = HsafaSDK(SdkOptions(core_url=CORE_URL, api_key=CORE_KEY, skill="robot_base"))
    try:
        print("Pushing event to Haseef...")
        await sdk.push_event({
            "type": "user_message",
            "data": {
                "text": "Show a happy expression.",
                "source": "test_push_event",
            },
            "haseefId": HASEEF_ID,
        })
        print("[OK] Event pushed successfully.")
    except Exception as e:
        import traceback
        print(f"[FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        await sdk.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
