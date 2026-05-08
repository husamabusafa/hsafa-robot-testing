"""Create a haseef for the voice+vision skill."""
import asyncio
import os
from hsafa_sdk import HsafaSDK, SdkOptions

CORE_URL = "https://core.hsafa.com"
API_KEY = "sk_prod_7f2e8d9c4b3a6f1e0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0"
SKILL = "robot_vision"

async def main():
    sdk = HsafaSDK(SdkOptions(
        core_url=CORE_URL,
        api_key=API_KEY,
        skill=SKILL,
    ))

    # Check if haseef already exists
    haseefs = await sdk.haseef.list()
    h = None
    for existing in haseefs:
        if existing.get("name") == "RobotVision" or SKILL in (existing.get("skills") or []):
            h = existing
            print(f"Found existing: {h['name']} (id={h['id']}, skills={h.get('skills')})")
            break

    if h:
        # Add skill if missing
        if SKILL not in (h.get("skills") or []):
            print(f"Adding skill '{SKILL}'...")
            await sdk.haseef.add_skill(h["id"], SKILL)
            print("Skill added.")
        haseef_id = h["id"]
    else:
        print(f"Creating new haseef for skill '{SKILL}'...")
        h = await sdk.haseef.create({
            "name": "RobotVision",
            "description": "Voice + Vision robot control haseef",
            "skills": [SKILL],
            "configJson": {
                "llm": {
                    "provider": "openrouter",
                    "base_url": "https://openrouter.ai/api/v1",
                    "model": "openai/gpt-5.4-mini",
                    "temperature": 0.7,
                    "max_tokens": 1024,
                },
                "system_prompt": (
                    "You are a robot with a camera and a movable head. "
                    "The user sends you voice messages with an image from their camera. "
                    "You can call `move_head(yaw_deg, pitch_deg)` to look around. "
                    "When you call it, the robot will move its head, capture a fresh image, and show it to you. "
                    "Use this to search for objects or look in specific directions. "
                    "yaw=0 is straight ahead, positive=left, negative=right. "
                    "pitch=0 is level, negative=up, positive=down. "
                    "Always be concise. If asked to find something, call move_head to look around autonomously."
                ),
            },
        })
        haseef_id = h["id"]
        print(f"Created haseef: {haseef_id}")

    # Verify
    h = await sdk.haseef.get(haseef_id)
    print(f"\nHaseef: {h['name']}")
    print(f"  ID: {h['id']}")
    print(f"  Skills: {h.get('skills')}")
    print(f"  Model: {h['configJson']['llm']['model']}")
    print(f"  System prompt: {h['configJson']['system_prompt'][:100]}...")

    await sdk.disconnect()
    print(f"\nHASEEF_ID={haseef_id}")
    print("Update hsafa_voice_vision.py with this ID.")

asyncio.run(main())
