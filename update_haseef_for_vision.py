"""Update haseef to use gpt-4o-mini with aggressive tool-prompting."""
import asyncio
from hsafa_sdk import HsafaSDK, SdkOptions

CORE_URL = "https://core.hsafa.com"
API_KEY = "sk_prod_7f2e8d9c4b3a6f1e0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0"
SKILL = "robot_vision"
HASEEF_ID = "8aa60ad7-cb23-4e44-a26c-e8b7c8332d11"

async def main():
    sdk = HsafaSDK(SdkOptions(core_url=CORE_URL, api_key=API_KEY, skill=SKILL))

    await sdk.haseef.update(HASEEF_ID, {
        "configJson": {
            "llm": {
                "provider": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "model": "openai/gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            "system_prompt": (
                "You are a robot with a camera and a movable head.\n\n"
                "INSTRUCTIONS:\n"
                "1. The user sends voice messages WITH an image from the camera.\n"
                "2. You MUST call the `move_head(yaw_deg, pitch_deg)` tool to look around.\n"
                "3. When you call move_head, the robot moves, captures a NEW image, and returns it to you.\n"
                "4. Use this to search for objects or look in different directions.\n"
                "5. Chain move_head calls autonomously — do NOT ask the user for permission.\n\n"
                "HEAD DIRECTIONS:\n"
                "- yaw=0, pitch=0  = straight ahead\n"
                "- yaw=+30, pitch=0 = look LEFT\n"
                "- yaw=-30, pitch=0 = look RIGHT\n"
                "- yaw=0, pitch=-15 = look UP\n"
                "- yaw=0, pitch=+15 = look DOWN\n\n"
                "RULES:\n"
                "- If the user asks to find/search/look for something, call move_head IMMEDIATELY.\n"
                "- If the object is not visible, try another direction.\n"
                "- Only speak AFTER you have found the object or checked all directions.\n"
                "- Be concise."
            ),
        }
    })
    h = await sdk.haseef.get(HASEEF_ID)
    print(f"Updated haseef: model={h['configJson']['llm']['model']}")
    await sdk.disconnect()

asyncio.run(main())
