import asyncio
import os
from hsafa_sdk import HsafaSDK, SdkOptions

sdk = HsafaSDK(SdkOptions(
    core_url="https://core.hsafa.com",
    api_key="sk_prod_7f2e8d9c4b3a6f1e0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0",
    skill="general_tester",
))

async def main():
    await sdk.haseef.update("b8f0ead5-036c-4a0b-8afb-e56314acdb9f", {
        "configJson": {
            "llm": {
                "provider": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "model": "openai/gpt-4o-mini",
                "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            "system_prompt": "You are a helpful test assistant. You have access to tools: echo, get_current_time, calculate, get_random_fact. ALWAYS use a tool when the user asks for time, math, or facts.",
        }
    })
    h = await sdk.haseef.get("b8f0ead5-036c-4a0b-8afb-e56314acdb9f")
    print("Updated haseef:")
    print(f"  model: {h['configJson']['llm']['model']}")
    print(f"  api_key set: {bool(h['configJson']['llm']['api_key'])}")
    await sdk.disconnect()

asyncio.run(main())
