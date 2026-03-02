import os
import asyncio
from openai import AsyncOpenAI
from config import OPENAI_API_KEY

async def main():
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not set")
        return

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    try:
        models = await client.models.list()
        print(f"✅ Found {len(models.data)} models.")
        
        # Filter for interesting ones
        interesting = [m.id for m in models.data if "gpt" in m.id or "o1" in m.id or "o3" in m.id]
        interesting.sort()
        
        print("--- Interesting Models ---")
        for m in interesting:
            print(m)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
