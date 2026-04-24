import logging
import asyncio
from app.main import process_prompt
from app.session_manager import SessionManager

logging.basicConfig(level=logging.DEBUG)

async def test():
    session = SessionManager("test_session_123")
    try:
        payload = process_prompt("police check", session)
        print("KEYS:", payload.keys())
        print("DRAFT LEN:", len(payload.get("generated", {}).get("draft", "")))
    except Exception as e:
        print("ERROR:", e)

if __name__ == "__main__":
    asyncio.run(test())
