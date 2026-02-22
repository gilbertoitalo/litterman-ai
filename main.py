import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env FIRST before any other imports
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Verify key
api_key = os.getenv("GEMINI_API_KEY")
print(f"API key loaded: {api_key[:8] if api_key else 'NOT FOUND'}")

# Only import after env is loaded
import asyncio
from core.voice_agent import run_voice_agent

if __name__ == "__main__":
    asyncio.run(run_voice_agent())