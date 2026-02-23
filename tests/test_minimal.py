import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

async def test_minimal():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Teste 1 — só texto, sem áudio
    config = types.LiveConnectConfig(
        response_modalities=["TEXT"],
    )
    
    try:
        async with client.aio.live.connect(
            model="models/gemini-2.5-flash-native-audio-latest",
            config=config
        ) as session:
            await session.send_client_content(
                turns=[types.Content(
                    role="user",
                    parts=[types.Part(text="Say hello.")]
                )],
                turn_complete=True
            )
            async for response in session.receive():
                if response.server_content:
                    if response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            print(f"Response: {part.text}")
                    if response.server_content.turn_complete:
                        break
        print("✓ TEXT mode OK")
    except Exception as e:
        print(f"✗ Erro: {e}")

asyncio.run(test_minimal())