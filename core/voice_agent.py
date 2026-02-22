import asyncio
import os
import numpy as np
from pathlib import Path

from google import genai
from google.genai import types
from .bl_engine import BlackLittermanEngine
from .gemini_agent import extract_views_from_news, views_to_matrices

# Load .env from project root

# Portfolio configuration
ASSETS = ['Stocks_USA', 'Stocks_EM', 'Bonds_USA']
WEIGHTS = np.array([0.60, 0.30, 0.10])
COV = np.diag([0.0225, 0.0324, 0.0025])

SYSTEM_PROMPT = """
You are Litterman, an AI co-pilot for asset managers.
Your role is to help portfolio managers analyse news and market events,
extract quantitative views, and suggest portfolio rebalancing based on
the Black-Litterman model.

When the manager describes a market event or news:
1. Acknowledge the event and its potential market impact
2. Explain what views you are extracting
3. Present the portfolio rebalancing recommendation clearly
4. Always mention the Sharpe ratio and key weight changes

Be concise, professional, and quantitative. Speak like a senior quant analyst.
Current portfolio assets: Stocks_USA, Stocks_EM, Bonds_USA
"""


async def run_voice_agent():
    """
    Runs the Litterman voice agent using Gemini Live API.
    The manager can speak naturally and get portfolio recommendations.
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=SYSTEM_PROMPT,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Charon"  # professional male voice
                )
            )
        )
    )

    print("Starting Litterman Voice Agent...")
    print("Speak to your AI portfolio co-pilot. Press Ctrl+C to stop.\n")

    async with client.aio.live.connect(
        model="models/gemini-2.5-flash-native-audio-latest",
        config=config
    ) as session:

        # Send initial greeting
        await session.send_client_content(
            turns = [types.Content(
                        role="user",
                        parts=[types.Part(text="Introduce yourself briefly.")]
                    )],
                    turn_complete=True
                )

        async for response in session.receive():
            if response.server_content:
                if response.server_content.model_turn:
                    for part in response.server_content.model_turn.parts:
                        if part.text:
                            print(f"Litterman: {part.text}")


if __name__ == "__main__":
    asyncio.run(run_voice_agent())