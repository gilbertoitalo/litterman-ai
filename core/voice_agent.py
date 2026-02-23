import asyncio
import os
import numpy as np
import pyaudio

from google import genai
from google.genai import types
from core.bl_engine import BlackLittermanEngine
from .gemini_agent import extract_views_from_news, views_to_matrices

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

FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK = 1024

MODEL = "models/gemini-2.5-flash-native-audio-latest"
CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    system_instruction=SYSTEM_PROMPT,
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name="Charon"
            )
        )
    )
)

pya = pyaudio.PyAudio()
audio_queue_output = asyncio.Queue()
audio_queue_mic = asyncio.Queue(maxsize=5)
mic_stream = None


async def listen_audio():
    """Captures microphone and puts chunks into mic queue."""
    global mic_stream
    mic_info = pya.get_default_input_device_info()
    mic_stream = await asyncio.to_thread(
        pya.open,
        format=FORMAT,
        channels=CHANNELS,
        rate=INPUT_RATE,
        input=True,
        input_device_index=mic_info["index"],
        frames_per_buffer=CHUNK,
    )
    print("Litterman Voice Agent started. Speak naturally.\n")
    while True:
        data = await asyncio.to_thread(
            mic_stream.read, CHUNK, **{"exception_on_overflow": False}
        )
        await audio_queue_mic.put({"data": data, "mime_type": "audio/pcm"})


async def send_realtime(session):
    """Sends mic audio from queue to Gemini session."""
    while True:
        msg = await audio_queue_mic.get()
        await session.send_realtime_input(audio=msg)


async def receive_audio(session):
    """Receives Gemini response and queues audio for playback."""
    while True:
        turn = session.receive()
        async for response in turn:
            if response.server_content and response.server_content.model_turn:
                for part in response.server_content.model_turn.parts:
                    if part.text:
                        print(f"Litterman: {part.text}")
                    if part.inline_data and isinstance(part.inline_data.data, bytes):
                        audio_queue_output.put_nowait(part.inline_data.data)

        # Clear output queue on interruption to stop playback immediately
        while not audio_queue_output.empty():
            audio_queue_output.get_nowait()
        print("[Listening...]\n")


async def play_audio():
    """Plays audio chunks from output queue through speakers."""
    stream = await asyncio.to_thread(
        pya.open,
        format=FORMAT,
        channels=CHANNELS,
        rate=OUTPUT_RATE,
        output=True,
    )
    while True:
        bytestream = await audio_queue_output.get()
        await asyncio.to_thread(stream.write, bytestream)


async def run_voice_agent():
    """Main entry point — connects to Gemini and runs all tasks."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    try:
        async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:

            # Send initial greeting
            await session.send_realtime_input(
                text="Introduce yourself briefly."
            )

            async with asyncio.TaskGroup() as tg:
                tg.create_task(listen_audio())
                tg.create_task(send_realtime(session))
                tg.create_task(receive_audio(session))
                tg.create_task(play_audio())

    except* KeyboardInterrupt:
        print("\nStopping agent...")
    finally:
        if mic_stream:
            mic_stream.close()
        pya.terminate()


if __name__ == "__main__":
    asyncio.run(run_voice_agent())