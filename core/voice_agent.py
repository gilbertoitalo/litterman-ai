import asyncio
import os
import re
import numpy as np
import pyaudio

from google import genai
from google.genai import types
from .bl_engine import BlackLittermanEngine
from .gemini_agent import run_bl_pipeline, format_bl_result_for_voice
from .shared_state import set_status, push_bl_result

ASSETS = ['Stocks_USA', 'Stocks_EM', 'Bonds_USA']
WEIGHTS = np.array([0.60, 0.30, 0.10])
COV = np.diag([0.0225, 0.0324, 0.0025])
ORIGINAL_WEIGHTS_DICT = {'Stocks_USA': 0.60, 'Stocks_EM': 0.30, 'Bonds_USA': 0.10}

SYSTEM_PROMPT = """
You are Litterman, a voice assistant for asset managers.

STRICT OUTPUT RULES — NEVER break these:
- Speak only plain sentences. Zero markdown. Zero asterisks. Zero headers.
- Never narrate your thinking. Never say what you are about to do. Just do it.
- Never start a response with phrases like "I am", "I will", "I'm now", "Let me", "Sure", "Of course", "Certainly", "Analyzing", "Formulating", "Processing".
- If you receive a text block with portfolio weights and a Sharpe ratio, read it aloud immediately in 3 sentences. Nothing before, nothing after.
- If the user says something that is not market news, respond in one short sentence only.

Current portfolio assets: Stocks_USA, Stocks_EM, Bonds_USA.
"""

FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK = 1024

DEBOUNCE_SECONDS = 4.0

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
    ),
    input_audio_transcription=types.AudioTranscriptionConfig(),
    thinking_config=types.ThinkingConfig(thinking_budget=0),
)

pya = pyaudio.PyAudio()
audio_queue_output = asyncio.Queue()
audio_queue_mic = asyncio.Queue(maxsize=5)

_raw_transcript_queue = asyncio.Queue()
transcript_queue = asyncio.Queue()

agent_is_speaking = asyncio.Event()
mic_stream = None
_bl_processing = asyncio.Event()


def _normalise_transcript(raw: str) -> str:
    text = raw
    text = re.sub(r'\b(\w)\s+(?=\w)', lambda m: m.group(1), text)
    text = re.sub(r' +', ' ', text).strip()
    return text


async def listen_audio():
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
    while True:
        msg = await audio_queue_mic.get()
        await session.send_realtime_input(audio=msg)


async def receive_audio(session):
    while True:
        turn = session.receive()
        transcript_chunks = []

        async for response in turn:
            if response.server_content:

                if response.server_content.input_transcription:
                    chunk = response.server_content.input_transcription.text
                    if chunk:
                        transcript_chunks.append(chunk)

                if response.server_content.model_turn:
                    agent_is_speaking.set()
                    for part in response.server_content.model_turn.parts:
                        if part.text:
                            print(f"Litterman: {part.text}")
                        if part.inline_data and isinstance(part.inline_data.data, bytes):
                            audio_queue_output.put_nowait(part.inline_data.data)

        agent_is_speaking.clear()

        if transcript_chunks:
            raw = "".join(transcript_chunks)
            full_transcript = _normalise_transcript(raw)
            print(f"[turn] {full_transcript}")

            # Always forward to BL — bl_listener decides if it's market news
            # _bl_processing flag prevents duplicate processing during cooldown
            if not _bl_processing.is_set() and len(full_transcript) >= 15:
                await _raw_transcript_queue.put(full_transcript)
            else:
                reason = []
                if _bl_processing.is_set():
                    reason.append("BL cooldown active")
                if len(full_transcript) < 15:
                    reason.append(f"too short ({len(full_transcript)} chars)")
                if reason:
                    print(f"[receive_audio] Skipped: {', '.join(reason)}")

        print("[Listening...]\n")


async def debounce_task():
    accumulated = []

    while True:
        try:
            if not accumulated:
                chunk = await _raw_transcript_queue.get()
                accumulated.append(chunk)

            chunk = await asyncio.wait_for(
                _raw_transcript_queue.get(),
                timeout=DEBOUNCE_SECONDS
            )
            accumulated.append(chunk)

        except asyncio.TimeoutError:
            if accumulated:
                full_utterance = " ".join(accumulated)
                print(f"Manager said: {full_utterance}")
                await transcript_queue.put(full_utterance)
                accumulated = []


async def play_audio():
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


async def bl_listener(session):
    classifier_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    while True:
        transcript = await transcript_queue.get()

        if _bl_processing.is_set():
            print("[BL Listener] Race condition caught — discarding.")
            continue

        _bl_processing.set()

        while not transcript_queue.empty():
            discarded = transcript_queue.get_nowait()
            print(f"[BL Listener] Discarded duplicate: {discarded[:60]}...")

        try:
            set_status("processing")

            classification = await asyncio.to_thread(
                classifier_client.models.generate_content,
                model="gemini-2.5-flash",
                contents=f"""Does this text describe a financial market event, news, or economic development
that could affect asset prices? Reply with only YES or NO.

Text: {transcript}"""
            )

            is_market_news = classification.text.strip().upper().startswith("YES")
            print(f"[BL Listener] Market news detected: {is_market_news}")

            if is_market_news:
                print("[BL Listener] Running Black-Litterman pipeline...")
                bl_result = await asyncio.to_thread(
                    run_bl_pipeline,
                    transcript,
                    ASSETS,
                    WEIGHTS,
                    COV
                )

                push_bl_result(
                    transcript=transcript,
                    views=bl_result["views"],
                    weights_after=bl_result["weights"],
                    sharpe_after=bl_result["sharpe_ratio"],
                )

                prompt = format_bl_result_for_voice(bl_result, ORIGINAL_WEIGHTS_DICT)
                set_status("speaking")

                # send_realtime_input(text=) reliably triggers an audio response.
                # send_client_content was not producing audio in all SDK versions.
                # The _bl_processing flag (set above) blocks any new transcript
                # from the cooldown window from re-triggering the pipeline.
                await session.send_realtime_input(text=prompt)
                print("[BL Listener] Results injected. Cooldown started (15s).")

                # Wait for agent to finish speaking before clearing cooldown
                await asyncio.sleep(5)   # give model time to start responding
                await asyncio.sleep(10)  # rest of cooldown
                print("[BL Listener] Cooldown ended — listening again.\n")

            else:
                await asyncio.sleep(2)

        except Exception as e:
            print(f"[BL Listener error] {e}")

        finally:
            _bl_processing.clear()
            set_status("listening")
            while not transcript_queue.empty():
                transcript_queue.get_nowait()
            while not _raw_transcript_queue.empty():
                _raw_transcript_queue.get_nowait()


async def run_voice_agent():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    try:
        async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:

            await session.send_realtime_input(
                text="I am Litterman, your AI co-pilot for portfolio management. How can I help you today."
            )

            async with asyncio.TaskGroup() as tg:
                tg.create_task(listen_audio())
                tg.create_task(send_realtime(session))
                tg.create_task(receive_audio(session))
                tg.create_task(play_audio())
                tg.create_task(debounce_task())
                tg.create_task(bl_listener(session))

    except* KeyboardInterrupt:
        print("\nStopping agent...")
    finally:
        if mic_stream:
            mic_stream.close()
        pya.terminate()


if __name__ == "__main__":
    asyncio.run(run_voice_agent())
