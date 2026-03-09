import asyncio
import os
import re
import numpy as np
import pyaudio

from google import genai
from google.genai import types
from .bl_engine import BlackLittermanEngine
from .gemini_agent import run_bl_pipeline, format_bl_result_for_voice
from .shared_state import get_state, set_status, push_bl_result

ASSETS = ['Stocks_USA', 'Stocks_EM', 'Bonds_USA']
WEIGHTS = np.array([0.60, 0.30, 0.10])   # fallback / reference only
COV = np.diag([0.0225, 0.0324, 0.0025])

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
BL_COOLDOWN_SECONDS = 20  # prevents BL re-trigger, but never blocks voice

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
interrupt_event = asyncio.Event()  # signals play_audio to drop buffered chunks

# Two separate flags — these are intentionally distinct:
# _bl_running: True while the BL pipeline is actively executing (prevents duplicate runs)
# _bl_cooldown: True after BL completes, blocks re-trigger but NOT normal conversation
_bl_running = asyncio.Event()
_bl_cooldown = asyncio.Event()


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
        transcript_chunks = []

        async for response in session.receive():
            if not response.server_content:
                continue

            # ── Barge-in: API signals user started speaking ──
            if response.server_content.interrupted:
                print("\n--- BARGE-IN DETECTED ---")
                interrupt_event.set()
                while not audio_queue_output.empty():
                    try:
                        audio_queue_output.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                agent_is_speaking.clear()
                continue

            if response.server_content.input_transcription:
                chunk = response.server_content.input_transcription.text
                if chunk:
                    transcript_chunks.append(chunk)

            if response.server_content.model_turn:
                interrupt_event.clear()  # new agent speech — reset interrupt
                agent_is_speaking.set()
                for part in response.server_content.model_turn.parts:
                    if part.text:
                        print(f"Litterman: {part.text}")
                    if part.inline_data and isinstance(part.inline_data.data, bytes):
                        audio_queue_output.put_nowait(part.inline_data.data)

            if response.server_content.turn_complete:
                agent_is_speaking.clear()
                break

        if transcript_chunks:
            raw = "".join(transcript_chunks)
            full_transcript = _normalise_transcript(raw)
            print(f"[turn] {full_transcript}")

            if len(full_transcript) < 15:
                print(f"[receive_audio] Skipped: too short ({len(full_transcript)} chars)")
            elif _bl_running.is_set():
                print("[receive_audio] Skipped: BL pipeline running")
            elif _bl_cooldown.is_set():
                print("[receive_audio] BL cooldown — forwarding to conversation only")
                await session.send_realtime_input(text=full_transcript)
            else:
                await _raw_transcript_queue.put(full_transcript)

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
        # Drop chunk if barge-in happened — don't play stale audio
        if interrupt_event.is_set():
            continue
        await asyncio.to_thread(stream.write, bytestream)


async def bl_listener(session):
    classifier_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    while True:
        transcript = await transcript_queue.get()

        if _bl_running.is_set():
            print("[BL Listener] Interrupted during classification — aborting.")
            continue

        _bl_running.set()
        _bl_cooldown.set()
        interrupt_event.clear()  # fresh cycle — reset interrupt

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

            # Check if user interrupted during classification
            if interrupt_event.is_set():
                print("[BL Listener] Interrupted during classification — aborting.")
                return

            is_market_news = classification.text.strip().upper().startswith("YES")
            print(f"[BL Listener] Market news detected: {is_market_news}")

            if is_market_news:
                print("[BL Listener] Running Black-Litterman pipeline...")

                # Acknowledge immediately
                await session.send_realtime_input(text="Understood. Stand by for analysis.")
                print("[BL Listener] Acknowledgment sent.")

                current_state = await asyncio.to_thread(get_state)
                current_weights_dict = current_state["portfolio"]["current"]
                weights_live = np.array([current_weights_dict[a] for a in ASSETS])
                print(f"[BL Listener] Using live weights: { {a: round(current_weights_dict[a], 4) for a in ASSETS} }")

                bl_result = await asyncio.to_thread(
                    run_bl_pipeline,
                    transcript,
                    ASSETS,
                    weights_live,
                    COV
                )

                # Check if user interrupted during BL calculation
                if interrupt_event.is_set():
                    print("[BL Listener] Interrupted during BL calc — skipping result injection.")
                else:
                    push_bl_result(
                        transcript=transcript,
                        views=bl_result["views"],
                        weights_after=bl_result["weights"],
                        sharpe_after=bl_result["sharpe_ratio"],
                    )

                    prompt = format_bl_result_for_voice(bl_result, current_weights_dict)
                    set_status("speaking")
                    await session.send_realtime_input(text=prompt)
                    print("[BL Listener] Results injected. Waiting for agent to finish speaking...")

                    await asyncio.wait_for(_wait_for_interrupt_or_silence(), timeout=30)
                    print("[BL Listener] Agent finished speaking.")

            else:
                await asyncio.sleep(1)

        except asyncio.TimeoutError:
            print("[BL Listener] Timeout — resuming.")
        except Exception as e:
            print(f"[BL Listener error] {e}")

        finally:
            _bl_running.clear()
            set_status("listening")
            asyncio.get_event_loop().call_later(
                BL_COOLDOWN_SECONDS,
                _bl_cooldown.clear
            )
            print(f"[BL Listener] BL cooldown active for {BL_COOLDOWN_SECONDS}s — conversation unblocked.\n")
            while not transcript_queue.empty():
                transcript_queue.get_nowait()
            while not _raw_transcript_queue.empty():
                _raw_transcript_queue.get_nowait()


async def _wait_for_interrupt_or_silence():
    """Wait until agent finishes speaking OR user interrupts — whichever comes first."""
    while True:
        if interrupt_event.is_set():
            return  # user interrupted
        if not agent_is_speaking.is_set():
            await asyncio.sleep(0.8)  # confirm silence
            if not agent_is_speaking.is_set():
                return  # agent finished naturally
        await asyncio.sleep(0.1)


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