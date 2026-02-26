import asyncio
import os
import re
import numpy as np
import pyaudio

from google import genai
from google.genai import types
from streamlit import form
from .bl_engine import BlackLittermanEngine
from .gemini_agent import run_bl_pipeline, format_bl_result_for_voice
from .shared_state import set_status, push_bl_result

ASSETS = ['Stocks_USA', 'Stocks_EM', 'Bonds_USA']
WEIGHTS = np.array([0.60, 0.30, 0.10])
COV = np.diag([0.0225, 0.0324, 0.0025])
ORIGINAL_WEIGHTS_DICT = {'Stocks_USA': 0.60, 'Stocks_EM': 0.30, 'Bonds_USA': 0.10}

SYSTEM_PROMPT = """
You are Litterman, an AI co-pilot for asset managers.
When given Black-Litterman results, read them aloud in exactly 3 sentences. No analysis, no headers, no thinking. Just speak the numbers.
Current portfolio: Stocks_USA, Stocks_EM, Bonds_USA
"""

FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK = 1024

# Debounce window: seconds of silence before treating utterance as complete.
# Gemini Live breaks long utterances into multiple turns — we wait for the
# manager to stop speaking before sending the full transcript to the BL pipeline.
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
    # thinking_config removed — not supported by Native Audio model

)

pya = pyaudio.PyAudio()
audio_queue_output = asyncio.Queue()
audio_queue_mic = asyncio.Queue(maxsize=5)

# Two-stage transcript pipeline:
# receive_audio -> _raw_transcript_queue -> debounce_task -> transcript_queue -> bl_listener
_raw_transcript_queue = asyncio.Queue()  # individual turn chunks (may be partial utterances)
transcript_queue = asyncio.Queue()        # complete debounced utterances ready for BL

agent_is_speaking = asyncio.Event()
mic_stream = None

# Global flag: set when BL pipeline is running or in cooldown.
# Blocks receive_audio from sending new transcripts during this period.
_bl_processing = asyncio.Event()


def _normalise_transcript(raw: str) -> str:
    """
    Normalises fragmented transcription from Gemini audio chunking.
    - Collapses single characters separated by spaces (chunking artefacts e.g. "eme rgen")
    - Collapses multiple spaces into one
    - Strips leading/trailing whitespace
    """
    text = raw
    text = re.sub(r'\b(\w)\s+(?=\w)', lambda m: m.group(1), text)
    text = re.sub(r' +', ' ', text).strip()
    return text


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
    """
    Receives Gemini response, queues audio for playback, and captures transcripts.

    Each Gemini turn may be a partial utterance — long sentences arrive across
    multiple turns. Each turn's transcript is sent to _raw_transcript_queue;
    debounce_task assembles them into complete utterances.

    Filtering:
    - Skips turns where the agent spoke (avoids echo triggering BL)
    - Skips turns during BL cooldown
    - Skips transcripts shorter than 15 chars
    """
    while True:
        turn = session.receive()
        transcript_chunks = []
        turn_had_agent_speech = False

        async for response in turn:
            if response.server_content:

                # Capture user input transcription
                if response.server_content.input_transcription:
                    chunk = response.server_content.input_transcription.text
                    if chunk:
                        transcript_chunks.append(chunk)

                # Agent is speaking
                if response.server_content.model_turn:
                    agent_is_speaking.set()
                    turn_had_agent_speech = True
                    for part in response.server_content.model_turn.parts:
                        if part.text:
                            print(f"Litterman: {part.text}")
                        if part.inline_data and isinstance(part.inline_data.data, bytes):
                            audio_queue_output.put_nowait(part.inline_data.data)

        # Turn complete
        agent_is_speaking.clear()

        # Clear output queue if interrupted mid-playback
        while not audio_queue_output.empty():
            audio_queue_output.get_nowait()

        # Normalise and forward to debounce stage
        if transcript_chunks:
            raw = "".join(transcript_chunks)
            full_transcript = _normalise_transcript(raw)
            print(f"[turn] {full_transcript}")

            if (
                not turn_had_agent_speech
                and not _bl_processing.is_set()
                and len(full_transcript) >= 15
            ):
                await _raw_transcript_queue.put(full_transcript)
            else:
                reason = []
                if turn_had_agent_speech:
                    reason.append("agent turn echo")
                if _bl_processing.is_set():
                    reason.append("BL cooldown active")
                if len(full_transcript) < 15:
                    reason.append(f"too short ({len(full_transcript)} chars)")
                print(f"[receive_audio] Skipped: {', '.join(reason)}")

        print("[Listening...]\n")


async def debounce_task():
    """
    Assembles partial turn transcripts into complete utterances.

    Gemini Live breaks long speech into multiple turns. This task collects
    all turns arriving within DEBOUNCE_SECONDS of each other, joins them,
    and sends the complete utterance to transcript_queue for the BL pipeline.

    Example:
        [turn] "The Federal Reserve announced today"      t=0.0s
        [turn] "an unexpected 50 basis point rate hike"   t=0.8s
        [turn] "citing persistent inflation above 4%"     t=1.5s
        --- 2.5s silence ---
        Manager said: "The Federal Reserve announced today an unexpected..."
    """
    accumulated = []

    while True:
        try:
            if not accumulated:
                # Blocking wait for first chunk
                chunk = await _raw_transcript_queue.get()
                accumulated.append(chunk)

            # Wait for more chunks within the debounce window
            chunk = await asyncio.wait_for(
                _raw_transcript_queue.get(),
                timeout=DEBOUNCE_SECONDS
            )
            accumulated.append(chunk)

        except asyncio.TimeoutError:
            # Silence window elapsed — utterance is complete
            if accumulated:
                full_utterance = " ".join(accumulated)
                print(f"Manager said: {full_utterance}")
                await transcript_queue.put(full_utterance)
                accumulated = []


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


async def bl_listener(session):
    """
    Listens to complete manager utterances, classifies them as market news or not,
    runs the Black-Litterman pipeline if relevant, and injects results into the session.
    Also writes results to shared_state.json for the Streamlit dashboard.
    """
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
            # Update dashboard status
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

                # ── Write to shared state for dashboard ──────────────────────
                push_bl_result(
                    transcript=transcript,
                    views=bl_result["views"],
                    weights_after=bl_result["weights"],
                    sharpe_after=bl_result["sharpe_ratio"],
                )

                # ── Inject voice response into session ───────────────────────
                prompt = format_bl_result_for_voice(bl_result, ORIGINAL_WEIGHTS_DICT)
                set_status("speaking")
                await session.send_realtime_input(text=prompt)
                print("[BL Listener] Results injected. Cooldown started (15s).")

                await asyncio.sleep(15)
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
    """Main entry point — connects to Gemini and runs all 6 async tasks."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    try:
        async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:

            # Send initial greeting
            await session.send_realtime_input(
                text="Say exactly this out loud: 'I am Litterman, your AI co-pilot for portfolio management. How can I help you today.'"
            )

            async with asyncio.TaskGroup() as tg:
                tg.create_task(listen_audio())
                tg.create_task(send_realtime(session))
                tg.create_task(receive_audio(session))
                tg.create_task(play_audio())
                tg.create_task(debounce_task())       # NEW: debounce stage
                tg.create_task(bl_listener(session))

    except* KeyboardInterrupt:
        print("\nStopping agent...")
    finally:
        if mic_stream:
            mic_stream.close()
        pya.terminate()


if __name__ == "__main__":
    asyncio.run(run_voice_agent())
