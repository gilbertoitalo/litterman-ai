import pyaudio
import numpy as np

CHUNK = 512
INPUT_RATE = 16000
CHANNELS = 1

pa = pyaudio.PyAudio()

# List available input devices
print("Available input devices:")
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f"  [{i}] {info['name']}")

stream = pa.open(
    format=pyaudio.paInt16,
    channels=CHANNELS,
    rate=INPUT_RATE,
    input=True,
    frames_per_buffer=CHUNK
)

print("\nReading microphone for 5 seconds — speak now...")
for _ in range(int(INPUT_RATE / CHUNK * 5)):
    data = stream.read(CHUNK, exception_on_overflow=False)
    arr = np.frombuffer(data, dtype=np.int16)
    volume = np.abs(arr).mean()
    bar = "#" * int(volume / 50)
    print(f"Volume: {volume:.0f} {bar}")

stream.stop_stream()
stream.close()
pa.terminate()