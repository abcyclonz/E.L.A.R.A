"""
E.L.A.R.A. — Pi Audio Client

Streams mic audio to RunPod over WebSocket.
Receives Kokoro TTS WAV back and plays it through the speaker.

Set ORCHESTRATOR_WS_URL in .env or export it before running.
"""

import asyncio
import io
import os
import queue
import threading
import wave

import pyaudio
import websockets
from dotenv import load_dotenv

load_dotenv()

WS_URL       = os.environ.get("ORCHESTRATOR_WS_URL", "ws://localhost:8001/ws/audio")
SAMPLE_RATE  = 16000
CHANNELS     = 1
CHUNK_FRAMES = 512   # ~32 ms per chunk

_pa           = pyaudio.PyAudio()
_send_queue: queue.Queue[bytes] = queue.Queue()
_playing      = threading.Event()   # mutes mic while TTS plays (no echo)


def _open_mic():
    def _cb(in_data, frame_count, time_info, status):
        if not _playing.is_set():
            _send_queue.put(in_data)
        return (None, pyaudio.paContinue)

    return _pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_FRAMES,
        stream_callback=_cb,
    )


def _play_wav(wav_bytes: bytes):
    _playing.set()
    try:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf) as wf:
            stream = _pa.open(
                format=_pa.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
            )
            data = wf.readframes(1024)
            while data:
                stream.write(data)
                data = wf.readframes(1024)
            stream.stop_stream()
            stream.close()
    except Exception as e:
        print(f"[Playback error] {e}")
    finally:
        _playing.clear()


async def run():
    print(f"Connecting to {WS_URL} ...")

    async with websockets.connect(WS_URL, max_size=10 * 1024 * 1024) as ws:
        print("Connected. Speak now — Elara is listening.\n")

        mic = _open_mic()
        mic.start_stream()

        async def sender():
            loop = asyncio.get_event_loop()
            while True:
                chunk = await loop.run_in_executor(None, _send_queue.get)
                try:
                    await ws.send(chunk)
                except websockets.ConnectionClosed:
                    break

        async def receiver():
            async for message in ws:
                if isinstance(message, bytes) and len(message) > 0:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, _play_wav, message)

        try:
            await asyncio.gather(sender(), receiver())
        finally:
            mic.stop_stream()
            mic.close()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        _pa.terminate()
