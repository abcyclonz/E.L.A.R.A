"""
audio_client.py — Thin Pi audio client for cloud-mode E.L.A.R.A.

Captures microphone audio and streams raw PCM to the RunPod orchestrator
over a WebSocket. Receives Kokoro TTS WAV back and plays it through the speaker.

Usage:
    python audio_client.py

Environment variables:
    ORCHESTRATOR_WS_URL  WebSocket URL of the orchestrator audio gateway
                         e.g. ws://your-pod-id-8003.proxy.runpod.net/ws/audio
                         Default: ws://localhost:8003/ws/audio
"""

import asyncio
import io
import os
import queue
import threading

import pyaudio
import websockets

# ── Config ────────────────────────────────────────────────────────────────────

WS_URL       = os.environ.get("ORCHESTRATOR_WS_URL", "ws://localhost:8003/ws/audio")
SAMPLE_RATE  = 16000
CHANNELS     = 1
SAMPLE_WIDTH = 2       # int16
CHUNK_FRAMES = 512     # frames per PyAudio callback (~32 ms at 16 kHz)

# ── Audio I/O ─────────────────────────────────────────────────────────────────

_pa           = pyaudio.PyAudio()
_send_queue: queue.Queue[bytes] = queue.Queue()
_playing      = threading.Event()   # set while TTS is playing — mutes mic


def _open_mic_stream():
    """Opens PyAudio input stream; each callback pushes a chunk to _send_queue."""
    def _callback(in_data, frame_count, time_info, status):
        if not _playing.is_set():   # don't echo TTS back into the mic
            _send_queue.put(in_data)
        return (None, pyaudio.paContinue)

    return _pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_FRAMES,
        stream_callback=_callback,
    )


def _play_wav(wav_bytes: bytes):
    """Plays a WAV buffer (Kokoro outputs 24 kHz WAV) through the default speaker."""
    import wave
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


# ── Main WebSocket loop ───────────────────────────────────────────────────────

async def run():
    print(f"Connecting to {WS_URL} …")

    async with websockets.connect(WS_URL, max_size=10 * 1024 * 1024) as ws:
        print("Connected. Speak now — Elara is listening.\n")

        mic_stream = _open_mic_stream()
        mic_stream.start_stream()

        async def sender():
            """Pulls audio chunks from the queue and sends over WebSocket."""
            loop = asyncio.get_event_loop()
            while True:
                chunk = await loop.run_in_executor(None, _send_queue.get)
                try:
                    await ws.send(chunk)
                except websockets.ConnectionClosed:
                    break

        async def receiver():
            """Receives TTS WAV bytes and plays them."""
            async for message in ws:
                if isinstance(message, bytes) and len(message) > 0:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, _play_wav, message)

        try:
            await asyncio.gather(sender(), receiver())
        finally:
            mic_stream.stop_stream()
            mic_stream.close()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        _pa.terminate()
