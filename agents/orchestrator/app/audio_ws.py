"""
audio_ws.py — WebSocket audio gateway, integrated into the orchestrator.

Pi connects here, streams raw PCM int16 audio chunks, receives Kokoro TTS WAV back.
Pipeline: PCM chunks → Silero VAD → Faster-Whisper STT → SpeechBrain Speaker ID
          → orchestrator pipeline (existing handle_input logic)
          → Elara Kokoro TTS → WAV bytes back to Pi
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Optional

import httpx
import numpy as np
import torch

from fastapi import WebSocket, WebSocketDisconnect

log = logging.getLogger("audio_ws")

SILENCE_TIMEOUT  = 0.8      # seconds of silence → end of phrase
MIN_AUDIO_LEN    = 0.5      # minimum phrase duration to process
SAMPLE_RATE      = 16000
BYTES_PER_SAMPLE = 2        # int16

ELARA_URL = os.environ.get("ELARA_URL", "http://elara:8002")

# Speaker embeddings persist across restarts
_SPEAKER_PATH = os.environ.get("SPEAKER_EMBEDDINGS_PATH", "/data/speaker_embeddings.json")


class AudioPipeline:
    """
    Loads VAD + STT + Speaker ID once at startup.
    Shared across all WebSocket connections (models are stateless per inference).
    """

    def __init__(self):
        self._vad_model       = None
        self._stt_model       = None
        self._encoder         = None
        self._known_speakers: dict[str, torch.Tensor] = {}
        self._next_id         = 1

    # ── Startup loader ────────────────────────────────────────────────────────

    def load(self):
        self._load_vad()
        self._load_stt()
        self._load_speaker_encoder()
        self._load_speaker_embeddings()

    def _load_vad(self):
        log.info("Loading Silero VAD…")
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self._vad_model = model
        log.info("VAD ready.")

    def _load_stt(self):
        log.info("Loading Faster-Whisper (base.en)…")
        from faster_whisper import WhisperModel
        device      = "cuda" if torch.cuda.is_available() else "cpu"
        compute     = "float16" if device == "cuda" else "int8"
        self._stt_model = WhisperModel("base.en", device=device, compute_type=compute)
        log.info("Whisper ready on %s (%s).", device, compute)

    def _load_speaker_encoder(self):
        log.info("Loading SpeechBrain ECAPA-TDNN…")
        from speechbrain.inference.speaker import EncoderClassifier
        self._encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/data/speaker_model",
        )
        log.info("Speaker encoder ready.")

    # ── Speaker persistence ───────────────────────────────────────────────────

    def _load_speaker_embeddings(self):
        if not os.path.exists(_SPEAKER_PATH):
            return
        try:
            with open(_SPEAKER_PATH) as f:
                data = json.load(f)
            self._next_id = data.get("next_id", 1)
            for sid, emb in data.get("speakers", {}).items():
                self._known_speakers[sid] = torch.tensor(emb)
            log.info("Loaded %d speaker(s) from disk.", len(self._known_speakers))
        except Exception as e:
            log.warning("Could not load speaker embeddings: %s", e)

    def _save_speaker_embeddings(self):
        try:
            os.makedirs(os.path.dirname(_SPEAKER_PATH), exist_ok=True)
            data = {
                "next_id": self._next_id,
                "speakers": {sid: emb.tolist() for sid, emb in self._known_speakers.items()},
            }
            with open(_SPEAKER_PATH, "w") as f:
                json.dump(data, f)
        except Exception as e:
            log.warning("Could not save speaker embeddings: %s", e)

    # ── VAD ───────────────────────────────────────────────────────────────────

    def is_speech(self, chunk_bytes: bytes) -> bool:
        audio   = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        tensor  = torch.from_numpy(audio)
        prob    = self._vad_model(tensor, SAMPLE_RATE).item()
        return prob > 0.5

    # ── STT ───────────────────────────────────────────────────────────────────

    def transcribe(self, audio_bytes: bytes) -> str:
        audio    = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._stt_model.transcribe(audio, beam_size=1, language="en")
        return " ".join(s.text for s in segments).strip()

    # ── Speaker ID ────────────────────────────────────────────────────────────

    def identify_speaker(self, audio_bytes: bytes) -> str:
        import torch.nn.functional as F
        audio    = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        signal   = torch.from_numpy(audio).unsqueeze(0)
        embedding = self._encoder.encode_batch(signal).squeeze(1)  # (1, 192)

        best_score, best_id = -1.0, None
        for sid, known_emb in self._known_speakers.items():
            score = F.cosine_similarity(embedding, known_emb.to(embedding.device), dim=1).item()
            if score > best_score:
                best_score, best_id = score, sid

        if best_score > 0.25 and best_id:
            return best_id

        # Auto-register if enough audio (≥ 1.5 s)
        duration = len(audio_bytes) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        if duration >= 1.5:
            new_id = f"User_{self._next_id}"
            self._known_speakers[new_id] = embedding
            self._next_id += 1
            self._save_speaker_embeddings()
            log.info("Registered new speaker: %s", new_id)
            return new_id

        return "User_1"  # fallback when audio is too short to register

    # ── Kokoro TTS ────────────────────────────────────────────────────────────

    async def synthesize(self, text: str) -> bytes:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{ELARA_URL}/tts",
                json={"text": text, "backend": "kokoro", "voice": "bf_emma", "speed": 0.9},
            )
            r.raise_for_status()
            return r.content


# Singleton — created at import time, loaded by orchestrator startup event
pipeline = AudioPipeline()


# ── WebSocket connection handler ───────────────────────────────────────────────

async def handle_audio_ws(websocket: WebSocket, handle_input_fn):
    """
    One WebSocket connection = one Pi/speaker session.

    Protocol:
      Incoming (Pi → RunPod): binary frames of raw PCM int16 at 16 kHz
      Outgoing (RunPod → Pi): binary WAV bytes (Kokoro output, 24 kHz)

    handle_input_fn is the orchestrator's handle_input() — passed in to
    avoid a circular import from audio_ws → main → audio_ws.
    """
    await websocket.accept()
    log.info("Pi audio connection opened from %s", websocket.client)

    audio_buffer: list[bytes] = []
    is_speaking               = False
    silence_start: Optional[float] = None

    try:
        while True:
            try:
                chunk = await asyncio.wait_for(websocket.receive_bytes(), timeout=30)
            except asyncio.TimeoutError:
                continue  # idle — keep connection alive

            # ── VAD ──────────────────────────────────────────────────────────
            has_speech = await asyncio.get_event_loop().run_in_executor(
                None, pipeline.is_speech, chunk
            )

            if has_speech:
                is_speaking   = True
                silence_start = None
                audio_buffer.append(chunk)
            else:
                if is_speaking:
                    audio_buffer.append(chunk)
                    if silence_start is None:
                        silence_start = time.monotonic()
                    elif time.monotonic() - silence_start > SILENCE_TIMEOUT:

                        # ── Complete phrase — process it ──────────────────────
                        full_audio    = b"".join(audio_buffer)
                        audio_buffer  = []
                        is_speaking   = False
                        silence_start = None

                        duration = len(full_audio) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                        if duration < MIN_AUDIO_LEN:
                            continue

                        loop = asyncio.get_event_loop()

                        # STT + Speaker ID in parallel (both are CPU/GPU bound)
                        text, speaker = await asyncio.gather(
                            loop.run_in_executor(None, pipeline.transcribe, full_audio),
                            loop.run_in_executor(None, pipeline.identify_speaker, full_audio),
                        )

                        if not text:
                            continue

                        log.info("[%s] %s", speaker, text)

                        # ── Orchestrator pipeline ─────────────────────────────
                        from app.models import AgentInput
                        req    = AgentInput(text=text, speaker=speaker)
                        result = await loop.run_in_executor(None, handle_input_fn, req)

                        if not result or not result.reply:
                            continue

                        # ── TTS → WAV back to Pi ──────────────────────────────
                        try:
                            wav_bytes = await pipeline.synthesize(result.reply)
                            await websocket.send_bytes(wav_bytes)
                        except Exception as e:
                            log.error("TTS synthesis failed: %s", e)

    except WebSocketDisconnect:
        log.info("Pi audio connection closed.")
    except Exception as e:
        log.error("WebSocket error: %s", e, exc_info=True)
