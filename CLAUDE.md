# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a real-time voice assistant pipeline built in Python. The `MIC/` module handles end-to-end speech processing: microphone capture → voice activity detection → speaker identification → speech-to-text transcription.

## Running the System

```bash
cd MIC
python main.py
```

Each module can also be run standalone for isolated testing:

```bash
python mic_stream.py       # Test mic input (5-second capture with volume bars)
python vad_module.py       # Test VAD (prints SPEECH DETECTED or dots for silence)
python stt_module.py       # Test Whisper model download and init
```

## Dependencies & Environment

The project uses a virtual environment at `MIC/.venv/`. Models are downloaded on first run and cached locally:
- Silero VAD → Torch Hub cache
- SpeechBrain ECAPA-TDNN → `MIC/tmp_model/`
- Faster-Whisper → Hugging Face cache

Audio format throughout the pipeline: **16kHz, mono, int16** (raw bytes from PyAudio), converted to **float32 normalized [-1.0, 1.0]** before ML inference.

## Architecture

The pipeline in `main.py` is a single-threaded loop with two parallelized stages:

```
MicrophoneStream (PyAudio callback → Queue)
    ↓ raw bytes (int16)
VADFilter (Silero VAD via torch.hub)
    ↓ bool: is_speech
[buffer accumulation + silence timeout logic in VoiceAssistantCore.run()]
    ↓ complete phrase (raw bytes)
VoiceAssistantCore.process_audio()
    ├── Thread 1 → Transcriber (Faster-Whisper "base.en", CPU int8)
    └── Thread 2 → SpeakerManager (SpeechBrain ECAPA-TDNN, cosine similarity)
                       └── auto-registers new speakers if audio ≥ 1.5s
```

**Key timing constants** (in `main.py`):
- `SILENCE_TIMEOUT = 0.8s` — silence duration to mark end of phrase
- `MIN_AUDIO_LEN = 0.5s` — minimum phrase length to run inference

**Speaker identification** stores embeddings in-memory only (`SpeakerManager.known_speakers` dict); there is no persistence across runs. Speaker IDs are assigned as `User_1`, `User_2`, etc. The similarity threshold is `0.25` (cosine similarity).

## GPU Acceleration

`Transcriber` defaults to `device="cpu"`, `compute_type="int8"`. To use a GPU, change these to `device="cuda"`, `compute_type="float16"` when instantiating `Transcriber` in `main.py`.
