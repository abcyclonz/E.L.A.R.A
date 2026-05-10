FROM python:3.11-slim
WORKDIR /app

# System libs needed by speechbrain / torchaudio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persistent directory for speaker embeddings + downloaded speaker model
RUN mkdir -p /data/speaker_model

EXPOSE 8001
