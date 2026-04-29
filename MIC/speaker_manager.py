import json
import os

import torch
import torch.nn.functional as F
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier

# Default persist path; override with SPEAKER_EMBEDDINGS_PATH env var.
_DEFAULT_PERSIST_PATH = os.path.join(os.path.dirname(__file__), "speaker_embeddings.json")


class SpeakerManager:
    def __init__(self, threshold=0.25, persist_path: str | None = None):
        self.threshold = threshold
        self.known_speakers: dict[str, torch.Tensor] = {}
        self.next_id = 1
        self.persist_path = persist_path or os.getenv(
            "SPEAKER_EMBEDDINGS_PATH", _DEFAULT_PERSIST_PATH
        )

        print("[*] Loading Speaker Recognition Model...")
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="tmp_model"
        )
        print("[*] Speaker Model Loaded.")

        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load embeddings and next_id from the JSON persist file (if it exists)."""
        if not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
            self.next_id = data.get("next_id", 1)
            for speaker_id, embedding_list in data.get("speakers", {}).items():
                self.known_speakers[speaker_id] = torch.tensor(embedding_list)
            print(f"[*] Loaded {len(self.known_speakers)} speaker(s) from {self.persist_path}")
        except Exception as exc:
            print(f"[!] Could not load speaker embeddings: {exc}")

    def _save(self) -> None:
        """Persist current embeddings and next_id to the JSON file."""
        try:
            data = {
                "next_id": self.next_id,
                "speakers": {
                    sid: emb.tolist()
                    for sid, emb in self.known_speakers.items()
                },
            }
            with open(self.persist_path, "w") as f:
                json.dump(data, f)
        except Exception as exc:
            print(f"[!] Could not save speaker embeddings: {exc}")

    # ── Core methods ──────────────────────────────────────────────────────────

    def extract_embedding(self, audio_data):
        """
        Input: Numpy array of audio (float32).
        Output: Embedding tensor of shape (1, 192).
        """
        # Convert to torch tensor
        signal = torch.from_numpy(audio_data)
        
        # SpeechBrain expects input shape (Batch_Size, Time) -> (1, Length)
        if len(signal.shape) == 1:
            signal = signal.unsqueeze(0)
            
        # Get embedding (fingerprint)
        # Result is usually [1, 1, 192]. We want [1, 192] to make math easier.
        embedding = self.classifier.encode_batch(signal)
        
        # Squeeze out the middle "time" dimension
        # (1, 1, 192) -> (1, 192)
        return embedding.squeeze(1)

    def identify_speaker(self, audio_data):
        """
        Compares audio against known speakers.
        Returns: (speaker_id, score) or (None, score) if unknown.
        """
        # 1. Get embedding for current audio
        new_embedding = self.extract_embedding(audio_data)
        
        best_score = -1.0
        best_speaker = None
        
        # 2. Compare against all known speakers
        for speaker_id, registered_embedding in self.known_speakers.items():
            # Calculate similarity between the two (1, 192) vectors
            # dim=1 means "calculate across the 192 features"
            score = F.cosine_similarity(new_embedding, registered_embedding, dim=1)
            score = score.item() # Now this will safely convert to a float
            
            if score > best_score:
                best_score = score
                best_speaker = speaker_id
                
        # 3. Decision Logic
        if best_score > self.threshold:
            return best_speaker, best_score
        else:
            return "UNKNOWN", best_score

    def register_new_speaker(self, audio_data):
        """Creates a new ID for this audio and persists to disk."""
        embedding = self.extract_embedding(audio_data)
        new_id = f"User_{self.next_id}"
        self.known_speakers[new_id] = embedding
        self.next_id += 1
        self._save()
        return new_id