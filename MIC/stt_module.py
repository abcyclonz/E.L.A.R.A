from faster_whisper import WhisperModel
import os

class Transcriber:
    def __init__(self, model_size="base.en", device="cpu", compute_type="int8"):
        print(f"[*] Loading Faster-Whisper ({model_size}) on {device}...")
        
        # 'device' should be 'cuda' if you have an NVIDIA GPU, else 'cpu'
        # 'compute_type' can be 'float16' (GPU) or 'int8' (CPU)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        
        print("[*] Whisper Model Loaded.")

    def transcribe(self, audio_data):
        """
        Input: Numpy array (float32)
        Output: String text
        """
        # We assume audio_data is already flat float32 16kHz
        # beam_size=1 is the fastest setting (greedy search)
        segments, info = self.model.transcribe(audio_data, beam_size=1, language="en")
        
        full_text = ""
        for segment in segments:
            full_text += segment.text + " "
            
        return full_text.strip()

# --- TEST CODE ---
if __name__ == "__main__":
    # This test requires a mic recording, so we will simulate or just load the model.
    print("Initializing Transcriber to download model...")
    stt = Transcriber()
    print("Success! Model downloaded and ready.")