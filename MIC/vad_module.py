
import torch
import numpy as np

class VADFilter:
    def __init__(self, sampling_rate=16000, threshold=0.5):
        self.sampling_rate = sampling_rate
        self.threshold = threshold
        
        print("[*] Loading Silero VAD model...")
        # Load the pre-trained model directly from Torch Hub (downloads once, then caches)
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        (self.get_speech_timestamps, _, self.read_audio, _, _) = utils
        print("[*] VAD Model Loaded.")

    def is_speech(self, audio_chunk):
        """
        Takes raw audio bytes (int16), converts to float32 tensor,
        and asks the AI: 'Is this human speech?'
        """
        # 1. Convert bytes to numpy float32 (normalized -1.0 to 1.0)
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        # 2. Convert to PyTorch tensor
        audio_tensor = torch.from_numpy(audio_float32)
        
        # 3. Get probability of speech
        # We assume the chunk is a small part of a stream
        speech_prob = self.model(audio_tensor, self.sampling_rate).item()
        
        # 4. Return True if above threshold
        return speech_prob > self.threshold

# --- TEST CODE (Run this file directly to test) ---
if __name__ == "__main__":
    from mic_stream import MicrophoneStream
    import time

    # Initialize Mic and VAD
    mic = MicrophoneStream()
    vad = VADFilter() # Downloads model on first run

    mic.start()
    print("\n--- LISTENING FOR HUMAN SPEECH ---\n")

    try:
        while True:
            chunk = mic.get_audio_chunk()
            
            if chunk:
                # Check if it's speech
                if vad.is_speech(chunk):
                    print("🗣️  SPEECH DETECTED")
                else:
                    # Optional: Print a dot for silence just to see it's working
                    print(".", end="", flush=True)
                    
            # Small sleep to prevent CPU hogging in this simple loop
            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()