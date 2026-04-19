import time
import numpy as np
import threading
import queue


from mic_stream import MicrophoneStream
from vad_module import VADFilter
from speaker_manager import SpeakerManager
from stt_module import Transcriber


SILENCE_TIMEOUT = 0.8  
MIN_AUDIO_LEN = 0.5    

class VoiceAssistantCore:
    def __init__(self):
        print("\n[1/4] Initializing Microphone...")
        self.mic = MicrophoneStream()
        
        print("[2/4] Initializing VAD...")
        self.vad = VADFilter()
        
        print("[3/4] Initializing Speaker ID...")
        self.speaker_mgr = SpeakerManager()
        
        print("[4/4] Initializing STT Engine...")
        self.stt = Transcriber()
        
        print("\n✅ SYSTEM READY. Speak now!")

    def process_audio(self, audio_data):
        """
        Runs STT and Speaker ID in parallel for speed.
        """
        # Convert bytes to float32 numpy array once
        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        # We use a dictionary to capture results from threads
        results = {"text": "", "speaker": "Unknown", "confidence": 0.0}

        # --- Define Worker Functions ---
        def run_stt():
            results["text"] = self.stt.transcribe(audio_float32)

        def run_id():
            spk, conf = self.speaker_mgr.identify_speaker(audio_float32)
            
            # Auto-register if unknown and audio is long enough (1.5s+)
            # Note: For production, you might want a specific 'Enroll' command instead.
            if spk == "UNKNOWN" and len(audio_float32) > 16000 * 1.5:
                print("   [+] New Speaker detected. Registering...")
                spk = self.speaker_mgr.register_new_speaker(audio_float32)
                
            results["speaker"] = spk
            results["confidence"] = conf

        # --- Run in Threads ---
        t1 = threading.Thread(target=run_stt)
        t2 = threading.Thread(target=run_id)
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
        
        return results

    def run(self):
        self.mic.start()
        
        audio_buffer = []      # Stores chunks of current phrase
        is_speaking = False
        silence_start = None
        
        try:
            while True:
                chunk = self.mic.get_audio_chunk()
                
                if chunk:
                    # 1. Check VAD
                    has_speech = self.vad.is_speech(chunk)
                    
                    if has_speech:
                        is_speaking = True
                        silence_start = None
                        audio_buffer.append(chunk)
                    else:
                        # Logic: We were speaking, now it's silent
                        if is_speaking:
                            audio_buffer.append(chunk) # Add a bit of trailing silence for naturalness
                            
                            if silence_start is None:
                                silence_start = time.time()
                            
                            # Check if silence has lasted long enough to stop
                            elif time.time() - silence_start > SILENCE_TIMEOUT:
                                # --- PHRASE COMPLETE: PROCESS NOW ---
                                full_audio = b''.join(audio_buffer)
                                duration = len(full_audio) / 32000 # 16khz * 2 bytes
                                
                                if duration >= MIN_AUDIO_LEN:
                                    print(f"\nProcessing {duration:.2f}s audio...")
                                    output = self.process_audio(full_audio)
                                    
                                    # --- FINAL OUTPUT ---
                                    print(f"🎤 {output['speaker']} ({output['confidence']:.2f}): {output['text']}")
                                    
                                    
                                # Reset state
                                audio_buffer = []
                                is_speaking = False
                                silence_start = None
                                print("Listening...")

                else:
                    time.sleep(0.001) # Sleep briefly if no audio to save CPU

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.mic.stop()

if __name__ == "__main__":
    assistant = VoiceAssistantCore()
    assistant.run()