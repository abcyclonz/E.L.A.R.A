import pyaudio
import queue
import threading
import numpy as np

class MicrophoneStream:
    def __init__(self, rate=16000, chunk=512, channels=1):
        self.rate = rate
        self.chunk = chunk
        self.channels = channels
        
        # This queue will hold the audio chunks. 
        # Other modules will "get" data from here.
        self.audio_queue = queue.Queue()
        
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False

    def _callback(self, in_data, frame_count, time_info, status):
        """This function runs whenever the mic has new data."""
        self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)

    def start(self):
        """Starts the microphone stream in a non-blocking way."""
        if self.is_running:
            return
            
        print(f"[*] Mic started: {self.rate}Hz, Mono")
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self._callback  # Runs in a separate thread automatically
        )
        self.stream.start_stream()
        self.is_running = True

    def stop(self):
        """Cleanly stops the stream."""
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        print("[*] Mic stopped.")

    def get_audio_chunk(self):
        """
        Retrieves the next chunk of audio. 
        Returns None if queue is empty.
        """
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

# --- TEST CODE (Run this file directly to test) ---
if __name__ == "__main__":
    import time
    
    # Initialize the stream
    mic = MicrophoneStream()
    mic.start()

    print("Listening... (Press Ctrl+C to stop)")
    
    try:
        start_time = time.time()
        # Listen for 5 seconds then stop
        while time.time() - start_time < 5:
            chunk = mic.get_audio_chunk()
            if chunk:
                # Calculate volume just to verify we are hearing something
                # Convert bytes to numpy array
                data_np = np.frombuffer(chunk, dtype=np.int16)
                volume = np.linalg.norm(data_np) / 10
                
                # Print a visual bar for volume
                print(f"Volume: {'|' * int(volume // 50)}")
                
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()