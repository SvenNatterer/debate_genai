import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import pygame
from audio_engine import EdgeTTSProvider

def test_solo_audio():
    print("Initialize pygame mixer...")
    pygame.mixer.init()

    print("Initialize EdgeTTSProvider...")
    provider = EdgeTTSProvider()
    
    text = "Dies ist ein erster Test der modularen Audio-Architektur."
    voice = "de-DE-KillianNeural"
    
    print(f"Requesting audio for text: '{text}' using voice '{voice}'...")
    try:
        audio_bytes = provider.generate_audio(text, voice)
        print("Audio generated successfully! Bytes received:", len(audio_bytes))
        
        # Load the bytes into pygame and play
        audio_stream = io.BytesIO(audio_bytes)
        pygame.mixer.music.load(audio_stream)
        
        print("Playing audio...")
        pygame.mixer.music.play()
        
        # Keep script running while audio plays
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
        print("Playback finished.")
    except Exception as e:
        print("Error during audio generation or playback:", e)

if __name__ == "__main__":
    test_solo_audio()
