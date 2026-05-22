import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import pygame
from audio_engine import EdgeTTSProvider

def test_panning():
    pygame.mixer.init()
    provider = EdgeTTSProvider()
    print("Generating audio...")
    audio_bytes = provider.generate_audio("Testing left panning.", "en-US-ChristopherNeural")
    
    print("Loading into Sound...")
    try:
        # EdgeTTS usually returns MP3 bytes
        stream = io.BytesIO(audio_bytes)
        sound = pygame.mixer.Sound(stream)
        channel = pygame.mixer.find_channel()
        
        print("Playing strictly left...")
        channel.set_volume(1.0, 0.0) # Left volume 1.0, Right volume 0.0
        channel.play(sound)
        
        while channel.get_busy():
            pygame.time.Clock().tick(10)
        print("Success.")
    except Exception as e:
        print("Error with Sound loading:", e)

if __name__ == "__main__":
    test_panning()
