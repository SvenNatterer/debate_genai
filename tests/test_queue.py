import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from audio_engine import EdgeTTSProvider, AudioQueueManager

def test_queue():
    provider = EdgeTTSProvider()
    queue_manager = AudioQueueManager(provider)
    
    print("Enqueueing sentence 1 (Left)...")
    queue_manager.enqueue("Hello, I am speaking from the left.", "en-US-ChristopherNeural", -1.0)
    
    print("Enqueueing sentence 2 (Right)...")
    queue_manager.enqueue("And I am responding from the right side.", "en-GB-RyanNeural", 1.0)
    
    print("Enqueueing sentence 3 (Center)...")
    queue_manager.enqueue("Finally, I conclude from the center.", "en-US-MichelleNeural", 0.0)
    
    print("Waiting for queue to finish...")
    queue_manager.queue.join()
    
    print("All audio played. Stopping queue manager...")
    queue_manager.stop()

if __name__ == "__main__":
    test_queue()
