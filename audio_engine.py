import io
import asyncio
import edge_tts
import queue
import threading
import pygame
import re
import time
import json
import os
from abc import ABC, abstractmethod
from openai import OpenAI
from dotenv import load_dotenv
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class AudioProvider(ABC):
    """Abstract interface for audio providers."""
    
    @abstractmethod
    def generate_audio(self, text: str, voice_id: str, rate: str = "+0%") -> bytes:
        """
        Generates audio from text using the given voice_id.
        rate: Edge TTS rate string, e.g. '+0%' (normal), '+50%' (1.5×), '-20%' (0.8×).
        Returns the raw MP3/WAV bytes.
        """
        pass

class OpenAITTSProvider(AudioProvider):
    """Audio provider using OpenAI's TTS API."""
    
    def __init__(self):
        # We assume OPENAI_API_KEY is in the environment
        self.client = OpenAI()

    def generate_audio(self, text: str, voice_id: str, rate: str = "+0%") -> bytes:
        """
        Calls OpenAI TTS API (tts-1) and returns MP3 bytes.
        Valid voice_ids for OpenAI: alloy, echo, fable, onyx, nova, shimmer.
        Note: rate is ignored for OpenAI TTS (no direct speed control in tts-1).
        """
        response = self.client.audio.speech.create(
            model="tts-1",
            voice=voice_id,
            input=text
        )
        return response.read()

import asyncio
import edge_tts

class EdgeTTSProvider(AudioProvider):
    """Audio provider using Microsoft Edge TTS (Free, no API key)."""
    
    def generate_audio(self, text: str, voice_id: str, rate: str = "+0%") -> bytes:
        """
        Calls Edge TTS API and returns MP3 bytes.
        rate: Edge TTS rate string, e.g. '+0%', '+50%', '-20%'.
        """
        async def _generate():
            communicate = edge_tts.Communicate(text, voice_id, rate=rate)
            audio_data = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.extend(chunk["data"])
            return bytes(audio_data)

        return asyncio.run(_generate())

class AudioQueueManager:
    """Manages asynchronous audio generation and playback with stereo panning."""
    def __init__(self, provider: AudioProvider):
        self.provider = provider
        self.queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.is_running = True
        pygame.mixer.init()
        self.generator_thread = threading.Thread(target=self._generator_loop, daemon=True)
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.generator_thread.start()
        self.worker_thread.start()
        
    def enqueue(self, text: str, voice_id: str, panning: float = 0.0, speaker_speed: float = 1.0):
        """Adds a text chunk to the playback queue.
        
        speaker_speed: playback speed multiplier (e.g. 1.0 = normal, 1.5 = 50% faster, 0.8 = 20% slower).
        """
        # Convert speed multiplier to Edge TTS rate string: 1.5 → '+50%', 0.8 → '-20%'
        rate_pct = round((speaker_speed - 1.0) * 100)
        rate = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"
        self.queue.put((text, voice_id, panning, rate))
        
    def _generator_loop(self):
        while self.is_running:
            try:
                text, voice_id, panning, rate = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                start_time = time.time()
                audio_bytes = self.provider.generate_audio(text, voice_id, rate)

                latency = time.time() - start_time
                self._log_latency(text, voice_id, latency)
                self.audio_queue.put((audio_bytes, text, voice_id, panning))
            except Exception as e:
                print(f"AudioGenerator Error: {e}")
                self.queue.task_done()

    def _worker_loop(self):
        while self.is_running or not self.audio_queue.empty():
            try:
                audio_bytes, text, voice_id, panning = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                # Load into pygame Sound
                stream = io.BytesIO(audio_bytes)
                sound = pygame.mixer.Sound(stream)

                # Find an available channel
                channel = pygame.mixer.find_channel()
                if not channel:
                    channel = pygame.mixer.Channel(0)

                # Calculate volume based on panning (-1.0 to 1.0)
                # panning = -1.0 -> left=1.0, right=0.0
                # panning = 0.0 -> left=1.0, right=1.0
                left_vol = min(1.0, 1.0 - panning)
                right_vol = min(1.0, 1.0 + panning)
                channel.set_volume(left_vol, right_vol)

                channel.play(sound)

                # Wait for playback to finish before processing the next chunk
                while channel.get_busy():
                    pygame.time.Clock().tick(30)
            except Exception as e:
                print(f"AudioWorker Error: {e}")
            finally:
                self.audio_queue.task_done()
                self.queue.task_done()
                
    def _log_latency(self, text: str, voice_id: str, latency: float):
        log_file = "audio_latency_log.json"
        log_entry = {
            "timestamp": time.time(),
            "voice_id": voice_id,
            "latency_seconds": round(latency, 4),
            "text_snippet": text[:50] + "..." if len(text) > 50 else text
        }
        logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                pass
        logs.append(log_entry)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4)
            
    def stop(self):
        self.is_running = False

_global_audio_manager = None
def get_audio_manager() -> AudioQueueManager:
    global _global_audio_manager
    if _global_audio_manager is None:
        _global_audio_manager = AudioQueueManager(EdgeTTSProvider())
    return _global_audio_manager

class SentenceChunker:
    """Buffers text and emits complete sentences."""
    def __init__(self, on_sentence_complete):
        self.buffer = ""
        self.on_sentence_complete = on_sentence_complete
        
    def add_token(self, token: str):
        self.buffer += token
        # Simple sentence boundary detection
        match = re.search(r'([.!?]+)(?:\s+|\n+)', self.buffer)
        if match:
            split_idx = match.end()
            sentence = self.buffer[:split_idx].strip()
            self.buffer = self.buffer[split_idx:]
            if sentence:
                self.on_sentence_complete(sentence)
                
    def flush(self):
        sentence = self.buffer.strip()
        if sentence:
            self.on_sentence_complete(sentence)
        self.buffer = ""
