"""Mock streaming backend for tests and development without GPU model."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator

import numpy as np

from neural_tts.audio.codec import AudioChunk, float32_to_pcm_s16le
from neural_tts.models.base import BackendInfo, StreamingTTS
from neural_tts.voice.schema import F5TTS_SUPPORTED_CONTROLS, VoiceConfig


class MockBackend(StreamingTTS):
    """Generates synthetic PCM chunks to validate streaming infrastructure."""

    def __init__(
        self,
        sample_rate: int = 24000,
        chunk_duration: float = 0.25,
        char_seconds: float = 0.06,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_duration = chunk_duration
        self._char_seconds = char_seconds
        self._load_time = 0.0
        self._ready = False

    async def load(self) -> None:
        started = time.monotonic()
        await asyncio.sleep(0.01)
        self._load_time = time.monotonic() - started
        self._ready = True

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="mock",
            model_name="sine-chunk-generator",
            sample_rate=self._sample_rate,
            device="cpu",
            load_time_seconds=self._load_time,
            supported_controls=F5TTS_SUPPORTED_CONTROLS,
            ready=self._ready,
        )

    async def synthesize(
        self,
        text: str,
        voice: VoiceConfig,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AudioChunk]:
        if not self._ready:
            await self.load()

        total_duration = max(len(text) * self._char_seconds / voice.speaking_rate, 0.5)
        frequency = 220.0 * (2.0 ** (voice.pitch * 0.5))  # pitch affects mock tone
        elapsed = 0.0
        sequence = 0

        while elapsed < total_duration:
            if cancel_event and cancel_event.is_set():
                return

            duration = min(self._chunk_duration, total_duration - elapsed)
            num_samples = int(duration * self._sample_rate)
            t = np.arange(num_samples, dtype=np.float32) / self._sample_rate
            phase = 2.0 * math.pi * frequency * (t + elapsed)
            envelope = min(voice.energy, 2.0)
            samples = (0.25 * envelope * np.sin(phase)).astype(np.float32)

            pcm = float32_to_pcm_s16le(samples)
            yield AudioChunk(
                sequence=sequence,
                sample_rate=self._sample_rate,
                pcm_s16le=pcm,
                duration_seconds=duration,
            )
            sequence += 1
            elapsed += duration
            await asyncio.sleep(0.02)

    async def aclose(self) -> None:
        self._ready = False
