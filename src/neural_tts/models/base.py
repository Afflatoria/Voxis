"""Streaming TTS backend abstraction."""

from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass

from neural_tts.audio.codec import AudioChunk
from neural_tts.voice.schema import VoiceConfig


@dataclass(frozen=True, slots=True)
class BackendInfo:
    name: str
    model_name: str
    sample_rate: int
    device: str
    load_time_seconds: float
    supported_controls: frozenset[str]
    ready: bool


class StreamingTTS(abc.ABC):
    """Model-independent streaming synthesis interface."""

    @abc.abstractmethod
    async def load(self) -> None:
        """Load model weights and warm up if needed."""

    @abc.abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: VoiceConfig,
        cancel_event: asyncio.Event | None = None,
    ):
        """
        Yield audio chunks as they become available.

        Each chunk is an AudioChunk with PCM s16le bytes.
        """
        raise NotImplementedError
        yield  # pragma: no cover

    @abc.abstractmethod
    def info(self) -> BackendInfo:
        """Return backend metadata for health/metrics endpoints."""

    @abc.abstractmethod
    async def aclose(self) -> None:
        """Release resources."""

    async def peak_gpu_memory_mb(self) -> float | None:
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception:
            return None
