"""Model-independent contracts for Phase 2 neural timbre conversion."""

from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neural_tts.voice.identity_space import SpeakerIdentity


@dataclass(frozen=True, slots=True)
class AudioBuffer:
    """Normalized mono floating-point audio."""

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=np.float32)
        if samples.ndim == 2 and 1 in samples.shape:
            samples = samples.reshape(-1)
        if samples.ndim != 1:
            raise ValueError("audio samples must be mono")
        if samples.size == 0:
            raise ValueError("audio samples must not be empty")
        if not np.all(np.isfinite(samples)):
            raise ValueError("audio samples must be finite")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        samples = np.clip(samples, -1.0, 1.0).copy()
        samples.setflags(write=False)
        object.__setattr__(self, "samples", samples)

    @property
    def duration_seconds(self) -> float:
        return self.samples.size / self.sample_rate


@dataclass(frozen=True, slots=True)
class ConverterInfo:
    name: str
    model: str
    embedding_dimensions: int
    native_sample_rate: int
    device: str
    ready: bool
    supports_streaming: bool


@dataclass(frozen=True, slots=True)
class ConversionResult:
    audio: AudioBuffer
    conversion_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rtf(self) -> float:
        return self.conversion_seconds / self.audio.duration_seconds


class TimbreConverter(abc.ABC):
    """Separates linguistic content from target speaker timbre."""

    @abc.abstractmethod
    async def load(self) -> None:
        """Load the encoder, converter, and vocoder."""

    @abc.abstractmethod
    def info(self) -> ConverterInfo:
        """Return converter capabilities and runtime state."""

    @abc.abstractmethod
    async def extract_identity(
        self,
        reference: AudioBuffer,
        *,
        identity_id: str,
        name: str,
    ) -> SpeakerIdentity:
        """Extract a normalized target-speaker embedding."""

    @abc.abstractmethod
    async def convert(
        self,
        source: AudioBuffer,
        *,
        target: SpeakerIdentity,
        source_identity: SpeakerIdentity | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> ConversionResult:
        """Convert source speech content into the target identity."""

    @abc.abstractmethod
    async def aclose(self) -> None:
        """Release model and accelerator resources."""
