"""Deterministic converter used to validate Phase 2 plumbing without models."""

from __future__ import annotations

import asyncio
import time

import numpy as np

from neural_tts.timbre.base import (
    AudioBuffer,
    ConversionResult,
    ConverterInfo,
    TimbreConverter,
)
from neural_tts.voice.identity_space import SpeakerIdentity


class MockTimbreConverter(TimbreConverter):
    """Extracts spectral test features and returns source audio unchanged."""

    EMBEDDING_DIMENSIONS = 32

    def __init__(self) -> None:
        self._ready = False

    async def load(self) -> None:
        self._ready = True

    def info(self) -> ConverterInfo:
        return ConverterInfo(
            name="mock",
            model="spectral-passthrough",
            embedding_dimensions=self.EMBEDDING_DIMENSIONS,
            native_sample_rate=24000,
            device="cpu",
            ready=self._ready,
            supports_streaming=False,
        )

    async def extract_identity(
        self,
        reference: AudioBuffer,
        *,
        identity_id: str,
        name: str,
    ) -> SpeakerIdentity:
        if not self._ready:
            await self.load()
        if float(np.max(np.abs(reference.samples))) <= 1e-6:
            raise ValueError("cannot extract an identity from silent audio")

        spectrum = np.abs(np.fft.rfft(reference.samples))
        bands = np.array_split(spectrum, self.EMBEDDING_DIMENSIONS)
        features = np.asarray(
            [np.log1p(float(np.mean(band))) for band in bands],
            dtype=np.float32,
        )
        return SpeakerIdentity(
            id=identity_id,
            name=name,
            embedding=features.tolist(),
            converter=self.info().model,
        )

    async def convert(
        self,
        source: AudioBuffer,
        *,
        target: SpeakerIdentity,
        source_identity: SpeakerIdentity | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> ConversionResult:
        if not self._ready:
            await self.load()
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError
        if len(target.embedding) != self.EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"target embedding must have {self.EMBEDDING_DIMENSIONS} dimensions"
            )

        started = time.perf_counter()
        output = AudioBuffer(source.samples.copy(), source.sample_rate)
        return ConversionResult(
            audio=output,
            conversion_seconds=time.perf_counter() - started,
            metadata={
                "passthrough": True,
                "target_identity": target.id,
                "source_identity": source_identity.id if source_identity else None,
            },
        )

    async def aclose(self) -> None:
        self._ready = False
