"""Streaming session orchestration."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator, Callable, Awaitable
from typing import Any

from neural_tts.audio.codec import AudioChunk
from neural_tts.logging_config import get_logger
from neural_tts.models.base import StreamingTTS
from neural_tts.streaming.state import SessionStatus, StreamState
from neural_tts.voice.schema import VoiceConfig

logger = get_logger(__name__)

_SPEECH_BOUNDARY = re.compile(r"(?<=[.!?;:,])\s+")


def split_streaming_text(text: str, max_chars: int = 48) -> list[str]:
    """Split text into short semantic units without losing or repeating words."""
    normalized = " ".join(text.split())
    if not normalized:
        return []

    units: list[str] = []
    for phrase in _SPEECH_BOUNDARY.split(normalized):
        words = phrase.split()
        current: list[str] = []
        current_length = 0
        for word in words:
            added_length = len(word) + (1 if current else 0)
            if current and current_length + added_length > max_chars:
                units.append(" ".join(current))
                current = []
                current_length = 0
            current.append(word)
            current_length += len(word) + (1 if len(current) > 1 else 0)
        if current:
            units.append(" ".join(current))
    return units


class StreamingSession:
    """Manages one streaming synthesis lifecycle with cancellation support."""

    def __init__(self, backend: StreamingTTS, gpu_lock: asyncio.Lock) -> None:
        self._backend = backend
        self._gpu_lock = gpu_lock
        self.state = StreamState()

    async def synthesize(
        self,
        text: str,
        voice: dict[str, Any] | None = None,
        request_id: str | None = None,
        on_chunk: Callable[[AudioChunk, StreamState], Awaitable[None]] | None = None,
    ) -> AsyncIterator[AudioChunk]:
        if not text or not text.strip():
            raise ValueError("text must not be empty")

        session_id = self.state.session_id
        current_voice = self.state.current_voice
        voice_version = self.state.voice_version
        self.state = StreamState(
            session_id=session_id,
            current_voice=current_voice,
            voice_version=voice_version,
        )
        self.state.request_id = request_id
        self.state.status = SessionStatus.STREAMING
        self.state.timestamps.request_received = time.monotonic()
        self.state.cancel_event = asyncio.Event()

        if voice:
            self.state.update_voice(voice)

        unsupported = self.state.current_voice.unsupported_active_controls()
        if unsupported:
            logger.warning(
                "unsupported_voice_controls",
                session_id=self.state.session_id,
                controls=unsupported,
            )

        generation_started = time.monotonic()

        segments = split_streaming_text(text.strip())

        async with self._gpu_lock:
            if self.state.is_cancelled():
                return

            self.state.timestamps.inference_started = time.monotonic()

            for segment_index, segment in enumerate(segments):
                if self.state.is_cancelled():
                    break

                # Snapshot at a text boundary. Mid-stream updates apply to the
                # next segment without restarting or duplicating spoken text.
                voice = self.state.current_voice.model_copy(deep=True)
                voice_version = self.state.voice_version
                logger.debug(
                    "synthesis_segment_started",
                    session_id=self.state.session_id,
                    segment_index=segment_index,
                    voice_version=voice_version,
                    text_preview=segment[:60],
                )

                async for backend_chunk in self._backend.synthesize(
                    text=segment,
                    voice=voice,
                    cancel_event=self.state.cancel_event,
                ):
                    if self.state.is_cancelled():
                        logger.info(
                            "synthesis_cancelled",
                            session_id=self.state.session_id,
                            sequence=self.state.sequence_number,
                        )
                        break

                    chunk = AudioChunk(
                        sequence=self.state.next_sequence(),
                        sample_rate=backend_chunk.sample_rate,
                        pcm_s16le=backend_chunk.pcm_s16le,
                        duration_seconds=backend_chunk.duration_seconds,
                    )
                    if self.state.timestamps.first_audio_generated is None:
                        self.state.timestamps.first_audio_generated = time.monotonic()

                    if on_chunk is not None:
                        await on_chunk(chunk, self.state)

                    if self.state.timestamps.first_audio_sent is None:
                        self.state.timestamps.first_audio_sent = time.monotonic()
                    self.state.mark_chunk_sent(chunk.duration_seconds)
                    yield chunk

        self.state.metrics.generation_seconds = time.monotonic() - generation_started
        self.state.timestamps.generation_completed = time.monotonic()

        if self.state.is_cancelled():
            self.state.status = SessionStatus.CANCELLED
        else:
            self.state.status = SessionStatus.COMPLETED

        logger.info(
            "synthesis_finished",
            session_id=self.state.session_id,
            status=self.state.status.value,
            rtf=self.state.metrics.rtf,
            chunks=self.state.metrics.chunk_count,
        )

    def cancel(self) -> None:
        self.state.request_cancel()

    def update_voice(self, partial: dict[str, Any]) -> VoiceConfig:
        """Apply an update at the next short text-segment boundary."""
        return self.state.update_voice(partial)
