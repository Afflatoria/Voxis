"""Streaming session orchestration."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Awaitable
from typing import Any

from neural_tts.audio.codec import AudioChunk
from neural_tts.logging_config import get_logger
from neural_tts.models.base import StreamingTTS
from neural_tts.streaming.state import SessionStatus, StreamState

logger = get_logger(__name__)


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
        self.state = StreamState(session_id=session_id)
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

        async with self._gpu_lock:
            if self.state.is_cancelled():
                return

            self.state.timestamps.inference_started = time.monotonic()

            async for chunk in self._backend.synthesize(
                text=text.strip(),
                voice=self.state.current_voice,
                cancel_event=self.state.cancel_event,
            ):
                if self.state.is_cancelled():
                    logger.info(
                        "synthesis_cancelled",
                        session_id=self.state.session_id,
                        sequence=chunk.sequence,
                    )
                    break

                if self.state.timestamps.first_audio_generated is None:
                    self.state.timestamps.first_audio_generated = time.monotonic()

                if on_chunk is not None:
                    await on_chunk(chunk, self.state)

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

    def update_voice(self, partial: dict[str, Any]) -> None:
        """Store voice update for current/future chunks (M1: affects next synthesis)."""
        self.state.update_voice(partial)
