"""Streaming session state."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from neural_tts.voice.schema import VoiceConfig


class SessionStatus(str, Enum):
    IDLE = "idle"
    STREAMING = "streaming"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class LatencyTimestamps:
    """Performance timestamps for one synthesis request (monotonic seconds)."""

    request_received: float | None = None
    inference_started: float | None = None
    first_audio_generated: float | None = None
    first_audio_sent: float | None = None
    generation_completed: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "request_received": self.request_received,
            "inference_started": self.inference_started,
            "first_audio_generated": self.first_audio_generated,
            "first_audio_sent": self.first_audio_sent,
            "generation_completed": self.generation_completed,
        }


@dataclass
class StreamMetrics:
    """Aggregated metrics for a streaming session."""

    chunk_count: int = 0
    total_audio_seconds: float = 0.0
    generation_seconds: float = 0.0
    chunk_durations: list[float] = field(default_factory=list)
    chunk_intervals: list[float] = field(default_factory=list)
    peak_gpu_memory_mb: float | None = None

    @property
    def rtf(self) -> float | None:
        if self.total_audio_seconds <= 0:
            return None
        return self.generation_seconds / self.total_audio_seconds

    @property
    def avg_chunk_duration(self) -> float | None:
        if not self.chunk_durations:
            return None
        return sum(self.chunk_durations) / len(self.chunk_durations)

    def to_dict(self, timestamps: LatencyTimestamps) -> dict[str, Any]:
        ttfa_server_ms: float | None = None
        if (
            timestamps.request_received is not None
            and timestamps.first_audio_sent is not None
        ):
            ttfa_server_ms = (
                timestamps.first_audio_sent - timestamps.request_received
            ) * 1000.0

        return {
            "ttfa_server_ms": ttfa_server_ms,
            "rtf": self.rtf,
            "generation_seconds": self.generation_seconds,
            "total_audio_seconds": self.total_audio_seconds,
            "chunk_count": self.chunk_count,
            "avg_chunk_duration_seconds": self.avg_chunk_duration,
            "chunk_intervals_seconds": self.chunk_intervals,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
            "timestamps": timestamps.to_dict(),
        }


@dataclass
class StreamState:
    """Mutable state for one WebSocket streaming session."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    current_voice: VoiceConfig = field(default_factory=VoiceConfig)
    sequence_number: int = 0
    cancelled: bool = False
    status: SessionStatus = SessionStatus.IDLE
    request_id: str | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    timestamps: LatencyTimestamps = field(default_factory=LatencyTimestamps)
    metrics: StreamMetrics = field(default_factory=StreamMetrics)
    _last_chunk_sent_at: float | None = field(default=None, repr=False)

    def request_cancel(self) -> None:
        self.cancelled = True
        self.cancel_event.set()
        if self.status == SessionStatus.STREAMING:
            self.status = SessionStatus.CANCELLED

    def is_cancelled(self) -> bool:
        return self.cancelled or self.cancel_event.is_set()

    def next_sequence(self) -> int:
        seq = self.sequence_number
        self.sequence_number += 1
        return seq

    def update_voice(self, partial: dict[str, Any]) -> VoiceConfig:
        """Apply a partial voice update; used now and for future mid-stream updates."""
        self.current_voice = self.current_voice.merge_update(partial)
        return self.current_voice

    def mark_chunk_sent(self, duration_seconds: float) -> None:
        now = time.monotonic()
        self.metrics.chunk_count += 1
        self.metrics.total_audio_seconds += duration_seconds
        self.metrics.chunk_durations.append(duration_seconds)
        if self._last_chunk_sent_at is not None:
            self.metrics.chunk_intervals.append(now - self._last_chunk_sent_at)
        self._last_chunk_sent_at = now
