"""Backend factory."""

from __future__ import annotations

from neural_tts.config import Settings
from neural_tts.models.base import StreamingTTS
from neural_tts.models.mock import MockBackend


def create_backend(settings: Settings) -> StreamingTTS:
    if settings.backend == "mock":
        return MockBackend()
    from neural_tts.models.f5tts import F5TTSBackend

    return F5TTSBackend(settings)
