"""Tests for model adapter with mock backend."""

import pytest

from neural_tts.models.mock import MockBackend
from neural_tts.voice.schema import VoiceConfig


@pytest.mark.asyncio
async def test_mock_backend_info():
    backend = MockBackend()
    info = backend.info()
    assert info.name == "mock"
    assert info.ready is False
    await backend.load()
    info = backend.info()
    assert info.ready is True


@pytest.mark.asyncio
async def test_mock_synthesize_yields_pcm():
    backend = MockBackend(chunk_duration=0.1, char_seconds=0.05)
    await backend.load()
    voice = VoiceConfig()
    chunks = []
    async for chunk in backend.synthesize("test", voice):
        chunks.append(chunk)
    assert len(chunks) > 0
    assert len(chunks[0].pcm_s16le) > 0
    assert chunks[0].sample_rate == 24000
