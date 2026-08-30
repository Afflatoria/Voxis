"""Tests for streaming session state and cancellation."""

import asyncio

import pytest

from neural_tts.models.mock import MockBackend
from neural_tts.streaming.scheduler import InferenceScheduler
from neural_tts.streaming.session import StreamingSession


@pytest.mark.asyncio
async def test_sequence_numbers_increase():
    backend = MockBackend(chunk_duration=0.05, char_seconds=0.02)
    await backend.load()
    session = StreamingSession(backend, InferenceScheduler().gpu_lock)

    sequences = []
    async for chunk in session.synthesize(text="hello world"):
        sequences.append(chunk.sequence)

    assert sequences == list(range(len(sequences)))
    assert len(sequences) > 0


@pytest.mark.asyncio
async def test_cancellation():
    backend = MockBackend(chunk_duration=0.2, char_seconds=0.15)
    await backend.load()
    session = StreamingSession(backend, InferenceScheduler().gpu_lock)

    async def collect():
        chunks = []
        async for chunk in session.synthesize(text="a" * 80):
            chunks.append(chunk)
            if len(chunks) >= 2:
                session.cancel()
        return chunks

    chunks = await collect()
    assert len(chunks) >= 2
    assert session.state.status.value in ("cancelled", "completed")


@pytest.mark.asyncio
async def test_voice_update_on_session():
    backend = MockBackend()
    await backend.load()
    session = StreamingSession(backend, InferenceScheduler().gpu_lock)
    session.update_voice({"speaking_rate": 1.5})
    assert session.state.current_voice.speaking_rate == 1.5


@pytest.mark.asyncio
async def test_empty_text_raises():
    backend = MockBackend()
    await backend.load()
    session = StreamingSession(backend, InferenceScheduler().gpu_lock)
    with pytest.raises(ValueError):
        async for _ in session.synthesize(text="   "):
            pass
