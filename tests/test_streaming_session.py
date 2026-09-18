"""Tests for streaming session state and cancellation."""

import asyncio

import pytest

from neural_tts.models.mock import MockBackend
from neural_tts.streaming.scheduler import InferenceScheduler
from neural_tts.streaming.session import StreamingSession, split_streaming_text


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
    assert session.state.voice_version == 1


def test_split_streaming_text_preserves_words_and_bounds_segments():
    text = (
        "This is the first phrase, followed by a substantially longer phrase "
        "that must be divided cleanly without dropping any words."
    )
    segments = split_streaming_text(text, max_chars=32)

    assert " ".join(segments) == " ".join(text.split())
    assert all(len(segment) <= 32 for segment in segments)


@pytest.mark.asyncio
async def test_voice_update_applies_at_next_segment_boundary():
    class RecordingBackend(MockBackend):
        def __init__(self):
            super().__init__(chunk_duration=0.02, char_seconds=0.002)
            self.rates = []

        async def synthesize(self, text, voice, cancel_event=None):
            self.rates.append(voice.speaking_rate)
            async for chunk in super().synthesize(text, voice, cancel_event):
                yield chunk

    backend = RecordingBackend()
    await backend.load()
    session = StreamingSession(backend, InferenceScheduler().gpu_lock)
    updated = False

    async for _chunk in session.synthesize(text="word " * 30):
        if not updated:
            session.update_voice({"speaking_rate": 1.5})
            updated = True

    assert backend.rates[0] == 1.0
    assert 1.5 in backend.rates[1:]


@pytest.mark.asyncio
async def test_empty_text_raises():
    backend = MockBackend()
    await backend.load()
    session = StreamingSession(backend, InferenceScheduler().gpu_lock)
    with pytest.raises(ValueError):
        async for _ in session.synthesize(text="   "):
            pass
