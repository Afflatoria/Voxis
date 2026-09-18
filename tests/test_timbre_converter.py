"""Contract tests for the Phase 2 timbre-conversion boundary."""

import asyncio

import numpy as np
import pytest

from neural_tts.timbre.base import AudioBuffer
from neural_tts.timbre.mock import MockTimbreConverter


def _tone(frequency: float = 220, seconds: float = 0.2) -> AudioBuffer:
    sample_rate = 24000
    time = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    samples = 0.25 * np.sin(2 * np.pi * frequency * time)
    return AudioBuffer(samples, sample_rate)


def test_audio_buffer_normalizes_shape_and_protects_samples():
    audio = AudioBuffer(np.ones((1, 100), dtype=np.float32) * 2, 16000)
    assert audio.samples.shape == (100,)
    assert float(audio.samples.max()) == 1.0
    assert not audio.samples.flags.writeable
    assert audio.duration_seconds == pytest.approx(100 / 16000)


@pytest.mark.asyncio
async def test_mock_converter_extracts_deterministic_normalized_identity():
    converter = MockTimbreConverter()
    reference = _tone()

    first = await converter.extract_identity(
        reference, identity_id="target", name="Target"
    )
    second = await converter.extract_identity(
        reference, identity_id="target", name="Target"
    )

    assert first == second
    assert len(first.embedding) == converter.EMBEDDING_DIMENSIONS
    assert np.linalg.norm(first.embedding) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_mock_conversion_obeys_contract():
    converter = MockTimbreConverter()
    source = _tone()
    target = await converter.extract_identity(
        _tone(330), identity_id="target", name="Target"
    )

    result = await converter.convert(source, target=target)

    assert result.audio.sample_rate == source.sample_rate
    assert result.audio.samples == pytest.approx(source.samples)
    assert result.rtf >= 0
    assert result.metadata["target_identity"] == "target"


@pytest.mark.asyncio
async def test_mock_conversion_honors_preexisting_cancellation():
    converter = MockTimbreConverter()
    target = await converter.extract_identity(
        _tone(), identity_id="target", name="Target"
    )
    cancel_event = asyncio.Event()
    cancel_event.set()

    with pytest.raises(asyncio.CancelledError):
        await converter.convert(_tone(), target=target, cancel_event=cancel_event)
