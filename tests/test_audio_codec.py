"""Tests for audio codec."""

import numpy as np

from neural_tts.audio.codec import (
    AudioChunk,
    decode_audio_frame,
    encode_audio_frame,
    float32_to_pcm_s16le,
    pcm_s16le_to_float32,
)


def test_pcm_roundtrip():
    samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    pcm = float32_to_pcm_s16le(samples)
    restored = pcm_s16le_to_float32(pcm)
    assert restored.shape == samples.shape
    assert np.allclose(restored, samples, atol=0.002)


def test_encode_decode_frame():
    pcm = float32_to_pcm_s16le(np.zeros(480, dtype=np.float32))
    chunk = AudioChunk(sequence=3, sample_rate=24000, pcm_s16le=pcm, duration_seconds=0.02)
    frame = encode_audio_frame(chunk)
    seq, sr, payload = decode_audio_frame(frame)
    assert seq == 3
    assert sr == 24000
    assert payload == pcm
