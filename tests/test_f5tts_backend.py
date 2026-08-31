"""Tests for F5-TTS backend helpers."""

import numpy as np

from neural_tts.models.f5tts import build_text_batches, energy_to_target_rms


def test_energy_to_target_rms_neutral():
    assert energy_to_target_rms(1.0) == 0.1


def test_energy_to_target_rms_clamps_low():
    assert energy_to_target_rms(0.0) == 0.05


def test_energy_to_target_rms_clamps_high():
    assert energy_to_target_rms(5.0) == 0.2


def test_build_text_batches_short_text():
    batches = build_text_batches(
        ref_audio_samples=24000,
        sample_rate=24000,
        ref_text="Hello world.",
        gen_text="Short test.",
        speed=1.0,
    )
    assert batches == ["Short test."]


def test_build_text_batches_splits_long_text():
    long_text = ". ".join(f"Sentence number {i} for batching" for i in range(80))
    batches = build_text_batches(
        ref_audio_samples=240_000,
        sample_rate=24000,
        ref_text="Short reference transcript for batching.",
        gen_text=long_text,
        speed=1.0,
    )
    assert len(batches) > 1
    assert all(batch.strip() for batch in batches)
