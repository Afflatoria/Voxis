"""Tests for VoiceConfig."""

import pytest
from pydantic import ValidationError

from neural_tts.voice.schema import VoiceConfig


def test_defaults():
    voice = VoiceConfig()
    assert voice.language == "English"
    assert voice.speaking_rate == 1.0
    assert voice.pitch == 0.0


def test_validation_empty_language():
    with pytest.raises(ValidationError):
        VoiceConfig(language="  ")


def test_serialization_roundtrip():
    voice = VoiceConfig(accent="Scottish", accent_strength=0.6, speaking_rate=1.2)
    data = voice.model_dump()
    restored = VoiceConfig.model_validate(data)
    assert restored == voice


def test_merge_update():
    voice = VoiceConfig()
    updated = voice.merge_update({"speaking_rate": 1.5, "energy": 1.2})
    assert updated.speaking_rate == 1.5
    assert updated.energy == 1.2
    assert voice.speaking_rate == 1.0


def test_unsupported_active_controls():
    voice = VoiceConfig(pitch=0.3, warmth=0.8)
    active = voice.unsupported_active_controls()
    assert "pitch" in active
    assert "warmth" in active
    assert "speaking_rate" not in active
