"""Tests for WebSocket protocol parsing."""

import pytest

from neural_tts.api.protocol import (
    StartMessage,
    StopMessage,
    VoiceUpdateMessage,
    parse_client_message,
)


def test_parse_start():
    msg = parse_client_message(
        {"type": "start", "text": "hello", "voice": {"language": "English"}}
    )
    assert isinstance(msg, StartMessage)
    assert msg.text == "hello"


def test_parse_stop():
    msg = parse_client_message({"type": "stop"})
    assert isinstance(msg, StopMessage)


def test_parse_voice_update():
    msg = parse_client_message({"type": "voice_update", "voice": {"pitch": 0.2}})
    assert isinstance(msg, VoiceUpdateMessage)
    assert msg.voice["pitch"] == 0.2


def test_malformed_type():
    with pytest.raises(ValueError):
        parse_client_message({"type": "unknown"})
