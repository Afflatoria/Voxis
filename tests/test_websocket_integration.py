"""WebSocket integration test using mock backend."""

import json

import pytest
from fastapi.testclient import TestClient

from neural_tts.audio.codec import decode_audio_frame
from neural_tts.main import create_app


@pytest.fixture
def mock_client(monkeypatch):
    monkeypatch.setenv("NEURAL_TTS_BACKEND", "mock")
    from neural_tts.config import get_settings

    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:
        yield client
    get_settings.cache_clear()


def _receive_json(ws):
    return json.loads(ws.receive_text())


def test_websocket_stream_mock(mock_client):
    with mock_client.websocket_connect("/v1/stream") as ws:
        ready = _receive_json(ws)
        assert ready["type"] == "ready"
        assert ready["backend"]["name"] == "mock"

        ws.send_text(
            json.dumps(
                {
                    "type": "start",
                    "text": "Hello",
                    "voice": {"language": "English"},
                }
            )
        )

        started = _receive_json(ws)
        assert started["type"] == "started"

        got_audio = False
        for _ in range(50):
            message = ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes"):
                seq, sr, pcm = decode_audio_frame(message["bytes"])
                assert sr == 24000
                assert len(pcm) > 0
                got_audio = True
            elif message.get("text"):
                data = json.loads(message["text"])
                if data["type"] == "complete":
                    assert data["metrics"]["chunk_count"] >= 1
                    break
                if data["type"] == "error":
                    pytest.fail(data["message"])
        else:
            pytest.fail("timed out waiting for completion")

        assert got_audio


def test_websocket_malformed_message(mock_client):
    with mock_client.websocket_connect("/v1/stream") as ws:
        ws.receive_text()
        ws.send_text("{not json")
        err = _receive_json(ws)
        assert err["type"] == "error"
        assert err["code"] == "malformed_message"


def test_websocket_empty_text(mock_client):
    with mock_client.websocket_connect("/v1/stream") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "start", "text": "  "}))
        err = _receive_json(ws)
        assert err["type"] == "error"
        assert err["code"] == "empty_text"


def test_websocket_accepts_midstream_voice_update(mock_client):
    with mock_client.websocket_connect("/v1/stream") as ws:
        ws.receive_text()
        ws.send_text(
            json.dumps(
                {
                    "type": "start",
                    "text": "This is a sufficiently long sentence to keep streaming.",
                }
            )
        )
        assert _receive_json(ws)["type"] == "started"

        ws.send_text(
            json.dumps(
                {
                    "type": "voice_update",
                    "voice": {"speaking_rate": 1.25, "energy": 0.8},
                }
            )
        )

        for _ in range(20):
            message = ws.receive()
            if message.get("text"):
                event = json.loads(message["text"])
                if event["type"] == "voice_updated":
                    assert event["voice_version"] == 1
                    assert event["voice"]["speaking_rate"] == 1.25
                    break
        else:
            pytest.fail("voice_updated event was not received")
