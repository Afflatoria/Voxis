"""Tests for deterministic Phase 1 voice-genome generation."""

import pytest
from fastapi.testclient import TestClient

from neural_tts.main import create_app
from neural_tts.voice.genome import (
    CandidateRequest,
    VoiceGenome,
    generate_candidates,
)


@pytest.fixture
def mock_client(monkeypatch):
    monkeypatch.setenv("NEURAL_TTS_BACKEND", "mock")
    from neural_tts.config import get_settings

    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:
        yield client
    get_settings.cache_clear()


def test_candidate_generation_is_reproducible():
    request = CandidateRequest(seed=1234, count=2)
    assert generate_candidates(request) == generate_candidates(request)


def test_mutations_remain_near_parent_and_inside_bounds():
    parent = VoiceGenome(
        id="parent",
        name="Parent",
        seed=1,
        speaking_rate=1.0,
        energy=1.0,
        warmth=0.2,
        brightness=-0.1,
        presence=0.3,
    )
    candidates = generate_candidates(
        CandidateRequest(
            seed=42,
            count=4,
            mutation_strength=0.1,
            parent=parent,
        )
    )

    assert len(candidates) == 4
    assert all(0.75 <= candidate.speaking_rate <= 1.3 for candidate in candidates)
    assert all(-1 <= candidate.warmth <= 1 for candidate in candidates)
    assert all(abs(candidate.warmth - parent.warmth) < 0.4 for candidate in candidates)


def test_candidate_api(mock_client):
    response = mock_client.post(
        "/v1/voice-genomes/candidates",
        json={"seed": 99, "count": 2, "mutation_strength": 0.25},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 2
    assert payload[0]["id"].startswith("vg-")
