"""Mathematical invariants for Phase 2 speaker-identity exploration."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from neural_tts.voice.identity_space import (
    SpeakerIdentity,
    cosine_similarity,
    generate_identity_candidates,
    spherical_interpolate,
    tangent_mutation,
)


def _identity(identity_id: str, embedding: list[float]) -> SpeakerIdentity:
    return SpeakerIdentity(
        id=identity_id,
        name=identity_id,
        embedding=embedding,
        converter="test-converter",
    )


def test_identity_normalizes_embedding():
    identity = _identity("a", [3.0, 4.0, 0.0])
    assert np.linalg.norm(identity.embedding) == pytest.approx(1.0)


def test_identity_rejects_invalid_embeddings():
    with pytest.raises(ValidationError):
        _identity("zero", [0.0, 0.0])
    with pytest.raises(ValidationError):
        _identity("nan", [math.nan, 1.0])


def test_slerp_preserves_endpoints_and_unit_norm():
    first = [1.0, 0.0, 0.0]
    second = [0.0, 1.0, 0.0]

    assert spherical_interpolate(first, second, 0.0) == pytest.approx(first)
    assert spherical_interpolate(first, second, 1.0) == pytest.approx(second)
    midpoint = spherical_interpolate(first, second, 0.5)
    assert np.linalg.norm(midpoint) == pytest.approx(1.0)
    assert cosine_similarity(midpoint, first) == pytest.approx(2**-0.5)


def test_tangent_mutation_is_deterministic_and_bounded():
    parent = [1.0, 0.0, 0.0, 0.0]
    first = tangent_mutation(parent, seed=42, strength=0.25)
    second = tangent_mutation(parent, seed=42, strength=0.25)

    assert first == pytest.approx(second)
    assert np.linalg.norm(first) == pytest.approx(1.0)
    expected_similarity = math.cos(0.25 * math.pi / 3)
    assert cosine_similarity(parent, first) == pytest.approx(expected_similarity)


def test_candidate_generation_is_reproducible_and_tracks_parents():
    anchors = [
        _identity("a", [1.0, 0.0, 0.0]),
        _identity("b", [0.0, 1.0, 0.0]),
        _identity("c", [0.0, 0.0, 1.0]),
    ]

    first = generate_identity_candidates(anchors, seed=7, count=4)
    second = generate_identity_candidates(anchors, seed=7, count=4)

    assert first == second
    assert len(first) == 4
    assert all(len(candidate.parent_ids) == 2 for candidate in first)
    assert all(np.linalg.norm(candidate.embedding) == pytest.approx(1.0) for candidate in first)


def test_candidate_generation_requires_compatible_anchors():
    with pytest.raises(ValueError, match="at least two"):
        generate_identity_candidates([_identity("a", [1.0, 0.0])], seed=1)

    with pytest.raises(ValueError, match="equal dimensions"):
        generate_identity_candidates(
            [
                _identity("a", [1.0, 0.0]),
                _identity("b", [1.0, 0.0, 0.0]),
            ],
            seed=1,
        )
