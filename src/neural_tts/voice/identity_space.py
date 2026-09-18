"""Geometry for exploring neural speaker-identity embedding spaces."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, Field, field_validator


class SpeakerIdentity(BaseModel):
    """A normalized speaker embedding plus reproducibility metadata."""

    id: str
    name: str
    embedding: list[float] = Field(min_length=2, max_length=4096)
    seed: int | None = None
    parent_ids: list[str] = Field(default_factory=list)
    converter: str | None = None

    @field_validator("embedding")
    @classmethod
    def normalize_and_validate_embedding(cls, values: list[float]) -> list[float]:
        vector = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(vector)):
            raise ValueError("embedding values must be finite")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("embedding must have non-zero magnitude")
        return (vector / norm).astype(np.float32).tolist()


def _unit_vector(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size < 2:
        raise ValueError("embedding must be a one-dimensional vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("embedding values must be finite")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("embedding must have non-zero magnitude")
    return vector / norm


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return cosine similarity after validating and normalizing both vectors."""
    first = _unit_vector(a)
    second = _unit_vector(b)
    if first.shape != second.shape:
        raise ValueError("embeddings must have equal dimensions")
    return float(np.clip(np.dot(first, second), -1.0, 1.0))


def spherical_interpolate(
    a: Sequence[float],
    b: Sequence[float],
    amount: float,
) -> np.ndarray:
    """Interpolate along the unit hypersphere between two speaker anchors."""
    if not 0.0 <= amount <= 1.0:
        raise ValueError("amount must be between 0 and 1")
    first = _unit_vector(a)
    second = _unit_vector(b)
    if first.shape != second.shape:
        raise ValueError("embeddings must have equal dimensions")

    dot = float(np.clip(np.dot(first, second), -1.0, 1.0))
    if dot > 0.9995:
        return _unit_vector((1.0 - amount) * first + amount * second)
    if dot < -0.9995:
        # Antipodal points have infinitely many valid arcs. Pick a stable
        # orthogonal direction so seeded experiments remain reproducible.
        axis = np.zeros_like(first)
        axis[int(np.argmin(np.abs(first)))] = 1.0
        tangent = _unit_vector(axis - np.dot(axis, first) * first)
        return (
            math.cos(math.pi * amount) * first
            + math.sin(math.pi * amount) * tangent
        )

    angle = math.acos(dot)
    scale = math.sin(angle)
    result = (
        math.sin((1.0 - amount) * angle) / scale * first
        + math.sin(amount * angle) / scale * second
    )
    return _unit_vector(result)


def tangent_mutation(
    parent: Sequence[float],
    *,
    seed: int,
    strength: float,
) -> np.ndarray:
    """Move a bounded angular distance from a parent on the unit sphere."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    origin = _unit_vector(parent)
    if strength == 0:
        return origin.copy()

    rng = np.random.default_rng(seed)
    noise = rng.normal(size=origin.shape)
    tangent = noise - np.dot(noise, origin) * origin
    tangent = _unit_vector(tangent)

    # Strength 1 means at most a 60-degree identity move, avoiding the
    # unrelated or unstable regions caused by unrestricted Gaussian noise.
    angle = strength * (math.pi / 3.0)
    return _unit_vector(math.cos(angle) * origin + math.sin(angle) * tangent)


def generate_identity_candidates(
    anchors: Sequence[SpeakerIdentity],
    *,
    seed: int,
    count: int = 2,
    mutation_strength: float = 0.15,
) -> list[SpeakerIdentity]:
    """Create deterministic candidates between and near valid anchor voices."""
    if len(anchors) < 2:
        raise ValueError("at least two anchor identities are required")
    if count < 1 or count > 32:
        raise ValueError("count must be between 1 and 32")

    dimensions = {len(anchor.embedding) for anchor in anchors}
    if len(dimensions) != 1:
        raise ValueError("all anchor embeddings must have equal dimensions")

    rng = np.random.default_rng(seed)
    candidates: list[SpeakerIdentity] = []
    for index in range(count):
        parent_indexes = rng.choice(len(anchors), size=2, replace=False)
        first = anchors[int(parent_indexes[0])]
        second = anchors[int(parent_indexes[1])]
        amount = float(rng.uniform(0.2, 0.8))
        base = spherical_interpolate(first.embedding, second.embedding, amount)
        mutation_seed = int(rng.integers(0, 2**31 - 1))
        embedding = tangent_mutation(
            base,
            seed=mutation_seed,
            strength=mutation_strength,
        )
        candidates.append(
            SpeakerIdentity(
                id=f"identity-{seed:x}-{index:02d}",
                name=f"Identity {index + 1}",
                embedding=embedding.tolist(),
                seed=mutation_seed,
                parent_ids=[first.id, second.id],
                converter=first.converter
                if first.converter == second.converter
                else None,
            )
        )
    return candidates
