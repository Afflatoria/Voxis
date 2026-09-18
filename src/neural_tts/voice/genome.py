"""Reproducible Phase 1 artificial voice identity parameters."""

from __future__ import annotations

import random

from pydantic import BaseModel, Field


class VoiceGenome(BaseModel):
    """A stable, serializable identity and delivery profile."""

    id: str
    name: str
    seed: int
    speaking_rate: float = Field(default=1.0, ge=0.75, le=1.3)
    pitch_semitones: float = Field(default=0.0, ge=-6.0, le=6.0)
    energy: float = Field(default=1.0, ge=0.6, le=1.4)
    warmth: float = Field(default=0.0, ge=-1.0, le=1.0)
    brightness: float = Field(default=0.0, ge=-1.0, le=1.0)
    presence: float = Field(default=0.0, ge=-1.0, le=1.0)


class CandidateRequest(BaseModel):
    seed: int = 1
    count: int = Field(default=2, ge=2, le=8)
    mutation_strength: float = Field(default=0.25, ge=0.02, le=0.75)
    parent: VoiceGenome | None = None


def _clamp(value: float, lower: float, upper: float) -> float:
    return round(min(upper, max(lower, value)), 3)


def generate_candidates(request: CandidateRequest) -> list[VoiceGenome]:
    """Generate deterministic candidates, optionally around a selected parent."""
    rng = random.Random(request.seed)
    candidates: list[VoiceGenome] = []

    for _ in range(request.count):
        candidate_seed = rng.getrandbits(31)
        if request.parent is None:
            rate = rng.uniform(0.84, 1.16)
            pitch = rng.uniform(-3.5, 3.5)
            energy = rng.uniform(0.78, 1.22)
            warmth = rng.uniform(-0.8, 0.8)
            brightness = rng.uniform(-0.8, 0.8)
            presence = rng.uniform(-0.7, 0.7)
        else:
            parent = request.parent
            strength = request.mutation_strength
            rate = parent.speaking_rate + rng.gauss(0, 0.16 * strength)
            pitch = parent.pitch_semitones + rng.gauss(0, 3.0 * strength)
            energy = parent.energy + rng.gauss(0, 0.2 * strength)
            warmth = parent.warmth + rng.gauss(0, strength)
            brightness = parent.brightness + rng.gauss(0, strength)
            presence = parent.presence + rng.gauss(0, strength)

        candidates.append(
            VoiceGenome(
                id=f"vg-{candidate_seed:08x}",
                name=f"Voice {candidate_seed & 0xFFFF:04X}",
                seed=candidate_seed,
                speaking_rate=_clamp(rate, 0.75, 1.3),
                pitch_semitones=_clamp(pitch, -6.0, 6.0),
                energy=_clamp(energy, 0.6, 1.4),
                warmth=_clamp(warmth, -1.0, 1.0),
                brightness=_clamp(brightness, -1.0, 1.0),
                presence=_clamp(presence, -1.0, 1.0),
            )
        )

    return candidates
