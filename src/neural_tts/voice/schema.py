"""Model-independent voice configuration schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


# Controls that CosyVoice 3 can approximate in M1 via instruct2 + speed.
COSYVOICE_SUPPORTED_CONTROLS = frozenset(
    {
        "language",
        "speaker",
        "accent",
        "accent_strength",
        "speaking_rate",
        "energy",
    }
)

# Reserved for future custom architecture — not implemented in M1 backend.
FUTURE_CONTROLS = frozenset(
    {
        "pitch",
        "warmth",
        "breathiness",
        "roughness",
        "expressiveness",
    }
)


class VoiceConfig(BaseModel):
    """Portable voice representation for all TTS backends."""

    language: str = "English"
    speaker: str | None = "default"

    accent: str | None = None
    accent_strength: float = Field(default=0.0, ge=0.0, le=1.0)

    pitch: float = Field(default=0.0, ge=-1.0, le=1.0)
    speaking_rate: float = Field(default=1.0, ge=0.5, le=2.0)
    energy: float = Field(default=1.0, ge=0.0, le=2.0)

    warmth: float = Field(default=0.0, ge=0.0, le=1.0)
    breathiness: float = Field(default=0.0, ge=0.0, le=1.0)
    roughness: float = Field(default=0.0, ge=0.0, le=1.0)

    expressiveness: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("language")
    @classmethod
    def language_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("language must not be empty")
        return value

    def merge_update(self, partial: dict[str, Any]) -> VoiceConfig:
        """Return a new VoiceConfig with partial fields updated."""
        return self.model_copy(update=partial)

    def unsupported_active_controls(self) -> list[str]:
        """Return future-only controls that differ from neutral defaults."""
        active: list[str] = []
        if self.pitch != 0.0:
            active.append("pitch")
        if self.warmth != 0.0:
            active.append("warmth")
        if self.breathiness != 0.0:
            active.append("breathiness")
        if self.roughness != 0.0:
            active.append("roughness")
        if self.expressiveness != 0.5:
            active.append("expressiveness")
        return active

    def to_public_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data["_supported_by_backend"] = sorted(COSYVOICE_SUPPORTED_CONTROLS)
        data["_unsupported_active"] = self.unsupported_active_controls()
        return data
