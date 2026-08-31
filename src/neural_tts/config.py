"""Application settings loaded from environment and config files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    backend: Literal["f5tts", "mock"] = Field(
        default="f5tts", validation_alias="NEURAL_TTS_BACKEND"
    )
    host: str = Field(default="0.0.0.0", validation_alias="NEURAL_TTS_HOST")
    port: int = Field(default=8000, validation_alias="NEURAL_TTS_PORT")
    log_level: str = Field(default="INFO", validation_alias="NEURAL_TTS_LOG_LEVEL")

    f5tts_model: str = Field(default="F5TTS_v1_Base", validation_alias="F5TTS_MODEL")
    f5tts_ref_wav: Path = Field(
        default=PROJECT_ROOT / "assets" / "voices" / "reference.wav",
        validation_alias="F5TTS_REF_WAV",
    )
    f5tts_ref_text: str = Field(
        default="",
        validation_alias="F5TTS_REF_TEXT",
    )
    f5tts_nfe_step: int = Field(default=32, validation_alias="F5TTS_NFE_STEP")
    f5tts_cfg_strength: float = Field(default=2.0, validation_alias="F5TTS_CFG_STRENGTH")
    f5tts_stream_chunk_size: int = Field(default=2048, validation_alias="F5TTS_STREAM_CHUNK_SIZE")

    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        validation_alias="NEURAL_TTS_CORS_ORIGINS",
    )

    max_concurrent_sessions: int = 1

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
