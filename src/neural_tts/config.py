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

    backend: Literal["cosyvoice", "mock"] = Field(
        default="cosyvoice", validation_alias="NEURAL_TTS_BACKEND"
    )
    host: str = Field(default="0.0.0.0", validation_alias="NEURAL_TTS_HOST")
    port: int = Field(default=8000, validation_alias="NEURAL_TTS_PORT")
    log_level: str = Field(default="INFO", validation_alias="NEURAL_TTS_LOG_LEVEL")

    cosyvoice_model_dir: Path = Field(
        default=PROJECT_ROOT / "pretrained_models" / "Fun-CosyVoice3-0.5B",
        validation_alias="COSYVOICE_MODEL_DIR",
    )
    cosyvoice_repo: Path = Field(
        default=PROJECT_ROOT / "third_party" / "CosyVoice",
        validation_alias="COSYVOICE_REPO",
    )
    cosyvoice_prompt_wav: Path = Field(
        default=PROJECT_ROOT / "third_party" / "CosyVoice" / "asset" / "zero_shot_prompt.wav",
        validation_alias="COSYVOICE_PROMPT_WAV",
    )
    cosyvoice_prompt_text: str = Field(
        default="You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
        validation_alias="COSYVOICE_PROMPT_TEXT",
    )

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
