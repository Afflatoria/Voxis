"""CosyVoice 3 backend — all CosyVoice-specific logic lives here."""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import numpy as np

from neural_tts.audio.codec import AudioChunk, float32_to_pcm_s16le
from neural_tts.config import Settings
from neural_tts.logging_config import get_logger
from neural_tts.models.base import BackendInfo, StreamingTTS
from neural_tts.voice.schema import COSYVOICE_SUPPORTED_CONTROLS, VoiceConfig

logger = get_logger(__name__)


def _patch_yaml_loader_compat() -> None:
    """hyperpyyaml + ruamel.yaml 0.19+ require Loader.max_depth on the class."""
    try:
        from ruamel.yaml import loader as ruamel_loader

        for cls in (
            ruamel_loader.Loader,
            ruamel_loader.RoundTripLoader,
            ruamel_loader.SafeLoader,
        ):
            if not hasattr(cls, "max_depth"):
                cls.max_depth = None
    except Exception:
        pass


LANGUAGE_INSTRUCT = {
    "english": "Please speak in English.",
    "en": "Please speak in English.",
    "chinese": "请用中文表达。",
    "zh": "请用中文表达。",
    "japanese": "Please speak in Japanese.",
    "ja": "Please speak in Japanese.",
    "korean": "Please speak in Korean.",
    "ko": "Please speak in Korean.",
    "french": "Please speak in French.",
    "german": "Please speak in German.",
    "spanish": "Please speak in Spanish.",
    "italian": "Please speak in Italian.",
    "russian": "Please speak in Russian.",
}


class CosyVoiceBackend(StreamingTTS):
    """StreamingTTS implementation backed by Fun-CosyVoice 3."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._sample_rate = 24000
        self._load_time = 0.0
        self._ready = False
        self._device = "cpu"
        self._model_name = "Fun-CosyVoice3-0.5B"

    def _ensure_cosyvoice_on_path(self) -> None:
        repo = self._settings.cosyvoice_repo.resolve()
        if not repo.exists():
            raise FileNotFoundError(
                f"CosyVoice repository not found at {repo}. "
                "Run: git clone --recursive https://github.com/FunAudioLLM/CosyVoice "
                f"{repo}"
            )
        matcha = repo / "third_party" / "Matcha-TTS"
        for path in (str(repo), str(matcha)):
            if path not in sys.path:
                sys.path.insert(0, path)

    def _validate_assets(self) -> None:
        model_dir = Path(self._settings.cosyvoice_model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"CosyVoice model not found at {model_dir}. "
                "Run: python scripts/download_models.py"
            )
        prompt_wav = Path(self._settings.cosyvoice_prompt_wav)
        if not prompt_wav.exists():
            raise FileNotFoundError(
                f"Reference prompt WAV not found at {prompt_wav}. "
                "Ensure CosyVoice repo is cloned with assets."
            )

    async def load(self) -> None:
        if self._ready:
            return

        started = time.monotonic()
        self._ensure_cosyvoice_on_path()
        self._validate_assets()

        def _load_model():
            import torch

            _patch_yaml_loader_compat()
            from cosyvoice.cli.cosyvoice import AutoModel

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            model = AutoModel(model_dir=str(self._settings.cosyvoice_model_dir))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return model, model.sample_rate, device

        self._model, self._sample_rate, self._device = await asyncio.to_thread(_load_model)
        self._load_time = time.monotonic() - started
        self._ready = True
        logger.info(
            "cosyvoice_loaded",
            model_dir=str(self._settings.cosyvoice_model_dir),
            device=self._device,
            load_time_seconds=self._load_time,
            sample_rate=self._sample_rate,
        )

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="cosyvoice",
            model_name=self._model_name,
            sample_rate=self._sample_rate,
            device=self._device,
            load_time_seconds=self._load_time,
            supported_controls=COSYVOICE_SUPPORTED_CONTROLS,
            ready=self._ready,
        )

    @staticmethod
    def build_instruct(voice: VoiceConfig) -> str:
        """Map VoiceConfig fields to CosyVoice3 instruct2 prompt."""
        parts: list[str] = ["You are a helpful assistant."]

        lang_key = voice.language.strip().lower()
        if lang_key in LANGUAGE_INSTRUCT:
            parts.append(LANGUAGE_INSTRUCT[lang_key])

        if voice.accent:
            strength = ""
            if voice.accent_strength >= 0.7:
                strength = "strong "
            elif voice.accent_strength >= 0.35:
                strength = "moderate "
            parts.append(f"Please speak with a {strength}{voice.accent} accent.")

        if voice.speaking_rate < 0.85:
            parts.append("Please speak slowly.")
        elif voice.speaking_rate > 1.15:
            parts.append("Please speak quickly.")

        if voice.energy < 0.7:
            parts.append("Please speak softly.")
        elif voice.energy > 1.3:
            parts.append("Please speak with high energy.")

        parts.append("<|endofprompt|>")
        return " ".join(parts)

    def _iter_model_outputs(
        self,
        text: str,
        voice: VoiceConfig,
        cancel_event: asyncio.Event | None,
    ) -> Iterator[dict]:
        assert self._model is not None
        instruct = self.build_instruct(voice)
        prompt_wav = str(self._settings.cosyvoice_prompt_wav)
        speed = float(np.clip(voice.speaking_rate, 0.5, 2.0))

        generator = self._model.inference_instruct2(
            text,
            instruct,
            prompt_wav,
            stream=True,
            speed=speed,
        )
        for output in generator:
            if cancel_event and cancel_event.is_set():
                break
            yield output

    async def synthesize(
        self,
        text: str,
        voice: VoiceConfig,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AudioChunk]:
        if not self._ready:
            await self.load()
        assert self._model is not None

        sequence = 0
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _producer() -> None:
            try:
                for output in self._iter_model_outputs(text, voice, cancel_event):
                    loop.call_soon_threadsafe(queue.put_nowait, ("chunk", output))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        producer_task = asyncio.create_task(asyncio.to_thread(_producer))

        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    break

                kind, payload = await queue.get()
                if kind == "done":
                    break
                if kind == "error":
                    raise payload  # type: ignore[misc]

                output = payload
                tensor = output["tts_speech"]
                samples = tensor.squeeze().detach().cpu().numpy().astype(np.float32)
                pcm = float32_to_pcm_s16le(samples)
                duration = len(samples) / self._sample_rate

                yield AudioChunk(
                    sequence=sequence,
                    sample_rate=self._sample_rate,
                    pcm_s16le=pcm,
                    duration_seconds=duration,
                )
                sequence += 1
        finally:
            if cancel_event:
                cancel_event.set()
            await producer_task

    async def aclose(self) -> None:
        self._model = None
        self._ready = False
