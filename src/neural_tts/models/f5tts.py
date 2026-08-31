"""F5-TTS backend — all F5-TTS-specific logic lives here."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import numpy as np
import torchaudio

from neural_tts.audio.codec import AudioChunk, float32_to_pcm_s16le
from neural_tts.config import Settings
from neural_tts.logging_config import get_logger
from neural_tts.models.base import BackendInfo, StreamingTTS
from neural_tts.voice.schema import F5TTS_SUPPORTED_CONTROLS, VoiceConfig

logger = get_logger(__name__)


def energy_to_target_rms(energy: float) -> float:
    """Map VoiceConfig energy (0–2, neutral=1) to F5-TTS target_rms."""
    return float(np.clip(0.1 * energy, 0.05, 0.2))


def build_text_batches(ref_audio_samples: int, sample_rate: int, ref_text: str, gen_text: str, speed: float) -> list[str]:
    """Split generation text the same way as f5_tts.infer.utils_infer.infer_process."""
    from f5_tts.infer.utils_infer import chunk_text

    duration_sec = ref_audio_samples / sample_rate
    max_chars = int(len(ref_text.encode("utf-8")) / duration_sec * (22 - duration_sec) * speed)
    batches = chunk_text(gen_text.strip(), max_chars=max_chars)
    return batches if batches else [gen_text.strip()] if gen_text.strip() else []


class F5TTSBackend(StreamingTTS):
    """StreamingTTS implementation backed by F5-TTS."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tts = None
        self._sample_rate = 24000
        self._load_time = 0.0
        self._ready = False
        self._device = "cpu"
        self._model_name = settings.f5tts_model

    def _validate_assets(self) -> None:
        ref_wav = Path(self._settings.f5tts_ref_wav)
        if not ref_wav.exists():
            raise FileNotFoundError(
                f"Reference voice WAV not found at {ref_wav}. "
                "Set F5TTS_REF_WAV in .env to a short reference clip (~12s max)."
            )

    async def load(self) -> None:
        if self._ready:
            return

        started = time.monotonic()
        self._validate_assets()

        def _load_model():
            import torch
            from f5_tts.api import F5TTS

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            tts = F5TTS(model=self._settings.f5tts_model)
            return tts, tts.target_sample_rate, tts.device

        self._tts, self._sample_rate, self._device = await asyncio.to_thread(_load_model)
        self._load_time = time.monotonic() - started
        self._ready = True
        logger.info(
            "f5tts_loaded",
            model=self._model_name,
            ref_wav=str(self._settings.f5tts_ref_wav),
            device=self._device,
            load_time_seconds=self._load_time,
            sample_rate=self._sample_rate,
        )

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="f5tts",
            model_name=self._model_name,
            sample_rate=self._sample_rate,
            device=self._device,
            load_time_seconds=self._load_time,
            supported_controls=F5TTS_SUPPORTED_CONTROLS,
            ready=self._ready,
        )

    def _iter_audio_chunks(
        self,
        text: str,
        voice: VoiceConfig,
        cancel_event: asyncio.Event | None,
    ) -> Iterator[tuple[np.ndarray, int]]:
        assert self._tts is not None
        from f5_tts.infer.utils_infer import infer_batch_process, preprocess_ref_audio_text

        speed = float(np.clip(voice.speaking_rate, 0.5, 2.0))
        target_rms = energy_to_target_rms(voice.energy)
        ref_file = str(self._settings.f5tts_ref_wav)
        ref_text = self._settings.f5tts_ref_text.strip()

        ref_audio_path, ref_text = preprocess_ref_audio_text(
            ref_file,
            ref_text,
            show_info=lambda *_args, **_kwargs: None,
        )
        audio, sr = torchaudio.load(ref_audio_path)
        batches = build_text_batches(audio.shape[-1], sr, ref_text, text, speed)
        if not batches:
            return

        logger.info(
            "f5tts_synthesize",
            text_preview=text[:80],
            batches=len(batches),
            speed=speed,
            target_rms=target_rms,
        )

        for chunk, rate in infer_batch_process(
            (audio, sr),
            ref_text,
            batches,
            self._tts.ema_model,
            self._tts.vocoder,
            mel_spec_type=self._tts.mel_spec_type,
            device=self._tts.device,
            target_rms=target_rms,
            speed=speed,
            nfe_step=self._settings.f5tts_nfe_step,
            cfg_strength=self._settings.f5tts_cfg_strength,
            streaming=True,
            chunk_size=self._settings.f5tts_stream_chunk_size,
            progress=None,
        ):
            if cancel_event and cancel_event.is_set():
                break
            yield chunk, rate

    async def synthesize(
        self,
        text: str,
        voice: VoiceConfig,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AudioChunk]:
        if not self._ready:
            await self.load()
        assert self._tts is not None

        sequence = 0
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _producer() -> None:
            try:
                for samples, sample_rate in self._iter_audio_chunks(text, voice, cancel_event):
                    loop.call_soon_threadsafe(queue.put_nowait, ("chunk", (samples, sample_rate)))
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

                samples, sample_rate = payload
                pcm = float32_to_pcm_s16le(np.asarray(samples, dtype=np.float32))
                duration = len(samples) / sample_rate

                yield AudioChunk(
                    sequence=sequence,
                    sample_rate=sample_rate,
                    pcm_s16le=pcm,
                    duration_seconds=duration,
                )
                sequence += 1
        finally:
            await producer_task

    async def aclose(self) -> None:
        self._tts = None
        self._ready = False
