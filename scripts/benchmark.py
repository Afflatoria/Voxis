#!/usr/bin/env python3
"""Benchmark streaming TTS latency (TTFA, RTF) using the mock or CosyVoice backend."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neural_tts.config import Settings
from neural_tts.models.factory import create_backend
from neural_tts.streaming.session import StreamingSession
from neural_tts.streaming.scheduler import InferenceScheduler
from neural_tts.voice.schema import VoiceConfig


DEFAULT_TEXT = (
    "Hello, this is a streaming text-to-speech benchmark. "
    "We measure time to first audio and real-time factor."
)


async def run_benchmark(text: str, backend_name: str | None) -> dict:
    settings = Settings()
    if backend_name:
        settings = settings.model_copy(update={"backend": backend_name})  # type: ignore[arg-type]

    backend = create_backend(settings)
    scheduler = InferenceScheduler()

    load_started = time.monotonic()
    await backend.load()
    load_time = time.monotonic() - load_started

    session = StreamingSession(backend, scheduler.gpu_lock)
    voice = VoiceConfig()

    first_chunk_at: float | None = None
    chunk_count = 0
    total_audio = 0.0
    started = time.monotonic()

    async for chunk in session.synthesize(text=text, voice=voice.model_dump()):
        if first_chunk_at is None:
            first_chunk_at = time.monotonic()
        chunk_count += 1
        total_audio += chunk.duration_seconds

    elapsed = time.monotonic() - started
    info = backend.info()
    peak_mb = await backend.peak_gpu_memory_mb()

    ts = session.state.timestamps
    ttfa_ms = None
    if ts.request_received is not None and ts.first_audio_generated is not None:
        ttfa_ms = (ts.first_audio_generated - ts.request_received) * 1000.0
    elif first_chunk_at is not None:
        ttfa_ms = (first_chunk_at - started) * 1000.0

    rtf = elapsed / total_audio if total_audio > 0 else None

    await backend.aclose()

    gpu_name = "n/a"
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return {
        "model": info.model_name,
        "backend": info.name,
        "device": info.device,
        "gpu": gpu_name,
        "text_length": len(text),
        "audio_duration_seconds": total_audio,
        "load_time_seconds": load_time,
        "generation_time_seconds": elapsed,
        "ttfa_ms": ttfa_ms,
        "rtf": rtf,
        "chunk_count": chunk_count,
        "peak_gpu_memory_mb": peak_mb,
    }


def print_report(report: dict) -> None:
    print("=" * 60)
    print("Neural TTS Benchmark")
    print("=" * 60)
    print(f"Model:        {report['model']}")
    print(f"Backend:      {report['backend']}")
    print(f"Device:       {report['device']}")
    print(f"GPU:          {report['gpu']}")
    print(f"Text length:  {report['text_length']} chars")
    print(f"Audio:        {report['audio_duration_seconds']:.2f}s")
    print(f"Model load:   {report['load_time_seconds']:.2f}s")
    print()
    print("TTFA:")
    if report["ttfa_ms"] is not None:
        print(f"  {report['ttfa_ms']:.1f} ms")
    else:
        print("  n/a")
    print()
    print("RTF:")
    if report["rtf"] is not None:
        print(f"  {report['rtf']:.3f}")
    else:
        print("  n/a")
    print()
    print("Chunks:")
    print(f"  count: {report['chunk_count']}")
    if report["peak_gpu_memory_mb"] is not None:
        print(f"Peak GPU memory: {report['peak_gpu_memory_mb']:.1f} MB")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument(
        "--backend",
        choices=["mock", "cosyvoice"],
        default=os.environ.get("NEURAL_TTS_BACKEND", "mock"),
    )
    args = parser.parse_args()

    report = asyncio.run(run_benchmark(args.text, args.backend))
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
