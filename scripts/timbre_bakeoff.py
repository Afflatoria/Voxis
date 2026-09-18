"""Offline Phase 2 timbre-converter evaluation harness."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neural_tts.timbre.base import AudioBuffer, TimbreConverter  # noqa: E402
from neural_tts.timbre.mock import MockTimbreConverter  # noqa: E402


def read_audio(path: Path) -> AudioBuffer:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = np.mean(samples, axis=1)
    return AudioBuffer(samples, sample_rate)


def create_converter(engine: str) -> TimbreConverter:
    if engine == "mock":
        return MockTimbreConverter()
    raise ValueError(f"unsupported converter engine: {engine}")


async def run(args: argparse.Namespace) -> dict[str, object]:
    converter = create_converter(args.engine)
    await converter.load()
    try:
        source = read_audio(args.source)
        reference = read_audio(args.reference)
        identity = await converter.extract_identity(
            reference,
            identity_id=args.identity_id,
            name=args.identity_name,
        )
        result = await converter.convert(source, target=identity)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            args.output,
            result.audio.samples,
            result.audio.sample_rate,
            subtype="PCM_16",
        )
        info = converter.info()
        return {
            "engine": info.name,
            "model": info.model,
            "device": info.device,
            "supports_streaming": info.supports_streaming,
            "embedding_dimensions": len(identity.embedding),
            "source_duration_seconds": source.duration_seconds,
            "output_duration_seconds": result.audio.duration_seconds,
            "conversion_seconds": result.conversion_seconds,
            "rtf": result.rtf,
            "target_identity": identity.id,
            "output": str(args.output),
            "metadata": result.metadata,
        }
    finally:
        await converter.aclose()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract an identity, convert one WAV, and report performance."
    )
    parser.add_argument("--engine", choices=["mock"], default="mock")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--identity-id", default="bakeoff-target")
    parser.add_argument("--identity-name", default="Bakeoff target")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = asyncio.run(run(args))
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
