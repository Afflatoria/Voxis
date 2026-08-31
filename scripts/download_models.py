#!/usr/bin/env python3
"""Prefetch F5-TTS model weights from Hugging Face into the local cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def prefetch(model: str = "F5TTS_v1_Base") -> None:
    from f5_tts.api import F5TTS

    print(f"Prefetching F5-TTS weights for {model} ...")
    tts = F5TTS(model=model)
    print(f"Ready on {tts.device} (sample_rate={tts.target_sample_rate})")
    print("Weights are cached under your Hugging Face cache directory.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download/cache F5-TTS model weights.")
    parser.add_argument(
        "--model",
        default="F5TTS_v1_Base",
        help="F5-TTS model variant (default: F5TTS_v1_Base).",
    )
    args = parser.parse_args()
    prefetch(args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
