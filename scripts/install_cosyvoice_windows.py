#!/usr/bin/env python3
"""Install Windows-compatible CosyVoice inference dependencies."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQ = PROJECT_ROOT / "requirements" / "cosyvoice-windows.txt"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> int:
    if not REQ.exists():
        print(f"Missing {REQ}", file=sys.stderr)
        return 1

    print("Installing minimal CosyVoice inference deps for Windows...")
    print()
    print("NOTE: Do not use third_party/CosyVoice/requirements.txt on Windows.")
    print("      It requires grpcio 1.57 source builds and Linux-only packages.")
    print()

    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    # Install CUDA PyTorch first so openai-whisper does not pull CPU-only torch.
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ]
    )

    # Whisper depends on torch but must not downgrade/replace the CUDA build.
    run([sys.executable, "-m", "pip", "install", "openai-whisper", "--no-deps"])

    run([sys.executable, "-m", "pip", "install", "-r", str(REQ)])

    print("\nDone. Verify with:")
    print('  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"')
    print('  python -c "import onnxruntime, whisper; print(\'ok\')"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
