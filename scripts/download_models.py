#!/usr/bin/env python3
"""Download Fun-CosyVoice3 model weights and clone CosyVoice if needed."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "pretrained_models" / "Fun-CosyVoice3-0.5B"
DEFAULT_REPO = PROJECT_ROOT / "third_party" / "CosyVoice"
MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

# Minimum files required before CosyVoice AutoModel can load.
REQUIRED_MODEL_FILES = (
    "cosyvoice3.yaml",
    "llm.pt",
    "flow.pt",
    "hift.pt",
    "campplus.onnx",
    "speech_tokenizer_v3.onnx",
    "spk2info.pt",
)


def model_is_complete(model_dir: Path) -> bool:
    return all((model_dir / name).exists() for name in REQUIRED_MODEL_FILES)


def clone_cosyvoice(repo: Path) -> None:
    if repo.exists():
        print(f"CosyVoice repo already exists: {repo}")
        return
    repo.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning CosyVoice into {repo} ...")
    subprocess.check_call(
        [
            "git",
            "clone",
            "--recursive",
            "https://github.com/FunAudioLLM/CosyVoice.git",
            str(repo),
        ]
    )


def download_model(model_dir: Path) -> None:
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    if model_is_complete(model_dir):
        print(f"Model already complete: {model_dir}")
        return
    if model_dir.exists() and any(model_dir.iterdir()):
        missing = [n for n in REQUIRED_MODEL_FILES if not (model_dir / n).exists()]
        print(f"Resuming incomplete download ({len(missing)} files missing): {missing}")

    print(f"Downloading {MODEL_ID} to {model_dir} ...")

    hf_error: Exception | None = None
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(MODEL_ID, local_dir=str(model_dir))
        print("Download complete via huggingface_hub.")
        return
    except ImportError:
        hf_error = None
    except Exception as exc:
        hf_error = exc
        print(f"huggingface_hub download failed: {exc}")

    try:
        from modelscope import snapshot_download as ms_download
    except ImportError as exc:
        raise SystemExit(
            "No model download library available.\n"
            "Install one of:\n"
            "  python -m pip install huggingface_hub\n"
            "  python -m pip install modelscope\n"
            "Or install project extras:\n"
            "  python -m pip install -e \".[download]\""
        ) from exc

    if hf_error is not None:
        print("Retrying via modelscope ...")
    ms_download(MODEL_ID, local_dir=str(model_dir))
    print("Download complete via modelscope.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download CosyVoice3 assets")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--skip-clone", action="store_true")
    args = parser.parse_args()

    if not args.skip_clone:
        clone_cosyvoice(args.repo)

    download_model(args.model_dir)

    prompt = args.repo / "asset" / "zero_shot_prompt.wav"
    if not prompt.exists():
        print(f"WARNING: reference prompt not found at {prompt}")
        print("Ensure CosyVoice was cloned with --recursive.")
        return 1

    print("\nSetup complete.")
    print(f"  Model: {args.model_dir}")
    print(f"  Repo:  {args.repo}")
    print(f"  Prompt WAV: {prompt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
