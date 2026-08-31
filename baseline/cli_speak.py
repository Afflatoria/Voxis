#!/usr/bin/env python3
"""Interactive CLI baseline tester for F5-TTS — no frontend required.

Type text, hear WAV output, inspect saved files under baseline/outputs/.

Examples:
  python baseline/cli_speak.py
  python baseline/cli_speak.py --text "Hello, this is a test."
  python baseline/cli_speak.py --speed 1.1 --no-play
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neural_tts.config import get_settings
from neural_tts.models.f5tts import F5TTSBackend
from neural_tts.voice.schema import VoiceConfig


def play_wav(path: Path) -> None:
    """Play a WAV file using the OS default mechanism."""
    path = path.resolve()
    if sys.platform == "win32":
        import winsound

        print(f"Playing {path.name} ...")
        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        return

    for cmd in (["ffplay", "-nodisp", "-autoexit", str(path)], ["aplay", str(path)]):
        try:
            import subprocess

            print(f"Playing via {cmd[0]} ...")
            subprocess.run(cmd, check=True)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    print(f"No audio player found. Open manually: {path}")


def save_wav(path: Path, samples: np.ndarray, sample_rate: int) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate)
    duration = len(samples) / sample_rate
    print(f"Saved {path} ({duration:.2f}s, {sample_rate} Hz)")
    return duration


class BaselineSpeaker:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.backend = F5TTSBackend(self.settings)
        self.sample_rate = 24000

    async def load(self) -> None:
        print("Loading F5-TTS model (first run downloads weights from Hugging Face) ...")
        started = time.monotonic()
        await self.backend.load()
        info = self.backend.info()
        self.sample_rate = info.sample_rate
        print(
            f"Ready in {time.monotonic() - started:.1f}s "
            f"(model={info.model_name}, sample_rate={info.sample_rate}, device={info.device})"
        )
        print(f"Reference WAV: {self.settings.f5tts_ref_wav}")

    async def synthesize(self, text: str, *, speed: float, energy: float) -> np.ndarray:
        voice = VoiceConfig(speaking_rate=speed, energy=energy)
        chunks: list[np.ndarray] = []
        started = time.monotonic()

        async for chunk in self.backend.synthesize(text, voice):
            samples = np.frombuffer(chunk.pcm_s16le, dtype=np.int16).astype(np.float32) / 32768.0
            chunks.append(samples)

        if not chunks:
            raise RuntimeError("Backend returned no audio chunks.")

        audio = np.concatenate(chunks)
        elapsed = time.monotonic() - started
        duration = len(audio) / self.sample_rate
        rtf = elapsed / duration if duration > 0 else 0.0
        print(
            f"Generated {len(chunks)} chunk(s), {duration:.2f}s audio, "
            f"RTF={rtf:.2f}, wall={elapsed:.1f}s"
        )
        return audio


def timestamped_path(prefix: str = "speech") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"{prefix}_{stamp}.wav"


async def run_once(
    speaker: BaselineSpeaker,
    text: str,
    *,
    speed: float,
    energy: float,
    play: bool,
    out_path: Path | None,
) -> Path:
    audio = await speaker.synthesize(text, speed=speed, energy=energy)
    path = out_path or timestamped_path()
    save_wav(path, audio, speaker.sample_rate)
    latest = OUTPUT_DIR / "last.wav"
    save_wav(latest, audio, speaker.sample_rate)
    if play:
        play_wav(path)
    return path


async def interactive_loop(
    speaker: BaselineSpeaker,
    *,
    speed: float,
    energy: float,
    play: bool,
) -> None:
    print()
    print("Baseline CLI TTS (F5-TTS)")
    print("  Enter text and press Enter to synthesize.")
    print("  Commands: :quit  :help  :speed <0.5-2.0>  :energy <0.5-2.0>  :play on|off")
    print(f"  Output folder: {OUTPUT_DIR}")
    print()

    current_speed = speed
    current_energy = energy
    current_play = play

    while True:
        try:
            line = input("tts> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.startswith(":"):
            parts = line[1:].split()
            cmd = parts[0].lower()
            arg = parts[1].lower() if len(parts) > 1 else ""

            if cmd in ("quit", "exit", "q"):
                break
            if cmd == "help":
                print("Commands: :quit :speed :energy :play")
                continue
            if cmd == "speed" and arg:
                current_speed = float(arg)
                print(f"speed={current_speed}")
                continue
            if cmd == "energy" and arg:
                current_energy = float(arg)
                print(f"energy={current_energy}")
                continue
            if cmd == "play" and arg in ("on", "off"):
                current_play = arg == "on"
                print(f"play={current_play}")
                continue

            print("Unknown command. Type :help")
            continue

        await run_once(
            speaker,
            line,
            speed=current_speed,
            energy=current_energy,
            play=current_play,
            out_path=None,
        )
        print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI baseline tester for F5-TTS (no frontend).")
    parser.add_argument(
        "--text",
        "-t",
        help="Synthesize once and exit. Omit for interactive mode.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speaking rate passed to F5-TTS (0.5-2.0).",
    )
    parser.add_argument(
        "--energy",
        type=float,
        default=1.0,
        help="Energy mapped to F5-TTS target_rms (0.5-2.0).",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Save WAV only; do not play audio.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output WAV path (default: baseline/outputs/speech_<timestamp>.wav).",
    )
    return parser


async def async_main() -> int:
    args = build_parser().parse_args()
    speaker = BaselineSpeaker()
    await speaker.load()

    play = not args.no_play
    if args.text:
        await run_once(
            speaker,
            args.text,
            speed=args.speed,
            energy=args.energy,
            play=play,
            out_path=args.output,
        )
        return 0

    await interactive_loop(
        speaker,
        speed=args.speed,
        energy=args.energy,
        play=play,
    )
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
