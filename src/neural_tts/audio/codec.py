"""Audio chunk representation and PCM encoding helpers."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

# Binary frame magic for WebSocket audio transport.
AUDIO_FRAME_MAGIC = b"NTTS"
AUDIO_FRAME_VERSION = 1
AUDIO_HEADER_SIZE = 16  # magic(4) + version(1) + reserved(3) + seq(4) + sr(4)


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """One chunk of synthesized speech."""

    sequence: int
    sample_rate: int
    pcm_s16le: bytes
    duration_seconds: float

    @property
    def num_samples(self) -> int:
        return len(self.pcm_s16le) // 2


def float32_to_pcm_s16le(samples: np.ndarray) -> bytes:
    """Convert float32 waveform in [-1, 1] to little-endian int16 PCM."""
    clipped = np.clip(samples, -1.0, 1.0)
    int16 = (clipped * 32767.0).astype(np.int16)
    return int16.tobytes()


def pcm_s16le_to_float32(pcm: bytes) -> np.ndarray:
    arr = np.frombuffer(pcm, dtype=np.int16)
    return arr.astype(np.float32) / 32767.0


def encode_audio_frame(chunk: AudioChunk) -> bytes:
    """Pack PCM into a binary WebSocket frame with a small header."""
    header = struct.pack(
        "<4sB3xII",
        AUDIO_FRAME_MAGIC,
        AUDIO_FRAME_VERSION,
        chunk.sequence,
        chunk.sample_rate,
    )
    return header + chunk.pcm_s16le


def decode_audio_frame(payload: bytes) -> tuple[int, int, bytes]:
    """Decode a binary audio frame. Returns (sequence, sample_rate, pcm)."""
    if len(payload) < AUDIO_HEADER_SIZE:
        raise ValueError("audio frame too short")
    magic, version, sequence, sample_rate = struct.unpack(
        "<4sB3xII", payload[:AUDIO_HEADER_SIZE]
    )
    if magic != AUDIO_FRAME_MAGIC:
        raise ValueError("invalid audio frame magic")
    if version != AUDIO_FRAME_VERSION:
        raise ValueError(f"unsupported audio frame version: {version}")
    return sequence, sample_rate, payload[AUDIO_HEADER_SIZE:]
