"""GPU inference scheduling."""

from __future__ import annotations

import asyncio


class InferenceScheduler:
    """Serializes GPU-bound synthesis jobs for M1.

    CosyVoice is not designed for efficient multi-session GPU concurrency.
    M1 allows one active synthesis at a time via a shared asyncio lock.
    """

    def __init__(self, max_concurrent: int = 1) -> None:
        if max_concurrent != 1:
            # M1 documents single-session GPU use; extend in later milestones.
            pass
        self._lock = asyncio.Lock()

    @property
    def gpu_lock(self) -> asyncio.Lock:
        return self._lock

    @property
    def has_active_job(self) -> bool:
        return self._lock.locked()
