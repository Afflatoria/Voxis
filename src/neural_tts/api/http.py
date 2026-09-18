"""HTTP routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from neural_tts.voice.schema import F5TTS_SUPPORTED_CONTROLS, FUTURE_CONTROLS
from neural_tts.voice.genome import CandidateRequest, VoiceGenome, generate_candidates

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    app_state = request.app.state
    backend = app_state.backend
    info = backend.info()
    return {
        "status": "ok" if info.ready else "loading",
        "backend": {
            "name": info.name,
            "model": info.model_name,
            "device": info.device,
            "sample_rate": info.sample_rate,
            "ready": info.ready,
            "load_time_seconds": info.load_time_seconds,
        },
        "gpu_busy": app_state.scheduler.has_active_job,
    }


@router.get("/v1/capabilities")
async def capabilities():
    return {
        "supported_controls": sorted(F5TTS_SUPPORTED_CONTROLS),
        "future_controls": sorted(FUTURE_CONTROLS),
        "notes": {
            "speaker": "M1 uses the reference voice from F5TTS_REF_WAV in .env.",
            "speaking_rate": "Mapped to F5-TTS speed parameter.",
            "energy": "Mapped to F5-TTS target_rms loudness normalization.",
            "language": "Reserved for future architecture; not applied by F5-TTS backend in M1.",
            "accent": "Reserved for future architecture; not applied by F5-TTS backend in M1.",
            "pitch": "Reserved for future architecture; not applied by F5-TTS backend in M1.",
            "warmth": "Reserved for future architecture; not applied by F5-TTS backend in M1.",
            "breathiness": "Reserved for future architecture; not applied by F5-TTS backend in M1.",
            "roughness": "Reserved for future architecture; not applied by F5-TTS backend in M1.",
            "expressiveness": "Reserved for future architecture; not applied by F5-TTS backend in M1.",
        },
    }


@router.post("/v1/voice-genomes/candidates", response_model=list[VoiceGenome])
async def voice_genome_candidates(request: CandidateRequest):
    """Create reproducible artificial voice candidates for A/B exploration."""
    return generate_candidates(request)
