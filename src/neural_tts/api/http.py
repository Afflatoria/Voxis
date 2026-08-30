"""HTTP routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from neural_tts.voice.schema import COSYVOICE_SUPPORTED_CONTROLS, FUTURE_CONTROLS

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
        "supported_controls": sorted(COSYVOICE_SUPPORTED_CONTROLS),
        "future_controls": sorted(FUTURE_CONTROLS),
        "notes": {
            "pitch": "Reserved for future architecture; not applied by CosyVoice backend in M1.",
            "warmth": "Reserved for future architecture; not applied by CosyVoice backend in M1.",
            "breathiness": "Reserved for future architecture; not applied by CosyVoice backend in M1.",
            "roughness": "Reserved for future architecture; not applied by CosyVoice backend in M1.",
            "expressiveness": "Reserved for future architecture; not applied by CosyVoice backend in M1.",
        },
    }
