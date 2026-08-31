"""WebSocket streaming endpoint."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from neural_tts.api.protocol import (
    CancelledEvent,
    CompleteEvent,
    ErrorEvent,
    PingMessage,
    ReadyEvent,
    StartedEvent,
    StopMessage,
    VoiceUpdateMessage,
    VoiceUpdatedEvent,
    parse_client_message,
)
from neural_tts.audio.codec import encode_audio_frame
from neural_tts.logging_config import get_logger
from neural_tts.streaming.session import StreamingSession
from neural_tts.streaming.state import SessionStatus
from neural_tts.voice.schema import F5TTS_SUPPORTED_CONTROLS, FUTURE_CONTROLS, VoiceConfig

logger = get_logger(__name__)
router = APIRouter()


async def _send_json(websocket: WebSocket, payload: dict[str, Any]) -> None:
    await websocket.send_text(json.dumps(payload))


@router.websocket("/v1/stream")
async def stream_tts(websocket: WebSocket) -> None:
    await websocket.accept()
    backend = websocket.app.state.backend
    scheduler = websocket.app.state.scheduler

    session: StreamingSession | None = None
    synth_task: asyncio.Task | None = None
    message_queue: asyncio.Queue[str | None] = asyncio.Queue()

    info = backend.info()
    await _send_json(
        websocket,
        ReadyEvent(
            backend={
                "name": info.name,
                "model": info.model_name,
                "device": info.device,
                "sample_rate": info.sample_rate,
                "ready": info.ready,
            },
            supported_controls=sorted(F5TTS_SUPPORTED_CONTROLS),
            future_controls=sorted(FUTURE_CONTROLS),
        ).model_dump(),
    )

    async def reader() -> None:
        try:
            while True:
                raw = await websocket.receive()
                if raw["type"] == "websocket.disconnect":
                    await message_queue.put(None)
                    return
                if raw.get("text") is not None:
                    await message_queue.put(raw["text"])
                elif raw.get("bytes") is not None:
                    await message_queue.put("__binary__")
        except WebSocketDisconnect:
            await message_queue.put(None)

    reader_task = asyncio.create_task(reader())

    async def cancel_synthesis() -> None:
        nonlocal synth_task, session
        if session is not None:
            session.cancel()
        if synth_task is not None and not synth_task.done():
            synth_task.cancel()
            try:
                await synth_task
            except asyncio.CancelledError:
                pass
        synth_task = None

    async def run_synthesis(text: str, voice: VoiceConfig, request_id: str | None) -> None:
        nonlocal session
        assert session is not None
        try:
            async for chunk in session.synthesize(
                text=text,
                voice=voice.model_dump(),
                request_id=request_id,
            ):
                if session.state.is_cancelled():
                    break
                frame = encode_audio_frame(chunk)
                await websocket.send_bytes(frame)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "inference_error",
                session_id=session.state.session_id,
                error=str(exc),
            )
            session.state.status = SessionStatus.ERROR
            await _send_json(
                websocket,
                ErrorEvent(
                    session_id=session.state.session_id,
                    request_id=request_id,
                    code="inference_error",
                    message=str(exc),
                ).model_dump(),
            )
            return

        peak_mb = await backend.peak_gpu_memory_mb()
        session.state.metrics.peak_gpu_memory_mb = peak_mb
        metrics = session.state.metrics.to_dict(session.state.timestamps)
        metrics["session_id"] = session.state.session_id
        metrics["request_id"] = request_id
        metrics["status"] = session.state.status.value

        if session.state.status == SessionStatus.CANCELLED:
            await _send_json(
                websocket,
                CancelledEvent(
                    session_id=session.state.session_id,
                    request_id=request_id,
                ).model_dump(),
            )
        elif session.state.status != SessionStatus.ERROR:
            await _send_json(
                websocket,
                CompleteEvent(
                    session_id=session.state.session_id,
                    request_id=request_id,
                    metrics=metrics,
                ).model_dump(),
            )

    try:
        while True:
            item = await message_queue.get()
            if item is None:
                break
            if item == "__binary__":
                await _send_json(
                    websocket,
                    ErrorEvent(
                        code="unexpected_binary",
                        message="client must send JSON text frames only",
                    ).model_dump(),
                )
                continue

            try:
                data = json.loads(item)
                message = parse_client_message(data)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                await _send_json(
                    websocket,
                    ErrorEvent(code="malformed_message", message=str(exc)).model_dump(),
                )
                continue

            if isinstance(message, PingMessage):
                await _send_json(websocket, {"type": "pong"})
                continue

            if isinstance(message, StopMessage):
                await cancel_synthesis()
                if session is not None:
                    await _send_json(
                        websocket,
                        CancelledEvent(session_id=session.state.session_id).model_dump(),
                    )
                continue

            if isinstance(message, VoiceUpdateMessage):
                if session is None:
                    await _send_json(
                        websocket,
                        ErrorEvent(
                            code="no_active_session",
                            message="voice_update requires an active session",
                        ).model_dump(),
                    )
                    continue
                voice = session.update_voice(message.voice)
                await _send_json(
                    websocket,
                    VoiceUpdatedEvent(
                        session_id=session.state.session_id,
                        voice=voice.to_public_dict(),
                    ).model_dump(),
                )
                continue

            # StartMessage — cancel any in-flight synthesis first
            await cancel_synthesis()

            if not message.text.strip():
                await _send_json(
                    websocket,
                    ErrorEvent(code="empty_text", message="text must not be empty").model_dump(),
                )
                continue

            try:
                voice = VoiceConfig.model_validate(message.voice or {})
            except ValidationError as exc:
                await _send_json(
                    websocket,
                    ErrorEvent(code="invalid_voice", message=str(exc)).model_dump(),
                )
                continue

            session = StreamingSession(backend, scheduler.gpu_lock)
            await _send_json(
                websocket,
                StartedEvent(
                    session_id=session.state.session_id,
                    request_id=message.request_id,
                ).model_dump(),
            )
            synth_task = asyncio.create_task(
                run_synthesis(message.text, voice, message.request_id)
            )

    finally:
        await cancel_synthesis()
        reader_task.cancel()
        try:
            await reader_task
        except asyncio.CancelledError:
            pass
