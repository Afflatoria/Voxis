"""WebSocket protocol message schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class StartMessage(BaseModel):
    type: Literal["start"] = "start"
    text: str
    voice: dict[str, Any] | None = None
    request_id: str | None = None


class StopMessage(BaseModel):
    type: Literal["stop"] = "stop"


class VoiceUpdateMessage(BaseModel):
    type: Literal["voice_update"] = "voice_update"
    voice: dict[str, Any]


class PingMessage(BaseModel):
    type: Literal["ping"] = "ping"


ClientMessage = StartMessage | StopMessage | VoiceUpdateMessage | PingMessage


def parse_client_message(data: dict[str, Any]) -> ClientMessage:
    msg_type = data.get("type")
    if msg_type == "start":
        return StartMessage.model_validate(data)
    if msg_type == "stop":
        return StopMessage.model_validate(data)
    if msg_type == "voice_update":
        return VoiceUpdateMessage.model_validate(data)
    if msg_type == "ping":
        return PingMessage.model_validate(data)
    raise ValueError(f"unknown message type: {msg_type!r}")


class ServerEvent(BaseModel):
    type: str
    session_id: str | None = None
    request_id: str | None = None


class ReadyEvent(ServerEvent):
    type: Literal["ready"] = "ready"
    backend: dict[str, Any]
    supported_controls: list[str]
    future_controls: list[str]


class StartedEvent(ServerEvent):
    type: Literal["started"] = "started"


class MetricsEvent(ServerEvent):
    type: Literal["metrics"] = "metrics"
    metrics: dict[str, Any]


class CompleteEvent(ServerEvent):
    type: Literal["complete"] = "complete"
    metrics: dict[str, Any]


class CancelledEvent(ServerEvent):
    type: Literal["cancelled"] = "cancelled"


class ErrorEvent(ServerEvent):
    type: Literal["error"] = "error"
    code: str
    message: str


class VoiceUpdatedEvent(ServerEvent):
    type: Literal["voice_updated"] = "voice_updated"
    voice: dict[str, Any]
    note: str = "Applied to session state; affects subsequent synthesis in M1."


class PongEvent(ServerEvent):
    type: Literal["pong"] = "pong"
