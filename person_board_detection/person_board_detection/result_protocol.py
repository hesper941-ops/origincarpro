"""Shared JSON protocol helpers for the person-board stage-2 nodes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional


SCHEMA_VERSION = 1
LLM_STATES = frozenset(
    {
        "IDLE",
        "RECEIVED",
        "VALIDATING",
        "SELECTING_IMAGE",
        "ENCODING_IMAGE",
        "ANALYZING",
        "PARSING",
        "SUCCEEDED",
        "FAILED",
        "REJECTED_DUPLICATE",
        "REJECTED_BUSY",
    }
)
DISPLAY_STATES = frozenset(
    {
        "STARTING",
        "WAITING",
        "ANALYZING",
        "SHOWING_RESULT",
        "SHOWING_ERROR",
        "GUI_UNAVAILABLE",
        "SHUTTING_DOWN",
    }
)


class ProtocolError(ValueError):
    """A stable machine-readable protocol error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class CaptureBatch:
    event_id: str
    request_id: str
    manifest_path: str
    image_paths: List[str]


def timestamp_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def dumps_message(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"))


def parse_json_object(
    text: str, reason: str = "invalid_json"
) -> Dict[str, Any]:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProtocolError(reason) from exc
    if not isinstance(value, dict):
        raise ProtocolError(reason)
    return value


def _required_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"missing_{field}")
    return value.strip()


def parse_capture_batch(text: str) -> CaptureBatch:
    payload = parse_json_object(text, "capture_batch_invalid_json")
    image_paths = payload.get("image_paths")
    if not isinstance(image_paths, list) or any(
        not isinstance(path, str) or not path.strip() for path in image_paths
    ):
        raise ProtocolError("capture_batch_image_paths_invalid")
    return CaptureBatch(
        event_id=_required_text(payload, "event_id"),
        request_id=_required_text(payload, "request_id"),
        manifest_path=_required_text(payload, "manifest_path"),
        image_paths=[path.strip() for path in image_paths],
    )


def make_llm_status(
    state: str,
    *,
    backend: str,
    busy: bool,
    event_id: str = "",
    request_id: str = "",
    progress_text: str = "",
    error_reason: str = "",
) -> Dict[str, Any]:
    if state not in LLM_STATES:
        raise ProtocolError("llm_state_invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "backend": backend,
        "busy": bool(busy),
        "event_id": event_id,
        "request_id": request_id,
        "progress_text": progress_text,
        "error_reason": error_reason,
        "updated_at": timestamp_now(),
    }


def make_llm_result(
    *,
    success: bool,
    backend: str,
    event_id: str,
    request_id: str,
    result_text: str,
    manifest_path: str,
    image_paths: List[str],
    started_at: str,
    completed_at: Optional[str] = None,
    elapsed_ms: float = 0.0,
    error_reason: str = "",
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "success": bool(success),
        "backend": backend,
        "event_id": event_id,
        "request_id": request_id,
        "result_text": result_text,
        "manifest_path": manifest_path,
        "image_paths": list(image_paths),
        "started_at": started_at,
        "completed_at": completed_at or timestamp_now(),
        "elapsed_ms": round(max(0.0, float(elapsed_ms)), 3),
        "error_reason": error_reason,
    }


def make_display_status(
    state: str,
    *,
    display_active: bool,
    fullscreen: bool,
    event_id: str = "",
    request_id: str = "",
    displayed_text: str = "",
    error_reason: str = "",
) -> Dict[str, Any]:
    if state not in DISPLAY_STATES:
        raise ProtocolError("display_state_invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "display_active": bool(display_active),
        "fullscreen": bool(fullscreen),
        "event_id": event_id,
        "request_id": request_id,
        "displayed_text": displayed_text,
        "error_reason": error_reason,
        "updated_at": timestamp_now(),
    }


def validate_llm_status(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError("schema_version_invalid")
    if payload.get("state") not in LLM_STATES:
        raise ProtocolError("llm_state_invalid")
    _required_text(payload, "backend")
    if not isinstance(payload.get("busy"), bool):
        raise ProtocolError("llm_busy_invalid")
    for field in (
        "event_id",
        "request_id",
        "progress_text",
        "error_reason",
        "updated_at",
    ):
        if not isinstance(payload.get(field), str):
            raise ProtocolError(f"llm_{field}_invalid")


def validate_llm_result(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError("schema_version_invalid")
    if not isinstance(payload.get("success"), bool):
        raise ProtocolError("result_success_invalid")
    _required_text(payload, "backend")
    _required_text(payload, "event_id")
    _required_text(payload, "request_id")
    for field in (
        "result_text",
        "manifest_path",
        "started_at",
        "completed_at",
        "error_reason",
    ):
        if not isinstance(payload.get(field), str):
            raise ProtocolError(f"result_{field}_invalid")
    image_paths = payload.get("image_paths")
    if not isinstance(image_paths, list) or any(
        not isinstance(path, str) for path in image_paths
    ):
        raise ProtocolError("result_image_paths_invalid")
    if payload["success"] and not payload["result_text"].strip():
        raise ProtocolError("result_text_empty")
    if not payload["success"] and not payload["error_reason"].strip():
        raise ProtocolError("result_error_reason_empty")
