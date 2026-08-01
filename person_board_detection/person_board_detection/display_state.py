"""Pure display state transitions for the person-board HDMI node."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Set, Tuple


ACTIVE_STATES = frozenset(
    {
        "RECEIVED",
        "VALIDATING",
        "SELECTING_IMAGE",
        "ENCODING_IMAGE",
        "ANALYZING",
        "PARSING",
    }
)
FAILED_STATES = frozenset(
    {"FAILED", "REJECTED_DUPLICATE", "REJECTED_BUSY"}
)


@dataclass(frozen=True)
class DisplayEvent:
    state: str
    text: str
    event_id: str = ""
    request_id: str = ""
    error_reason: str = ""


class DisplayStateReducer:
    """Resolve LLM/control messages into ordered, persistent UI events."""

    def __init__(
        self, waiting_text: str, analyzing_text: str, failed_text: str
    ) -> None:
        self.waiting_text = waiting_text
        self.analyzing_text = analyzing_text
        self.failed_text = failed_text
        self.current_event = DisplayEvent("WAITING", waiting_text)
        self.current_event_id = ""
        self.current_request_id = ""
        self._superseded_request_ids: Set[str] = set()
        self._superseded_event_ids: Set[str] = set()

    def _remember_current_as_superseded(self) -> None:
        if self.current_request_id:
            self._superseded_request_ids.add(self.current_request_id)
        if self.current_event_id:
            self._superseded_event_ids.add(self.current_event_id)

    def _is_superseded(self, event_id: str, request_id: str) -> bool:
        if request_id:
            return request_id in self._superseded_request_ids
        return bool(event_id and event_id in self._superseded_event_ids)

    def _is_current_identity(self, event_id: str, request_id: str) -> bool:
        if request_id:
            return request_id == self.current_request_id
        if event_id:
            return event_id == self.current_event_id
        return True

    def _adopt(self, event_id: str, request_id: str) -> None:
        if request_id and request_id != self.current_request_id:
            self._remember_current_as_superseded()
            self.current_request_id = request_id
            self.current_event_id = event_id
        elif (
            not request_id
            and event_id
            and event_id != self.current_event_id
        ):
            self._remember_current_as_superseded()
            self.current_request_id = ""
            self.current_event_id = event_id
        elif event_id:
            self.current_event_id = event_id

    def _emit(self, event: DisplayEvent) -> DisplayEvent:
        self.current_event = event
        self.current_event_id = event.event_id
        self.current_request_id = event.request_id
        return event

    def handle_status(
        self, payload: Mapping[str, Any]
    ) -> Optional[DisplayEvent]:
        state = str(payload["state"])
        event_id = str(payload.get("event_id", ""))
        request_id = str(payload.get("request_id", ""))
        error_reason = str(payload.get("error_reason", ""))

        # IDLE/heartbeat messages never erase a result or an in-progress task.
        if state in {"IDLE", "SUCCEEDED"}:
            return None
        if self._is_superseded(event_id, request_id):
            return None
        if (
            self.current_event.state
            in {"SHOWING_RESULT", "SHOWING_ERROR"}
            and self._is_current_identity(event_id, request_id)
        ):
            return None
        if state in ACTIVE_STATES:
            self._adopt(event_id, request_id)
            return self._emit(
                DisplayEvent(
                    "ANALYZING",
                    self.analyzing_text,
                    self.current_event_id,
                    self.current_request_id,
                )
            )
        if state in FAILED_STATES:
            self._adopt(event_id, request_id)
            return self._emit(
                DisplayEvent(
                    "SHOWING_ERROR",
                    self.failed_text,
                    self.current_event_id,
                    self.current_request_id,
                    error_reason,
                )
            )
        return None

    def handle_result(
        self, payload: Mapping[str, Any]
    ) -> Optional[DisplayEvent]:
        event_id = str(payload.get("event_id", ""))
        request_id = str(payload.get("request_id", ""))
        if self._is_superseded(event_id, request_id):
            return None
        self._adopt(event_id, request_id)
        if bool(payload["success"]):
            return self._emit(
                DisplayEvent(
                    "SHOWING_RESULT",
                    str(payload["result_text"]).strip(),
                    self.current_event_id,
                    self.current_request_id,
                )
            )
        return self._emit(
            DisplayEvent(
                "SHOWING_ERROR",
                self.failed_text,
                self.current_event_id,
                self.current_request_id,
                str(payload.get("error_reason", "unknown_error")),
            )
        )

    def clear(self) -> DisplayEvent:
        self._remember_current_as_superseded()
        self.current_event_id = ""
        self.current_request_id = ""
        return self._emit(DisplayEvent("WAITING", self.waiting_text))

    def handle_control_text(
        self, text: str
    ) -> Tuple[Optional[DisplayEvent], str]:
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return None, "display_control_invalid_json"
        if not isinstance(payload, dict):
            return None, "display_control_invalid_json"
        if payload.get("command") != "clear":
            return None, "display_control_unknown_command"
        return self.clear(), ""
