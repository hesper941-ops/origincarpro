"""Unit tests for HDMI display transitions; no ROS or Tk display required."""

from person_board_detection.person_board_detection.display_state import (
    DisplayStateReducer,
)


def reducer():
    return DisplayStateReducer("等待识别", "正在识别……", "识别失败，请重试")


def status(state, request_id="req-a", event_id="event-a", error_reason=""):
    return {
        "state": state,
        "request_id": request_id,
        "event_id": event_id,
        "error_reason": error_reason,
    }


def result(text, request_id="req-a", event_id="event-a", success=True):
    return {
        "success": success,
        "result_text": text if success else "",
        "request_id": request_id,
        "event_id": event_id,
        "error_reason": "request_failed" if not success else "",
    }


def test_successful_result_persists():
    state = reducer()
    shown = state.handle_result(result("A"))
    for _ in range(5):
        assert state.handle_status(status("IDLE", "", "")) is None
    assert state.current_event == shown
    assert state.current_event.text == "A"


def test_clear_command():
    state = reducer()
    state.handle_result(result("A"))
    event, reason = state.handle_control_text('{"command":"clear"}')
    assert reason == ""
    assert event.state == "WAITING"
    assert event.text == "等待识别"
    assert event.event_id == ""
    assert event.request_id == ""


def test_next_result_after_clear():
    state = reducer()
    state.handle_result(result("A"))
    state.handle_control_text('{"command":"clear"}')
    state.handle_result(result("B", "req-b", "event-b"))
    assert state.current_event.text == "B"
    assert state.current_request_id == "req-b"


def test_new_result_replaces_old():
    state = reducer()
    state.handle_result(result("A"))
    state.handle_result(result("B", "req-b", "event-b"))
    assert state.current_event.text == "B"
    assert state.current_request_id == "req-b"


def test_stale_result_is_ignored():
    state = reducer()
    state.handle_status(status("VALIDATING"))
    state.handle_status(status("VALIDATING", "req-b", "event-b"))
    assert state.handle_result(result("A")) is None
    assert state.current_request_id == "req-b"
    state.handle_result(result("B", "req-b", "event-b"))
    assert state.current_event.text == "B"


def test_invalid_control_keeps_current_result():
    state = reducer()
    shown = state.handle_result(result("A"))
    event, reason = state.handle_control_text("abc")
    assert event is None
    assert reason == "display_control_invalid_json"
    assert state.current_event == shown


def test_unknown_command_keeps_current_result():
    state = reducer()
    shown = state.handle_result(result("A"))
    event, reason = state.handle_control_text('{"command":"xxx"}')
    assert event is None
    assert reason == "display_control_unknown_command"
    assert state.current_event == shown


def test_failed_new_request():
    state = reducer()
    state.handle_result(result("A"))
    state.handle_status(status("VALIDATING", "req-b", "event-b"))
    failed = state.handle_status(
        status("FAILED", "req-b", "event-b", "socket_timeout")
    )
    assert failed.text == "识别失败，请重试"
    assert "socket_timeout" not in failed.text
    assert state.current_event.text != "A"
