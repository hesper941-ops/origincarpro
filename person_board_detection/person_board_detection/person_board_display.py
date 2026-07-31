"""Persistent Tkinter HDMI display for person-board recognition results."""

from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass
from typing import Any, List, Optional

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .result_protocol import (
    ProtocolError,
    dumps_message,
    make_display_status,
    parse_json_object,
    validate_llm_result,
    validate_llm_status,
)


@dataclass(frozen=True)
class DisplayEvent:
    state: str
    text: str
    event_id: str = ""
    request_id: str = ""
    error_reason: str = ""


class PersonBoardDisplayNode(Node):
    def __init__(self) -> None:
        super().__init__("person_board_display")
        self._declare_parameters()
        self.llm_status_topic = str(
            self.get_parameter("llm_status_topic").value
        )
        self.llm_result_topic = str(
            self.get_parameter("llm_result_topic").value
        )
        self.display_status_topic = str(
            self.get_parameter("display_status_topic").value
        )
        self.fullscreen = bool(self.get_parameter("fullscreen").value)
        self.always_on_top = bool(self.get_parameter("always_on_top").value)
        self.hide_cursor = bool(self.get_parameter("hide_cursor").value)
        self.allow_keyboard_exit = bool(
            self.get_parameter("allow_keyboard_exit").value
        )
        self.waiting_text = str(self.get_parameter("waiting_text").value)
        self.analyzing_text = str(self.get_parameter("analyzing_text").value)
        self.failed_text = str(self.get_parameter("failed_text").value)
        self.font_family = str(self.get_parameter("font_family").value)
        self.font_size = int(self.get_parameter("font_size").value)
        self.wrap_length_ratio = float(
            self.get_parameter("wrap_length_ratio").value
        )
        self.window_title = str(self.get_parameter("window_title").value)
        self.display_width = int(self.get_parameter("display_width").value)
        self.display_height = int(self.get_parameter("display_height").value)
        self._validate_parameters()
        latched_qos = QoSProfile(depth=1)
        latched_qos.reliability = ReliabilityPolicy.RELIABLE
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.status_publisher = self.create_publisher(
            String, self.display_status_topic, latched_qos
        )
        self.status_subscription = self.create_subscription(
            String,
            self.llm_status_topic,
            self._llm_status_callback,
            latched_qos,
        )
        self.result_subscription = self.create_subscription(
            String,
            self.llm_result_topic,
            self._llm_result_callback,
            latched_qos,
        )
        self.events: "queue.Queue[DisplayEvent]" = queue.Queue(maxsize=8)

    def _declare_parameters(self) -> None:
        defaults = {
            "llm_status_topic": "/person_board/llm_status",
            "llm_result_topic": "/person_board/llm_result",
            "display_status_topic": "/person_board/display_status",
            "fullscreen": True,
            "always_on_top": False,
            "hide_cursor": True,
            "allow_keyboard_exit": True,
            "waiting_text": "等待识别",
            "analyzing_text": "正在分析……",
            "failed_text": "识别失败",
            "font_family": "Noto Sans CJK SC",
            "font_size": 56,
            "wrap_length_ratio": 0.80,
            "window_title": "人形立牌识别",
            "display_width": 0,
            "display_height": 0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _validate_parameters(self) -> None:
        for name, value in (
            ("llm_status_topic", self.llm_status_topic),
            ("llm_result_topic", self.llm_result_topic),
            ("display_status_topic", self.display_status_topic),
            ("waiting_text", self.waiting_text),
            ("analyzing_text", self.analyzing_text),
            ("failed_text", self.failed_text),
        ):
            if not value:
                raise ValueError(f"{name} must not be empty")
        if self.font_size < 8:
            raise ValueError("font_size must be at least 8")
        if not 0.1 <= self.wrap_length_ratio <= 1.0:
            raise ValueError("wrap_length_ratio must be in [0.1, 1.0]")
        if self.display_width < 0 or self.display_height < 0:
            raise ValueError("display dimensions must be non-negative")

    def _put_event(self, event: DisplayEvent) -> None:
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            self.events.put_nowait(event)

    def _llm_status_callback(self, message: String) -> None:
        try:
            payload = parse_json_object(
                message.data, "llm_status_invalid_json"
            )
            validate_llm_status(payload)
        except ProtocolError as exc:
            self.get_logger().error(f"invalid llm_status: {exc.reason}")
            self._put_event(
                DisplayEvent(
                    "SHOWING_ERROR", self.failed_text, error_reason=exc.reason
                )
            )
            return
        state = str(payload["state"])
        event_id = str(payload.get("event_id", ""))
        request_id = str(payload.get("request_id", ""))
        error_reason = str(payload.get("error_reason", ""))
        if state in {"RECEIVED", "VALIDATING"}:
            self._put_event(
                DisplayEvent(
                    "WAITING", self.waiting_text, event_id, request_id
                )
            )
        elif state == "ANALYZING":
            self._put_event(
                DisplayEvent(
                    "ANALYZING", self.analyzing_text, event_id, request_id
                )
            )
        elif state in {"FAILED", "REJECTED_DUPLICATE", "REJECTED_BUSY"}:
            text = self.failed_text
            if error_reason:
                text += f"\n{error_reason}"
            self._put_event(
                DisplayEvent(
                    "SHOWING_ERROR", text, event_id, request_id, error_reason
                )
            )

    def _llm_result_callback(self, message: String) -> None:
        try:
            payload = parse_json_object(
                message.data, "llm_result_invalid_json"
            )
            validate_llm_result(payload)
        except ProtocolError as exc:
            self.get_logger().error(f"invalid llm_result: {exc.reason}")
            self._put_event(
                DisplayEvent(
                    "SHOWING_ERROR", self.failed_text, error_reason=exc.reason
                )
            )
            return
        event_id = str(payload.get("event_id", ""))
        request_id = str(payload.get("request_id", ""))
        if bool(payload["success"]):
            text = str(payload.get("result_text", "")).strip()
            if not text:
                text = self.failed_text
                self._put_event(
                    DisplayEvent(
                        "SHOWING_ERROR",
                        text + "\nresult_text_empty",
                        event_id,
                        request_id,
                        "result_text_empty",
                    )
                )
                return
            self._put_event(
                DisplayEvent("SHOWING_RESULT", text, event_id, request_id)
            )
        else:
            reason = str(payload.get("error_reason", "unknown_error"))
            self._put_event(
                DisplayEvent(
                    "SHOWING_ERROR",
                    f"{self.failed_text}\n{reason}",
                    event_id,
                    request_id,
                    reason,
                )
            )

    def publish_display_status(
        self, event: DisplayEvent, display_active: bool, fullscreen: bool
    ) -> None:
        payload = make_display_status(
            event.state,
            display_active=display_active,
            fullscreen=fullscreen,
            event_id=event.event_id,
            request_id=event.request_id,
            displayed_text=event.text,
            error_reason=event.error_reason,
        )
        message = String()
        message.data = dumps_message(payload)
        self.status_publisher.publish(message)


class TkDisplayApplication:
    def __init__(
        self, node: PersonBoardDisplayNode, tk: Any, tkfont: Any
    ) -> None:
        self.node = node
        self.tk = tk
        self.closed = False
        self.root = tk.Tk()
        self.root.title(node.window_title)
        self.root.configure(background="black")
        self.root.attributes("-topmost", node.always_on_top)
        self.root.attributes("-fullscreen", node.fullscreen)
        self.fullscreen = node.fullscreen
        if node.hide_cursor:
            self.root.configure(cursor="none")
        if node.display_width > 0 and node.display_height > 0:
            self.root.geometry(
                f"{node.display_width}x{node.display_height}+0+0"
            )
        families = set(tkfont.families(self.root))
        candidates = [
            node.font_family,
            "Noto Sans CJK SC",
            "WenQuanYi Zen Hei",
            "Microsoft YaHei",
            "Arial",
            "TkDefaultFont",
        ]
        actual_family = next(
            (family for family in candidates if family in families),
            "TkDefaultFont",
        )
        self.node.get_logger().info(f"display font: {actual_family}")
        self.font = tkfont.Font(family=actual_family, size=node.font_size)
        screen_width = node.display_width or self.root.winfo_screenwidth()
        wrap_length = max(100, int(screen_width * node.wrap_length_ratio))
        self.label = tk.Label(
            self.root,
            text=node.waiting_text,
            font=self.font,
            foreground="white",
            background="black",
            justify="center",
            anchor="center",
            wraplength=wrap_length,
        )
        self.label.pack(fill="both", expand=True, padx=30, pady=30)
        self.root.bind("<Escape>", self._leave_fullscreen)
        self.root.bind("<F11>", self._toggle_fullscreen)
        self.root.bind("q", self._keyboard_exit)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        starting = DisplayEvent("STARTING", node.waiting_text)
        self.current_event = starting
        node.publish_display_status(starting, True, self.fullscreen)
        self._show(DisplayEvent("WAITING", node.waiting_text))
        self.root.after(50, self._poll_events)

    def _show(self, event: DisplayEvent) -> None:
        self.current_event = event
        self.label.configure(text=event.text)
        self.node.publish_display_status(event, True, self.fullscreen)

    def _poll_events(self) -> None:
        if self.closed:
            return
        latest: Optional[DisplayEvent] = None
        while True:
            try:
                latest = self.node.events.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self._show(latest)
        self.root.after(50, self._poll_events)

    def _leave_fullscreen(self, _event: Any = None) -> str:
        self.fullscreen = False
        self.root.attributes("-fullscreen", False)
        self.node.publish_display_status(self.current_event, True, False)
        return "break"

    def _toggle_fullscreen(self, _event: Any = None) -> str:
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)
        self.node.publish_display_status(
            self.current_event, True, self.fullscreen
        )
        return "break"

    def _keyboard_exit(self, _event: Any = None) -> str:
        if self.node.allow_keyboard_exit:
            self.close()
        return "break"

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.node.publish_display_status(
                DisplayEvent("SHUTTING_DOWN", self.label.cget("text")),
                False,
                self.fullscreen,
            )
        finally:
            self.root.quit()
            self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def _publish_gui_unavailable(
    node: PersonBoardDisplayNode, reason: str, spin_once: bool = False
) -> None:
    node.get_logger().error(reason)
    node.publish_display_status(
        DisplayEvent("GUI_UNAVAILABLE", node.failed_text, error_reason=reason),
        False,
        node.fullscreen,
    )
    if spin_once:
        # Give DDS a short opportunity to match subscribers before teardown.
        rclpy.spin_once(node, timeout_sec=0.1)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[PersonBoardDisplayNode] = None
    executor: Optional[SingleThreadedExecutor] = None
    ros_thread: Optional[threading.Thread] = None
    try:
        node = PersonBoardDisplayNode()
        if not os.environ.get("DISPLAY"):
            _publish_gui_unavailable(
                node, "DISPLAY is not set", spin_once=True
            )
            return
        try:
            import tkinter as tk
            import tkinter.font as tkfont
        except ImportError:
            _publish_gui_unavailable(
                node,
                "tkinter is unavailable; install python3-tk",
                spin_once=True,
            )
            return
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        ros_thread = threading.Thread(
            target=executor.spin,
            name="person-board-display-ros",
            daemon=True,
        )
        ros_thread.start()
        try:
            application = TkDisplayApplication(node, tk, tkfont)
            application.run()
        except tk.TclError as exc:
            _publish_gui_unavailable(
                node, f"Tk GUI initialization failed: {exc}"
            )
    except KeyboardInterrupt:
        pass
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=2.0)
        if ros_thread is not None and ros_thread.is_alive():
            ros_thread.join(timeout=2.0)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
