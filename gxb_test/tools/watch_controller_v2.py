#!/usr/bin/env python3
"""Low-rate console watcher for the dry-run lane controller diagnostics."""

import json
import signal
import sys
import time
from typing import Any, Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class ControllerWatcher(Node):
    def __init__(self) -> None:
        super().__init__("gxb_controller_watcher")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._last_print = 0.0
        self.create_subscription(
            String,
            "/gxb_test/controller/status",
            self._callback,
            qos,
        )

    def _callback(self, message: String) -> None:
        now = time.monotonic()
        if now - self._last_print < 1.0:
            return
        self._last_print = now
        try:
            status: Dict[str, Any] = json.loads(message.data)
            ready = bool(status.get("controller_ready", False))
            print(
                "CONTROLLER "
                f"{'READY' if ready else 'BLOCKED'} "
                f"quality={status.get('control_quality_level', '-')} "
                f"reason={status.get('control_block_reason', '-')} "
                f"mode={status.get('centerline_mode', '-')} "
                f"points={status.get('path_point_count', 0)} "
                f"span={float(status.get('path_forward_span_m', 0.0)):.3f} "
                f"lat={float(status.get('lateral_error_m', 0.0)):+.3f} "
                f"heading={float(status.get('heading_error_deg', 0.0)):+.2f} "
                f"v={float(status.get('suggested_linear_x', 0.0)):.3f} "
                f"w={float(status.get('suggested_angular_z', 0.0)):+.3f}",
                flush=True,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"CONTROLLER INVALID_JSON error={exc}", flush=True)


def main() -> int:
    rclpy.init(args=None)
    node = ControllerWatcher()

    def stop(_signum: int, _frame: Any) -> None:
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
