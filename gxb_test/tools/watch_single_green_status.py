#!/usr/bin/env python3
"""Low-rate watcher for single-green fusion and corridor-DP diagnostics."""

import json
import signal
import sys
import time
from typing import Any, Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class SingleGreenWatcher(Node):
    def __init__(self) -> None:
        super().__init__("gxb_single_green_watcher")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._last_print = 0.0
        self.create_subscription(
            String,
            "/gxb_test/pipeline/status",
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
            print(
                "SINGLE_GREEN "
                f"mode={status.get('centerline_mode', '-')} "
                f"valid={status.get('centerline_valid', False)} "
                f"points={status.get('centerline_point_count', 0)} "
                f"span={float(status.get('centerline_forward_span_m', 0.0)):.3f} "
                f"side={status.get('single_green_curve_side', '-')} "
                f"accepted={status.get('single_green_curve_accepted_count', 0)} "
                f"fusion={status.get('single_green_curve_pre_fusion_count', 0)}"
                f"->{status.get('single_green_curve_post_fusion_count', 0)} "
                f"fusion_reason={status.get('single_green_curve_fusion_reject_reason', '-')} "
                f"dp={status.get('single_green_dp_attempted', False)}/"
                f"{status.get('single_green_dp_used', False)} "
                f"stations={status.get('single_green_dp_station_count', 0)} "
                f"candidates={status.get('single_green_dp_candidate_count', 0)} "
                f"output={status.get('single_green_dp_output_count', 0)} "
                f"dp_span={float(status.get('single_green_dp_output_span_m', 0.0)):.3f} "
                f"gaps={status.get('single_green_dp_gap_count', 0)} "
                f"cost={float(status.get('single_green_dp_total_cost', 0.0)):.3f} "
                f"dp_ms={float(status.get('single_green_dp_ms', 0.0)):.3f} "
                f"dp_reason={status.get('single_green_dp_failure_reason', '-')}",
                flush=True,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"SINGLE_GREEN INVALID_JSON error={exc}", flush=True)


def main() -> int:
    rclpy.init(args=None)
    node = SingleGreenWatcher()

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
