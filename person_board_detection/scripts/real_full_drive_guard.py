#!/usr/bin/env python3
"""Odom-bounded straight-drive guard used only by the Stage-3 real-full test."""

import argparse
import json
import math
import signal
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32MultiArray, String


STOP_SIGNAL = False


def request_stop(_signum, _frame):
    global STOP_SIGNAL
    STOP_SIGNAL = True


class DriveGuard(Node):
    def __init__(self, args):
        super().__init__("person_board_real_full_drive_guard")
        self.args = args
        self.cmd_pub = self.create_publisher(Twist, args.cmd_topic, 10)
        self.create_subscription(Odometry, args.odom_topic, self.on_odom, 20)
        self.create_subscription(String, "/person_board/capture_status",
                                 self.on_capture_status, 20)
        self.create_subscription(Bool, "/person_board/detected",
                                 self.on_detected, 20)
        self.create_subscription(Float32, "/person_board/score",
                                 self.on_score, 20)
        self.create_subscription(Int32MultiArray, "/person_board/box",
                                 self.on_box, 20)
        self.create_subscription(Float32, "/person_board/inference_ms",
                                 self.on_inference, 20)
        self.create_subscription(String, "/person_board/capture_done",
                                 self.on_capture_done, 10)
        self.create_subscription(String, "/person_board/llm_status",
                                 self.on_llm_status, 20)
        self.create_subscription(String, "/person_board/display_status",
                                 self.on_display_status, 20)
        self.started_mono = time.monotonic()
        self.drive_started_mono = None
        self.last_odom_mono = None
        self.x0 = self.y0 = self.x = self.y = None
        self.distance = 0.0
        self.stop_reason = ""
        self.detected = False
        self.capture_state = ""
        self.llm_state = ""
        self.latest_score = None
        self.latest_box = []
        self.last_periodic_log = -1.0
        self.timeline_path = Path(args.timeline)
        self.timeline_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeline = self.timeline_path.open("w", encoding="utf-8")
        self.ready_path = Path(args.ready_file)
        self.arm_path = Path(args.arm_file)
        self.summary_path = Path(args.summary)
        self.events = {}
        self.inference_samples = []
        self._event("guard_started")

    def elapsed(self):
        return time.monotonic() - self.started_mono

    def _write(self, event, **extra):
        value = {
            "elapsed_sec": round(self.elapsed(), 3),
            "event": event,
            "odom_x": self.x,
            "odom_y": self.y,
            "distance_m": round(self.distance, 4),
            "cmd_speed": 0.0 if self.drive_started_mono is None else self.args.speed,
            "person_board_state": self.capture_state,
            "detected": self.detected,
            "llm_state": self.llm_state,
        }
        value.update(extra)
        self.timeline.write(json.dumps(value, ensure_ascii=False) + "\n")
        self.timeline.flush()

    def _event(self, name, **extra):
        if name not in self.events:
            self.events[name] = round(self.elapsed(), 3)
        self._write(name, **extra)

    @staticmethod
    def twist(speed=0.0):
        msg = Twist()
        msg.linear.x = float(speed)
        return msg

    def on_odom(self, msg):
        self.last_odom_mono = time.monotonic()
        self.x = float(msg.pose.pose.position.x)
        self.y = float(msg.pose.pose.position.y)
        if self.x0 is None:
            self.x0, self.y0 = self.x, self.y
            self._event("first_odom")
        self.distance = math.hypot(self.x - self.x0, self.y - self.y0)

    def _json(self, msg):
        try:
            return json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return {}

    def on_capture_status(self, msg):
        payload = self._json(msg)
        state = str(payload.get("state", ""))
        if state and state != self.capture_state:
            self.capture_state = state
            self._event("capture_state", state=state,
                        stable_count=payload.get("stable_count"),
                        bbox=payload.get("bbox"),
                        confidence=payload.get("confidence"))
        if state == "FAILED":
            self.stop_reason = "test_failure"

    def on_detected(self, msg):
        if msg.data and not self.detected:
            self.detected = True
            self._event("first_yolo_detected",
                        score=self.latest_score, box=self.latest_box)

    def on_score(self, msg):
        self.latest_score = float(msg.data)

    def on_box(self, msg):
        self.latest_box = list(msg.data)
        self._write("bbox", box=self.latest_box, score=self.latest_score)

    def on_inference(self, msg):
        self.inference_samples.append(float(msg.data))

    def on_capture_done(self, msg):
        payload = self._json(msg)
        self._event("capture_done", success=payload.get("success"),
                    request_id=payload.get("request_id"),
                    error_reason=payload.get("error_reason"))
        if payload.get("success") is False:
            self.stop_reason = "test_failure"

    def on_llm_status(self, msg):
        payload = self._json(msg)
        state = str(payload.get("state", ""))
        if state and state != self.llm_state:
            self.llm_state = state
            self._event("llm_state", state=state,
                        request_id=payload.get("request_id"),
                        error_reason=payload.get("error_reason"))
        if state == "ANALYZING":
            self.events.setdefault("api_request", round(self.elapsed(), 3))
        if state == "SUCCEEDED":
            self.events.setdefault("api_success", round(self.elapsed(), 3))
        if state == "FAILED":
            self.stop_reason = "test_failure"

    def on_display_status(self, msg):
        payload = self._json(msg)
        if payload.get("state") == "SHOWING_RESULT":
            self._event("hdmi_showing_result",
                        request_id=payload.get("request_id"),
                        displayed_text=payload.get("displayed_text"))

    def publish_zero(self, seconds=3.0):
        end = time.monotonic() + seconds
        while time.monotonic() < end and rclpy.ok():
            self.cmd_pub.publish(self.twist())
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.05)
        for _ in range(10):
            self.cmd_pub.publish(self.twist())
            time.sleep(0.02)

    def run(self):
        ready_deadline = time.monotonic() + self.args.odom_start_timeout
        while rclpy.ok() and not STOP_SIGNAL:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.x0 is not None and self.cmd_pub.get_subscription_count() > 0:
                break
            if time.monotonic() >= ready_deadline:
                self.stop_reason = "odom_timeout"
                self._event("ready_failed",
                            cmd_subscribers=self.cmd_pub.get_subscription_count())
                self.publish_zero()
                return 2
        self.publish_zero(1.0)
        self.ready_path.write_text("ready\n", encoding="utf-8")
        self._event("ready",
                    cmd_subscribers=self.cmd_pub.get_subscription_count())

        while rclpy.ok() and not STOP_SIGNAL and not self.arm_path.exists():
            rclpy.spin_once(self, timeout_sec=0.1)
            if (self.last_odom_mono is None or
                    time.monotonic() - self.last_odom_mono > self.args.odom_stale):
                self.stop_reason = "odom_timeout"
                break
        if STOP_SIGNAL:
            self.stop_reason = "signal"
        if self.stop_reason:
            self.publish_zero()
            return 3

        self.drive_started_mono = time.monotonic()
        self._event("drive_started")
        next_pub = time.monotonic()
        while rclpy.ok() and not STOP_SIGNAL:
            now = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.01)
            drive_elapsed = now - self.drive_started_mono
            if self.stop_reason:
                break
            if self.distance >= self.args.max_distance:
                self.stop_reason = "distance_limit"
                break
            if drive_elapsed >= self.args.hard_timeout:
                self.stop_reason = "timeout"
                break
            if self.last_odom_mono is None or now - self.last_odom_mono > self.args.odom_stale:
                self.stop_reason = "odom_timeout"
                break
            if self.cmd_pub.get_subscription_count() < 1:
                self.stop_reason = "test_failure"
                break
            if now >= next_pub:
                self.cmd_pub.publish(self.twist(self.args.speed))
                next_pub = now + 1.0 / self.args.rate
            if drive_elapsed - self.last_periodic_log >= self.args.log_period:
                self.last_periodic_log = drive_elapsed
                self._write("drive_sample", drive_elapsed_sec=round(drive_elapsed, 3))
        if STOP_SIGNAL and not self.stop_reason:
            self.stop_reason = "signal"
        if not self.stop_reason:
            self.stop_reason = "other"
        self._event("stopping", stop_reason=self.stop_reason)
        self.publish_zero()
        return 0

    def close(self):
        drive_elapsed = 0.0
        if self.drive_started_mono is not None:
            drive_elapsed = time.monotonic() - self.drive_started_mono
        summary = {
            "x0": self.x0, "y0": self.y0, "x1": self.x, "y1": self.y,
            "final_distance_m": round(self.distance, 4),
            "drive_elapsed_sec": round(drive_elapsed, 3),
            "stop_reason": self.stop_reason or "other",
            "detected": self.detected,
            "capture_state": self.capture_state,
            "llm_state": self.llm_state,
            "events": self.events,
            "inference_sample_count": len(self.inference_samples),
            "inference_ms_mean": (
                round(sum(self.inference_samples) / len(self.inference_samples), 3)
                if self.inference_samples else None),
        }
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        self.timeline.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--cmd-topic", default="/cmd_vel")
    parser.add_argument("--speed", type=float, default=0.10)
    parser.add_argument("--max-distance", type=float, default=3.0)
    parser.add_argument("--hard-timeout", type=float, default=35.0)
    parser.add_argument("--odom-start-timeout", type=float, default=8.0)
    parser.add_argument("--odom-stale", type=float, default=1.0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--log-period", type=float, default=0.5)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--arm-file", required=True)
    args, ros_args = parser.parse_known_args()
    if not (0.0 < args.speed <= 0.10):
        raise SystemExit("unsafe speed")
    if not (0.0 < args.max_distance <= 3.0):
        raise SystemExit("unsafe distance")
    if not (0.0 < args.hard_timeout <= 35.0):
        raise SystemExit("unsafe timeout")
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    rclpy.init(args=ros_args)
    node = DriveGuard(args)
    code = 1
    try:
        code = node.run()
    except Exception as exc:
        node.stop_reason = "test_failure"
        node._event("guard_exception", error=type(exc).__name__)
        node.publish_zero(5.0)
        code = 1
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
