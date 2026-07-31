#!/usr/bin/env python3
"""
安全定速直行测试节点

功能：
1. 等待 /cmd_vel 出现订阅者；
2. 启动前持续发送零速度；
3. 平滑加速到指定速度；
4. 定速运行指定时间；
5. 平滑减速并持续发送零速度；
6. Ctrl+C / SIGTERM / 异常时优先发送停车指令。

示例：
python3 safe_constant_speed_drive.py \
  --topic /cmd_vel \
  --speed 0.03 \
  --duration 3 \
  --delay 3
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

try:
    from rclpy.signals import SignalHandlerOptions
except ImportError:  # 兼容少数旧版 rclpy
    SignalHandlerOptions = None


STOP_REQUESTED = False


def _request_stop(signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\n[安全] 收到信号 {signum}，准备停车……", flush=True)


class SafeConstantSpeedDrive(Node):
    def __init__(
        self,
        topic: str,
        publish_rate_hz: float,
    ) -> None:
        super().__init__("safe_constant_speed_drive")
        self.publisher = self.create_publisher(Twist, topic, 10)
        self.topic = topic
        self.publish_rate_hz = publish_rate_hz
        self.period_sec = 1.0 / publish_rate_hz

    @staticmethod
    def build_twist(linear_x: float) -> Twist:
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        return msg

    def spin_discovery_once(self, timeout_sec: float = 0.05) -> None:
        if rclpy.ok():
            rclpy.spin_once(self, timeout_sec=timeout_sec)

    def wait_for_subscriber(self, timeout_sec: float) -> bool:
        start = time.monotonic()

        while not STOP_REQUESTED:
            self.spin_discovery_once(0.05)
            count = self.publisher.get_subscription_count()

            if count > 0:
                self.get_logger().info(
                    f"{self.topic} 已发现 {count} 个订阅者。"
                )
                return True

            if time.monotonic() - start >= timeout_sec:
                return False

            time.sleep(0.05)

        return False

    def publish_for(
        self,
        speed: float,
        duration_sec: float,
        label: str,
    ) -> None:
        if duration_sec <= 0.0:
            return

        msg = self.build_twist(speed)
        start = time.monotonic()
        next_tick = start

        while not STOP_REQUESTED:
            now = time.monotonic()
            if now - start >= duration_sec:
                break

            self.publisher.publish(msg)
            self.spin_discovery_once(0.0)

            next_tick += self.period_sec
            sleep_sec = next_tick - time.monotonic()
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
            else:
                next_tick = time.monotonic()

        self.get_logger().info(
            f"{label}结束：速度={speed:.3f} m/s，"
            f"计划时长={duration_sec:.2f} s"
        )

    def publish_ramp(
        self,
        start_speed: float,
        end_speed: float,
        duration_sec: float,
        label: str,
    ) -> None:
        if duration_sec <= 0.0:
            self.publisher.publish(self.build_twist(end_speed))
            return

        start = time.monotonic()
        next_tick = start

        while not STOP_REQUESTED:
            elapsed = time.monotonic() - start
            if elapsed >= duration_sec:
                break

            ratio = max(0.0, min(1.0, elapsed / duration_sec))
            speed = start_speed + (end_speed - start_speed) * ratio
            self.publisher.publish(self.build_twist(speed))
            self.spin_discovery_once(0.0)

            next_tick += self.period_sec
            sleep_sec = next_tick - time.monotonic()
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
            else:
                next_tick = time.monotonic()

        self.publisher.publish(self.build_twist(end_speed))
        self.get_logger().info(f"{label}结束。")

    def emergency_stop(self, hold_sec: float) -> None:
        """
        持续发送零速度。
        即使 STOP_REQUESTED=True，也必须完整执行停车保持时间。
        """
        zero = self.build_twist(0.0)
        hold_sec = max(hold_sec, 1.0)

        self.get_logger().warning(
            f"持续发送零速度 {hold_sec:.1f} 秒……"
        )

        start = time.monotonic()
        next_tick = start

        while time.monotonic() - start < hold_sec:
            try:
                self.publisher.publish(zero)
                self.spin_discovery_once(0.0)
            except Exception as exc:
                print(f"[停车警告] 发布零速度异常：{exc}", flush=True)

            next_tick += self.period_sec
            sleep_sec = next_tick - time.monotonic()
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
            else:
                next_tick = time.monotonic()

        # 再补发若干次，降低最后一条消息丢失的风险。
        for _ in range(10):
            try:
                self.publisher.publish(zero)
                self.spin_discovery_once(0.0)
            except Exception:
                pass
            time.sleep(0.02)

        self.get_logger().warning("零速度发送完成。")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="RDK X5 智能车安全定速直行测试"
    )
    parser.add_argument("--topic", default="/cmd_vel")
    parser.add_argument("--speed", type=float, default=0.03)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--ramp", type=float, default=0.5)
    parser.add_argument("--pre-stop", type=float, default=1.0)
    parser.add_argument("--stop-hold", type=float, default=3.0)
    parser.add_argument("--subscriber-timeout", type=float, default=10.0)
    parser.add_argument(
        "--max-abs-speed",
        type=float,
        default=0.20,
        help="安全速度上限，超过时拒绝运行",
    )
    return parser.parse_known_args()


def validate_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.speed):
        raise ValueError("speed 必须是有限数值")
    if abs(args.speed) > args.max_abs_speed:
        raise ValueError(
            f"请求速度 {args.speed:.3f} m/s 超过安全上限 "
            f"{args.max_abs_speed:.3f} m/s"
        )
    if args.duration <= 0.0:
        raise ValueError("duration 必须大于 0")
    if args.rate < 5.0:
        raise ValueError("rate 不应低于 5 Hz")
    if args.stop_hold < 1.0:
        raise ValueError("stop-hold 不应低于 1 秒")


def main() -> int:
    global STOP_REQUESTED

    args, ros_args = parse_args()

    try:
        validate_args(args)
    except ValueError as exc:
        print(f"[参数错误] {exc}", file=sys.stderr)
        return 2

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    init_kwargs = {"args": ros_args}
    if SignalHandlerOptions is not None:
        init_kwargs["signal_handler_options"] = SignalHandlerOptions.NO

    rclpy.init(**init_kwargs)

    node: Optional[SafeConstantSpeedDrive] = None

    try:
        node = SafeConstantSpeedDrive(
            topic=args.topic,
            publish_rate_hz=args.rate,
        )

        node.get_logger().info(
            f"目标话题={args.topic}，速度={args.speed:.3f} m/s，"
            f"定速时长={args.duration:.2f} s"
        )

        if not node.wait_for_subscriber(args.subscriber_timeout):
            node.get_logger().error(
                f"{args.topic} 在 {args.subscriber_timeout:.1f} 秒内"
                "没有发现订阅者，禁止启动小车。"
            )
            node.emergency_stop(args.stop_hold)
            return 1

        # 建立连接后先发送一段零速度。
        node.publish_for(
            speed=0.0,
            duration_sec=args.pre_stop,
            label="启动前零速度保持",
        )

        if STOP_REQUESTED:
            node.emergency_stop(args.stop_hold)
            return 130

        for remaining in range(int(math.ceil(args.delay)), 0, -1):
            if STOP_REQUESTED:
                break
            node.get_logger().info(f"{remaining} 秒后开始运动……")
            time.sleep(1.0)

        if STOP_REQUESTED:
            node.emergency_stop(args.stop_hold)
            return 130

        node.publish_ramp(
            start_speed=0.0,
            end_speed=args.speed,
            duration_sec=args.ramp,
            label="平滑加速",
        )

        if not STOP_REQUESTED:
            node.publish_for(
                speed=args.speed,
                duration_sec=args.duration,
                label="定速运行",
            )

        # 无论定速是否被 Ctrl+C 中断，都立即进入减速/停车。
        if not STOP_REQUESTED:
            node.publish_ramp(
                start_speed=args.speed,
                end_speed=0.0,
                duration_sec=args.ramp,
                label="平滑减速",
            )

        node.emergency_stop(args.stop_hold)

        if STOP_REQUESTED:
            return 130

        node.get_logger().info("测试完成，小车已发送停车指令。")
        return 0

    except Exception as exc:
        print(f"[运行异常] {exc}", file=sys.stderr)
        if node is not None:
            try:
                node.emergency_stop(max(args.stop_hold, 5.0))
            except Exception as stop_exc:
                print(
                    f"[严重警告] 异常停车发布失败：{stop_exc}",
                    file=sys.stderr,
                )
        return 1

    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
