#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class RGBDatasetCollector(Node):
    """按固定频率从深度相机 RGB 话题保存 JPEG 图片。"""

    def __init__(self) -> None:
        super().__init__("person_board_rgb_dataset_collector")

        self.declare_parameter("image_topic", "/rgb/image_raw")
        self.declare_parameter(
            "output_dir",
            "/root/intelligent_car_ws/datasets/person_board/raw/test_run",
        )
        self.declare_parameter("save_fps", 2.0)
        self.declare_parameter("max_images", 200)
        self.declare_parameter("jpeg_quality", 95)
        self.declare_parameter("warmup_sec", 2.0)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.output_dir = Path(str(self.get_parameter("output_dir").value))
        self.save_fps = float(self.get_parameter("save_fps").value)
        self.max_images = int(self.get_parameter("max_images").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.warmup_sec = float(self.get_parameter("warmup_sec").value)

        if self.save_fps <= 0:
            raise ValueError("save_fps 必须大于 0")
        if self.max_images <= 0:
            raise ValueError("max_images 必须大于 0")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()
        self.start_monotonic = time.monotonic()
        self.last_save_monotonic = 0.0
        self.saved_count = 0
        self.received_count = 0
        self.finished = False

        self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(f"订阅 RGB 话题：{self.image_topic}")
        self.get_logger().info(f"输出目录：{self.output_dir}")
        self.get_logger().info(
            f"保存频率：{self.save_fps:.2f} FPS，最大数量：{self.max_images}"
        )

    def image_callback(self, msg: Image) -> None:
        if self.finished:
            return

        self.received_count += 1
        now = time.monotonic()

        if now - self.start_monotonic < self.warmup_sec:
            return

        if now - self.last_save_monotonic < 1.0 / self.save_fps:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"RGB 图像转换失败：{exc}")
            return

        if frame is None or frame.size == 0:
            self.get_logger().warning("收到空图像，已跳过")
            return

        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(
            msg.header.stamp.nanosec
        )
        filename = f"frame_{self.saved_count:06d}_{stamp_ns}.jpg"
        output_path = self.output_dir / filename

        ok = cv2.imwrite(
            str(output_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            self.get_logger().error(f"保存失败：{output_path}")
            return

        self.saved_count += 1
        self.last_save_monotonic = now

        if self.saved_count % 10 == 0 or self.saved_count == 1:
            self.get_logger().info(
                f"已保存 {self.saved_count}/{self.max_images} 张，"
                f"共接收 {self.received_count} 帧"
            )

        if self.saved_count >= self.max_images:
            self.finished = True
            self.get_logger().info("达到最大图片数，采集完成。")
            rclpy.shutdown()


def main() -> None:
    rclpy.init()
    node = None
    try:
        node = RGBDatasetCollector()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().error(f"采集异常：{exc}")
        else:
            print(f"初始化异常：{exc}")
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
