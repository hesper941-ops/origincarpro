#!/usr/bin/env python3
import time
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class USBCameraPublisher(Node):
    def __init__(self):
        super().__init__("usb_camera_publisher")

        self.declare_parameter("device", "/dev/video0")
        self.declare_parameter("image_topic", "/image_out")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 15.0)
        self.declare_parameter("frame_id", "usb_camera")

        self.device = self.get_parameter("device").value
        self.image_topic = self.get_parameter("image_topic").value
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.fps = float(self.get_parameter("fps").value)
        self.frame_id = self.get_parameter("frame_id").value

        self.pub = self.create_publisher(Image, self.image_topic, 10)

        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open USB camera: {self.device}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        timer_period = 1.0 / max(self.fps, 1.0)
        self.timer = self.create_timer(timer_period, self.timer_cb)

        self.last_log = 0.0
        self.get_logger().info(f"USB camera opened: {self.device}")
        self.get_logger().info(f"Publish image topic: {self.image_topic}")
        self.get_logger().info(f"Resolution: {self.width}x{self.height}, fps={self.fps}")

    def timer_cb(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            now = time.time()
            if now - self.last_log > 1.0:
                self.get_logger().warn("Failed to read USB camera frame")
                self.last_log = now
            return

        h, w = frame.shape[:2]

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = h
        msg.width = w
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = w * 3
        msg.data = frame.tobytes()

        self.pub.publish(msg)

    def destroy_node(self):
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = None
    try:
        node = USBCameraPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
