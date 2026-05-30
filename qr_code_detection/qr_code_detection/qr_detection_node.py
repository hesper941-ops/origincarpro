#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time

import cv2
import cv_bridge
import rclpy
from ament_index_python.packages import get_package_share_directory
from origincar_msg.msg import Sign
from pyzbar.pyzbar import decode
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String


class QrCodeDetection(Node):
    def __init__(self):
        super().__init__('qrcode_detect')
        self.bridge = cv_bridge.CvBridge()
        self.last_decoded_data = None

        # Subscribe to /image_out from vision_camera/hbm_image_bridge.
        self.image_sub = self.create_subscription(
            Image,
            '/image_out',
            self.image_callback,
            1,
        )

        self.pub_car_signal = self.create_publisher(Sign, '/sign_switch', 1)
        self.pub_qrcode_info = self.create_publisher(String, '/qrcode_detected/info_result', 1)
        self.qr_start_pub = self.create_publisher(Int32, '/close_signal', 10)

        model_path = os.path.join(get_package_share_directory('qr_code_detection'), 'model')
        self.detect_obj = cv2.wechat_qrcode_WeChatQRCode(
            os.path.join(model_path, 'detect.prototxt'),
            os.path.join(model_path, 'detect.caffemodel'),
            os.path.join(model_path, 'sr.prototxt'),
            os.path.join(model_path, 'sr.caffemodel'),
        )

    def handle_qr_data(self, qr_data):
        if qr_data and qr_data != self.last_decoded_data:
            self.get_logger().info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] QR: {qr_data}")
            self.last_decoded_data = qr_data
        return self.last_decoded_data

    def parse_sign_data(self, qr_text):
        normalized = str(qr_text).strip()
        lowered = normalized.lower()
        if lowered in ('clockwise', 'clock wise'):
            return 3
        if lowered in ('anticlockwise', 'anti-clockwise', 'anti clockwise', 'counterclockwise'):
            return 4
        try:
            value = int(normalized)
        except ValueError:
            self.get_logger().warn(f'Unknown QR content, publishing text only: {qr_text}')
            return None
        return 4 if value % 2 == 0 else 3

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'Failed to convert image for QR detection: {exc}')
            return

        qr_codes = decode(cv_image)
        if not qr_codes:
            return

        for qr in qr_codes:
            try:
                qr_data = qr.data.decode('utf-8')
            except UnicodeDecodeError:
                self.get_logger().warn('Failed to decode QR content as UTF-8')
                continue
            except Exception as exc:
                self.get_logger().warn(f'QR decode error: {exc}')
                continue

            latest = self.handle_qr_data(qr_data)
            if latest is None:
                continue

            self.pub_qrcode_info.publish(String(data=latest))

            sign_data = self.parse_sign_data(latest)
            if sign_data is not None:
                sign = Sign()
                sign.sign_data = sign_data
                self.pub_car_signal.publish(sign)

            self.qr_start_pub.publish(Int32(data=1))


def main(args=None):
    rclpy.init(args=args)
    node = QrCodeDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
