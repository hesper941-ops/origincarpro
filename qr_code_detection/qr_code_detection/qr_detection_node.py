#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import cv_bridge
import rclpy
from ai_msgs.msg import PerceptionTargets
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
        self.latest_rois = []
        self.detection_stamp = None
        self.detection_receive_time = None
        self.last_stale_warn_time = 0.0
        self.declare_parameter('image_topic', '/image_out')
        self.declare_parameter('detection_topic', '/hobot_dnn_detection')
        self.declare_parameter('qr_labels', ['QR_code'])
        self.declare_parameter('max_detection_age_sec', 0.5)
        self.declare_parameter('publish_deprecated_outputs', False)
        self.declare_parameter('debug_allow_numeric_sign', False)
        self.qr_labels = {
            str(label).strip().lower()
            for label in self.get_parameter('qr_labels').value
        }
        self.max_detection_age_sec = float(self.get_parameter('max_detection_age_sec').value)
        self.publish_deprecated_outputs = bool(self.get_parameter('publish_deprecated_outputs').value)
        self.debug_allow_numeric_sign = bool(self.get_parameter('debug_allow_numeric_sign').value)

        self.image_sub = self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self.image_callback,
            1,
        )
        self.detection_sub = self.create_subscription(
            PerceptionTargets,
            self.get_parameter('detection_topic').value,
            self.detection_callback,
            10,
        )

        self.pub_car_signal = self.create_publisher(Sign, '/sign_switch', 1)
        self.pub_qrcode_info = self.create_publisher(String, '/qrcode_detected/info_result', 1)
        self.qr_start_pub = self.create_publisher(Int32, '/close_signal', 10)

    def handle_qr_data(self, qr_data):
        if qr_data and qr_data != self.last_decoded_data:
            self.get_logger().info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] QR: {qr_data}")
            self.last_decoded_data = qr_data
        return self.last_decoded_data

    def parse_sign_data(self, qr_text):
        normalized = str(qr_text).strip()
        lowered = normalized.lower()
        if lowered in ('clockwise', 'clock wise') or normalized in ('顺时针', '順時針'):
            return 3
        if (
            lowered in (
                'anticlockwise',
                'anti-clockwise',
                'anti clockwise',
                'counterclockwise',
                'counter-clockwise',
                'counter clockwise',
            )
            or normalized in ('逆时针', '逆時針')
        ):
            return 4
        if self.debug_allow_numeric_sign:
            try:
                value = int(normalized)
            except ValueError:
                pass
            else:
                self.get_logger().warn('Using debug numeric QR sign fallback; this is not the formal rule')
                return 4 if value % 2 == 0 else 3
        self.get_logger().warn(f'Unknown QR content, publishing text only: {qr_text}')
        return None

    def detection_callback(self, msg):
        rois = []
        for target in msg.targets:
            if str(target.type).strip().lower() not in self.qr_labels:
                continue
            for roi in target.rois:
                rect = roi.rect
                if rect.width <= 0 or rect.height <= 0:
                    continue
                rois.append((
                    int(rect.x_offset),
                    int(rect.y_offset),
                    int(rect.width),
                    int(rect.height),
                ))
        self.latest_rois = rois
        self.detection_stamp = self.msg_stamp_to_sec(msg)
        self.detection_receive_time = self.now_sec()

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'Failed to convert image for QR detection: {exc}')
            return

        if not self.latest_rois:
            return

        if self.rois_are_stale(msg):
            return

        height, width = cv_image.shape[:2]
        for x, y, w, h in self.latest_rois:
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)
            if x2 <= x1 or y2 <= y1:
                continue

            roi_image = cv_image[y1:y2, x1:x2]
            qr_codes = decode(roi_image)
            if not qr_codes:
                continue

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
                if self.publish_deprecated_outputs and sign_data is not None:
                    sign = Sign()
                    sign.sign_data = sign_data
                    self.pub_car_signal.publish(sign)

                if self.publish_deprecated_outputs:
                    self.qr_start_pub.publish(Int32(data=1))

    def rois_are_stale(self, image_msg):
        if self.detection_receive_time is None:
            return True

        image_stamp = self.msg_stamp_to_sec(image_msg)
        if image_stamp is not None and self.detection_stamp is not None:
            age = abs(image_stamp - self.detection_stamp)
        else:
            age = self.now_sec() - self.detection_receive_time

        if age <= self.max_detection_age_sec:
            return False

        now = self.now_sec()
        if now - self.last_stale_warn_time > 2.0:
            self.get_logger().warn(
                f'Skip QR ROI decode: detection/image timestamp gap {age:.3f}s '
                f'exceeds max_detection_age_sec={self.max_detection_age_sec:.3f}'
            )
            self.last_stale_warn_time = now
        return True

    def msg_stamp_to_sec(self, msg):
        stamp = getattr(getattr(msg, 'header', None), 'stamp', None)
        if stamp is None:
            return None
        sec = int(stamp.sec)
        nanosec = int(stamp.nanosec)
        if sec == 0 and nanosec == 0:
            return None
        return sec + nanosec * 1e-9

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9


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
