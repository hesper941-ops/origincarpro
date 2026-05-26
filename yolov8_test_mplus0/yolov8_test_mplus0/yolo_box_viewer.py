#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from ai_msgs.msg import PerceptionTargets


class YoloBoxViewer(Node):
    def __init__(self):
        super().__init__('yolo_box_viewer')

        self.declare_parameter('image_topic', '/image')
        self.declare_parameter('detection_topic', '/hobot_dnn_detection')
        self.declare_parameter('output_compressed_topic', '/yolov8_result/compressed')
        self.declare_parameter('jpeg_quality', 80)

        self.image_topic = self.get_parameter('image_topic').value
        self.detection_topic = self.get_parameter('detection_topic').value
        self.output_topic = self.get_parameter('output_compressed_topic').value
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)

        self.latest_targets = []

        self.image_sub = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            10
        )

        self.det_sub = self.create_subscription(
            PerceptionTargets,
            self.detection_topic,
            self.detection_callback,
            10
        )

        self.result_pub = self.create_publisher(
            CompressedImage,
            self.output_topic,
            10
        )

        self.get_logger().info(f'Input image topic: {self.image_topic}')
        self.get_logger().info(f'Detection topic: {self.detection_topic}')
        self.get_logger().info(f'Output compressed topic: {self.output_topic}')

    def detection_callback(self, msg):
        targets = []

        for target in msg.targets:
            target_type = str(target.type)

            for roi in target.rois:
                rect = roi.rect

                x = int(rect.x_offset)
                y = int(rect.y_offset)
                w = int(rect.width)
                h = int(rect.height)
                score = float(roi.confidence)

                if w <= 0 or h <= 0:
                    continue

                targets.append({
                    'label': target_type,
                    'score': score,
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                })

        self.latest_targets = targets

    def image_callback(self, msg):
        data = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn('Failed to decode compressed image')
            return

        for t in self.latest_targets:
            x1 = max(0, t['x'])
            y1 = max(0, t['y'])
            x2 = min(frame.shape[1] - 1, t['x'] + t['w'])
            y2 = min(frame.shape[0] - 1, t['y'] + t['h'])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = f"{t['label']} {t['score']:.2f}"
            cv2.putText(
                frame,
                text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            'YOLOv8 X5',
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        ok, encoded = cv2.imencode('.jpg', frame, encode_param)

        if not ok:
            self.get_logger().warn('Failed to encode result image')
            return

        out = CompressedImage()
        out.header = msg.header
        out.format = 'jpeg'
        out.data = encoded.tobytes()
        self.result_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = YoloBoxViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
