#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

try:
    from ai_msgs.msg import PerceptionTargets
    AI_MSGS_AVAILABLE = True
except Exception:
    AI_MSGS_AVAILABLE = False
    PerceptionTargets = None


class YoloBoxViewer(Node):
    def __init__(self):
        super().__init__('yolo_box_viewer')

        self.declare_parameter('image_topic', '/image_out')
        self.declare_parameter('detection_topic', '/hobot_dnn_detection')
        self.declare_parameter('show_window', True)
        self.declare_parameter('save_debug_image', False)
        self.declare_parameter('debug_image_path', '/tmp/yolov8_test_mplus0_debug.jpg')

        self.image_topic = self.get_parameter('image_topic').value
        self.detection_topic = self.get_parameter('detection_topic').value
        self.show_window = self.get_parameter('show_window').value
        self.save_debug_image = self.get_parameter('save_debug_image').value
        self.debug_image_path = self.get_parameter('debug_image_path').value

        self.bridge = CvBridge()
        self.latest_targets = []

        self.class_names = {
            0: 'QR_code',
            1: 'line',
            2: 'end',
            3: 'roadblock',
            4: 'yellow_green_boundary',
            5: 'item_area',
            6: 'sign_board',
        }

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        if AI_MSGS_AVAILABLE:
            self.det_sub = self.create_subscription(
                PerceptionTargets,
                self.detection_topic,
                self.detection_callback,
                10
            )
            self.get_logger().info('Subscribed detection topic as ai_msgs/msg/PerceptionTargets')
        else:
            self.det_sub = None
            self.get_logger().warn(
                'ai_msgs is not available in Python. '
                'Image display will work, but detection boxes cannot be parsed yet.'
            )

        self.get_logger().info(f'Image topic: {self.image_topic}')
        self.get_logger().info(f'Detection topic: {self.detection_topic}')
        self.get_logger().info('YoloBoxViewer started.')

    def detection_callback(self, msg):
        targets = []

        for target in msg.targets:
            target_type = getattr(target, 'type', '')
            class_id = self.parse_class_id(target_type)

            rois = getattr(target, 'rois', [])
            for roi in rois:
                rect = getattr(roi, 'rect', None)
                if rect is None:
                    continue

                x = int(getattr(rect, 'x_offset', 0))
                y = int(getattr(rect, 'y_offset', 0))
                w = int(getattr(rect, 'width', 0))
                h = int(getattr(rect, 'height', 0))

                confidence = float(getattr(roi, 'confidence', 0.0))

                if w <= 0 or h <= 0:
                    continue

                label = self.class_names.get(class_id, target_type if target_type else str(class_id))

                targets.append({
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'label': label,
                    'score': confidence,
                })

        self.latest_targets = targets

        if len(targets) > 0:
            text = ', '.join([f"{t['label']}:{t['score']:.2f}" for t in targets])
            self.get_logger().info(f'Detected: {text}')

    def parse_class_id(self, target_type):
        if target_type is None:
            return -1

        text = str(target_type)

        if text.isdigit():
            return int(text)

        for class_id, name in self.class_names.items():
            if text == name:
                return class_id

        return -1

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        for target in self.latest_targets:
            x = target['x']
            y = target['y']
            w = target['w']
            h = target['h']
            label = target['label']
            score = target['score']

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame.shape[1] - 1, x + w)
            y2 = min(frame.shape[0] - 1, y + h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = f'{label} {score:.2f}'
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
            'yolov8_test_mplus0',
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        if self.save_debug_image:
            cv2.imwrite(self.debug_image_path, frame)

        if self.show_window:
            cv2.imshow('YOLOv8 Test Mplus0', frame)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YoloBoxViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
