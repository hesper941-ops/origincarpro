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

        # =========================
        # 图像和检测话题参数
        # =========================
        self.declare_parameter('image_topic', '/image_out/compressed')
        self.declare_parameter('detection_topic', '/hobot_dnn_detection')
        self.declare_parameter('output_compressed_topic', '/yolov8_result/compressed')
        self.declare_parameter('jpeg_quality', 80)

        # =========================
        # 可视化参考参数
        # 这里只用于画触发线和显示文字，不控制小车
        # =========================
        self.declare_parameter('avoid_start_y_ratio', 0.65)
        self.declare_parameter('center_deadzone_ratio', 0.08)
        self.declare_parameter('obstacle_labels', ['roadblock'])

        self.image_topic = self.get_parameter('image_topic').value
        self.detection_topic = self.get_parameter('detection_topic').value
        self.output_topic = self.get_parameter('output_compressed_topic').value
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)

        self.avoid_start_y_ratio = float(
            self.get_parameter('avoid_start_y_ratio').value
        )
        self.center_deadzone_ratio = float(
            self.get_parameter('center_deadzone_ratio').value
        )

        self.obstacle_labels = {
            str(label).strip().lower()
            for label in self.get_parameter('obstacle_labels').value
        }

        # 保存最近一次 YOLO 检测结果
        self.latest_targets = []

        # =========================
        # 订阅图像
        # =========================
        self.image_sub = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            10
        )

        # =========================
        # 订阅 YOLO 检测结果
        # =========================
        self.det_sub = self.create_subscription(
            PerceptionTargets,
            self.detection_topic,
            self.detection_callback,
            10
        )

        # =========================
        # 发布画框后的图像
        # =========================
        self.result_pub = self.create_publisher(
            CompressedImage,
            self.output_topic,
            10
        )

        self.get_logger().info('yolo_box_viewer started')
        self.get_logger().info('Mode: visualization only')
        self.get_logger().info(f'Input image topic: {self.image_topic}')
        self.get_logger().info(f'Detection topic: {self.detection_topic}')
        self.get_logger().info(f'Output compressed topic: {self.output_topic}')
        self.get_logger().info('This node does NOT publish /cmd_vel, /yolo_cmd_vel, or /yolo_avoid_active')

    def detection_callback(self, msg):
        """
        接收 YOLO 检测结果，只保存目标信息，不做控制。
        """
        targets = []

        for target in msg.targets:
            target_type = str(target.type).strip()

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
        """
        接收压缩图像，在图像上画 YOLO 结果，然后重新发布。
        这个函数不发布任何速度控制。
        """
        data = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn('Failed to decode compressed image')
            return

        frame_h, frame_w = frame.shape[:2]

        frame_center_x = frame_w / 2.0
        avoid_start_y = frame_h * self.avoid_start_y_ratio
        deadzone = frame_w * self.center_deadzone_ratio

        # 记录最靠近车的 roadblock
        # 图像坐标中 y 越大，越靠近画面底部
        nearest_roadblock = None
        max_bottom_y = -1

        display_action = 'no roadblock'

        # =========================
        # 遍历 YOLO 检测目标
        # =========================
        for t in self.latest_targets:
            x1 = max(0, int(t['x']))
            y1 = max(0, int(t['y']))
            x2 = min(frame_w - 1, int(t['x'] + t['w']))
            y2 = min(frame_h - 1, int(t['y'] + t['h']))

            label = str(t['label']).strip()
            label_lower = label.lower()
            score = float(t['score'])

            # =========================
            # roadblock / 锥形筒：只画底部红线
            # =========================
            if label_lower in self.obstacle_labels:
                bottom_y = y2
                red_center_x = (x1 + x2) / 2.0

                # 记录最靠近车的 roadblock
                if bottom_y > max_bottom_y:
                    max_bottom_y = bottom_y
                    nearest_roadblock = {
                        'center_x': red_center_x,
                        'bottom_y': bottom_y,
                        'x1': x1,
                        'x2': x2,
                        'y1': y1,
                        'y2': y2,
                        'score': score,
                        'label': label,
                    }

                # 画 roadblock 底部红线
                cv2.line(
                    frame,
                    (x1, bottom_y),
                    (x2, bottom_y),
                    (0, 0, 255),
                    2
                )

                # 画红线中心点
                cv2.circle(
                    frame,
                    (int(red_center_x), int(bottom_y)),
                    5,
                    (0, 0, 255),
                    -1
                )

                text = f"{label} {score:.2f}"
                cv2.putText(
                    frame,
                    text,
                    (x1, max(20, bottom_y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            # =========================
            # 其他目标：正常画绿色框
            # =========================
            else:
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                text = f"{label} {score:.2f}"
                cv2.putText(
                    frame,
                    text,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # =========================
        # 画画面中心线
        # =========================
        cv2.line(
            frame,
            (int(frame_center_x), 0),
            (int(frame_center_x), frame_h),
            (255, 255, 0),
            1
        )

        # =========================
        # 画避障参考触发线
        # 注意：这里只是显示参考线，不控制小车
        # =========================
        cv2.line(
            frame,
            (0, int(avoid_start_y)),
            (frame_w, int(avoid_start_y)),
            (255, 0, 255),
            1
        )

        # =========================
        # 只生成显示文字，不发布速度
        # =========================
        if nearest_roadblock is not None:
            red_center_x = nearest_roadblock['center_x']
            red_bottom_y = nearest_roadblock['bottom_y']

            offset = red_center_x - frame_center_x

            if red_bottom_y >= avoid_start_y:
                if offset < -deadzone:
                    display_action = 'near obstacle: left side'
                elif offset > deadzone:
                    display_action = 'near obstacle: right side'
                else:
                    display_action = 'near obstacle: center'
            else:
                if offset < -deadzone:
                    display_action = 'far obstacle: left side'
                elif offset > deadzone:
                    display_action = 'far obstacle: right side'
                else:
                    display_action = 'far obstacle: center'

        # =========================
        # 显示调试信息
        # =========================
        cv2.putText(
            frame,
            'YOLOv8 X5 Viewer',
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            'mode: visualization only',
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f'obstacle: {display_action}',
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        if nearest_roadblock is not None:
            red_center_x = nearest_roadblock['center_x']
            red_bottom_y = nearest_roadblock['bottom_y']

            cv2.putText(
                frame,
                f'red_center_x: {int(red_center_x)}  bottom_y: {int(red_bottom_y)}',
                (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

        # =========================
        # 编码并发布可视化图像
        # =========================
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