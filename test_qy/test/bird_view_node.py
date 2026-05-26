#!/usr/bin/env python3
#没订阅话题的,有节点

import rclpy
from rclpy.node import Node
import cv2
import numpy as np


class PerspectiveNode(Node):

    def __init__(self):
        super().__init__('perspective_node')

        # 打开摄像头
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.get_logger().error("Camera open failed")
            return

        # 原图4个点
        self.src = np.float32([
            [264, 427],  # 左下
            [335, 422],  # 右下
            [315, 325],  # 右上
            [271, 329]   # 左上
        ])

        # 定时器（30Hz）
        self.timer = self.create_timer(0.03, self.loop)

        self.get_logger().info("Perspective node started")

    def loop(self):

        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warn("Frame read failed")
            return

        h, w = frame.shape[:2]

        # 目标俯视图4点
        dst = np.float32([
            [0, h],
            [w, h],
            [w, 0],
            [0, 0]
        ])

        # 计算透视矩阵
        M = cv2.getPerspectiveTransform(self.src, dst)

        # 透视变换
        bird = cv2.warpPerspective(frame, M, (w, h))

        # 原图画点
        for p in self.src:
            cv2.circle(frame, (int(p[0]), int(p[1])), 8, (0, 0, 255), -1)

        # 显示
        cv2.imshow("Original", frame)
        cv2.imshow("Bird View", bird)

        cv2.waitKey(1)

    def destroy_node(self):

        self.cap.release()
        cv2.destroyAllWindows()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = PerspectiveNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()