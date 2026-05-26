#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.node import Node


class PerspectiveNode(Node):
    def __init__(self):
        super().__init__('perspective_node')

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Camera open failed")
            return

        self.src = np.float32([
            [264, 427],
            [335, 422],
            [315, 325],
            [271, 329]
        ])
        self.timer = self.create_timer(0.03, self.loop)

    def loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Frame read failed")
            return

        h, w = frame.shape[:2]
        dst = np.float32([
            [0, h],
            [w, h],
            [w, 0],
            [0, 0]
        ])

        matrix = cv2.getPerspectiveTransform(self.src, dst)
        bird = cv2.warpPerspective(frame, matrix, (w, h))

        for p in self.src:
            cv2.circle(frame, (int(p[0]), int(p[1])), 8, (0, 0, 255), -1)

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
