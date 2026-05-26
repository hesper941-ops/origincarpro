#!/usr/bin/env python3

import cv2
import cv_bridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image


class PerspectiveNode(Node):
    def __init__(self):
        super().__init__('perspective_node')

        self.bridge = cv_bridge.CvBridge()
        publish_compressed = self.declare_parameter('publish_compressed', True).value
        if isinstance(publish_compressed, str):
            self.publish_compressed = publish_compressed.lower() in ('1', 'true', 'yes', 'on')
        else:
            self.publish_compressed = bool(publish_compressed)

        self.src = np.float32([
            [264, 427],
            [335, 422],
            [315, 325],
            [271, 329],
        ])

        self.image_sub = self.create_subscription(
            Image,
            '/image_out',
            self.image_callback,
            10,
        )
        self.image_pub = self.create_publisher(Image, '/bird_view/image', 10)
        self.compressed_pub = self.create_publisher(
            CompressedImage,
            '/bird_view/image/compressed',
            10,
        )

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width = frame.shape[:2]
        dst = np.float32([
            [0, height],
            [width, height],
            [width, 0],
            [0, 0],
        ])

        matrix = cv2.getPerspectiveTransform(self.src, dst)
        bird = cv2.warpPerspective(frame, matrix, (width, height))

        bird_msg = self.bridge.cv2_to_imgmsg(bird, encoding='bgr8')
        bird_msg.header = msg.header
        self.image_pub.publish(bird_msg)

        if self.publish_compressed:
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(bird, dst_format='jpg')
            compressed_msg.header = msg.header
            self.compressed_pub.publish(compressed_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PerspectiveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
