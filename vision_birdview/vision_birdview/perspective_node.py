#!/usr/bin/env python3

import cv2
import cv_bridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage


def get_bool_param(value):
    if isinstance(value, str):
        return value.lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


class PerspectiveNode(Node):
    def __init__(self):
        super().__init__('perspective_node')

        self.bridge = cv_bridge.CvBridge()

        # 话题参数：默认走统一图像入口 /image_out
        self.input_topic = self.declare_parameter('input_topic', '/image_out').value
        self.output_topic = self.declare_parameter('output_topic', '/bird_view/image').value
        self.compressed_output_topic = self.declare_parameter(
            'compressed_output_topic',
            '/bird_view/image/compressed'
        ).value

        self.publish_compressed = get_bool_param(
            self.declare_parameter('publish_compressed', True).value
        )

        # 默认不要 imshow，X5 / SSH / launch 环境下容易没有显示窗口
        self.show_window = get_bool_param(
            self.declare_parameter('show_window', False).value
        )

        self.image_width = int(self.declare_parameter('image_width', 640).value)
        self.image_height = int(self.declare_parameter('image_height', 480).value)

        self.declare_parameter(
            'src_points',
            [
                257.0, 391.0,
                315.0, 395.0,
                307.0, 277.0,
                278.0, 275.0
            ]
        )

        self.declare_parameter(
            'dst_points',
            [
                0.0, 480.0,
                640.0, 480.0,
                640.0, 0.0,
                0.0, 0.0
            ]
        )

        src_points = self.get_parameter('src_points').value
        dst_points = self.get_parameter('dst_points').value

        self.src = np.array(src_points, dtype=np.float32).reshape((4, 2))
        self.dst = np.array(dst_points, dtype=np.float32).reshape((4, 2))
        self.matrix = cv2.getPerspectiveTransform(self.src, self.dst)

        self.image_sub = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            10
        )

        self.image_pub = self.create_publisher(
            Image,
            self.output_topic,
            10
        )

        self.compressed_pub = self.create_publisher(
            CompressedImage,
            self.compressed_output_topic,
            10
        )

        self.get_logger().info('Perspective node started')
        self.get_logger().info(f'input_topic: {self.input_topic}')
        self.get_logger().info(f'output_topic: {self.output_topic}')
        self.get_logger().info(f'compressed_output_topic: {self.compressed_output_topic}')
        self.get_logger().info(f'publish_compressed: {self.publish_compressed}')
        self.get_logger().info(f'show_window: {self.show_window}')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge Error: {e}')
            return

        out_w = self.image_width if self.image_width > 0 else frame.shape[1]
        out_h = self.image_height if self.image_height > 0 else frame.shape[0]

        bird = cv2.warpPerspective(frame, self.matrix, (out_w, out_h))

        if self.show_window:
            cv2.imshow('bird_view', bird)
            cv2.waitKey(1)

        bird_msg = self.bridge.cv2_to_imgmsg(bird, encoding='bgr8')
        bird_msg.header = msg.header
        self.image_pub.publish(bird_msg)

        if self.publish_compressed:
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(
                bird,
                dst_format='jpg'
            )
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
        if node.show_window:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
