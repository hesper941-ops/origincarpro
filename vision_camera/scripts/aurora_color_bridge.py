#!/usr/bin/env python3
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage


class AuroraColorBridge(Node):
    def __init__(self):
        super().__init__('aurora_color_bridge')

        self.declare_parameter('color_topic', '')
        self.declare_parameter('frame_id', 'aurora_color_frame')
        self.declare_parameter('jpeg_quality', 85)

        self.color_topic = self.get_parameter('color_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)

        self.bridge = CvBridge()

        self.image_pub = self.create_publisher(Image, '/image_out', 10)
        self.compressed_pub = self.create_publisher(CompressedImage, '/image_out/compressed', 10)
        self.codec_input_pub = self.create_publisher(CompressedImage, '/image', 10)

        self.sub = None

        if self.color_topic:
            self.subscribe_color_topic(self.color_topic)
        else:
            self.get_logger().info('No color_topic set, auto searching Aurora color Image topic...')
            self.timer = self.create_timer(1.0, self.auto_select_color_topic)

    def auto_select_color_topic(self):
        topics = self.get_topic_names_and_types()

        candidates = []
        for name, types in topics:
            if 'sensor_msgs/msg/Image' not in types:
                continue

            lower = name.lower()
            if any(k in lower for k in ['depth', 'ir', 'infra', 'point']):
                continue
            if any(k in lower for k in ['color', 'rgb', 'image_raw', 'image']):
                candidates.append(name)

        if not candidates:
            self.get_logger().warn('No suitable Aurora color Image topic found yet...')
            return

        # Prefer names containing color/rgb
        candidates.sort(key=lambda n: (('color' not in n.lower()) and ('rgb' not in n.lower()), len(n)))
        topic = candidates[0]

        self.timer.cancel()
        self.subscribe_color_topic(topic)

    def subscribe_color_topic(self, topic):
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.color_topic = topic
        self.sub = self.create_subscription(Image, topic, self.image_callback, qos)
        self.get_logger().info(f'Aurora color bridge subscribed: {topic}')
        self.get_logger().info('Publishing: /image_out, /image_out/compressed, /image')

    def image_callback(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Failed to convert image to bgr8: {exc}')
            return

        header = msg.header
        if not header.frame_id:
            header.frame_id = self.frame_id

        image_msg = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
        image_msg.header = header
        self.image_pub.publish(image_msg)

        ok, jpg = cv2.imencode(
            '.jpg',
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            self.get_logger().warn('Failed to encode JPEG')
            return

        compressed_msg = CompressedImage()
        compressed_msg.header = header
        compressed_msg.format = 'jpeg'
        compressed_msg.data = jpg.tobytes()

        # /image_out/compressed: for display / YOLO box viewer background
        self.compressed_pub.publish(compressed_msg)

        # /image: for hobot_codec input, then codec publishes /hbmem_img
        self.codec_input_pub.publish(compressed_msg)


def main():
    rclpy.init()
    node = AuroraColorBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
