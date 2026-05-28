#!/usr/bin/env python3
#连yaml文件
import cv2
import cv_bridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage

class PerspectiveNode(Node):

    def __init__(self):
        super().__init__('perspective_node')

        self.bridge = cv_bridge.CvBridge()

        publish_compressed = self.declare_parameter('publish_compressed',True).value
        if isinstance(publish_compressed, str):
            self.publish_compressed = (publish_compressed.lower()in ('1', 'true', 'yes', 'on'))
        else:
            self.publish_compressed = bool(publish_compressed)

        self.declare_parameter(
            'src_points',
            [
                257.0, 391.0,
                315.0, 395.0,
                307.0, 277.0,
                278.0, 275.0
            ]
        )

        src_points = self.get_parameter('src_points').value
        self.src = np.array(src_points,dtype=np.float32).reshape((4, 2))
        self.declare_parameter(
            'dst_points',
            [
                0.0, 480.0,
                640.0, 480.0,
                640.0, 0.0,
                0.0, 0.0
            ]
        )

        dst_points = self.get_parameter('dst_points').value
        self.dst = np.array(dst_points,dtype=np.float32).reshape((4, 2))
        self.image_sub = self.create_subscription(Image,'/aurora/rgb/image_raw',self.image_callback,10,)
        self.image_pub = self.create_publisher(Image,'/bird_view/image',10)

        self.compressed_pub = self.create_publisher(CompressedImage,'/bird_view/image/compressed',10)
        self.get_logger().info('Perspective node started')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2( msg,desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge Error: {e}')
            return
        height, width = frame.shape[:2]
        matrix = cv2.getPerspectiveTransform(
            self.src,
            self.dst
        )
        bird = cv2.warpPerspective(frame,matrix,(width, height))
        cv2.imshow("bird_view", bird)
        cv2.waitKey(1)
        bird_msg = self.bridge.cv2_to_imgmsg(bird,encoding='bgr8')
        bird_msg.header = msg.header
        self.image_pub.publish(bird_msg)
        if self.publish_compressed:
            compressed_msg = (self.bridge.cv2_to_compressed_imgmsg(bird,dst_format='jpg'))
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
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()