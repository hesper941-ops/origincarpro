#!/usr/bin/env python3

import cv2
import cv_bridge
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Pose2D
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Float32
from ai_msgs.msg import PerceptionTargets

def get_bool_param(value):
    if isinstance(value, str):
        return value.lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


class PerspectiveNode(Node):
    def __init__(self):
        super().__init__('perspective_node')

        self.bridge = cv_bridge.CvBridge()
        self.input_topic = self.declare_parameter('input_topic', '/image_out').value
        self.output_topic = self.declare_parameter('output_topic', '/bird_view/image').value
        self.compressed_output_topic = self.declare_parameter(
            'compressed_output_topic',
            '/bird_view/image/compressed'
        ).value
        self.declare_parameter('perspective_config', '')
        self.publish_compressed = get_bool_param(
            self.declare_parameter('publish_compressed', True).value
        )
        self.show_window = get_bool_param(
            self.declare_parameter('show_window', False).value
        )
        self.enable_dummy_output = get_bool_param(
            self.declare_parameter('enable_dummy_output', False).value
        )

        self.src, self.dst, self.output_size = self.load_perspective_config()
        self.matrix = cv2.getPerspectiveTransform(self.src, self.dst)
        # 保存最新检测到的 end 中心
        self.latest_end = None

        # YOLO检测结果
        self.det_sub = self.create_subscription(
            PerceptionTargets,
            '/hobot_dnn_detection',
            self.det_callback,
            10
        )


        self.image_sub = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            10
        )
        self.image_pub = self.create_publisher(Image, self.output_topic, 10)
        self.compressed_pub = self.create_publisher(
            CompressedImage,
            self.compressed_output_topic,
            10
        )
        self.p_valid_pub = self.create_publisher(Bool, '/bird_view/p_point_valid', 10)
        self.p_pose_pub = self.create_publisher(Pose2D, '/bird_view/p_point_pose', 10)
        self.line_error_pub = self.create_publisher(Float32, '/bird_view/line_error', 10)
        self.heading_error_pub = self.create_publisher(Float32, '/bird_view/heading_error', 10)

        self.interface_timer = self.create_timer(0.5, self.publish_return_interface)
        

    def load_perspective_config(self):
        default_src = np.float32([
            [264, 427],
            [335, 422],
            [315, 325],
            [271, 329],
        ])
        default_size = [400, 400]
        default_dst = np.float32([
            [0, default_size[1]],
            [default_size[0], default_size[1]],
            [default_size[0], 0],
            [0, 0],
        ])
        path = self.get_parameter('perspective_config').value
        if not path:
            return default_src, default_dst, default_size

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            src_points = data.get('src_points', default_src.tolist())
            output_size = data.get('output_size', default_size)
            dst_points = data.get('dst_points')
            if len(src_points) != 4:
                raise ValueError('src_points must contain four points')
            if dst_points is None:
                dst_points = [
                    [0, output_size[1]],
                    [output_size[0], output_size[1]],
                    [output_size[0], 0],
                    [0, 0],
                ]
            if len(dst_points) != 4:
                raise ValueError('dst_points must contain four points')
            return np.float32(src_points), np.float32(dst_points), output_size
        except Exception as exc:
            self.get_logger().warn(f'Failed to load perspective config {path}: {exc}')
            return default_src, default_dst, default_size

    def det_callback(self, msg):

        self.latest_end = None

        for target in msg.targets:

            if target.type != "end":
                continue

            if len(target.rois) == 0:
                continue

            roi = target.rois[0]

            x = roi.rect.x_offset
            y = roi.rect.y_offset
            w = roi.rect.width
            h = roi.rect.height

            cx = x + w * 0.5
            cy = y + h * 0.5

            self.latest_end = (cx, cy)

            break

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'CV Bridge Error: {exc}')
            return

        out_width = int(self.output_size[0]) if self.output_size else frame.shape[1]
        out_height = int(self.output_size[1]) if self.output_size else frame.shape[0]
        bird = cv2.warpPerspective(frame, self.matrix, (out_width, out_height))

        p_valid = False
        p_pose = Pose2D()

        if self.latest_end is not None:

            cx, cy = self.latest_end

            src_pt = np.array(
                [[[cx, cy]]],
                dtype=np.float32
            )

            bird_pt = cv2.perspectiveTransform(
                src_pt,
                self.matrix
            )

            u = bird_pt[0][0][0]
            v = bird_pt[0][0][1]

            cv2.circle(
                bird,
                (int(u), int(v)),
                8,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                bird,
                "P",
                (int(u) + 10, int(v)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )      
            car_u = 320     #逆透视框中底部中心坐标x轴
            car_v = 475     #逆透视框中底部中心坐标y轴

            scale = 0.01 / 219      #逆透视比例尺

            dx_pixel = u - car_u
            dy_pixel = car_v - v

            p_pose.x = dy_pixel * scale + 0.085      #逆透视框中底部中心与车前端距离8.5cm
            p_pose.y = dx_pixel * scale             #这里单位是米


            p_valid = True

        if self.show_window:
            cv2.imshow('bird_view', bird)
            cv2.waitKey(1)

        bird_msg = self.bridge.cv2_to_imgmsg(bird, encoding='bgr8')
        bird_msg.header = msg.header
        self.image_pub.publish(bird_msg)

        if self.publish_compressed:
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(bird, dst_format='jpg')
            compressed_msg.header = msg.header
            self.compressed_pub.publish(compressed_msg)

        self.p_valid_pub.publish(
            Bool(data=p_valid)
        )

        self.p_pose_pub.publish(
            p_pose
        )

    def publish_return_interface(self):
        self.line_error_pub.publish(Float32(data=0.0))
        self.heading_error_pub.publish(Float32(data=0.0))


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
