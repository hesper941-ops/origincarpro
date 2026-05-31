#!/usr/bin/env python3

import math
import os

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class SemanticMapPreviewNode(Node):
    def __init__(self):
        super().__init__('semantic_map_preview_node')

        self.declare_parameter('semantic_map_file', '')
        self.declare_parameter('odom_topic', '/odom_combined')
        self.declare_parameter('goal_topic', '/current_goal')
        self.declare_parameter('preview_image_topic', '/semantic_map/preview')
        self.declare_parameter('preview_compressed_topic', '/semantic_map/preview/compressed')
        self.declare_parameter('preview_image_size', 800)
        self.declare_parameter('preview_publish_rate', 1.0)
        self.declare_parameter('jpeg_quality', 85)

        self.image_size = int(self.get_parameter('preview_image_size').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.semantic_map = self.load_semantic_map()
        self.points = self.semantic_map.get('points', {})
        self.routes = self.semantic_map.get('routes', {})
        self.boundary = self.semantic_map.get('boundary', {})

        self.current_goal = None
        self.current_pose = None
        self.current_yaw = 0.0

        self.image_pub = self.create_publisher(
            Image,
            self.get_parameter('preview_image_topic').value,
            10,
        )
        self.compressed_pub = self.create_publisher(
            CompressedImage,
            self.get_parameter('preview_compressed_topic').value,
            10,
        )

        self.create_subscription(
            PoseStamped,
            self.get_parameter('goal_topic').value,
            self.goal_callback,
            10,
        )
        self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self.odom_callback,
            10,
        )

        if cv2 is None or np is None:
            self.get_logger().warn('OpenCV is not available; semantic map preview images will not be published')

        rate = float(self.get_parameter('preview_publish_rate').value)
        self.timer = self.create_timer(1.0 / max(rate, 0.1), self.publish_preview)

    def load_semantic_map(self):
        path = self.get_parameter('semantic_map_file').value
        if not path:
            try:
                from ament_index_python.packages import get_package_share_directory
                path = os.path.join(
                    get_package_share_directory('origincar_nav'),
                    'config',
                    'semantic_map.yaml',
                )
            except Exception:
                path = os.path.join(os.getcwd(), 'origincar_nav', 'config', 'semantic_map.yaml')

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            self.get_logger().info(f'Loaded semantic map preview file: {path}')
            return data
        except Exception as exc:
            self.get_logger().warn(f'Failed to load semantic map preview file {path}: {exc}')
            return {'points': {}, 'routes': {}, 'boundary': {}}

    def goal_callback(self, msg):
        self.current_goal = (msg.pose.position.x, msg.pose.position.y)

    def odom_callback(self, msg):
        pose = msg.pose.pose
        self.current_pose = (pose.position.x, pose.position.y)
        self.current_yaw = yaw_from_quaternion(pose.orientation)

    def publish_preview(self):
        if cv2 is None or np is None:
            return

        image = np.full((self.image_size, self.image_size, 3), 255, dtype=np.uint8)
        transform = self.build_transform()

        self.draw_boundary(image, transform)
        self.draw_routes(image, transform)
        self.draw_points(image, transform)
        self.draw_goal(image, transform)
        self.draw_pose(image, transform)
        self.draw_overlay_text(image)

        stamp = self.get_clock().now().to_msg()
        self.image_pub.publish(self.to_image_msg(image, stamp))
        compressed = self.to_compressed_msg(image, stamp)
        if compressed is not None:
            self.compressed_pub.publish(compressed)

    def build_transform(self):
        xs = []
        ys = []
        for cfg in self.points.values():
            pose = cfg.get('pose', [])
            if len(pose) >= 2:
                xs.append(float(pose[0]))
                ys.append(float(pose[1]))

        for key in ('x_min', 'x_max'):
            if key in self.boundary:
                xs.append(float(self.boundary[key]))
        for key in ('y_min', 'y_max'):
            if key in self.boundary:
                ys.append(float(self.boundary[key]))

        if self.current_goal is not None:
            xs.append(float(self.current_goal[0]))
            ys.append(float(self.current_goal[1]))
        if self.current_pose is not None:
            xs.append(float(self.current_pose[0]))
            ys.append(float(self.current_pose[1]))

        if not xs or not ys:
            xs = [0.0, 1.0]
            ys = [0.0, 1.0]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if abs(x_max - x_min) < 1e-6:
            x_max = x_min + 1.0
        if abs(y_max - y_min) < 1e-6:
            y_max = y_min + 1.0

        margin = max(40, int(self.image_size * 0.08))
        scale = min(
            (self.image_size - 2 * margin) / (x_max - x_min),
            (self.image_size - 2 * margin) / (y_max - y_min),
        )

        def world_to_pixel(x, y):
            px = int(margin + (float(x) - x_min) * scale)
            py = int(self.image_size - margin - (float(y) - y_min) * scale)
            return px, py

        return world_to_pixel

    def draw_boundary(self, image, transform):
        keys = ('x_min', 'x_max', 'y_min', 'y_max')
        if not all(key in self.boundary for key in keys):
            return
        x_min = float(self.boundary['x_min'])
        x_max = float(self.boundary['x_max'])
        y_min = float(self.boundary['y_min'])
        y_max = float(self.boundary['y_max'])
        p1 = transform(x_min, y_min)
        p2 = transform(x_max, y_max)
        top_left = (min(p1[0], p2[0]), min(p1[1], p2[1]))
        bottom_right = (max(p1[0], p2[0]), max(p1[1], p2[1]))
        cv2.rectangle(image, top_left, bottom_right, (40, 40, 40), 2)

    def draw_routes(self, image, transform):
        self.draw_route(image, transform, 'clockwise', (0, 150, 0))
        self.draw_route(image, transform, 'anticlockwise', (180, 120, 0))

    def draw_route(self, image, transform, route_name, color):
        route = self.routes.get(route_name, {})
        order = route.get('channel_order', [])
        pixel_points = []
        for name in order:
            pose = self.point_pose(name)
            if pose is not None:
                pixel_points.append(transform(pose[0], pose[1]))
        for start, end in zip(pixel_points, pixel_points[1:]):
            cv2.line(image, start, end, color, 2, lineType=cv2.LINE_AA)
        if pixel_points:
            cv2.putText(image, route_name, pixel_points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def draw_points(self, image, transform):
        colors = {
            'p_start': (0, 90, 255),
            'task_station': (255, 0, 0),
            'channel_entry': (0, 160, 160),
        }
        for name in (
            'p_start',
            'task_station',
            'channel_entry',
            'channel_p1',
            'channel_p2',
            'channel_p3',
            'channel_p4',
        ):
            pose = self.point_pose(name)
            if pose is None:
                continue
            pixel = transform(pose[0], pose[1])
            color = colors.get(name, (80, 80, 220))
            cv2.circle(image, pixel, 7, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(
                image,
                name,
                (pixel[0] + 8, pixel[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                lineType=cv2.LINE_AA,
            )

    def draw_goal(self, image, transform):
        if self.current_goal is None:
            return
        pixel = transform(self.current_goal[0], self.current_goal[1])
        cv2.drawMarker(image, pixel, (0, 0, 255), cv2.MARKER_STAR, 22, 2, line_type=cv2.LINE_AA)
        cv2.putText(image, 'current_goal', (pixel[0] + 10, pixel[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    def draw_pose(self, image, transform):
        if self.current_pose is None:
            return
        pixel = transform(self.current_pose[0], self.current_pose[1])
        heading_len = 28
        end = (
            int(pixel[0] + math.cos(self.current_yaw) * heading_len),
            int(pixel[1] - math.sin(self.current_yaw) * heading_len),
        )
        cv2.circle(image, pixel, 8, (0, 0, 0), 2, lineType=cv2.LINE_AA)
        cv2.arrowedLine(image, pixel, end, (0, 0, 0), 2, tipLength=0.3, line_type=cv2.LINE_AA)
        cv2.putText(image, 'car', (pixel[0] + 10, pixel[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def draw_overlay_text(self, image):
        goal_name = self.nearest_point_name(self.current_goal) if self.current_goal else 'none'
        if self.current_pose is None:
            pose_text = 'pose: none'
        else:
            pose_text = f'pose: {self.current_pose[0]:.2f}, {self.current_pose[1]:.2f}, {self.current_yaw:.2f}'
        lines = [
            'Semantic Map Preview',
            f'goal: {goal_name}',
            pose_text,
        ]
        for index, text in enumerate(lines):
            cv2.putText(
                image,
                text,
                (18, 30 + index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (20, 20, 20),
                2,
                lineType=cv2.LINE_AA,
            )

    def point_pose(self, name):
        cfg = self.points.get(name)
        if not cfg:
            return None
        pose = cfg.get('pose', [])
        if len(pose) < 2:
            return None
        return float(pose[0]), float(pose[1])

    def nearest_point_name(self, point):
        best_name = 'unknown'
        best_dist = None
        for name in self.points:
            pose = self.point_pose(name)
            if pose is None:
                continue
            dist = math.hypot(pose[0] - point[0], pose[1] - point[1])
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_name = name
        return best_name

    def to_image_msg(self, image, stamp):
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = 'semantic_map_preview'
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = int(image.shape[1] * 3)
        msg.data = image.tobytes()
        return msg

    def to_compressed_msg(self, image, stamp):
        ok, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            self.get_logger().warn('Failed to encode semantic map preview as JPEG')
            return None
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = 'semantic_map_preview'
        msg.format = 'jpeg'
        msg.data = encoded.tobytes()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapPreviewNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
