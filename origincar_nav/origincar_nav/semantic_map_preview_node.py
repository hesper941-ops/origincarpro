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
        self.declare_parameter('field_map_image', '')
        self.declare_parameter('odom_topic', '/odom_combined')
        self.declare_parameter('goal_topic', '/current_goal')
        self.declare_parameter('preview_image_topic', '/semantic_map/preview')
        self.declare_parameter('preview_compressed_topic', '/semantic_map/preview/compressed')
        self.declare_parameter('preview_image_size', 800)
        self.declare_parameter('preview_publish_rate', 1.0)
        self.declare_parameter('jpeg_quality', 85)
        self.declare_parameter('path_max_length', 500)

        self.image_size = int(self.get_parameter('preview_image_size').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.path_max_length = int(self.get_parameter('path_max_length').value)
        self.semantic_map = self.load_semantic_map()
        self.points = self.semantic_map.get('points', {})
        self.routes = self.semantic_map.get('routes', {})
        self.boundary = self.semantic_map.get('boundary', {})

        self.current_goal = None
        self.current_pose = None
        self.current_yaw = 0.0
        self.vehicle_path = []

        self.background = None
        if cv2 is None or np is None:
            self.get_logger().warn('OpenCV is not available; semantic map preview images will not be published')
        else:
            self.background = self.load_field_map_background()

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

        rate = float(self.get_parameter('preview_publish_rate').value)
        self.timer = self.create_timer(1.0 / max(rate, 0.1), self.publish_preview)

    def load_semantic_map(self):
        path = self.get_parameter('semantic_map_file').value
        if not path:
            path = self.package_file('config', 'semantic_map.yaml')

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            self.get_logger().info(f'Loaded semantic map preview file: {path}')
            return data
        except Exception as exc:
            self.get_logger().warn(f'Failed to load semantic map preview file {path}: {exc}')
            return {'points': {}, 'routes': {}, 'boundary': {}}

    def load_field_map_background(self):
        path = self.get_parameter('field_map_image').value
        if not path:
            path = self.package_file('maps', 'field_map.png')

        if not os.path.exists(path):
            self.get_logger().warn(
                f'Field map image not found: {path}; using simple white fallback preview'
            )
            return None

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().warn(
                f'Failed to read field map image: {path}; using simple white fallback preview'
            )
            return None

        self.get_logger().info(f'Loaded field map preview background: {path}')
        return cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

    def package_file(self, *path_parts):
        try:
            from ament_index_python.packages import get_package_share_directory
            return os.path.join(get_package_share_directory('origincar_nav'), *path_parts)
        except Exception:
            return os.path.join(os.getcwd(), 'origincar_nav', *path_parts)

    def goal_callback(self, msg):
        self.current_goal = (msg.pose.position.x, msg.pose.position.y)

    def odom_callback(self, msg):
        pose = msg.pose.pose
        self.current_pose = (pose.position.x, pose.position.y)
        self.current_yaw = yaw_from_quaternion(pose.orientation)
        self.vehicle_path.append(self.current_pose)
        if len(self.vehicle_path) > self.path_max_length:
            self.vehicle_path = self.vehicle_path[-self.path_max_length:]

    def publish_preview(self):
        if cv2 is None or np is None:
            return

        if self.background is None:
            image = np.full((self.image_size, self.image_size, 3), 255, dtype=np.uint8)
        else:
            image = self.background.copy()

        transform = self.world_to_pixel
        self.draw_boundary(image, transform)
        self.draw_routes(image, transform)
        self.draw_vehicle_path(image, transform)
        self.draw_points(image, transform)
        self.draw_goal(image, transform)
        self.draw_pose(image, transform)
        self.draw_overlay_text(image)

        stamp = self.get_clock().now().to_msg()
        self.image_pub.publish(self.to_image_msg(image, stamp))
        compressed = self.to_compressed_msg(image, stamp)
        if compressed is not None:
            self.compressed_pub.publish(compressed)

    def world_to_pixel(self, x, y):
        x_min = float(self.boundary.get('x_min', 0.0))
        x_max = float(self.boundary.get('x_max', 5.0))
        y_min = float(self.boundary.get('y_min', 0.0))
        y_max = float(self.boundary.get('y_max', 5.0))
        width = max(x_max - x_min, 1e-6)
        height = max(y_max - y_min, 1e-6)
        px = int((float(x) - x_min) / width * (self.image_size - 1))
        py = int((y_max - float(y)) / height * (self.image_size - 1))
        return self.clamp_pixel(px, py)

    def clamp_pixel(self, px, py):
        px = max(0, min(self.image_size - 1, int(px)))
        py = max(0, min(self.image_size - 1, int(py)))
        return px, py

    def draw_boundary(self, image, transform):
        keys = ('x_min', 'x_max', 'y_min', 'y_max')
        if not all(key in self.boundary for key in keys):
            return
        p1 = transform(self.boundary['x_min'], self.boundary['y_min'])
        p2 = transform(self.boundary['x_max'], self.boundary['y_max'])
        top_left = (min(p1[0], p2[0]), min(p1[1], p2[1]))
        bottom_right = (max(p1[0], p2[0]), max(p1[1], p2[1]))
        cv2.rectangle(image, top_left, bottom_right, (40, 40, 40), 2)

    def draw_routes(self, image, transform):
        self.draw_route(image, transform, 'clockwise', (0, 150, 0))
        self.draw_route(image, transform, 'anticlockwise', (180, 100, 0))

    def draw_route(self, image, transform, route_name, color):
        route = self.routes.get(route_name, {})
        order = route.get('channel_order', [])
        pixels = []
        for name in order:
            pose = self.point_pose(name)
            if pose is not None:
                pixels.append(transform(pose[0], pose[1]))
        for start, end in zip(pixels, pixels[1:]):
            cv2.line(image, start, end, color, 3, lineType=cv2.LINE_AA)
        if pixels:
            cv2.putText(image, route_name, (pixels[0][0] + 8, pixels[0][1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, lineType=cv2.LINE_AA)

    def draw_vehicle_path(self, image, transform):
        if len(self.vehicle_path) < 2:
            return
        pixels = [transform(x, y) for x, y in self.vehicle_path]
        for start, end in zip(pixels, pixels[1:]):
            cv2.line(image, start, end, (60, 60, 60), 2, lineType=cv2.LINE_AA)

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
            cv2.circle(image, pixel, 8, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(image, pixel, 10, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(image, name, (pixel[0] + 10, pixel[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, lineType=cv2.LINE_AA)

    def draw_goal(self, image, transform):
        if self.current_goal is None:
            return
        pixel = transform(self.current_goal[0], self.current_goal[1])
        cv2.drawMarker(image, pixel, (0, 0, 255), cv2.MARKER_STAR, 24, 2, line_type=cv2.LINE_AA)
        cv2.putText(image, 'current_goal', (pixel[0] + 10, pixel[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)

    def draw_pose(self, image, transform):
        if self.current_pose is None:
            return
        pixel = transform(self.current_pose[0], self.current_pose[1])
        heading_len = 30
        end = (
            int(pixel[0] + math.cos(self.current_yaw) * heading_len),
            int(pixel[1] - math.sin(self.current_yaw) * heading_len),
        )
        cv2.circle(image, pixel, 8, (0, 0, 0), 2, lineType=cv2.LINE_AA)
        cv2.arrowedLine(image, pixel, end, (0, 0, 0), 2, tipLength=0.3, line_type=cv2.LINE_AA)
        cv2.putText(image, 'car', (pixel[0] + 10, pixel[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

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
            org = (18, 30 + index * 24)
            cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 4, lineType=cv2.LINE_AA)
            cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (20, 20, 20), 2, lineType=cv2.LINE_AA)

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
