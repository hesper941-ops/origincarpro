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
        self.declare_parameter('preview_image_size', 500)
        self.declare_parameter('preview_publish_rate', 1.0)
        self.declare_parameter('publish_raw_image', False)
        self.declare_parameter('jpeg_quality', 85)
        self.declare_parameter('max_path_points', 300)
        self.declare_parameter('overlay_font_scale', 0.45)
        self.declare_parameter('overlay_font_thickness', 1)

        self.image_size = int(self.get_parameter('preview_image_size').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.publish_raw_image = self.as_bool(self.get_parameter('publish_raw_image').value)
        self.max_path_points = int(self.get_parameter('max_path_points').value)
        self.overlay_font_scale = float(self.get_parameter('overlay_font_scale').value)
        self.overlay_font_thickness = int(self.get_parameter('overlay_font_thickness').value)
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

        self.image_pub = None
        if self.publish_raw_image:
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
        self.get_logger().info(
            'Semantic map preview config: '
            f'background_shape={None if self.background is None else self.background.shape}, '
            f'preview_image_size={self.image_size}, '
            f'publish_raw_image={self.publish_raw_image}, '
            f'preview_publish_rate={rate}'
        )
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

        self.get_logger().info(f'Field map path: {path}')
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().warn(
                f'Failed to read field map image: {path}; using simple white fallback preview'
            )
            return None

        self.get_logger().info(f'Loaded field map preview background: {path}')
        if image.shape[0] == self.image_size and image.shape[1] == self.image_size:
            return image
        return cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

    def package_file(self, *path_parts):
        try:
            from ament_index_python.packages import get_package_share_directory
            return os.path.join(get_package_share_directory('origincar_nav'), *path_parts)
        except Exception:
            return os.path.join(os.getcwd(), 'origincar_nav', *path_parts)

    def goal_callback(self, msg):
        point = (msg.pose.position.x, msg.pose.position.y)
        self.current_goal = point if self.valid_xy(*point) else None

    def odom_callback(self, msg):
        pose = msg.pose.pose
        point = (pose.position.x, pose.position.y)
        yaw = yaw_from_quaternion(pose.orientation)
        if not self.valid_xy(*point) or not math.isfinite(yaw):
            return
        self.current_pose = point
        self.current_yaw = yaw
        self.vehicle_path.append(self.current_pose)
        if len(self.vehicle_path) > self.max_path_points:
            self.vehicle_path = self.vehicle_path[-self.max_path_points:]

    def publish_preview(self):
        try:
            if cv2 is None or np is None:
                return

            if self.background is None:
                frame = np.full((self.image_size, self.image_size, 3), 255, dtype=np.uint8)
            else:
                frame = self.background.copy()

            transform = self.world_to_pixel
            self.draw_boundary(frame, transform)
            self.draw_routes(frame, transform)
            self.draw_vehicle_path(frame, transform)
            self.draw_points(frame, transform)
            self.draw_goal(frame, transform)
            self.draw_pose(frame, transform)
            self.draw_overlay_text(frame)

            stamp = self.get_clock().now().to_msg()
            if self.publish_raw_image and self.image_pub is not None:
                self.image_pub.publish(self.to_image_msg(frame, stamp))
            compressed = self.to_compressed_msg(frame, stamp)
            if compressed is not None:
                self.compressed_pub.publish(compressed)
        except Exception as exc:
            self.get_logger().error(f'Failed to publish semantic map preview: {exc}')

    def world_to_pixel(self, x, y):
        if not self.valid_xy(x, y):
            return None
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
        if px is None or py is None:
            return None
        px = max(0, min(self.image_size - 1, int(px)))
        py = max(0, min(self.image_size - 1, int(py)))
        return px, py

    def draw_boundary(self, image, transform):
        keys = ('x_min', 'x_max', 'y_min', 'y_max')
        if not all(key in self.boundary for key in keys):
            return
        if not self.valid_xy(self.boundary['x_min'], self.boundary['y_min']):
            return
        if not self.valid_xy(self.boundary['x_max'], self.boundary['y_max']):
            return
        p1 = transform(self.boundary['x_min'], self.boundary['y_min'])
        p2 = transform(self.boundary['x_max'], self.boundary['y_max'])
        if p1 is None or p2 is None:
            return
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
                pixel = transform(pose[0], pose[1])
                if pixel is not None:
                    pixels.append(pixel)
        for start, end in zip(pixels, pixels[1:]):
            cv2.line(image, start, end, color, 2, lineType=cv2.LINE_AA)

    def draw_vehicle_path(self, image, transform):
        if len(self.vehicle_path) < 2:
            return
        pixels = [transform(x, y) for x, y in self.vehicle_path if self.valid_xy(x, y)]
        pixels = [pixel for pixel in pixels if pixel is not None]
        for start, end in zip(pixels, pixels[1:]):
            cv2.line(image, start, end, (60, 60, 60), 2, lineType=cv2.LINE_AA)

    def draw_points(self, image, transform):
        colors = {
            'p_start': (0, 90, 255),
            'task_station': (255, 0, 0),
            'channel_entry': (0, 160, 160),
            'channel_entry_lower': (0, 130, 210),
            'channel_entry_upper': (0, 130, 210),
        }
        label_offsets = {
            'p_start': (10, -12),
            'task_station': (-118, -12),
            'channel_entry': (-105, 4),
            'channel_entry_lower': (10, 20),
            'channel_entry_upper': (10, -12),
            'channel_p1': (10, -12),
            'channel_p2': (10, -12),
            'channel_p3': (10, -12),
            'channel_p4': (10, -12),
        }
        for name in (
            'p_start',
            'task_station',
            'channel_entry',
            'channel_entry_lower',
            'channel_entry_upper',
            'channel_p1',
            'channel_p2',
            'channel_p3',
            'channel_p4',
        ):
            pose = self.point_pose(name)
            if pose is None:
                continue
            pixel = transform(pose[0], pose[1])
            if pixel is None:
                continue
            color = colors.get(name, (80, 80, 220))
            cv2.circle(image, pixel, 8, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(image, pixel, 10, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            dx, dy = label_offsets.get(name, (10, -8))
            label_pos = self.clamp_label_position(pixel[0] + dx, pixel[1] + dy)
            cv2.putText(image, name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, lineType=cv2.LINE_AA)

    def draw_goal(self, image, transform):
        if self.current_goal is None:
            return
        pixel = transform(self.current_goal[0], self.current_goal[1])
        if pixel is None:
            return
        cv2.drawMarker(image, pixel, (0, 0, 255), cv2.MARKER_STAR, 24, 2, line_type=cv2.LINE_AA)
        cv2.putText(image, 'current_goal', (pixel[0] + 10, pixel[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)

    def draw_pose(self, image, transform):
        if self.current_pose is None:
            return
        pixel = transform(self.current_pose[0], self.current_pose[1])
        if pixel is None:
            return
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
            org = (12, 22 + index * 17)
            cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, self.overlay_font_scale,
                        (255, 255, 255), self.overlay_font_thickness + 2, lineType=cv2.LINE_AA)
            cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, self.overlay_font_scale,
                        (20, 20, 20), self.overlay_font_thickness, lineType=cv2.LINE_AA)

    def point_pose(self, name):
        cfg = self.points.get(name)
        if not cfg:
            return None
        pose = cfg.get('pose', [])
        if len(pose) < 2:
            return None
        if not self.valid_xy(pose[0], pose[1]):
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

    def valid_xy(self, x, y):
        try:
            return math.isfinite(float(x)) and math.isfinite(float(y))
        except (TypeError, ValueError):
            return False

    def clamp_label_position(self, x, y):
        return (
            max(4, min(self.image_size - 120, int(x))),
            max(14, min(self.image_size - 6, int(y))),
        )

    def as_bool(self, value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ('1', 'true', 'yes', 'on')

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
