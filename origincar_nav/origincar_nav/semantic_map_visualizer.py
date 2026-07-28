#!/usr/bin/env python3

import math
import os

import rclpy
import yaml
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw):
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)
    return qz, qw


class SemanticMapVisualizer(Node):
    def __init__(self):
        super().__init__('semantic_map_visualizer')

        self.declare_parameter('semantic_map_file', '')
        self.declare_parameter('odom_topic', '/odom_combined')
        self.declare_parameter('visualization_frame', '')
        self.declare_parameter('path_max_length', 500)
        self.declare_parameter('publish_rate_hz', 2.0)

        self.semantic_map = self.load_semantic_map()
        viz_cfg = self.semantic_map.get('visualization', {})
        self.frame_id = (
            self.get_parameter('visualization_frame').value
            or viz_cfg.get('frame_id')
            or self.semantic_map.get('map_frame', 'map')
        )
        self.path_max_length = int(
            self.get_parameter('path_max_length').value
            or viz_cfg.get('path_max_length', 500)
        )
        self.show_text_labels = bool(viz_cfg.get('show_text_labels', True))
        self.show_routes = bool(viz_cfg.get('show_routes', True))
        self.show_boundary = bool(viz_cfg.get('show_boundary', True))

        self.points = self.semantic_map.get('points', {})
        self.routes = self.semantic_map.get('routes', {})
        self.boundary = self.semantic_map.get('boundary', {})

        self.current_goal = None
        self.current_pose = None
        self.mission_state = 'INIT'
        self.perception_mode = 'idle'
        self.avoid_active = False
        self.obstacle_debug_point = None
        self.path = Path()
        self.path.header.frame_id = self.frame_id

        self.marker_pub = self.create_publisher(MarkerArray, '/semantic_map/markers', 10)
        self.goal_marker_pub = self.create_publisher(Marker, '/current_goal_marker', 10)
        self.path_pub = self.create_publisher(Path, '/vehicle_path', 10)
        self.text_marker_pub = self.create_publisher(Marker, '/mission_text_marker', 10)
        self.obstacle_marker_pub = self.create_publisher(Marker, '/avoid_obstacle_marker', 10)
        self.pose_marker_pub = self.create_publisher(Marker, '/semantic_map/current_pose_marker', 10)

        self.create_subscription(Odometry, self.get_parameter('odom_topic').value, self.odom_callback, 10)
        self.create_subscription(PoseStamped, '/current_goal', self.goal_callback, 10)
        self.create_subscription(String, '/mission_state', self.state_callback, 10)
        self.create_subscription(String, '/perception_mode', self.perception_callback, 10)
        self.create_subscription(Bool, '/avoid_active', self.avoid_callback, 10)
        self.create_subscription(PointStamped, '/avoid/obstacle_debug_point', self.obstacle_callback, 10)

        rate = float(self.get_parameter('publish_rate_hz').value)
        self.timer = self.create_timer(1.0 / max(rate, 0.5), self.publish_visualization)

    def load_semantic_map(self):
        path = self.get_parameter('semantic_map_file').value
        if not path:
            try:
                from ament_index_python.packages import get_package_share_directory
                path = os.path.join(get_package_share_directory('origincar_nav'), 'config', 'semantic_map.yaml')
            except Exception:
                path = os.path.join(os.getcwd(), 'origincar_nav', 'config', 'semantic_map.yaml')
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as exc:
            self.get_logger().warn(f'Visualization disabled map details; failed to read semantic map: {exc}')
            return {}

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = self.frame_id
        pose.pose = msg.pose.pose
        self.path.poses.append(pose)
        if len(self.path.poses) > self.path_max_length:
            self.path.poses = self.path.poses[-self.path_max_length:]

    def goal_callback(self, msg):
        self.current_goal = msg.pose

    def state_callback(self, msg):
        self.mission_state = msg.data

    def perception_callback(self, msg):
        self.perception_mode = msg.data

    def avoid_callback(self, msg):
        self.avoid_active = bool(msg.data)

    def obstacle_callback(self, msg):
        self.obstacle_debug_point = msg.point

    def publish_visualization(self):
        self.marker_pub.publish(self.build_semantic_markers())
        self.publish_goal_marker()
        self.publish_path()
        self.publish_text_marker()
        self.publish_obstacle_marker()
        self.publish_pose_marker()

    def build_semantic_markers(self):
        markers = MarkerArray()
        marker_id = 0
        if self.show_boundary and self.boundary:
            markers.markers.append(self.boundary_marker(marker_id))
            marker_id += 1

        for name, cfg in self.points.items():
            markers.markers.append(self.point_marker(marker_id, name, cfg))
            marker_id += 1
            if self.show_text_labels:
                markers.markers.append(self.text_label_marker(marker_id, name, cfg))
                marker_id += 1

        if self.show_routes:
            for route_name, route_cfg in self.routes.items():
                markers.markers.append(self.route_marker(marker_id, route_name, route_cfg))
                marker_id += 1
        return markers

    def base_marker(self, marker_id, marker_type, ns):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.frame_id
        marker.ns = ns
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def boundary_marker(self, marker_id):
        marker = self.base_marker(marker_id, Marker.LINE_STRIP, 'semantic_boundary')
        marker.scale.x = 0.03
        marker.color.r = 0.1
        marker.color.g = 0.7
        marker.color.b = 1.0
        marker.color.a = 1.0
        x_min = float(self.boundary.get('x_min', 0.0))
        x_max = float(self.boundary.get('x_max', 0.0))
        y_min = float(self.boundary.get('y_min', 0.0))
        y_max = float(self.boundary.get('y_max', 0.0))
        marker.points = [
            Point(x=x_min, y=y_min, z=0.02),
            Point(x=x_max, y=y_min, z=0.02),
            Point(x=x_max, y=y_max, z=0.02),
            Point(x=x_min, y=y_max, z=0.02),
            Point(x=x_min, y=y_min, z=0.02),
        ]
        return marker

    def point_marker(self, marker_id, name, cfg):
        pose = cfg.get('pose', [0.0, 0.0, 0.0])
        marker = self.base_marker(marker_id, Marker.SPHERE, 'semantic_points')
        marker.pose.position.x = float(pose[0])
        marker.pose.position.y = float(pose[1])
        marker.pose.position.z = 0.08
        marker.scale.x = 0.12
        marker.scale.y = 0.12
        marker.scale.z = 0.12
        marker.color.r = 1.0 if name == 'p_start' else 0.2
        marker.color.g = 0.8
        marker.color.b = 0.2
        marker.color.a = 1.0
        return marker

    def text_label_marker(self, marker_id, name, cfg):
        pose = cfg.get('pose', [0.0, 0.0, 0.0])
        marker = self.base_marker(marker_id, Marker.TEXT_VIEW_FACING, 'semantic_labels')
        marker.pose.position.x = float(pose[0])
        marker.pose.position.y = float(pose[1])
        marker.pose.position.z = 0.25
        marker.scale.z = 0.16
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.text = name
        return marker

    def route_marker(self, marker_id, route_name, route_cfg):
        marker = self.base_marker(marker_id, Marker.LINE_STRIP, f'route_{route_name}')
        marker.scale.x = 0.02
        marker.color.r = 1.0 if route_name == 'clockwise' else 0.8
        marker.color.g = 0.5 if route_name == 'clockwise' else 0.2
        marker.color.b = 0.1 if route_name == 'clockwise' else 1.0
        marker.color.a = 0.9
        for point_name in route_cfg.get('channel_order', []):
            point_cfg = self.points.get(point_name)
            if not point_cfg:
                continue
            pose = point_cfg.get('pose', [0.0, 0.0, 0.0])
            marker.points.append(Point(x=float(pose[0]), y=float(pose[1]), z=0.05))
        return marker

    def publish_goal_marker(self):
        marker = self.base_marker(0, Marker.ARROW, 'current_goal')
        if self.current_goal is None:
            marker.action = Marker.DELETE
        else:
            marker.pose = self.current_goal
            marker.scale.x = 0.35
            marker.scale.y = 0.08
            marker.scale.z = 0.08
            marker.color.r = 1.0
            marker.color.g = 0.9
            marker.color.b = 0.1
            marker.color.a = 1.0
        self.goal_marker_pub.publish(marker)

    def publish_path(self):
        self.path.header.stamp = self.get_clock().now().to_msg()
        self.path.header.frame_id = self.frame_id
        self.path_pub.publish(self.path)

    def publish_text_marker(self):
        marker = self.base_marker(0, Marker.TEXT_VIEW_FACING, 'mission_text')
        marker.pose.position.x = 0.2
        marker.pose.position.y = 0.2
        marker.pose.position.z = 0.7
        marker.scale.z = 0.18
        marker.color.r = 0.9
        marker.color.g = 0.9
        marker.color.b = 0.9
        marker.color.a = 1.0
        marker.text = f'state: {self.mission_state}\nmode: {self.perception_mode}'
        self.text_marker_pub.publish(marker)

    def publish_obstacle_marker(self):
        marker = self.base_marker(0, Marker.SPHERE, 'avoid_obstacle_debug')
        if self.obstacle_debug_point is None:
            marker.action = Marker.DELETE
        else:
            marker.pose.position = self.obstacle_debug_point
            marker.pose.position.z = 0.12
            marker.scale.x = 0.18
            marker.scale.y = 0.18
            marker.scale.z = 0.18
            marker.color.r = 1.0
            marker.color.g = 0.1
            marker.color.b = 0.1
            marker.color.a = 1.0
        self.obstacle_marker_pub.publish(marker)

    def publish_pose_marker(self):
        marker = self.base_marker(0, Marker.ARROW, 'current_pose')
        if self.current_pose is None:
            marker.action = Marker.DELETE
        else:
            marker.pose = self.current_pose
            marker.scale.x = 0.28
            marker.scale.y = 0.07
            marker.scale.z = 0.07
            marker.color.r = 0.1
            marker.color.g = 0.9
            marker.color.b = 0.9
            marker.color.a = 1.0
        self.pose_marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
