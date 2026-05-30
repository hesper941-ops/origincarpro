#!/usr/bin/env python3

import math
import os

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped, Twist, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class TargetTracker(Node):
    def __init__(self):
        super().__init__('target_tracker')

        self.declare_parameter('max_linear_speed', 0.20)
        self.declare_parameter('min_linear_speed', 0.05)
        self.declare_parameter('max_angular_speed', 0.80)
        self.declare_parameter('k_linear', 0.6)
        self.declare_parameter('k_angular', 1.5)
        self.declare_parameter('goal_reached_dist', 0.15)
        self.declare_parameter('goal_reached_yaw', 0.35)
        self.declare_parameter('semantic_map_file', '')
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('odom_topic', '/odom_combined')
        self.declare_parameter('goal_topic', '/current_goal')
        self.declare_parameter('cmd_topic', '/track_cmd_vel')
        self.declare_parameter('enable_topic', '/track_enable')

        self.max_linear_speed = float(self.get_parameter('max_linear_speed').value)
        self.min_linear_speed = float(self.get_parameter('min_linear_speed').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)
        self.k_linear = float(self.get_parameter('k_linear').value)
        self.k_angular = float(self.get_parameter('k_angular').value)
        thresholds = self.load_thresholds()
        self.goal_reached_dist = float(
            self.get_parameter('goal_reached_dist').value
            or thresholds.get('goal_reached_dist', 0.15)
        )
        self.goal_reached_yaw = float(
            self.get_parameter('goal_reached_yaw').value
            or thresholds.get('goal_reached_yaw', 0.35)
        )

        self.current_pose = None
        self.current_yaw = 0.0
        self.goal = None
        self.goal_yaw = 0.0
        self.goal_signature = None
        self.enabled = False
        self.last_warn_time = 0.0
        self.goal_was_reached = False

        self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self.odom_callback,
            10,
        )
        self.create_subscription(
            PoseStamped,
            self.get_parameter('goal_topic').value,
            self.goal_callback,
            10,
        )
        self.create_subscription(
            Bool,
            self.get_parameter('enable_topic').value,
            self.enable_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(Twist, self.get_parameter('cmd_topic').value, 10)
        self.goal_reached_pub = self.create_publisher(Bool, '/goal_reached', 10)
        self.error_pub = self.create_publisher(Vector3, '/tracking_error', 10)

        rate = float(self.get_parameter('control_rate_hz').value)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self.control_loop)

    def load_thresholds(self):
        path = self.get_parameter('semantic_map_file').value
        if not path:
            try:
                from ament_index_python.packages import get_package_share_directory
                path = os.path.join(get_package_share_directory('origincar_nav'), 'config', 'semantic_map.yaml')
            except Exception:
                return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return (yaml.safe_load(f) or {}).get('thresholds', {})
        except Exception as exc:
            self.get_logger().warn(f'Failed to read tracker thresholds from semantic map: {exc}')
            return {}

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        self.current_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

    def goal_callback(self, msg):
        signature = self.make_goal_signature(msg)
        self.goal = msg.pose
        self.goal_yaw = yaw_from_quaternion(msg.pose.orientation)
        if signature != self.goal_signature:
            self.goal_signature = signature
            self.goal_was_reached = False

    def enable_callback(self, msg):
        self.enabled = bool(msg.data)
        if not self.enabled:
            self.publish_zero()

    def control_loop(self):
        if not self.enabled:
            self.publish_zero()
            return

        if self.current_pose is None or self.goal is None:
            self.publish_zero()
            now = self.get_clock().now().nanoseconds / 1e9
            if now - self.last_warn_time > 2.0:
                self.get_logger().warn('Waiting for odom and current goal before tracking')
                self.last_warn_time = now
            return

        dx = self.goal.position.x - self.current_pose.position.x
        dy = self.goal.position.y - self.current_pose.position.y
        distance = math.hypot(dx, dy)
        target_heading = math.atan2(dy, dx) if distance > 1e-6 else self.goal_yaw
        heading_error = normalize_angle(target_heading - self.current_yaw)
        target_yaw_error = normalize_angle(self.goal_yaw - self.current_yaw)

        self.error_pub.publish(Vector3(x=distance, y=heading_error, z=target_yaw_error))

        cmd = Twist()
        reached_msg = Bool()

        if distance > self.goal_reached_dist:
            angular = self.k_angular * heading_error
            cmd.angular.z = max(-self.max_angular_speed, min(self.max_angular_speed, angular))

            speed_scale = max(0.0, 1.0 - abs(heading_error) / 1.2)
            linear = min(self.max_linear_speed, self.k_linear * distance) * speed_scale
            if abs(heading_error) < 1.4:
                cmd.linear.x = max(self.min_linear_speed, linear)
            reached_msg.data = False
            self.goal_was_reached = False
        else:
            if abs(target_yaw_error) > self.goal_reached_yaw:
                cmd.angular.z = max(
                    -self.max_angular_speed,
                    min(self.max_angular_speed, self.k_angular * target_yaw_error),
                )
                reached_msg.data = False
            else:
                reached_msg.data = not self.goal_was_reached
                if reached_msg.data:
                    self.goal_was_reached = True

        self.cmd_pub.publish(cmd)
        self.goal_reached_pub.publish(reached_msg)

    def publish_zero(self):
        self.cmd_pub.publish(Twist())

    def make_goal_signature(self, msg):
        return (
            msg.header.frame_id,
            round(float(msg.pose.position.x), 3),
            round(float(msg.pose.position.y), 3),
            round(float(msg.pose.position.z), 3),
            round(float(yaw_from_quaternion(msg.pose.orientation)), 3),
        )


def main(args=None):
    rclpy.init(args=args)
    node = TargetTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_zero()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
