#!/usr/bin/env python3

import math
import os

import rclpy
import yaml
from ai_msgs.msg import PerceptionTargets
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class YoloAvoidController(Node):
    def __init__(self):
        super().__init__('yolo_avoid_controller')

        self.declare_parameter('detection_topic', '/hobot_dnn_detection')
        self.declare_parameter('odom_topic', '/odom_combined')
        self.declare_parameter('semantic_map_file', '')
        self.declare_parameter('obstacle_labels', ['obstacle'])
        self.declare_parameter('image_width', 640.0)
        self.declare_parameter('trigger_area_ratio', 0.08)
        self.declare_parameter('center_deadband_ratio', 0.12)
        self.declare_parameter('avoid_duration_sec', 1.0)
        self.declare_parameter('cooldown_sec', 0.2)
        self.declare_parameter('lateral_check_dist', 0.35)
        self.declare_parameter('avoid_linear_speed', 0.08)
        self.declare_parameter('avoid_angular_speed', 0.65)
        self.declare_parameter('publish_rate_hz', 20.0)

        self.boundary = self.load_boundary()
        self.obstacle_labels = {
            str(label).strip().lower()
            for label in self.get_parameter('obstacle_labels').value
        }
        self.image_width = float(self.get_parameter('image_width').value)
        self.trigger_area_ratio = float(self.get_parameter('trigger_area_ratio').value)
        self.center_deadband_ratio = float(self.get_parameter('center_deadband_ratio').value)
        self.avoid_duration_sec = float(self.get_parameter('avoid_duration_sec').value)
        self.cooldown_sec = float(self.get_parameter('cooldown_sec').value)
        self.lateral_check_dist = float(self.get_parameter('lateral_check_dist').value)
        self.avoid_linear_speed = float(self.get_parameter('avoid_linear_speed').value)
        self.avoid_angular_speed = float(self.get_parameter('avoid_angular_speed').value)

        self.current_x = None
        self.current_y = None
        self.current_yaw = 0.0
        self.avoid_active = False
        self.avoid_direction = 0
        self.avoid_end_time = 0.0
        self.cooldown_until = 0.0

        self.create_subscription(
            PerceptionTargets,
            self.get_parameter('detection_topic').value,
            self.detection_callback,
            10,
        )
        self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self.odom_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(Twist, '/avoid_cmd_vel', 10)
        self.legacy_cmd_pub = self.create_publisher(Twist, '/yolo_cmd_vel', 10)
        self.active_pub = self.create_publisher(Bool, '/avoid_active', 10)
        self.legacy_active_pub = self.create_publisher(Bool, '/yolo_avoid_active', 10)
        self.finished_pub = self.create_publisher(Bool, '/avoid_finished', 10)
        self.legacy_finished_pub = self.create_publisher(Bool, '/yolo_avoid_finished', 10)

        rate = float(self.get_parameter('publish_rate_hz').value)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self.tick)

    def load_boundary(self):
        path = self.get_parameter('semantic_map_file').value
        if not path:
            try:
                from ament_index_python.packages import get_package_share_directory
                path = os.path.join(get_package_share_directory('origincar_nav'), 'config', 'semantic_map.yaml')
            except Exception:
                path = ''
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return (yaml.safe_load(f) or {}).get('boundary', {})
        except Exception as exc:
            self.get_logger().warn(f'Using fallback map boundary; failed to read semantic map: {exc}')
            return {'x_min': 0.0, 'x_max': 4.0, 'y_min': 0.0, 'y_max': 3.0}

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

    def detection_callback(self, msg):
        now = self.now_sec()
        if self.avoid_active or now < self.cooldown_until:
            return

        obstacle = self.select_obstacle(msg)
        if obstacle is None:
            return

        direction = self.choose_direction(obstacle['center_x'])
        self.avoid_active = True
        self.avoid_direction = direction
        self.avoid_end_time = now + self.avoid_duration_sec
        self.publish_active(True)

    def select_obstacle(self, msg):
        best = None
        for target in msg.targets:
            if str(target.type).strip().lower() not in self.obstacle_labels:
                continue
            for roi in target.rois:
                rect = roi.rect
                area_ratio = (float(rect.width) * float(rect.height)) / max(self.image_width * self.image_width, 1.0)
                if area_ratio < self.trigger_area_ratio:
                    continue
                center_x = float(rect.x_offset) + float(rect.width) * 0.5
                if best is None or area_ratio > best['area_ratio']:
                    best = {'center_x': center_x, 'area_ratio': area_ratio}
        return best

    def choose_direction(self, center_x):
        center_offset = (center_x - self.image_width * 0.5) / max(self.image_width, 1.0)
        left_ok = self.side_inside_boundary(left=True)
        right_ok = self.side_inside_boundary(left=False)

        if not left_ok and not right_ok:
            self.get_logger().warn('Both avoidance sides are outside boundary; stopping')
            return 0
        if not left_ok:
            return -1
        if not right_ok:
            return 1
        if center_offset < -self.center_deadband_ratio:
            return -1
        if center_offset > self.center_deadband_ratio:
            return 1
        return 1

    def side_inside_boundary(self, left):
        if self.current_x is None or self.current_y is None:
            return True
        sign = 1.0 if left else -1.0
        x = self.current_x - sign * math.sin(self.current_yaw) * self.lateral_check_dist
        y = self.current_y + sign * math.cos(self.current_yaw) * self.lateral_check_dist
        return (
            float(self.boundary.get('x_min', -999.0)) <= x <= float(self.boundary.get('x_max', 999.0))
            and float(self.boundary.get('y_min', -999.0)) <= y <= float(self.boundary.get('y_max', 999.0))
        )

    def tick(self):
        cmd = Twist()
        now = self.now_sec()

        if self.avoid_active and now <= self.avoid_end_time:
            if self.avoid_direction != 0:
                cmd.linear.x = self.avoid_linear_speed
                cmd.angular.z = self.avoid_angular_speed * self.avoid_direction
            self.publish_cmd(cmd)
            self.publish_active(True)
            return

        if self.avoid_active:
            self.avoid_active = False
            self.cooldown_until = now + self.cooldown_sec
            self.publish_cmd(Twist())
            self.publish_active(False)
            self.publish_finished()

    def publish_cmd(self, cmd):
        self.cmd_pub.publish(cmd)
        self.legacy_cmd_pub.publish(cmd)

    def publish_active(self, active):
        msg = Bool(data=bool(active))
        self.active_pub.publish(msg)
        self.legacy_active_pub.publish(msg)

    def publish_finished(self):
        msg = Bool(data=True)
        self.finished_pub.publish(msg)
        self.legacy_finished_pub.publish(msg)

    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9


def main(args=None):
    rclpy.init(args=args)
    node = YoloAvoidController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_cmd(Twist())
        node.publish_active(False)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
