#!/usr/bin/env python3

import math
import os
from enum import Enum

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, Int32, String

try:
    from origincar_msg.msg import Sign
except ImportError:
    Sign = None


class MissionState(Enum):
    INIT = 'INIT'
    TRACK_TO_TASK_STATION = 'TRACK_TO_TASK_STATION'
    WAIT_QR_RESULT = 'WAIT_QR_RESULT'
    TRACK_TO_CHANNEL_ENTRY = 'TRACK_TO_CHANNEL_ENTRY'
    CHANNEL_NAV = 'CHANNEL_NAV'
    RETURN_PREPARE = 'RETURN_PREPARE'
    BIRDVIEW_RETURN = 'BIRDVIEW_RETURN'
    FINISH = 'FINISH'


def quaternion_from_yaw(yaw):
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)
    return qz, qw


class TaskManager(Node):
    def __init__(self):
        super().__init__('task_manager')

        self.declare_parameter('semantic_map_file', '')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_topic', '/odom_combined')
        self.declare_parameter('sign_topic_type', 'origincar_msg/Sign')
        self.declare_parameter('publish_rate_hz', 2.0)

        self.semantic_map = self.load_semantic_map()
        self.map_frame = self.semantic_map.get('map_frame', self.get_parameter('map_frame').value)
        self.points = self.semantic_map.get('points', {})
        self.routes = self.semantic_map.get('routes', {})

        self.state = MissionState.INIT
        self.route_direction = None
        self.channel_order = []
        self.channel_index = 0
        self.current_goal_name = None
        self.last_published_goal_name = None
        self.avoid_active = False
        self.birdview_valid = False
        self.last_goal_reached = False

        self.goal_pub = self.create_publisher(PoseStamped, '/current_goal', 10)
        self.track_enable_pub = self.create_publisher(Bool, '/track_enable', 10)
        self.state_pub = self.create_publisher(String, '/mission_state', 10)
        self.perception_pub = self.create_publisher(String, '/perception_mode', 10)

        self.create_subscription(Bool, '/goal_reached', self.goal_reached_callback, 10)
        self.create_subscription(String, '/qrcode_detected/info_result', self.qr_callback, 10)
        self.create_subscription(Bool, '/avoid_active', self.avoid_active_callback, 10)
        self.create_subscription(Bool, '/yolo_avoid_active', self.avoid_active_callback, 10)
        self.create_subscription(Bool, '/avoid_finished', self.avoid_finished_callback, 10)
        self.create_subscription(Bool, '/yolo_avoid_finished', self.avoid_finished_callback, 10)
        self.create_subscription(Bool, '/bird_view/p_point_valid', self.birdview_valid_callback, 10)
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value, self.odom_callback, 10)

        if self.get_parameter('sign_topic_type').value == 'std_msgs/Int32' or Sign is None:
            self.create_subscription(Int32, '/sign_switch', self.sign_int_callback, 10)
        else:
            self.create_subscription(Sign, '/sign_switch', self.sign_msg_callback, 10)

        rate = float(self.get_parameter('publish_rate_hz').value)
        self.timer = self.create_timer(1.0 / max(rate, 0.5), self.tick)

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
            self.get_logger().info(f'Loaded semantic map: {path}')
            return data
        except Exception as exc:
            self.get_logger().warn(f'Failed to load semantic map {path}: {exc}')
            return {'points': {}, 'routes': {}, 'map_frame': self.get_parameter('map_frame').value}

    def tick(self):
        if self.state == MissionState.INIT:
            self.set_state(MissionState.TRACK_TO_TASK_STATION)

        if self.avoid_active:
            self.publish_track_enable(False)
            self.publish_status('target_track')
            return

        if self.state == MissionState.TRACK_TO_TASK_STATION:
            self.publish_named_goal('task_station')
            self.publish_track_enable(True)
            self.publish_status('target_track')
        elif self.state == MissionState.WAIT_QR_RESULT:
            self.publish_track_enable(False)
            self.publish_status('qr')
        elif self.state == MissionState.TRACK_TO_CHANNEL_ENTRY:
            self.publish_named_goal('channel_entry')
            self.publish_track_enable(True)
            self.publish_status('target_track')
        elif self.state == MissionState.CHANNEL_NAV:
            self.publish_track_enable(True)
            self.publish_status('channel_visual_correction')
            if self.channel_index < len(self.channel_order):
                self.publish_named_goal(self.channel_order[self.channel_index])
            else:
                self.set_state(MissionState.RETURN_PREPARE)
        elif self.state == MissionState.RETURN_PREPARE:
            self.publish_status('target_track')
            self.publish_named_goal('p_start')
            self.publish_track_enable(True)
            if self.birdview_valid:
                self.set_state(MissionState.BIRDVIEW_RETURN)
        elif self.state == MissionState.BIRDVIEW_RETURN:
            self.publish_track_enable(False)
            self.publish_status('birdview_return')
        elif self.state == MissionState.FINISH:
            self.publish_track_enable(False)
            self.publish_status('idle')

    def set_state(self, state):
        if self.state == state:
            return
        self.state = state
        self.last_published_goal_name = None
        self.last_goal_reached = False
        self.get_logger().info(f'Mission state -> {state.value}')

    def publish_status(self, perception_mode):
        self.state_pub.publish(String(data=self.state.value))
        self.perception_pub.publish(String(data=perception_mode))

    def publish_track_enable(self, enabled):
        self.track_enable_pub.publish(Bool(data=bool(enabled)))

    def publish_named_goal(self, name):
        if name not in self.points:
            self.get_logger().warn(f'Semantic point not found: {name}')
            self.publish_track_enable(False)
            return

        if name == self.last_published_goal_name:
            self.current_goal_name = name
            return

        self.current_goal_name = name
        self.last_published_goal_name = name
        self.goal_pub.publish(self.pose_from_point(name))

    def republish_current_goal(self):
        if self.current_goal_name:
            self.goal_pub.publish(self.pose_from_point(self.current_goal_name))
            self.last_published_goal_name = self.current_goal_name

    def pose_from_point(self, name):
        pose = self.points[name].get('pose', [0.0, 0.0, 0.0])
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.position.x = float(pose[0])
        msg.pose.position.y = float(pose[1])
        msg.pose.position.z = 0.0
        qz, qw = quaternion_from_yaw(float(pose[2]) if len(pose) > 2 else 0.0)
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg

    def goal_reached_callback(self, msg):
        if not msg.data or self.avoid_active or self.last_goal_reached:
            return

        self.last_goal_reached = True
        if self.state == MissionState.TRACK_TO_TASK_STATION:
            self.set_state(MissionState.WAIT_QR_RESULT)
        elif self.state == MissionState.TRACK_TO_CHANNEL_ENTRY:
            self.prepare_channel_route()
            self.set_state(MissionState.CHANNEL_NAV)
        elif self.state == MissionState.CHANNEL_NAV:
            self.channel_index += 1
            if self.channel_index >= len(self.channel_order):
                self.set_state(MissionState.RETURN_PREPARE)
            else:
                self.last_goal_reached = False
        elif self.state == MissionState.RETURN_PREPARE:
            self.set_state(MissionState.FINISH)

    def qr_callback(self, msg):
        direction = self.parse_direction(msg.data)
        if direction is None:
            return
        self.route_direction = direction
        if self.state == MissionState.WAIT_QR_RESULT:
            self.prepare_channel_route()
            self.set_state(MissionState.TRACK_TO_CHANNEL_ENTRY)

    def sign_int_callback(self, msg):
        self.apply_sign_value(int(msg.data))

    def sign_msg_callback(self, msg):
        self.apply_sign_value(int(msg.sign_data))

    def apply_sign_value(self, value):
        self.route_direction = 'clockwise' if value % 2 else 'anticlockwise'
        if self.state == MissionState.WAIT_QR_RESULT:
            self.prepare_channel_route()
            self.set_state(MissionState.TRACK_TO_CHANNEL_ENTRY)

    def parse_direction(self, text):
        normalized = str(text).strip().lower()
        if normalized in ('clockwise', 'clock wise'):
            return 'clockwise'
        if normalized in ('anticlockwise', 'anti-clockwise', 'anti clockwise', 'counterclockwise'):
            return 'anticlockwise'
        try:
            value = int(normalized)
            return 'clockwise' if value % 2 else 'anticlockwise'
        except ValueError:
            self.get_logger().warn(f'Unknown QR route content: {text}')
            return None

    def prepare_channel_route(self):
        direction = self.route_direction or 'clockwise'
        route = self.routes.get(direction, {})
        self.channel_order = list(route.get('channel_order', []))
        self.channel_index = 0
        if not self.channel_order:
            self.get_logger().warn(f'No channel route configured for {direction}')

    def avoid_active_callback(self, msg):
        self.avoid_active = bool(msg.data)

    def avoid_finished_callback(self, msg):
        if msg.data:
            self.avoid_active = False
            self.last_goal_reached = False
            self.last_published_goal_name = None
            self.republish_current_goal()
            self.publish_track_enable(True)

    def birdview_valid_callback(self, msg):
        self.birdview_valid = bool(msg.data)

    def odom_callback(self, msg):
        del msg


def main(args=None):
    rclpy.init(args=args)
    node = TaskManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_track_enable(False)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
