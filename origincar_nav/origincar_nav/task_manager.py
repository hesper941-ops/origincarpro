#!/usr/bin/env python3

import math
import os
from enum import Enum

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, String


class MissionState(Enum):
    WAIT_START = 'WAIT_START'
    INIT = 'INIT'
    TRACK_TO_TASK_STATION = 'TRACK_TO_TASK_STATION'
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
        self.declare_parameter('publish_rate_hz', 2.0)
        self.declare_parameter('qr_scan_start_dist_to_task_station', 1.0)
        self.declare_parameter('current_goal_republish_period_sec', 1.0)
        self.declare_parameter('debug_allow_numeric_qr_route', False)
        self.declare_parameter('debug_mode', False)
        self.declare_parameter('debug_start_state', 'TRACK_TO_TASK_STATION')
        self.declare_parameter('debug_route_direction', 'clockwise')
        self.declare_parameter('debug_channel_index', 0)
        self.declare_parameter('debug_goal_name', '')
        self.declare_parameter('debug_auto_start', False)

        self.semantic_map = self.load_semantic_map()
        self.map_frame = self.semantic_map.get('map_frame', self.get_parameter('map_frame').value)
        self.points = self.semantic_map.get('points', {})
        self.routes = self.semantic_map.get('routes', {})
        task_thresholds = self.semantic_map.get('task_manager', {})
        self.qr_scan_start_dist_to_task_station = float(
            self.get_parameter('qr_scan_start_dist_to_task_station').value
            or task_thresholds.get('qr_scan_start_dist_to_task_station', 1.0)
        )
        self.debug_allow_numeric_qr_route = bool(
            self.get_parameter('debug_allow_numeric_qr_route').value
        )
        self.current_goal_republish_period_sec = float(
            self.get_parameter('current_goal_republish_period_sec').value
        )
        self.debug_mode = bool(self.get_parameter('debug_mode').value)
        self.debug_start_state = str(self.get_parameter('debug_start_state').value)
        self.debug_route_direction = str(self.get_parameter('debug_route_direction').value).lower()
        self.debug_channel_index = int(self.get_parameter('debug_channel_index').value)
        self.debug_goal_name = str(self.get_parameter('debug_goal_name').value)

        self.state = MissionState.WAIT_START
        self.competition_started = bool(self.get_parameter('debug_auto_start').value)
        self.debug_started = False
        self.route_direction = None
        self.channel_order = []
        self.channel_index = 0
        self.current_goal_name = None
        self.last_published_goal_name = None
        self.last_goal_republish_time = 0.0
        self.avoid_active = False
        self.birdview_valid = False
        self.last_goal_reached = False
        self.current_x = None
        self.current_y = None
        self.qr_scan_active = False

        self.goal_pub = self.create_publisher(PoseStamped, '/current_goal', 10)
        self.track_enable_pub = self.create_publisher(Bool, '/track_enable', 10)
        self.state_pub = self.create_publisher(String, '/mission_state', 10)
        self.perception_pub = self.create_publisher(String, '/perception_mode', 10)

        self.create_subscription(Bool, '/goal_reached', self.goal_reached_callback, 10)
        self.create_subscription(String, '/qrcode_detected/info_result', self.qr_callback, 10)
        self.create_subscription(Bool, '/avoid_active', self.avoid_active_callback, 10)
        self.create_subscription(Bool, '/avoid_finished', self.avoid_finished_callback, 10)
        self.create_subscription(Bool, '/bird_view/p_point_valid', self.birdview_valid_callback, 10)
        self.create_subscription(Bool, '/competition/started', self.competition_started_callback, 10)
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value, self.odom_callback, 10)

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
        if not self.competition_started:
            if self.state != MissionState.WAIT_START:
                self.set_state(MissionState.WAIT_START)
            self.current_goal_name = None
            self.last_published_goal_name = None
            self.publish_track_enable(False)
            self.publish_status('standby')
            return

        if self.state == MissionState.WAIT_START:
            if self.debug_mode:
                self.apply_debug_start()
            else:
                self.set_state(MissionState.INIT)

        if self.state == MissionState.INIT:
            self.set_state(MissionState.TRACK_TO_TASK_STATION)

        if self.avoid_active:
            self.publish_track_enable(False)
            self.publish_status('target_track')
            return

        if self.state == MissionState.TRACK_TO_TASK_STATION:
            self.publish_named_goal(self.goal_name_for_state('task_station'))
            self.publish_track_enable(True)
            self.qr_scan_active = self.should_scan_qr()
            self.publish_status('qr_scan' if self.qr_scan_active else 'target_track')
        elif self.state == MissionState.TRACK_TO_CHANNEL_ENTRY:
            self.publish_named_goal(self.goal_name_for_state('channel_entry'))
            self.publish_track_enable(True)
            self.publish_status('target_track')
        elif self.state == MissionState.CHANNEL_NAV:
            self.publish_track_enable(True)
            self.publish_status('channel_visual_correction')
            if self.channel_index < len(self.channel_order):
                self.publish_named_goal(self.goal_name_for_state(self.channel_order[self.channel_index]))
            else:
                self.set_state(MissionState.RETURN_PREPARE)
        elif self.state == MissionState.RETURN_PREPARE:
            self.publish_status('target_track')
            self.publish_named_goal(self.goal_name_for_state('p_start'))
            self.publish_track_enable(True)
            if self.birdview_valid:
                self.set_state(MissionState.BIRDVIEW_RETURN)
        elif self.state == MissionState.BIRDVIEW_RETURN:
            self.publish_track_enable(False)
            self.publish_status('birdview_return')
        elif self.state == MissionState.FINISH:
            self.publish_track_enable(False)
            self.publish_status('idle')

        self.maybe_republish_current_goal()

    def set_state(self, state):
        if self.state == state:
            return
        self.state = state
        self.last_published_goal_name = None
        self.last_goal_reached = False
        if state != MissionState.TRACK_TO_TASK_STATION:
            self.qr_scan_active = False
        self.get_logger().info(f'Mission state -> {state.value}')

    def apply_debug_start(self):
        if self.debug_started:
            return

        self.route_direction = self.debug_route_direction
        self.prepare_channel_route()
        self.channel_index = max(0, min(self.debug_channel_index, len(self.channel_order)))

        try:
            self.set_state(MissionState[self.debug_start_state])
        except KeyError:
            self.get_logger().warn(
                f'Unknown debug_start_state={self.debug_start_state}, using TRACK_TO_TASK_STATION'
            )
            self.set_state(MissionState.TRACK_TO_TASK_STATION)

        if self.debug_goal_name:
            self.publish_named_goal(self.debug_goal_name)
        self.debug_started = True

    def goal_name_for_state(self, default_name):
        if self.debug_mode and self.debug_goal_name:
            return self.debug_goal_name
        return default_name

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
        self.last_goal_republish_time = self.now_sec()

    def republish_current_goal(self):
        if self.current_goal_name and self.current_goal_name in self.points:
            self.goal_pub.publish(self.pose_from_point(self.current_goal_name))
            self.last_published_goal_name = self.current_goal_name
            self.last_goal_republish_time = self.now_sec()

    def maybe_republish_current_goal(self):
        tracking_states = {
            MissionState.TRACK_TO_TASK_STATION,
            MissionState.TRACK_TO_CHANNEL_ENTRY,
            MissionState.CHANNEL_NAV,
            MissionState.RETURN_PREPARE,
        }
        if self.state not in tracking_states or not self.current_goal_name:
            return
        if self.current_goal_name not in self.points:
            return

        now = self.now_sec()
        if now - self.last_goal_republish_time >= self.current_goal_republish_period_sec:
            self.republish_current_goal()

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
            self.qr_scan_active = True
            self.last_goal_reached = False
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
        if self.state != MissionState.TRACK_TO_TASK_STATION or not self.qr_scan_active:
            return
        direction = self.parse_direction(msg.data)
        if direction is None:
            return
        self.route_direction = direction
        if self.state == MissionState.TRACK_TO_TASK_STATION:
            self.prepare_channel_route()
            self.set_state(MissionState.TRACK_TO_CHANNEL_ENTRY)
            self.publish_named_goal('channel_entry')

    def parse_direction(self, text):
        normalized = str(text).strip().lower()
        if normalized in ('clockwise', 'clock wise'):
            return 'clockwise'
        if normalized in ('anticlockwise', 'anti-clockwise', 'anti clockwise', 'counterclockwise'):
            return 'anticlockwise'
        if normalized == '顺时针':
            return 'clockwise'
        if normalized == '逆时针':
            return 'anticlockwise'
        if self.debug_allow_numeric_qr_route:
            try:
                value = int(normalized)
                self.get_logger().warn('Using debug numeric QR route fallback; this is not the formal rule')
                return 'clockwise' if value % 2 else 'anticlockwise'
            except ValueError:
                pass
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

    def competition_started_callback(self, msg):
        self.competition_started = bool(msg.data)
        if not self.competition_started:
            self.debug_started = False

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def should_scan_qr(self):
        if self.current_x is None or self.current_y is None:
            return False
        task_station = self.points.get('task_station')
        if not task_station:
            return False
        pose = task_station.get('pose', [0.0, 0.0, 0.0])
        distance = math.hypot(float(pose[0]) - self.current_x, float(pose[1]) - self.current_y)
        return distance <= self.qr_scan_start_dist_to_task_station

    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9


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
