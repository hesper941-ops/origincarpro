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


ANSI_GREEN = '\033[92m'
ANSI_BOLD = '\033[1m'
ANSI_RESET = '\033[0m'


class MissionState(Enum):
    WAIT_START = 'WAIT_START'
    INIT = 'INIT'
    TRACK_TO_TASK_STATION = 'TRACK_TO_TASK_STATION'
    TRACK_TO_CHANNEL_ENTRY = 'TRACK_TO_CHANNEL_ENTRY'
    TRACK_TO_CHANNEL_ENTRY_LOWER = 'TRACK_TO_CHANNEL_ENTRY_LOWER'
    TRACK_TO_CHANNEL_ENTRY_UPPER = 'TRACK_TO_CHANNEL_ENTRY_UPPER'
    CHANNEL_NAV = 'CHANNEL_NAV'
    RETURN_PREPARE = 'RETURN_PREPARE'
    BIRDVIEW_RETURN = 'BIRDVIEW_RETURN'
    SINGLE_GOAL_TRACK = 'SINGLE_GOAL_TRACK'
    SINGLE_GOAL_REACHED = 'SINGLE_GOAL_REACHED'
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
        self.declare_parameter('debug_allow_numeric_qr_route', True)
        self.declare_parameter('debug_mode', False)
        self.declare_parameter('debug_start_state', 'TRACK_TO_TASK_STATION')
        self.declare_parameter('debug_route_direction', 'clockwise')
        self.declare_parameter('debug_channel_index', 0)
        self.declare_parameter('debug_goal_name', '')
        self.declare_parameter('debug_auto_start', False)
        self.declare_parameter('enable_birdview_return', False)
        self.declare_parameter('single_goal_mode', False)
        self.declare_parameter('single_goal_name', 'task_station')

        # 新增：二维码内容无法解析方向时，默认走哪个方向
        self.declare_parameter('default_route_direction', 'clockwise')

        self.semantic_map = self.load_semantic_map()
        self.map_frame = self.semantic_map.get('map_frame', self.get_parameter('map_frame').value)
        self.points = self.semantic_map.get('points', {})
        self.p_start_map = self.load_p_start_map()
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
        self.enable_birdview_return = bool(self.get_parameter('enable_birdview_return').value)
        self.single_goal_mode = bool(self.get_parameter('single_goal_mode').value)
        self.single_goal_name = str(self.get_parameter('single_goal_name').value)
        self.single_goal_valid = (not self.single_goal_mode) or self.single_goal_name in self.points
        self.single_goal_error_logged = False

        self.default_route_direction = str(
            self.get_parameter('default_route_direction').value
        ).strip().lower()

        if self.default_route_direction not in ('clockwise', 'anticlockwise'):
            self.get_logger().warn(
                f'Invalid default_route_direction={self.default_route_direction}, use clockwise'
            )
            self.default_route_direction = 'clockwise'

        if self.single_goal_mode:
            if self.single_goal_valid:
                self.get_logger().info(f'Single goal calibration mode enabled: {self.single_goal_name}')
            else:
                self.get_logger().error(
                    f'single_goal_name={self.single_goal_name} not found in semantic_map.yaml'
                )
                self.single_goal_error_logged = True

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

        # 新增：防止同一个二维码短时间内重复触发
        self.qr_already_used = False

        self.goal_pub = self.create_publisher(PoseStamped, '/current_goal', 10)
        self.track_enable_pub = self.create_publisher(Bool, '/track_enable', 10)
        self.state_pub = self.create_publisher(String, '/mission_state', 10)
        self.perception_pub = self.create_publisher(String, '/perception_mode', 10)

        # 新增：任务事件广播。
        # 终端会用绿色打印，同时也会发布到 /mission_event，方便你 ros2 topic echo 查看。
        self.mission_event_pub = self.create_publisher(String, '/mission_event', 10)

        self.create_subscription(Bool, '/goal_reached', self.goal_reached_callback, 10)
        self.create_subscription(String, '/qrcode_detected/info_result', self.qr_callback, 10)
        self.create_subscription(Bool, '/avoid_active', self.avoid_active_callback, 10)
        self.create_subscription(Bool, '/avoid_finished', self.avoid_finished_callback, 10)
        self.create_subscription(Bool, '/bird_view/p_point_valid', self.birdview_valid_callback, 10)
        self.create_subscription(Bool, '/competition/started', self.competition_started_callback, 10)
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value, self.odom_callback, 10)

        rate = float(self.get_parameter('publish_rate_hz').value)
        self.timer = self.create_timer(1.0 / max(rate, 0.5), self.tick)

    def broadcast_green(self, text):
        """在终端用绿色醒目提示，并同步发布到 /mission_event。"""
        message = str(text)
        self.get_logger().info(f'{ANSI_GREEN}{ANSI_BOLD}{message}{ANSI_RESET}')

        # mission_event_pub 在 __init__ 后才存在；加保护避免初始化阶段异常。
        if hasattr(self, 'mission_event_pub'):
            self.mission_event_pub.publish(String(data=message))

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
            return {
                'points': {},
                'routes': {},
                'map_frame': self.get_parameter('map_frame').value,
            }

    def load_p_start_map(self):
        pose = self.points.get('p_start', {}).get('pose', [0.0, 0.0, 0.0])
        return float(pose[0]), float(pose[1])

    def tick(self):
        if not self.competition_started:
            if self.state != MissionState.WAIT_START:
                self.set_state(MissionState.WAIT_START)

            self.current_goal_name = None
            self.last_published_goal_name = None
            self.qr_already_used = False

            self.publish_track_enable(False)
            self.publish_status('standby')
            return

        if self.state == MissionState.WAIT_START:
            if self.single_goal_mode:
                self.tick_single_goal()
                return

            if self.debug_mode:
                self.apply_debug_start()
            else:
                self.set_state(MissionState.INIT)

        if self.single_goal_mode:
            self.tick_single_goal()
            return

        if self.state == MissionState.INIT:
            self.qr_already_used = False
            self.set_state(MissionState.TRACK_TO_TASK_STATION)

        if self.avoid_active:
            self.publish_track_enable(False)
            self.publish_status('target_track')
            return

        if self.state == MissionState.TRACK_TO_TASK_STATION:
            self.publish_named_goal(self.goal_name_for_state('task_station'))
            self.publish_track_enable(True)

            # 这个只影响 perception_mode 的显示，不再限制二维码是否能触发下一目标
            self.qr_scan_active = self.should_scan_qr()
            self.publish_status('qr_scan' if self.qr_scan_active else 'target_track')

        elif self.state == MissionState.TRACK_TO_CHANNEL_ENTRY:
            self.publish_named_goal(self.goal_name_for_state('channel_entry'))
            self.publish_track_enable(True)
            self.publish_status('target_track')
        elif self.state == MissionState.TRACK_TO_CHANNEL_ENTRY_LOWER:
            self.publish_named_goal(self.goal_name_for_state('channel_entry_lower'))
            self.publish_track_enable(True)
            self.publish_status('target_track')
        elif self.state == MissionState.TRACK_TO_CHANNEL_ENTRY_UPPER:
            self.publish_named_goal(self.goal_name_for_state('channel_entry_upper'))
            self.publish_track_enable(True)
            self.publish_status('target_track')
        elif self.state == MissionState.CHANNEL_NAV:
            self.publish_track_enable(True)
            self.publish_status('channel_visual_correction')

            if self.channel_index < len(self.channel_order):
                self.publish_named_goal(
                    self.goal_name_for_state(self.channel_order[self.channel_index])
                )
            else:
                self.set_state(MissionState.RETURN_PREPARE)

        elif self.state == MissionState.RETURN_PREPARE:
            self.publish_status('target_track')
            self.publish_named_goal(self.goal_name_for_state('p_start'))
            self.publish_track_enable(True)
            if self.enable_birdview_return and self.birdview_valid:
                self.set_state(MissionState.BIRDVIEW_RETURN)

        elif self.state == MissionState.BIRDVIEW_RETURN:
            self.publish_track_enable(False)
            self.publish_status('birdview_return')

        elif self.state == MissionState.FINISH:
            self.publish_track_enable(False)
            self.publish_status('idle')

        self.maybe_republish_current_goal()

    def tick_single_goal(self):
        if not self.single_goal_valid:
            if not self.single_goal_error_logged:
                self.get_logger().error(
                    f'single_goal_name={self.single_goal_name} not found in semantic_map.yaml'
                )
                self.single_goal_error_logged = True
            self.publish_track_enable(False)
            self.publish_status('single_goal_invalid')
            return

        if self.state == MissionState.SINGLE_GOAL_REACHED:
            self.publish_track_enable(False)
            self.publish_status('single_goal_reached')
            return

        if self.state != MissionState.SINGLE_GOAL_TRACK:
            self.set_state(MissionState.SINGLE_GOAL_TRACK)

        self.publish_named_goal(self.single_goal_name)
        self.publish_track_enable(True)
        self.publish_status('single_goal_track')
        self.maybe_republish_current_goal()

    def set_state(self, state):
        if self.state == state:
            return

        self.state = state
        self.last_published_goal_name = None
        self.last_goal_reached = False

        if state != MissionState.TRACK_TO_TASK_STATION:
            self.qr_scan_active = False

        self.broadcast_green(f'📍 任务状态切换 -> {state.value}')

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

        pose = self.points[name].get('pose', [0.0, 0.0, 0.0])
        yaw = float(pose[2]) if len(pose) > 2 else 0.0
        self.broadcast_green(
            f'🚗 前往目标点 -> {name}  '
            f'x={float(pose[0]):.2f}, y={float(pose[1]):.2f}, yaw={yaw:.2f}'
        )

    def republish_current_goal(self):
        if self.current_goal_name and self.current_goal_name in self.points:
            self.goal_pub.publish(self.pose_from_point(self.current_goal_name))
            self.last_published_goal_name = self.current_goal_name
            self.last_goal_republish_time = self.now_sec()

    def maybe_republish_current_goal(self):
        tracking_states = {
            MissionState.TRACK_TO_TASK_STATION,
            MissionState.TRACK_TO_CHANNEL_ENTRY,
            MissionState.TRACK_TO_CHANNEL_ENTRY_LOWER,
            MissionState.TRACK_TO_CHANNEL_ENTRY_UPPER,
            MissionState.CHANNEL_NAV,
            MissionState.RETURN_PREPARE,
            MissionState.SINGLE_GOAL_TRACK,
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

        yaw = float(pose[2]) if len(pose) > 2 else 0.0
        qz, qw = quaternion_from_yaw(yaw)
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        return msg

    def goal_reached_callback(self, msg):
        if not msg.data or self.avoid_active or self.last_goal_reached:
            return

        self.last_goal_reached = True

        finished_goal = self.current_goal_name or 'unknown'
        self.broadcast_green(f'✅ 完成目标点 -> {finished_goal}')

        if self.state == MissionState.TRACK_TO_TASK_STATION:
            # 到达任务站但还没识别到二维码时，不切下一目标，继续等待二维码
            self.qr_scan_active = True
            self.last_goal_reached = False

        elif self.state == MissionState.SINGLE_GOAL_TRACK:
            self.publish_track_enable(False)
            self.set_state(MissionState.SINGLE_GOAL_REACHED)

        elif self.state == MissionState.TRACK_TO_CHANNEL_ENTRY:
            self.prepare_channel_route()
            self.set_state(MissionState.CHANNEL_NAV)
        elif self.state == MissionState.TRACK_TO_CHANNEL_ENTRY_LOWER:
            self.set_state(MissionState.TRACK_TO_CHANNEL_ENTRY_UPPER)
        elif self.state == MissionState.TRACK_TO_CHANNEL_ENTRY_UPPER:
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
        """
        关键改动：
        一旦在 TRACK_TO_TASK_STATION 阶段识别到二维码，
        不管二维码内容是不是标准方向，都立刻前往 channel_entry。
        """

        if self.state != MissionState.TRACK_TO_TASK_STATION:
            return

        if self.qr_already_used:
            return

        qr_text = str(msg.data).strip()
        if not qr_text:
            return

        self.qr_already_used = True

        self.broadcast_green(
            f'QR detected -> {qr_text}; switch to next target'
        )

        direction = self.parse_direction(qr_text)

        if direction is None:
            self.get_logger().warn(
                f'QR content "{qr_text}" is not a valid route direction, '
                f'use default route direction: {self.default_route_direction}'
            )
            direction = self.default_route_direction

        self.route_direction = direction
        self.prepare_channel_route()
        self.broadcast_green(
            f'Route direction -> {direction}; channel order -> {self.channel_order}'
        )

        self.set_state(MissionState.TRACK_TO_CHANNEL_ENTRY_LOWER)
        self.publish_named_goal('channel_entry_lower')
        self.publish_track_enable(True)
        self.publish_status('target_track')

    def parse_direction(self, text):
        normalized = str(text).strip().lower()

        if normalized in ('clockwise', 'clock wise'):
            return 'clockwise'

        if normalized in (
            'anticlockwise',
            'anti-clockwise',
            'anti clockwise',
            'counterclockwise',
            'counter-clockwise',
            'counter clockwise',
        ):
            return 'anticlockwise'

        if normalized in ('顺时针', '順時針'):
            return 'clockwise'

        if normalized in ('逆时针', '逆時針'):
            return 'anticlockwise'

        if self.debug_allow_numeric_qr_route:
            try:
                value = int(normalized)
            except ValueError:
                pass
            else:
                self.get_logger().warn(
                    'Using debug numeric QR route fallback; this is not the formal rule'
                )
                return 'clockwise' if value % 2 else 'anticlockwise'

        self.get_logger().warn(f'Unknown QR route content: {text}')
        return None

    def prepare_channel_route(self):
        direction = self.route_direction or self.default_route_direction

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
            self.broadcast_green('🟢 避障结束，恢复当前目标跟踪')
            self.republish_current_goal()
            self.publish_track_enable(True)

    def birdview_valid_callback(self, msg):
        self.birdview_valid = bool(msg.data)

    def competition_started_callback(self, msg):
        self.competition_started = bool(msg.data)

        if not self.competition_started:
            self.debug_started = False
            self.qr_already_used = False

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
        # /odom_combined starts near (0, 0) at P; semantic points are in field/map
        # coordinates, so QR scan distance is checked in map coordinates.
        current_map_x = self.current_x + self.p_start_map[0]
        current_map_y = self.current_y + self.p_start_map[1]
        distance = math.hypot(float(pose[0]) - current_map_x, float(pose[1]) - current_map_y)
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
