#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


class TargetTracker(Node):
    def __init__(self):
        super().__init__('target_tracker')

        # =========================
        # 速度参数
        # =========================
        # 最大前进速度，场地小，不建议太快
        self.declare_parameter('max_linear_speed', 0.30)

        # 最小前进速度：用于中等角度转弯时，避免速度太低
        self.declare_parameter('min_linear_speed', 0.07)

        # 最大角速度
        self.declare_parameter('max_angular_speed', 0.90)

        # 大角度但还没有进入倒车模式时，低速前进摆头
        self.declare_parameter('crawl_linear_speed', 0.08)

        # 是否允许倒车追目标
        self.declare_parameter('enable_reverse_turn', True)

        # 超过这个角度，认为目标在车后方，进入倒车模式
        # 2.35 rad ≈ 135°
        self.declare_parameter('reverse_enter_angle', 2.35)

        # 退出倒车模式角度
        # 加一个退出阈值，避免在临界角度附近反复前进/倒车切换
        # 2.00 rad ≈ 115°
        self.declare_parameter('reverse_exit_angle', 2.00)

        # 倒车速度
        self.declare_parameter('reverse_linear_speed', 0.08)

        # 普通倒车模式的最小距离：目标很远并且在车后方时才使用普通倒车
        # 注意：靠近目标点的智能倒车不使用这个阈值，而使用下面的 close_reverse_* 参数
        self.declare_parameter('reverse_min_dist', 0.45)

        # =========================
        # 靠近目标点智能倒车参数
        # =========================
        # 用途：快到目标点时，如果目标点落在车尾方向，直接小速度倒过去，
        # 避免原地慢慢扭头或者卡住不动。
        self.declare_parameter('enable_close_reverse', True)

        # 距离目标小于这个值，允许进入“近点智能倒车”
        self.declare_parameter('close_reverse_max_dist', 0.65)

        # 距离目标小于这个值，不再倒车，避免到点附近过冲
        # 建议略大于 goal_reached_dist，默认 0.22m
        self.declare_parameter('close_reverse_min_dist', 0.22)

        # 靠近目标点时，角度误差超过这个值才倒车
        # 2.05 rad ≈ 117°，目标明显在车后方才触发
        self.declare_parameter('close_reverse_enter_angle', 2.05)

        # 近点倒车退出角度
        # 1.55 rad ≈ 89°
        self.declare_parameter('close_reverse_exit_angle', 1.55)

        # 近点倒车速度一定要小，防止倒过头
        self.declare_parameter('close_reverse_linear_speed', 0.050)


        # =========================
        # 近点防抖 / 近点前进吃点参数
        # =========================
        # 目标已经很近并且在车头前方附近时，直接认为到点。
        # 这样可以避免小车为了最后几厘米反复扭动。
        self.declare_parameter('relaxed_goal_reached_dist', 0.24)

        # relaxed 到点只在目标大致位于前方时生效，避免目标在车尾还误判到点。
        # 1.40 rad ≈ 80°
        self.declare_parameter('relaxed_goal_reached_angle', 1.40)

        # 近距离且目标在前方/侧前方时，禁止倒车，改用小弧线前进吃点。
        self.declare_parameter('close_forward_max_dist', 0.45)
        # 1.60 rad ≈ 92°
        self.declare_parameter('close_forward_angle', 1.60)
        self.declare_parameter('close_forward_linear_speed', 0.075)
        self.declare_parameter('close_forward_max_angular_speed', 0.45)
        self.declare_parameter('close_forward_angular_gain', 0.45)

        # =========================
        # 控制参数
        # =========================
        self.declare_parameter('k_linear', 0.8)
        self.declare_parameter('k_angular', 0.90)

        # 角速度阻尼，抑制左右摇摆
        self.declare_parameter('k_yaw_rate_damping', 0.35)

        # 小角度死区，避免一直微调
        self.declare_parameter('heading_deadzone', 0.08)

        # 小于这个角度，基本认为方向很好，可以全速走
        # 0.35 rad ≈ 20°
        self.declare_parameter('full_speed_angle', 0.35)

        # 超过这个角度，进入“大角度慢速转弯”
        # 1.55 rad ≈ 89°
        self.declare_parameter('large_turn_angle', 1.55)

        # 距离目标多近开始减速
        self.declare_parameter('slow_down_dist', 0.45)

        # 加速度限制，避免速度突变
        self.declare_parameter('linear_acc_limit', 0.65)
        self.declare_parameter('angular_acc_limit', 2.50)

        # =========================
        # 到点阈值
        # =========================
        self.declare_parameter('goal_reached_dist', 0.15)

        # 你的底盘之前不支持纯自转，所以默认不做终点原地朝向对齐
        self.declare_parameter('align_final_yaw', False)
        self.declare_parameter('goal_reached_yaw', 0.35)
        self.declare_parameter('yaw_deadzone', 0.05)

        # =========================
        # ROS 参数
        # =========================
        self.declare_parameter('semantic_map_file', '')
        self.declare_parameter('odom_origin_mode', 'p_start_map')
        # Compensates a fixed yaw bias between /odom_combined and the semantic map.
        # Official placement: car nose points to map +X, so yaw should be 0 deg.
        self.declare_parameter('yaw_offset_deg', 0.0)
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('odom_topic', '/odom_combined')
        self.declare_parameter('goal_topic', '/current_goal')
        self.declare_parameter('cmd_topic', '/track_cmd_vel')
        self.declare_parameter('enable_topic', '/track_enable')

        self.max_linear_speed = float(self.get_parameter('max_linear_speed').value)
        self.min_linear_speed = float(self.get_parameter('min_linear_speed').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)
        self.crawl_linear_speed = float(self.get_parameter('crawl_linear_speed').value)

        self.enable_reverse_turn = bool(self.get_parameter('enable_reverse_turn').value)
        self.reverse_enter_angle = float(self.get_parameter('reverse_enter_angle').value)
        self.reverse_exit_angle = float(self.get_parameter('reverse_exit_angle').value)
        self.reverse_linear_speed = float(self.get_parameter('reverse_linear_speed').value)
        self.reverse_min_dist = float(self.get_parameter('reverse_min_dist').value)

        self.enable_close_reverse = bool(self.get_parameter('enable_close_reverse').value)
        self.close_reverse_max_dist = float(self.get_parameter('close_reverse_max_dist').value)
        self.close_reverse_min_dist = float(self.get_parameter('close_reverse_min_dist').value)
        self.close_reverse_enter_angle = float(self.get_parameter('close_reverse_enter_angle').value)
        self.close_reverse_exit_angle = float(self.get_parameter('close_reverse_exit_angle').value)
        self.close_reverse_linear_speed = float(self.get_parameter('close_reverse_linear_speed').value)


        self.relaxed_goal_reached_dist = float(self.get_parameter('relaxed_goal_reached_dist').value)
        self.relaxed_goal_reached_angle = float(self.get_parameter('relaxed_goal_reached_angle').value)
        self.close_forward_max_dist = float(self.get_parameter('close_forward_max_dist').value)
        self.close_forward_angle = float(self.get_parameter('close_forward_angle').value)
        self.close_forward_linear_speed = float(self.get_parameter('close_forward_linear_speed').value)
        self.close_forward_max_angular_speed = float(self.get_parameter('close_forward_max_angular_speed').value)
        self.close_forward_angular_gain = float(self.get_parameter('close_forward_angular_gain').value)

        self.k_linear = float(self.get_parameter('k_linear').value)
        self.k_angular = float(self.get_parameter('k_angular').value)
        self.k_yaw_rate_damping = float(self.get_parameter('k_yaw_rate_damping').value)

        self.heading_deadzone = float(self.get_parameter('heading_deadzone').value)
        self.full_speed_angle = float(self.get_parameter('full_speed_angle').value)
        self.large_turn_angle = float(self.get_parameter('large_turn_angle').value)
        self.slow_down_dist = float(self.get_parameter('slow_down_dist').value)

        self.linear_acc_limit = float(self.get_parameter('linear_acc_limit').value)
        self.angular_acc_limit = float(self.get_parameter('angular_acc_limit').value)

        self.align_final_yaw = bool(self.get_parameter('align_final_yaw').value)
        self.goal_reached_yaw = float(self.get_parameter('goal_reached_yaw').value)
        self.yaw_deadzone = float(self.get_parameter('yaw_deadzone').value)

        self.semantic_map = self.load_semantic_map()
        thresholds = self.semantic_map.get('thresholds', {})
        self.odom_origin_mode = str(self.get_parameter('odom_origin_mode').value)
        self.p_start_map = self.load_p_start_map()
        self.yaw_offset_deg = float(self.get_parameter('yaw_offset_deg').value)
        self.yaw_offset = math.radians(self.yaw_offset_deg)
        self.goal_reached_dist = float(
            self.get_parameter('goal_reached_dist').value
            or thresholds.get('goal_reached_dist', 0.15)
        )

        # 近点倒车的最小距离必须大于到点阈值，否则可能刚到点又倒车。
        self.close_reverse_min_dist = max(
            self.close_reverse_min_dist,
            self.goal_reached_dist + 0.03,
        )


        # relaxed 到点距离必须至少大于普通到点阈值，否则没有意义。
        self.relaxed_goal_reached_dist = max(
            self.relaxed_goal_reached_dist,
            self.goal_reached_dist,
        )

        # 最大距离必须大于最小距离，否则关闭近点倒车更安全。
        if self.close_reverse_max_dist <= self.close_reverse_min_dist:
            self.get_logger().warn(
                'close_reverse_max_dist <= close_reverse_min_dist, disable close reverse'
            )
            self.enable_close_reverse = False

        self.current_pose = None
        self.current_yaw = 0.0
        self.current_yaw_rate = 0.0

        self.goal = None
        self.goal_map = None
        self.goal_yaw = 0.0
        self.goal_signature = None

        self.enabled = False
        self.last_warn_time = 0.0
        self.goal_was_reached = False

        self.last_cmd = Twist()
        self.last_control_time = None

        # 是否正在倒车模式
        # 用状态锁存，避免角度在阈值附近来回抖动
        self.reverse_mode = False
        # 'normal'：远距离目标在车后方，倒车追目标
        # 'close' ：靠近目标点时，目标在车尾方向，低速智能倒车到点
        self.reverse_mode_kind = None

        self.last_drive_mode = None
        self.last_mode_log_time = 0.0
        self.last_debug_log_time = 0.0

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

        self.cmd_pub = self.create_publisher(
            Twist,
            self.get_parameter('cmd_topic').value,
            10,
        )

        self.goal_reached_pub = self.create_publisher(Bool, '/goal_reached', 10)
        self.error_pub = self.create_publisher(Vector3, '/tracking_error', 10)

        rate = float(self.get_parameter('control_rate_hz').value)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self.control_loop)

        self.get_logger().info('target_tracker started with reverse-turn controller')
        self.get_logger().info(f'max_linear_speed={self.max_linear_speed}')
        self.get_logger().info(f'min_linear_speed={self.min_linear_speed}')
        self.get_logger().info(f'crawl_linear_speed={self.crawl_linear_speed}')
        self.get_logger().info(f'max_angular_speed={self.max_angular_speed}')
        self.get_logger().info(f'enable_reverse_turn={self.enable_reverse_turn}')
        self.get_logger().info(f'reverse_enter_angle={self.reverse_enter_angle}')
        self.get_logger().info(f'reverse_exit_angle={self.reverse_exit_angle}')
        self.get_logger().info(f'reverse_linear_speed={self.reverse_linear_speed}')
        self.get_logger().info(f'reverse_min_dist={self.reverse_min_dist}')
        self.get_logger().info(f'enable_close_reverse={self.enable_close_reverse}')
        self.get_logger().info(f'close_reverse_max_dist={self.close_reverse_max_dist}')
        self.get_logger().info(f'close_reverse_min_dist={self.close_reverse_min_dist}')
        self.get_logger().info(f'close_reverse_enter_angle={self.close_reverse_enter_angle}')
        self.get_logger().info(f'close_reverse_exit_angle={self.close_reverse_exit_angle}')
        self.get_logger().info(f'close_reverse_linear_speed={self.close_reverse_linear_speed}')
        self.get_logger().info(f'relaxed_goal_reached_dist={self.relaxed_goal_reached_dist}')
        self.get_logger().info(f'relaxed_goal_reached_angle={self.relaxed_goal_reached_angle}')
        self.get_logger().info(f'close_forward_max_dist={self.close_forward_max_dist}')
        self.get_logger().info(f'close_forward_angle={self.close_forward_angle}')
        self.get_logger().info(f'close_forward_linear_speed={self.close_forward_linear_speed}')
        self.get_logger().info(f'full_speed_angle={self.full_speed_angle}')
        self.get_logger().info(f'large_turn_angle={self.large_turn_angle}')
        self.get_logger().info(f'align_final_yaw={self.align_final_yaw}')
        self.get_logger().info(
            'Target tracker odom alignment: '
            f'odom_origin_mode={self.odom_origin_mode}, '
            f'p_start_map=({self.p_start_map[0]:.3f}, {self.p_start_map[1]:.3f}, {self.p_start_map[2]:.3f}), '
            f'yaw_offset_deg={self.yaw_offset_deg:.2f}'
        )

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
                return {}

        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as exc:
            self.get_logger().warn(f'Failed to read tracker semantic map: {exc}')
            return {}

    def load_p_start_map(self):
        points = self.semantic_map.get('points', {})
        pose = points.get('p_start', {}).get('pose', [0.0, 0.0, 0.0])
        yaw = float(pose[2]) if len(pose) > 2 else 0.0
        return float(pose[0]), float(pose[1]), yaw

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        self.current_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.current_yaw_rate = msg.twist.twist.angular.z

    def goal_callback(self, msg):
        signature = self.make_goal_signature(msg)
        self.goal_map = msg.pose
        self.goal = self.map_goal_to_odom_goal(msg.pose)
        self.goal_yaw = yaw_from_quaternion(self.goal.orientation)

        if signature != self.goal_signature:
            self.goal_signature = signature
            self.goal_was_reached = False
            self.reverse_mode = False
            self.reverse_mode_kind = None
            self.last_drive_mode = None
            self.last_cmd = Twist()
            map_yaw = yaw_from_quaternion(msg.pose.orientation)
            self.get_logger().info(
                'Accepted current goal with map-to-odom alignment: '
                f'goal_map=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {map_yaw:.3f}), '
                f'goal_odom=({self.goal.position.x:.3f}, {self.goal.position.y:.3f}, {self.goal_yaw:.3f})'
            )

    def enable_callback(self, msg):
        self.enabled = bool(msg.data)

        if not self.enabled:
            self.reverse_mode = False
            self.reverse_mode_kind = None
            self.last_drive_mode = None
            self.publish_zero()

    def control_loop(self):
        now = self.get_clock().now().nanoseconds / 1e9

        if self.last_control_time is None:
            dt = 1.0 / 20.0
        else:
            dt = max(1e-3, now - self.last_control_time)

        self.last_control_time = now

        if not self.enabled:
            self.reverse_mode = False
            self.reverse_mode_kind = None
            self.last_drive_mode = None
            self.publish_zero()
            return

        if self.current_pose is None or self.goal is None:
            self.reverse_mode = False
            self.reverse_mode_kind = None
            self.last_drive_mode = None
            self.publish_zero()

            if now - self.last_warn_time > 2.0:
                self.get_logger().warn(
                    'Waiting for odom and current goal before tracking'
                )
                self.last_warn_time = now
            return

        dx = self.goal.position.x - self.current_pose.position.x
        dy = self.goal.position.y - self.current_pose.position.y
        distance = math.hypot(dx, dy)

        if distance > 1e-6:
            target_heading = math.atan2(dy, dx)
        else:
            target_heading = self.goal_yaw

        current_yaw_map = normalize_angle(self.current_yaw + self.yaw_offset)
        heading_error = normalize_angle(target_heading - current_yaw_map)
        target_yaw_error = normalize_angle(self.goal_yaw - current_yaw_map)
        abs_heading_error = abs(heading_error)

        self.error_pub.publish(
            Vector3(
                x=distance,
                y=heading_error,
                z=target_yaw_error,
            )
        )

        raw_cmd = Twist()
        reached_msg = Bool()
        reached_msg.data = False
        drive_mode = 'stop'

        # =========================================================
        # 0. 近点 relaxed 到点
        # =========================================================
        # 原来的 goal_reached_dist=0.15m 太精确。
        # 在真实车上，近点几厘米误差会让 atan2 算出的角度剧烈变化，
        # 于是车会为了最后一点点距离反复扭动。
        #
        # 这里增加一个“近点前方到点”判断：
        # 目标已经很近，并且位于车头前方/侧前方，就直接算完成。
        strict_position_reached = distance <= self.goal_reached_dist
        relaxed_position_reached = (
            not strict_position_reached
            and not self.align_final_yaw
            and distance <= self.relaxed_goal_reached_dist
            and abs_heading_error <= self.relaxed_goal_reached_angle
        )

        # =========================================================
        # 1. 还没到目标点：追踪目标点
        # =========================================================
        if not strict_position_reached and not relaxed_position_reached:
            # 近距离并且目标还在车头前方/侧前方时，优先前进吃点，禁止倒车。
            # 这就是解决“明明就在前方附近一点点还想倒车”的关键。
            in_close_forward_zone = (
                distance <= self.close_forward_max_dist
                and abs_heading_error <= self.close_forward_angle
            )

            # =====================================================
            # A. 倒车模式判断
            # =====================================================
            if not self.enable_reverse_turn:
                self.reverse_mode = False
                self.reverse_mode_kind = None
            else:
                in_close_reverse_zone = (
                    self.enable_close_reverse
                    and not in_close_forward_zone
                    and distance >= self.close_reverse_min_dist
                    and distance <= self.close_reverse_max_dist
                )

                if self.reverse_mode:
                    if self.reverse_mode_kind == 'close':
                        exit_min_dist = self.close_reverse_min_dist
                        exit_angle = self.close_reverse_exit_angle
                    else:
                        exit_min_dist = self.reverse_min_dist
                        exit_angle = self.reverse_exit_angle

                    # 如果目标重新回到车头前方附近，也立刻退出倒车。
                    if (
                        distance < exit_min_dist
                        or abs_heading_error < exit_angle
                        or in_close_forward_zone
                    ):
                        self.reverse_mode = False
                        self.reverse_mode_kind = None
                else:
                    # 普通倒车：只有目标明确在车尾后方才倒车。
                    # 阈值从 129° 提高到 135°，减少误判。
                    if (
                        distance > self.reverse_min_dist
                        and abs_heading_error > self.reverse_enter_angle
                    ):
                        self.reverse_mode = True
                        self.reverse_mode_kind = 'normal'

                    # 近点智能倒车：只在目标明显在车后方时触发。
                    # 阈值从 95° 提高到 117°，避免前方附近的点误倒车。
                    elif (
                        in_close_reverse_zone
                        and abs_heading_error > self.close_reverse_enter_angle
                    ):
                        self.reverse_mode = True
                        self.reverse_mode_kind = 'close'

            # =====================================================
            # B. 倒车追踪
            # =====================================================
            if self.reverse_mode:
                reverse_heading = normalize_angle(target_heading + math.pi)
                reverse_error = normalize_angle(reverse_heading - current_yaw_map)
                abs_reverse_error = abs(reverse_error)

                if abs_reverse_error < self.heading_deadzone:
                    angular_cmd = 0.0
                else:
                    angular_cmd = (
                        self.k_angular * reverse_error
                        - self.k_yaw_rate_damping * self.current_yaw_rate
                    )

                raw_cmd.angular.z = clamp(
                    angular_cmd,
                    -self.max_angular_speed,
                    self.max_angular_speed,
                )

                if self.reverse_mode_kind == 'close':
                    drive_mode = 'close_reverse'
                    reverse_speed = self.close_reverse_linear_speed
                    close_range = max(
                        self.close_reverse_max_dist - self.close_reverse_min_dist,
                        1e-3,
                    )
                    close_scale = clamp(
                        (distance - self.close_reverse_min_dist) / close_range,
                        0.25,
                        1.0,
                    )
                    reverse_speed *= close_scale
                else:
                    drive_mode = 'normal_reverse'
                    reverse_speed = self.reverse_linear_speed

                    if distance < self.slow_down_dist:
                        dist_scale = max(
                            0.35,
                            distance / max(self.slow_down_dist, 1e-3),
                        )
                        reverse_speed *= dist_scale

                raw_cmd.linear.x = -clamp(
                    reverse_speed,
                    0.0,
                    self.max_linear_speed,
                )

            # =====================================================
            # C. 近点前进吃点
            # =====================================================
            elif in_close_forward_zone:
                drive_mode = 'close_forward'

                # 近点时不要用大角速度猛拧，容易左右摇摆。
                # 这里用较小角速度，让车走小弧线把点吃掉。
                if abs_heading_error < self.heading_deadzone:
                    angular_cmd = 0.0
                else:
                    angular_cmd = (
                        self.close_forward_angular_gain * heading_error
                        - self.k_yaw_rate_damping * self.current_yaw_rate
                    )

                raw_cmd.angular.z = clamp(
                    angular_cmd,
                    -self.close_forward_max_angular_speed,
                    self.close_forward_max_angular_speed,
                )

                # 距离越接近 relaxed 到点阈值，速度越低。
                close_range = max(
                    self.close_forward_max_dist - self.relaxed_goal_reached_dist,
                    1e-3,
                )
                close_scale = clamp(
                    (distance - self.relaxed_goal_reached_dist) / close_range,
                    0.35,
                    1.0,
                )

                linear_cmd = self.close_forward_linear_speed * close_scale
                raw_cmd.linear.x = clamp(
                    linear_cmd,
                    0.0,
                    self.max_linear_speed,
                )

            # =====================================================
            # D. 正常前进追踪
            # =====================================================
            else:
                drive_mode = 'forward'

                if abs_heading_error < self.heading_deadzone:
                    angular_cmd = 0.0
                else:
                    angular_cmd = (
                        self.k_angular * heading_error
                        - self.k_yaw_rate_damping * self.current_yaw_rate
                    )

                raw_cmd.angular.z = clamp(
                    angular_cmd,
                    -self.max_angular_speed,
                    self.max_angular_speed,
                )

                base_linear = min(
                    self.max_linear_speed,
                    self.k_linear * distance,
                )

                if distance < self.slow_down_dist:
                    dist_scale = max(
                        0.30,
                        distance / max(self.slow_down_dist, 1e-3),
                    )
                    base_linear *= dist_scale

                if abs_heading_error <= self.full_speed_angle:
                    linear_cmd = base_linear

                elif abs_heading_error < self.large_turn_angle:
                    ratio = (
                        (abs_heading_error - self.full_speed_angle)
                        / max(self.large_turn_angle - self.full_speed_angle, 1e-3)
                    )
                    heading_scale = max(0.35, 1.0 - ratio * ratio)
                    linear_cmd = base_linear * heading_scale

                    if distance > self.relaxed_goal_reached_dist:
                        linear_cmd = max(self.min_linear_speed, linear_cmd)

                else:
                    # 大角度但还没有明确到该倒车：
                    # 不再原地慢慢扭，给一点前进速度走弧线。
                    if distance > self.relaxed_goal_reached_dist:
                        linear_cmd = self.crawl_linear_speed
                    else:
                        linear_cmd = 0.0

                raw_cmd.linear.x = clamp(
                    linear_cmd,
                    0.0,
                    self.max_linear_speed,
                )

            self.goal_was_reached = False

        # =========================================================
        # 2. 已经到目标点附近
        # =========================================================
        else:
            self.reverse_mode = False
            self.reverse_mode_kind = None
            drive_mode = 'reached_relaxed' if relaxed_position_reached else 'reached'

            # 你的底盘不支持/不适合纯自转，所以默认到点就算完成。
            if not self.align_final_yaw or relaxed_position_reached:
                raw_cmd.linear.x = 0.0
                raw_cmd.angular.z = 0.0

                reached_msg.data = not self.goal_was_reached
                if reached_msg.data:
                    self.goal_was_reached = True

            else:
                if abs(target_yaw_error) > self.goal_reached_yaw:
                    drive_mode = 'final_yaw_align'
                    if abs(target_yaw_error) < self.yaw_deadzone:
                        angular_cmd = 0.0
                    else:
                        angular_cmd = (
                            self.k_angular * target_yaw_error
                            - self.k_yaw_rate_damping * self.current_yaw_rate
                        )

                    raw_cmd.angular.z = clamp(
                        angular_cmd,
                        -self.max_angular_speed,
                        self.max_angular_speed,
                    )
                    raw_cmd.linear.x = 0.0
                    reached_msg.data = False

                else:
                    raw_cmd.linear.x = 0.0
                    raw_cmd.angular.z = 0.0

                    reached_msg.data = not self.goal_was_reached
                    if reached_msg.data:
                        self.goal_was_reached = True

        cmd = self.limit_cmd_acceleration(raw_cmd, dt)

        self.cmd_pub.publish(cmd)
        self.goal_reached_pub.publish(reached_msg)
        self.log_drive_mode(drive_mode, distance, heading_error, now)
        self.log_tracking_debug(
            now,
            target_heading,
            heading_error,
            current_yaw_map,
            cmd,
        )

        self.last_cmd = cmd

    def log_tracking_debug(self, now, desired_yaw, heading_error, current_yaw_map, cmd):
        if now - self.last_debug_log_time < 1.0:
            return

        self.last_debug_log_time = now
        self.get_logger().info(
            'tracker debug: '
            f'current_x={self.current_pose.position.x:.3f}, '
            f'current_y={self.current_pose.position.y:.3f}, '
            f'current_yaw_deg={math.degrees(self.current_yaw):.2f}, '
            f'yaw_offset_deg={self.yaw_offset_deg:.2f}, '
            f'current_yaw_map_deg={math.degrees(current_yaw_map):.2f}, '
            f'goal_x={self.goal.position.x:.3f}, '
            f'goal_y={self.goal.position.y:.3f}, '
            f'desired_yaw_deg={math.degrees(desired_yaw):.2f}, '
            f'heading_error_deg={math.degrees(heading_error):.2f}, '
            f'linear_cmd={cmd.linear.x:.3f}, '
            f'angular_cmd={cmd.angular.z:.3f}'
        )

    def log_drive_mode(self, mode, distance, heading_error, now):
        # 只在模式变化时打印，避免终端刷屏。
        if mode == self.last_drive_mode:
            return

        self.last_drive_mode = mode
        self.last_mode_log_time = now
        self.get_logger().info(
            '[92m'
            f'tracker mode: {mode}, '
            f'dist={distance:.3f}m, '
            f'heading={math.degrees(heading_error):.1f}deg'
            '[0m'
        )

    def limit_cmd_acceleration(self, raw_cmd, dt):
        cmd = Twist()

        max_dv = self.linear_acc_limit * dt
        max_dw = self.angular_acc_limit * dt

        cmd.linear.x = self.last_cmd.linear.x + clamp(
            raw_cmd.linear.x - self.last_cmd.linear.x,
            -max_dv,
            max_dv,
        )

        cmd.angular.z = self.last_cmd.angular.z + clamp(
            raw_cmd.angular.z - self.last_cmd.angular.z,
            -max_dw,
            max_dw,
        )

        return cmd

    def publish_zero(self):
        zero = Twist()
        self.cmd_pub.publish(zero)
        self.last_cmd = zero

    def make_goal_signature(self, msg):
        return (
            msg.header.frame_id,
            round(float(msg.pose.position.x), 3),
            round(float(msg.pose.position.y), 3),
            round(float(msg.pose.position.z), 3),
            round(float(yaw_from_quaternion(msg.pose.orientation)), 3),
        )

    def map_goal_to_odom_goal(self, pose):
        # semantic_map.yaml uses field/map coordinates. /odom_combined starts near
        # (0, 0) at P, so formal goals are converted with: goal_odom = goal_map - p_start_map.
        if self.odom_origin_mode != 'p_start_map':
            return pose

        converted = PoseStamped().pose
        converted.position.x = float(pose.position.x) - self.p_start_map[0]
        converted.position.y = float(pose.position.y) - self.p_start_map[1]
        converted.position.z = float(pose.position.z)

        goal_yaw_map = yaw_from_quaternion(pose.orientation)
        goal_yaw_odom = normalize_angle(goal_yaw_map - self.p_start_map[2])
        qz = math.sin(goal_yaw_odom * 0.5)
        qw = math.cos(goal_yaw_odom * 0.5)
        converted.orientation.z = qz
        converted.orientation.w = qw
        return converted


def main(args=None):
    rclpy.init(args=args)
    node = TargetTracker()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_zero()
        except Exception:
            pass

        try:
            node.destroy_node()
        except Exception:
            pass

        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
