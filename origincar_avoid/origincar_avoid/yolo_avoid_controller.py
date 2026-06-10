#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os

import rclpy
import yaml
from rclpy.node import Node

from ai_msgs.msg import PerceptionTargets
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool


def yaw_from_quaternion(q):
    """
    从四元数中提取 yaw 角。
    小车平面运动主要用 yaw。
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class YoloAvoidController(Node):
    def __init__(self):
        super().__init__('yolo_avoid_controller')

        # =========================
        # 输入话题
        # =========================
        self.declare_parameter('detection_topic', '/hobot_dnn_detection')
        self.declare_parameter('odom_topic', '/odom_combined')

        # =========================
        # 输出话题
        # 注意：这里和你的 cmd_vel_mux.py 保持一致
        # =========================
        self.declare_parameter('avoid_cmd_topic', '/avoid_cmd_vel')
        self.declare_parameter('avoid_active_topic', '/avoid_active')
        self.declare_parameter('avoid_finished_topic', '/avoid_finished')
        self.declare_parameter('debug_point_topic', '/avoid/obstacle_debug_point')

        # =========================
        # 语义地图边界文件
        # =========================
        self.declare_parameter('semantic_map_file', '')

        # =========================
        # YOLO 检测类别
        # 默认只躲 roadblock
        # 如果模型类别叫 cone，可以在 launch 里传：
        # obstacle_labels: ['roadblock', 'cone']
        # =========================
        self.declare_parameter('obstacle_labels', ['roadblock'])

        # =========================
        # 图像尺寸参数
        # 你的日志是：
        # nv12, h: 400, w: 640
        # 所以这里必须用 640 x 400
        # =========================
        self.declare_parameter('image_width', 640.0)
        self.declare_parameter('image_height', 400.0)

        # =========================
        # 红线避障逻辑参数
        # 图像坐标 y 向下增大：
        # image_height=400, avoid_start_y_ratio=0.60
        # 触发线就是 y=240
        # bottom_y >= 240 时开始避障
        # =========================
        self.declare_parameter('avoid_start_y_ratio', 0.60)
        self.declare_parameter('center_deadzone_ratio', 0.08)
        self.declare_parameter('min_box_area_ratio', 0.003)

        # =========================
        # 避障动作参数
        # 原来 avoid_linear_speed=0.30 太快，容易撞
        # 先用保守速度，稳定后再慢慢加
        # =========================
        self.declare_parameter('avoid_duration_sec', 1.20)
        self.declare_parameter('cooldown_sec', 0.45)
        self.declare_parameter('avoid_linear_speed', 0.16)
        self.declare_parameter('avoid_angular_speed', 0.75)

        # =========================
        # 边界保护参数
        # 原来 lateral_check_dist=0.35 靠边时很容易判定左右都越界
        # 先降到 0.22
        # =========================
        self.declare_parameter('lateral_check_dist', 0.22)
        self.declare_parameter('obstacle_debug_forward_dist', 0.6)
        self.declare_parameter('obstacle_debug_lateral_scale', 0.5)

        # 发布控制频率
        self.declare_parameter('publish_rate_hz', 20.0)

        # =========================
        # 读取参数
        # =========================
        self.detection_topic = self.get_parameter('detection_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value

        self.avoid_cmd_topic = self.get_parameter('avoid_cmd_topic').value
        self.avoid_active_topic = self.get_parameter('avoid_active_topic').value
        self.avoid_finished_topic = self.get_parameter('avoid_finished_topic').value
        self.debug_point_topic = self.get_parameter('debug_point_topic').value

        self.obstacle_labels = {
            str(label).strip().lower()
            for label in self.get_parameter('obstacle_labels').value
        }

        self.image_width = float(self.get_parameter('image_width').value)
        self.image_height = float(self.get_parameter('image_height').value)

        self.avoid_start_y_ratio = float(
            self.get_parameter('avoid_start_y_ratio').value
        )
        self.center_deadzone_ratio = float(
            self.get_parameter('center_deadzone_ratio').value
        )
        self.min_box_area_ratio = float(
            self.get_parameter('min_box_area_ratio').value
        )

        self.avoid_duration_sec = float(
            self.get_parameter('avoid_duration_sec').value
        )
        self.cooldown_sec = float(
            self.get_parameter('cooldown_sec').value
        )
        self.avoid_linear_speed = float(
            self.get_parameter('avoid_linear_speed').value
        )
        self.avoid_angular_speed = float(
            self.get_parameter('avoid_angular_speed').value
        )

        self.lateral_check_dist = float(
            self.get_parameter('lateral_check_dist').value
        )
        self.obstacle_debug_forward_dist = float(
            self.get_parameter('obstacle_debug_forward_dist').value
        )
        self.obstacle_debug_lateral_scale = float(
            self.get_parameter('obstacle_debug_lateral_scale').value
        )

        # =========================
        # 地图边界
        # =========================
        self.boundary = self.load_boundary()

        # =========================
        # 小车当前位姿
        # =========================
        self.current_x = None
        self.current_y = None
        self.current_yaw = 0.0

        # =========================
        # 避障状态
        # =========================
        self.avoid_active = False
        self.avoid_direction = 0
        self.avoid_end_time = 0.0
        self.cooldown_until = 0.0

        # =========================
        # 订阅 YOLO 检测
        # =========================
        self.create_subscription(
            PerceptionTargets,
            self.detection_topic,
            self.detection_callback,
            10,
        )

        # =========================
        # 订阅里程计
        # =========================
        self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10,
        )

        # =========================
        # 发布避障速度：/avoid_cmd_vel
        # =========================
        self.cmd_pub = self.create_publisher(
            Twist,
            self.avoid_cmd_topic,
            10
        )

        # =========================
        # 发布避障是否激活：/avoid_active
        # =========================
        self.active_pub = self.create_publisher(
            Bool,
            self.avoid_active_topic,
            10
        )

        # =========================
        # 发布避障完成信号
        # =========================
        self.finished_pub = self.create_publisher(
            Bool,
            self.avoid_finished_topic,
            10
        )

        # =========================
        # 发布调试点
        # =========================
        self.debug_point_pub = self.create_publisher(
            PointStamped,
            self.debug_point_topic,
            10
        )

        rate = float(self.get_parameter('publish_rate_hz').value)
        self.timer = self.create_timer(
            1.0 / max(rate, 1.0),
            self.tick
        )

        self.get_logger().info('yolo_avoid_controller started')
        self.get_logger().info(f'Detection topic: {self.detection_topic}')
        self.get_logger().info(f'Odom topic: {self.odom_topic}')
        self.get_logger().info(f'Avoid cmd topic: {self.avoid_cmd_topic}')
        self.get_logger().info(f'Avoid active topic: {self.avoid_active_topic}')
        self.get_logger().info(f'Obstacle labels: {self.obstacle_labels}')
        self.get_logger().info(f'Image size: {self.image_width} x {self.image_height}')
        self.get_logger().info(f'Avoid start y ratio: {self.avoid_start_y_ratio}')
        self.get_logger().info(
            f'Avoid trigger y: {self.image_height * self.avoid_start_y_ratio:.1f}'
        )
        self.get_logger().info(f'Center deadzone ratio: {self.center_deadzone_ratio}')
        self.get_logger().info(f'Avoid duration: {self.avoid_duration_sec}')
        self.get_logger().info(f'Cooldown: {self.cooldown_sec}')
        self.get_logger().info(f'Avoid linear speed: {self.avoid_linear_speed}')
        self.get_logger().info(f'Avoid angular speed: {self.avoid_angular_speed}')
        self.get_logger().info(f'Lateral check dist: {self.lateral_check_dist}')
        self.get_logger().info(f'Map boundary: {self.boundary}')

    def load_boundary(self):
        """
        读取 semantic_map.yaml 里的 boundary。
        如果读取失败，就使用 fallback 边界。
        """
        path = self.get_parameter('semantic_map_file').value

        if not path:
            try:
                from ament_index_python.packages import get_package_share_directory
                path = os.path.join(
                    get_package_share_directory('origincar_nav'),
                    'config',
                    'semantic_map.yaml'
                )
            except Exception:
                path = ''

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                boundary = data.get('boundary', {})

                if not boundary:
                    self.get_logger().warn(
                        f'No boundary field found in semantic map: {path}'
                    )

                return boundary

        except Exception as exc:
            self.get_logger().warn(
                f'Using fallback map boundary; failed to read semantic map: {exc}'
            )
            return {
                'x_min': 0.0,
                'x_max': 4.0,
                'y_min': 0.0,
                'y_max': 3.0,
            }

    def odom_callback(self, msg):
        """
        保存小车当前位姿。
        """
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

    def detection_callback(self, msg):
        """
        收到 YOLO 检测结果后，判断是否需要开始避障。
        """
        now = self.now_sec()

        # 已经在避障，或者还在冷却时间内，不重复触发
        if self.avoid_active or now < self.cooldown_until:
            return

        obstacle = self.select_obstacle_by_redline(msg)

        if obstacle is None:
            return

        direction = self.choose_direction(obstacle['center_x'])

        self.publish_obstacle_debug_point(obstacle['center_x'])

        self.avoid_active = True
        self.avoid_direction = direction
        self.avoid_end_time = now + self.avoid_duration_sec

        self.publish_active(True)

        self.get_logger().info(
            f'Start avoid: center_x={obstacle["center_x"]:.1f}, '
            f'bottom_y={obstacle["bottom_y"]:.1f}, '
            f'trigger_y={self.image_height * self.avoid_start_y_ratio:.1f}, '
            f'direction={self.avoid_direction}'
        )

    def select_obstacle_by_redline(self, msg):
        """
        使用红线逻辑选择障碍物。

        逻辑：
        1. 只看 obstacle_labels，比如 roadblock
        2. 计算每个框的 bottom_y 和 center_x
        3. 选择 bottom_y 最大的那个，也就是画面最下面、通常最近的障碍物
        4. 如果 bottom_y 超过触发线，则返回该障碍物
        5. 否则返回 None

        注意：
        图像坐标系 y 轴向下增大。
        所以 bottom_y 越大，障碍物越靠近画面底部，通常越近。
        """
        nearest = None
        max_bottom_y = -1.0

        avoid_start_y = self.image_height * self.avoid_start_y_ratio
        image_area = max(self.image_width * self.image_height, 1.0)

        for target in msg.targets:
            label = str(target.type).strip().lower()

            if label not in self.obstacle_labels:
                continue

            for roi in target.rois:
                rect = roi.rect

                width = float(rect.width)
                height = float(rect.height)

                if width <= 0.0 or height <= 0.0:
                    continue

                area_ratio = (width * height) / image_area

                # 太小的框一般是远处或误检，先过滤掉
                if area_ratio < self.min_box_area_ratio:
                    continue

                x1 = float(rect.x_offset)
                y1 = float(rect.y_offset)

                center_x = x1 + width * 0.5
                bottom_y = y1 + height

                if bottom_y > max_bottom_y:
                    max_bottom_y = bottom_y
                    nearest = {
                        'label': label,
                        'center_x': center_x,
                        'bottom_y': bottom_y,
                        'area_ratio': area_ratio,
                    }

        if nearest is None:
            return None

        # 红线没到触发线，不接管
        # bottom_y 越大，说明框越靠近画面底部。
        # 所以 bottom_y < avoid_start_y 时，不避障。
        if nearest['bottom_y'] < avoid_start_y:
            return None

        return nearest

    def choose_direction(self, center_x):
        """
        根据障碍物横向位置和地图边界决定避障方向。

        返回值：
        1  ：向左转
        -1 ：向右转
        0  ：停车保护
        """
        center_offset = center_x - self.image_width * 0.5
        deadzone = self.image_width * self.center_deadzone_ratio

        left_ok = self.side_inside_boundary(left=True)
        right_ok = self.side_inside_boundary(left=False)

        # 两边都不能走，停车保护
        if not left_ok and not right_ok:
            self.get_logger().warn(
                'Protective stop: both avoidance candidates are outside boundary.'
            )
            return 0

        # 左边不能走，只能向右
        if not left_ok:
            return -1

        # 右边不能走，只能向左
        if not right_ok:
            return 1

        # 障碍物在左边，小车向右绕
        if center_offset < -deadzone:
            return -1

        # 障碍物在右边，小车向左绕
        if center_offset > deadzone:
            return 1

        # 障碍物接近中心，默认向左绕
        return 1

    def side_inside_boundary(self, left):
        """
        检查小车左侧或右侧的候选点是否还在地图 boundary 里面。
        """
        if self.current_x is None or self.current_y is None:
            # 还没收到里程计时，不阻止避障
            return True

        sign = 1.0 if left else -1.0

        candidate_x = self.current_x - sign * math.sin(self.current_yaw) * self.lateral_check_dist
        candidate_y = self.current_y + sign * math.cos(self.current_yaw) * self.lateral_check_dist

        x_min = float(self.boundary.get('x_min', -999.0))
        x_max = float(self.boundary.get('x_max', 999.0))
        y_min = float(self.boundary.get('y_min', -999.0))
        y_max = float(self.boundary.get('y_max', 999.0))

        return (
            x_min <= candidate_x <= x_max
            and y_min <= candidate_y <= y_max
        )

    def tick(self):
        """
        定时发布避障速度。
        """
        cmd = Twist()
        now = self.now_sec()

        if self.avoid_active and now <= self.avoid_end_time:
            if self.avoid_direction != 0:
                cmd.linear.x = self.avoid_linear_speed
                cmd.angular.z = self.avoid_angular_speed * self.avoid_direction
            else:
                # direction == 0，停车保护
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0

            self.publish_cmd(cmd)
            self.publish_active(True)
            return

        # 避障刚刚结束
        if self.avoid_active:
            self.avoid_active = False
            self.cooldown_until = now + self.cooldown_sec

            self.publish_cmd(Twist())
            self.publish_active(False)
            self.publish_finished()

            self.get_logger().info('Avoid finished, return control to tracker')

    def publish_cmd(self, cmd):
        self.cmd_pub.publish(cmd)

    def publish_active(self, active):
        msg = Bool()
        msg.data = bool(active)
        self.active_pub.publish(msg)

    def publish_finished(self):
        msg = Bool()
        msg.data = True
        self.finished_pub.publish(msg)

    def publish_obstacle_debug_point(self, center_x):
        """
        发布一个调试点，方便 RViz 显示估计的障碍物位置。
        注意：这个点只是粗略估计，不是真实深度定位。
        """
        if self.current_x is None or self.current_y is None:
            return

        lateral_offset = (
            (center_x - self.image_width * 0.5)
            / max(self.image_width, 1.0)
            * self.obstacle_debug_lateral_scale
        )

        point = PointStamped()
        point.header.stamp = self.get_clock().now().to_msg()
        point.header.frame_id = 'odom_combined'

        point.point.x = (
            self.current_x
            + math.cos(self.current_yaw) * self.obstacle_debug_forward_dist
            - math.sin(self.current_yaw) * lateral_offset
        )

        point.point.y = (
            self.current_y
            + math.sin(self.current_yaw) * self.obstacle_debug_forward_dist
            + math.cos(self.current_yaw) * lateral_offset
        )

        point.point.z = 0.0

        self.debug_point_pub.publish(point)

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
        # 退出前释放避障控制权并停车
        node.publish_cmd(Twist())
        node.publish_active(False)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()