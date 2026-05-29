#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import Empty
from geometry_msgs.msg import PoseStamped

from nav2_msgs.action import NavigateToPose


def yaw_to_quaternion(yaw):
    """
    2D yaw 转 quaternion
    """
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)

    return 0.0, 0.0, qz, qw


class WaypointManager(Node):
    def __init__(self):
        super().__init__('waypoint_manager')

        # =========================
        # 参数
        # =========================
        self.declare_parameter('global_frame', 'map')

        # 路点坐标
        # 注意：三个数组长度必须一样
        self.declare_parameter('waypoint_x', [0.0])
        self.declare_parameter('waypoint_y', [0.0])
        self.declare_parameter('waypoint_yaw', [0.0])

        # 是否循环巡航
        self.declare_parameter('loop', True)

        # 收到避障完成后，至少间隔多少秒才允许再次重发目标
        # 防止 /yolo_avoid_finished 抖动导致疯狂重发
        self.declare_parameter('replan_cooldown_sec', 1.0)

        # 是否启动后自动发送第一个目标点
        self.declare_parameter('autostart', True)

        self.global_frame = self.get_parameter('global_frame').value

        self.waypoint_x = list(self.get_parameter('waypoint_x').value)
        self.waypoint_y = list(self.get_parameter('waypoint_y').value)
        self.waypoint_yaw = list(self.get_parameter('waypoint_yaw').value)

        self.loop = bool(self.get_parameter('loop').value)
        self.replan_cooldown_sec = float(
            self.get_parameter('replan_cooldown_sec').value
        )
        self.autostart = bool(self.get_parameter('autostart').value)

        # =========================
        # 检查路点参数
        # =========================
        if not (
            len(self.waypoint_x)
            == len(self.waypoint_y)
            == len(self.waypoint_yaw)
        ):
            raise RuntimeError(
                'waypoint_x, waypoint_y, waypoint_yaw length must be same'
            )

        if len(self.waypoint_x) == 0:
            raise RuntimeError('No waypoint configured')

        self.total_waypoints = len(self.waypoint_x)

        # 当前正在导航的路点编号
        self.current_index = 0

        # 当前是否已经发送过目标
        self.goal_sent = False

        # goal 序号，用来忽略旧 goal 的回调
        self.goal_seq = 0

        # 上一次重规划时间
        self.last_replan_time = None

        # =========================
        # Nav2 NavigateToPose Action Client
        # =========================
        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

        # =========================
        # 订阅 YOLO 避障完成事件
        # =========================
        self.avoid_finished_sub = self.create_subscription(
            Empty,
            '/yolo_avoid_finished',
            self.avoid_finished_callback,
            10
        )

        # 启动后稍微等一下 Nav2
        self.start_timer = self.create_timer(
            1.0,
            self.start_timer_callback
        )

        self.get_logger().info('waypoint_manager started')
        self.get_logger().info(f'global_frame: {self.global_frame}')
        self.get_logger().info(f'total waypoints: {self.total_waypoints}')
        self.get_logger().info(f'loop: {self.loop}')
        self.get_logger().info(
            f'replan_cooldown_sec: {self.replan_cooldown_sec}'
        )

        for i in range(self.total_waypoints):
            self.get_logger().info(
                f'waypoint[{i}]: '
                f'x={self.waypoint_x[i]:.3f}, '
                f'y={self.waypoint_y[i]:.3f}, '
                f'yaw={self.waypoint_yaw[i]:.3f}'
            )

    def start_timer_callback(self):
        """
        启动后自动发送第一个目标点
        """
        if not self.autostart:
            self.start_timer.cancel()
            return

        if self.goal_sent:
            self.start_timer.cancel()
            return

        if not self.nav_client.server_is_ready():
            self.get_logger().info('Waiting for Nav2 navigate_to_pose server...')
            return

        self.get_logger().info('Nav2 navigate_to_pose server is ready')
        self.send_current_goal(reason='autostart')
        self.start_timer.cancel()

    def make_goal_msg(self, index):
        """
        根据路点编号生成 NavigateToPose goal
        """
        goal_msg = NavigateToPose.Goal()

        pose = PoseStamped()
        pose.header.frame_id = self.global_frame
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = float(self.waypoint_x[index])
        pose.pose.position.y = float(self.waypoint_y[index])
        pose.pose.position.z = 0.0

        qx, qy, qz, qw = yaw_to_quaternion(float(self.waypoint_yaw[index]))

        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        goal_msg.pose = pose

        return goal_msg

    def send_current_goal(self, reason='normal'):
        """
        发送当前路点给 Nav2。

        关键点：
        再次发送同一个目标点，会让 Nav2 从当前机器人位置重新规划。
        """
        if not self.nav_client.server_is_ready():
            self.get_logger().warn(
                'Nav2 navigate_to_pose server not ready, cannot send goal'
            )
            return

        self.goal_seq += 1
        current_seq = self.goal_seq

        goal_msg = self.make_goal_msg(self.current_index)

        self.get_logger().info(
            f'Send goal seq={current_seq}, '
            f'index={self.current_index}, '
            f'reason={reason}, '
            f'x={self.waypoint_x[self.current_index]:.3f}, '
            f'y={self.waypoint_y[self.current_index]:.3f}, '
            f'yaw={self.waypoint_yaw[self.current_index]:.3f}'
        )

        send_future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=lambda feedback_msg: self.feedback_callback(
                feedback_msg,
                current_seq
            )
        )

        send_future.add_done_callback(
            lambda future: self.goal_response_callback(
                future,
                current_seq
            )
        )

        self.goal_sent = True

    def goal_response_callback(self, future, seq):
        """
        Nav2 是否接受目标点
        """
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warn(
                f'Goal seq={seq} rejected by Nav2'
            )
            return

        self.get_logger().info(
            f'Goal seq={seq} accepted by Nav2'
        )

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda future_result: self.result_callback(
                future_result,
                seq
            )
        )

    def feedback_callback(self, feedback_msg, seq):
        """
        Nav2 导航反馈。

        这里不频繁打印，否则终端会刷屏。
        需要调试时可以打开。
        """
        pass

    def result_callback(self, future, seq):
        """
        当前目标点导航结果
        """
        # 如果这是旧 goal 的结果，直接忽略
        if seq != self.goal_seq:
            self.get_logger().info(
                f'Ignore old goal result seq={seq}, current seq={self.goal_seq}'
            )
            return

        result = future.result()
        status = result.status

        self.get_logger().info(
            f'Goal seq={seq}, index={self.current_index} finished, status={status}'
        )

        # Nav2 action status 常见：
        # 4 = SUCCEEDED
        # 5 = CANCELED
        # 6 = ABORTED
        if status == 4:
            self.handle_goal_succeeded()
        elif status == 5:
            self.get_logger().warn('Goal canceled')
        elif status == 6:
            self.get_logger().warn(
                'Goal aborted. You can wait for /yolo_avoid_finished '
                'or manually resend goal.'
            )
        else:
            self.get_logger().warn(f'Goal finished with status={status}')

    def handle_goal_succeeded(self):
        """
        到达当前路点后，切换到下一个路点
        """
        self.get_logger().info(
            f'Reached waypoint {self.current_index}'
        )

        next_index = self.current_index + 1

        if next_index >= self.total_waypoints:
            if self.loop:
                next_index = 0
                self.get_logger().info('Loop enabled, restart from waypoint 0')
            else:
                self.get_logger().info('All waypoints finished')
                return

        self.current_index = next_index
        self.send_current_goal(reason='next_waypoint')

    def can_replan_now(self):
        """
        检查是否允许重规划
        """
        now = self.get_clock().now()

        if self.last_replan_time is None:
            return True

        dt = (now - self.last_replan_time).nanoseconds / 1e9

        return dt >= self.replan_cooldown_sec

    def avoid_finished_callback(self, msg):
        """
        收到 YOLO 避障完成事件后，重新发送当前目标点。

        这样 Nav2 会以当前位置作为新的起点，重新调用 planner 规划路径。
        """
        if not self.goal_sent:
            self.get_logger().warn(
                'Received /yolo_avoid_finished, but no goal has been sent yet'
            )
            return

        if not self.can_replan_now():
            self.get_logger().warn(
                'Received /yolo_avoid_finished, but replan cooldown active'
            )
            return

        self.last_replan_time = self.get_clock().now()

        self.get_logger().info(
            f'Received /yolo_avoid_finished, resend waypoint {self.current_index}'
        )

        self.send_current_goal(reason='yolo_avoid_finished_replan')


def main(args=None):
    rclpy.init(args=args)

    node = WaypointManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()