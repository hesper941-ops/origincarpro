#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


class CmdVelMux(Node):
    def __init__(self):
        super().__init__('cmd_vel_mux')

        # =========================
        # 参数
        # =========================
        self.declare_parameter('nav2_cmd_vel_topic', '/nav2_cmd_vel')
        self.declare_parameter('yolo_cmd_vel_topic', '/yolo_cmd_vel')
        self.declare_parameter('yolo_avoid_active_topic', '/yolo_avoid_active')
        self.declare_parameter('output_cmd_vel_topic', '/cmd_vel')

        # YOLO 避障指令超时时间
        # 如果超过这个时间没有收到 YOLO 速度，就自动停车，防止小车一直乱跑
        self.declare_parameter('yolo_cmd_timeout', 0.5)

        self.nav2_cmd_vel_topic = self.get_parameter('nav2_cmd_vel_topic').value
        self.yolo_cmd_vel_topic = self.get_parameter('yolo_cmd_vel_topic').value
        self.yolo_avoid_active_topic = self.get_parameter('yolo_avoid_active_topic').value
        self.output_cmd_vel_topic = self.get_parameter('output_cmd_vel_topic').value
        self.yolo_cmd_timeout = float(self.get_parameter('yolo_cmd_timeout').value)

        # =========================
        # 状态变量
        # =========================
        self.yolo_avoid_active = False

        self.latest_nav2_cmd = Twist()
        self.latest_yolo_cmd = Twist()

        self.last_yolo_cmd_time = None

        # =========================
        # 订阅 NAV2 速度
        # =========================
        self.nav2_sub = self.create_subscription(
            Twist,
            self.nav2_cmd_vel_topic,
            self.nav2_cmd_callback,
            10
        )

        # =========================
        # 订阅 YOLO 避障速度
        # =========================
        self.yolo_sub = self.create_subscription(
            Twist,
            self.yolo_cmd_vel_topic,
            self.yolo_cmd_callback,
            10
        )

        # =========================
        # 订阅 YOLO 避障是否接管控制
        # =========================
        self.yolo_active_sub = self.create_subscription(
            Bool,
            self.yolo_avoid_active_topic,
            self.yolo_active_callback,
            10
        )

        # =========================
        # 最终发布给底盘的速度
        # =========================
        self.cmd_pub = self.create_publisher(
            Twist,
            self.output_cmd_vel_topic,
            10
        )

        # =========================
        # 定时发布
        # 20Hz，也就是每 0.05 秒发布一次
        # =========================
        self.timer = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info('cmd_vel_mux started')
        self.get_logger().info(f'NAV2 cmd topic: {self.nav2_cmd_vel_topic}')
        self.get_logger().info(f'YOLO cmd topic: {self.yolo_cmd_vel_topic}')
        self.get_logger().info(f'YOLO active topic: {self.yolo_avoid_active_topic}')
        self.get_logger().info(f'Output cmd topic: {self.output_cmd_vel_topic}')

    def nav2_cmd_callback(self, msg):
        """
        保存 NAV2 发来的速度
        """
        self.latest_nav2_cmd = msg

    def yolo_cmd_callback(self, msg):
        """
        保存 YOLO 避障发来的速度
        """
        self.latest_yolo_cmd = msg
        self.last_yolo_cmd_time = self.get_clock().now()

    def yolo_active_callback(self, msg):
        """
        YOLO 是否接管控制权
        True  ：使用 YOLO 的速度
        False ：使用 NAV2 的速度
        """
        self.yolo_avoid_active = bool(msg.data)

    def stop_cmd(self):
        """
        生成停车速度
        """
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = 0.0
        return cmd

    def timer_callback(self):
        """
        根据当前状态选择发布哪个速度
        """
        if self.yolo_avoid_active:
            # YOLO 正在避障，优先使用 YOLO 速度
            if self.last_yolo_cmd_time is None:
                # YOLO 说要接管，但是还没发速度，先停车
                final_cmd = self.stop_cmd()
            else:
                now = self.get_clock().now()
                dt = (now - self.last_yolo_cmd_time).nanoseconds / 1e9

                if dt > self.yolo_cmd_timeout:
                    # YOLO 速度太久没更新，停车保护
                    final_cmd = self.stop_cmd()
                    self.get_logger().warn('YOLO cmd timeout, stop robot')
                else:
                    final_cmd = self.latest_yolo_cmd

        else:
            # YOLO 没有避障，使用 NAV2 速度
            final_cmd = self.latest_nav2_cmd

        self.cmd_pub.publish(final_cmd)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelMux()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 退出前停车
        stop = Twist()
        node.cmd_pub.publish(stop)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()