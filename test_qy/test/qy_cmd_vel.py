#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.qos import QoSProfile

class CmdVel2AckermannDriveNode(Node):
    def __init__(self):
        super().__init__('cmd_vel_to_ackermann_drive')
        # 发布 Ackermann 控制
        self.publisher = self.create_publisher(
            AckermannDriveStamped,
            '/ackermann_cmd',
            QoSProfile(depth=10)
        )
        # 订阅 cmd_vel
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_callback,
            QoSProfile(depth=10)
        )
        # =========================
        # 参数
        # =========================

        # 轴距（单位：米）
        self.wheelbase = 0.143

        self.frame_id = 'odom_combined'

        # False:
        # angular.z 表示角速度
        self.cmd_angle_instead_rotvel = False

        # 最大速度（你后面主要调这个）
        self.max_speed = 1.2

        # 最低速度
        self.min_speed = 0.3

        # 转弯减速系数（越大，转弯越慢）
        self.turn_speed_gain = 2.0

        # 转向平滑系数
        # 越大越平滑
        self.alpha = 0.7

        # 上一次转向角
        self.last_steering = 0.0

        self.get_logger().info("CmdVel -> Ackermann 节点已启动")

    # =====================================================
    # 角速度 -> 转向角
    # =====================================================

    def convert_trans_rot_vel_to_steering_angle(self, vel, omega):
        if omega == 0 or vel == 0:
            return 0.0
        radius = vel / omega
        steering = math.atan(self.wheelbase / radius)
        return steering

    # =====================================================
    # 回调函数
    # =====================================================

    def cmd_callback(self, data):
        # 原始线速度
        vel = data.linear.x
        # =================================================
        # 计算转向角
        # =================================================
        if self.cmd_angle_instead_rotvel:
            steering = data.angular.z
        else:
            steering = self.convert_trans_rot_vel_to_steering_angle(
                vel,
                data.angular.z
            )
        # =================================================
        # 转向平滑（防止抖动）
        # =================================================
        steering_filtered = (
            self.alpha * self.last_steering
            + (1.0 - self.alpha) * steering
        )
        self.last_steering = steering_filtered
        # =================================================
        # 动态转弯减速
        # 转角越大 -> 速度越低
        # =================================================
        speed = (
            self.max_speed
            - self.turn_speed_gain * abs(steering_filtered)
        )

        # 限制最低速度
        speed = max(self.min_speed, speed)

        # =================================================
        # 发布 Ackermann 消息
        # =================================================

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.drive.steering_angle = steering_filtered
        msg.drive.speed = speed
        self.publisher.publish(msg)

        # =================================================
        # 调试输出
        # =================================================
        self.get_logger().info(
            f"speed={speed:.2f}  steering={steering_filtered:.2f}"
        )

# =========================================================
# main
# =========================================================

def main():
    rclpy.init()
    node = CmdVel2AckermannDriveNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()