#!/usr/bin/env python3

import math

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile


class CmdVel2AckermannDriveNode(Node):
    def __init__(self):
        super().__init__('cmd_vel_to_ackermann_drive')
        self.publisher = self.create_publisher(
            AckermannDriveStamped,
            '/ackermann_cmd',
            QoSProfile(depth=10)
        )
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_callback,
            QoSProfile(depth=10)
        )

        self.wheelbase = 0.143
        self.frame_id = 'odom_combined'
        self.cmd_angle_instead_rotvel = False
        self.max_speed = 1.2
        self.min_speed = 0.3
        self.turn_speed_gain = 2.0
        self.alpha = 0.7
        self.last_steering = 0.0

    def convert_trans_rot_vel_to_steering_angle(self, vel, omega):
        if omega == 0 or vel == 0:
            return 0.0
        radius = vel / omega
        return math.atan(self.wheelbase / radius)

    def cmd_callback(self, data):
        vel = data.linear.x
        if self.cmd_angle_instead_rotvel:
            steering = data.angular.z
        else:
            steering = self.convert_trans_rot_vel_to_steering_angle(vel, data.angular.z)

        steering_filtered = (
            self.alpha * self.last_steering
            + (1.0 - self.alpha) * steering
        )
        self.last_steering = steering_filtered

        speed = self.max_speed - self.turn_speed_gain * abs(steering_filtered)
        speed = max(self.min_speed, speed)

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.drive.steering_angle = steering_filtered
        msg.drive.speed = speed
        self.publisher.publish(msg)


def main():
    rclpy.init()
    node = CmdVel2AckermannDriveNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
