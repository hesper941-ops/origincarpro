#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool


class CmdVelMux(Node):
    def __init__(self):
        super().__init__('cmd_vel_mux')

        self.declare_parameter('track_cmd_topic', '/track_cmd_vel')
        self.declare_parameter('legacy_nav_cmd_topic', '/nav2_cmd_vel')
        self.declare_parameter('avoid_cmd_topic', '/avoid_cmd_vel')
        self.declare_parameter('legacy_avoid_cmd_topic', '/yolo_cmd_vel')
        self.declare_parameter('avoid_active_topic', '/avoid_active')
        self.declare_parameter('legacy_avoid_active_topic', '/yolo_avoid_active')
        self.declare_parameter('output_cmd_topic', '/cmd_vel')
        self.declare_parameter('cmd_timeout_sec', 0.5)
        self.declare_parameter('publish_rate_hz', 20.0)

        self.cmd_timeout_sec = float(self.get_parameter('cmd_timeout_sec').value)
        self.track_cmd = Twist()
        self.avoid_cmd = Twist()
        self.track_stamp = None
        self.avoid_stamp = None
        self.avoid_active = False

        self.create_subscription(Twist, self.get_parameter('track_cmd_topic').value, self.track_callback, 10)
        self.create_subscription(Twist, self.get_parameter('legacy_nav_cmd_topic').value, self.track_callback, 10)
        self.create_subscription(Twist, self.get_parameter('avoid_cmd_topic').value, self.avoid_callback, 10)
        self.create_subscription(Twist, self.get_parameter('legacy_avoid_cmd_topic').value, self.avoid_callback, 10)
        self.create_subscription(Bool, self.get_parameter('avoid_active_topic').value, self.active_callback, 10)
        self.create_subscription(Bool, self.get_parameter('legacy_avoid_active_topic').value, self.active_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, self.get_parameter('output_cmd_topic').value, 10)
        rate = float(self.get_parameter('publish_rate_hz').value)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self.publish_selected_cmd)

    def track_callback(self, msg):
        self.track_cmd = msg
        self.track_stamp = self.now_sec()

    def avoid_callback(self, msg):
        self.avoid_cmd = msg
        self.avoid_stamp = self.now_sec()

    def active_callback(self, msg):
        self.avoid_active = bool(msg.data)

    def publish_selected_cmd(self):
        now = self.now_sec()
        if self.avoid_active:
            if self.avoid_stamp is not None and now - self.avoid_stamp <= self.cmd_timeout_sec:
                self.cmd_pub.publish(self.avoid_cmd)
            else:
                self.cmd_pub.publish(Twist())
            return

        if self.track_stamp is not None and now - self.track_stamp <= self.cmd_timeout_sec:
            self.cmd_pub.publish(self.track_cmd)
        else:
            self.cmd_pub.publish(Twist())

    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
