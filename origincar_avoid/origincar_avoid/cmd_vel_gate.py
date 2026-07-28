#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool


class CmdVelGate(Node):
    def __init__(self):
        super().__init__('cmd_vel_gate')

        self.declare_parameter('input_cmd_topic', '/cmd_vel_raw')
        self.declare_parameter('output_cmd_topic', '/cmd_vel')
        self.declare_parameter('started_topic', '/competition/started')
        self.declare_parameter('emergency_stop_topic', '/competition/emergency_stop')
        self.declare_parameter('cmd_timeout_sec', 0.5)
        self.declare_parameter('publish_zero_rate', 10.0)

        self.cmd_timeout_sec = float(self.get_parameter('cmd_timeout_sec').value)
        self.started = False
        self.emergency_stop = False
        self.latest_cmd = Twist()
        self.latest_cmd_time = None

        self.create_subscription(
            Twist,
            self.get_parameter('input_cmd_topic').value,
            self.cmd_callback,
            10,
        )
        self.create_subscription(
            Bool,
            self.get_parameter('started_topic').value,
            self.started_callback,
            10,
        )
        self.create_subscription(
            Bool,
            self.get_parameter('emergency_stop_topic').value,
            self.emergency_stop_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(Twist, self.get_parameter('output_cmd_topic').value, 10)
        rate = float(self.get_parameter('publish_zero_rate').value)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self.tick)

    def cmd_callback(self, msg):
        self.latest_cmd = msg
        self.latest_cmd_time = self.now_sec()

    def started_callback(self, msg):
        self.started = bool(msg.data)

    def emergency_stop_callback(self, msg):
        self.emergency_stop = bool(msg.data)

    def tick(self):
        if not self.started or self.emergency_stop:
            self.cmd_pub.publish(Twist())
            return

        now = self.now_sec()
        if self.latest_cmd_time is None or now - self.latest_cmd_time > self.cmd_timeout_sec:
            self.cmd_pub.publish(Twist())
            return

        self.cmd_pub.publish(self.latest_cmd)

    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelGate()
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
