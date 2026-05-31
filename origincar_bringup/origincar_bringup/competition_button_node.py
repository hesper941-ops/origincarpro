#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


class CompetitionButtonNode(Node):
    def __init__(self):
        super().__init__('competition_button_node')

        self.declare_parameter('button_backend', 'topic')
        self.declare_parameter('start_button_topic', '/competition/start_button')
        self.declare_parameter('reset_button_topic', '/competition/reset_button')
        self.declare_parameter('started_topic', '/competition/started')
        self.declare_parameter('emergency_stop_topic', '/competition/emergency_stop')
        self.declare_parameter('userkey_topic', '/UserKey_Value')
        self.declare_parameter('debug_auto_start', False)
        self.declare_parameter('publish_rate_hz', 5.0)

        self.button_backend = str(self.get_parameter('button_backend').value)
        self.started = bool(self.get_parameter('debug_auto_start').value)
        self.emergency_stop = False

        self.started_pub = self.create_publisher(
            Bool,
            self.get_parameter('started_topic').value,
            10,
        )
        self.emergency_pub = self.create_publisher(
            Bool,
            self.get_parameter('emergency_stop_topic').value,
            10,
        )
        self.create_subscription(
            Bool,
            self.get_parameter('emergency_stop_topic').value,
            self.emergency_stop_callback,
            10,
        )

        if self.button_backend == 'topic':
            self.create_subscription(
                Bool,
                self.get_parameter('start_button_topic').value,
                self.start_button_callback,
                10,
            )
            self.create_subscription(
                Bool,
                self.get_parameter('reset_button_topic').value,
                self.reset_button_callback,
                10,
            )
        elif self.button_backend == 'userkey':
            self.get_logger().warn(
                'button_backend=userkey is experimental and disabled by default; '
                '/UserKey_Value is not used as the formal competition start input.'
            )
        else:
            self.get_logger().warn(
                f'Unknown button_backend={self.button_backend}; topic backend is the formal default.'
            )

        rate = float(self.get_parameter('publish_rate_hz').value)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self.publish_state)

    def start_button_callback(self, msg):
        if msg.data:
            self.started = True

    def reset_button_callback(self, msg):
        if msg.data:
            self.started = False

    def emergency_stop_callback(self, msg):
        self.emergency_stop = bool(msg.data)

    def publish_state(self):
        self.started_pub.publish(Bool(data=self.started))
        self.emergency_pub.publish(Bool(data=self.emergency_stop))


def main(args=None):
    rclpy.init(args=args)
    node = CompetitionButtonNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.started_pub.publish(Bool(data=False))
        node.emergency_pub.publish(Bool(data=True))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
