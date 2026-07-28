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
        self.declare_parameter('emergency_stop_cmd_topic', '/competition/emergency_stop_cmd')
        self.declare_parameter('userkey_topic', '/UserKey_Value')
        self.declare_parameter('debug_auto_start', False)
        self.declare_parameter('publish_rate_hz', 5.0)

        self.button_backend = str(self.get_parameter('button_backend').value)
        self.started_topic = self.get_parameter('started_topic').value
        self.emergency_stop_topic = self.get_parameter('emergency_stop_topic').value
        self.emergency_stop_cmd_topic = self.get_parameter('emergency_stop_cmd_topic').value
        self.started = bool(self.get_parameter('debug_auto_start').value)
        self.emergency_stop = False

        self.started_pub = self.create_publisher(
            Bool,
            self.started_topic,
            10,
        )

        self.emergency_pub = self.create_publisher(
            Bool,
            self.emergency_stop_topic,
            10,
        )

        self.create_subscription(
            Bool,
            self.emergency_stop_cmd_topic,
            self.emergency_stop_cmd_callback,
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

        self.get_logger().info(
            f'competition_button_node started, backend={self.button_backend}, '
            f'auto_start={self.started}'
        )

    def safe_publish(self, publisher, msg, topic_name):
        try:
            publisher.publish(msg)
            return True
        except Exception as e:
            self.get_logger().warn(f'Failed to publish {topic_name}: {e}')
            return False

    def start_button_callback(self, msg):
        if msg.data:
            self.started = True
            self.emergency_stop = False
            self.get_logger().info('Competition started by topic button.')

    def reset_button_callback(self, msg):
        if msg.data:
            self.started = False
            self.emergency_stop = False
            self.get_logger().info('Competition reset by topic button.')

    def emergency_stop_cmd_callback(self, msg):
        if msg.data:
            self.started = False
            self.emergency_stop = True
            self.get_logger().warn('Emergency stop command received.')
        else:
            self.started = False
            self.emergency_stop = False
            self.get_logger().info(
                'Emergency stop cleared; start_button must be pressed again to start.'
            )

    def publish_state(self):
        self.safe_publish(
            self.started_pub,
            Bool(data=self.started),
            self.started_topic,
        )

        self.safe_publish(
            self.emergency_pub,
            Bool(data=self.emergency_stop),
            self.emergency_stop_topic,
        )


def main(args=None):
    rclpy.init(args=args)
    node = CompetitionButtonNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        try:
            node.get_logger().error(f'competition_button_node crashed: {e}')
        except Exception:
            pass
    finally:
        try:
            node.safe_publish(
                node.started_pub,
                Bool(data=False),
                node.started_topic,
            )
            node.safe_publish(
                node.emergency_pub,
                Bool(data=True),
                node.emergency_stop_topic,
            )
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
