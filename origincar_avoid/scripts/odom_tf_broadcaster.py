#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdomTfBroadcaster(Node):
    def __init__(self):
        super().__init__('odom_tf_broadcaster')

        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        self.odom_topic = self.get_parameter('odom_topic').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.tf_broadcaster = TransformBroadcaster(self)

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            20
        )

        self.get_logger().info(f'Subscribe odom topic: {self.odom_topic}')
        self.get_logger().info(f'Publish TF: {self.odom_frame} -> {self.base_frame}')

    def odom_callback(self, msg):
        t = TransformStamped()

        # 使用 odom 消息的时间戳
        t.header.stamp = msg.header.stamp

        # 如果 /odom 里面 frame_id 是空的，就使用参数
        if msg.header.frame_id:
            t.header.frame_id = msg.header.frame_id
        else:
            t.header.frame_id = self.odom_frame

        # 如果 /odom 里面 child_frame_id 是空的，就使用参数
        if msg.child_frame_id:
            t.child_frame_id = msg.child_frame_id
        else:
            t.child_frame_id = self.base_frame

        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        t.transform.rotation = msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdomTfBroadcaster()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()