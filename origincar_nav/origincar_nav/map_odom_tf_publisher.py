#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseWithCovarianceStamped

from tf2_ros import TransformBroadcaster


def yaw_to_quaternion(yaw):
    """
    2D yaw 转 quaternion
    """
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)
    return 0.0, 0.0, qz, qw


def quaternion_to_yaw(q):
    """
    quaternion 转 2D yaw
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    """
    角度归一化到 [-pi, pi]
    """
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class MapOdomTfPublisher(Node):
    def __init__(self):
        super().__init__('map_odom_tf_publisher')

        # =========================
        # 参数
        # =========================
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom_combined')
        self.declare_parameter('base_frame', 'base_footprint')

        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('qr_pose_topic', '/qr_pose_in_map')

        # 初始 map -> odom_combined
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw', 0.0)

        self.declare_parameter('publish_rate', 30.0)

        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.odom_topic = self.get_parameter('odom_topic').value
        self.qr_pose_topic = self.get_parameter('qr_pose_topic').value

        self.map_to_odom_x = float(self.get_parameter('initial_x').value)
        self.map_to_odom_y = float(self.get_parameter('initial_y').value)
        self.map_to_odom_yaw = float(self.get_parameter('initial_yaw').value)

        publish_rate = float(self.get_parameter('publish_rate').value)

        # 最新 odom_combined -> base_footprint
        self.latest_odom_base_x = None
        self.latest_odom_base_y = None
        self.latest_odom_base_yaw = None

        # =========================
        # TF broadcaster
        # =========================
        self.tf_broadcaster = TransformBroadcaster(self)

        # =========================
        # 订阅融合里程计
        # Odometry.pose 表示 base 在 odom_combined 下的位姿
        # =========================
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            20
        )

        # =========================
        # 订阅二维码定位结果
        # 这个话题表示：
        #   base_footprint 在 map 坐标系下的位姿
        #
        # 以后你的二维码节点算出小车在 map 中的位置后，
        # 就发布 PoseWithCovarianceStamped 到 /qr_pose_in_map
        # =========================
        self.qr_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.qr_pose_topic,
            self.qr_pose_callback,
            10
        )

        # =========================
        # 也订阅 /initialpose
        # 方便你在 RViz 里用 2D Pose Estimate 手动给初始位姿
        # =========================
        self.initial_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.initial_pose_callback,
            10
        )

        self.timer = self.create_timer(
            1.0 / publish_rate,
            self.publish_tf
        )

        self.get_logger().info('map_odom_tf_publisher started')
        self.get_logger().info(f'Publish TF: {self.map_frame} -> {self.odom_frame}')
        self.get_logger().info(f'Odom topic: {self.odom_topic}')
        self.get_logger().info(f'QR pose topic: {self.qr_pose_topic}')
        self.get_logger().info(
            f'Initial map->odom: x={self.map_to_odom_x:.3f}, '
            f'y={self.map_to_odom_y:.3f}, yaw={self.map_to_odom_yaw:.3f}'
        )

    def odom_callback(self, msg):
        """
        保存 odom_combined -> base_footprint
        """
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.latest_odom_base_x = float(p.x)
        self.latest_odom_base_y = float(p.y)
        self.latest_odom_base_yaw = quaternion_to_yaw(q)

    def initial_pose_callback(self, msg):
        """
        RViz 2D Pose Estimate 输入：
        表示 base_footprint 在 map 下的位姿
        """
        self.update_map_to_odom_from_map_base_pose(msg, source='initialpose')

    def qr_pose_callback(self, msg):
        """
        二维码定位输入：
        表示 base_footprint 在 map 下的位姿
        """
        self.update_map_to_odom_from_map_base_pose(msg, source='qr')

    def update_map_to_odom_from_map_base_pose(self, msg, source='unknown'):
        """
        已知：
            T_map_base
            T_odom_base

        求：
            T_map_odom

        关系：
            T_map_base = T_map_odom * T_odom_base

        所以：
            T_map_odom = T_map_base * inverse(T_odom_base)
        """

        if self.latest_odom_base_x is None:
            self.get_logger().warn(
                f'Received {source} pose, but no odom received yet. Ignore.'
            )
            return

        if msg.header.frame_id and msg.header.frame_id != self.map_frame:
            self.get_logger().warn(
                f'{source} pose frame_id is {msg.header.frame_id}, '
                f'expected {self.map_frame}'
            )

        # base 在 map 下的位姿
        map_base_x = float(msg.pose.pose.position.x)
        map_base_y = float(msg.pose.pose.position.y)
        map_base_yaw = quaternion_to_yaw(msg.pose.pose.orientation)

        # base 在 odom 下的位姿
        odom_base_x = self.latest_odom_base_x
        odom_base_y = self.latest_odom_base_y
        odom_base_yaw = self.latest_odom_base_yaw

        # yaw_map_odom = yaw_map_base - yaw_odom_base
        map_odom_yaw = normalize_angle(map_base_yaw - odom_base_yaw)

        c = math.cos(map_odom_yaw)
        s = math.sin(map_odom_yaw)

        # t_map_odom = p_map_base - R_map_odom * p_odom_base
        map_odom_x = map_base_x - (c * odom_base_x - s * odom_base_y)
        map_odom_y = map_base_y - (s * odom_base_x + c * odom_base_y)

        self.map_to_odom_x = map_odom_x
        self.map_to_odom_y = map_odom_y
        self.map_to_odom_yaw = map_odom_yaw

        self.get_logger().info(
            f'Update map->odom from {source}: '
            f'x={self.map_to_odom_x:.3f}, '
            f'y={self.map_to_odom_y:.3f}, '
            f'yaw={self.map_to_odom_yaw:.3f}'
        )

    def publish_tf(self):
        """
        持续发布 map -> odom_combined
        """
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame

        t.transform.translation.x = float(self.map_to_odom_x)
        t.transform.translation.y = float(self.map_to_odom_y)
        t.transform.translation.z = 0.0

        qx, qy, qz, qw = yaw_to_quaternion(self.map_to_odom_yaw)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = MapOdomTfPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()