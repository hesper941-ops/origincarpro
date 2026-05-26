#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='qr_code_detection',
            executable='qr_detection_node',
            output='screen',
        ),
    ])
