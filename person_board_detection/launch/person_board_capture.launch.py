#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_path = os.path.join(
        get_package_share_directory("person_board_detection"),
        "config",
        "person_board_capture.yaml",
    )
    return LaunchDescription(
        [
            Node(
                package="person_board_detection",
                executable="person_board_detector",
                name="person_board_detector",
                output="screen",
                parameters=[config_path],
                respawn=False,
            ),
        ]
    )
