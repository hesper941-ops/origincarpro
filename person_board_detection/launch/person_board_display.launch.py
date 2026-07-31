"""Launch the standalone persistent HDMI display."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("person_board_detection")
    parameters = os.path.join(
        package_share, "config", "person_board_display.yaml"
    )
    return LaunchDescription(
        [
            Node(
                package="person_board_detection",
                executable="person_board_display",
                name="person_board_display",
                output="screen",
                parameters=[parameters],
            )
        ]
    )
