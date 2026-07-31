"""Launch the standalone mock multimodal worker."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("person_board_detection")
    parameters = os.path.join(
        package_share, "config", "person_board_mock_llm.yaml"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "allowed_capture_directory",
                default_value=(
                    "/root/intelligent_car_ws/runtime/person_board/"
                    "latest_capture"
                ),
            ),
            Node(
                package="person_board_detection",
                executable="person_board_mock_llm_worker",
                name="person_board_mock_llm_worker",
                output="screen",
                parameters=[
                    parameters,
                    {
                        "allowed_capture_directory": LaunchConfiguration(
                            "allowed_capture_directory"
                        )
                    },
                ],
            ),
        ]
    )
