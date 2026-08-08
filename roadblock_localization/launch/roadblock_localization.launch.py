import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("roadblock_localization")
    params_file = os.path.join(share, "config", "roadblock_localization.yaml")
    calibration_file = os.path.join(share, "config", "ipm_calibration.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument("debug_log", default_value="false"),
            Node(
                package="roadblock_localization",
                executable="roadblock_ground_localizer",
                name="roadblock_ground_localizer",
                output="screen",
                parameters=[
                    params_file,
                    {"calibration_file": calibration_file, "debug_log": LaunchConfiguration("debug_log")},
                ],
            ),
        ]
    )
