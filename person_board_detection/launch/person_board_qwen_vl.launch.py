"""Launch only the real Qwen VL worker."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("person_board_detection"),
        "config", "person_board_qwen_vl.yaml")
    allowed = LaunchConfiguration("allowed_capture_directory")
    return LaunchDescription([
        DeclareLaunchArgument(
            "allowed_capture_directory",
            default_value="/root/intelligent_car_ws/runtime/person_board/latest_capture"),
        Node(
        package="person_board_detection",
        executable="person_board_qwen_vl_worker",
        name="person_board_qwen_vl_worker",
        output="screen",
        parameters=[config, {"allowed_capture_directory": allowed}],
    )])
