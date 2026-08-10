#!/usr/bin/env python3

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('roadblock_map_filter'),
        'config',
        'roadblock_map_filter.yaml',
    )
    return LaunchDescription([
        Node(
            package='roadblock_map_filter',
            executable='roadblock_map_filter',
            name='roadblock_map_filter',
            output='screen',
            parameters=[config],
        ),
    ])
