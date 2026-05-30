import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory('origincar_nav')
    default_map = os.path.join(package_share, 'config', 'semantic_map.yaml')

    semantic_map_file = LaunchConfiguration('semantic_map_file')
    odom_topic = LaunchConfiguration('odom_topic')

    return LaunchDescription([
        DeclareLaunchArgument('semantic_map_file', default_value=default_map),
        DeclareLaunchArgument('odom_topic', default_value='/odom_combined'),
        Node(
            package='origincar_nav',
            executable='task_manager',
            output='screen',
            parameters=[{
                'semantic_map_file': semantic_map_file,
                'odom_topic': odom_topic,
            }],
        ),
        Node(
            package='origincar_nav',
            executable='target_tracker',
            output='screen',
            parameters=[{
                'odom_topic': odom_topic,
                'goal_topic': '/current_goal',
                'cmd_topic': '/track_cmd_vel',
                'enable_topic': '/track_enable',
                'semantic_map_file': semantic_map_file,
            }],
        ),
    ])
