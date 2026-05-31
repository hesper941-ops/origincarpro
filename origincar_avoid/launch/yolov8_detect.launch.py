#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    semantic_map_file = LaunchConfiguration('semantic_map_file')
    odom_topic = LaunchConfiguration('odom_topic')
    config_file = LaunchConfiguration('config_file')
    image_topic = LaunchConfiguration('image_topic')
    detection_topic = LaunchConfiguration('detection_topic')
    output_compressed_topic = LaunchConfiguration('output_compressed_topic')

    return LaunchDescription([
        DeclareLaunchArgument(
            'semantic_map_file',
            default_value=os.path.join(
                get_package_share_directory('origincar_nav'),
                'config',
                'semantic_map.yaml',
            ),
        ),
        DeclareLaunchArgument('odom_topic', default_value='/odom_combined'),
        DeclareLaunchArgument(
            'config_file',
            default_value=TextSubstitution(
                text=os.path.join(
                    get_package_share_directory('yolov8_test_mplus0'),
                    'config',
                    'yolov8_test_mplus0_workconfig.json',
                )
            ),
        ),
        DeclareLaunchArgument('image_topic', default_value='/image_out/compressed'),
        DeclareLaunchArgument('detection_topic', default_value='/hobot_dnn_detection'),
        DeclareLaunchArgument('output_compressed_topic', default_value='/yolov8_result/compressed'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('yolov8_test_mplus0'),
                    'launch',
                    'yolov8_detect.launch.py',
                )
            ),
            launch_arguments={
                'config_file': config_file,
                'image_topic': image_topic,
                'detection_topic': detection_topic,
                'output_compressed_topic': output_compressed_topic,
            }.items(),
        ),
        Node(
            package='origincar_avoid',
            executable='yolo_avoid_controller',
            output='screen',
            parameters=[{
                'semantic_map_file': semantic_map_file,
                'odom_topic': odom_topic,
                'detection_topic': detection_topic,
            }],
        ),
        Node(
            package='origincar_avoid',
            executable='cmd_vel_mux',
            output='screen',
            parameters=[{
                'track_cmd_topic': '/track_cmd_vel',
                'avoid_cmd_topic': '/avoid_cmd_vel',
                'avoid_active_topic': '/avoid_active',
                'output_cmd_topic': '/cmd_vel_raw',
            }],
        ),
    ])
