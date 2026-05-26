#!/usr/bin/env python3

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    package_share = get_package_share_directory('yolov8_test_mplus0')

    config_file = LaunchConfiguration('config_file')
    image_topic = LaunchConfiguration('image_topic')
    detection_topic = LaunchConfiguration('detection_topic')
    output_compressed_topic = LaunchConfiguration('output_compressed_topic')

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=TextSubstitution(
                text=os.path.join(package_share, 'config', 'yolov8_test_mplus0_workconfig.json')
            ),
            description='YOLOv8 DNN workconfig file'
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value=TextSubstitution(text='/image_out/compressed'),
            description='Compressed image topic for visualization'
        ),
        DeclareLaunchArgument(
            'detection_topic',
            default_value=TextSubstitution(text='/hobot_dnn_detection'),
            description='YOLO detection result topic'
        ),
        DeclareLaunchArgument(
            'output_compressed_topic',
            default_value=TextSubstitution(text='/yolov8_result/compressed'),
            description='Output compressed result topic'
        ),
        Node(
            package='dnn_node_example',
            executable='example',
            output='screen',
            parameters=[
                {'config_file': config_file},
                {'dump_render_img': 0},
                {'feed_type': 1},
                {'is_shared_mem_sub': 1},
                {'msg_pub_topic_name': detection_topic},
            ],
            arguments=['--ros-args', '--log-level', 'warn']
        ),
        Node(
            package='yolov8_test_mplus0',
            executable='yolo_box_viewer',
            output='screen',
            parameters=[
                {'image_topic': image_topic},
                {'detection_topic': detection_topic},
                {'output_compressed_topic': output_compressed_topic},
            ]
        ),
    ])
