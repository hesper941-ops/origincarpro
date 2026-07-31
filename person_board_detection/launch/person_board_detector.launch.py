#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='person_board_detection',
            executable='person_board_detector',
            name='person_board_detector',
            output='screen',
            parameters=[{
                'model_path': (
                    '/root/intelligent_car_ws/models/person_board/'
                    'person_board_yolov8n_v0_bayese_640x640_nv12.bin'
                ),
                'image_topic': '/aurora/rgb/image_raw',
                'confidence_threshold': 0.35,
                'nms_threshold': 0.50,
                'max_detections': 10,
                'publish_result_image': True,
                'save_debug_image': True,
                'debug_image_path': (
                    '/root/person_board_debug/latest.jpg'
                ),
                'log_every_n': 30,
            }],
        ),
    ])
