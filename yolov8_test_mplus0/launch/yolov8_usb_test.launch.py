#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config_file = LaunchConfiguration('config_file')
    device = LaunchConfiguration('device')
    image_topic = LaunchConfiguration('image_topic')
    detection_topic = LaunchConfiguration('detection_topic')
    show_window = LaunchConfiguration('show_window')
    usb_pixel_format = LaunchConfiguration('usb_pixel_format')

    declare_config_file = DeclareLaunchArgument(
        'config_file',
        default_value=TextSubstitution(
            text='/root/intelligent_car_ws/src/yolov8_test_mplus0/config/yolov8_test_mplus0_workconfig.json'
        ),
        description='YOLOv8 DNN workconfig file'
    )

    declare_device = DeclareLaunchArgument(
        'device',
        default_value=TextSubstitution(text='/dev/video0'),
        description='USB camera device'
    )

    declare_image_topic = DeclareLaunchArgument(
        'image_topic',
        default_value=TextSubstitution(text='/image_out'),
        description='BGR image topic for OpenCV visualization'
    )

    declare_detection_topic = DeclareLaunchArgument(
        'detection_topic',
        default_value=TextSubstitution(text='/hobot_dnn_detection'),
        description='YOLO detection result topic'
    )

    declare_show_window = DeclareLaunchArgument(
        'show_window',
        default_value=TextSubstitution(text='True'),
        description='Whether to show OpenCV window'
    )

    declare_usb_pixel_format = DeclareLaunchArgument(
        'usb_pixel_format',
        default_value=TextSubstitution(text='mjpeg2rgb'),
        description='USB camera pixel format: mjpeg, mjpeg2rgb, yuyv, yuyv2rgb'
    )

    dnn_test_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('dnn_node_example'),
                'launch',
                'dnn_node_example.launch.py'
            )
        ),
        launch_arguments={
            'device': device,
            'usb_video_device': device,
            'usb_image_width': '640',
            'usb_image_height': '480',
            'usb_pixel_format': usb_pixel_format,
            'dnn_example_config_file': config_file,
            'dnn_example_msg_pub_topic_name': 'hobot_dnn_detection',
            'dnn_example_image_width': '640',
            'dnn_example_image_height': '640',
            'log_level': 'warn',
        }.items()
    )

    image_transport_node = Node(
        package='utils',
        executable='image_transport_node',
        output='screen',
        arguments=['--ros-args', '--log-level', 'info']
    )

    yolo_box_viewer_node = Node(
        package='yolov8_test_mplus0',
        executable='yolo_box_viewer',
        output='screen',
        parameters=[
            {'image_topic': image_topic},
            {'detection_topic': detection_topic},
            {'show_window': show_window},
        ]
    )

    return LaunchDescription([
        declare_config_file,
        declare_device,
        declare_image_topic,
        declare_detection_topic,
        declare_show_window,
        declare_usb_pixel_format,

        dnn_test_launch,
        image_transport_node,
        yolo_box_viewer_node,
    ])
