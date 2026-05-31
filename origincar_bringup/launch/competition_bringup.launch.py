#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_file(package_name, *path_parts):
    return os.path.join(get_package_share_directory(package_name), 'launch', *path_parts)


def generate_launch_description():
    enable_base = LaunchConfiguration('enable_base')
    enable_camera = LaunchConfiguration('enable_camera')
    enable_yolo_avoid = LaunchConfiguration('enable_yolo_avoid')
    enable_qr = LaunchConfiguration('enable_qr')
    enable_birdview = LaunchConfiguration('enable_birdview')
    enable_nav = LaunchConfiguration('enable_nav')
    enable_visualization = LaunchConfiguration('enable_visualization')
    enable_cobridge = LaunchConfiguration('enable_cobridge')
    odom_topic = LaunchConfiguration('odom_topic')
    visualization_frame = LaunchConfiguration('visualization_frame')
    button_backend = LaunchConfiguration('button_backend')
    debug_mode = LaunchConfiguration('debug_mode')
    debug_start_state = LaunchConfiguration('debug_start_state')
    debug_route_direction = LaunchConfiguration('debug_route_direction')
    debug_channel_index = LaunchConfiguration('debug_channel_index')
    debug_goal_name = LaunchConfiguration('debug_goal_name')
    debug_auto_start = LaunchConfiguration('debug_auto_start')

    safety_note = (
        'Safety: if enable_base=true with navigation and cmd_vel_mux enabled, '
        'the system may publish motion commands to /cmd_vel. Lift the wheels '
        'or confirm the field is safe before testing on the real car.'
    )

    return LaunchDescription([
        DeclareLaunchArgument('enable_base', default_value='true'),
        DeclareLaunchArgument('enable_camera', default_value='true'),
        DeclareLaunchArgument('enable_yolo_avoid', default_value='true'),
        DeclareLaunchArgument('enable_qr', default_value='true'),
        DeclareLaunchArgument('enable_birdview', default_value='true'),
        DeclareLaunchArgument('enable_nav', default_value='true'),
        DeclareLaunchArgument('enable_visualization', default_value='true'),
        DeclareLaunchArgument(
            'enable_cobridge',
            default_value='false',
            description='Optional web/rosbridge display bridge; disabled by default to avoid port conflicts.',
        ),
        DeclareLaunchArgument('odom_topic', default_value='/odom_combined'),
        DeclareLaunchArgument('visualization_frame', default_value='odom_combined'),
        DeclareLaunchArgument('button_backend', default_value='topic'),
        DeclareLaunchArgument('debug_mode', default_value='false'),
        DeclareLaunchArgument('debug_start_state', default_value='TRACK_TO_TASK_STATION'),
        DeclareLaunchArgument('debug_route_direction', default_value='clockwise'),
        DeclareLaunchArgument('debug_channel_index', default_value='0'),
        DeclareLaunchArgument('debug_goal_name', default_value=''),
        DeclareLaunchArgument('debug_auto_start', default_value='false'),

        LogInfo(msg=safety_note),

        Node(
            package='origincar_bringup',
            executable='competition_button_node.py',
            output='screen',
            parameters=[{
                'button_backend': button_backend,
                'debug_auto_start': debug_auto_start,
            }],
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_file('origincar_base', 'origincar_bringup.launch.py')),
            condition=IfCondition(enable_base),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_file('vision_camera', 'camera_bridge.launch.py')),
            condition=IfCondition(enable_camera),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_file('origincar_avoid', 'yolov8_detect.launch.py')),
            condition=IfCondition(enable_yolo_avoid),
            launch_arguments={
                'odom_topic': odom_topic,
            }.items(),
        ),
        Node(
            package='qr_code_detection',
            executable='qr_detection_node',
            output='screen',
            condition=IfCondition(enable_qr),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_file('vision_birdview', 'bird_view.launch.py')),
            condition=IfCondition(enable_birdview),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_file('origincar_nav', 'competition_nav.launch.py')),
            condition=IfCondition(enable_nav),
            launch_arguments={
                'odom_topic': odom_topic,
                'enable_visualization': enable_visualization,
                'visualization_frame': visualization_frame,
                'debug_mode': debug_mode,
                'debug_start_state': debug_start_state,
                'debug_route_direction': debug_route_direction,
                'debug_channel_index': debug_channel_index,
                'debug_goal_name': debug_goal_name,
                'debug_auto_start': debug_auto_start,
            }.items(),
        ),
        Node(
            package='origincar_avoid',
            executable='cmd_vel_gate',
            output='screen',
            parameters=[{
                'input_cmd_topic': '/cmd_vel_raw',
                'output_cmd_topic': '/cmd_vel',
                'started_topic': '/competition/started',
                'emergency_stop_topic': '/competition/emergency_stop',
            }],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_file('origincar_bringup', 'usb_websocket_display.launch.py')),
            condition=IfCondition(enable_cobridge),
            launch_arguments={
                'websocket_image_topic': '/image_out/compressed',
            }.items(),
        ),
    ])
