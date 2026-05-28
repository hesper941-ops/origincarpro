#!/usr/bin/env python3

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    package_share = get_package_share_directory('origincar_avoid')

    config_file = LaunchConfiguration('config_file')
    image_topic = LaunchConfiguration('image_topic')
    detection_topic = LaunchConfiguration('detection_topic')
    output_compressed_topic = LaunchConfiguration('output_compressed_topic')

    yolo_cmd_vel_topic = LaunchConfiguration('yolo_cmd_vel_topic')
    yolo_avoid_active_topic = LaunchConfiguration('yolo_avoid_active_topic')

    nav2_cmd_vel_topic = LaunchConfiguration('nav2_cmd_vel_topic')
    output_cmd_vel_topic = LaunchConfiguration('output_cmd_vel_topic')

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

        DeclareLaunchArgument(
            'yolo_cmd_vel_topic',
            default_value=TextSubstitution(text='/yolo_cmd_vel'),
            description='YOLO obstacle avoidance cmd_vel topic'
        ),

        DeclareLaunchArgument(
            'yolo_avoid_active_topic',
            default_value=TextSubstitution(text='/yolo_avoid_active'),
            description='YOLO obstacle avoidance active topic'
        ),

        DeclareLaunchArgument(
            'nav2_cmd_vel_topic',
            default_value=TextSubstitution(text='/nav2_cmd_vel'),
            description='NAV2 cmd_vel topic before mux'
        ),

        DeclareLaunchArgument(
            'output_cmd_vel_topic',
            default_value=TextSubstitution(text='/cmd_vel'),
            description='Final cmd_vel topic to robot base'
        ),

        # =========================
        # DNN YOLO 推理节点
        # =========================
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

        # =========================
        # YOLO 画框 + 避障判断节点
        # 输出 /yolo_cmd_vel 和 /yolo_avoid_active
        # =========================
        Node(
            package='origincar_avoid',
            executable='yolo_box_viewer',
            output='screen',
            parameters=[
                {'image_topic': image_topic},
                {'detection_topic': detection_topic},
                {'output_compressed_topic': output_compressed_topic},

                {'yolo_cmd_vel_topic': yolo_cmd_vel_topic},
                {'yolo_avoid_active_topic': yolo_avoid_active_topic},

                # 避障触发线，越小越早避障
                {'avoid_start_y_ratio': 0.65},

                # 中心死区，越大越不敏感
                {'center_deadzone_ratio': 0.08},

                # YOLO 避障速度
                {'linear_speed': 0.15},
                {'turn_speed': 0.45},

                # 和 NAV2 结合时，建议 False
                # 没障碍物时不让 YOLO 自己前进，而是交给 NAV2
                {'forward_when_no_roadblock': False},
            ]
        ),

        # =========================
        # 速度仲裁节点
        # 输入：
        #   /nav2_cmd_vel
        #   /yolo_cmd_vel
        #   /yolo_avoid_active
        # 输出：
        #   /cmd_vel
        # =========================
        Node(
            package='origincar_avoid',
            executable='cmd_vel_mux',
            output='screen',
            parameters=[
                {'nav2_cmd_vel_topic': nav2_cmd_vel_topic},
                {'yolo_cmd_vel_topic': yolo_cmd_vel_topic},
                {'yolo_avoid_active_topic': yolo_avoid_active_topic},
                {'output_cmd_vel_topic': output_cmd_vel_topic},
                {'yolo_cmd_timeout': 0.5},
            ]
        ),
    ])