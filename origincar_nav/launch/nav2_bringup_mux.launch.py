#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node, SetRemap

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    origincar_nav_dir = get_package_share_directory('origincar_nav')

    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    map_yaml_file = LaunchConfiguration('map')

    return LaunchDescription([
        # =========================
        # 基础参数
        # =========================
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock'
        ),

        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(
                origincar_nav_dir,
                'config',
                'nav2_params.yaml'
            ),
            description='Full path to Nav2 params file'
        ),

        # 这里非常关键：必须声明 map
        DeclareLaunchArgument(
            'map',
            default_value=os.path.join(
                origincar_nav_dir,
                'maps',
                'map.yaml'
            ),
            description='Full path to map yaml file'
        ),

        # =========================
        # 1. map_server
        # 只发布静态地图，不启动 AMCL
        # =========================
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'yaml_filename': map_yaml_file},
            ]
        ),

        # =========================
        # 2. map_server 生命周期管理
        # =========================
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_map_server',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'autostart': True},
                {'node_names': ['map_server']},
            ]
        ),

        # =========================
        # 3. 发布 map -> odom_combined
        # =========================
        Node(
            package='origincar_nav',
            executable='map_odom_tf_publisher',
            name='map_odom_tf_publisher',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time},

                {'map_frame': 'map'},
                {'odom_frame': 'odom_combined'},
                {'base_frame': 'base_footprint'},

                # 如果你的融合里程计话题不是 /odom，这里要改
                {'odom_topic': '/odom'},

                # 二维码定位节点以后发布到这个话题
                {'qr_pose_topic': '/qr_pose_in_map'},

                # 初始 map -> odom_combined
                {'initial_x': 0.0},
                {'initial_y': 0.0},
                {'initial_yaw': 0.0},

                {'publish_rate': 30.0},
            ]
        ),

        # =========================
        # 4. Nav2 navigation
        # 不启动 AMCL
        # 不启动 SLAM
        # =========================
        GroupAction([
            # Nav2 平滑后的速度输出到 /nav2_cmd_vel
            # 再交给你的 cmd_vel_mux
            SetRemap(
                src='cmd_vel_smoothed',
                dst='/nav2_cmd_vel'
            ),

            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        nav2_bringup_dir,
                        'launch',
                        'navigation_launch.py'
                    )
                ),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'params_file': params_file,
                    'autostart': 'true',
                }.items()
            ),
        ]),

        # =========================
        # 5. 延迟启动 waypoint_manager
        # 避免 Nav2 / TF / map 还没起来就发目标点
        # =========================
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='origincar_nav',
                    executable='waypoint_manager',
                    name='waypoint_manager',
                    output='screen',
                    parameters=[
                        {'use_sim_time': use_sim_time},
                        {'global_frame': 'map'},

                        # =========================
                        # 改成你自己的路点
                        # x/y 单位：米
                        # yaw 单位：弧度
                        # =========================
                        {'waypoint_x': [3, 1.5, 1.5, 0.5]},
                        {'waypoint_y': [1.0, 0.0, 1.0, 1.0]},
                        {'waypoint_yaw': [0.0, 1.57, 3.14, -1.57]},

                        {'loop': True},
                        {'replan_cooldown_sec': 1.0},
                        {'autostart': True},
                    ]
                )
            ]
        ),
    ])