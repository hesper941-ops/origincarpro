#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import SetRemap

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    origincar_nav_dir = get_package_share_directory('origincar_nav')

    use_sim_time = LaunchConfiguration('use_sim_time')
    map_file = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock'
        ),

        DeclareLaunchArgument(
            'map',
            default_value=os.path.join(origincar_nav_dir, 'maps', 'map.yaml'),
            description='Full path to map yaml file'
        ),

        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(origincar_nav_dir, 'config', 'nav2_params.yaml'),
            description='Full path to NAV2 params file'
        ),

        # 关键点：
        # 把 NAV2 原本输出的 /cmd_vel 重映射到 /nav2_cmd_vel
        # 这样 NAV2 不会直接控制底盘
        GroupAction([
            
            SetRemap(src='cmd_vel_smoothed', dst='/nav2_cmd_vel'),

            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')
                ),
                launch_arguments={
                    'map': map_file,
                    'use_sim_time': use_sim_time,
                    
                    'params_file': params_file,
                    'autostart': 'true',
                }.items()
            ),
        ]),
    ])