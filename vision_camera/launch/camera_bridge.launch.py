#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    color_topic = LaunchConfiguration('color_topic')
    jpeg_quality = LaunchConfiguration('jpeg_quality')

    aurora_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('deptrum-ros-driver-aurora930'),
            '/launch/aurora930_launch.py'
        ])
    )

    aurora_color_bridge = Node(
        package='vision_camera',
        executable='aurora_color_bridge.py',
        name='aurora_color_bridge',
        output='screen',
        parameters=[{
            'color_topic': color_topic,
            'jpeg_quality': jpeg_quality,
        }]
    )

    # Keep original YOLO/BPU input topic unchanged:
    # /image (CompressedImage) -> hobot_codec -> /hbmem_img (HbmMsg1080P)
    hobot_codec_decode = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('hobot_codec'),
            '/launch/hobot_codec_decode.launch.py'
        ]),
        launch_arguments={
            'codec_sub_topic': '/image',
            'codec_pub_topic': '/hbmem_img',
            'codec_in_mode': 'ros',
            'codec_out_mode': 'shared_mem',
            'codec_in_format': 'jpeg',
            'codec_out_format': 'nv12',
        }.items()
    )

    return LaunchDescription([
        SetEnvironmentVariable('ROS_DISABLE_LOANED_MESSAGES', '1'),

        DeclareLaunchArgument(
            'color_topic',
            default_value='',
            description='Aurora color Image topic. Empty means auto select.'
        ),
        DeclareLaunchArgument(
            'jpeg_quality',
            default_value='85',
            description='JPEG quality for /image and /image_out/compressed'
        ),

        aurora_launch,

        TimerAction(
            period=2.0,
            actions=[aurora_color_bridge]
        ),

        TimerAction(
            period=3.0,
            actions=[hobot_codec_decode]
        ),
    ])
