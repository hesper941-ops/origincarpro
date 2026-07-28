import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory('origincar_nav')
    default_map = os.path.join(package_share, 'config', 'semantic_map.yaml')

    semantic_map_file = LaunchConfiguration('semantic_map_file')
    odom_topic = LaunchConfiguration('odom_topic')
    enable_visualization = LaunchConfiguration('enable_visualization')
    enable_preview_image = LaunchConfiguration('enable_preview_image')
    visualization_frame = LaunchConfiguration('visualization_frame')
    preview_image_topic = LaunchConfiguration('preview_image_topic')
    preview_compressed_topic = LaunchConfiguration('preview_compressed_topic')
    preview_image_size = LaunchConfiguration('preview_image_size')
    preview_publish_rate = LaunchConfiguration('preview_publish_rate')
    publish_raw_image = LaunchConfiguration('publish_raw_image')
    debug_mode = LaunchConfiguration('debug_mode')
    debug_start_state = LaunchConfiguration('debug_start_state')
    debug_route_direction = LaunchConfiguration('debug_route_direction')
    debug_channel_index = LaunchConfiguration('debug_channel_index')
    debug_goal_name = LaunchConfiguration('debug_goal_name')
    debug_auto_start = LaunchConfiguration('debug_auto_start')
    enable_birdview_return = LaunchConfiguration('enable_birdview_return')
    yaw_offset_deg = LaunchConfiguration('yaw_offset_deg')
    single_goal_mode = LaunchConfiguration('single_goal_mode')
    single_goal_name = LaunchConfiguration('single_goal_name')

    return LaunchDescription([
        DeclareLaunchArgument('semantic_map_file', default_value=default_map),
        DeclareLaunchArgument('odom_topic', default_value='/odom_combined'),
        DeclareLaunchArgument('enable_visualization', default_value='true'),
        DeclareLaunchArgument('enable_preview_image', default_value='true'),
        DeclareLaunchArgument('visualization_frame', default_value='map'),
        DeclareLaunchArgument('preview_image_topic', default_value='/semantic_map/preview'),
        DeclareLaunchArgument('preview_compressed_topic', default_value='/semantic_map/preview/compressed'),
        DeclareLaunchArgument('preview_image_size', default_value='500'),
        DeclareLaunchArgument('preview_publish_rate', default_value='1.0'),
        DeclareLaunchArgument('publish_raw_image', default_value='false'),
        DeclareLaunchArgument('debug_mode', default_value='false'),
        DeclareLaunchArgument('debug_start_state', default_value='TRACK_TO_TASK_STATION'),
        DeclareLaunchArgument('debug_route_direction', default_value='clockwise'),
        DeclareLaunchArgument('debug_channel_index', default_value='0'),
        DeclareLaunchArgument('debug_goal_name', default_value=''),
        DeclareLaunchArgument('debug_auto_start', default_value='false'),
        DeclareLaunchArgument('enable_birdview_return', default_value='false'),
        DeclareLaunchArgument('yaw_offset_deg', default_value='0.0'),
        DeclareLaunchArgument('single_goal_mode', default_value='false'),
        DeclareLaunchArgument('single_goal_name', default_value='task_station'),
        Node(
            package='origincar_nav',
            executable='task_manager',
            output='screen',
            parameters=[{
                'semantic_map_file': semantic_map_file,
                'odom_topic': odom_topic,
                'debug_mode': debug_mode,
                'debug_start_state': debug_start_state,
                'debug_route_direction': debug_route_direction,
                'debug_channel_index': debug_channel_index,
                'debug_goal_name': debug_goal_name,
                'debug_auto_start': debug_auto_start,
                'enable_birdview_return': enable_birdview_return,
                'single_goal_mode': single_goal_mode,
                'single_goal_name': single_goal_name,
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
                'yaw_offset_deg': yaw_offset_deg,
            }],
        ),
        Node(
            package='origincar_nav',
            executable='semantic_map_visualizer',
            output='screen',
            condition=IfCondition(enable_visualization),
            parameters=[{
                'semantic_map_file': semantic_map_file,
                'odom_topic': odom_topic,
                'visualization_frame': visualization_frame,
            }],
        ),
        Node(
            package='origincar_nav',
            executable='semantic_map_preview_node',
            output='screen',
            condition=IfCondition(PythonExpression([
                "'",
                enable_visualization,
                "' == 'true' and '",
                enable_preview_image,
                "' == 'true'",
            ])),
            parameters=[{
                'semantic_map_file': semantic_map_file,
                'odom_topic': odom_topic,
                'goal_topic': '/current_goal',
                'preview_image_topic': preview_image_topic,
                'preview_compressed_topic': preview_compressed_topic,
                'preview_image_size': preview_image_size,
                'preview_publish_rate': preview_publish_rate,
                'publish_raw_image': publish_raw_image,
            }],
        ),
    ])
