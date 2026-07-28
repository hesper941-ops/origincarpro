import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory('vision_birdview'),
        'config',
        'perspective.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'input_topic',
            default_value=TextSubstitution(text='/image_out'),
            description='Input image topic for bird view'
        ),
        DeclareLaunchArgument(
            'perspective_config',
            default_value=TextSubstitution(text=default_config),
            description='Perspective transform YAML file'
        ),
        DeclareLaunchArgument(
            'publish_compressed',
            default_value=TextSubstitution(text='true'),
            description='Whether to publish compressed bird view image'
        ),
        DeclareLaunchArgument(
            'show_window',
            default_value=TextSubstitution(text='false'),
            description='Whether to show OpenCV imshow window'
        ),
        DeclareLaunchArgument(
            'enable_dummy_output',
            default_value=TextSubstitution(text='False'),
            description='Publish dummy local return perception for interface testing'
        ),
        Node(
            package='vision_birdview',
            executable='perspective_node',
            output='screen',
            parameters=[{
                'input_topic': LaunchConfiguration('input_topic'),
                'perspective_config': LaunchConfiguration('perspective_config'),
                'publish_compressed': LaunchConfiguration('publish_compressed'),
                'show_window': LaunchConfiguration('show_window'),
                'enable_dummy_output': LaunchConfiguration('enable_dummy_output'),
            }],
        ),
    ])
