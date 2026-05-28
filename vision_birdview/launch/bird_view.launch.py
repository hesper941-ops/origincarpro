from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('vision_birdview'),
        'config',
        'aurora_ipm.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'input_topic',
            default_value=TextSubstitution(text='/image_out'),
            description='Input image topic for bird view'
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

        Node(
            package='vision_birdview',
            executable='perspective_node',
            output='screen',
            parameters=[
                config,
                {
                    'input_topic': LaunchConfiguration('input_topic'),
                    'publish_compressed': LaunchConfiguration('publish_compressed'),
                    'show_window': LaunchConfiguration('show_window'),
                }
            ],
        )
    ])
