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
            'publish_compressed',
            default_value=TextSubstitution(text='True'),
            description='Whether to publish /bird_view/image/compressed'
        ),
        Node(
            package='vision_birdview',
            executable='perspective_node',
            output='screen',

            parameters=[
                config,
                {
                    'publish_compressed': LaunchConfiguration('publish_compressed'),
                }

            ],
        )
    ])
