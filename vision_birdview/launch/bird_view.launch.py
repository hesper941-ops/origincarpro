from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
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
            parameters=[{
                'publish_compressed': LaunchConfiguration('publish_compressed'),
            }],
        ),
    ])
