from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        # 声明参数
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Log level for the vision AI node'
        ),
        
        # Vision AI节点
        Node(
            package='racing_vision_ai',
            executable='vision_ai_node',
            name='vision_ai_node',
            output='screen',
            parameters=[
                {'use_sim_time': False}
            ],
            arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
        ),
    ])
