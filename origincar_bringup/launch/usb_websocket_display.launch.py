import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource, PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'dnn_example_msg_pub_topic_name',
            default_value=TextSubstitution(text='hobot_dnn_detection')
        ),
        DeclareLaunchArgument(
            'websocket_image_topic',
            default_value=TextSubstitution(text='/image_out/compressed')
        ),
    ]

    web_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            get_package_share_directory('websocket') + '/launch/websocket.launch.py'
        ),
        launch_arguments={
            'websocket_image_topic': LaunchConfiguration('websocket_image_topic'),
            'websocket_image_type': 'mjpeg',
            'websocket_smart_topic': LaunchConfiguration('dnn_example_msg_pub_topic_name'),
        }.items()
    )

    rosbridge_node = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('rosbridge_server'),
                'launch',
                'rosbridge_websocket_launch.xml'
            )
        ),
        launch_arguments={'port': '9091'}.items()
    )

    return LaunchDescription(launch_args + [
        web_node,
        rosbridge_node,
    ])
