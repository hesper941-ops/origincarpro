from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    device = LaunchConfiguration('device')
    usb_image_width = LaunchConfiguration('usb_image_width')
    usb_image_height = LaunchConfiguration('usb_image_height')
    usb_pixel_format = LaunchConfiguration('usb_pixel_format')

    return LaunchDescription([
        DeclareLaunchArgument(
            'device',
            default_value=TextSubstitution(text='/dev/video0'),
            description='USB camera device path'
        ),
        DeclareLaunchArgument(
            'usb_image_width',
            default_value=TextSubstitution(text='640'),
            description='USB camera image width'
        ),
        DeclareLaunchArgument(
            'usb_image_height',
            default_value=TextSubstitution(text='480'),
            description='USB camera image height'
        ),
        DeclareLaunchArgument(
            'usb_pixel_format',
            default_value=TextSubstitution(text='mjpeg'),
            description='USB camera pixel format'
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                get_package_share_directory('hobot_usb_cam') + '/launch/hobot_usb_cam.launch.py'
            ),
            launch_arguments={
                'usb_video_device': device,
                'usb_image_width': usb_image_width,
                'usb_image_height': usb_image_height,
                'usb_pixel_format': usb_pixel_format,
            }.items()
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                get_package_share_directory('hobot_codec') + '/launch/hobot_codec_decode.launch.py'
            ),
            launch_arguments={
                'codec_in_mode': 'ros',
                'codec_out_mode': 'shared_mem',
                'codec_sub_topic': '/image',
                'codec_pub_topic': '/hbmem_img',
                'codec_out_format': 'nv12',
            }.items()
        ),
        Node(
            package='vision_camera',
            executable='hbm_image_bridge',
            output='screen',
            arguments=['--ros-args', '--log-level', 'info']
        ),
    ])
