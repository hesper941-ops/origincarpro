import os
from launch.substitutions import TextSubstitution
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument

from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python import get_package_share_directory
from ament_index_python.packages import get_package_prefix
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription

def generate_launch_description():
    
    dnn_node_example_path = os.path.join(
        get_package_prefix('dnn_node_example'),
        "lib/dnn_node_example")
    print("dnn_node_example_path is ", dnn_node_example_path)
    cp_cmd = "cp -r " + dnn_node_example_path + "/config ."
    print("cp_cmd is ", cp_cmd)
    os.system(cp_cmd)
    # args that can be set from the command line or a default will be used
    config_file_launch_arg = DeclareLaunchArgument(
        "dnn_example_config_file", default_value=TextSubstitution(text="/userdata/dev_ws/yolov5workconfig.json")
    )
    dump_render_launch_arg = DeclareLaunchArgument(
        "dnn_example_dump_render_img", default_value=TextSubstitution(text="0")
    )
    image_width_launch_arg = DeclareLaunchArgument(
        "dnn_example_image_width", default_value=TextSubstitution(text="640")
    )
    image_height_launch_arg = DeclareLaunchArgument(
        "dnn_example_image_height", default_value=TextSubstitution(text="480")
    )
    msg_pub_topic_name_launch_arg = DeclareLaunchArgument(
        "dnn_example_msg_pub_topic_name", default_value=TextSubstitution(text="hobot_dnn_detection")
    )


    nv12_codec_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hobot_codec'),
                'launch/hobot_codec_decode.launch.py')),
        launch_arguments={
            'codec_in_mode': 'ros',
            'codec_out_mode': 'shared_mem',
            'codec_sub_topic': '/image',
            'codec_pub_topic': '/hbmem_img',
            'codec_out_format': 'nv12'
        }.items()
    )
    
    dnn_node_example_node = Node(
        package='dnn_node_example',
        executable='example',
        output='screen',
        parameters=[
            {"config_file": LaunchConfiguration('dnn_example_config_file')},
            {"dump_render_img": LaunchConfiguration(
                'dnn_example_dump_render_img')},
            {"feed_type": 1},
            {"is_shared_mem_sub": 1},
            {"msg_pub_topic_name": LaunchConfiguration(
                "dnn_example_msg_pub_topic_name")}
        ],
        arguments=['--ros-args', '--log-level', 'warn']
    )
    
    # image->image_raw
    hobot_codec_to_raw = IncludeLaunchDescription(  
        PythonLaunchDescriptionSource(get_package_share_directory('hobot_codec') + '/launch/hobot_codec.launch.py'),  
        launch_arguments={'codec_sub_topic': '/image', 'codec_pub_topic': '/image_raw'}.items()  
    )  
    
    return LaunchDescription([
            config_file_launch_arg,
            dump_render_launch_arg,
            image_width_launch_arg,
            image_height_launch_arg,
            msg_pub_topic_name_launch_arg,
            
            dnn_node_example_node,
            nv12_codec_node,
            
            #hobot_codec_to_raw,
        ])
