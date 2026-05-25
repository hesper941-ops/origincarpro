#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2022, www.guyuehome.com

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from launch import LaunchDescription
from launch_ros.actions import Node
from launch import LaunchDescription
import os
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from ament_index_python import get_package_share_directory
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    usb_node = IncludeLaunchDescription(PythonLaunchDescriptionSource(get_package_share_directory('hobot_usb_cam') + '/launch/hobot_usb_cam.launch.py'),
                                       launch_arguments={'usb_image_width': '1920', 'usb_image_height': '1080',
                                                         'usb_video_device': '/dev/video8'}.items())

    codec_node = IncludeLaunchDescription(PythonLaunchDescriptionSource(get_package_share_directory('hobot_codec') + '/launch/hobot_codec.launch.py'),
                                               launch_arguments={'codec_in_mode': 'ros', 'codec_out_mode': 'ros','codec_in_format':'jpeg','codec_out_format':'bgr8',
                                                                 'codec_sub_topic': '/image', 'codec_pub_topic': '/image_bgr8'}.items())
    raw_node = IncludeLaunchDescription(PythonLaunchDescriptionSource(get_package_share_directory('hobot_codec') + '/launch/hobot_codec.launch.py'),
                                               launch_arguments={'codec_sub_topic': '/image', 'codec_pub_topic': '/image_raw'}.items())
    qr_detection_node = Node(
        package='qr_code_detection',
        executable='qr_detection_node',
    )
    image_resize_node = Node(
        package='my_package',
        executable='my_node'
    )
    
    xml_node = IncludeLaunchDescription(  
        XMLLaunchDescriptionSource(get_package_share_directory('rosbridge_server') + '/launch/rosbridge_websocket_launch.xml'),
    ) 
    ai_model_node = Node(
        package='racing_vision_ai',
        executable='vision_ai_node'
    )
    return LaunchDescription([
        qr_detection_node,
        usb_node,
        #raw_node,
        codec_node,
        #image_resize_node,
        xml_node,
        
        ai_model_node,
    ])
