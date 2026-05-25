#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2022, www.guyuehome.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
import cv2
import cv_bridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Int8
from std_msgs.msg import Int32
from geometry_msgs.msg import Pose
import numpy as np
from origincar_msg.msg import Sign

from pyzbar.pyzbar import decode
import time

import os
from ament_index_python.packages import get_package_share_directory

class QrCodeDetection(Node):
    def __init__(self):
        super().__init__('qrcode_detect')
        self.bridge = cv_bridge.CvBridge()
        self.state = 0;
        self.i = 0;
        
        self.last_decoded_data = None
        self.decoded_data = []  # 存储所有解码结果
#        self.close_sub = self.create_subscription(
#            Int32, "/close_signal", self.close_callback, 1)
        # 接受来自utils/NV122BGR的imgae_out
        self.image_sub = self.create_subscription(
            Image, "/image_bgr8", self.image_callback, 1)

        self.pub_car_signal = self.create_publisher(
            Sign, "/sign_switch", 1)
        self.car_signal = Sign()
        self.qr_start_pub = self.create_publisher(Int32, '/close_signal', 10)
        #  接收二维码信息
        self.pub_qrcode_info = self.create_publisher(
            String, "/qrcode_detected/info_result", 1)#识别出的文字信息
        self.info_result = String()

        # model路径
        modelPath = os.path.join(get_package_share_directory('qr_code_detection'), 'model/')

        # 重点: 使用 微信的qrcode 识别功能;
        self.detect_obj = cv2.wechat_qrcode_WeChatQRCode(
            modelPath+'detect.prototxt', modelPath+'detect.caffemodel',
            modelPath+'sr.prototxt', modelPath+'sr.caffemodel')
    
    # 处理二维码数据
    def handle_qr_data(self, QR_data):
        if QR_data != self.last_decoded_data and QR_data != "": #and QR_data not in self.decoded_data:
            #self.decoded_data.append(QR_data)
            self.get_logger().info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 解码内容: {QR_data}")
            return QR_data
            
        return self.last_decoded_data

    def image_callback(self, msg):
      if self.state == 0:
        cv_image = self.bridge.imgmsg_to_cv2(msg)
        # 识别图像中的二维码
        #qrInfo, qrPoints = self.detect_obj.detectAndDecode(cv_image)
        QR_codes = decode(cv_image)
        if QR_codes:
            for QR in QR_codes:
                try:
                    QR_data = QR.data.decode('utf-8')
                except UnicodeDecodeError:
                    self.get_logger().warn("二维码内容解码失败")
                    continue
                except Exception as e:
                    self.get_logger().error(f"解码异常: {e}")
                    continue

                rect = QR.rect
                self.last_decoded_data = self.handle_qr_data(QR_data)
            if self.last_decoded_data != None:
                self.get_logger().info('qrInfo: "{0}"'.format(self.last_decoded_data))
                qrInfo_str = self.last_decoded_data
                self.info_result.data = qrInfo_str
                self.pub_qrcode_info.publish(self.info_result)
                
                if self.last_decoded_data == "ClockWise":
                    self.car_signal.sign_data = 3
                    self.pub_car_signal.publish(self.car_signal)
                elif self.last_decoded_data == "AntiClockWise":
                    self.car_signal.sign_data = 4
                    self.pub_car_signal.publish(self.car_signal)
                elif int(self.last_decoded_data) % 2 == 0:
                    self.car_signal.sign_data = 4
                    self.pub_car_signal.publish(self.car_signal)
                elif int(self.last_decoded_data) % 2 != 0:
                    self.car_signal.sign_data = 3
                    self.pub_car_signal.publish(self.car_signal)
                    
                close_msg = Int32()
                close_msg.data = 1
                self.qr_start_pub.publish(close_msg)
        # 返回的结果包含了2个参数，第一个为识别到的二维码信息，第2个为二维码的位置点，二者都是list类型。
#        if qrInfo != emptyList:
#            self.get_logger().info('qrInfo: "{0}"'.format(qrInfo))
#
#            qrInfo_str = qrInfo[0]
#            self.info_result.data = qrInfo_str
#            self.pub_qrcode_info.publish(self.info_result)
#            



def main(args=None):

    rclpy.init(args=args)

    qrCodeDetection = QrCodeDetection()
    while rclpy.ok():
        rclpy.spin(qrCodeDetection)

    qrCodeDetection.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
