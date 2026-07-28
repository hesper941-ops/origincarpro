#!/usr/bin/env python3
import base64
import os
import io
import cv2
import numpy as np
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Int32,String
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import yaml

# 导入火山引擎SDK
try:
    from volcenginesdkarkruntime import Ark
except ImportError:
    print("Warning: volcengine-python-sdk[ark] not installed. Please install with: pip install volcengine-python-sdk[ark]")
    Ark = None


class VisionAINode(Node):
    """
    Racing Vision AI Node
    订阅sign4return信号，当接收到值为9时，订阅一帧图像并发送给火山引擎大模型进行图生文字分析
    """
    
    def __init__(self):
        super().__init__('vision_ai_node')
        
        # 初始化CV Bridge用于图像转换
        self.bridge = CvBridge()
        
        # 声明参数，允许用户指定配置文件路径
        self.declare_parameter('config_path', '')
        
        # 加载配置文件
        self.config = self.load_config()
        
        # 初始化火山引擎客户端
        self.ark_client = None
        if Ark is not None:
            api_key = self.config.get('volcengine', {}).get('api_key')
            if api_key:
                self.ark_client = Ark(api_key=api_key)
                self.get_logger().info("Volcengine Ark client initialized successfully")
            else:
                self.get_logger().warn("ARK_API_KEY not set in config file")
                # 尝试从环境变量获取API密钥
                api_key = os.environ.get('ARK_API_KEY')
                if api_key:
                    self.ark_client = Ark(api_key=api_key)
                    self.get_logger().info("Volcengine Ark client initialized using environment variable")
                else:
                    self.get_logger().error("ARK_API_KEY not set in config file or environment variable")
        else:
            self.get_logger().warn("Volcengine SDK not available")
        
        # 从配置文件获取模型ID，如果不存在尝试从环境变量获取
        self.model_id = self.config.get('volcengine', {}).get('model_id')
        if not self.model_id:
            self.model_id = os.environ.get('ARK_MODEL_ID', 'your-model-id')
            if self.model_id != 'your-model-id':
                self.get_logger().info(f"Using model ID from environment variable: {self.model_id}")
        
        # 目标sign值从配置文件获取，如果不存在则使用默认值9
        self.target_sign = self.config.get('detection', {}).get('target_sign', 9)
        
        # 状态变量
        self.waiting_for_image = False
        self.latest_image = None
        
        # QoS配置
        qos_profile = QoSProfile(depth=10)
        
        # 订阅sign4return话题
        self.sign_subscriber = self.create_subscription(
            Int32,
            self.config.get('detection', {}).get('sign_topic', 'sign4return'),  # 状态切换话题
            self.sign_callback,
            qos_profile
        )
        #发布输出结果
        self.ai_publisher = self.create_publisher(
            String, 
            'ai_model_output', 
            10
        )
        # 图像话题名称
        self.image_topic = self.config.get('detection', {}).get('image_topic', '/image')
        
        # 订阅图像话题（仅在需要时订阅）
        self.image_sub = None
        
        self.get_logger().info("Vision AI Node initialized")
        self.get_logger().info(f"Waiting for sign4return={self.target_sign} signal...")
    
    def load_config(self):
        """加载配置文件"""
        # 使用ament_index_python查找包的安装路径（如果已安装）
#        try:
#            from ament_index_python.packages import get_package_share_directory
#            package_share_dir = get_package_share_directory('racing_vision_ai')
#            installed_config_path = os.path.join(package_share_dir, 'config', 'vision_ai_config.yaml')
#        except Exception:
#            installed_config_path = None
#            self.get_logger().debug("找不到已安装的包共享目录")
#        
        # 尝试多个可能的配置文件路径
        possible_paths = [
            # 1. 检查当前工作目录的配置文件
            os.path.join('/userdata/dev_ws/src/origincar/racing_vision_ai/config/', 'vision_ai_config.yaml'),
            
            # 2. 开发环境相对路径
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'vision_ai_config.yaml'),
        ]
        
        # 如果找到了已安装的配置路径，添加到搜索列表
#        if installed_config_path:
#            possible_paths.append(installed_config_path)
#        
        # 获取用户通过参数指定的配置文件路径
#        param_config_path = self.get_parameter('config_path').get_parameter_value().string_value
#        
#        if param_config_path:
#            possible_paths.insert(0, param_config_path)  # 将用户指定的路径放在最前面
#            self.get_logger().info(f"用户指定的配置文件路径: {param_config_path}")
#        
        # 尝试每个可能的路径
        for config_path in possible_paths:
            try:
                with open(config_path, 'r', encoding='utf-8') as file:
                    config = yaml.safe_load(file)
                    self.get_logger().info(f"成功加载配置文件: {config_path}")
                    return config
            except Exception as e:
                self.get_logger().debug(f"无法从 {config_path} 加载配置文件: {str(e)}")
                continue
        
        # 如果所有路径都失败，记录警告并返回空字典
        self.get_logger().warn("无法找到配置文件，将使用默认值")
        return {}
    
    def sign_callback(self, msg):
        """处理sign4return信号"""
        self.get_logger().info(f"Received sign4return: {msg.data}")
        
        if msg.data == self.target_sign:
            self.get_logger().info(f"Received trigger signal (sign4return={self.target_sign}), subscribing to image topic...")
            self.waiting_for_image = True
            
            # 创建图像订阅者
            if self.image_sub is None:
                qos_profile = QoSProfile(depth=10)
                self.image_sub = self.create_subscription(
                    CompressedImage,
                    self.image_topic,
                    self.image_callback,
                    qos_profile
                )
                self.get_logger().info(f"Subscribed to {self.image_topic} topic")
    
    def image_callback(self, msg):
        """处理接收到的压缩图像"""
        if not self.waiting_for_image:
            return
        
        self.get_logger().info("Received compressed image, processing with AI model...")
        
        try:
            # 将压缩图像消息转换为OpenCV格式
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 调用AI模型分析图像
            result = self.analyze_image_with_ai(cv_image)
            
            if result:
                ai_model_msg = String()
                ai_model_msg.data = result
                self.ai_publisher.publish(ai_model_msg)
                self.get_logger().error(f"AI Analysis Result: {result}")
            else:
                self.get_logger().warn("Failed to get AI analysis result")
        
        except Exception as e:
            self.get_logger().error(f"Error processing compressed image: {str(e)}")
        
        finally:
            # 重置状态，取消订阅（如果只需要处理一帧）
            self.waiting_for_image = False
            if self.image_sub is not None:
                self.destroy_subscription(self.image_sub)
                self.image_sub = None
                self.get_logger().info("Unsubscribed from image topic")
    
    def encode_image_to_base64(self, cv_image):
        """将OpenCV图像编码为Base64字符串"""
        try:
            # 将BGR图像转换为RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # 转换为PIL Image
            pil_image = Image.fromarray(rgb_image)
            
            # 从配置文件获取图像质量和格式
            quality = self.config.get('image', {}).get('jpeg_quality', 85)
            format_type = self.config.get('image', {}).get('format', 'JPEG')
            
            # 编码为指定格式的Base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format=format_type, quality=quality)
            
            # 获取Base64编码
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return img_base64
        
        except Exception as e:
            self.get_logger().error(f"Error encoding image to base64: {str(e)}")
            return None
    
    def analyze_image_with_ai(self, cv_image):
        """使用火山引擎AI模型分析图像"""
        if self.ark_client is None:
            self.get_logger().warn("Ark client not available, skipping AI analysis")
            return "AI client not available"
        
        try:
            # 将图像编码为Base64
            base64_image = self.encode_image_to_base64(cv_image)
            if base64_image is None:
                return None
            
            # 调用火山引擎API
            prompt = self.config.get('volcengine', {}).get('prompt', "请详细描述这张图片中的内容，包括物体、场景、颜色等信息。")
            
            response = self.ark_client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },         
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            )
            
            # 提取响应内容
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                return content
            else:
                self.get_logger().warn("No response from AI model")
                return None
        
        except Exception as e:
            self.get_logger().error(f"Error calling AI model: {str(e)}")
            return None


def main(args=None):
    rclpy.init(args=args)
    
    node = VisionAINode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
