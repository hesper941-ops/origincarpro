import rclpy 
from rclpy.node import Node 
from ai_msgs.msg import PerceptionTargets 
from std_msgs.msg import Int16MultiArray 
from sensor_msgs.msg import Image  
from cv_bridge import CvBridge  
import cv2  
import numpy as np  

 
class MinimalSubscriber(Node): 
    def __init__(self): 
        super().__init__('minimal_subscriber') 
        self.bridge = CvBridge()
        self.subscription = self.create_subscription( 
            Int16MultiArray, 
            'Coordinate',  
            self.coord_callback, 
            10) 
        self.image_sub = self.create_subscription(  
            Image,  
            'image_bgr8',  
            self.image_callback,  
            10) 
        self.image_pub = self.create_publisher(  
            Image,  
            'camera/process_image',  
            10)  
        self.subscription  # prevent unused variable warning 
        self.count = 0
        self.getCoord_Flag = 0
        self.x = 0
        self.y = 0

    def coord_callback(self, msg):         #坐标获取
        self.x = msg.data[0];
        self.y = msg.data[1];
        print('x=',self.x,'y=', self.y)
        
    def image_callback(self, msg):  
        try:  
            # 将ROS图像消息转换为OpenCV图像  
            cv_image = self.bridge.imgmsg_to_cv2(msg,'bgr8')  
  
            # 使用OpenCV进行缩放处理  
            resized_image = cv2.circle(cv_image, (self.x, self.y), 2, (0, 0, 255), -1)  
            #jpeg_image = cv2.imencode('.jpg', resized_image)[1]
            # 将处理后的OpenCV图像转换回ROS图像消息  
            output_msg = self.bridge.cv2_to_imgmsg(resized_image, 'bgr8')  
            output_msg.header.stamp = self.get_clock().now().to_msg()  
            output_msg.header.frame_id = msg.header.frame_id  
  
            # 发布处理后的图像消息  
            self.image_pub.publish(output_msg)  
        except Exception as e:  
            self.get_logger().error(f'Error in image_callback: {e}')  
 
def main(args=None): 
    rclpy.init(args=args) 
    minimal_subscriber = MinimalSubscriber() 
    rclpy.spin(minimal_subscriber) 
    minimal_subscriber.destroy_node() 
    rclpy.shutdown() 

# 给main函数一个入口，省得colcon build编译
if __name__ == '__main__': 
    main()