#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

import sys
import select
import termios
import tty
import threading

# 提示信息
msg = """
---------------------------
ROS 2 键盘遥控控制面板
---------------------------
移动控制：
   w
a    d
   s

a/d : 向左 / 向右转弯
w/s : 前进 / 后退

空格键 (space) : 紧急停止

CTRL-C 退出程序
"""

# 速度参数（流畅版）
MAX_LINEAR_SPEED = 0.3      # 最大前进速度
MAX_ANGULAR_SPEED = 1.6     # 最大转向速度
LINEAR_ACC = 0.15           # 线性加速度
ANGULAR_ACC = 0.4           # 角加速度


class TeleopSmoothNode(Node):
    def __init__(self):
        super().__init__('teleop_smooth')
        
        # ROS 2 创建发布者，QoS 设为 10
        self.twist_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # 流畅控制核心变量
        self.target_linear = 0.0
        self.target_angular = 0.0
        self.current_linear = 0.0
        self.current_angular = 0.0

        # ROS 2 使用定时器替代原先独立的 while 循环 (20Hz = 0.05秒)
        self.timer = self.create_timer(0.05, self.publish_vel_loop)

    def publish_vel_loop(self):
        """定时器回调：持续平滑发布速度"""
        # 平滑加速度
        if self.current_linear < self.target_linear:
            self.current_linear = min(self.current_linear + LINEAR_ACC, self.target_linear)
        elif self.current_linear > self.target_linear:
            self.current_linear = max(self.current_linear - LINEAR_ACC, self.target_linear)
        
        if self.current_angular < self.target_angular:
            self.current_angular = min(self.current_angular + ANGULAR_ACC, self.target_angular)
        elif self.current_angular > self.target_angular:
            self.current_angular = max(self.current_angular - ANGULAR_ACC, self.target_angular)

        # 发布速度
        twist = Twist()
        twist.linear.x = self.current_linear
        twist.angular.z = self.current_angular
        self.twist_pub.publish(twist)


def get_key(settings):
    """非阻塞获取键盘输入"""
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
    key = sys.stdin.read(1) if rlist else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def main(args=None):
    rclpy.init(args=args)
    teleop_node = TeleopSmoothNode()

    # 启动后台线程来处理 ROS 2 的回调（包括定时器）
    # 这样主线程就不会被 spin 阻塞，可以继续监听键盘
    spin_thread = threading.Thread(target=rclpy.spin, args=(teleop_node,), daemon=True)
    spin_thread.start()

    settings = termios.tcgetattr(sys.stdin)
    print("\033[1;32m" + msg + "\033[0m")

    try:
        while rclpy.ok():
            key = get_key(settings)
            
            if key == 'w':
                teleop_node.target_linear = MAX_LINEAR_SPEED
            elif key == 's':
                teleop_node.target_linear = -MAX_LINEAR_SPEED
            elif key == 'a':
                teleop_node.target_angular = MAX_ANGULAR_SPEED
            elif key == 'd':
                teleop_node.target_angular = -MAX_ANGULAR_SPEED
            elif key == ' ':
                teleop_node.target_linear = 0.0
                teleop_node.target_angular = 0.0
            elif key == '\x03':  # CTRL-C
                break
            else:
                # 无按键 → 只减速角速度，线速度保持（更像真实车）
                teleop_node.target_angular = 0.0

    except Exception as e:
        print(e)
    finally:
        # 程序退出前，发送停止指令并恢复终端设置
        teleop_node.target_linear = 0.0
        teleop_node.target_angular = 0.0
        
        # 强制立即发布一次 0 速度，防止由于退出过快导致最后一次减速没有发出
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        teleop_node.twist_pub.publish(twist)
        
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        
        # 优雅关闭 ROS 2 节点
        teleop_node.destroy_node()
        rclpy.shutdown()
        spin_thread.join()

if __name__ == '__main__':
    main()