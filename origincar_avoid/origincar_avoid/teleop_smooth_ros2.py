#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

import sys
import select
import termios
import tty
import threading
import time


msg = """
---------------------------
ROS 2 普通小车键盘点动遥控
---------------------------

移动控制：
   w
a    d
   s

w : 向前点动一下
s : 向后点动一下
a : 向左转一下
d : 向右转一下

空格键 : 立即停止

CTRL-C 退出程序
"""


# =========================
# 点动速度参数
# =========================

LINEAR_SPEED = 0.4          # 前进/后退速度
TURN_LINEAR_SPEED = 0.25    # 转弯时的前进速度，普通小车建议保留一点前进速度
ANGULAR_SPEED = 2.0         # 左转/右转角速度

MOVE_DURATION = 0.2         # 每按一下运动多久，单位秒
PUBLISH_PERIOD = 0.01       # 发布频率，0.01 = 100Hz


class TeleopStepNode(Node):
    def __init__(self):
        super().__init__('teleop_step')

        self.twist_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.linear = 0.0
        self.angular = 0.0
        self.active_until = 0.0

        self.timer = self.create_timer(PUBLISH_PERIOD, self.publish_loop)

    def set_motion(self, linear, angular):
        """
        设置一次点动运动
        """
        self.linear = float(linear)
        self.angular = float(angular)
        self.active_until = time.time() + MOVE_DURATION

        self.publish_twist(self.linear, self.angular)

    def stop(self):
        """
        立即停止
        """
        self.linear = 0.0
        self.angular = 0.0
        self.active_until = 0.0
        self.publish_twist(0.0, 0.0)

    def publish_twist(self, linear, angular):
        twist = Twist()

        # 普通小车只控制前后速度 linear.x
        twist.linear.x = float(linear)
        twist.linear.y = 0.0
        twist.linear.z = 0.0

        # 普通小车左右转向用 angular.z
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = float(angular)

        self.twist_pub.publish(twist)

    def publish_loop(self):
        """
        定时发布速度。
        如果还在点动时间内，就继续发布速度；
        超过时间后，自动停止。
        """
        now = time.time()

        if now < self.active_until:
            self.publish_twist(self.linear, self.angular)
        else:
            self.publish_twist(0.0, 0.0)


def get_key(settings):
    """
    非阻塞获取键盘输入
    """
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.02)

    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def main(args=None):
    rclpy.init(args=args)

    node = TeleopStepNode()

    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True
    )
    spin_thread.start()

    settings = termios.tcgetattr(sys.stdin)

    print("\033[1;32m" + msg + "\033[0m")

    try:
        while rclpy.ok():
            key = get_key(settings)

            if key == 'w':
                # 前进
                node.set_motion(LINEAR_SPEED, 0.0)

            elif key == 's':
                # 后退
                node.set_motion(-LINEAR_SPEED, 0.0)

            elif key == 'a':
                # 左转：普通小车建议一边前进一边左转
                node.set_motion(TURN_LINEAR_SPEED, ANGULAR_SPEED)

            elif key == 'd':
                # 右转：普通小车建议一边前进一边右转
                node.set_motion(TURN_LINEAR_SPEED, -ANGULAR_SPEED)

            elif key == ' ':
                node.stop()

            elif key == '\x03':
                break

    except Exception as e:
        print(e)

    finally:
        node.stop()

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

        node.destroy_node()
        rclpy.shutdown()

        try:
            spin_thread.join(timeout=1.0)
        except RuntimeError:
            pass


if __name__ == '__main__':
    main()