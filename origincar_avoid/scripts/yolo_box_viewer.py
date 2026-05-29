#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Empty

from ai_msgs.msg import PerceptionTargets


class YoloBoxViewer(Node):
    def __init__(self):
        super().__init__('yolo_box_viewer')

        # =========================
        # 图像和检测话题参数
        # =========================
        self.declare_parameter('image_topic', '/image_out/compressed')
        self.declare_parameter('detection_topic', '/hobot_dnn_detection')
        self.declare_parameter('output_compressed_topic', '/yolov8_result/compressed')
        self.declare_parameter('jpeg_quality', 80)

        # =========================
        # YOLO 避障输出话题
        # =========================
        self.declare_parameter('yolo_cmd_vel_topic', '/yolo_cmd_vel')
        self.declare_parameter('yolo_avoid_active_topic', '/yolo_avoid_active')
        self.declare_parameter('yolo_avoid_finished_topic', '/yolo_avoid_finished')

        # =========================
        # 避障判断参数
        # =========================
        self.declare_parameter('avoid_start_y_ratio', 0.65)
        self.declare_parameter('center_deadzone_ratio', 0.08)
        self.declare_parameter('linear_speed', 0.15)
        self.declare_parameter('turn_speed', 0.45)

        # 没检测到 roadblock 时是否让 YOLO 自己继续前进
        # 和 Nav2 配合时建议 False
        self.declare_parameter('forward_when_no_roadblock', False)

        # =========================
        # 更稳的避障完成判断参数
        # =========================

        # 连续多少帧 yolo_active=False，才认为避障完成
        # 建议 5~10，帧率高就设大一点
        self.declare_parameter('avoid_finish_confirm_frames', 6)

        # 避障完成事件最短间隔，单位秒
        # 防止短时间内重复发布 /yolo_avoid_finished
        self.declare_parameter('avoid_finished_cooldown_sec', 1.0)

        # =========================
        # 读取参数
        # =========================
        self.image_topic = self.get_parameter('image_topic').value
        self.detection_topic = self.get_parameter('detection_topic').value
        self.output_topic = self.get_parameter('output_compressed_topic').value
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)

        self.yolo_cmd_vel_topic = self.get_parameter('yolo_cmd_vel_topic').value
        self.yolo_avoid_active_topic = self.get_parameter('yolo_avoid_active_topic').value
        self.yolo_avoid_finished_topic = self.get_parameter(
            'yolo_avoid_finished_topic'
        ).value

        self.avoid_start_y_ratio = float(
            self.get_parameter('avoid_start_y_ratio').value
        )
        self.center_deadzone_ratio = float(
            self.get_parameter('center_deadzone_ratio').value
        )
        self.linear_speed = float(self.get_parameter('linear_speed').value)
        self.turn_speed = float(self.get_parameter('turn_speed').value)

        self.forward_when_no_roadblock = bool(
            self.get_parameter('forward_when_no_roadblock').value
        )

        self.avoid_finish_confirm_frames = int(
            self.get_parameter('avoid_finish_confirm_frames').value
        )
        self.avoid_finished_cooldown_sec = float(
            self.get_parameter('avoid_finished_cooldown_sec').value
        )

        self.latest_targets = []

        # =========================
        # 避障状态机变量
        # =========================

        # 当前是否处在一次 YOLO 避障过程里
        # 只要 YOLO 接管过一次，就进入 True
        # 避障完成确认后，再变回 False
        self.avoid_session_active = False

        # 连续多少帧不需要 YOLO 接管
        self.inactive_frame_count = 0

        # 上一次发布 /yolo_avoid_finished 的时间
        self.last_finished_time = None

        # =========================
        # 订阅图像
        # =========================
        self.image_sub = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            10
        )

        # =========================
        # 订阅 YOLO 检测结果
        # =========================
        self.det_sub = self.create_subscription(
            PerceptionTargets,
            self.detection_topic,
            self.detection_callback,
            10
        )

        # =========================
        # 发布画框后的图像
        # =========================
        self.result_pub = self.create_publisher(
            CompressedImage,
            self.output_topic,
            10
        )

        # =========================
        # 发布 YOLO 避障速度
        # =========================
        self.yolo_cmd_pub = self.create_publisher(
            Twist,
            self.yolo_cmd_vel_topic,
            10
        )

        # =========================
        # 发布 YOLO 是否接管控制权
        # True  ：cmd_vel_mux 使用 YOLO 速度
        # False ：cmd_vel_mux 使用 Nav2 速度
        # =========================
        self.yolo_active_pub = self.create_publisher(
            Bool,
            self.yolo_avoid_active_topic,
            10
        )

        # =========================
        # 发布 YOLO 避障完成事件
        # waypoint_manager 订阅该话题后重新发送当前目标点
        # =========================
        self.yolo_finished_pub = self.create_publisher(
            Empty,
            self.yolo_avoid_finished_topic,
            10
        )

        self.get_logger().info(f'Input image topic: {self.image_topic}')
        self.get_logger().info(f'Detection topic: {self.detection_topic}')
        self.get_logger().info(f'Output compressed topic: {self.output_topic}')
        self.get_logger().info(f'YOLO cmd vel topic: {self.yolo_cmd_vel_topic}')
        self.get_logger().info(f'YOLO avoid active topic: {self.yolo_avoid_active_topic}')
        self.get_logger().info(f'YOLO avoid finished topic: {self.yolo_avoid_finished_topic}')
        self.get_logger().info(f'Linear speed: {self.linear_speed}')
        self.get_logger().info(f'Turn speed: {self.turn_speed}')
        self.get_logger().info(
            f'Avoid finish confirm frames: {self.avoid_finish_confirm_frames}'
        )
        self.get_logger().info(
            f'Avoid finished cooldown: {self.avoid_finished_cooldown_sec} sec'
        )

    def detection_callback(self, msg):
        targets = []

        for target in msg.targets:
            target_type = str(target.type)

            for roi in target.rois:
                rect = roi.rect

                x = int(rect.x_offset)
                y = int(rect.y_offset)
                w = int(rect.width)
                h = int(rect.height)
                score = float(roi.confidence)

                if w <= 0 or h <= 0:
                    continue

                targets.append({
                    'label': target_type,
                    'score': score,
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                })

        self.latest_targets = targets

    def publish_yolo_active(self, active):
        msg = Bool()
        msg.data = bool(active)
        self.yolo_active_pub.publish(msg)

    def publish_cmd(self, linear_x, angular_z):
        cmd = Twist()

        cmd.linear.x = float(linear_x)
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0

        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = float(angular_z)

        self.yolo_cmd_pub.publish(cmd)

    def stop_yolo_cmd(self):
        self.publish_cmd(0.0, 0.0)

    def can_publish_finished_event(self):
        """
        检查是否允许发布避障完成事件。
        用 cooldown 防止短时间内重复发布。
        """
        now = self.get_clock().now()

        if self.last_finished_time is None:
            return True

        dt = (now - self.last_finished_time).nanoseconds / 1e9

        return dt >= self.avoid_finished_cooldown_sec

    def publish_avoid_finished(self):
        """
        发布避障完成事件。

        waypoint_manager 订阅 /yolo_avoid_finished 后，
        重新发送当前目标点，让 Nav2 重新规划。
        """
        if not self.can_publish_finished_event():
            return

        msg = Empty()
        self.yolo_finished_pub.publish(msg)
        self.last_finished_time = self.get_clock().now()

        self.get_logger().info(
            'YOLO avoid finished, published /yolo_avoid_finished'
        )

    def update_avoid_finish_state(self, yolo_active):
        """
        更稳的避障完成判断。

        逻辑：
        1. 只要 yolo_active=True，说明 YOLO 正在接管避障，
           进入一次避障会话 avoid_session_active=True。
        2. 当 yolo_active=False 时，不立刻认为完成，
           而是累计 inactive_frame_count。
        3. 只有连续 N 帧 yolo_active=False，
           才认为本次避障真正完成。
        4. 避障完成后发布一次 /yolo_avoid_finished。
        """

        # YOLO 正在接管
        if yolo_active:
            self.avoid_session_active = True
            self.inactive_frame_count = 0
            return False

        # YOLO 没有接管，但之前也没有进入过避障会话
        # 说明一直是普通 Nav2 控制，不需要发布完成事件
        if not self.avoid_session_active:
            self.inactive_frame_count = 0
            return False

        # 到这里说明：
        # 之前进入过避障会话，现在 yolo_active=False
        # 开始累计连续非接管帧数
        self.inactive_frame_count += 1

        if self.inactive_frame_count >= self.avoid_finish_confirm_frames:
            self.publish_avoid_finished()

            # 本次避障会话结束，重置状态
            self.avoid_session_active = False
            self.inactive_frame_count = 0

            return True

        return False

    def image_callback(self, msg):
        data = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn('Failed to decode compressed image')
            return

        frame_h, frame_w = frame.shape[:2]

        frame_center_x = frame_w / 2.0
        avoid_start_y = frame_h * self.avoid_start_y_ratio
        deadzone = frame_w * self.center_deadzone_ratio

        # 用来记录最靠近车的 roadblock
        nearest_roadblock = None
        max_bottom_y = -1

        avoid_action = 'nav2 control'
        yolo_active = False

        # =========================
        # 遍历 YOLO 检测目标
        # =========================
        for t in self.latest_targets:
            x1 = max(0, t['x'])
            y1 = max(0, t['y'])
            x2 = min(frame_w - 1, t['x'] + t['w'])
            y2 = min(frame_h - 1, t['y'] + t['h'])

            label = t['label']
            score = t['score']

            # =========================
            # roadblock：只画底部红线
            # =========================
            if label == 'roadblock':
                bottom_y = y2
                red_center_x = (x1 + x2) / 2.0

                # 记录最靠近车的 roadblock
                if bottom_y > max_bottom_y:
                    max_bottom_y = bottom_y
                    nearest_roadblock = {
                        'center_x': red_center_x,
                        'bottom_y': bottom_y,
                        'x1': x1,
                        'x2': x2,
                        'y1': y1,
                        'y2': y2,
                        'score': score,
                    }

                # 画 roadblock 底部红线
                cv2.line(
                    frame,
                    (x1, bottom_y),
                    (x2, bottom_y),
                    (0, 0, 255),
                    2
                )

                # 画红线中心点
                cv2.circle(
                    frame,
                    (int(red_center_x), int(bottom_y)),
                    5,
                    (0, 0, 255),
                    -1
                )

                text = f"{label} {score:.2f}"
                cv2.putText(
                    frame,
                    text,
                    (x1, max(20, bottom_y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            # =========================
            # 其他目标：正常画绿色框
            # =========================
            else:
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                text = f"{label} {score:.2f}"
                cv2.putText(
                    frame,
                    text,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # =========================
        # 画画面中心线和避障触发线
        # =========================
        cv2.line(
            frame,
            (int(frame_center_x), 0),
            (int(frame_center_x), frame_h),
            (255, 255, 0),
            1
        )

        cv2.line(
            frame,
            (0, int(avoid_start_y)),
            (frame_w, int(avoid_start_y)),
            (255, 0, 255),
            1
        )

        # =========================
        # 避障逻辑
        # =========================
        if nearest_roadblock is not None:
            red_center_x = nearest_roadblock['center_x']
            red_bottom_y = nearest_roadblock['bottom_y']

            # 只有红线超过设定 Y 值，才让 YOLO 接管控制
            if red_bottom_y >= avoid_start_y:
                yolo_active = True
                offset = red_center_x - frame_center_x

                # 障碍物在左边，小车向右转
                if offset < -deadzone:
                    self.publish_cmd(self.linear_speed, -self.turn_speed)
                    avoid_action = 'YOLO active: roadblock left -> turn right'

                # 障碍物在右边，小车向左转
                elif offset > deadzone:
                    self.publish_cmd(self.linear_speed, self.turn_speed)
                    avoid_action = 'YOLO active: roadblock right -> turn left'

                # 障碍物居中，默认向左转
                else:
                    self.publish_cmd(self.linear_speed, self.turn_speed)
                    avoid_action = 'YOLO active: roadblock center -> turn left'

            else:
                # 检测到 roadblock，但是还没到避障距离
                yolo_active = False
                self.stop_yolo_cmd()
                avoid_action = 'roadblock far -> NAV2 control'

        else:
            # 没有检测到 roadblock
            yolo_active = False

            if self.forward_when_no_roadblock:
                # 不建议在 Nav2 模式下开启
                yolo_active = True
                self.publish_cmd(self.linear_speed, 0.0)
                avoid_action = 'YOLO active: no roadblock -> forward'
            else:
                self.stop_yolo_cmd()
                avoid_action = 'no roadblock -> NAV2 control'

        # =========================
        # 更稳的避障完成判断
        # =========================
        finished_now = self.update_avoid_finish_state(yolo_active)

        if finished_now:
            avoid_action = avoid_action + ' | avoid finished -> request replan'

        # 发布 YOLO 是否接管
        self.publish_yolo_active(yolo_active)

        # =========================
        # 显示调试信息
        # =========================
        cv2.putText(
            frame,
            'YOLOv8 X5',
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f'avoid: {avoid_action}',
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f'yolo_active: {yolo_active}',
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f'session: {self.avoid_session_active}  inactive: {self.inactive_frame_count}/{self.avoid_finish_confirm_frames}',
            (10, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        if nearest_roadblock is not None:
            red_center_x = nearest_roadblock['center_x']
            red_bottom_y = nearest_roadblock['bottom_y']

            cv2.putText(
                frame,
                f'red_center_x: {int(red_center_x)}  bottom_y: {int(red_bottom_y)}',
                (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

        # =========================
        # 编码并发布可视化图像
        # =========================
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        ok, encoded = cv2.imencode('.jpg', frame, encode_param)

        if not ok:
            self.get_logger().warn('Failed to encode result image')
            return

        out = CompressedImage()
        out.header = msg.header
        out.format = 'jpeg'
        out.data = encoded.tobytes()
        self.result_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = YoloBoxViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_yolo_active(False)
        node.stop_yolo_cmd()

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()