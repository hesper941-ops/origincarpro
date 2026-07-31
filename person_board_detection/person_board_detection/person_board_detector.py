#!/usr/bin/env python3

import os
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Float32, Int32MultiArray

from hobot_dnn import pyeasy_dnn as dnn


INPUT_WIDTH = 640
INPUT_HEIGHT = 640
STRIDES = (8, 16, 32)
REG_MAX = 16


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def ros_image_to_bgr(msg: Image) -> np.ndarray:
    """将常见 sensor_msgs/Image 编码转换为 BGR。"""
    encoding = msg.encoding.lower()
    raw = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding in ('bgr8', 'rgb8', '8uc3'):
        required = msg.height * msg.step
        if raw.size < required:
            raise ValueError(
                f'图像数据不足：{raw.size} < {required}'
            )

        rows = raw[:required].reshape(msg.height, msg.step)
        packed = rows[:, :msg.width * 3]
        image = packed.reshape(msg.height, msg.width, 3).copy()

        if encoding == 'rgb8':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image

    if encoding in ('mono8', '8uc1'):
        required = msg.height * msg.step
        rows = raw[:required].reshape(msg.height, msg.step)
        gray = rows[:, :msg.width].copy()
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if encoding in ('yuyv', 'yuy2', 'yuv422', 'yuv422_yuy2'):
        required = msg.height * msg.step
        rows = raw[:required].reshape(msg.height, msg.step)
        packed = rows[:, :msg.width * 2]
        yuyv = packed.reshape(msg.height, msg.width, 2)
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)

    if encoding in ('uyvy', 'yuv422_uyvy'):
        required = msg.height * msg.step
        rows = raw[:required].reshape(msg.height, msg.step)
        packed = rows[:, :msg.width * 2]
        uyvy = packed.reshape(msg.height, msg.width, 2)
        return cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)

    if encoding == 'nv12':
        required = msg.width * msg.height * 3 // 2
        if raw.size < required:
            raise ValueError(
                f'NV12 数据不足：{raw.size} < {required}'
            )

        nv12 = raw[:required].reshape(
            msg.height * 3 // 2,
            msg.width,
        )

        return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)

    raise ValueError(f'暂不支持图像编码：{msg.encoding}')


def letterbox(
    image: np.ndarray,
    target_width: int = INPUT_WIDTH,
    target_height: int = INPUT_HEIGHT,
) -> Tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]

    scale = min(
        target_width / float(width),
        target_height / float(height),
    )

    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_width = target_width - resized_width
    pad_height = target_height - resized_height

    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    return padded, scale, pad_left, pad_top


def bgr_to_nv12(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]

    if height % 2 or width % 2:
        raise ValueError(
            f'NV12 要求偶数宽高，当前：{width}x{height}'
        )

    i420 = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
    flattened = i420.reshape(-1)

    y_size = width * height
    uv_size = y_size // 4

    y_plane = flattened[:y_size]
    u_plane = flattened[y_size:y_size + uv_size]
    v_plane = flattened[
        y_size + uv_size:y_size + uv_size * 2
    ]

    nv12 = np.empty(y_size + y_size // 2, dtype=np.uint8)
    nv12[:y_size] = y_plane
    nv12[y_size::2] = u_plane
    nv12[y_size + 1::2] = v_plane

    return np.ascontiguousarray(
        nv12.reshape(height * 3 // 2, width)
    )


def decode_outputs(
    outputs: Sequence,
    confidence_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(outputs) != 6:
        raise RuntimeError(
            f'模型输出数量为 {len(outputs)}，期望 6'
        )

    projection = np.arange(REG_MAX, dtype=np.float32)

    all_boxes: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []

    for scale_index, stride in enumerate(STRIDES):
        cls_output = np.asarray(
            outputs[scale_index * 2].buffer,
            dtype=np.float32,
        ).squeeze(axis=0)

        box_output = np.asarray(
            outputs[scale_index * 2 + 1].buffer,
            dtype=np.float32,
        ).squeeze(axis=0)

        if cls_output.ndim == 2:
            cls_output = cls_output[..., np.newaxis]

        if cls_output.shape[-1] != 1:
            raise RuntimeError(
                f'stride={stride} 分类输出异常：'
                f'{cls_output.shape}'
            )

        if box_output.shape[-1] != 4 * REG_MAX:
            raise RuntimeError(
                f'stride={stride} DFL 输出异常：'
                f'{box_output.shape}'
            )

        score_map = sigmoid(cls_output[..., 0])

        selected_y, selected_x = np.where(
            score_map >= confidence_threshold
        )

        if selected_x.size == 0:
            continue

        scores = score_map[selected_y, selected_x]

        logits = box_output[
            selected_y,
            selected_x,
            :,
        ].reshape(-1, 4, REG_MAX)

        distributions = softmax(logits, axis=-1)

        distances = np.sum(
            distributions * projection,
            axis=-1,
        ) * float(stride)

        center_x = (
            selected_x.astype(np.float32) + 0.5
        ) * float(stride)

        center_y = (
            selected_y.astype(np.float32) + 0.5
        ) * float(stride)

        x1 = center_x - distances[:, 0]
        y1 = center_y - distances[:, 1]
        x2 = center_x + distances[:, 2]
        y2 = center_y + distances[:, 3]

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        all_boxes.append(boxes)
        all_scores.append(scores)

    if not all_boxes:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)

    boxes[:, [0, 2]] = np.clip(
        boxes[:, [0, 2]],
        0.0,
        INPUT_WIDTH - 1.0,
    )

    boxes[:, [1, 3]] = np.clip(
        boxes[:, [1, 3]],
        0.0,
        INPUT_HEIGHT - 1.0,
    )

    return boxes, scores


def apply_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    confidence_threshold: float,
    nms_threshold: float,
    max_detections: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if boxes.shape[0] == 0:
        return boxes, scores

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    valid = np.logical_and(widths > 1.0, heights > 1.0)
    boxes = boxes[valid]
    scores = scores[valid]

    if boxes.shape[0] == 0:
        return boxes, scores

    boxes_xywh = np.column_stack(
        (
            boxes[:, 0],
            boxes[:, 1],
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
        )
    )

    indices = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(),
        scores.tolist(),
        float(confidence_threshold),
        float(nms_threshold),
    )

    if indices is None or len(indices) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    indices = np.asarray(indices).reshape(-1)
    indices = indices[np.argsort(scores[indices])[::-1]]
    indices = indices[:max_detections]

    return boxes[indices], scores[indices]


def restore_boxes(
    boxes: np.ndarray,
    scale: float,
    pad_left: int,
    pad_top: int,
    original_width: int,
    original_height: int,
) -> np.ndarray:
    if boxes.shape[0] == 0:
        return boxes

    restored = boxes.copy()

    restored[:, [0, 2]] -= float(pad_left)
    restored[:, [1, 3]] -= float(pad_top)

    restored[:, [0, 2]] /= scale
    restored[:, [1, 3]] /= scale

    restored[:, [0, 2]] = np.clip(
        restored[:, [0, 2]],
        0.0,
        original_width - 1.0,
    )

    restored[:, [1, 3]] = np.clip(
        restored[:, [1, 3]],
        0.0,
        original_height - 1.0,
    )

    return restored


class PersonBoardDetector(Node):

    def __init__(self) -> None:
        super().__init__('person_board_detector')

        self.declare_parameter(
            'model_path',
            (
                '/root/intelligent_car_ws/models/person_board/'
                'person_board_yolov8n_v0_bayese_640x640_nv12.bin'
            ),
        )
        self.declare_parameter(
            'image_topic',
            '/aurora/rgb/image_raw',
        )
        self.declare_parameter('confidence_threshold', 0.35)
        self.declare_parameter('nms_threshold', 0.50)
        self.declare_parameter('max_detections', 10)
        self.declare_parameter('publish_result_image', True)
        self.declare_parameter('save_debug_image', True)
        self.declare_parameter(
            'debug_image_path',
            '/root/person_board_debug/latest.jpg',
        )
        self.declare_parameter('log_every_n', 30)

        self.model_path = str(
            self.get_parameter('model_path').value
        )
        self.image_topic = str(
            self.get_parameter('image_topic').value
        )
        self.confidence_threshold = float(
            self.get_parameter(
                'confidence_threshold'
            ).value
        )
        self.nms_threshold = float(
            self.get_parameter('nms_threshold').value
        )
        self.max_detections = int(
            self.get_parameter('max_detections').value
        )
        self.publish_result_image = bool(
            self.get_parameter(
                'publish_result_image'
            ).value
        )
        self.save_debug_image = bool(
            self.get_parameter('save_debug_image').value
        )
        self.debug_image_path = str(
            self.get_parameter('debug_image_path').value
        )
        self.log_every_n = max(
            1,
            int(self.get_parameter('log_every_n').value),
        )

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f'模型不存在：{self.model_path}'
            )

        self.get_logger().info(
            f'加载模型：{self.model_path}'
        )

        models = dnn.load(self.model_path)

        if not models:
            raise RuntimeError('pyeasy_dnn 未加载到模型')

        self.model = models[0]

        self.get_logger().info(
            f'模型加载成功：{self.model.name}，'
            f'输出数量={len(self.model.outputs)}'
        )

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.image_subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos,
        )

        self.detected_publisher = self.create_publisher(
            Bool,
            '/person_board/detected',
            10,
        )
        self.score_publisher = self.create_publisher(
            Float32,
            '/person_board/score',
            10,
        )
        self.box_publisher = self.create_publisher(
            Int32MultiArray,
            '/person_board/box',
            10,
        )
        self.inference_publisher = self.create_publisher(
            Float32,
            '/person_board/inference_ms',
            10,
        )
        self.result_publisher = self.create_publisher(
            CompressedImage,
            '/person_board/result/compressed',
            10,
        )

        self.frame_count = 0
        self.last_detected = False
        self.last_debug_save_time = 0.0

        self.get_logger().info(
            f'等待图像：{self.image_topic}'
        )
        self.get_logger().info(
            '输出：/person_board/detected、'
            '/person_board/score、/person_board/box、'
            '/person_board/result/compressed'
        )

    def publish_detection(
        self,
        detected: bool,
        score: float,
        box: Sequence[int],
        inference_ms: float,
    ) -> None:
        detected_msg = Bool()
        detected_msg.data = detected
        self.detected_publisher.publish(detected_msg)

        score_msg = Float32()
        score_msg.data = float(score)
        self.score_publisher.publish(score_msg)

        box_msg = Int32MultiArray()
        box_msg.data = [int(value) for value in box]
        self.box_publisher.publish(box_msg)

        inference_msg = Float32()
        inference_msg.data = float(inference_ms)
        self.inference_publisher.publish(inference_msg)

    def draw_result(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        result = image.copy()

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = [
                int(round(value)) for value in box
            ]

            cv2.rectangle(
                result,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            text = f'person_board {float(score):.3f}'

            cv2.putText(
                result,
                text,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return result

    def publish_result(
        self,
        result: np.ndarray,
        source_msg: Image,
    ) -> None:
        if not self.publish_result_image:
            return

        success, encoded = cv2.imencode(
            '.jpg',
            result,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )

        if not success:
            return

        output = CompressedImage()
        output.header = source_msg.header
        output.format = 'jpeg'
        output.data = encoded.tobytes()

        self.result_publisher.publish(output)

    def save_debug_result(
        self,
        result: np.ndarray,
        detected: bool,
    ) -> None:
        if not self.save_debug_image or not detected:
            return

        now = time.monotonic()

        if now - self.last_debug_save_time < 1.0:
            return

        path = Path(self.debug_image_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(path), result)
        self.last_debug_save_time = now

    def image_callback(self, msg: Image) -> None:
        total_start = time.perf_counter()

        try:
            image = ros_image_to_bgr(msg)

            original_height, original_width = image.shape[:2]

            padded, scale, pad_left, pad_top = letterbox(
                image,
                INPUT_WIDTH,
                INPUT_HEIGHT,
            )

            nv12 = bgr_to_nv12(padded)

            inference_start = time.perf_counter()
            outputs = self.model.forward(nv12)
            inference_ms = (
                time.perf_counter() - inference_start
            ) * 1000.0

            boxes, scores = decode_outputs(
                outputs,
                self.confidence_threshold,
            )

            boxes, scores = apply_nms(
                boxes,
                scores,
                self.confidence_threshold,
                self.nms_threshold,
                self.max_detections,
            )

            boxes = restore_boxes(
                boxes,
                scale,
                pad_left,
                pad_top,
                original_width,
                original_height,
            )

            detected = boxes.shape[0] > 0

            if detected:
                best_index = int(np.argmax(scores))
                best_score = float(scores[best_index])
                best_box = [
                    int(round(value))
                    for value in boxes[best_index]
                ]
            else:
                best_score = 0.0
                best_box = []

            self.publish_detection(
                detected,
                best_score,
                best_box,
                inference_ms,
            )

            result = self.draw_result(
                image,
                boxes,
                scores,
            )

            self.publish_result(result, msg)
            self.save_debug_result(result, detected)

            self.frame_count += 1

            total_ms = (
                time.perf_counter() - total_start
            ) * 1000.0

            state_changed = detected != self.last_detected

            if (
                state_changed
                or self.frame_count % self.log_every_n == 0
            ):
                self.get_logger().info(
                    f'frame={self.frame_count}, '
                    f'encoding={msg.encoding}, '
                    f'detected={detected}, '
                    f'score={best_score:.3f}, '
                    f'box={best_box}, '
                    f'infer={inference_ms:.2f} ms, '
                    f'total={total_ms:.2f} ms'
                )

            self.last_detected = detected

        except Exception as exc:
            self.get_logger().error(
                f'处理图像失败：{type(exc).__name__}: {exc}'
            )


def main(args=None) -> None:
    rclpy.init(args=args)

    node = PersonBoardDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
