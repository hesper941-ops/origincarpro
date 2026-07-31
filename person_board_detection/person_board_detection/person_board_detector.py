#!/usr/bin/env python3
"""ROS 2 person-board detector with optional gated three-frame capture."""

import json
import os
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Float32, Int32MultiArray, String

from hobot_dnn import pyeasy_dnn as dnn

from .adaptive_scheduler import AdaptiveScheduler
from .bbox_utils import (
    bbox_iou,
    center_shift_ratio,
    edge_touch_ratio,
    normalized_area,
    padded_bbox,
    select_best_detection,
    short_side,
)
from .capture_manager import CaptureManager, CapturedFrame
from .crop_quality import evaluate_crop


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
        super().__init__("person_board_detector")
        self._declare_parameters()
        self._read_parameters()
        self._validate_parameters()
        self.image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.detected_publisher = self.create_publisher(
            Bool, "/person_board/detected", 10
        )
        self.score_publisher = self.create_publisher(Float32, "/person_board/score", 10)
        self.box_publisher = self.create_publisher(
            Int32MultiArray, "/person_board/box", 10
        )
        self.inference_publisher = self.create_publisher(
            Float32, "/person_board/inference_ms", 10
        )
        self.result_publisher = self.create_publisher(
            CompressedImage, "/person_board/result/compressed", 10
        )
        self.status_publisher = self.create_publisher(
            String, "/person_board/capture_status", 10
        )
        self.crop_publisher = self.create_publisher(
            CompressedImage, "/person_board/crop/compressed", 10
        )
        self.batch_publisher = self.create_publisher(
            String, "/person_board/capture_batch", 10
        )
        self.done_publisher = self.create_publisher(
            String, "/person_board/capture_done", 10
        )
        self.control_subscription = self.create_subscription(
            String, self.control_topic, self.control_callback, 10
        )

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"model file does not exist: {self.model_path}")
        self.get_logger().info(f"Loading model: {self.model_path}")
        models = dnn.load(self.model_path)
        if not models:
            raise RuntimeError("pyeasy_dnn returned no models")
        self.model = models[0]
        self.get_logger().info(
            f"Model preloaded: {self.model.name}, outputs={len(self.model.outputs)}"
        )

        self.scheduler = AdaptiveScheduler(
            {
                "SEARCH": self.search_inference_hz,
                "TRACK_FAR": self.far_inference_hz,
                "TRACK_MIDDLE": self.middle_inference_hz,
                "TRACK_NEAR": self.near_inference_hz,
                "CAPTURE_BURST": self.capture_inference_hz,
            },
            self.bbox_size_ema_alpha,
            self.frequency_switch_confirm_count,
            self.middle_enter_short_side_px,
            self.middle_exit_short_side_px,
            self.near_enter_short_side_px,
            self.near_exit_short_side_px,
        )
        self.capture_manager = CaptureManager(
            self.runtime_directory,
            self.fixed_capture_subdirectory,
            self.capture_required_count,
            self.capture_max_attempts,
            self.capture_interval_ms,
            self.capture_timeout_sec,
            self.jpeg_quality,
            self.image_topic,
        )
        self.image_subscription = None
        self.state = "PRELOADED_IDLE" if self.capture_mode else "SEARCH"
        self.enabled = not self.capture_mode
        self.event_id = ""
        self.used_event_ids = set()
        self.frame_received_count = 0
        self.inference_count = 0
        self.frame_count = 0
        self.last_detected = False
        self.last_debug_save_time = 0.0
        self.selected_bbox: List[int] = []
        self.selected_score = 0.0
        self.selected_short_side = 0.0
        self.selected_normalized_area = 0.0
        self.stable_count = 0
        self.previous_bbox: Optional[Tuple[int, int, int, int]] = None
        self.error_reason = ""
        self.last_rejection_reason = ""
        self.shutdown_timer = None
        self.status_timer = self.create_timer(0.5, self.status_timer_callback)
        if not self.capture_mode or self.subscribe_image_while_idle:
            self._create_image_subscription()
        self.publish_status()
        subscription_active = self.image_subscription is not None
        self.get_logger().info(
            f"State={self.state}, image_subscription_active={subscription_active}"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "model_path": (
                "/root/intelligent_car_ws/models/person_board/"
                "person_board_yolov8n_v0_bayese_640x640_nv12.bin"
            ),
            "image_topic": "/aurora/rgb/image_raw",
            "control_topic": "/person_board/control",
            "capture_mode": False,
            "preload_model": True,
            "subscribe_image_while_idle": True,
            "one_shot_mode": False,
            "shutdown_after_capture": False,
            "shutdown_publish_grace_sec": 0.5,
            "confidence_threshold": 0.35,
            "nms_threshold": 0.50,
            "max_detections": 10,
            "publish_result_image": True,
            "save_debug_image": True,
            "debug_image_path": "/root/person_board_debug/latest.jpg",
            "log_every_n": 30,
            "search_inference_hz": 5.0,
            "far_inference_hz": 3.0,
            "middle_inference_hz": 6.0,
            "near_inference_hz": 10.0,
            "capture_inference_hz": 10.0,
            "bbox_size_ema_alpha": 0.30,
            "frequency_switch_confirm_count": 3,
            "middle_enter_short_side_px": 60.0,
            "middle_exit_short_side_px": 50.0,
            "near_enter_short_side_px": 100.0,
            "near_exit_short_side_px": 85.0,
            "minimum_yolo_confidence": 0.60,
            "minimum_detection_area": 16.0,
            "capture_min_short_side_px": 90.0,
            "stable_detection_count": 3,
            "stable_bbox_iou": 0.60,
            "maximum_center_shift_ratio": 0.20,
            "crop_padding_ratio": 0.10,
            "minimum_crop_width": 80,
            "minimum_crop_height": 80,
            "jpeg_quality": 88,
            "minimum_sharpness": 30.0,
            "maximum_edge_touch_ratio": 0.20,
            "capture_required_count": 3,
            "capture_max_attempts": 8,
            "capture_interval_ms": 120,
            "capture_timeout_sec": 2.0,
            "publish_debug_crop": True,
            "runtime_directory": "/root/intelligent_car_ws/runtime/person_board",
            "fixed_capture_subdirectory": CaptureManager.DEFAULT_SUBDIRECTORY,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)

    def _read_parameters(self) -> None:
        for name in self._parameters:
            setattr(self, name, self.get_parameter(name).value)
        for name in (
            "model_path",
            "image_topic",
            "control_topic",
            "debug_image_path",
            "runtime_directory",
            "fixed_capture_subdirectory",
        ):
            setattr(self, name, str(getattr(self, name)))

    def _validate_parameters(self) -> None:
        errors = []
        if not self.preload_model:
            errors.append("preload_model must be true")
        if self.image_topic != "/aurora/rgb/image_raw":
            errors.append("image_topic must be /aurora/rgb/image_raw")
        if not self.control_topic:
            errors.append("control_topic must not be empty")
        unit_values = (
            "confidence_threshold",
            "nms_threshold",
            "minimum_yolo_confidence",
            "stable_bbox_iou",
            "maximum_center_shift_ratio",
            "bbox_size_ema_alpha",
            "crop_padding_ratio",
            "maximum_edge_touch_ratio",
        )
        for name in unit_values:
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                errors.append(f"{name} must be in [0, 1]")
        positive_values = (
            "search_inference_hz",
            "far_inference_hz",
            "middle_inference_hz",
            "near_inference_hz",
            "capture_inference_hz",
            "capture_timeout_sec",
            "minimum_detection_area",
            "capture_min_short_side_px",
            "minimum_sharpness",
        )
        for name in positive_values:
            if float(getattr(self, name)) <= 0.0:
                errors.append(f"{name} must be > 0")
        positive_integers = (
            "max_detections",
            "log_every_n",
            "frequency_switch_confirm_count",
            "stable_detection_count",
            "minimum_crop_width",
            "minimum_crop_height",
            "capture_max_attempts",
            "capture_interval_ms",
        )
        for name in positive_integers:
            if int(getattr(self, name)) <= 0:
                errors.append(f"{name} must be > 0")
        if int(self.capture_required_count) != 3:
            errors.append("capture_required_count must be 3")
        if int(self.capture_max_attempts) < int(self.capture_required_count):
            errors.append("capture_max_attempts must be >= capture_required_count")
        if not 1 <= int(self.jpeg_quality) <= 100:
            errors.append("jpeg_quality must be in [1, 100]")
        if not (
            0.0
            < self.middle_exit_short_side_px
            < self.middle_enter_short_side_px
            < self.near_enter_short_side_px
        ):
            errors.append("require 0 < middle_exit < middle_enter < near_enter")
        if (
            not self.middle_enter_short_side_px
            < self.near_exit_short_side_px
            < self.near_enter_short_side_px
        ):
            errors.append("near_exit must be between middle_enter and near_enter")
        if self.shutdown_after_capture and not self.one_shot_mode:
            errors.append("shutdown_after_capture requires one_shot_mode=true")
        if float(self.shutdown_publish_grace_sec) < 0.0:
            errors.append("shutdown_publish_grace_sec must be >= 0")
        if self.save_debug_image and not self.debug_image_path:
            errors.append("debug_image_path must not be empty when saving is enabled")
        if not self.runtime_directory:
            errors.append("runtime_directory must not be empty")
        if errors:
            raise ValueError("; ".join(errors))

    def _create_image_subscription(self) -> None:
        if self.image_subscription is None:
            self.image_subscription = self.create_subscription(
                Image, self.image_topic, self.image_callback, self.image_qos
            )

    def _destroy_image_subscription(self) -> None:
        if self.image_subscription is not None:
            self.destroy_subscription(self.image_subscription)
            self.image_subscription = None

    def control_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                raise ValueError("control payload must be a JSON object")
            command = payload.get("command")
            event_id = payload.get("event_id", "")
            if command not in ("start", "stop", "reset"):
                raise ValueError("command must be start, stop, or reset")
            if command == "start":
                self._handle_start(event_id)
            elif command == "stop":
                self._handle_stop(event_id)
            else:
                self._handle_reset(event_id)
        except Exception as exc:
            self.error_reason = f"invalid_control: {exc}"
            self.get_logger().error(self.error_reason)
            self.publish_status()

    def _handle_start(self, event_id: object) -> None:
        if not self.capture_mode:
            raise ValueError("start is only valid in capture_mode")
        if not isinstance(event_id, str) or not CaptureManager.is_valid_event_id(
            event_id
        ):
            raise ValueError("event_id is missing or contains unsafe characters")
        if self.enabled:
            raise ValueError(f"capture already active for event_id={self.event_id}")
        if event_id in self.used_event_ids:
            raise ValueError(f"duplicate event_id rejected: {event_id}")
        self.used_event_ids.add(event_id)
        self.event_id = event_id
        self.enabled = True
        self.error_reason = ""
        self.last_rejection_reason = ""
        self.stable_count = 0
        self.previous_bbox = None
        self.scheduler.reset()
        self.capture_manager.reset_batch()
        self.state = "SEARCH"
        self._create_image_subscription()
        self.get_logger().info(f"Started event_id={event_id}")
        self.publish_status()

    def _handle_stop(self, event_id: object) -> None:
        if event_id and event_id != self.event_id:
            raise ValueError(f"stop event_id does not match active event: {event_id}")
        self.enabled = False
        self._destroy_image_subscription()
        self.state = "ABORTED"
        self.error_reason = "stopped_by_control"
        self.capture_manager.reset_batch()
        self.publish_status()

    def _handle_reset(self, event_id: object) -> None:
        if event_id and not isinstance(event_id, str):
            raise ValueError("reset event_id must be a string")
        self.enabled = False
        if self.capture_mode and not self.subscribe_image_while_idle:
            self._destroy_image_subscription()
        self.used_event_ids.clear()
        self.event_id = ""
        self.error_reason = ""
        self.last_rejection_reason = ""
        self.stable_count = 0
        self.previous_bbox = None
        self.scheduler.reset()
        self.capture_manager.reset_batch()
        self.state = "PRELOADED_IDLE" if self.capture_mode else "SEARCH"
        self.publish_status()

    def publish_detection(
        self, detected: bool, score: float, box: Sequence[int], inference_ms: float
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
        self, image: np.ndarray, boxes: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        result = image.copy()
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = [int(round(value)) for value in box]
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                result,
                f"person_board {float(score):.3f}",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return result

    def publish_result(self, result: np.ndarray, source_msg: Image) -> None:
        if not self.publish_result_image:
            return
        success, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if success:
            output = CompressedImage()
            output.header = source_msg.header
            output.format = "jpeg"
            output.data = encoded.tobytes()
            self.result_publisher.publish(output)

    def save_debug_result(self, result: np.ndarray, detected: bool) -> None:
        if not self.save_debug_image or not detected:
            return
        now = time.monotonic()
        if now - self.last_debug_save_time < 1.0:
            return
        path = Path(self.debug_image_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), result):
            raise OSError(f"failed to save debug image: {path}")
        self.last_debug_save_time = now

    def _run_inference(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        original_height, original_width = image.shape[:2]
        padded, scale, pad_left, pad_top = letterbox(image, INPUT_WIDTH, INPUT_HEIGHT)
        nv12 = bgr_to_nv12(padded)
        inference_start = time.perf_counter()
        outputs = self.model.forward(nv12)
        inference_ms = (time.perf_counter() - inference_start) * 1000.0
        boxes, scores = decode_outputs(outputs, self.confidence_threshold)
        boxes, scores = apply_nms(
            boxes,
            scores,
            self.confidence_threshold,
            self.nms_threshold,
            self.max_detections,
        )
        boxes = restore_boxes(
            boxes, scale, pad_left, pad_top, original_width, original_height
        )
        return boxes, scores, inference_ms

    def image_callback(self, msg: Image) -> None:
        if self.capture_mode and not self.enabled:
            return
        self.frame_received_count += 1
        now = time.monotonic()
        if self.capture_mode and not self.scheduler.should_infer(now):
            return
        if self.state == "CAPTURE_BURST" and not self.capture_manager.may_attempt(now):
            return
        total_start = time.perf_counter()
        try:
            image = ros_image_to_bgr(msg)
            boxes, scores, inference_ms = self._run_inference(image)
            self.inference_count += 1
            throttle_state = (
                "CAPTURE_BURST"
                if self.state == "CAPTURE_BURST"
                else self.scheduler.state
            )
            self.scheduler.mark_inference(throttle_state, now)
            selected = self._select_target(
                boxes, scores, image.shape[1], image.shape[0]
            )
            detected = selected is not None
            if selected is None:
                best_box: List[int] = []
                best_score = 0.0
            else:
                selected_box, best_score = selected
                best_box = list(selected_box)
            self.publish_detection(detected, best_score, best_box, inference_ms)
            result = self.draw_result(image, boxes, scores)
            self.publish_result(result, msg)
            self.save_debug_result(result, detected)
            self.frame_count += 1
            if self.capture_mode:
                self._update_capture_state(image, msg, selected, now)
                self.publish_status()
            total_ms = (time.perf_counter() - total_start) * 1000.0
            state_changed = detected != self.last_detected
            if state_changed or self.frame_count % int(self.log_every_n) == 0:
                self.get_logger().info(
                    f"frame={self.frame_count}, encoding={msg.encoding}, "
                    f"detected={detected}, score={best_score:.3f}, box={best_box}, "
                    f"infer={inference_ms:.2f} ms, total={total_ms:.2f} ms"
                )
            self.last_detected = detected
        except Exception as exc:
            self.error_reason = f"image_processing_failed: {type(exc).__name__}: {exc}"
            self.get_logger().error(self.error_reason)
            if self.state == "CAPTURE_BURST":
                self._fail_capture(self.error_reason)
            else:
                self.publish_status()

    def _select_target(
        self, boxes: np.ndarray, scores: np.ndarray, width: int, height: int
    ) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
        if not self.capture_mode:
            if boxes.shape[0] == 0:
                return None
            best_index = int(np.argmax(scores))
            return tuple(int(round(value)) for value in boxes[best_index]), float(
                scores[best_index]
            )
        return select_best_detection(
            boxes,
            scores,
            width,
            height,
            self.minimum_yolo_confidence,
            self.minimum_detection_area,
        )

    def _update_capture_state(
        self,
        image: np.ndarray,
        msg: Image,
        selected: Optional[Tuple[Tuple[int, int, int, int], float]],
        now: float,
    ) -> None:
        if selected is None:
            if self.state == "CAPTURE_BURST" and self.capture_manager.may_attempt(now):
                self.capture_manager.record_rejection("no_target", now)
                self.last_rejection_reason = "no_target"
                self._check_capture_failure(now)
            else:
                self.state = "TARGET_LOST"
                self.publish_status()
                self.scheduler.target_lost()
                self.state = "SEARCH"
                self.stable_count = 0
                self.previous_bbox = None
                self.selected_bbox = []
                self.selected_score = 0.0
                self.selected_short_side = 0.0
                self.selected_normalized_area = 0.0
            return
        box, score = selected
        height, width = image.shape[:2]
        self.selected_bbox = list(box)
        self.selected_score = score
        self.selected_short_side = short_side(box)
        self.selected_normalized_area = normalized_area(box, width, height)
        if self.state == "CAPTURE_BURST":
            self._capture_attempt(image, msg, box, score, now)
            return
        self.state = self.scheduler.update(self.selected_short_side)
        stable = (
            score >= self.minimum_yolo_confidence
            and self.selected_short_side >= self.capture_min_short_side_px
            and edge_touch_ratio(box, width, height) <= self.maximum_edge_touch_ratio
        )
        padded = padded_bbox(box, self.crop_padding_ratio, width, height)
        if (
            padded is None
            or padded[2] - padded[0] < self.minimum_crop_width
            or padded[3] - padded[1] < self.minimum_crop_height
        ):
            stable = False
        if stable and self.previous_bbox is not None:
            stable = (
                bbox_iou(self.previous_bbox, box) >= self.stable_bbox_iou
                and center_shift_ratio(self.previous_bbox, box)
                <= self.maximum_center_shift_ratio
            )
        self.stable_count = self.stable_count + 1 if stable else 0
        self.previous_bbox = box
        if self.stable_count >= self.stable_detection_count:
            request_id = self.capture_manager.begin(self.event_id, now)
            self.state = "CAPTURE_BURST"
            self.get_logger().info(f"Capture burst started: request_id={request_id}")

    def _capture_attempt(
        self,
        image: np.ndarray,
        msg: Image,
        box: Tuple[int, int, int, int],
        score: float,
        now: float,
    ) -> None:
        if not self.capture_manager.may_attempt(now):
            return
        height, width = image.shape[:2]
        reason = ""
        padded = padded_bbox(box, self.crop_padding_ratio, width, height)
        if score < self.minimum_yolo_confidence:
            reason = "confidence_below_minimum"
        elif short_side(box) < self.capture_min_short_side_px:
            reason = "short_side_below_minimum"
        elif padded is None:
            reason = "invalid_padded_bbox"
        if reason:
            self.capture_manager.record_rejection(reason, now)
            self.last_rejection_reason = reason
            self._check_capture_failure(now)
            return
        assert padded is not None
        x1, y1, x2, y2 = padded
        crop = image[y1:y2, x1:x2].copy()
        quality = evaluate_crop(
            crop,
            box,
            width,
            height,
            self.minimum_crop_width,
            self.minimum_crop_height,
            self.minimum_sharpness,
            self.maximum_edge_touch_ratio,
        )
        if not quality.accepted:
            self.capture_manager.record_rejection(quality.reason, now)
            self.last_rejection_reason = quality.reason
            self._check_capture_failure(now)
            return
        sec, nanosec = int(msg.header.stamp.sec), int(msg.header.stamp.nanosec)
        if sec == 0 and nanosec == 0:
            fallback = self.get_clock().now().to_msg()
            sec, nanosec = int(fallback.sec), int(fallback.nanosec)
            self.get_logger().warning(
                "Source timestamp was zero; using node receive time"
            )
        self.capture_manager.add_frame(
            CapturedFrame(crop, sec, nanosec, box, padded, score, quality.sharpness),
            now,
        )
        if self.publish_debug_crop:
            self._publish_crop(crop, msg)
        if self.capture_manager.complete:
            self._finish_capture()
        else:
            self._check_capture_failure(now)

    def _publish_crop(self, crop: np.ndarray, source_msg: Image) -> None:
        success, encoded = cv2.imencode(
            ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not success:
            raise OSError("debug crop JPEG encoding failed")
        output = CompressedImage()
        output.header = source_msg.header
        output.format = "jpeg"
        output.data = encoded.tobytes()
        self.crop_publisher.publish(output)

    def _check_capture_failure(self, now: float) -> None:
        if self.capture_manager.exhausted:
            self._fail_capture("capture_max_attempts_exceeded")
        elif self.capture_manager.timed_out(now):
            self._fail_capture("capture_timeout")

    def _finish_capture(self) -> None:
        try:
            batch = self.capture_manager.commit()
            self.state = "BATCH_READY"
            self.publish_status()
            batch_msg = String()
            batch_msg.data = json.dumps(batch, ensure_ascii=False)
            self.batch_publisher.publish(batch_msg)
            done_msg = String()
            done_msg.data = json.dumps(
                {
                    "success": True,
                    "event_id": self.event_id,
                    "request_id": self.capture_manager.request_id,
                    "reason": "capture_batch_ready",
                },
                ensure_ascii=False,
            )
            self.done_publisher.publish(done_msg)
            self.state = "DONE"
            self.enabled = False
            self._destroy_image_subscription()
            self.publish_status()
            if self.one_shot_mode and self.shutdown_after_capture:
                self.shutdown_timer = self.create_timer(
                    float(self.shutdown_publish_grace_sec), self._shutdown_once
                )
        except Exception as exc:
            self._fail_capture(f"batch_write_failed: {type(exc).__name__}: {exc}")

    def _fail_capture(self, reason: str) -> None:
        self.state = "CAPTURE_FAILED"
        self.enabled = False
        self.error_reason = reason
        self.last_rejection_reason = self.capture_manager.last_rejection_reason
        self._destroy_image_subscription()
        self.get_logger().error(reason)
        self.publish_status()

    def _shutdown_once(self) -> None:
        if self.shutdown_timer is not None:
            self.shutdown_timer.cancel()
            self.shutdown_timer = None
        if rclpy.ok():
            self.get_logger().info(
                "Capture publication grace period complete; shutting down"
            )
            rclpy.shutdown()

    def status_timer_callback(self) -> None:
        if self.state == "CAPTURE_BURST":
            self._check_capture_failure(time.monotonic())
        self.publish_status()

    def publish_status(self) -> None:
        active_states = (
            "SEARCH",
            "TRACK_FAR",
            "TRACK_MIDDLE",
            "TRACK_NEAR",
            "CAPTURE_BURST",
        )
        hz_state = (
            "CAPTURE_BURST" if self.state == "CAPTURE_BURST" else self.scheduler.state
        )
        current_hz = (
            self.scheduler.current_hz(hz_state) if self.state in active_states else 0.0
        )
        payload = {
            "state": self.state,
            "enabled": self.enabled,
            "event_id": self.event_id,
            "image_subscription_active": self.image_subscription is not None,
            "current_inference_hz": current_hz,
            "frame_received_count": self.frame_received_count,
            "inference_count": self.inference_count,
            "selected_bbox": self.selected_bbox,
            "short_side_px": self.selected_short_side,
            "smoothed_short_side_px": self.scheduler.smoothed_short_side,
            "normalized_area": self.selected_normalized_area,
            "yolo_confidence": self.selected_score,
            "stable_count": self.stable_count,
            "capture_count": len(self.capture_manager.frames),
            "capture_attempts": self.capture_manager.attempts,
            "request_id": self.capture_manager.request_id,
            "error_reason": self.error_reason,
            "last_rejection_reason": self.last_rejection_reason,
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_publisher.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = PersonBoardDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"person_board_detector fatal error: {type(exc).__name__}: {exc}")
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
