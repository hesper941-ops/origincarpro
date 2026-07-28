#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
黄色区域视觉寻路独立测试脚本。

本脚本不接入正式 task_manager，只订阅相机并向 cmd_vel_gate 的输入话题发布
速度。默认发布到 /cmd_vel_raw，仍需使用比赛启动按钮放行。

推荐测试流程
============

1. 启动底盘和相机，不启动正式导航：

    cd /root/intelligent_car_ws
    source /opt/tros/humble/setup.bash
    source install/setup.bash

    ros2 launch origincar_bringup competition_bringup.launch.py \
      enable_base:=true \
      enable_camera:=true \
      enable_yolo_avoid:=false \
      enable_qr:=false \
      enable_birdview:=false \
      enable_nav:=false \
      enable_cobridge:=false \
      button_backend:=topic

如果当前 launch 不支持 enable_nav，则删除 enable_nav 参数。

2. 运行黄色区域寻路：

    python3 tools/yellow_area_follow.py \
      --ros-args \
      --params-file tools/yellow_area_params.yaml

只看可视化、不让车运动时：

    python3 tools/yellow_area_follow.py \
      --ros-args \
      --params-file tools/yellow_area_params.yaml \
      -p publish_debug_image:=true \
      -p dry_run:=true

此时可在 CoStudio / Foxglove / RDK 面板中查看
/yellow_area/debug_image（bgr8）和 /yellow_area/debug_mask（mono8）。

3. 发车门锁：

    ros2 topic pub --once /competition/start_button \
      std_msgs/msg/Bool "{data: true}"

4. 急停：

    ros2 topic pub --once /competition/emergency_stop_cmd \
      std_msgs/msg/Bool "{data: true}"

5. 查看速度链路：

    timeout 3s ros2 topic echo /cmd_vel_raw || true
    timeout 3s ros2 topic echo /cmd_vel || true

Ctrl+C 退出时，本脚本会连续发布多次零速。首次测试请架空车轮或确保急停可用。
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


MORPHOLOGY_KERNEL_SIZE = 5
STOP_PUBLISH_COUNT = 5
STOP_PUBLISH_INTERVAL_SEC = 0.05


@dataclass
class BoundaryBandResult:
    """Yellow/green boundary geometry for one horizontal ROI band."""

    y0: int
    y1: int
    center_y: int
    valid: bool
    left: Optional[float] = None
    right: Optional[float] = None
    center: Optional[float] = None
    left_virtual: bool = False
    right_virtual: bool = False
    confidence: float = 0.0


@dataclass
class VisionResult:
    """Boundary path, masks, look-ahead targets, and visibility metrics."""

    valid: bool
    source: str
    yellow_pixels: int
    green_pixels: int
    yellow_mask: np.ndarray
    green_mask: np.ndarray
    boundary_mask: np.ndarray
    bands: List[BoundaryBandResult]
    path_points: List[Tuple[float, float]]
    valid_center_count: int
    left_visible_count: int
    right_visible_count: int
    both_visible_count: int
    near_center_x: Optional[float]
    far_center_x: Optional[float]
    near_y: Optional[float]
    far_y: Optional[float]
    near_error: float
    far_error: float
    control_error: float
    fallback_centroid: Optional[Tuple[float, float]]


def image_message_to_bgr(msg: Image) -> np.ndarray:
    """Convert bgr8/rgb8/mono8/nv12 sensor_msgs/Image data to packed BGR."""
    encoding = msg.encoding.strip().lower()
    width = int(msg.width)
    height = int(msg.height)
    step = int(msg.step)
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {width}x{height}")

    raw = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding in ("bgr8", "rgb8"):
        row_bytes = width * 3
        step = step or row_bytes
        required = step * height
        if step < row_bytes or raw.size < required:
            raise ValueError(
                f"invalid {encoding} buffer: step={step}, bytes={raw.size}"
            )
        packed = raw[:required].reshape(height, step)[:, :row_bytes]
        image = packed.reshape(height, width, 3)
        if encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(image)

    if encoding == "mono8":
        row_bytes = width
        step = step or row_bytes
        required = step * height
        if step < row_bytes or raw.size < required:
            raise ValueError(
                f"invalid mono8 buffer: step={step}, bytes={raw.size}"
            )
        mono = raw[:required].reshape(height, step)[:, :row_bytes]
        return cv2.cvtColor(np.ascontiguousarray(mono), cv2.COLOR_GRAY2BGR)

    if encoding == "nv12":
        if width % 2 or height % 2:
            raise ValueError(f"NV12 image size must be even: {width}x{height}")
        row_bytes = width
        step = step or row_bytes
        yuv_rows = height + height // 2
        required = step * yuv_rows
        if step < row_bytes or raw.size < required:
            raise ValueError(
                f"invalid nv12 buffer: step={step}, expected at least "
                f"{required} bytes, got {raw.size}"
            )
        yuv = raw[:required].reshape(yuv_rows, step)[:, :row_bytes]
        return cv2.cvtColor(
            np.ascontiguousarray(yuv), cv2.COLOR_YUV2BGR_NV12
        )

    raise ValueError(f"unsupported image encoding: {msg.encoding!r}")


def _local_transition_candidates(
    yellow_score: np.ndarray,
    green_score: np.ndarray,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Find robust green->yellow and yellow->green column transitions."""
    width = int(yellow_score.size)
    smooth_kernel = np.ones(5, dtype=np.float32) / 5.0
    yellow = np.convolve(yellow_score, smooth_kernel, mode="same")
    green = np.convolve(green_score, smooth_kernel, mode="same")
    span = 6
    left_metric = np.zeros(width, dtype=np.float32)
    right_metric = np.zeros(width, dtype=np.float32)
    indices = np.arange(span, width - span)
    yellow_sum = np.concatenate(([0.0], np.cumsum(yellow)))
    green_sum = np.concatenate(([0.0], np.cumsum(green)))
    yellow_left = (
        yellow_sum[indices] - yellow_sum[indices - span]
    ) / span
    yellow_right = (
        yellow_sum[indices + span] - yellow_sum[indices]
    ) / span
    green_left = (
        green_sum[indices] - green_sum[indices - span]
    ) / span
    green_right = (
        green_sum[indices + span] - green_sum[indices]
    ) / span

    left_valid = (green_left >= 0.12) & (yellow_right >= 0.12)
    left_contrast = np.maximum(
        0.0,
        (green_left - green_right) + (yellow_right - yellow_left),
    )
    left_values = (
        np.minimum(green_left, yellow_right) + 0.5 * left_contrast
    )
    left_metric[indices[left_valid]] = left_values[left_valid]

    right_valid = (yellow_left >= 0.12) & (green_right >= 0.12)
    right_contrast = np.maximum(
        0.0,
        (yellow_left - yellow_right) + (green_right - green_left),
    )
    right_values = (
        np.minimum(yellow_left, green_right) + 0.5 * right_contrast
    )
    right_metric[indices[right_valid]] = right_values[right_valid]

    def select_peaks(metric: np.ndarray) -> List[Tuple[float, float]]:
        order = np.argsort(metric)[::-1]
        selected: List[Tuple[float, float]] = []
        for index in order:
            strength = float(metric[index])
            if strength < 0.18:
                break
            if all(abs(float(index) - point[0]) >= 10.0 for point in selected):
                selected.append((float(index), strength))
            if len(selected) >= 12:
                break
        return selected

    return select_peaks(left_metric), select_peaks(right_metric)


def _select_boundary_pair(
    left_candidates: List[Tuple[float, float]],
    right_candidates: List[Tuple[float, float]],
    previous_left: Optional[float],
    previous_right: Optional[float],
    predicted_center: float,
    min_boundary_gap_px: float,
    max_boundary_jump_px: float,
    expected_lane_width_px: float,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Choose a plausible, continuous left/right boundary pair."""
    best_pair = None
    best_score = -float("inf")
    for left in left_candidates:
        if (
            previous_left is not None
            and abs(left[0] - previous_left) > max_boundary_jump_px
        ):
            continue
        for right in right_candidates:
            if (
                previous_right is not None
                and abs(right[0] - previous_right) > max_boundary_jump_px
            ):
                continue
            lane_width = right[0] - left[0]
            if lane_width < min_boundary_gap_px:
                continue
            center = (left[0] + right[0]) * 0.5
            width_penalty = (
                abs(lane_width - expected_lane_width_px)
                / max(expected_lane_width_px, 1.0)
            )
            center_penalty = (
                abs(center - predicted_center)
                / max(max_boundary_jump_px, 1.0)
            )
            score = (
                left[1] + right[1]
                - 0.30 * width_penalty
                - 0.12 * center_penalty
            )
            if score > best_score:
                best_score = score
                best_pair = (left, right)
    return best_pair


def _select_single_boundary(
    candidates: List[Tuple[float, float]],
    previous: Optional[float],
    expected: float,
    max_boundary_jump_px: float,
) -> Optional[Tuple[float, float]]:
    """Choose one continuous boundary when its opposite side is unavailable."""
    best = None
    best_score = -float("inf")
    for candidate in candidates:
        if (
            previous is not None
            and abs(candidate[0] - previous) > max_boundary_jump_px
        ):
            continue
        distance_penalty = (
            abs(candidate[0] - expected)
            / max(max_boundary_jump_px, 1.0)
        )
        score = candidate[1] - 0.12 * distance_penalty
        if score > best_score:
            best_score = score
            best = candidate
    return best


def _fit_center_path(
    center_points: List[Tuple[float, float, float]],
    roi_height: int,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Fit a robust weighted polynomial x(y) and return drawable points."""
    x_values = np.asarray([point[0] for point in center_points], dtype=float)
    y_values = np.asarray([point[1] for point in center_points], dtype=float)
    confidence = np.asarray([point[2] for point in center_points], dtype=float)
    near_weight = 0.55 + 0.45 * y_values / max(roi_height - 1, 1)
    weights = np.sqrt(np.clip(confidence * near_weight, 0.05, None))
    degree = 2 if len(center_points) >= 4 else min(1, len(center_points) - 1)
    coefficients = np.polyfit(y_values, x_values, degree, w=weights)

    if len(center_points) >= 5:
        residuals = np.abs(np.polyval(coefficients, y_values) - x_values)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        threshold = max(18.0, median + 2.5 * max(mad, 1.0))
        keep = residuals <= threshold
        if int(np.count_nonzero(keep)) >= max(degree + 1, 3):
            coefficients = np.polyfit(
                y_values[keep],
                x_values[keep],
                degree,
                w=weights[keep],
            )

    sorted_y = np.sort(y_values)
    path_points = [
        (float(np.polyval(coefficients, y)), float(y)) for y in sorted_y
    ]
    return coefficients, path_points


def _largest_yellow_centroid(
    yellow_mask: np.ndarray,
) -> Optional[Tuple[float, float]]:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        yellow_mask, connectivity=8
    )
    if component_count <= 1:
        return None
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    moments = cv2.moments(component, binaryImage=True)
    if moments["m00"] <= 0.0:
        return None
    return (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )


def extract_boundary_path(
    bgr: np.ndarray,
    yellow_lower: Tuple[int, int, int],
    yellow_upper: Tuple[int, int, int],
    green_lower: Tuple[int, int, int],
    green_upper: Tuple[int, int, int],
    roi_y_start_ratio: float,
    roi_y_end_ratio: float,
    boundary_scan_count: int,
    boundary_band_half_height: int,
    min_boundary_gap_px: float,
    max_boundary_jump_px: float,
    expected_lane_width_px: float,
    use_virtual_boundary: bool,
    min_valid_bands: int,
    min_yellow_pixels: int,
    near_y_ratio: float,
    far_y_ratio: float,
    head_gain: float,
) -> VisionResult:
    """Extract yellow/green boundaries and fit a look-ahead center path."""
    image_height, image_width = bgr.shape[:2]
    roi_y_start = int(image_height * roi_y_start_ratio)
    roi_y_start = min(max(roi_y_start, 0), image_height - 1)
    roi_y_end = int(image_height * roi_y_end_ratio)
    roi_y_end = min(max(roi_y_end, roi_y_start + 1), image_height)
    roi = bgr[roi_y_start:roi_y_end, :]
    roi_height = roi.shape[0]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(
        hsv,
        np.asarray(yellow_lower, dtype=np.uint8),
        np.asarray(yellow_upper, dtype=np.uint8),
    )
    green_mask = cv2.inRange(
        hsv,
        np.asarray(green_lower, dtype=np.uint8),
        np.asarray(green_upper, dtype=np.uint8),
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE),
    )
    for mask in (yellow_mask, green_mask):
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, dst=mask)
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, dst=mask)

    yellow_pixels = int(cv2.countNonZero(yellow_mask))
    green_pixels = int(cv2.countNonZero(green_mask))
    yellow_edge = cv2.morphologyEx(
        yellow_mask, cv2.MORPH_GRADIENT, kernel
    )
    green_edge = cv2.morphologyEx(green_mask, cv2.MORPH_GRADIENT, kernel)
    boundary_mask = cv2.bitwise_and(
        cv2.dilate(yellow_edge, kernel),
        cv2.dilate(green_edge, kernel),
    )

    scan_ratios = np.linspace(0.08, 0.92, boundary_scan_count)
    half_height = max(1, int(boundary_band_half_height))
    bands: List[BoundaryBandResult] = []
    center_points: List[Tuple[float, float, float]] = []
    previous_left: Optional[float] = None
    previous_right: Optional[float] = None
    previous_center = image_width * 0.5
    left_visible_count = 0
    right_visible_count = 0
    both_visible_count = 0

    for ratio in scan_ratios:
        center_y = int(round((roi_height - 1) * float(ratio)))
        y0 = max(0, center_y - half_height)
        y1 = min(roi_height, center_y + half_height + 1)
        yellow_score = np.mean(yellow_mask[y0:y1, :] > 0, axis=0)
        green_score = np.mean(green_mask[y0:y1, :] > 0, axis=0)
        left_candidates, right_candidates = _local_transition_candidates(
            yellow_score.astype(np.float32),
            green_score.astype(np.float32),
        )
        pair = _select_boundary_pair(
            left_candidates,
            right_candidates,
            previous_left,
            previous_right,
            previous_center,
            min_boundary_gap_px,
            max_boundary_jump_px,
            expected_lane_width_px,
        )

        left_actual = None
        right_actual = None
        left_strength = 0.0
        right_strength = 0.0
        if pair is not None:
            left_actual, left_strength = pair[0]
            right_actual, right_strength = pair[1]
        else:
            left_choice = _select_single_boundary(
                left_candidates,
                previous_left,
                previous_center - expected_lane_width_px * 0.5,
                max_boundary_jump_px,
            )
            right_choice = _select_single_boundary(
                right_candidates,
                previous_right,
                previous_center + expected_lane_width_px * 0.5,
                max_boundary_jump_px,
            )
            if left_choice is not None:
                left_actual, left_strength = left_choice
            if right_choice is not None:
                right_actual, right_strength = right_choice
            if (
                left_actual is not None
                and right_actual is not None
                and right_actual - left_actual < min_boundary_gap_px
            ):
                if left_strength >= right_strength:
                    right_actual = None
                else:
                    left_actual = None

        left_visible = left_actual is not None
        right_visible = right_actual is not None
        left_visible_count += int(left_visible)
        right_visible_count += int(right_visible)
        both_visible_count += int(left_visible and right_visible)

        left = left_actual
        right = right_actual
        left_virtual = False
        right_virtual = False
        confidence = 0.0
        if left_visible and right_visible:
            confidence = 1.0
        elif use_virtual_boundary and left_visible:
            right = min(
                image_width - 1.0, left_actual + expected_lane_width_px
            )
            right_virtual = True
            confidence = 0.55
        elif use_virtual_boundary and right_visible:
            left = max(0.0, right_actual - expected_lane_width_px)
            left_virtual = True
            confidence = 0.55

        valid = left is not None and right is not None
        center = (left + right) * 0.5 if valid else None
        bands.append(
            BoundaryBandResult(
                y0,
                y1,
                center_y,
                valid,
                left,
                right,
                center,
                left_virtual,
                right_virtual,
                confidence,
            )
        )
        if valid and center is not None:
            center_points.append((center, float(center_y), confidence))
            previous_left = left
            previous_right = right
            previous_center = center

    valid_center_count = len(center_points)
    image_center = image_width * 0.5
    near_y = float((roi_height - 1) * near_y_ratio)
    far_y = float((roi_height - 1) * far_y_ratio)

    if valid_center_count >= min_valid_bands:
        coefficients, path_points = _fit_center_path(
            center_points, roi_height
        )
        near_center_x = float(
            np.clip(
                np.polyval(coefficients, near_y),
                0.0,
                image_width - 1.0,
            )
        )
        far_center_x = float(
            np.clip(
                np.polyval(coefficients, far_y),
                0.0,
                image_width - 1.0,
            )
        )
        near_error = (near_center_x - image_center) / image_center
        far_error = (far_center_x - image_center) / image_center
        control_error = near_error + head_gain * (far_error - near_error)
        return VisionResult(
            True,
            "boundary_path",
            yellow_pixels,
            green_pixels,
            yellow_mask,
            green_mask,
            boundary_mask,
            bands,
            path_points,
            valid_center_count,
            left_visible_count,
            right_visible_count,
            both_visible_count,
            near_center_x,
            far_center_x,
            near_y,
            far_y,
            near_error,
            far_error,
            control_error,
            None,
        )

    fallback_centroid = None
    if yellow_pixels >= min_yellow_pixels:
        fallback_centroid = _largest_yellow_centroid(yellow_mask)
    if fallback_centroid is not None:
        fallback_x, fallback_y = fallback_centroid
        fallback_error = (fallback_x - image_center) / image_center
        return VisionResult(
            True,
            "yellow_area_center",
            yellow_pixels,
            green_pixels,
            yellow_mask,
            green_mask,
            boundary_mask,
            bands,
            [(fallback_x, fallback_y)],
            valid_center_count,
            left_visible_count,
            right_visible_count,
            both_visible_count,
            fallback_x,
            fallback_x,
            fallback_y,
            fallback_y,
            fallback_error,
            fallback_error,
            fallback_error,
            fallback_centroid,
        )

    return VisionResult(
        False,
        "lost",
        yellow_pixels,
        green_pixels,
        yellow_mask,
        green_mask,
        boundary_mask,
        bands,
        [],
        valid_center_count,
        left_visible_count,
        right_visible_count,
        both_visible_count,
        None,
        None,
        None,
        None,
        0.0,
        0.0,
        0.0,
        None,
    )


def compute_control(
    raw_error: float,
    previous_error: Optional[float],
    previous_angular: Optional[float],
    elapsed_sec: float,
    smoothing_alpha: float,
    angular_smoothing_alpha: float,
    error_deadband: float,
    linear_speed: float,
    min_linear_speed: float,
    speed_scale: float,
    kp: float,
    kd: float,
    max_angular: float,
    max_angular_delta_per_sec: float,
) -> Tuple[float, float, float]:
    """Calculate smoothed PD steering with deadband and angular slew limiting."""
    elapsed_sec = float(np.clip(elapsed_sec, 0.02, 0.20))
    if previous_error is None:
        filtered_error = raw_error
        derivative = 0.0
    else:
        filtered_error = (
            smoothing_alpha * raw_error
            + (1.0 - smoothing_alpha) * previous_error
        )
        derivative = (filtered_error - previous_error) / elapsed_sec

    control_error = (
        0.0 if abs(filtered_error) <= error_deadband else filtered_error
    )
    target_angular = -kp * control_error - kd * derivative
    target_angular = float(
        np.clip(target_angular, -max_angular, max_angular)
    )
    if previous_angular is None:
        smoothed_angular = target_angular
    else:
        smoothed_angular = (
            angular_smoothing_alpha * target_angular
            + (1.0 - angular_smoothing_alpha) * previous_angular
        )
        max_delta = max_angular_delta_per_sec * elapsed_sec
        smoothed_angular = float(
            np.clip(
                smoothed_angular,
                previous_angular - max_delta,
                previous_angular + max_delta,
            )
        )
    angular = float(
        np.clip(smoothed_angular, -max_angular, max_angular)
    )

    error_magnitude = min(abs(filtered_error), 1.0)
    adaptive_speed = (
        linear_speed
        - (linear_speed - min_linear_speed) * error_magnitude
    )
    linear = float(
        np.clip(adaptive_speed * speed_scale, 0.0, linear_speed)
    )
    return filtered_error, linear, angular


def numpy_to_image_message(
    image: np.ndarray,
    encoding: str,
    source_msg: Image,
) -> Image:
    """Pack a contiguous uint8 numpy image without using cv_bridge."""
    packed = np.ascontiguousarray(image, dtype=np.uint8)
    output = Image()
    output.header = source_msg.header
    output.height = int(packed.shape[0])
    output.width = int(packed.shape[1])
    output.encoding = encoding
    output.is_bigendian = 0
    channels = 1 if packed.ndim == 2 else int(packed.shape[2])
    output.step = output.width * channels
    output.data = packed.tobytes()
    return output


def render_debug_image(
    bgr: np.ndarray,
    result: VisionResult,
    roi_y_start_ratio: float,
    roi_y_end_ratio: float,
    linear: float,
    angular: float,
    lost_frames: int,
    single_boundary_frames: int,
    kp: float,
    kd: float,
    head_gain: float,
    dry_run: bool,
) -> np.ndarray:
    """Draw V3 masks, boundary points, center path, and controller state."""
    debug = bgr.copy()
    image_height, image_width = debug.shape[:2]
    roi_y_start = int(image_height * roi_y_start_ratio)
    roi_y_start = min(max(roi_y_start, 0), image_height - 1)
    roi_y_end = int(image_height * roi_y_end_ratio)
    roi_y_end = min(max(roi_y_end, roi_y_start + 1), image_height)
    roi_y_end_line = min(roi_y_end, image_height - 1)

    # BGR colors chosen to remain distinct from the yellow road.
    blue = (255, 0, 0)
    red = (0, 0, 255)
    cyan = (255, 255, 0)
    dim_cyan = (128, 128, 0)
    green = (0, 255, 0)
    purple = (255, 0, 255)
    white = (255, 255, 255)
    orange = (0, 165, 255)

    # Semi-transparent yellow road and green-region contours.
    yellow_overlay = debug.copy()
    overlay_roi = yellow_overlay[roi_y_start:roi_y_end, :]
    overlay_roi[result.yellow_mask > 0] = (0, 255, 255)
    cv2.addWeighted(yellow_overlay, 0.25, debug, 0.75, 0.0, debug)
    green_contours, _ = cv2.findContours(
        result.green_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    debug_roi = debug[roi_y_start:roi_y_end, :]
    cv2.drawContours(debug_roi, green_contours, -1, (0, 180, 0), 1)

    # ROI boundary and scan-band geometry.
    cv2.line(
        debug,
        (0, roi_y_start),
        (image_width - 1, roi_y_start),
        cyan,
        2,
    )
    cv2.line(
        debug,
        (0, roi_y_end_line),
        (image_width - 1, roi_y_end_line),
        cyan,
        2,
    )
    for band in result.bands:
        y0 = roi_y_start + band.y0
        y1 = roi_y_start + band.y1 - 1
        color = cyan if band.valid else dim_cyan
        cv2.rectangle(
            debug,
            (0, y0),
            (image_width - 1, y1),
            color,
            1,
        )
        if (
            band.valid
            and band.left is not None
            and band.right is not None
            and band.center is not None
        ):
            point_y = roi_y_start + band.center_y
            left_point = (int(round(band.left)), point_y)
            right_point = (int(round(band.right)), point_y)
            center_point = (int(round(band.center)), point_y)
            for point, is_virtual in (
                (left_point, band.left_virtual),
                (right_point, band.right_virtual),
            ):
                if is_virtual:
                    cv2.circle(debug, point, 6, orange, 2)
                else:
                    cv2.circle(debug, point, 6, (0, 0, 0), -1)
                    cv2.circle(debug, point, 4, green, -1)
            cv2.circle(debug, center_point, 6, (0, 0, 0), -1)
            cv2.circle(debug, center_point, 4, white, -1)

    # Vehicle center, fitted center trajectory, and near/far look-ahead points.
    image_center_x = image_width // 2
    cv2.line(
        debug,
        (image_center_x, roi_y_start),
        (image_center_x, image_height - 1),
        blue,
        2,
    )
    if len(result.path_points) >= 2:
        path = np.asarray(
            [
                (
                    int(np.clip(round(x), 0, image_width - 1)),
                    int(
                        np.clip(
                            round(roi_y_start + y),
                            roi_y_start,
                            roi_y_end - 1,
                        )
                    ),
                )
                for x, y in result.path_points
            ],
            dtype=np.int32,
        )
        cv2.polylines(debug, [path], False, purple, 3, cv2.LINE_AA)

    if (
        result.near_center_x is not None
        and result.near_y is not None
    ):
        near_point = (
            int(np.clip(round(result.near_center_x), 0, image_width - 1)),
            int(
                np.clip(
                    round(roi_y_start + result.near_y),
                    roi_y_start,
                    roi_y_end - 1,
                )
            ),
        )
        cv2.circle(debug, near_point, 8, red, -1)
        cv2.arrowedLine(
            debug,
            (image_center_x, image_height - 1),
            near_point,
            white,
            3,
            cv2.LINE_AA,
            tipLength=0.08,
        )
        cv2.putText(
            debug,
            "near",
            (near_point[0] + 8, near_point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            red,
            1,
            cv2.LINE_AA,
        )

    if result.far_center_x is not None and result.far_y is not None:
        far_point = (
            int(np.clip(round(result.far_center_x), 0, image_width - 1)),
            int(
                np.clip(
                    round(roi_y_start + result.far_y),
                    roi_y_start,
                    roi_y_end - 1,
                )
            ),
        )
        cv2.circle(debug, far_point, 7, blue, -1)
        cv2.putText(
            debug,
            "far",
            (far_point[0] + 8, far_point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            blue,
            1,
            cv2.LINE_AA,
        )

    if result.source == "yellow_area_center" and result.fallback_centroid:
        fallback_x, fallback_y = result.fallback_centroid
        point = (
            int(np.clip(round(fallback_x), 0, image_width - 1)),
            int(
                np.clip(
                    round(roi_y_start + fallback_y),
                    roi_y_start,
                    image_height - 1,
                )
            ),
        )
        cv2.drawMarker(
            debug,
            point,
            purple,
            cv2.MARKER_CROSS,
            24,
            3,
        )

    status_lines = [
        f"yellow_pixels={result.yellow_pixels}",
        f"green_pixels={result.green_pixels}",
        f"valid_center_count={result.valid_center_count}",
        f"left_visible_count={result.left_visible_count}",
        f"right_visible_count={result.right_visible_count}",
        f"both_visible_count={result.both_visible_count}",
        f"near_error={result.near_error:.3f}",
        f"far_error={result.far_error:.3f}",
        f"control_error={result.control_error:.3f}",
        f"linear.x={linear:.3f}",
        f"angular.z={angular:.3f}",
        f"lost_frames={lost_frames}",
        f"single_boundary_frames={single_boundary_frames}",
        f"kp={kp:.3f}",
        f"kd={kd:.3f}",
        f"head_gain={head_gain:.3f}",
    ]
    if result.source == "yellow_area_center":
        status_lines.append("fallback=yellow_area_center")
    if dry_run:
        status_lines.append("DRY RUN / ZERO CMD")

    line_height = 20
    lost_title_height = 28 if result.source == "lost" else 0
    panel_height = min(
        image_height,
        12 + lost_title_height + line_height * len(status_lines),
    )
    panel_width = min(image_width, 360)
    overlay = debug.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (panel_width - 1, panel_height - 1),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.58, debug, 0.42, 0.0, debug)

    text_y = 20
    if result.source == "lost":
        cv2.putText(
            debug,
            "BOUNDARIES LOST / STOP",
            (8, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            red,
            2,
            cv2.LINE_AA,
        )
        text_y += lost_title_height
    for line in status_lines:
        if text_y >= image_height:
            break
        color = cyan if line.startswith("fallback=") else white
        if line.startswith("DRY RUN"):
            color = purple
        cv2.putText(
            debug,
            line,
            (8, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
        text_y += line_height

    return debug


class YellowAreaFollower(Node):
    """ROS2 node for standalone yellow-area following tests."""

    def __init__(self) -> None:
        super().__init__("yellow_area_follower")
        defaults = {
            "image_topic": "/image_out",
            "cmd_topic": "/cmd_vel_raw",
            "h_min": 15,
            "h_max": 40,
            "s_min": 45,
            "s_max": 255,
            "v_min": 80,
            "v_max": 255,
            "green_h_min": 35,
            "green_h_max": 95,
            "green_s_min": 20,
            "green_s_max": 255,
            "green_v_min": 30,
            "green_v_max": 255,
            "roi_y_start_ratio": 0.25,
            "roi_y_end_ratio": 0.85,
            "scan_band_half_height": 8,
            "min_yellow_pixels": 800,
            "min_band_pixels": 40,
            "min_valid_bands": 2,
            "boundary_scan_count": 10,
            "boundary_band_half_height": 6,
            "min_boundary_gap_px": 40,
            "max_boundary_jump_px": 80,
            "expected_lane_width_px": 260,
            "use_virtual_boundary": True,
            "near_y_ratio": 0.75,
            "far_y_ratio": 0.45,
            "head_gain": 0.35,
            "boundary_loss_slowdown": 0.5,
            "min_left_right_visible_rows": 2,
            "single_boundary_max_frames": 10,
            "linear_speed": 0.04,
            "min_linear_speed": 0.02,
            "kp": 0.22,
            "kd": 0.12,
            "max_angular": 0.25,
            "smoothing_alpha": 0.20,
            "error_deadband": 0.02,
            "angular_smoothing_alpha": 0.35,
            "max_angular_delta_per_sec": 0.60,
            "lost_stop_frames": 5,
            "search_on_lost": False,
            "search_angular": 0.25,
            "publish_debug_image": False,
            "debug_image_topic": "/yellow_area/debug_image",
            "debug_mask_topic": "/yellow_area/debug_mask",
            "publish_boundary_debug_mask": False,
            "boundary_debug_mask_topic": "/yellow_area/boundary_mask",
            "debug_rate_hz": 10.0,
            "control_only_when_started": False,
            "dry_run": False,
            "log_interval_sec": 0.5,
            "image_timeout_sec": 1.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.params = {
            name: self.get_parameter(name).value for name in defaults
        }
        self._validate_parameters()

        self.image_topic = str(self.params["image_topic"])
        self.cmd_topic = str(self.params["cmd_topic"])
        self.cmd_publisher = self.create_publisher(
            Twist, self.cmd_topic, 10
        )
        self.create_subscription(
            Image,
            self.image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )

        self.debug_image_publisher = None
        self.debug_mask_publisher = None
        self.boundary_debug_mask_publisher = None
        if bool(self.params["publish_debug_image"]):
            self.debug_image_publisher = self.create_publisher(
                Image, str(self.params["debug_image_topic"]), 1
            )
            self.debug_mask_publisher = self.create_publisher(
                Image, str(self.params["debug_mask_topic"]), 1
            )
        if bool(self.params["publish_boundary_debug_mask"]):
            self.boundary_debug_mask_publisher = self.create_publisher(
                Image,
                str(self.params["boundary_debug_mask_topic"]),
                1,
            )

        self.previous_error: Optional[float] = None
        self.previous_angular: Optional[float] = 0.0
        self.last_control_time = time.monotonic()
        self.last_turn_direction = 1.0
        self.lost_frames = 0
        self.single_boundary_frames = 0
        self.last_image_time = time.monotonic()
        self.last_log_time = 0.0
        self.last_error_log_time = 0.0
        self.last_debug_time = 0.0
        self.watchdog_stopped = False
        self.create_timer(0.1, self._watchdog_callback)
        self._log_configuration()

    def _validate_parameters(self) -> None:
        p = self.params
        if not str(p["image_topic"]) or not str(p["cmd_topic"]):
            raise ValueError("image_topic and cmd_topic must not be empty")
        if not (
            0 <= int(p["h_min"]) <= int(p["h_max"]) <= 179
            and 0 <= int(p["s_min"]) <= int(p["s_max"]) <= 255
            and 0 <= int(p["v_min"]) <= int(p["v_max"]) <= 255
        ):
            raise ValueError("invalid HSV threshold ranges")
        if not (
            0 <= int(p["green_h_min"]) <= int(p["green_h_max"]) <= 179
            and 0 <= int(p["green_s_min"]) <= int(p["green_s_max"]) <= 255
            and 0 <= int(p["green_v_min"]) <= int(p["green_v_max"]) <= 255
        ):
            raise ValueError("invalid green HSV threshold ranges")
        roi_y_start_ratio = float(p["roi_y_start_ratio"])
        roi_y_end_ratio = float(p["roi_y_end_ratio"])
        if not 0.0 <= roi_y_start_ratio <= 0.95:
            raise ValueError("roi_y_start_ratio must be in [0.0, 0.95]")
        if not 0.05 <= roi_y_end_ratio <= 1.0:
            raise ValueError("roi_y_end_ratio must be in [0.05, 1.0]")
        if roi_y_end_ratio <= roi_y_start_ratio + 0.10:
            corrected_end_ratio = min(0.95, roi_y_start_ratio + 0.25)
            self.get_logger().warning(
                "roi_y_end_ratio is too close to roi_y_start_ratio; "
                f"correcting {roi_y_end_ratio:.3f} to "
                f"{corrected_end_ratio:.3f}"
            )
            if corrected_end_ratio <= roi_y_start_ratio:
                raise ValueError(
                    "unable to create a non-empty ROI from the supplied ratios"
                )
            p["roi_y_end_ratio"] = corrected_end_ratio
        for name in (
            "scan_band_half_height",
            "min_yellow_pixels",
            "min_band_pixels",
            "min_valid_bands",
            "boundary_scan_count",
            "boundary_band_half_height",
            "min_boundary_gap_px",
            "max_boundary_jump_px",
            "expected_lane_width_px",
            "min_left_right_visible_rows",
            "single_boundary_max_frames",
            "lost_stop_frames",
        ):
            if int(p[name]) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        linear_speed = float(p["linear_speed"])
        min_linear_speed = float(p["min_linear_speed"])
        if linear_speed < 0.0 or not 0.0 <= min_linear_speed <= linear_speed:
            raise ValueError(
                "speeds must satisfy 0 <= min_linear_speed <= linear_speed"
            )
        for name in (
            "kp",
            "kd",
            "max_angular",
            "search_angular",
            "head_gain",
            "max_angular_delta_per_sec",
        ):
            if float(p[name]) < 0.0:
                raise ValueError(f"{name} must not be negative")
        for name in ("smoothing_alpha", "angular_smoothing_alpha"):
            if not 0.0 < float(p[name]) <= 1.0:
                raise ValueError(f"{name} must be in (0.0, 1.0]")
        if not 0.0 <= float(p["error_deadband"]) < 1.0:
            raise ValueError("error_deadband must be in [0.0, 1.0)")
        if not (
            0.0 <= float(p["far_y_ratio"]) <= 1.0
            and 0.0 <= float(p["near_y_ratio"]) <= 1.0
            and float(p["far_y_ratio"]) < float(p["near_y_ratio"])
        ):
            raise ValueError(
                "look-ahead ratios must satisfy 0 <= far < near <= 1"
            )
        if not 0.0 <= float(p["boundary_loss_slowdown"]) <= 1.0:
            raise ValueError("boundary_loss_slowdown must be in [0.0, 1.0]")
        if int(p["min_valid_bands"]) > int(p["boundary_scan_count"]):
            raise ValueError(
                "min_valid_bands must not exceed boundary_scan_count"
            )
        if float(p["max_angular_delta_per_sec"]) <= 0.0:
            raise ValueError(
                "max_angular_delta_per_sec must be greater than zero"
            )
        if float(p["debug_rate_hz"]) <= 0.0:
            raise ValueError("debug_rate_hz must be greater than zero")
        if float(p["log_interval_sec"]) <= 0.0:
            raise ValueError("log_interval_sec must be greater than zero")
        if float(p["image_timeout_sec"]) <= 0.0:
            raise ValueError("image_timeout_sec must be greater than zero")

    def _log_configuration(self) -> None:
        p = self.params
        self.get_logger().info(
            f"Yellow area follower started: image_topic={self.image_topic}, "
            f"cmd_topic={self.cmd_topic}"
        )
        self.get_logger().info(
            "Debug: "
            f"debug_image_topic={p['debug_image_topic']}, "
            f"debug_mask_topic={p['debug_mask_topic']}, "
            f"publish_debug_image={p['publish_debug_image']}, "
            f"boundary_debug_mask_topic={p['boundary_debug_mask_topic']}, "
            f"publish_boundary_debug_mask="
            f"{p['publish_boundary_debug_mask']}, "
            f"debug_rate_hz={p['debug_rate_hz']}, "
            f"dry_run={p['dry_run']}, "
            f"control_only_when_started={p['control_only_when_started']}"
        )
        self.get_logger().info(
            "HSV threshold: "
            f"H={p['h_min']}..{p['h_max']}, "
            f"S={p['s_min']}..{p['s_max']}, "
            f"V={p['v_min']}..{p['v_max']}; "
            f"roi_y_start_ratio={p['roi_y_start_ratio']}, "
            f"roi_y_end_ratio={p['roi_y_end_ratio']}"
        )
        self.get_logger().info(
            "Green HSV threshold: "
            f"H={p['green_h_min']}..{p['green_h_max']}, "
            f"S={p['green_s_min']}..{p['green_s_max']}, "
            f"V={p['green_v_min']}..{p['green_v_max']}"
        )
        self.get_logger().info(
            "Control: "
            f"linear={p['linear_speed']}, min_linear={p['min_linear_speed']}, "
            f"kp={p['kp']}, kd={p['kd']}, max_angular={p['max_angular']}, "
            f"smoothing_alpha={p['smoothing_alpha']}, "
            f"angular_smoothing_alpha={p['angular_smoothing_alpha']}, "
            f"head_gain={p['head_gain']}, "
            f"error_deadband={p['error_deadband']}, "
            f"lost_stop_frames={p['lost_stop_frames']}, "
            f"search_on_lost={p['search_on_lost']}"
        )

    def _image_callback(self, msg: Image) -> None:
        self.last_image_time = time.monotonic()
        self.watchdog_stopped = False
        try:
            bgr = image_message_to_bgr(msg)
            p = self.params
            result = extract_boundary_path(
                bgr,
                (
                    int(p["h_min"]),
                    int(p["s_min"]),
                    int(p["v_min"]),
                ),
                (
                    int(p["h_max"]),
                    int(p["s_max"]),
                    int(p["v_max"]),
                ),
                (
                    int(p["green_h_min"]),
                    int(p["green_s_min"]),
                    int(p["green_v_min"]),
                ),
                (
                    int(p["green_h_max"]),
                    int(p["green_s_max"]),
                    int(p["green_v_max"]),
                ),
                float(p["roi_y_start_ratio"]),
                float(p["roi_y_end_ratio"]),
                int(p["boundary_scan_count"]),
                int(p["boundary_band_half_height"]),
                float(p["min_boundary_gap_px"]),
                float(p["max_boundary_jump_px"]),
                float(p["expected_lane_width_px"]),
                bool(p["use_virtual_boundary"]),
                int(p["min_valid_bands"]),
                int(p["min_yellow_pixels"]),
                float(p["near_y_ratio"]),
                float(p["far_y_ratio"]),
                float(p["head_gain"]),
            )
            if result.valid:
                self._handle_valid_result(bgr, msg, result)
            else:
                self._handle_lost_result(bgr, msg, result)
        except Exception as exc:
            self.lost_frames += 1
            self.single_boundary_frames = 0
            self.previous_error = None
            self.previous_angular = 0.0
            self._publish_stop()
            self._throttled_error(
                f"Image processing failed; stopping: {exc}"
            )

    def _handle_valid_result(
        self,
        bgr: np.ndarray,
        source_msg: Image,
        result: VisionResult,
    ) -> None:
        p = self.params
        speed_scale = 1.0
        enough_both = (
            result.both_visible_count
            >= int(p["min_left_right_visible_rows"])
        )
        if result.source == "yellow_area_center":
            self.lost_frames += 1
            self.single_boundary_frames = 0
            speed_scale = float(p["boundary_loss_slowdown"])
            if self.lost_frames > int(p["lost_stop_frames"]):
                self.previous_error = None
                self.previous_angular = 0.0
                self._publish_stop()
                self._throttled_status_log(result, 0.0, 0.0)
                self._publish_debug(bgr, source_msg, result, 0.0, 0.0)
                return
        else:
            self.lost_frames = 0
            if enough_both:
                self.single_boundary_frames = 0
            else:
                self.single_boundary_frames += 1
                speed_scale = float(p["boundary_loss_slowdown"])
                if self.single_boundary_frames > int(
                    p["single_boundary_max_frames"]
                ):
                    speed_scale *= float(p["boundary_loss_slowdown"])

        now = time.monotonic()
        elapsed = now - self.last_control_time
        self.last_control_time = now
        filtered_error, linear, angular = compute_control(
            result.control_error,
            self.previous_error,
            self.previous_angular,
            elapsed,
            float(p["smoothing_alpha"]),
            float(p["angular_smoothing_alpha"]),
            float(p["error_deadband"]),
            float(p["linear_speed"]),
            float(p["min_linear_speed"]),
            speed_scale,
            float(p["kp"]),
            float(p["kd"]),
            float(p["max_angular"]),
            float(p["max_angular_delta_per_sec"]),
        )
        self.previous_error = filtered_error
        self.previous_angular = angular
        if abs(filtered_error) > 1e-6:
            self.last_turn_direction = -1.0 if filtered_error > 0.0 else 1.0
        self._publish_velocity(linear, angular)
        self._throttled_status_log(result, linear, angular)
        self._publish_debug(bgr, source_msg, result, linear, angular)

    def _handle_lost_result(
        self,
        bgr: np.ndarray,
        source_msg: Image,
        result: VisionResult,
    ) -> None:
        self.lost_frames += 1
        self.single_boundary_frames = 0
        self.previous_error = None
        self.previous_angular = 0.0
        self.last_control_time = time.monotonic()
        p = self.params
        angular = 0.0
        if (
            bool(p["search_on_lost"])
            and self.lost_frames <= int(p["lost_stop_frames"])
        ):
            angular = (
                self.last_turn_direction * float(p["search_angular"])
            )
            self._publish_velocity(0.0, angular)
        else:
            self._publish_stop()
        self._throttled_status_log(result, 0.0, angular)
        self._publish_debug(bgr, source_msg, result, 0.0, angular)

    def _publish_velocity(self, linear: float, angular: float) -> None:
        command = Twist()
        if bool(self.params["dry_run"]):
            command.linear.x = 0.0
            command.angular.z = 0.0
        else:
            command.linear.x = float(linear)
            command.angular.z = float(angular)
        self.cmd_publisher.publish(command)

    def _publish_stop(self) -> None:
        self._publish_velocity(0.0, 0.0)

    def publish_stop_repeated(self) -> None:
        """Publish several stop messages before shutdown."""
        for _ in range(STOP_PUBLISH_COUNT):
            self._publish_stop()
            time.sleep(STOP_PUBLISH_INTERVAL_SEC)

    def _watchdog_callback(self) -> None:
        elapsed = time.monotonic() - self.last_image_time
        if elapsed < float(self.params["image_timeout_sec"]):
            return
        self._publish_stop()
        if not self.watchdog_stopped:
            self.previous_error = None
            self.previous_angular = 0.0
            self.single_boundary_frames = 0
            self.watchdog_stopped = True
            self.get_logger().error(
                f"No image received for {elapsed:.2f}s; stopping."
            )

    def _throttled_status_log(
        self,
        result: VisionResult,
        linear: float,
        angular: float,
    ) -> None:
        now = time.monotonic()
        if now - self.last_log_time < float(
            self.params["log_interval_sec"]
        ):
            return
        self.last_log_time = now
        self.get_logger().info(
            f"yellow_pixels={result.yellow_pixels}, "
            f"green_pixels={result.green_pixels}, "
            f"valid_center_count={result.valid_center_count}, "
            f"left_visible_count={result.left_visible_count}, "
            f"right_visible_count={result.right_visible_count}, "
            f"both_visible_count={result.both_visible_count}, "
            f"near_error={result.near_error:.3f}, "
            f"far_error={result.far_error:.3f}, "
            f"control_error={result.control_error:.3f}, "
            f"linear.x={linear:.3f}, angular.z={angular:.3f}, "
            f"lost_frames={self.lost_frames}, "
            f"single_boundary_frames={self.single_boundary_frames}, "
            f"source={result.source}"
        )

    def _throttled_error(self, message: str) -> None:
        now = time.monotonic()
        if now - self.last_error_log_time < float(
            self.params["log_interval_sec"]
        ):
            return
        self.last_error_log_time = now
        self.get_logger().error(message)

    def _publish_debug(
        self,
        bgr: np.ndarray,
        source_msg: Image,
        result: VisionResult,
        linear: float,
        angular: float,
    ) -> None:
        if (
            self.debug_image_publisher is None
            or self.debug_mask_publisher is None
        ) and self.boundary_debug_mask_publisher is None:
            return

        p = self.params
        now = time.monotonic()
        if now - self.last_debug_time < 1.0 / float(p["debug_rate_hz"]):
            return
        self.last_debug_time = now

        image_height, image_width = bgr.shape[:2]
        roi_y_start = int(image_height * float(p["roi_y_start_ratio"]))
        roi_y_start = min(max(roi_y_start, 0), image_height - 1)
        roi_y_end = int(image_height * float(p["roi_y_end_ratio"]))
        roi_y_end = min(max(roi_y_end, roi_y_start + 1), image_height)

        if (
            self.debug_image_publisher is not None
            and self.debug_mask_publisher is not None
        ):
            debug = render_debug_image(
                bgr,
                result,
                float(p["roi_y_start_ratio"]),
                float(p["roi_y_end_ratio"]),
                linear,
                angular,
                self.lost_frames,
                self.single_boundary_frames,
                float(p["kp"]),
                float(p["kd"]),
                float(p["head_gain"]),
                bool(p["dry_run"]),
            )
            full_yellow_mask = np.zeros(
                (image_height, image_width), dtype=np.uint8
            )
            full_yellow_mask[roi_y_start:roi_y_end, :] = result.yellow_mask
            self.debug_image_publisher.publish(
                numpy_to_image_message(debug, "bgr8", source_msg)
            )
            self.debug_mask_publisher.publish(
                numpy_to_image_message(
                    full_yellow_mask, "mono8", source_msg
                )
            )

        if self.boundary_debug_mask_publisher is not None:
            full_boundary_mask = np.zeros(
                (image_height, image_width), dtype=np.uint8
            )
            full_boundary_mask[
                roi_y_start:roi_y_end, :
            ] = result.boundary_mask
            self.boundary_debug_mask_publisher.publish(
                numpy_to_image_message(
                    full_boundary_mask, "mono8", source_msg
                )
            )


def main(args=None) -> int:
    rclpy.init(args=args)
    node: Optional[YellowAreaFollower] = None
    exit_code = 1
    try:
        node = YellowAreaFollower()
        rclpy.spin(node)
        exit_code = 0
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("Ctrl+C received; publishing repeated stop.")
            node.publish_stop_repeated()
            exit_code = 0
    except Exception as exc:
        if node is not None:
            node.get_logger().error(f"Fatal error; stopping: {exc}")
            node.publish_stop_repeated()
        else:
            print(f"Failed to start yellow area follower: {exc}", flush=True)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
