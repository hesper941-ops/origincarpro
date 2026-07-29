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
    path_mode: str
    yellow_pixels: int
    green_pixels: int
    yellow_mask: np.ndarray
    green_mask: np.ndarray
    boundary_mask: np.ndarray
    bands: List[BoundaryBandResult]
    boundary_sequences: List[List[Tuple[float, float, float]]]
    left_boundary_points: List[Tuple[float, float, float]]
    right_boundary_points: List[Tuple[float, float, float]]
    single_boundary_points: List[Tuple[float, float, float]]
    target_path_points: List[Tuple[float, float]]
    boundary_points_count: int
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
    history_age: int
    yellow_fallback_frames: int


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


def _extract_boundary_path_v3(
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


def extract_boundary_points_from_mask(
    boundary_mask: np.ndarray,
    component_min_area: int,
    keep_component_ratio: float,
    y_bucket_px: int,
    min_bucket_points: int,
    max_x_jump_px: float,
    min_points: int,
) -> List[List[Tuple[float, float, float]]]:
    """Extract long, y-bucketed, continuous boundary point sequences."""
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        boundary_mask, connectivity=8
    )
    if component_count <= 1:
        return []
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    maximum_area = int(np.max(component_areas))
    minimum_kept_area = max(
        int(component_min_area),
        int(round(maximum_area * keep_component_ratio)),
    )
    sequences: List[List[Tuple[float, float, float]]] = []

    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_kept_area:
            continue
        y_values, x_values = np.nonzero(labels == label)
        buckets = y_values // max(1, int(y_bucket_px))
        component_points: List[Tuple[float, float, float]] = []
        for bucket in np.unique(buckets):
            selected = buckets == bucket
            count = int(np.count_nonzero(selected))
            if count < min_bucket_points:
                continue
            bucket_x = x_values[selected]
            bucket_y = y_values[selected]
            confidence = min(
                1.0,
                count / max(float(min_bucket_points * 3), 1.0),
            )
            component_points.append(
                (
                    float(np.median(bucket_x)),
                    float(np.median(bucket_y)),
                    confidence,
                )
            )
        component_points.sort(key=lambda point: point[1])
        if not component_points:
            continue

        continuous_parts: List[List[Tuple[float, float, float]]] = [[]]
        for point in component_points:
            if (
                continuous_parts[-1]
                and abs(point[0] - continuous_parts[-1][-1][0])
                > max_x_jump_px
            ):
                continuous_parts.append([])
            continuous_parts[-1].append(point)
        for part in continuous_parts:
            if len(part) >= min_points:
                sequences.append(part)

    sequences.sort(
        key=lambda points: (
            len(points),
            points[-1][1] - points[0][1],
            sum(point[2] for point in points),
        ),
        reverse=True,
    )
    return sequences


def _fit_x_of_y(
    points: List[Tuple[float, float, float]],
) -> Optional[np.ndarray]:
    """Fit a robust x(y) curve to boundary or target points."""
    if not points:
        return None
    x_values = np.asarray([point[0] for point in points], dtype=float)
    y_values = np.asarray([point[1] for point in points], dtype=float)
    confidence = np.asarray([point[2] for point in points], dtype=float)
    degree = 2 if len(points) >= 5 and np.ptp(y_values) >= 24.0 else 1
    degree = min(degree, len(points) - 1)
    if degree <= 0:
        return np.asarray([float(np.median(x_values))], dtype=float)
    try:
        coefficients = np.polyfit(
            y_values,
            x_values,
            degree,
            w=np.sqrt(np.clip(confidence, 0.05, None)),
        )
    except (ValueError, np.linalg.LinAlgError):
        return None
    if len(points) >= 6:
        residuals = np.abs(np.polyval(coefficients, y_values) - x_values)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        keep = residuals <= max(15.0, median + 2.5 * max(mad, 1.0))
        if int(np.count_nonzero(keep)) >= degree + 2:
            try:
                coefficients = np.polyfit(
                    y_values[keep],
                    x_values[keep],
                    degree,
                    w=np.sqrt(np.clip(confidence[keep], 0.05, None)),
                )
            except (ValueError, np.linalg.LinAlgError):
                pass
    return coefficients


def _choose_boundary_pair(
    sequences: List[List[Tuple[float, float, float]]],
    min_gap_px: float,
    expected_width_px: float,
) -> Optional[
    Tuple[
        List[Tuple[float, float, float]],
        List[Tuple[float, float, float]],
        np.ndarray,
        np.ndarray,
        float,
        float,
    ]
]:
    """Choose two overlapping continuous sequences as left/right boundaries."""
    best = None
    best_score = -float("inf")
    candidates = sequences[:6]
    for first_index, first in enumerate(candidates):
        first_fit = _fit_x_of_y(first)
        if first_fit is None:
            continue
        for second in candidates[first_index + 1:]:
            second_fit = _fit_x_of_y(second)
            if second_fit is None:
                continue
            overlap_start = max(first[0][1], second[0][1])
            overlap_end = min(first[-1][1], second[-1][1])
            if overlap_end - overlap_start < 24.0:
                continue
            middle_y = (overlap_start + overlap_end) * 0.5
            first_x = float(np.polyval(first_fit, middle_y))
            second_x = float(np.polyval(second_fit, middle_y))
            gap = abs(second_x - first_x)
            if gap < min_gap_px:
                continue
            if first_x <= second_x:
                left, right = first, second
                left_fit, right_fit = first_fit, second_fit
            else:
                left, right = second, first
                left_fit, right_fit = second_fit, first_fit
            width_penalty = abs(gap - expected_width_px) / max(
                expected_width_px, 1.0
            )
            score = (
                len(left)
                + len(right)
                + 0.03 * (overlap_end - overlap_start)
                - 2.0 * width_penalty
            )
            if score > best_score:
                best_score = score
                best = (
                    left,
                    right,
                    left_fit,
                    right_fit,
                    overlap_start,
                    overlap_end,
                )
    return best


def _yellow_vote(
    yellow_mask: np.ndarray,
    x: float,
    y: float,
    radius: int,
) -> int:
    """Count yellow pixels in a small square around a floating-point sample."""
    height, width = yellow_mask.shape
    center_x = int(round(x))
    center_y = int(round(y))
    x0 = max(0, center_x - radius)
    x1 = min(width, center_x + radius + 1)
    y0 = max(0, center_y - radius)
    y1 = min(height, center_y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return 0
    return int(cv2.countNonZero(yellow_mask[y0:y1, x0:x1]))


def _single_boundary_offset_path(
    boundary_points: List[Tuple[float, float, float]],
    yellow_mask: np.ndarray,
    offset_px: float,
    sample_count: int,
    y_min_ratio: float,
    y_max_ratio: float,
    normal_sample_px: float,
    vote_radius: int,
) -> Tuple[List[Tuple[float, float]], int]:
    """Offset a single boundary along the normal pointing into yellow."""
    coefficients = _fit_x_of_y(boundary_points)
    if coefficients is None:
        return [], 0
    roi_height, roi_width = yellow_mask.shape
    point_y_min = boundary_points[0][1]
    point_y_max = boundary_points[-1][1]
    sample_y_min = max(point_y_min, (roi_height - 1) * y_min_ratio)
    sample_y_max = min(point_y_max, (roi_height - 1) * y_max_ratio)
    if sample_y_max - sample_y_min < 8.0:
        sample_y_min, sample_y_max = point_y_min, point_y_max
    if sample_y_max <= sample_y_min:
        return [], 0
    sample_y_values = np.linspace(
        sample_y_min,
        sample_y_max,
        max(2, int(sample_count)),
    )
    derivative_coefficients = np.polyder(coefficients)
    vote_distance = max(float(normal_sample_px), vote_radius + 2.0)
    plus_votes = 0
    minus_votes = 0
    geometry = []
    for y in sample_y_values:
        x = float(np.polyval(coefficients, y))
        dx_dy = (
            float(np.polyval(derivative_coefficients, y))
            if derivative_coefficients.size
            else 0.0
        )
        normal = np.asarray((1.0, -dx_dy), dtype=float)
        normal /= max(float(np.linalg.norm(normal)), 1e-6)
        plus_votes += _yellow_vote(
            yellow_mask,
            x + normal[0] * vote_distance,
            y + normal[1] * vote_distance,
            vote_radius,
        )
        minus_votes += _yellow_vote(
            yellow_mask,
            x - normal[0] * vote_distance,
            y - normal[1] * vote_distance,
            vote_radius,
        )
        geometry.append((x, float(y), normal))
    if plus_votes == minus_votes:
        return [], 0
    inward_sign = 1 if plus_votes > minus_votes else -1
    target_path = []
    for x, y, normal in geometry:
        target_x = float(
            np.clip(x + inward_sign * normal[0] * offset_px, 0, roi_width - 1)
        )
        target_y = float(
            np.clip(y + inward_sign * normal[1] * offset_px, 0, roi_height - 1)
        )
        target_path.append((target_x, target_y))
    target_path.sort(key=lambda point: point[1])
    return target_path, inward_sign


def _target_path_lookahead(
    target_path: List[Tuple[float, float]],
    image_width: int,
    image_height: int,
    roi_y_start: int,
    roi_y_end: int,
    near_y_ratio: float,
    far_y_ratio: float,
    head_gain: float,
    use_heading_error: bool,
    heading_gain: float,
) -> Tuple[float, float, float, float, float, float, float]:
    """Evaluate one target path at image-relative near/far y positions."""
    if not target_path:
        raise ValueError("target_path must not be empty")
    points = [(x, y, 1.0) for x, y in target_path]
    coefficients = _fit_x_of_y(points)
    if coefficients is None:
        raise ValueError("failed to fit target path")
    roi_height = roi_y_end - roi_y_start
    near_y = float(
        np.clip(
            image_height * near_y_ratio - roi_y_start,
            0.0,
            roi_height - 1.0,
        )
    )
    far_y = float(
        np.clip(
            image_height * far_y_ratio - roi_y_start,
            0.0,
            roi_height - 1.0,
        )
    )
    near_x = float(
        np.clip(np.polyval(coefficients, near_y), 0.0, image_width - 1.0)
    )
    far_x = float(
        np.clip(np.polyval(coefficients, far_y), 0.0, image_width - 1.0)
    )
    image_center = image_width * 0.5
    near_error = (near_x - image_center) / image_center
    far_error = (far_x - image_center) / image_center
    control_error = near_error + head_gain * (far_error - near_error)
    if use_heading_error:
        heading_error = (far_x - near_x) / image_center
        control_error += heading_gain * heading_error
    return (
        near_x,
        far_x,
        near_y,
        far_y,
        near_error,
        far_error,
        control_error,
    )


def _make_v4_result(
    path_mode: str,
    target_path: List[Tuple[float, float]],
    yellow_pixels: int,
    green_pixels: int,
    yellow_mask: np.ndarray,
    green_mask: np.ndarray,
    boundary_mask: np.ndarray,
    sequences: List[List[Tuple[float, float, float]]],
    left_points: List[Tuple[float, float, float]],
    right_points: List[Tuple[float, float, float]],
    single_points: List[Tuple[float, float, float]],
    valid_center_count: int,
    left_visible_count: int,
    right_visible_count: int,
    both_visible_count: int,
    fallback_centroid: Optional[Tuple[float, float]],
    image_width: int,
    image_height: int,
    roi_y_start: int,
    roi_y_end: int,
    params: dict,
) -> VisionResult:
    """Create a VisionResult and calculate common near/far look-ahead."""
    if target_path:
        (
            near_x,
            far_x,
            near_y,
            far_y,
            near_error,
            far_error,
            control_error,
        ) = _target_path_lookahead(
            target_path,
            image_width,
            image_height,
            roi_y_start,
            roi_y_end,
            float(params["near_y_ratio"]),
            float(params["far_y_ratio"]),
            float(params["head_gain"]),
            bool(params["use_heading_error"]),
            float(params["heading_gain"]),
        )
    else:
        near_x = far_x = near_y = far_y = None
        near_error = far_error = control_error = 0.0
    return VisionResult(
        bool(target_path),
        path_mode,
        yellow_pixels,
        green_pixels,
        yellow_mask,
        green_mask,
        boundary_mask,
        [],
        sequences,
        left_points,
        right_points,
        single_points,
        target_path,
        sum(len(sequence) for sequence in sequences),
        valid_center_count,
        left_visible_count,
        right_visible_count,
        both_visible_count,
        near_x,
        far_x,
        near_y,
        far_y,
        near_error,
        far_error,
        control_error,
        fallback_centroid,
        0,
        0,
    )


def extract_boundary_path(
    bgr: np.ndarray,
    params: dict,
) -> VisionResult:
    """V4 path extraction with double, single-normal, and yellow fallback."""
    image_height, image_width = bgr.shape[:2]
    roi_y_start = int(image_height * float(params["roi_y_start_ratio"]))
    roi_y_start = min(max(roi_y_start, 0), image_height - 1)
    roi_y_end = int(image_height * float(params["roi_y_end_ratio"]))
    roi_y_end = min(max(roi_y_end, roi_y_start + 1), image_height)
    roi = bgr[roi_y_start:roi_y_end, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(
        hsv,
        np.asarray(
            (
                int(params["h_min"]),
                int(params["s_min"]),
                int(params["v_min"]),
            ),
            dtype=np.uint8,
        ),
        np.asarray(
            (
                int(params["h_max"]),
                int(params["s_max"]),
                int(params["v_max"]),
            ),
            dtype=np.uint8,
        ),
    )
    green_mask = cv2.inRange(
        hsv,
        np.asarray(
            (
                int(params["green_h_min"]),
                int(params["green_s_min"]),
                int(params["green_v_min"]),
            ),
            dtype=np.uint8,
        ),
        np.asarray(
            (
                int(params["green_h_max"]),
                int(params["green_s_max"]),
                int(params["green_v_max"]),
            ),
            dtype=np.uint8,
        ),
    )
    # The configurable hue ranges may overlap (the recommended defaults do).
    # Assign overlapping pixels to the nearer hue-range center so the contact
    # mask remains a yellow/green interface instead of covering a whole region.
    overlap = (yellow_mask > 0) & (green_mask > 0)
    if np.any(overlap):
        hue = hsv[:, :, 0].astype(np.float32)
        yellow_center = (
            float(params["h_min"]) + float(params["h_max"])
        ) * 0.5
        green_center = (
            float(params["green_h_min"]) + float(params["green_h_max"])
        ) * 0.5
        yellow_distance = np.abs(hue - yellow_center)
        green_distance = np.abs(hue - green_center)
        assign_yellow = overlap & (yellow_distance <= green_distance)
        assign_green = overlap & ~assign_yellow
        green_mask[assign_yellow] = 0
        yellow_mask[assign_green] = 0

    open_size = max(1, int(params["mask_open_kernel_size"]))
    close_size = max(1, int(params["mask_close_kernel_size"]))
    if open_size % 2 == 0:
        open_size += 1
    if close_size % 2 == 0:
        close_size += 1
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_size, open_size)
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_size, close_size)
    )
    for mask in (yellow_mask, green_mask):
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, dst=mask)
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, dst=mask)

    contact_size = max(1, int(params["boundary_contact_kernel_size"]))
    if contact_size % 2 == 0:
        contact_size += 1
    contact_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (contact_size, contact_size)
    )
    yellow_contact = cv2.bitwise_and(
        cv2.dilate(yellow_mask, contact_kernel), green_mask
    )
    green_contact = cv2.bitwise_and(
        cv2.dilate(green_mask, contact_kernel), yellow_mask
    )
    boundary_mask = cv2.bitwise_or(yellow_contact, green_contact)
    boundary_mask = cv2.dilate(
        boundary_mask,
        contact_kernel,
        iterations=max(0, int(params["boundary_dilate_iterations"])),
    )

    yellow_pixels = int(cv2.countNonZero(yellow_mask))
    green_pixels = int(cv2.countNonZero(green_mask))
    sequences = extract_boundary_points_from_mask(
        boundary_mask,
        int(params["boundary_component_min_area"]),
        float(params["boundary_keep_component_ratio"]),
        int(params["boundary_y_bucket_px"]),
        int(params["boundary_min_bucket_points"]),
        float(params["boundary_max_x_jump_px"]),
        int(params["boundary_min_points"]),
    )

    pair = _choose_boundary_pair(
        sequences,
        float(params["min_boundary_gap_px"]),
        float(params["expected_lane_width_px"]),
    )
    if pair is not None:
        left, right, left_fit, right_fit, y0, y1 = pair
        sample_count = max(
            int(params["boundary_scan_count"]),
            int(params["min_left_right_visible_rows"]),
        )
        sample_y_values = np.linspace(y0, y1, sample_count)
        target_path = [
            (
                float(
                    np.clip(
                        (
                            np.polyval(left_fit, y)
                            + np.polyval(right_fit, y)
                        )
                        * 0.5,
                        0.0,
                        image_width - 1.0,
                    )
                ),
                float(y),
            )
            for y in sample_y_values
        ]
        return _make_v4_result(
            "both_boundary_center",
            target_path,
            yellow_pixels,
            green_pixels,
            yellow_mask,
            green_mask,
            boundary_mask,
            sequences,
            left,
            right,
            [],
            len(target_path),
            len(left),
            len(right),
            len(target_path),
            None,
            image_width,
            image_height,
            roi_y_start,
            roi_y_end,
            params,
        )

    single_points: List[Tuple[float, float, float]] = []
    if sequences:
        single_points = max(
            sequences,
            key=lambda points: (
                len(points),
                points[-1][1] - points[0][1],
            ),
        )
    single_span = (
        single_points[-1][1] - single_points[0][1]
        if single_points
        else 0.0
    )
    if (
        bool(params["single_boundary_enable"])
        and len(single_points) >= int(params["single_boundary_min_points"])
        and single_span >= float(params["single_boundary_min_y_span_px"])
    ):
        target_path, inward_sign = _single_boundary_offset_path(
            single_points,
            yellow_mask,
            float(params["single_boundary_offset_px"]),
            int(params["single_boundary_sample_count"]),
            float(params["single_boundary_y_min_ratio"]),
            float(params["single_boundary_y_max_ratio"]),
            float(params["single_boundary_normal_sample_px"]),
            int(params["single_boundary_yellow_vote_radius"]),
        )
        if target_path:
            left_count = len(single_points) if inward_sign > 0 else 0
            right_count = len(single_points) if inward_sign < 0 else 0
            return _make_v4_result(
                "single_boundary_offset",
                target_path,
                yellow_pixels,
                green_pixels,
                yellow_mask,
                green_mask,
                boundary_mask,
                sequences,
                single_points if inward_sign > 0 else [],
                single_points if inward_sign < 0 else [],
                single_points,
                len(target_path),
                left_count,
                right_count,
                0,
                None,
                image_width,
                image_height,
                roi_y_start,
                roi_y_end,
                params,
            )

    fallback_centroid = None
    if (
        bool(params["yellow_fallback_enable"])
        and yellow_pixels >= int(params["yellow_fallback_min_pixels"])
    ):
        fallback_centroid = _largest_yellow_centroid(yellow_mask)
    if fallback_centroid is not None:
        fallback_x, fallback_y = fallback_centroid
        roi_height = roi_y_end - roi_y_start
        target_path = [
            (fallback_x, max(0.0, fallback_y - roi_height * 0.15)),
            (fallback_x, min(roi_height - 1.0, fallback_y + roi_height * 0.15)),
        ]
        return _make_v4_result(
            "yellow_area_center_fallback",
            target_path,
            yellow_pixels,
            green_pixels,
            yellow_mask,
            green_mask,
            boundary_mask,
            sequences,
            [],
            [],
            single_points,
            0,
            0,
            0,
            0,
            fallback_centroid,
            image_width,
            image_height,
            roi_y_start,
            roi_y_end,
            params,
        )

    return _make_v4_result(
        "lost_stop",
        [],
        yellow_pixels,
        green_pixels,
        yellow_mask,
        green_mask,
        boundary_mask,
        sequences,
        [],
        [],
        single_points,
        0,
        0,
        0,
        0,
        None,
        image_width,
        image_height,
        roi_y_start,
        roi_y_end,
        params,
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


def _render_debug_image_legacy(
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
    single_boundary_offset_px: float,
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

    # Actual boundary-mask points and their continuous sequences.
    for sequence in result.boundary_sequences:
        if not sequence:
            continue
        boundary_polyline = np.asarray(
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
                for x, y, _ in sequence
            ],
            dtype=np.int32,
        )
        if len(boundary_polyline) >= 2:
            cv2.polylines(
                debug,
                [boundary_polyline],
                False,
                green,
                2,
                cv2.LINE_AA,
            )
        for point in boundary_polyline:
            cv2.circle(debug, tuple(point), 3, green, -1)

    # Vehicle center, fitted center trajectory, and near/far look-ahead points.
    image_center_x = image_width // 2
    cv2.line(
        debug,
        (image_center_x, roi_y_start),
        (image_center_x, image_height - 1),
        blue,
        2,
    )
    if len(result.target_path_points) >= 2:
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
                for x, y in result.target_path_points
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

    if (
        result.path_mode == "yellow_area_center_fallback"
        and result.fallback_centroid
    ):
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
        f"path_mode={result.path_mode}",
        f"yellow_pixels={result.yellow_pixels}",
        f"green_pixels={result.green_pixels}",
        f"boundary_points_count={result.boundary_points_count}",
        f"target_path_points_count={len(result.target_path_points)}",
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
        f"history_age={result.history_age}",
        f"yellow_fallback_frames={result.yellow_fallback_frames}",
        f"single_boundary_offset_px={single_boundary_offset_px:.1f}",
        f"kp={kp:.3f}",
        f"kd={kd:.3f}",
        f"head_gain={head_gain:.3f}",
    ]
    if result.path_mode == "yellow_area_center_fallback":
        status_lines.append("fallback=yellow_area_center")
    if dry_run:
        status_lines.append("DRY RUN / ZERO CMD")

    line_height = 20
    lost_title_height = 28 if result.path_mode == "lost_stop" else 0
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
    if result.path_mode == "lost_stop":
        cv2.putText(
            debug,
            "LOST / STOP",
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


def _debug_path_polyline(
    points: List[Tuple[float, float]],
    interpolate: bool,
    interpolate_count: int,
    image_width: int,
    roi_y_start: int,
    roi_y_end: int,
) -> np.ndarray:
    """Create a smooth display-only polyline without changing control points."""
    if not points:
        return np.empty((0, 2), dtype=np.int32)
    display_points = list(points)
    if interpolate and len(points) >= 2:
        fitted = _fit_x_of_y([(x, y, 1.0) for x, y in points])
        if fitted is not None:
            y_values = np.linspace(
                min(point[1] for point in points),
                max(point[1] for point in points),
                max(2, int(interpolate_count)),
            )
            display_points = [
                (float(np.polyval(fitted, y)), float(y)) for y in y_values
            ]
    return np.asarray(
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
            for x, y in display_points
        ],
        dtype=np.int32,
    )


def _path_mode_color(path_mode: str) -> Tuple[int, int, int]:
    """Return the standardized BGR color for one path mode."""
    return {
        "both_boundary_center": (0, 255, 0),
        "single_boundary_offset": (255, 255, 0),
        "history_prediction": (0, 255, 255),
        "yellow_area_center_fallback": (0, 165, 255),
        "lost_stop": (0, 0, 255),
    }.get(path_mode, (255, 255, 255))


def draw_info_panel(
    canvas: np.ndarray,
    x0: int,
    panel_width: int,
    result: VisionResult,
    linear: float,
    angular: float,
    speed_scale: float,
    lost_frames: int,
    params: dict,
    fill_background: bool = True,
) -> None:
    """Draw a compact, grouped status panel beside the camera image."""
    panel_height = canvas.shape[0]
    x1 = min(canvas.shape[1], x0 + panel_width)
    if fill_background:
        canvas[:, x0:x1] = (24, 24, 24)
    left = x0 + 12
    white = (235, 235, 235)
    muted = (175, 175, 175)
    cyan = (255, 255, 0)
    purple = (255, 0, 255)
    mode_color = _path_mode_color(result.path_mode)

    y = 20
    if bool(params["dry_run"]):
        cv2.putText(
            canvas,
            "DRY RUN",
            (left, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            purple,
            2,
            cv2.LINE_AA,
        )
        y += 20
    cv2.putText(
        canvas,
        "MODE:",
        (left, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        muted,
        1,
        cv2.LINE_AA,
    )
    y += 20
    cv2.putText(
        canvas,
        result.path_mode,
        (left, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        mode_color,
        2,
        cv2.LINE_AA,
    )
    y += 16

    groups = [
        (
            "PATH",
            [
                f"boundary_points: {result.boundary_points_count}",
                f"target_points: {len(result.target_path_points)}",
                f"valid_center_count: {result.valid_center_count}",
                f"left_visible: {result.left_visible_count}",
                f"right_visible: {result.right_visible_count}",
                f"both_visible: {result.both_visible_count}",
            ],
        ),
        (
            "ERROR",
            [
                f"near_error: {result.near_error:.3f}",
                f"far_error: {result.far_error:.3f}",
                f"control_error: {result.control_error:.3f}",
            ],
        ),
        (
            "CMD",
            [
                f"linear.x: {linear:.3f}",
                f"angular.z: {angular:.3f}",
                f"speed_scale: {speed_scale:.3f}",
            ],
        ),
        (
            "VISION",
            [
                f"yellow_pixels: {result.yellow_pixels}",
                f"green_pixels: {result.green_pixels}",
                f"lost_frames: {lost_frames}",
                f"history_age: {result.history_age}",
                f"fallback_frames: {result.yellow_fallback_frames}",
            ],
        ),
        (
            "PARAM",
            [
                f"offset_px: {float(params['single_boundary_offset_px']):.1f}",
                f"kp: {float(params['kp']):.3f}",
                f"kd: {float(params['kd']):.3f}",
                f"head_gain: {float(params['head_gain']):.3f}",
            ],
        ),
    ]
    remaining_lines = sum(1 + len(lines) for _, lines in groups)
    line_height = max(
        12,
        min(16, int((panel_height - y - 4) / max(remaining_lines, 1))),
    )
    field_scale = max(0.32, min(0.42, line_height / 36.0))
    for title, lines in groups:
        if y >= panel_height - 2:
            break
        y += 3
        cv2.putText(
            canvas,
            f"[{title}]",
            (left, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            field_scale,
            cyan,
            1,
            cv2.LINE_AA,
        )
        y += line_height
        for line in lines:
            if y >= panel_height - 2:
                break
            cv2.putText(
                canvas,
                line,
                (left, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                field_scale,
                white,
                1,
                cv2.LINE_AA,
            )
            y += line_height


def render_debug_image(
    bgr: np.ndarray,
    result: VisionResult,
    linear: float,
    angular: float,
    speed_scale: float,
    lost_frames: int,
    params: dict,
) -> np.ndarray:
    """Render a clean main image plus an optional right-side status panel."""
    main = bgr.copy()
    image_height, image_width = main.shape[:2]
    roi_y_start = int(
        image_height * float(params["roi_y_start_ratio"])
    )
    roi_y_start = min(max(roi_y_start, 0), image_height - 1)
    roi_y_end = int(image_height * float(params["roi_y_end_ratio"]))
    roi_y_end = min(max(roi_y_end, roi_y_start + 1), image_height)
    roi_y_end_line = min(roi_y_end, image_height - 1)

    blue = (255, 0, 0)
    red = (0, 0, 255)
    cyan = (255, 255, 0)
    light_blue = (255, 210, 40)
    green = (0, 255, 0)
    purple = (255, 0, 255)
    white = (255, 255, 255)
    orange = (0, 165, 255)
    yellow = (0, 255, 255)

    yellow_alpha = float(
        np.clip(params["debug_yellow_overlay_alpha"], 0.0, 1.0)
    )
    if yellow_alpha > 0.0:
        overlay = main.copy()
        overlay_roi = overlay[roi_y_start:roi_y_end, :]
        overlay_roi[result.yellow_mask > 0] = yellow
        cv2.addWeighted(
            overlay, yellow_alpha, main, 1.0 - yellow_alpha, 0.0, main
        )
    green_alpha = float(
        np.clip(params["debug_green_overlay_alpha"], 0.0, 1.0)
    )
    if green_alpha > 0.0:
        overlay = main.copy()
        overlay_roi = overlay[roi_y_start:roi_y_end, :]
        overlay_roi[result.green_mask > 0] = (0, 180, 0)
        cv2.addWeighted(
            overlay, green_alpha, main, 1.0 - green_alpha, 0.0, main
        )
    green_contours, _ = cv2.findContours(
        result.green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(
        main[roi_y_start:roi_y_end, :],
        green_contours,
        -1,
        (0, 180, 0),
        1,
    )

    cv2.line(
        main, (0, roi_y_start), (image_width - 1, roi_y_start), cyan, 2
    )
    cv2.line(
        main,
        (0, roi_y_end_line),
        (image_width - 1, roi_y_end_line),
        cyan,
        2,
    )
    image_center_x = image_width // 2
    cv2.line(
        main,
        (image_center_x, roi_y_start),
        (image_center_x, image_height - 1),
        blue,
        2,
    )

    # Draw actual points in green and a smooth display-only boundary fit.
    for sequence in result.boundary_sequences:
        for x, y, _ in sequence:
            point = (
                int(np.clip(round(x), 0, image_width - 1)),
                int(
                    np.clip(
                        round(roi_y_start + y),
                        roi_y_start,
                        roi_y_end - 1,
                    )
                ),
            )
            cv2.circle(main, point, 3, green, -1)
        fitted = _fit_x_of_y(sequence)
        if fitted is not None and len(sequence) >= 2:
            y_values = np.linspace(
                sequence[0][1], sequence[-1][1], 40
            )
            fitted_line = np.asarray(
                [
                    (
                        int(
                            np.clip(
                                round(np.polyval(fitted, y)),
                                0,
                                image_width - 1,
                            )
                        ),
                        int(
                            np.clip(
                                round(roi_y_start + y),
                                roi_y_start,
                                roi_y_end - 1,
                            )
                        ),
                    )
                    for y in y_values
                ],
                dtype=np.int32,
            )
            cv2.polylines(
                main, [fitted_line], False, light_blue, 1, cv2.LINE_AA
            )

    if result.path_mode != "lost_stop":
        display_path = _debug_path_polyline(
            result.target_path_points,
            bool(params["debug_path_interpolate"]),
            int(params["debug_path_interpolate_count"]),
            image_width,
            roi_y_start,
            roi_y_end,
        )
        if result.path_mode == "history_prediction":
            path_color = yellow
        elif result.path_mode == "yellow_area_center_fallback":
            path_color = yellow
        else:
            path_color = purple
        if len(display_path) >= 2:
            cv2.polylines(
                main, [display_path], False, path_color, 3, cv2.LINE_AA
            )
        elif len(display_path) == 1:
            cv2.circle(main, tuple(display_path[0]), 5, path_color, -1)

        near_point = None
        if result.near_center_x is not None and result.near_y is not None:
            near_point = (
                int(
                    np.clip(
                        round(result.near_center_x), 0, image_width - 1
                    )
                ),
                int(
                    np.clip(
                        round(roi_y_start + result.near_y),
                        roi_y_start,
                        roi_y_end - 1,
                    )
                ),
            )
            cv2.circle(main, near_point, 8, (0, 0, 0), -1)
            cv2.circle(main, near_point, 6, red, -1)
            cv2.putText(
                main,
                "near",
                (near_point[0] + 8, near_point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                red,
                1,
                cv2.LINE_AA,
            )
            cv2.arrowedLine(
                main,
                (image_center_x, image_height - 1),
                near_point,
                white,
                3,
                cv2.LINE_AA,
                tipLength=0.08,
            )
        if result.far_center_x is not None and result.far_y is not None:
            far_point = (
                int(
                    np.clip(
                        round(result.far_center_x), 0, image_width - 1
                    )
                ),
                int(
                    np.clip(
                        round(roi_y_start + result.far_y),
                        roi_y_start,
                        roi_y_end - 1,
                    )
                ),
            )
            cv2.circle(main, far_point, 8, (0, 0, 0), -1)
            cv2.circle(main, far_point, 6, orange, -1)
            cv2.putText(
                main,
                "far",
                (far_point[0] + 8, far_point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                orange,
                1,
                cv2.LINE_AA,
            )
        if result.path_mode == "yellow_area_center_fallback":
            cv2.putText(
                main,
                "fallback: yellow area center",
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                orange,
                2,
                cv2.LINE_AA,
            )
    else:
        text = "LOST / STOP"
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 1.15, 3
        )
        origin = (
            max(8, (image_width - text_width) // 2),
            max(text_height + 8, image_height // 2),
        )
        cv2.putText(
            main,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.15,
            (0, 0, 0),
            6,
            cv2.LINE_AA,
        )
        cv2.putText(
            main,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.15,
            red,
            3,
            cv2.LINE_AA,
        )

    if bool(params["dry_run"]):
        cv2.putText(
            main,
            "DRY RUN / ZERO CMD",
            (10, image_height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            main,
            "DRY RUN / ZERO CMD",
            (10, image_height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            purple,
            1,
            cv2.LINE_AA,
        )

    requested_panel_width = max(1, int(params["debug_info_panel_width"]))
    if bool(params["debug_side_panel"]):
        canvas = np.full(
            (image_height, image_width + requested_panel_width, 3),
            (24, 24, 24),
            dtype=np.uint8,
        )
        canvas[:, :image_width] = main
        panel_x = image_width
        panel_width = requested_panel_width
    else:
        canvas = main
        panel_width = min(requested_panel_width, image_width)
        panel_x = image_width - panel_width
        overlay = canvas.copy()
        overlay[:, panel_x:] = (24, 24, 24)
        cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0.0, canvas)
    draw_info_panel(
        canvas,
        panel_x,
        panel_width,
        result,
        linear,
        angular,
        speed_scale,
        lost_frames,
        params,
        fill_background=bool(params["debug_side_panel"]),
    )
    return canvas


class YellowAreaFollower(Node):
    """ROS2 node for standalone yellow-area following tests."""

    def __init__(self) -> None:
        super().__init__("yellow_area_follower")
        defaults = {
            "image_topic": "/image_out",
            "cmd_topic": "/cmd_vel_raw",
            "h_min": 5,
            "h_max": 55,
            "s_min": 18,
            "s_max": 255,
            "v_min": 35,
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
            "mask_open_kernel_size": 5,
            "mask_close_kernel_size": 5,
            "boundary_contact_kernel_size": 5,
            "boundary_dilate_iterations": 1,
            "boundary_component_min_area": 80,
            "boundary_keep_component_ratio": 0.20,
            "boundary_y_bucket_px": 8,
            "boundary_min_bucket_points": 3,
            "boundary_max_x_jump_px": 100,
            "boundary_min_points": 5,
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
            "both_boundary_speed_scale": 1.0,
            "single_boundary_enable": True,
            "single_boundary_offset_px": 110,
            "single_boundary_min_points": 5,
            "single_boundary_min_y_span_px": 60,
            "single_boundary_sample_count": 8,
            "single_boundary_y_min_ratio": 0.35,
            "single_boundary_y_max_ratio": 0.85,
            "single_boundary_normal_sample_px": 12,
            "single_boundary_yellow_vote_radius": 5,
            "single_boundary_speed_scale": 0.75,
            "history_enable": True,
            "history_max_frames": 5,
            "history_speed_scale": 0.45,
            "history_confidence_decay": 0.75,
            "yellow_fallback_enable": True,
            "yellow_fallback_speed_scale": 0.30,
            "yellow_fallback_max_frames": 6,
            "yellow_fallback_min_pixels": 3000,
            "use_heading_error": False,
            "heading_gain": 0.20,
            "linear_speed": 0.04,
            "min_linear_speed": 0.02,
            "kp": 0.22,
            "kd": 0.12,
            "max_angular": 0.25,
            "smoothing_alpha": 0.20,
            "error_deadband": 0.03,
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
            "debug_info_panel_width": 300,
            "debug_side_panel": True,
            "debug_path_interpolate": True,
            "debug_path_interpolate_count": 40,
            "debug_yellow_overlay_alpha": 0.18,
            "debug_green_overlay_alpha": 0.10,
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
        self.last_good_path: List[Tuple[float, float]] = []
        self.last_good_roi_shape: Optional[Tuple[int, int]] = None
        self.history_age = 0
        self.yellow_fallback_frames = 0
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
            "mask_open_kernel_size",
            "mask_close_kernel_size",
            "boundary_contact_kernel_size",
            "boundary_component_min_area",
            "boundary_y_bucket_px",
            "boundary_min_bucket_points",
            "boundary_max_x_jump_px",
            "boundary_min_points",
            "min_boundary_gap_px",
            "max_boundary_jump_px",
            "expected_lane_width_px",
            "min_left_right_visible_rows",
            "single_boundary_max_frames",
            "single_boundary_offset_px",
            "single_boundary_min_points",
            "single_boundary_min_y_span_px",
            "single_boundary_sample_count",
            "single_boundary_normal_sample_px",
            "single_boundary_yellow_vote_radius",
            "history_max_frames",
            "yellow_fallback_max_frames",
            "yellow_fallback_min_pixels",
            "debug_info_panel_width",
            "debug_path_interpolate_count",
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
            "heading_gain",
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
        for name in (
            "boundary_loss_slowdown",
            "both_boundary_speed_scale",
            "single_boundary_speed_scale",
            "history_speed_scale",
            "history_confidence_decay",
            "yellow_fallback_speed_scale",
            "boundary_keep_component_ratio",
        ):
            if not 0.0 <= float(p[name]) <= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0]")
        if not (
            0.0 <= float(p["single_boundary_y_min_ratio"]) < 1.0
            and 0.0 < float(p["single_boundary_y_max_ratio"]) <= 1.0
            and float(p["single_boundary_y_min_ratio"])
            < float(p["single_boundary_y_max_ratio"])
        ):
            raise ValueError(
                "single-boundary y ratios must satisfy 0 <= min < max <= 1"
            )
        if int(p["boundary_dilate_iterations"]) < 0:
            raise ValueError(
                "boundary_dilate_iterations must not be negative"
            )
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
        for name in (
            "debug_yellow_overlay_alpha",
            "debug_green_overlay_alpha",
        ):
            if not 0.0 <= float(p[name]) <= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0]")
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
            f"debug_side_panel={p['debug_side_panel']}, "
            f"debug_info_panel_width={p['debug_info_panel_width']}, "
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
            result = extract_boundary_path(bgr, p)
            result = self._apply_path_priority(result, bgr.shape[:2])
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

    def _apply_path_priority(
        self,
        observation: VisionResult,
        image_shape: Tuple[int, int],
    ) -> VisionResult:
        """Apply boundary, history, yellow-fallback, then lost priority."""
        image_height, image_width = image_shape
        p = self.params
        roi_y_start = int(image_height * float(p["roi_y_start_ratio"]))
        roi_y_start = min(max(roi_y_start, 0), image_height - 1)
        roi_y_end = int(image_height * float(p["roi_y_end_ratio"]))
        roi_y_end = min(max(roi_y_end, roi_y_start + 1), image_height)
        roi_shape = (roi_y_end - roi_y_start, image_width)

        if observation.path_mode in (
            "both_boundary_center",
            "single_boundary_offset",
        ):
            self.last_good_path = list(observation.target_path_points)
            self.last_good_roi_shape = roi_shape
            self.history_age = 0
            self.yellow_fallback_frames = 0
            observation.history_age = 0
            observation.yellow_fallback_frames = 0
            return observation

        history_available = (
            bool(p["history_enable"])
            and bool(self.last_good_path)
            and self.last_good_roi_shape == roi_shape
            and self.history_age < int(p["history_max_frames"])
        )
        if history_available:
            self.history_age += 1
            self.yellow_fallback_frames = 0
            history_result = _make_v4_result(
                "history_prediction",
                list(self.last_good_path),
                observation.yellow_pixels,
                observation.green_pixels,
                observation.yellow_mask,
                observation.green_mask,
                observation.boundary_mask,
                observation.boundary_sequences,
                observation.left_boundary_points,
                observation.right_boundary_points,
                observation.single_boundary_points,
                len(self.last_good_path),
                observation.left_visible_count,
                observation.right_visible_count,
                observation.both_visible_count,
                None,
                image_width,
                image_height,
                roi_y_start,
                roi_y_end,
                p,
            )
            history_result.history_age = self.history_age
            return history_result

        if observation.path_mode == "yellow_area_center_fallback":
            self.yellow_fallback_frames += 1
            observation.history_age = self.history_age
            observation.yellow_fallback_frames = self.yellow_fallback_frames
            if self.yellow_fallback_frames <= int(
                p["yellow_fallback_max_frames"]
            ):
                return observation
        else:
            self.yellow_fallback_frames = 0

        lost_result = _make_v4_result(
            "lost_stop",
            [],
            observation.yellow_pixels,
            observation.green_pixels,
            observation.yellow_mask,
            observation.green_mask,
            observation.boundary_mask,
            observation.boundary_sequences,
            observation.left_boundary_points,
            observation.right_boundary_points,
            observation.single_boundary_points,
            0,
            observation.left_visible_count,
            observation.right_visible_count,
            observation.both_visible_count,
            None,
            image_width,
            image_height,
            roi_y_start,
            roi_y_end,
            p,
        )
        lost_result.history_age = self.history_age
        lost_result.yellow_fallback_frames = self.yellow_fallback_frames
        return lost_result

    def _handle_valid_result(
        self,
        bgr: np.ndarray,
        source_msg: Image,
        result: VisionResult,
    ) -> None:
        p = self.params
        self.lost_frames = 0
        if result.path_mode == "both_boundary_center":
            speed_scale = float(p["both_boundary_speed_scale"])
            self.single_boundary_frames = 0
        elif result.path_mode == "single_boundary_offset":
            speed_scale = float(p["single_boundary_speed_scale"])
            self.single_boundary_frames += 1
        elif result.path_mode == "history_prediction":
            speed_scale = float(p["history_speed_scale"]) * (
                float(p["history_confidence_decay"]) ** result.history_age
            )
            self.single_boundary_frames = 0
        elif result.path_mode == "yellow_area_center_fallback":
            speed_scale = float(p["yellow_fallback_speed_scale"])
            self.single_boundary_frames = 0
        else:
            speed_scale = 0.0

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
            f"history_age={result.history_age}, "
            f"yellow_fallback_frames={result.yellow_fallback_frames}, "
            f"path_mode={result.path_mode}"
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
            if result.path_mode == "both_boundary_center":
                speed_scale = float(p["both_boundary_speed_scale"])
            elif result.path_mode == "single_boundary_offset":
                speed_scale = float(p["single_boundary_speed_scale"])
            elif result.path_mode == "history_prediction":
                speed_scale = float(p["history_speed_scale"]) * (
                    float(p["history_confidence_decay"])
                    ** result.history_age
                )
            elif result.path_mode == "yellow_area_center_fallback":
                speed_scale = float(p["yellow_fallback_speed_scale"])
            else:
                speed_scale = 0.0
            debug = render_debug_image(
                bgr,
                result,
                linear,
                angular,
                speed_scale,
                self.lost_frames,
                p,
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
