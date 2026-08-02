#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第五阶段：车道路径控制计算与中间命令节点。

运行：
  source /opt/tros/humble/setup.bash
  source /root/intelligent_car_ws/install/setup.bash
  python3 /root/intelligent_car_ws/src/gxb_test/5_lane_path_controller.py \
    --ros-args -p dry_run:=true

查看：
  ros2 topic echo /gxb_test/controller/status
  ros2 topic echo /gxb_test/lane_control_cmd

本节点订阅感知 Path/status，发布 JSON 诊断和 JSON 中间控制命令。它不导入
底盘速度消息、不发布 /cmd_vel，也不直接控制车辆；只有 lane_control_gate.py
可以把通过安全检查的中间命令转换为底盘 Twist。
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path as FilePath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SELF_TEST = "--self-test" in sys.argv

if not SELF_TEST:
    import rclpy
    from nav_msgs.msg import Path as RosPath
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from std_msgs.msg import String
else:
    # 自测环境无需 ROS 安装；这些桩只覆盖纯算法测试需要的最小接口。
    class _Policy:
        BEST_EFFORT = "best_effort"
        VOLATILE = "volatile"
        KEEP_LAST = "keep_last"

    ReliabilityPolicy = DurabilityPolicy = HistoryPolicy = _Policy

    class QoSProfile:
        def __init__(
            self,
            *,
            reliability: Any,
            durability: Any,
            history: Any,
            depth: int,
        ) -> None:
            self.reliability = reliability
            self.durability = durability
            self.history = history
            self.depth = depth

    class Node:  # pragma: no cover - ROS node is not instantiated by self-test.
        pass

    class RosPath:
        pass

    class String:
        def __init__(self) -> None:
            self.data = ""


PREFERRED_MODES = frozenset(
    {
        "green_dual_inner_edge",
        "yellow_corridor_dual_edge",
        "yellow_corridor_center_gap_filled",
    }
)
DEGRADED_MODES = frozenset(
    {
        "green_yellow_hybrid",
        # Two independently fitted physical boundaries remain dual-source,
        # but this fallback is intentionally capped at degraded quality.
        "dual_boundary_midpoint",
    }
)
RECOVERY_MODES = frozenset({"single_green_width_offset"})
UNSUPPORTED_SINGLE_MODES = frozenset({"single_boundary_normal_offset"})


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def to_json_safe(value: Any) -> Any:
    """Recursively convert NumPy values into strict JSON-compatible values."""
    if isinstance(value, np.generic):
        return to_json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [to_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {
            str(key): to_json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    raise TypeError(
        f"unsupported JSON value type: {type(value).__name__}"
    )


def serialize_json(value: Any) -> str:
    """The single strict JSON serialization boundary used by this file."""
    return json.dumps(
        to_json_safe(value),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def angle_difference(target: float, reference: float) -> float:
    return normalize_angle(target - reference)


def make_observation_qos() -> QoSProfile:
    """Match the integrated pipeline's BEST_EFFORT depth-one publishers."""
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def enforce_dry_run(requested: bool) -> bool:
    """This version has no control-output path, regardless of the parameter."""
    _ = requested
    return True


@dataclass
class ControllerConfig:
    path_topic: str = "/gxb_test/pipeline/centerline_path"
    status_topic: str = "/gxb_test/pipeline/status"
    controller_status_topic: str = "/gxb_test/controller/status"
    control_cmd_topic: str = "/gxb_test/lane_control_cmd"
    dry_run: bool = True
    log_rate_hz: float = 2.0
    status_publish_rate_hz: float = 5.0
    min_path_points: int = 3
    min_path_span_m: float = 0.10
    path_timeout_sec: float = 0.40
    status_timeout_sec: float = 1.00
    max_point_lateral_jump_m: float = 0.15
    min_centerline_confidence: float = 0.30
    min_centerline_span_m: float = 0.10
    min_process_fps: float = 4.0
    max_perception_age_ms: float = 350.0
    near_error_distance_m: float = 0.35
    lookahead_distance_m: float = 0.45
    heading_fit_start_m: float = 0.30
    heading_fit_end_m: float = 0.55
    min_heading_segment_dx_m: float = 0.025
    heading_outlier_mad_scale: float = 3.0
    heading_outlier_min_threshold_deg: float = 3.0
    max_heading_estimator_disagreement_deg: float = 8.0
    max_heading_chord_disagreement_deg: float = 8.0
    heading_filter_time_constant_sec: float = 0.25
    heading_state_reset_timeout_sec: float = 0.80
    max_heading_rate_deg_per_sec: float = 60.0
    lateral_gain: float = 1.8
    heading_gain: float = 1.2
    max_suggested_angular_z: float = 0.60
    nominal_suggested_linear_x: float = 0.50
    normal_min_path_points: int = 10
    normal_min_path_span_m: float = 0.45
    normal_min_confidence: float = 0.80
    normal_suggested_linear_x: float = 0.50
    normal_max_angular_z: float = 0.60
    degraded_min_path_points: int = 6
    degraded_min_path_span_m: float = 0.25
    degraded_min_confidence: float = 0.75
    degraded_suggested_linear_x: float = 0.35
    short_path_suggested_linear_x: float = 0.20
    degraded_max_angular_z: float = 0.35
    recovery_min_path_points: int = 6
    recovery_min_path_span_m: float = 0.25
    recovery_min_confidence: float = 0.30

    # 急弯单绿救车档：只允许 single_green_width_offset 使用。
    # 目的不是正常巡航，而是极低速继续改变车头姿态，
    # 直到重新获得 >=6点/0.25m 的普通 recovery 路径。
    sharp_recovery_min_path_points: int = 3
    sharp_recovery_min_path_span_m: float = 0.10
    sharp_recovery_suggested_linear_x: float = 0.08
    sharp_recovery_max_angular_z: float = 0.30

    # 3~5点急弯路径天然比普通 recovery 抖动更大。
    # 保留方向一致性和几何一致性，但使用独立稳定窗口。
    sharp_recovery_required_stable_frames: int = 2
    sharp_recovery_max_heading_std_deg: float = 8.0
    sharp_recovery_max_target_y_delta_m: float = 0.05
    sharp_recovery_max_lateral_delta_m: float = 0.05

    recovery_required_stable_frames: int = 4
    recovery_suggested_linear_x: float = 0.32
    recovery_max_angular_z: float = 0.30
    recovery_max_heading_std_deg: float = 4.0
    recovery_max_target_y_delta_m: float = 0.025
    recovery_max_lateral_delta_m: float = 0.025
    recovery_max_direction_flips: int = 0
    recovery_history_size: int = 5
    max_angular_accel_rad_s2: float = 0.60
    max_angular_decel_rad_s2: float = 1.00
    direction_flip_deadband_rad_s: float = 0.03
    direction_flip_window_size: int = 6
    max_direction_flips_in_window: int = 1

    def validate(self) -> None:
        if self.min_path_points < 3:
            raise ValueError("min_path_points must be at least 3")
        positive = {
            "min_path_span_m": self.min_path_span_m,
            "path_timeout_sec": self.path_timeout_sec,
            "status_timeout_sec": self.status_timeout_sec,
            "max_point_lateral_jump_m": self.max_point_lateral_jump_m,
            "near_error_distance_m": self.near_error_distance_m,
            "lookahead_distance_m": self.lookahead_distance_m,
            "max_suggested_angular_z": self.max_suggested_angular_z,
            "min_heading_segment_dx_m": self.min_heading_segment_dx_m,
            "heading_filter_time_constant_sec": (
                self.heading_filter_time_constant_sec
            ),
            "heading_state_reset_timeout_sec": (
                self.heading_state_reset_timeout_sec
            ),
            "max_heading_rate_deg_per_sec": (
                self.max_heading_rate_deg_per_sec
            ),
            "max_angular_accel_rad_s2": self.max_angular_accel_rad_s2,
            "max_angular_decel_rad_s2": self.max_angular_decel_rad_s2,
            "normal_min_path_span_m": self.normal_min_path_span_m,
            "degraded_min_path_span_m": self.degraded_min_path_span_m,
            "recovery_min_path_span_m": self.recovery_min_path_span_m,
            "sharp_recovery_min_path_span_m": (
                self.sharp_recovery_min_path_span_m
            ),
            "sharp_recovery_max_angular_z": (
                self.sharp_recovery_max_angular_z
            ),
            "sharp_recovery_max_heading_std_deg": (
                self.sharp_recovery_max_heading_std_deg
            ),
            "sharp_recovery_max_target_y_delta_m": (
                self.sharp_recovery_max_target_y_delta_m
            ),
            "sharp_recovery_max_lateral_delta_m": (
                self.sharp_recovery_max_lateral_delta_m
            ),
            "normal_max_angular_z": self.normal_max_angular_z,
            "degraded_max_angular_z": self.degraded_max_angular_z,
            "recovery_max_angular_z": self.recovery_max_angular_z,
            "heading_outlier_mad_scale": self.heading_outlier_mad_scale,
            "heading_outlier_min_threshold_deg": (
                self.heading_outlier_min_threshold_deg
            ),
            "max_heading_estimator_disagreement_deg": (
                self.max_heading_estimator_disagreement_deg
            ),
            "max_heading_chord_disagreement_deg": (
                self.max_heading_chord_disagreement_deg
            ),
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.heading_fit_end_m <= self.heading_fit_start_m:
            raise ValueError("heading fit end must exceed start")
        if not 0.0 <= self.min_centerline_confidence <= 1.0:
            raise ValueError("min_centerline_confidence must be in [0, 1]")
        for name in (
            "normal_min_confidence",
            "degraded_min_confidence",
            "recovery_min_confidence",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "normal_min_path_points",
            "degraded_min_path_points",
            "recovery_min_path_points",
            "sharp_recovery_min_path_points",
        ):
            if int(getattr(self, name)) < self.min_path_points:
                raise ValueError(
                    f"{name} cannot be below min_path_points"
                )
        self.log_rate_hz = max(0.1, finite_float(self.log_rate_hz, 2.0))
        self.status_publish_rate_hz = max(
            0.1, finite_float(self.status_publish_rate_hz, 5.0)
        )
        self.nominal_suggested_linear_x = max(
            0.0, finite_float(self.nominal_suggested_linear_x)
        )
        self.normal_suggested_linear_x = max(
            0.0, finite_float(self.normal_suggested_linear_x)
        )
        self.degraded_suggested_linear_x = max(
            0.0, finite_float(self.degraded_suggested_linear_x)
        )
        self.short_path_suggested_linear_x = max(
            0.0, finite_float(self.short_path_suggested_linear_x)
        )
        self.recovery_suggested_linear_x = max(
            0.0, finite_float(self.recovery_suggested_linear_x)
        )
        self.sharp_recovery_suggested_linear_x = max(
            0.0, finite_float(self.sharp_recovery_suggested_linear_x)
        )
        self.recovery_required_stable_frames = max(
            1, int(self.recovery_required_stable_frames)
        )
        self.sharp_recovery_required_stable_frames = max(
            1, int(self.sharp_recovery_required_stable_frames)
        )
        self.recovery_history_size = max(
            self.recovery_required_stable_frames,
            self.sharp_recovery_required_stable_frames,
            int(self.recovery_history_size),
        )
        self.direction_flip_window_size = max(
            2, int(self.direction_flip_window_size)
        )
        self.recovery_max_direction_flips = max(
            0, int(self.recovery_max_direction_flips)
        )
        self.max_direction_flips_in_window = max(
            0, int(self.max_direction_flips_in_window)
        )
        self.dry_run = enforce_dry_run(self.dry_run)
        assert self.dry_run is True


@dataclass
class PerceptionSnapshot:
    received: bool = False
    received_monotonic: float = 0.0
    parse_error: str = ""
    centerline_valid: bool = False
    centerline_mode: str = ""
    centerline_confidence: float = 0.0
    centerline_forward_span_m: float = 0.0
    fallback_boundary_pipeline_used: bool = False
    green_boundary_fast_path_used: bool = False
    processing_time_ms: float = 0.0
    capture_to_process_age_ms: float = 0.0
    process_fps: float = 0.0
    path_publish_fps: float = 0.0

    @classmethod
    def from_document(
        cls, document: Dict[str, Any], received_monotonic: float
    ) -> "PerceptionSnapshot":
        if not isinstance(document, dict):
            raise ValueError("status JSON root must be an object")
        return cls(
            received=True,
            received_monotonic=received_monotonic,
            centerline_valid=bool(document.get("centerline_valid", False)),
            centerline_mode=str(document.get("centerline_mode", "")).strip(),
            centerline_confidence=clamp(
                finite_float(document.get("centerline_confidence")), 0.0, 1.0
            ),
            centerline_forward_span_m=max(
                0.0, finite_float(document.get("centerline_forward_span_m"))
            ),
            fallback_boundary_pipeline_used=bool(
                document.get("fallback_boundary_pipeline_used", False)
            ),
            green_boundary_fast_path_used=bool(
                document.get("green_boundary_fast_path_used", False)
            ),
            processing_time_ms=max(
                0.0, finite_float(document.get("processing_time_ms"))
            ),
            capture_to_process_age_ms=max(
                0.0, finite_float(document.get("capture_to_process_age_ms"))
            ),
            process_fps=max(0.0, finite_float(document.get("process_fps"))),
            path_publish_fps=max(
                0.0, finite_float(document.get("path_publish_fps"))
            ),
        )


@dataclass
class HeadingEstimate:
    valid: bool = False
    reason: str = "heading_estimation_failed"
    method: str = "segment_mad_weighted_mean"
    fit_point_count: int = 0
    segment_count_raw: int = 0
    segment_count_used: int = 0
    segment_median_rad: float = 0.0
    segment_mad_rad: float = 0.0
    ols_slope: float = 0.0
    ols_rad: float = 0.0
    robust_rad: float = 0.0
    chord_rad: float = 0.0
    estimator_disagreement: bool = False
    estimator_disagreement_deg: float = 0.0
    chord_disagreement: bool = False
    chord_disagreement_deg: float = 0.0


@dataclass
class PathAnalysis:
    valid: bool = False
    reason: str = "path_not_received"
    frame_id: str = ""
    point_count: int = 0
    span_m: float = 0.0
    points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64),
        repr=False,
    )
    near_error_distance_m: float = 0.0
    lateral_error_m: float = 0.0
    near_error_clamped: bool = False
    lookahead_distance_m: float = 0.0
    lookahead_target_x_m: float = 0.0
    lookahead_target_y_m: float = 0.0
    lookahead_target_distance_m: float = 0.0
    lookahead_clamped: bool = False
    heading_fit_point_count: int = 0
    heading_slope: float = 0.0
    heading_error_rad: float = 0.0
    heading_method: str = ""
    heading_segment_count_raw: int = 0
    heading_segment_count_used: int = 0
    heading_segment_median_rad: float = 0.0
    heading_segment_mad_rad: float = 0.0
    heading_ols_rad: float = 0.0
    heading_robust_rad: float = 0.0
    heading_chord_rad: float = 0.0
    heading_estimator_disagreement: bool = False
    heading_estimator_disagreement_deg: float = 0.0
    heading_chord_disagreement: bool = False
    heading_chord_disagreement_deg: float = 0.0


@dataclass
class ControllerRuntimeState:
    last_heading_error_rad: Optional[float] = None
    last_heading_timestamp: float = 0.0
    last_suggested_angular_z: float = 0.0
    last_angular_timestamp: float = 0.0
    last_centerline_mode: str = ""
    last_path_span_m: float = 0.0
    recovery_stable_frame_count: int = 0
    recovery_heading_history: List[float] = field(default_factory=list)
    recovery_target_y_history: List[float] = field(default_factory=list)
    recovery_lateral_history: List[float] = field(default_factory=list)
    angular_direction_history: List[int] = field(default_factory=list)

    def reset(self) -> None:
        self.last_heading_error_rad = None
        self.last_heading_timestamp = 0.0
        self.last_suggested_angular_z = 0.0
        self.last_angular_timestamp = 0.0
        self.last_centerline_mode = ""
        self.last_path_span_m = 0.0
        self.reset_recovery()
        self.angular_direction_history.clear()

    def reset_recovery(self) -> None:
        self.recovery_stable_frame_count = 0
        self.recovery_heading_history.clear()
        self.recovery_target_y_history.clear()
        self.recovery_lateral_history.clear()


@dataclass
class ControllerPreview:
    timestamp: str = ""
    controller_ready: bool = False
    control_block_reason: str = "waiting_for_path"
    degraded_mode: bool = False
    control_quality_level: str = "blocked"
    mode_policy_reason: str = "waiting_for_path"
    normal_mode_active: bool = False
    degraded_mode_active: bool = False
    recovery_mode_active: bool = False
    path_received: bool = False
    status_received: bool = False
    path_frame_id: str = ""
    path_point_count: int = 0
    path_span_m: float = 0.0
    path_age_ms: float = -1.0
    status_age_ms: float = -1.0
    centerline_valid: bool = False
    centerline_mode: str = ""
    centerline_confidence: float = 0.0
    fallback_boundary_pipeline_used: bool = False
    green_boundary_fast_path_used: bool = False
    processing_time_ms: float = 0.0
    capture_to_process_age_ms: float = 0.0
    process_fps: float = 0.0
    path_publish_fps: float = 0.0
    near_error_distance_m: float = 0.0
    lateral_error_m: float = 0.0
    lateral_error_abs_m: float = 0.0
    near_error_clamped: bool = False
    lookahead_distance_m: float = 0.0
    lookahead_target_x_m: float = 0.0
    lookahead_target_y_m: float = 0.0
    lookahead_target_distance_m: float = 0.0
    lookahead_clamped: bool = False
    heading_fit_point_count: int = 0
    heading_slope: float = 0.0
    heading_method: str = ""
    heading_segment_count_raw: int = 0
    heading_segment_count_used: int = 0
    heading_segment_median_rad: float = 0.0
    heading_segment_median_deg: float = 0.0
    heading_segment_mad_rad: float = 0.0
    heading_segment_mad_deg: float = 0.0
    heading_ols_rad: float = 0.0
    heading_ols_deg: float = 0.0
    heading_robust_rad: float = 0.0
    heading_robust_deg: float = 0.0
    heading_chord_rad: float = 0.0
    heading_chord_deg: float = 0.0
    heading_estimator_disagreement: bool = False
    heading_estimator_disagreement_deg: float = 0.0
    heading_chord_disagreement: bool = False
    heading_chord_disagreement_deg: float = 0.0
    heading_error_raw_rad: float = 0.0
    heading_error_raw_deg: float = 0.0
    heading_error_filtered_rad: float = 0.0
    heading_error_filtered_deg: float = 0.0
    heading_temporal_filter_used: bool = False
    heading_rate_limited: bool = False
    heading_dt_sec: float = 0.0
    heading_error_rad: float = 0.0
    heading_error_deg: float = 0.0
    recovery_stable_frame_count: int = 0
    recovery_required_stable_frames: int = 0
    recovery_history_size: int = 0
    recovery_heading_std_deg: float = 0.0
    recovery_target_y_delta_m: float = 0.0
    recovery_lateral_delta_m: float = 0.0
    suggested_linear_x: float = 0.0
    suggested_angular_z_raw: float = 0.0
    suggested_angular_z_unfiltered: float = 0.0
    suggested_angular_z_mode_limited: float = 0.0
    suggested_angular_z_rate_limited: float = 0.0
    suggested_angular_z: float = 0.0
    angular_command_saturated: bool = False
    angular_rate_limited: bool = False
    angular_delta_raw: float = 0.0
    angular_delta_limited: float = 0.0
    angular_dt_sec: float = 0.0
    previous_preview_angular_z: float = 0.0
    angular_direction_sign: int = 0
    angular_direction_flip_count: int = 0
    angular_direction_unstable: bool = False
    controller_confidence: float = 0.0
    update_rate_hz: float = 0.0
    calculation_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_control_command(self) -> Dict[str, Any]:
        """Return the stable JSON interface consumed by the safety gate."""
        return {
            "schema_version": 1,
            "interface_version": "gxb_lane_control_v1",
            "timestamp": self.timestamp,
            "linear_x": self.suggested_linear_x,
            "angular_z": self.suggested_angular_z,
            "ready": self.controller_ready,
            "quality": self.control_quality_level,
            "mode": self.centerline_mode,
            "lateral_error": self.lateral_error_m,
            "heading_error": self.heading_error_rad,
            "controller_confidence": self.controller_confidence,
            "reason": self.control_block_reason,
            "path_valid": bool(
                self.path_received
                and self.path_point_count > 0
                and self.path_span_m > 0.0
            ),
            "path_age_ms": self.path_age_ms,
            "pipeline_age_ms": self.status_age_ms,
            "path_point_count": self.path_point_count,
            "path_span_m": self.path_span_m,
            "process_fps": self.process_fps,
            "processing_time_ms": self.processing_time_ms,
        }


class RateMeter:
    def __init__(self, window_sec: float = 2.0) -> None:
        self.window_sec = window_sec
        self.samples: List[float] = []

    def tick(self, now: Optional[float] = None) -> float:
        instant = time.monotonic() if now is None else now
        self.samples.append(instant)
        cutoff = instant - self.window_sec
        self.samples = [item for item in self.samples if item >= cutoff]
        return self.value()

    def value(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        elapsed = self.samples[-1] - self.samples[0]
        return (len(self.samples) - 1) / elapsed if elapsed > 1.0e-9 else 0.0


def interpolate_y_at_forward(
    points: np.ndarray, target_x: float
) -> Tuple[float, float, bool]:
    """Return used x, interpolated y, and whether the target was clamped."""
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(values) == 0:
        raise ValueError("empty path")
    if target_x <= values[0, 0]:
        return (
            float(values[0, 0]),
            float(values[0, 1]),
            bool(target_x < values[0, 0]),
        )
    if target_x >= values[-1, 0]:
        return (
            float(values[-1, 0]),
            float(values[-1, 1]),
            bool(target_x > values[-1, 0]),
        )
    upper = int(np.searchsorted(values[:, 0], target_x, side="left"))
    lower = max(0, upper - 1)
    x0, y0 = values[lower]
    x1, y1 = values[upper]
    if x1 <= x0:
        raise ValueError("non-increasing interpolation segment")
    ratio = (target_x - x0) / (x1 - x0)
    return float(target_x), float(y0 + ratio * (y1 - y0)), False


def select_lookahead_target(
    points: np.ndarray, lookahead_distance_m: float
) -> Tuple[float, float, float, bool]:
    x_value, y_value, clamped = interpolate_y_at_forward(
        points, lookahead_distance_m
    )
    return x_value, y_value, math.hypot(x_value, y_value), clamped


def estimate_heading(
    points: np.ndarray,
    config: ControllerConfig,
    near_point: Tuple[float, float],
    lookahead_point: Tuple[float, float],
) -> HeadingEstimate:
    """Estimate heading from robust local segments, retaining OLS diagnostics."""
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    result = HeadingEstimate()
    finite = np.all(np.isfinite(values), axis=1)
    values = values[finite]
    if len(values) < 3:
        result.reason = "heading_too_few_finite_points"
        return result
    selected = values[
        (values[:, 0] >= config.heading_fit_start_m)
        & (values[:, 0] <= config.heading_fit_end_m)
    ]
    result.fit_point_count = int(len(selected))
    if len(selected) < 3:
        result.reason = "heading_fit_interval_too_few_points"
        return result

    delta = np.diff(selected, axis=0)
    valid_segments = (
        np.isfinite(delta[:, 0])
        & np.isfinite(delta[:, 1])
        & (delta[:, 0] >= config.min_heading_segment_dx_m)
    )
    segment_delta = delta[valid_segments]
    if len(segment_delta) < 2:
        result.reason = "heading_too_few_valid_segments"
        return result
    headings = np.arctan2(segment_delta[:, 1], segment_delta[:, 0])
    weights = segment_delta[:, 0]
    result.segment_count_raw = int(len(headings))
    median = float(np.median(headings))
    deviations = np.array(
        [abs(angle_difference(float(item), median)) for item in headings],
        dtype=np.float64,
    )
    mad = float(np.median(deviations))
    threshold = max(
        math.radians(config.heading_outlier_min_threshold_deg),
        config.heading_outlier_mad_scale * mad,
    )
    keep = deviations <= threshold
    used = headings[keep]
    used_weights = weights[keep]
    result.segment_count_used = int(len(used))
    result.segment_median_rad = normalize_angle(median)
    result.segment_mad_rad = mad
    if len(used) < 2:
        result.reason = "heading_too_few_inlier_segments"
        return result

    sine = float(np.sum(np.sin(used) * used_weights))
    cosine = float(np.sum(np.cos(used) * used_weights))
    if abs(sine) + abs(cosine) <= 1.0e-12:
        result.reason = "heading_robust_mean_undefined"
        return result
    result.robust_rad = normalize_angle(math.atan2(sine, cosine))

    design = np.column_stack(
        (selected[:, 0], np.ones(len(selected), dtype=np.float64))
    )
    coefficients, _, _, _ = np.linalg.lstsq(
        design, selected[:, 1], rcond=None
    )
    result.ols_slope = float(coefficients[0])
    if not math.isfinite(result.ols_slope):
        result.reason = "heading_ols_not_finite"
        return result
    result.ols_rad = normalize_angle(math.atan(result.ols_slope))

    chord_dx = float(lookahead_point[0] - near_point[0])
    chord_dy = float(lookahead_point[1] - near_point[1])
    if chord_dx < config.min_heading_segment_dx_m:
        result.reason = "heading_chord_too_short"
        return result
    result.chord_rad = normalize_angle(math.atan2(chord_dy, chord_dx))
    result.estimator_disagreement_deg = math.degrees(
        abs(angle_difference(result.ols_rad, result.robust_rad))
    )
    result.estimator_disagreement = bool(
        result.estimator_disagreement_deg
        > config.max_heading_estimator_disagreement_deg
    )
    result.chord_disagreement_deg = math.degrees(
        abs(angle_difference(result.chord_rad, result.robust_rad))
    )
    result.chord_disagreement = bool(
        result.chord_disagreement_deg
        > config.max_heading_chord_disagreement_deg
    )
    result.valid = True
    result.reason = ""
    return result


def validate_path(
    frame_id: str, points: Sequence[Sequence[float]], config: ControllerConfig
) -> PathAnalysis:
    analysis = PathAnalysis(
        frame_id=str(frame_id),
        near_error_distance_m=config.near_error_distance_m,
        lookahead_distance_m=config.lookahead_distance_m,
    )
    try:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        analysis.points = values
        analysis.point_count = len(values)
        if frame_id != "base_link":
            analysis.reason = "path_frame_mismatch"
            return analysis
        if len(values) < config.min_path_points:
            analysis.reason = "path_too_few_points"
            return analysis
        if not bool(np.all(np.isfinite(values))):
            analysis.reason = "path_non_finite"
            return analysis
        delta_x = np.diff(values[:, 0])
        if bool(np.any(delta_x <= 0.0)):
            analysis.reason = "path_forward_not_monotonic"
            return analysis
        if not bool(np.any(values[:, 0] > 0.0)):
            analysis.reason = "path_no_positive_forward_point"
            return analysis
        analysis.span_m = float(values[-1, 0] - values[0, 0])
        if analysis.span_m + 1.0e-9 < config.min_path_span_m:
            analysis.reason = "path_span_too_short"
            return analysis
        if bool(
            np.any(
                np.abs(np.diff(values[:, 1]))
                > config.max_point_lateral_jump_m
            )
        ):
            analysis.reason = "path_lateral_jump"
            return analysis

        (
            near_x,
            analysis.lateral_error_m,
            analysis.near_error_clamped,
        ) = interpolate_y_at_forward(values, config.near_error_distance_m)
        (
            analysis.lookahead_target_x_m,
            analysis.lookahead_target_y_m,
            analysis.lookahead_target_distance_m,
            analysis.lookahead_clamped,
        ) = select_lookahead_target(values, config.lookahead_distance_m)
        heading = estimate_heading(
            values,
            config,
            (near_x, analysis.lateral_error_m),
            (
                analysis.lookahead_target_x_m,
                analysis.lookahead_target_y_m,
            ),
        )
        if not heading.valid:
            analysis.reason = "heading_estimation_failed"
            analysis.heading_method = heading.reason
            return analysis
        analysis.heading_fit_point_count = heading.fit_point_count
        analysis.heading_slope = heading.ols_slope
        analysis.heading_error_rad = heading.robust_rad
        analysis.heading_method = heading.method
        analysis.heading_segment_count_raw = heading.segment_count_raw
        analysis.heading_segment_count_used = heading.segment_count_used
        analysis.heading_segment_median_rad = heading.segment_median_rad
        analysis.heading_segment_mad_rad = heading.segment_mad_rad
        analysis.heading_ols_rad = heading.ols_rad
        analysis.heading_robust_rad = heading.robust_rad
        analysis.heading_chord_rad = heading.chord_rad
        analysis.heading_estimator_disagreement = (
            heading.estimator_disagreement
        )
        analysis.heading_estimator_disagreement_deg = (
            heading.estimator_disagreement_deg
        )
        analysis.heading_chord_disagreement = heading.chord_disagreement
        analysis.heading_chord_disagreement_deg = (
            heading.chord_disagreement_deg
        )
        analysis.valid = True
        analysis.reason = ""
    except Exception as exc:
        analysis.valid = False
        analysis.reason = f"path_analysis_error:{type(exc).__name__}"
    return analysis


def basic_block_reason(
    analysis: PathAnalysis,
    perception: PerceptionSnapshot,
    path_age_sec: float,
    status_age_sec: float,
    config: ControllerConfig,
) -> str:
    if path_age_sec < 0.0:
        return "path_not_received"
    if path_age_sec > config.path_timeout_sec:
        return "path_timeout"
    if not analysis.valid:
        return analysis.reason or "path_invalid"
    if not perception.received:
        return "status_not_received"
    if perception.parse_error:
        return "status_json_invalid"
    if status_age_sec > config.status_timeout_sec:
        return "status_timeout"
    if not perception.centerline_valid:
        return "centerline_invalid"
    if perception.capture_to_process_age_ms > config.max_perception_age_ms:
        return "perception_age_too_high"
    if perception.process_fps < config.min_process_fps:
        return "process_fps_too_low"
    return ""


def evaluate_mode_policy(
    analysis: PathAnalysis,
    perception: PerceptionSnapshot,
    config: ControllerConfig,
) -> Tuple[str, str]:
    mode = perception.centerline_mode
    if mode in UNSUPPORTED_SINGLE_MODES:
        return "blocked", "unsupported_single_boundary_mode"
    if mode in PREFERRED_MODES:
        normal_requirements = (
            (
                analysis.point_count >= config.normal_min_path_points,
                "normal_path_points_low",
            ),
            (
                min(
                    analysis.span_m,
                    perception.centerline_forward_span_m,
                )
                + 1.0e-9
                >= config.normal_min_path_span_m,
                "normal_path_span_low",
            ),
            (
                perception.centerline_confidence
                >= config.normal_min_confidence,
                "normal_confidence_low",
            ),
        )
        normal_failure = next(
            (reason for passed, reason in normal_requirements if not passed),
            "",
        )
        if not normal_failure:
            return "normal", "normal_mode_accepted"
        short_path_requirements = (
            (
                analysis.point_count >= config.degraded_min_path_points,
                "normal_path_points_low",
            ),
            (
                min(
                    analysis.span_m,
                    perception.centerline_forward_span_m,
                )
                + 1.0e-9
                >= config.degraded_min_path_span_m,
                "normal_path_span_low",
            ),
            (
                perception.centerline_confidence
                >= config.degraded_min_confidence,
                "normal_confidence_low",
            ),
        )
        if all(passed for passed, _reason in short_path_requirements):
            return "degraded", "short_path_degraded"
        return "blocked", normal_failure
    elif mode in DEGRADED_MODES:
        level = "degraded"
        requirements = (
            (
                analysis.point_count >= config.degraded_min_path_points,
                "degraded_path_points_low",
            ),
            (
                min(
                    analysis.span_m,
                    perception.centerline_forward_span_m,
                )
                + 1.0e-9
                >= config.degraded_min_path_span_m,
                "degraded_path_span_low",
            ),
            (
                perception.centerline_confidence
                >= config.degraded_min_confidence,
                "degraded_confidence_low",
            ),
        )
    elif mode in RECOVERY_MODES:
        recovery_span = min(
            analysis.span_m,
            perception.centerline_forward_span_m,
        )

        standard_points_ok = (
            analysis.point_count >= config.recovery_min_path_points
        )
        standard_span_ok = (
            recovery_span + 1.0e-9
            >= config.recovery_min_path_span_m
        )
        confidence_ok = (
            perception.centerline_confidence
            >= config.recovery_min_confidence
        )

        # 普通 recovery 的几何条件已经满足时，
        # 不允许再降级到 sharp recovery 绕过其判据。
        if standard_points_ok and standard_span_ok:
            if not confidence_ok:
                return "blocked", "recovery_confidence_low"
            return "recovery", "recovery_mode_accepted"

        # sharp recovery 只解决 single-green 急弯中的
        # 短视野（点数/跨度不足），不用于绕过低置信度。
        if mode == "single_green_width_offset":
            if (
                analysis.point_count
                < config.sharp_recovery_min_path_points
            ):
                return "blocked", "sharp_recovery_path_points_low"

            if (
                recovery_span + 1.0e-9
                < config.sharp_recovery_min_path_span_m
            ):
                return "blocked", "sharp_recovery_path_span_low"

            if not confidence_ok:
                return "blocked", "sharp_recovery_confidence_low"

            return "recovery", "sharp_recovery"

        if not standard_points_ok:
            return "blocked", "recovery_path_points_low"

        return "blocked", "recovery_path_span_low"
    else:
        return "blocked", "centerline_mode_not_allowed"
    for passed, reason in requirements:
        if not passed:
            return "blocked", reason
    return level, f"{level}_mode_accepted"


def determine_block_reason(
    analysis: PathAnalysis,
    perception: PerceptionSnapshot,
    path_age_sec: float,
    status_age_sec: float,
    config: ControllerConfig,
) -> Tuple[str, bool]:
    reason = basic_block_reason(
        analysis, perception, path_age_sec, status_age_sec, config
    )
    if reason:
        return reason, False
    level, policy_reason = evaluate_mode_policy(analysis, perception, config)
    return (
        (policy_reason if level == "blocked" else ""),
        level in {"degraded", "recovery"},
    )


def filter_heading_temporal(
    raw_heading: float,
    now: float,
    mode: str,
    state: ControllerRuntimeState,
    config: ControllerConfig,
    advance_state: bool,
) -> Tuple[float, bool, bool, float]:
    previous = state.last_heading_error_rad
    dt = now - state.last_heading_timestamp
    reset = (
        previous is None
        or state.last_heading_timestamp <= 0.0
        or dt <= 0.0
        or dt > config.heading_state_reset_timeout_sec
    )
    if reset:
        filtered = normalize_angle(raw_heading)
        used = False
        rate_limited = False
        reported_dt = 0.0
    elif not advance_state:
        filtered = float(previous)
        used = True
        rate_limited = False
        reported_dt = 0.0
    else:
        alpha = dt / (config.heading_filter_time_constant_sec + dt)
        previous_rank = (
            0
            if state.last_centerline_mode in PREFERRED_MODES
            else 1
            if state.last_centerline_mode in DEGRADED_MODES
            else 2
        )
        current_rank = (
            0 if mode in PREFERRED_MODES else 1 if mode in DEGRADED_MODES else 2
        )
        if current_rank > previous_rank:
            alpha *= 0.5
        raw_delta = angle_difference(raw_heading, float(previous))
        candidate_delta = alpha * raw_delta
        maximum_delta = math.radians(
            config.max_heading_rate_deg_per_sec
        ) * dt
        rate_limited = abs(raw_delta) > maximum_delta
        candidate_delta = clamp(
            candidate_delta, -maximum_delta, maximum_delta
        )
        filtered = normalize_angle(float(previous) + candidate_delta)
        used = True
        reported_dt = dt
    if advance_state:
        state.last_heading_error_rad = float(filtered)
        state.last_heading_timestamp = float(now)
        state.last_centerline_mode = str(mode)
    return float(filtered), used, rate_limited, float(reported_dt)


def angular_slew_limit(
    target: float,
    now: float,
    state: ControllerRuntimeState,
    config: ControllerConfig,
    advance_state: bool,
) -> Tuple[float, bool, float, float, float, float]:
    previous = float(state.last_suggested_angular_z)
    dt = now - state.last_angular_timestamp
    if (
        state.last_angular_timestamp <= 0.0
        or dt <= 0.0
        or dt > config.heading_state_reset_timeout_sec
    ):
        limited = float(target)
        was_limited = False
        dt = 0.0
    elif not advance_state:
        limited = previous
        was_limited = False
        dt = 0.0
    else:
        raw_delta = target - previous
        if target * previous < 0.0:
            time_to_zero = (
                abs(previous) / config.max_angular_decel_rad_s2
            )
            if dt <= time_to_zero:
                limited = math.copysign(
                    max(
                        0.0,
                        abs(previous)
                        - config.max_angular_decel_rad_s2 * dt,
                    ),
                    previous,
                )
            else:
                remaining = dt - time_to_zero
                limited = math.copysign(
                    min(
                        abs(target),
                        config.max_angular_accel_rad_s2 * remaining,
                    ),
                    target,
                )
        else:
            increasing = abs(target) > abs(previous)
            rate = (
                config.max_angular_accel_rad_s2
                if increasing
                else config.max_angular_decel_rad_s2
            )
            limited_delta = clamp(raw_delta, -rate * dt, rate * dt)
            limited = previous + limited_delta
        limited_delta = limited - previous
        was_limited = not math.isclose(
            limited_delta, raw_delta, rel_tol=1.0e-9, abs_tol=1.0e-9
        )
    raw_delta = float(target - previous)
    limited_delta = float(limited - previous)
    if advance_state:
        state.last_suggested_angular_z = float(limited)
        state.last_angular_timestamp = float(now)
    return (
        float(limited),
        bool(was_limited),
        raw_delta,
        limited_delta,
        float(dt),
        previous,
    )


def angular_direction_sign(value: float, deadband: float) -> int:
    if abs(value) < deadband:
        return 0
    return 1 if value > 0.0 else -1


def update_direction_history(
    value: float,
    state: ControllerRuntimeState,
    config: ControllerConfig,
    advance_state: bool,
) -> Tuple[int, int]:
    sign = angular_direction_sign(
        value, config.direction_flip_deadband_rad_s
    )
    if advance_state and sign:
        state.angular_direction_history.append(sign)
        state.angular_direction_history = state.angular_direction_history[
            -config.direction_flip_window_size :
        ]
    history = state.angular_direction_history
    flips = sum(
        1
        for first, second in zip(history, history[1:])
        if first != second
    )
    return sign, int(flips)


def update_recovery_stability(
    analysis: PathAnalysis,
    heading_raw: float,
    heading_rate_limited: bool,
    direction_flip_count: int,
    state: ControllerRuntimeState,
    config: ControllerConfig,
    advance_state: bool,
    sharp_recovery: bool = False,
) -> Tuple[int, float, float, float, bool]:
    if advance_state:
        state.recovery_heading_history.append(float(heading_raw))
        state.recovery_target_y_history.append(
            float(analysis.lookahead_target_y_m)
        )
        state.recovery_lateral_history.append(
            float(analysis.lateral_error_m)
        )
        size = config.recovery_history_size
        state.recovery_heading_history = state.recovery_heading_history[-size:]
        state.recovery_target_y_history = (
            state.recovery_target_y_history[-size:]
        )
        state.recovery_lateral_history = state.recovery_lateral_history[-size:]
    headings = state.recovery_heading_history
    targets = state.recovery_target_y_history
    laterals = state.recovery_lateral_history
    heading_std = (
        math.degrees(float(np.std(headings))) if headings else 0.0
    )
    target_delta = max(targets) - min(targets) if targets else 0.0
    lateral_delta = max(laterals) - min(laterals) if laterals else 0.0

    if sharp_recovery:
        max_heading_std_deg = config.sharp_recovery_max_heading_std_deg
        max_target_y_delta_m = config.sharp_recovery_max_target_y_delta_m
        max_lateral_delta_m = config.sharp_recovery_max_lateral_delta_m

        # 急弯短 Path 的 raw heading 抖动较大。
        # 输出本身仍经过 heading rate limiter，因此不把
        # “rate limiter 正在工作”直接视为 sharp recovery 失败。
        heading_rate_ok = True
    else:
        max_heading_std_deg = config.recovery_max_heading_std_deg
        max_target_y_delta_m = config.recovery_max_target_y_delta_m
        max_lateral_delta_m = config.recovery_max_lateral_delta_m
        heading_rate_ok = not heading_rate_limited

    stable = bool(
        headings
        and math.isfinite(heading_raw)
        and heading_rate_ok
        and not analysis.heading_chord_disagreement
        and heading_std <= max_heading_std_deg
        and target_delta <= max_target_y_delta_m
        and lateral_delta <= max_lateral_delta_m
        and direction_flip_count <= config.recovery_max_direction_flips
    )
    if advance_state:
        if stable:
            state.recovery_stable_frame_count += 1
        else:
            state.reset_recovery()
    return (
        int(state.recovery_stable_frame_count),
        float(heading_std),
        float(target_delta),
        float(lateral_delta),
        stable,
    )


def compute_controller_confidence(
    analysis: PathAnalysis,
    perception: PerceptionSnapshot,
    preview: ControllerPreview,
    config: ControllerConfig,
) -> float:
    if not analysis.valid or not preview.controller_ready:
        return 0.0
    if preview.control_quality_level == "normal":
        point_min = config.normal_min_path_points
        span_min = config.normal_min_path_span_m
        cap = 1.0
    elif preview.control_quality_level == "degraded":
        point_min = config.degraded_min_path_points
        span_min = config.degraded_min_path_span_m
        cap = 0.70
    else:
        point_min = config.recovery_min_path_points
        span_min = config.recovery_min_path_span_m
        cap = 0.35
    confidence = perception.centerline_confidence
    confidence *= clamp(analysis.point_count / max(point_min * 1.5, 1.0), 0.6, 1.0)
    confidence *= clamp(analysis.span_m / max(span_min * 1.25, 1.0e-6), 0.6, 1.0)
    confidence *= clamp(analysis.heading_segment_count_used / 4.0, 0.65, 1.0)
    if analysis.lookahead_clamped:
        confidence *= 0.86
    if analysis.heading_estimator_disagreement:
        confidence *= 0.72
    if analysis.heading_chord_disagreement:
        confidence *= 0.72
    if preview.heading_rate_limited:
        confidence *= 0.78
    if preview.angular_rate_limited:
        confidence *= 0.88
    if preview.angular_direction_unstable:
        confidence *= 0.55
    return clamp(confidence, 0.0, cap)


def compute_preview_command(
    analysis: PathAnalysis,
    perception: PerceptionSnapshot,
    config: ControllerConfig,
    path_age_sec: float,
    status_age_sec: float,
    update_rate_hz: float = 0.0,
    runtime_state: Optional[ControllerRuntimeState] = None,
    now: Optional[float] = None,
    advance_state: bool = True,
) -> ControllerPreview:
    started = time.perf_counter()
    instant = time.monotonic() if now is None else float(now)
    state = runtime_state if runtime_state is not None else ControllerRuntimeState()
    reason = basic_block_reason(
        analysis, perception, path_age_sec, status_age_sec, config
    )
    level = "blocked"
    policy_reason = reason
    if not reason:
        level, policy_reason = evaluate_mode_policy(
            analysis, perception, config
        )
        if level == "blocked":
            reason = policy_reason
    if (
        not reason
        and level == "normal"
        and (
            analysis.heading_estimator_disagreement
            or analysis.heading_chord_disagreement
        )
    ):
        level = "degraded"
        policy_reason = "heading_diagnostic_disagreement"
    degraded = level in {"degraded", "recovery"}
    preview = ControllerPreview(
        timestamp=datetime.now(timezone.utc).isoformat(),
        controller_ready=bool(not reason),
        control_block_reason=reason,
        degraded_mode=bool(degraded),
        control_quality_level=level if not reason else "blocked",
        mode_policy_reason=policy_reason,
        normal_mode_active=bool(level == "normal"),
        degraded_mode_active=bool(level == "degraded"),
        recovery_mode_active=bool(level == "recovery"),
        path_received=bool(path_age_sec >= 0.0),
        status_received=bool(perception.received),
        path_frame_id=analysis.frame_id,
        path_point_count=analysis.point_count,
        path_span_m=analysis.span_m,
        path_age_ms=path_age_sec * 1000.0 if path_age_sec >= 0.0 else -1.0,
        status_age_ms=status_age_sec * 1000.0 if status_age_sec >= 0.0 else -1.0,
        centerline_valid=bool(perception.centerline_valid),
        centerline_mode=perception.centerline_mode,
        centerline_confidence=perception.centerline_confidence,
        fallback_boundary_pipeline_used=bool(
            perception.fallback_boundary_pipeline_used
        ),
        green_boundary_fast_path_used=bool(
            perception.green_boundary_fast_path_used
        ),
        processing_time_ms=perception.processing_time_ms,
        capture_to_process_age_ms=perception.capture_to_process_age_ms,
        process_fps=perception.process_fps,
        path_publish_fps=perception.path_publish_fps,
        near_error_distance_m=analysis.near_error_distance_m,
        lateral_error_m=analysis.lateral_error_m,
        lateral_error_abs_m=abs(analysis.lateral_error_m),
        near_error_clamped=bool(analysis.near_error_clamped),
        lookahead_distance_m=analysis.lookahead_distance_m,
        lookahead_target_x_m=analysis.lookahead_target_x_m,
        lookahead_target_y_m=analysis.lookahead_target_y_m,
        lookahead_target_distance_m=analysis.lookahead_target_distance_m,
        lookahead_clamped=bool(analysis.lookahead_clamped),
        heading_fit_point_count=analysis.heading_fit_point_count,
        heading_slope=analysis.heading_slope,
        heading_method=analysis.heading_method,
        heading_segment_count_raw=analysis.heading_segment_count_raw,
        heading_segment_count_used=analysis.heading_segment_count_used,
        heading_segment_median_rad=analysis.heading_segment_median_rad,
        heading_segment_median_deg=math.degrees(
            analysis.heading_segment_median_rad
        ),
        heading_segment_mad_rad=analysis.heading_segment_mad_rad,
        heading_segment_mad_deg=math.degrees(
            analysis.heading_segment_mad_rad
        ),
        heading_ols_rad=analysis.heading_ols_rad,
        heading_ols_deg=math.degrees(analysis.heading_ols_rad),
        heading_robust_rad=analysis.heading_robust_rad,
        heading_robust_deg=math.degrees(analysis.heading_robust_rad),
        heading_chord_rad=analysis.heading_chord_rad,
        heading_chord_deg=math.degrees(analysis.heading_chord_rad),
        heading_estimator_disagreement=bool(
            analysis.heading_estimator_disagreement
        ),
        heading_estimator_disagreement_deg=(
            analysis.heading_estimator_disagreement_deg
        ),
        heading_chord_disagreement=bool(
            analysis.heading_chord_disagreement
        ),
        heading_chord_disagreement_deg=(
            analysis.heading_chord_disagreement_deg
        ),
        heading_error_raw_rad=analysis.heading_robust_rad,
        heading_error_raw_deg=math.degrees(analysis.heading_robust_rad),
        heading_error_rad=analysis.heading_robust_rad,
        heading_error_deg=math.degrees(analysis.heading_robust_rad),
        recovery_stable_frame_count=state.recovery_stable_frame_count,
        recovery_required_stable_frames=(
            config.recovery_required_stable_frames
        ),
        recovery_history_size=config.recovery_history_size,
        update_rate_hz=max(0.0, update_rate_hz),
    )
    if reason:
        if advance_state:
            state.reset()
        preview.calculation_time_ms = (
            time.perf_counter() - started
        ) * 1000.0
        return preview

    (
        filtered_heading,
        heading_filter_used,
        heading_rate_limited,
        heading_dt,
    ) = filter_heading_temporal(
        analysis.heading_robust_rad,
        instant,
        perception.centerline_mode,
        state,
        config,
        advance_state,
    )
    preview.heading_error_filtered_rad = filtered_heading
    preview.heading_error_filtered_deg = math.degrees(filtered_heading)
    preview.heading_temporal_filter_used = bool(heading_filter_used)
    preview.heading_rate_limited = bool(heading_rate_limited)
    preview.heading_dt_sec = heading_dt
    preview.heading_error_rad = filtered_heading
    preview.heading_error_deg = math.degrees(filtered_heading)

    preview.suggested_angular_z_unfiltered = (
        config.lateral_gain * preview.lateral_error_m
        + config.heading_gain * filtered_heading
    )
    preview.suggested_angular_z_raw = (
        preview.suggested_angular_z_unfiltered
    )
    if level == "normal":
        angular_limit = config.normal_max_angular_z
        base_speed = config.normal_suggested_linear_x
    elif level == "degraded":
        angular_limit = config.degraded_max_angular_z
        base_speed = (
            config.short_path_suggested_linear_x
            if policy_reason == "short_path_degraded"
            else config.degraded_suggested_linear_x
        )
    else:
        if policy_reason == "sharp_recovery":
            angular_limit = config.sharp_recovery_max_angular_z
            base_speed = config.sharp_recovery_suggested_linear_x
        else:
            angular_limit = config.recovery_max_angular_z
            base_speed = config.recovery_suggested_linear_x
    preview.suggested_angular_z_mode_limited = clamp(
        preview.suggested_angular_z_unfiltered,
        -angular_limit,
        angular_limit,
    )
    preview.angular_command_saturated = not math.isclose(
        preview.suggested_angular_z_mode_limited,
        preview.suggested_angular_z_unfiltered,
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    )
    (
        preview.suggested_angular_z_rate_limited,
        preview.angular_rate_limited,
        preview.angular_delta_raw,
        preview.angular_delta_limited,
        preview.angular_dt_sec,
        preview.previous_preview_angular_z,
    ) = angular_slew_limit(
        preview.suggested_angular_z_mode_limited,
        instant,
        state,
        config,
        advance_state,
    )
    capped_rate_limited = clamp(
        preview.suggested_angular_z_rate_limited,
        -angular_limit,
        angular_limit,
    )
    if not math.isclose(
        capped_rate_limited,
        preview.suggested_angular_z_rate_limited,
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    ):
        preview.angular_rate_limited = True
        preview.suggested_angular_z_rate_limited = capped_rate_limited
        preview.angular_delta_limited = (
            capped_rate_limited - preview.previous_preview_angular_z
        )
        if advance_state:
            state.last_suggested_angular_z = capped_rate_limited
    preview.angular_direction_sign, preview.angular_direction_flip_count = (
        update_direction_history(
            preview.suggested_angular_z_rate_limited,
            state,
            config,
            advance_state,
        )
    )
    flip_limit = (
        config.recovery_max_direction_flips
        if level == "recovery"
        else config.max_direction_flips_in_window
    )
    preview.angular_direction_unstable = bool(
        preview.angular_direction_flip_count > flip_limit
    )

    if level == "recovery":
        sharp_recovery_active = policy_reason == "sharp_recovery"

        (
            preview.recovery_stable_frame_count,
            preview.recovery_heading_std_deg,
            preview.recovery_target_y_delta_m,
            preview.recovery_lateral_delta_m,
            recovery_stable,
        ) = update_recovery_stability(
            analysis,
            analysis.heading_robust_rad,
            preview.heading_rate_limited,
            preview.angular_direction_flip_count,
            state,
            config,
            advance_state,
            sharp_recovery=sharp_recovery_active,
        )

        required_stable_frames = (
            config.sharp_recovery_required_stable_frames
            if sharp_recovery_active
            else config.recovery_required_stable_frames
        )
        preview.recovery_required_stable_frames = required_stable_frames

        if (
            not recovery_stable
            or preview.recovery_stable_frame_count
            < required_stable_frames
        ):
            reason = "recovery_not_stable"
    elif advance_state:
        state.reset_recovery()

    if (
        level in {"degraded", "recovery"}
        and preview.angular_direction_unstable
    ):
        reason = "angular_direction_unstable"

    if reason:
        preview.controller_ready = False
        preview.control_block_reason = reason
        preview.control_quality_level = "blocked"
        preview.mode_policy_reason = reason
        preview.suggested_linear_x = 0.0
        preview.suggested_angular_z = 0.0
        if advance_state:
            state.last_suggested_angular_z = 0.0
            state.last_angular_timestamp = instant
            if reason != "recovery_not_stable":
                state.reset_recovery()
        preview.controller_confidence = 0.0
    else:
        preview.suggested_angular_z = (
            preview.suggested_angular_z_rate_limited
        )
        preview.controller_confidence = compute_controller_confidence(
            analysis,
            perception,
            preview,
            config,
        )
        confidence_factor = clamp(
            0.55 + 0.45 * preview.controller_confidence, 0.45, 1.0
        )
        lateral_factor = clamp(
            1.0 - abs(preview.lateral_error_m) / 0.30, 0.45, 1.0
        )
        heading_factor = clamp(
            1.0 - abs(preview.heading_error_rad) / math.radians(40.0),
            0.45,
            1.0,
        )
        coverage_factor = 0.80 if preview.lookahead_clamped else 1.0
        angular_factor = clamp(
            1.0
            - abs(preview.suggested_angular_z)
            / max(2.0 * angular_limit, 1.0e-6),
            0.55,
            1.0,
        )
        continuity_factor = (
            0.82
            if preview.heading_rate_limited
            or preview.angular_rate_limited
            else 1.0
        )
        disagreement_factor = (
            0.80
            if preview.heading_estimator_disagreement
            or preview.heading_chord_disagreement
            else 1.0
        )
        fps_factor = clamp(
            perception.process_fps / max(config.min_process_fps * 1.5, 1.0),
            0.70,
            1.0,
        )
        preview.suggested_linear_x = max(
            0.0,
            base_speed
            * confidence_factor
            * lateral_factor
            * heading_factor
            * coverage_factor
            * angular_factor
            * continuity_factor
            * disagreement_factor
            * fps_factor,
        )
    if advance_state:
        state.last_path_span_m = float(analysis.span_m)
    preview.calculation_time_ms = (
        time.perf_counter() - started
    ) * 1000.0
    return preview


def safe_compute_preview(
    analysis: PathAnalysis,
    perception: PerceptionSnapshot,
    config: ControllerConfig,
    path_age_sec: float,
    status_age_sec: float,
    update_rate_hz: float = 0.0,
    runtime_state: Optional[ControllerRuntimeState] = None,
    now: Optional[float] = None,
    advance_state: bool = True,
) -> ControllerPreview:
    try:
        return compute_preview_command(
            analysis,
            perception,
            config,
            path_age_sec,
            status_age_sec,
            update_rate_hz,
            runtime_state,
            now,
            advance_state,
        )
    except Exception as exc:
        return ControllerPreview(
            timestamp=datetime.now(timezone.utc).isoformat(),
            control_block_reason=f"calculation_error:{type(exc).__name__}",
            path_received=path_age_sec >= 0.0,
            status_received=perception.received,
            path_frame_id=analysis.frame_id,
            path_point_count=analysis.point_count,
            path_span_m=analysis.span_m,
        )


class LanePathControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("lane_path_controller_observer")
        defaults = ControllerConfig()
        for name, value in asdict(defaults).items():
            self.declare_parameter(name, value)
        requested_dry_run = bool(self.get_parameter("dry_run").value)
        values = {
            name: self.get_parameter(name).value
            for name in asdict(defaults)
        }
        values["dry_run"] = requested_dry_run
        self.config = ControllerConfig(**values)
        self.config.validate()
        if not requested_dry_run:
            self.get_logger().error(
                "control output is intentionally disabled in this version"
            )
        assert self.config.dry_run is True

        self.perception = PerceptionSnapshot()
        self.path_analysis = PathAnalysis()
        self.path_received_monotonic = 0.0
        self.preview = ControllerPreview()
        self.runtime_state = ControllerRuntimeState()
        self.path_sequence = 0
        self.processed_path_sequence = -1
        self.path_rate = RateMeter()
        self.last_log_monotonic = 0.0
        self.last_exception_log_monotonic = 0.0
        qos = make_observation_qos()
        self.path_subscription = self.create_subscription(
            RosPath,
            self.config.path_topic,
            self._path_callback,
            qos,
        )
        self.status_subscription = self.create_subscription(
            String,
            self.config.status_topic,
            self._status_callback,
            qos,
        )
        self.controller_status_publisher = self.create_publisher(
            String,
            self.config.controller_status_topic,
            qos,
        )
        self.control_cmd_publisher = self.create_publisher(
            String,
            self.config.control_cmd_topic,
            qos,
        )
        self.create_timer(
            1.0 / self.config.status_publish_rate_hz,
            self._publish_observation,
        )
        self.get_logger().info(
            "lane controller observer started: "
            f"path={self.config.path_topic} status={self.config.status_topic} "
            f"status_output={self.config.controller_status_topic} "
            f"control_output={self.config.control_cmd_topic} dry_run=true "
            "direct_cmd_vel=false "
            "qos=BEST_EFFORT/VOLATILE/depth1"
        )

    def _path_callback(self, message: RosPath) -> None:
        now = time.monotonic()
        try:
            points = [
                (float(pose.pose.position.x), float(pose.pose.position.y))
                for pose in message.poses
            ]
            self.path_analysis = validate_path(
                str(message.header.frame_id), points, self.config
            )
        except Exception as exc:
            self.path_analysis = PathAnalysis(
                reason=f"path_callback_error:{type(exc).__name__}"
            )
        self.path_received_monotonic = now
        self.path_sequence += 1
        self.path_rate.tick(now)
        self._refresh_preview(now)

    def _status_callback(self, message: String) -> None:
        now = time.monotonic()
        try:
            document = json.loads(message.data)
            self.perception = PerceptionSnapshot.from_document(document, now)
        except Exception as exc:
            self.perception = PerceptionSnapshot(
                received=True,
                received_monotonic=now,
                parse_error=f"{type(exc).__name__}: {exc}",
            )
        # Deliberately do not recalculate geometry/command here. A new Path is
        # the trigger; the timer only refreshes freshness gates for that result.

    def _refresh_preview(self, now: float) -> None:
        path_age = (
            now - self.path_received_monotonic
            if self.path_received_monotonic > 0.0
            else -1.0
        )
        status_age = (
            now - self.perception.received_monotonic
            if self.perception.received
            else -1.0
        )
        advance_state = bool(
            self.path_sequence != self.processed_path_sequence
            and (self.perception.received or not self.path_analysis.valid)
        )
        self.preview = safe_compute_preview(
            self.path_analysis,
            self.perception,
            self.config,
            path_age,
            status_age,
            self.path_rate.value(),
            self.runtime_state,
            now,
            advance_state,
        )
        if advance_state:
            self.processed_path_sequence = self.path_sequence
        if (
            not self.preview.controller_ready
            and self.preview.control_block_reason
            != "recovery_not_stable"
        ):
            self.runtime_state.reset()

    def _publish_observation(self) -> None:
        try:
            self._publish_observation_impl()
        except Exception as exc:
            self._handle_observation_exception(exc)

    def _publish_observation_impl(self) -> None:
        now = time.monotonic()
        self._refresh_preview(now)
        message = String()
        message.data = serialize_json(self.preview.to_dict())
        self.controller_status_publisher.publish(message)
        control_message = String()
        control_message.data = serialize_json(
            self.preview.to_control_command()
        )
        self.control_cmd_publisher.publish(control_message)
        if now - self.last_log_monotonic >= 1.0 / self.config.log_rate_hz:
            self.last_log_monotonic = now
            if self.preview.controller_ready:
                self.get_logger().info(
                    f"READY quality={self.preview.control_quality_level} "
                    f"mode={self.preview.centerline_mode} "
                    f"lat={self.preview.lateral_error_m:+.3f}m "
                    f"heading_raw={self.preview.heading_error_raw_deg:+.1f}deg "
                    f"heading={self.preview.heading_error_deg:+.1f}deg "
                    f"v={self.preview.suggested_linear_x:.3f} "
                    f"w={self.preview.suggested_angular_z:+.3f} "
                    f"conf={self.preview.controller_confidence:.2f}"
                )
            else:
                self.get_logger().warning(
                    f"BLOCKED reason={self.preview.control_block_reason} "
                    f"mode={self.preview.centerline_mode or '-'} "
                    f"stable={self.preview.recovery_stable_frame_count}/"
                    f"{self.preview.recovery_required_stable_frames} "
                    f"path_age={self.preview.path_age_ms:.0f}ms"
                )

    def _handle_observation_exception(self, exc: Exception) -> None:
        """Contain one failed timer cycle; never rethrow into the executor."""
        now = time.monotonic()
        self.preview = ControllerPreview(
            timestamp=datetime.now(timezone.utc).isoformat(),
            controller_ready=False,
            control_block_reason="controller_exception",
            degraded_mode=False,
            suggested_linear_x=0.0,
            suggested_angular_z_raw=0.0,
            suggested_angular_z=0.0,
            controller_confidence=0.0,
        )
        try:
            self.runtime_state.reset()
        except Exception:
            pass
        # This minimal fault status contains Python-native values only. Even its
        # construction, conversion, or publication is isolated from the timer.
        try:
            message = String()
            message.data = serialize_json(self.preview.to_dict())
            self.controller_status_publisher.publish(message)
            control_message = String()
            control_message.data = serialize_json(
                self.preview.to_control_command()
            )
            self.control_cmd_publisher.publish(control_message)
        except Exception:
            pass
        try:
            if (
                now - self.last_exception_log_monotonic
                >= 1.0 / self.config.log_rate_hz
            ):
                self.last_exception_log_monotonic = now
                self.get_logger().error(
                    "controller observation exception: "
                    f"{type(exc).__name__}: {exc}"
                )
        except Exception:
            pass


def _healthy_perception(
    *,
    mode: str = "green_dual_inner_edge",
    confidence: float = 0.98,
    valid: bool = True,
) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        received=True,
        received_monotonic=10.0,
        centerline_valid=valid,
        centerline_mode=mode,
        centerline_confidence=confidence,
        centerline_forward_span_m=0.70,
        green_boundary_fast_path_used=True,
        processing_time_ms=145.0,
        capture_to_process_age_ms=25.0,
        process_fps=6.5,
        path_publish_fps=6.5,
    )


def _path_points(
    lateral: float = 0.0,
    slope: float = 0.0,
    start: float = 0.05,
    end: float = 0.75,
    count: int = 15,
) -> np.ndarray:
    forward = np.linspace(start, end, count)
    left = lateral + slope * forward
    return np.column_stack((forward, left))


def run_self_test() -> None:
    tests: List[Tuple[str, Any]] = []

    def test(name: str):
        def decorate(function: Any) -> Any:
            tests.append((name, function))
            return function
        return decorate

    config = ControllerConfig()
    config.validate()

    @test("01 centered straight lateral zero")
    def _() -> None:
        result = validate_path("base_link", _path_points(), config)
        assert result.valid and abs(result.lateral_error_m) < 1.0e-9

    @test("02 left path positive lateral")
    def _() -> None:
        result = validate_path("base_link", _path_points(lateral=0.05), config)
        assert result.lateral_error_m > 0.0

    @test("03 right path negative lateral")
    def _() -> None:
        result = validate_path("base_link", _path_points(lateral=-0.05), config)
        assert result.lateral_error_m < 0.0

    @test("04 left curve positive heading")
    def _() -> None:
        result = validate_path("base_link", _path_points(slope=0.12), config)
        assert result.heading_error_rad > 0.0

    @test("05 right curve negative heading")
    def _() -> None:
        result = validate_path("base_link", _path_points(slope=-0.12), config)
        assert result.heading_error_rad < 0.0

    @test("06 left path positive preview angular")
    def _() -> None:
        result = validate_path("base_link", _path_points(lateral=0.05), config)
        preview = compute_preview_command(
            result, _healthy_perception(), config, 0.01, 0.01
        )
        assert preview.suggested_angular_z > 0.0

    @test("07 right path negative preview angular")
    def _() -> None:
        result = validate_path("base_link", _path_points(lateral=-0.05), config)
        preview = compute_preview_command(
            result, _healthy_perception(), config, 0.01, 0.01
        )
        assert preview.suggested_angular_z < 0.0

    @test("08 lookahead interpolation")
    def _() -> None:
        points = np.array([[0.2, 0.0], [0.6, 0.2]])
        x, y, distance, clipped = select_lookahead_target(points, 0.4)
        assert abs(x - 0.4) < 1e-9 and abs(y - 0.1) < 1e-9
        assert distance > 0.4 and not clipped

    @test("09 short lookahead clamps")
    def _() -> None:
        points = np.array([[0.1, 0.0], [0.3, 0.02]])
        x, _, _, clipped = select_lookahead_target(points, 0.4)
        assert clipped and abs(x - 0.3) < 1e-9

    @test("10 near error interpolation")
    def _() -> None:
        points = np.array([[0.1, 0.0], [0.4, 0.3]])
        x, y, clipped = interpolate_y_at_forward(points, 0.25)
        assert abs(x - 0.25) < 1e-9 and abs(y - 0.15) < 1e-9
        assert not clipped

    @test("11 wrong frame blocks")
    def _() -> None:
        result = validate_path("camera", _path_points(), config)
        assert not result.valid and result.reason == "path_frame_mismatch"

    @test("12 below structural point minimum blocks")
    def _() -> None:
        result = validate_path("base_link", _path_points(count=2), config)
        assert not result.valid and result.reason == "path_too_few_points"

    @test("13 below structural span minimum blocks")
    def _() -> None:
        result = validate_path(
            "base_link", _path_points(start=0.10, end=0.19), config
        )
        assert not result.valid and result.reason == "path_span_too_short"

    @test("14 nonfinite coordinate blocks")
    def _() -> None:
        points = _path_points()
        points[4, 1] = np.nan
        result = validate_path("base_link", points, config)
        assert not result.valid and result.reason == "path_non_finite"

    @test("15 nonmonotonic forward blocks")
    def _() -> None:
        points = _path_points()
        points[5, 0] = points[4, 0]
        result = validate_path("base_link", points, config)
        assert not result.valid and result.reason == "path_forward_not_monotonic"

    @test("16 lateral jump blocks")
    def _() -> None:
        points = _path_points()
        points[7:, 1] += 0.20
        result = validate_path("base_link", points, config)
        assert not result.valid and result.reason == "path_lateral_jump"

    @test("17 status age 0.5 remains usable")
    def _() -> None:
        result = validate_path("base_link", _path_points(), config)
        preview = compute_preview_command(
            result, _healthy_perception(), config, 0.01, 0.5
        )
        assert preview.controller_ready

    @test("17b status age over 1.0 blocks")
    def _() -> None:
        result = validate_path("base_link", _path_points(), config)
        preview = compute_preview_command(
            result, _healthy_perception(), config, 0.01, 1.01
        )
        assert preview.control_block_reason == "status_timeout"

    @test("18 path timeout blocks")
    def _() -> None:
        result = validate_path("base_link", _path_points(), config)
        preview = compute_preview_command(
            result, _healthy_perception(), config, 0.5, 0.01
        )
        assert preview.control_block_reason == "path_timeout"

    @test("19 invalid centerline blocks")
    def _() -> None:
        result = validate_path("base_link", _path_points(), config)
        preview = compute_preview_command(
            result, _healthy_perception(valid=False), config, 0.01, 0.01
        )
        assert preview.control_block_reason == "centerline_invalid"

    @test("20 low confidence blocks")
    def _() -> None:
        result = validate_path("base_link", _path_points(), config)
        preview = compute_preview_command(
            result,
            _healthy_perception(confidence=0.50),
            config,
            0.01,
            0.01,
        )
        assert preview.control_block_reason == "normal_confidence_low"

    @test("21 degraded reduces speed and confidence")
    def _() -> None:
        result = validate_path("base_link", _path_points(), config)
        normal = compute_preview_command(
            result, _healthy_perception(), config, 0.01, 0.01
        )
        degraded = compute_preview_command(
            result,
            _healthy_perception(mode="green_yellow_hybrid"),
            config,
            0.01,
            0.01,
        )
        assert degraded.controller_ready and degraded.degraded_mode
        assert degraded.suggested_linear_x < normal.suggested_linear_x
        assert degraded.controller_confidence < normal.controller_confidence

    @test("22 angular signs remain standard")
    def _() -> None:
        left = validate_path(
            "base_link", _path_points(lateral=0.02, slope=0.08), config
        )
        right = validate_path(
            "base_link", _path_points(lateral=-0.02, slope=-0.08), config
        )
        assert compute_preview_command(
            left, _healthy_perception(), config, 0.01, 0.01
        ).suggested_angular_z > 0
        assert compute_preview_command(
            right, _healthy_perception(), config, 0.01, 0.01
        ).suggested_angular_z < 0

    @test("23 angular saturation")
    def _() -> None:
        result = validate_path(
            "base_link", _path_points(lateral=0.14, slope=0.4), config
        )
        preview = compute_preview_command(
            result, _healthy_perception(), config, 0.01, 0.01
        )
        assert preview.angular_command_saturated
        assert abs(preview.suggested_angular_z - 0.60) < 1e-9

    @test("24 status JSON serializable")
    def _() -> None:
        result = validate_path("base_link", _path_points(), config)
        preview = compute_preview_command(
            result, _healthy_perception(), config, 0.01, 0.01
        )
        packed = serialize_json(preview.to_dict())
        assert json.loads(packed)["controller_ready"]

    @test("25 compatible best effort QoS")
    def _() -> None:
        qos = make_observation_qos()
        assert qos.reliability == ReliabilityPolicy.BEST_EFFORT
        assert qos.durability == DurabilityPolicy.VOLATILE
        assert qos.depth == 1

    @test("26 source has no chassis message publisher")
    def _() -> None:
        source = FilePath(__file__).read_text(encoding="utf-8")
        forbidden_type = "Twi" + "st"
        assert f"create_publisher({forbidden_type}" not in source

    @test("27 source has no chassis velocity publisher")
    def _() -> None:
        source = FilePath(__file__).read_text(encoding="utf-8")
        forbidden = "create_" + "publisher(" + "Twi" + "st"
        assert forbidden not in source

    @test("28 source has no raw velocity target")
    def _() -> None:
        source = FilePath(__file__).read_text(encoding="utf-8")
        target = "/" + "cmd" + "_vel" + "_raw"
        assert target not in source

    @test("29 false dry run cannot enable control")
    def _() -> None:
        assert enforce_dry_run(False) is True

    @test("30 calculation exception safely blocks")
    def _() -> None:
        result = validate_path("base_link", _path_points(), config)
        broken = ControllerConfig(normal_max_angular_z="broken")  # type: ignore[arg-type]
        preview = safe_compute_preview(
            result, _healthy_perception(), broken, 0.01, 0.01
        )
        assert not preview.controller_ready
        assert preview.control_block_reason.startswith("calculation_error:")

    @test("31 JSON parse failure is represented")
    def _() -> None:
        snapshot = PerceptionSnapshot(
            received=True, received_monotonic=10.0, parse_error="JSONDecodeError"
        )
        result = validate_path("base_link", _path_points(), config)
        preview = compute_preview_command(result, snapshot, config, 0.01, 0.01)
        assert preview.control_block_reason == "status_json_invalid"

    @test("32 low process FPS blocks")
    def _() -> None:
        snapshot = _healthy_perception()
        snapshot.process_fps = 3.9
        result = validate_path("base_link", _path_points(), config)
        preview = compute_preview_command(result, snapshot, config, 0.01, 0.01)
        assert preview.control_block_reason == "process_fps_too_low"

    @test("33 perception capture age blocks")
    def _() -> None:
        snapshot = _healthy_perception()
        snapshot.capture_to_process_age_ms = 351.0
        result = validate_path("base_link", _path_points(), config)
        preview = compute_preview_command(result, snapshot, config, 0.01, 0.01)
        assert preview.control_block_reason == "perception_age_too_high"

    @test("34 unknown mode blocks")
    def _() -> None:
        result = validate_path("base_link", _path_points(), config)
        preview = compute_preview_command(
            result,
            _healthy_perception(mode="future_unknown_mode"),
            config,
            0.01,
            0.01,
        )
        assert preview.control_block_reason == "centerline_mode_not_allowed"

    @test("35 numpy true becomes JSON true")
    def _() -> None:
        packed = serialize_json({"value": np.bool_(True)})
        assert json.loads(packed)["value"] is True

    @test("36 numpy false becomes JSON false")
    def _() -> None:
        packed = serialize_json({"value": np.bool_(False)})
        assert json.loads(packed)["value"] is False

    @test("37 numpy integers become integers")
    def _() -> None:
        packed = serialize_json(
            {"small": np.int32(7), "large": np.int64(9)}
        )
        assert json.loads(packed) == {"small": 7, "large": 9}

    @test("38 numpy floats become numbers")
    def _() -> None:
        packed = serialize_json(
            {"single": np.float32(1.25), "double": np.float64(2.5)}
        )
        document = json.loads(packed)
        assert abs(document["single"] - 1.25) < 1.0e-6
        assert abs(document["double"] - 2.5) < 1.0e-12

    @test("39 ndarray recursively becomes list")
    def _() -> None:
        document = json.loads(
            serialize_json(np.array([[1, 2], [3, 4]], dtype=np.int32))
        )
        assert document == [[1, 2], [3, 4]]

    @test("40 nested numpy values are safe")
    def _() -> None:
        value = {
            5: [
                np.bool_(True),
                {"count": np.int64(3), "ratio": np.float32(0.5)},
            ]
        }
        document = json.loads(serialize_json(value))
        assert document == {
            "5": [True, {"count": 3, "ratio": 0.5}]
        }

    @test("41 nonfinite floats become null")
    def _() -> None:
        value = [
            float("nan"),
            float("inf"),
            float("-inf"),
            np.float64(np.nan),
        ]
        assert json.loads(serialize_json(value)) == [None, None, None, None]

    @test("42 strict JSON accepts converted payload")
    def _() -> None:
        packed = serialize_json(
            {"flag": np.bool_(True), "bad": np.float32(np.inf)}
        )
        assert json.loads(packed) == {"flag": True, "bad": None}

    @test("43 complete real path preview is serializable")
    def _() -> None:
        result = validate_path(
            "base_link",
            _path_points(lateral=0.02, slope=0.05),
            config,
        )
        preview = compute_preview_command(
            result, _healthy_perception(), config, 0.01, 0.01, 6.5
        )
        document = json.loads(serialize_json(preview.to_dict()))
        assert document["controller_ready"] is True
        assert document["path_point_count"] == 15

    @test("44 numpy lookahead clamp cannot break JSON")
    def _() -> None:
        preview = ControllerPreview(lookahead_clamped=np.bool_(True))
        document = json.loads(serialize_json(preview.to_dict()))
        assert document["lookahead_clamped"] is True

    @test("45 numpy angular saturation cannot break JSON")
    def _() -> None:
        preview = ControllerPreview(
            angular_command_saturated=np.bool_(True)
        )
        document = json.loads(serialize_json(preview.to_dict()))
        assert document["angular_command_saturated"] is True

    @test("46 exception handler never rethrows")
    def _() -> None:
        class FailingPublisher:
            def publish(self, _message: Any) -> None:
                raise RuntimeError("simulated publish failure")

        class FailingLogger:
            def error(self, _message: str) -> None:
                raise RuntimeError("simulated log failure")

        node = LanePathControllerNode.__new__(LanePathControllerNode)
        node.config = config
        node.path_received_monotonic = 1.0
        node.perception = _healthy_perception()
        node.path_analysis = validate_path(
            "base_link", _path_points(), config
        )
        node.controller_status_publisher = FailingPublisher()
        node.last_exception_log_monotonic = 0.0
        node.get_logger = lambda: FailingLogger()
        node._handle_observation_exception(RuntimeError("simulated"))
        assert not node.preview.controller_ready
        assert node.preview.control_block_reason == "controller_exception"

    @test("47 timer exception stays contained and zero preview")
    def _() -> None:
        node = LanePathControllerNode.__new__(LanePathControllerNode)
        handled: List[Exception] = []
        node._publish_observation_impl = (
            lambda: (_ for _ in ()).throw(TypeError("simulated"))
        )
        node._handle_observation_exception = handled.append
        node._publish_observation()
        assert len(handled) == 1
        fault = ControllerPreview(
            controller_ready=False,
            control_block_reason="controller_exception",
            suggested_linear_x=0.0,
            suggested_angular_z=0.0,
        )
        assert not fault.controller_ready
        assert fault.suggested_linear_x == 0.0
        assert fault.suggested_angular_z == 0.0

    @test("48 robust straight heading zero")
    def _() -> None:
        result = validate_path("base_link", _path_points(), config)
        assert abs(result.heading_robust_rad) < 1.0e-9

    @test("49 robust left heading positive")
    def _() -> None:
        result = validate_path(
            "base_link", _path_points(slope=0.08), config
        )
        assert result.heading_robust_rad > 0.0

    @test("50 robust right heading negative")
    def _() -> None:
        result = validate_path(
            "base_link", _path_points(slope=-0.08), config
        )
        assert result.heading_robust_rad < 0.0

    @test("51 one far outlier does not create heading jump")
    def _() -> None:
        points = _path_points(start=0.30, end=0.55, count=6, slope=0.05)
        points[-1, 1] += 0.08
        result = validate_path("base_link", points, config)
        assert result.valid
        assert abs(math.degrees(result.heading_robust_rad)) < 6.0

    @test("52 segment median matches constant slope")
    def _() -> None:
        result = validate_path(
            "base_link", _path_points(slope=0.10), config
        )
        assert abs(result.heading_segment_median_rad - math.atan(0.10)) < 1e-9

    @test("53 MAD removes segment outlier")
    def _() -> None:
        points = _path_points(start=0.30, end=0.55, count=6, slope=0.05)
        points[-1, 1] += 0.08
        result = validate_path("base_link", points, config)
        assert result.heading_segment_count_used < result.heading_segment_count_raw

    @test("54 OLS disagreement retains robust heading")
    def _() -> None:
        points = _path_points(start=0.30, end=0.55, count=6, slope=0.05)
        points[-1, 1] += 0.12
        result = validate_path("base_link", points, config)
        assert result.heading_estimator_disagreement
        assert result.heading_error_rad == result.heading_robust_rad

    @test("55 chord heading is correct")
    def _() -> None:
        result = validate_path(
            "base_link", _path_points(slope=0.10), config
        )
        assert abs(result.heading_chord_rad - math.atan(0.10)) < 1e-9

    @test("56 chord disagreement is diagnosed")
    def _() -> None:
        points = _path_points(start=0.30, end=0.60, count=7, slope=0.04)
        points[3, 1] += 0.10
        result = validate_path("base_link", points, config)
        assert result.valid
        assert result.heading_chord_disagreement or result.heading_estimator_disagreement

    @test("57 tiny dx segment is excluded")
    def _() -> None:
        x = np.array([0.30, 0.305, 0.35, 0.40, 0.45, 0.50, 0.55])
        points = np.column_stack((x, 0.05 * x))
        result = validate_path("base_link", points, config)
        assert result.valid
        assert result.heading_segment_count_raw == 5

    @test("58 insufficient valid segments fails safely")
    def _() -> None:
        x = np.array([0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.55])
        points = np.column_stack((x, 0.05 * x))
        result = validate_path("base_link", points, config)
        assert not result.valid
        assert result.reason == "heading_estimation_failed"

    @test("59 first heading frame initializes directly")
    def _() -> None:
        state = ControllerRuntimeState()
        value, used, limited, dt = filter_heading_temporal(
            0.10, 1.0, "green_dual_inner_edge", state, config, True
        )
        assert abs(value - 0.10) < 1e-9 and not used and not limited and dt == 0

    @test("60 small heading change is filtered normally")
    def _() -> None:
        state = ControllerRuntimeState(
            last_heading_error_rad=0.05,
            last_heading_timestamp=1.0,
            last_centerline_mode="green_dual_inner_edge",
        )
        value, used, limited, _ = filter_heading_temporal(
            0.06, 1.2, "green_dual_inner_edge", state, config, True
        )
        assert 0.05 < value < 0.06 and used and not limited

    @test("61 one-frame heading jump is rate limited")
    def _() -> None:
        state = ControllerRuntimeState(
            last_heading_error_rad=math.radians(3.0),
            last_heading_timestamp=1.0,
            last_centerline_mode="green_dual_inner_edge",
        )
        value, _, limited, _ = filter_heading_temporal(
            math.radians(18.0),
            1.1,
            "single_green_width_offset",
            state,
            config,
            True,
        )
        assert limited and math.degrees(value) < 10.0

    @test("62 sustained turn is followed over time")
    def _() -> None:
        state = ControllerRuntimeState()
        values = []
        for index in range(12):
            value, _, _, _ = filter_heading_temporal(
                math.radians(15.0),
                1.0 + 0.2 * index,
                "green_dual_inner_edge",
                state,
                config,
                True,
            )
            values.append(value)
        assert values[-1] > math.radians(14.0)

    @test("63 heading history resets after timeout")
    def _() -> None:
        state = ControllerRuntimeState(
            last_heading_error_rad=0.0,
            last_heading_timestamp=1.0,
            last_centerline_mode="green_dual_inner_edge",
        )
        value, used, _, _ = filter_heading_temporal(
            0.20, 2.0, "green_dual_inner_edge", state, config, True
        )
        assert abs(value - 0.20) < 1e-9 and not used

    @test("64 explicit blocked reset clears heading")
    def _() -> None:
        state = ControllerRuntimeState(
            last_heading_error_rad=0.1, last_heading_timestamp=1.0
        )
        state.reset()
        assert state.last_heading_error_rad is None

    @test("65 backward timestamp resets heading")
    def _() -> None:
        state = ControllerRuntimeState(
            last_heading_error_rad=0.1,
            last_heading_timestamp=2.0,
            last_centerline_mode="green_dual_inner_edge",
        )
        value, used, _, _ = filter_heading_temporal(
            -0.1, 1.0, "green_dual_inner_edge", state, config, True
        )
        assert abs(value + 0.1) < 1e-9 and not used

    @test("66 normal mode policy")
    def _() -> None:
        analysis = validate_path("base_link", _path_points(), config)
        level, reason = evaluate_mode_policy(
            analysis, _healthy_perception(), config
        )
        assert level == "normal" and reason == "normal_mode_accepted"

    @test("67 hybrid short path enters degraded")
    def _() -> None:
        points = _path_points(start=0.30, end=0.60, count=7, slope=0.05)
        analysis = validate_path("base_link", points, config)
        snapshot = _healthy_perception(
            mode="green_yellow_hybrid", confidence=0.90
        )
        level, _ = evaluate_mode_policy(analysis, snapshot, config)
        assert level == "degraded"

    @test("68 recovery first frame is blocked")
    def _() -> None:
        state = ControllerRuntimeState()
        analysis = validate_path(
            "base_link",
            _path_points(start=0.30, end=0.60, count=7, slope=0.05),
            config,
        )
        preview = compute_preview_command(
            analysis,
            _healthy_perception(
                mode="single_green_width_offset", confidence=0.80
            ),
            config,
            0.01,
            0.01,
            runtime_state=state,
            now=1.0,
        )
        assert not preview.controller_ready
        assert preview.control_block_reason == "recovery_not_stable"

    @test("69 recovery becomes ready on fourth stable frame")
    def _() -> None:
        state = ControllerRuntimeState()
        analysis = validate_path(
            "base_link",
            _path_points(start=0.30, end=0.60, count=7, slope=0.05),
            config,
        )
        snapshot = _healthy_perception(
            mode="single_green_width_offset", confidence=0.80
        )
        previews = [
            compute_preview_command(
                analysis,
                snapshot,
                config,
                0.01,
                0.01,
                runtime_state=state,
                now=1.0 + 0.2 * index,
            )
            for index in range(4)
        ]
        assert all(not item.controller_ready for item in previews[:3])
        assert previews[3].controller_ready
        assert previews[3].control_quality_level == "recovery"

    @test("70 recovery anomaly resets stable count")
    def _() -> None:
        state = ControllerRuntimeState()
        snapshot = _healthy_perception(
            mode="single_green_width_offset", confidence=0.80
        )
        stable = validate_path(
            "base_link",
            _path_points(start=0.30, end=0.60, count=7, slope=0.03),
            config,
        )
        compute_preview_command(
            stable, snapshot, config, 0.01, 0.01,
            runtime_state=state, now=1.0
        )
        unstable = validate_path(
            "base_link",
            _path_points(
                lateral=0.04, start=0.30, end=0.60, count=7, slope=0.03
            ),
            config,
        )
        preview = compute_preview_command(
            unstable, snapshot, config, 0.01, 0.01,
            runtime_state=state, now=1.2
        )
        assert preview.recovery_stable_frame_count == 0

    @test("71 unsupported single boundary always blocks")
    def _() -> None:
        analysis = validate_path("base_link", _path_points(), config)
        preview = compute_preview_command(
            analysis,
            _healthy_perception(mode="single_boundary_normal_offset"),
            config, 0.01, 0.01
        )
        assert preview.control_block_reason == "unsupported_single_boundary_mode"

    @test("72 invalid mode always blocks")
    def _() -> None:
        analysis = validate_path("base_link", _path_points(), config)
        preview = compute_preview_command(
            analysis, _healthy_perception(mode="invalid"),
            config, 0.01, 0.01
        )
        assert not preview.controller_ready

    @test("73 unknown mode always blocks")
    def _() -> None:
        analysis = validate_path("base_link", _path_points(), config)
        preview = compute_preview_command(
            analysis, _healthy_perception(mode="mystery"),
            config, 0.01, 0.01
        )
        assert preview.control_block_reason == "centerline_mode_not_allowed"

    @test("74 sharp recovery span below minimum blocks")
    def _() -> None:
        local = ControllerConfig(
            min_path_span_m=0.05,
            min_centerline_span_m=0.05,
        )
        local.validate()
        analysis = validate_path(
            "base_link",
            _path_points(start=0.30, end=0.38, count=3),
            local,
        )
        assert analysis.valid
        level, reason = evaluate_mode_policy(
            analysis,
            _healthy_perception(mode="single_green_width_offset"),
            local,
        )
        assert (
            level == "blocked"
            and reason == "sharp_recovery_path_span_low"
        )

    @test("75 recovery low confidence blocks")
    def _() -> None:
        analysis = validate_path(
            "base_link",
            _path_points(start=0.30, end=0.60, count=7),
            config,
        )
        level, reason = evaluate_mode_policy(
            analysis,
            _healthy_perception(
                mode="single_green_width_offset", confidence=0.29
            ),
            config,
        )
        assert level == "blocked" and reason == "recovery_confidence_low"

    @test("76 left curve angular remains positive")
    def _() -> None:
        analysis = validate_path(
            "base_link", _path_points(slope=0.10), config
        )
        preview = compute_preview_command(
            analysis, _healthy_perception(), config, 0.01, 0.01
        )
        assert preview.suggested_angular_z > 0

    @test("77 right curve angular remains negative")
    def _() -> None:
        analysis = validate_path(
            "base_link", _path_points(slope=-0.10), config
        )
        preview = compute_preview_command(
            analysis, _healthy_perception(), config, 0.01, 0.01
        )
        assert preview.suggested_angular_z < 0

    @test("78 normal angular mode limit")
    def _() -> None:
        analysis = validate_path(
            "base_link", _path_points(lateral=0.14, slope=0.4), config
        )
        preview = compute_preview_command(
            analysis, _healthy_perception(), config, 0.01, 0.01
        )
        assert abs(preview.suggested_angular_z_mode_limited) <= 0.60

    @test("79 degraded angular mode limit")
    def _() -> None:
        analysis = validate_path(
            "base_link", _path_points(lateral=0.14, slope=0.4), config
        )
        preview = compute_preview_command(
            analysis,
            _healthy_perception(mode="green_yellow_hybrid"),
            config, 0.01, 0.01
        )
        assert abs(preview.suggested_angular_z_mode_limited) <= 0.35

    @test("80 recovery angular mode limit")
    def _() -> None:
        analysis = validate_path(
            "base_link",
            _path_points(
                lateral=0.10, start=0.30, end=0.60, count=7, slope=0.2
            ),
            config,
        )
        preview = compute_preview_command(
            analysis,
            _healthy_perception(mode="single_green_width_offset"),
            config, 0.01, 0.01
        )
        assert abs(preview.suggested_angular_z_mode_limited) <= 0.30

    @test("81 angular slew rate limit")
    def _() -> None:
        state = ControllerRuntimeState(
            last_suggested_angular_z=0.10,
            last_angular_timestamp=1.0,
        )
        value, limited, _, _, _, _ = angular_slew_limit(
            0.30, 1.2, state, config, True
        )
        assert limited and value <= 0.2200001

    @test("82 blocked command is immediately zero")
    def _() -> None:
        analysis = validate_path("base_link", _path_points(), config)
        preview = compute_preview_command(
            analysis, _healthy_perception(valid=False),
            config, 0.01, 0.01
        )
        assert preview.suggested_angular_z == 0.0

    @test("83 measured angular jump is suppressed")
    def _() -> None:
        state = ControllerRuntimeState(
            last_suggested_angular_z=0.115,
            last_angular_timestamp=1.0,
        )
        value, limited, _, _, _, _ = angular_slew_limit(
            0.300, 1.2, state, config, True
        )
        assert limited and value < 0.300

    @test("84 sustained angular target is eventually reached")
    def _() -> None:
        state = ControllerRuntimeState(
            last_suggested_angular_z=0.0,
            last_angular_timestamp=1.0,
        )
        value = 0.0
        for index in range(1, 8):
            value, _, _, _, _, _ = angular_slew_limit(
                0.30, 1.0 + 0.2 * index, state, config, True
            )
        assert abs(value - 0.30) < 1e-9

    @test("85 frequent direction flips are detected")
    def _() -> None:
        state = ControllerRuntimeState()
        flips = 0
        for value in (0.1, -0.1, 0.1, -0.1):
            _, flips = update_direction_history(
                value, state, config, True
            )
        assert flips == 3

    @test("86 deadband values do not count as flips")
    def _() -> None:
        state = ControllerRuntimeState()
        for value in (0.01, -0.02, 0.0):
            sign, flips = update_direction_history(
                value, state, config, True
            )
            assert sign == 0 and flips == 0

    @test("87 normal speed exceeds degraded")
    def _() -> None:
        analysis = validate_path("base_link", _path_points(), config)
        normal = compute_preview_command(
            analysis, _healthy_perception(), config, 0.01, 0.01
        )
        degraded = compute_preview_command(
            analysis,
            _healthy_perception(mode="green_yellow_hybrid"),
            config, 0.01, 0.01
        )
        assert normal.suggested_linear_x > degraded.suggested_linear_x

    @test("88 degraded base speed is at least recovery base")
    def _() -> None:
        assert (
            config.degraded_suggested_linear_x
            >= config.recovery_suggested_linear_x
        )

    @test("89 blocked linear speed is zero")
    def _() -> None:
        analysis = validate_path("base_link", _path_points(), config)
        preview = compute_preview_command(
            analysis, _healthy_perception(valid=False),
            config, 0.01, 0.01
        )
        assert preview.suggested_linear_x == 0.0

    @test("90 recovery confidence cap")
    def _() -> None:
        state = ControllerRuntimeState()
        analysis = validate_path(
            "base_link",
            _path_points(start=0.30, end=0.60, count=7, slope=0.03),
            config,
        )
        snapshot = _healthy_perception(
            mode="single_green_width_offset", confidence=0.99
        )
        preview = None
        for index in range(4):
            preview = compute_preview_command(
                analysis, snapshot, config, 0.01, 0.01,
                runtime_state=state, now=1.0 + 0.2 * index
            )
        assert preview is not None
        assert preview.controller_confidence <= 0.35

    @test("91 estimator disagreement lowers confidence")
    def _() -> None:
        clean = validate_path("base_link", _path_points(slope=0.05), config)
        points = _path_points(start=0.30, end=0.55, count=6, slope=0.05)
        points[-1, 1] += 0.12
        noisy = validate_path("base_link", points, config)
        clean_preview = compute_preview_command(
            clean, _healthy_perception(), config, 0.01, 0.01
        )
        noisy_preview = compute_preview_command(
            noisy,
            _healthy_perception(mode="green_yellow_hybrid"),
            config, 0.01, 0.01
        )
        assert noisy_preview.controller_confidence < clean_preview.controller_confidence

    @test("92 heading rate limit lowers confidence")
    def _() -> None:
        analysis = validate_path(
            "base_link", _path_points(slope=0.30), config
        )
        state = ControllerRuntimeState(
            last_heading_error_rad=0.0,
            last_heading_timestamp=1.0,
            last_centerline_mode="green_dual_inner_edge",
            last_suggested_angular_z=0.0,
            last_angular_timestamp=1.0,
        )
        limited = compute_preview_command(
            analysis, _healthy_perception(), config, 0.01, 0.01,
            runtime_state=state, now=1.1
        )
        baseline = compute_preview_command(
            analysis, _healthy_perception(), config, 0.01, 0.01
        )
        assert limited.heading_rate_limited
        assert limited.controller_confidence < baseline.controller_confidence

    @test("93 new status fields serialize")
    def _() -> None:
        analysis = validate_path("base_link", _path_points(), config)
        preview = compute_preview_command(
            analysis, _healthy_perception(), config, 0.01, 0.01
        )
        document = json.loads(serialize_json(preview.to_dict()))
        for key in (
            "control_quality_level",
            "heading_method",
            "heading_robust_rad",
            "heading_chord_rad",
            "heading_error_filtered_rad",
            "suggested_angular_z_mode_limited",
            "angular_direction_flip_count",
        ):
            assert key in document

    @test("94 numpy values remain serializable")
    def _() -> None:
        assert json.loads(
            serialize_json({"flag": np.bool_(True), "value": np.float32(1.0)})
        ) == {"flag": True, "value": 1.0}

    @test("95 timer containment remains active")
    def _() -> None:
        node = LanePathControllerNode.__new__(LanePathControllerNode)
        handled: List[Exception] = []
        node._publish_observation_impl = (
            lambda: (_ for _ in ()).throw(RuntimeError("test"))
        )
        node._handle_observation_exception = handled.append
        node._publish_observation()
        assert len(handled) == 1

    @test("96 source has no forbidden message import")
    def _() -> None:
        source = FilePath(__file__).read_text(encoding="utf-8")
        message_name = "Twi" + "st"
        package_name = "geometry" + "_msgs.msg"
        assert f"from {package_name} import {message_name}" not in source

    @test("97 exactly two JSON publishers remain")
    def _() -> None:
        source = FilePath(__file__).read_text(encoding="utf-8")
        assert source.count("self.create_" + "publisher(") == 2

    @test("98 source does not import chassis Twist")
    def _() -> None:
        source = FilePath(__file__).read_text(encoding="utf-8")
        forbidden = (
            "from geometry" + "_msgs.msg import " + "Twi" + "st"
        )
        assert forbidden not in source

    @test("99 false dry run still forced true")
    def _() -> None:
        assert enforce_dry_run(False)

    @test("100 exception preview has zero commands")
    def _() -> None:
        preview = ControllerPreview(
            controller_ready=False,
            control_block_reason="controller_exception",
        )
        assert preview.suggested_linear_x == 0.0
        assert preview.suggested_angular_z == 0.0

    @test("101 angular direction switch decelerates through zero")
    def _() -> None:
        state = ControllerRuntimeState(
            last_suggested_angular_z=0.10,
            last_angular_timestamp=1.0,
        )
        value, limited, _, _, _, _ = angular_slew_limit(
            -0.10, 1.2, state, config, True
        )
        assert limited
        assert -0.061 <= value <= -0.059

    @test("102 perception span participates in mode gate")
    def _() -> None:
        analysis = validate_path("base_link", _path_points(), config)
        snapshot = _healthy_perception()
        snapshot.centerline_forward_span_m = 0.30
        level, reason = evaluate_mode_policy(analysis, snapshot, config)
        assert level == "degraded" and reason == "short_path_degraded"

    @test("103 degraded direction instability blocks")
    def _() -> None:
        state = ControllerRuntimeState(
            angular_direction_history=[1, -1, 1]
        )
        analysis = validate_path(
            "base_link", _path_points(slope=0.05), config
        )
        preview = compute_preview_command(
            analysis,
            _healthy_perception(mode="green_yellow_hybrid"),
            config,
            0.01,
            0.01,
            runtime_state=state,
            now=1.0,
        )
        assert not preview.controller_ready
        assert preview.control_block_reason == "angular_direction_unstable"

    @test("104 mode cap holds without new path state advance")
    def _() -> None:
        state = ControllerRuntimeState(
            last_heading_error_rad=0.2,
            last_heading_timestamp=1.0,
            last_suggested_angular_z=0.5,
            last_angular_timestamp=1.0,
            last_centerline_mode="green_dual_inner_edge",
        )
        analysis = validate_path(
            "base_link", _path_points(slope=0.2), config
        )
        preview = compute_preview_command(
            analysis,
            _healthy_perception(mode="green_yellow_hybrid"),
            config,
            0.01,
            0.01,
            runtime_state=state,
            now=1.2,
            advance_state=False,
        )
        assert abs(preview.suggested_angular_z) <= 0.35

    @test("105 dual boundary midpoint is capped degraded")
    def _() -> None:
        analysis = validate_path(
            "base_link",
            _path_points(start=0.30, end=0.60, count=7),
            config,
        )
        level, _ = evaluate_mode_policy(
            analysis,
            _healthy_perception(
                mode="dual_boundary_midpoint", confidence=0.85
            ),
            config,
        )
        assert level == "degraded"

    @test("106 intermediate control interface is complete")
    def _() -> None:
        analysis = validate_path("base_link", _path_points(), config)
        preview = compute_preview_command(
            analysis, _healthy_perception(), config, 0.01, 0.01
        )
        command = preview.to_control_command()
        for key in (
            "linear_x",
            "angular_z",
            "ready",
            "quality",
            "mode",
            "lateral_error",
            "heading_error",
            "timestamp",
        ):
            assert key in command
        assert command["interface_version"] == "gxb_lane_control_v1"

    @test("107 blocked intermediate command is zero")
    def _() -> None:
        preview = ControllerPreview(
            controller_ready=False,
            control_block_reason="test_block",
        )
        command = preview.to_control_command()
        assert not command["ready"]
        assert command["linear_x"] == 0.0
        assert command["angular_z"] == 0.0

    @test("108 preferred six-point path stays ready degraded")
    def _() -> None:
        analysis = validate_path(
            "base_link", _path_points(start=0.30, end=0.55, count=6), config
        )
        perception = _healthy_perception()
        perception.centerline_forward_span_m = 0.25
        preview = compute_preview_command(
            analysis, perception, config, 0.01, 0.50
        )
        assert preview.controller_ready
        assert preview.control_quality_level == "degraded"
        assert preview.mode_policy_reason == "short_path_degraded"
        assert 0.0 < preview.suggested_linear_x <= 0.20

    @test("109 preferred five-point path remains blocked")
    def _() -> None:
        analysis = validate_path("base_link", _path_points(count=5), config)
        preview = compute_preview_command(
            analysis, _healthy_perception(), config, 0.01, 0.01
        )
        assert not preview.controller_ready
        assert preview.suggested_linear_x == 0.0

    @test("110 preferred span below 0.25 remains blocked")
    def _() -> None:
        analysis = validate_path(
            "base_link", _path_points(start=0.30, end=0.54, count=6), config
        )
        preview = compute_preview_command(
            analysis, _healthy_perception(), config, 0.01, 0.01
        )
        assert not preview.controller_ready
        assert preview.suggested_linear_x == 0.0

    @test("111 clean normal path can approach 0.50")
    def _() -> None:
        analysis = validate_path("base_link", _path_points(), config)
        preview = compute_preview_command(
            analysis, _healthy_perception(confidence=1.0), config, 0.01, 0.01
        )
        assert preview.controller_ready
        assert 0.45 <= preview.suggested_linear_x <= 0.50

    failures: List[str] = []
    for name, function in tests:
        try:
            function()
            print(f"PASS {name}")
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"FAIL {failures[-1]}")
    print(f"self-test: {len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        raise SystemExit(1)


def main() -> None:
    if SELF_TEST:
        run_self_test()
        return
    rclpy.init(args=sys.argv)
    node: Optional[LanePathControllerNode] = None
    try:
        node = LanePathControllerNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().error(
                f"controller observer stopped by exception: {type(exc).__name__}: {exc}"
            )
        else:
            print(
                f"controller observer startup failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        raise
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
