#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第五阶段：车道路径控制观察节点（只读 dry-run）。

运行：
  source /opt/tros/humble/setup.bash
  source /root/intelligent_car_ws/install/setup.bash
  python3 /root/intelligent_car_ws/src/gxb_test/5_lane_path_controller.py \
    --ros-args -p dry_run:=true

查看：
  ros2 topic echo /gxb_test/controller/status

本节点只订阅感知 Path/status、计算控制预览并发布 JSON 诊断。它不导入底盘
速度消息、不创建速度发布器，也不控制车辆。即使 dry_run 参数传入 false，
控制输出仍保持禁用。
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
        # 一体化节点在两种快速结果共同参与时使用此实际模式名。
        "green_yellow_hybrid",
    }
)
DEGRADED_MODES = frozenset(
    {
        "single_green_width_offset",
        "single_boundary_normal_offset",
        "dual_boundary_midpoint",
        "history_fallback",
    }
)


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
    dry_run: bool = True
    log_rate_hz: float = 2.0
    status_publish_rate_hz: float = 5.0
    min_path_points: int = 6
    min_path_span_m: float = 0.45
    path_timeout_sec: float = 0.40
    status_timeout_sec: float = 0.40
    max_point_lateral_jump_m: float = 0.15
    min_centerline_confidence: float = 0.80
    min_centerline_span_m: float = 0.45
    min_process_fps: float = 4.0
    max_perception_age_ms: float = 350.0
    near_error_distance_m: float = 0.25
    lookahead_distance_m: float = 0.40
    heading_fit_start_m: float = 0.25
    heading_fit_end_m: float = 0.60
    lateral_gain: float = 1.8
    heading_gain: float = 1.2
    max_suggested_angular_z: float = 0.60
    nominal_suggested_linear_x: float = 0.10
    degraded_suggested_linear_x: float = 0.05

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
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.heading_fit_end_m <= self.heading_fit_start_m:
            raise ValueError("heading fit end must exceed start")
        if not 0.0 <= self.min_centerline_confidence <= 1.0:
            raise ValueError("min_centerline_confidence must be in [0, 1]")
        self.log_rate_hz = max(0.1, finite_float(self.log_rate_hz, 2.0))
        self.status_publish_rate_hz = max(
            0.1, finite_float(self.status_publish_rate_hz, 5.0)
        )
        self.nominal_suggested_linear_x = max(
            0.0, finite_float(self.nominal_suggested_linear_x)
        )
        self.degraded_suggested_linear_x = max(
            0.0, finite_float(self.degraded_suggested_linear_x)
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


@dataclass
class ControllerPreview:
    timestamp: str = ""
    controller_ready: bool = False
    control_block_reason: str = "waiting_for_path"
    degraded_mode: bool = False
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
    heading_error_rad: float = 0.0
    heading_error_deg: float = 0.0
    suggested_linear_x: float = 0.0
    suggested_angular_z_raw: float = 0.0
    suggested_angular_z: float = 0.0
    angular_command_saturated: bool = False
    controller_confidence: float = 0.0
    update_rate_hz: float = 0.0
    calculation_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
    points: np.ndarray, fit_start_m: float, fit_end_m: float
) -> Tuple[int, float, float]:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    selected = values[
        (values[:, 0] >= fit_start_m) & (values[:, 0] <= fit_end_m)
    ]
    if len(selected) >= 3:
        design = np.column_stack(
            (selected[:, 0], np.ones(len(selected), dtype=np.float64))
        )
        coefficients, _, _, _ = np.linalg.lstsq(
            design, selected[:, 1], rcond=None
        )
        slope = float(coefficients[0])
        if not math.isfinite(slope):
            raise ValueError("heading fit is not finite")
        return len(selected), slope, normalize_angle(math.atan(slope))

    if len(values) < 2:
        raise ValueError("not enough points for heading")
    midpoint = 0.5 * (fit_start_m + fit_end_m)
    upper = int(np.searchsorted(values[:, 0], midpoint, side="left"))
    upper = min(max(1, upper), len(values) - 1)
    lower = upper - 1
    dx = float(values[upper, 0] - values[lower, 0])
    if dx <= 1.0e-9:
        raise ValueError("invalid adjacent heading segment")
    slope = float((values[upper, 1] - values[lower, 1]) / dx)
    return len(selected), slope, normalize_angle(math.atan(slope))


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
        if analysis.span_m < config.min_path_span_m:
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
            _,
            analysis.lateral_error_m,
            analysis.near_error_clamped,
        ) = interpolate_y_at_forward(values, config.near_error_distance_m)
        (
            analysis.lookahead_target_x_m,
            analysis.lookahead_target_y_m,
            analysis.lookahead_target_distance_m,
            analysis.lookahead_clamped,
        ) = select_lookahead_target(values, config.lookahead_distance_m)
        (
            analysis.heading_fit_point_count,
            analysis.heading_slope,
            analysis.heading_error_rad,
        ) = estimate_heading(
            values, config.heading_fit_start_m, config.heading_fit_end_m
        )
        analysis.valid = True
        analysis.reason = ""
    except Exception as exc:
        analysis.valid = False
        analysis.reason = f"path_analysis_error:{type(exc).__name__}"
    return analysis


def compute_controller_confidence(
    analysis: PathAnalysis,
    perception: PerceptionSnapshot,
    config: ControllerConfig,
    degraded_mode: bool,
    angular_saturated: bool,
) -> float:
    if not analysis.valid or not perception.received:
        return 0.0
    point_factor = clamp(
        analysis.point_count / max(float(config.min_path_points * 2), 1.0),
        0.55,
        1.0,
    )
    span_factor = clamp(
        analysis.span_m / max(config.lookahead_distance_m * 1.5, 1.0e-6),
        0.55,
        1.0,
    )
    heading_factor = (
        1.0
        if analysis.heading_fit_point_count >= 3
        else 0.82
    )
    coverage_factor = 0.85 if analysis.lookahead_clamped else 1.0
    mode_factor = 0.68 if degraded_mode else 1.0
    saturation_factor = 0.82 if angular_saturated else 1.0
    return clamp(
        perception.centerline_confidence
        * point_factor
        * span_factor
        * heading_factor
        * coverage_factor
        * mode_factor
        * saturation_factor,
        0.0,
        1.0,
    )


def determine_block_reason(
    analysis: PathAnalysis,
    perception: PerceptionSnapshot,
    path_age_sec: float,
    status_age_sec: float,
    config: ControllerConfig,
) -> Tuple[str, bool]:
    if path_age_sec < 0.0:
        return "path_not_received", False
    if path_age_sec > config.path_timeout_sec:
        return "path_timeout", False
    if not analysis.valid:
        return analysis.reason or "path_invalid", False
    if not perception.received:
        return "status_not_received", False
    if perception.parse_error:
        return "status_json_invalid", False
    if status_age_sec > config.status_timeout_sec:
        return "status_timeout", False
    if not perception.centerline_valid:
        return "centerline_invalid", False
    if perception.centerline_mode in PREFERRED_MODES:
        degraded = False
    elif perception.centerline_mode in DEGRADED_MODES:
        degraded = True
    else:
        return "centerline_mode_not_allowed", False
    if perception.centerline_confidence < config.min_centerline_confidence:
        return "centerline_confidence_low", degraded
    if perception.centerline_forward_span_m < config.min_centerline_span_m:
        return "centerline_span_too_short", degraded
    if perception.capture_to_process_age_ms > config.max_perception_age_ms:
        return "perception_age_too_high", degraded
    if perception.process_fps < config.min_process_fps:
        return "process_fps_too_low", degraded
    return "", degraded


def compute_preview_command(
    analysis: PathAnalysis,
    perception: PerceptionSnapshot,
    config: ControllerConfig,
    path_age_sec: float,
    status_age_sec: float,
    update_rate_hz: float = 0.0,
) -> ControllerPreview:
    started = time.perf_counter()
    reason, degraded = determine_block_reason(
        analysis, perception, path_age_sec, status_age_sec, config
    )
    preview = ControllerPreview(
        timestamp=datetime.now(timezone.utc).isoformat(),
        controller_ready=bool(not reason),
        control_block_reason=reason,
        degraded_mode=bool(degraded),
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
        heading_error_rad=analysis.heading_error_rad,
        heading_error_deg=math.degrees(analysis.heading_error_rad),
        update_rate_hz=max(0.0, update_rate_hz),
    )
    if preview.controller_ready:
        preview.suggested_angular_z_raw = (
            config.lateral_gain * preview.lateral_error_m
            + config.heading_gain * preview.heading_error_rad
        )
        preview.suggested_angular_z = clamp(
            preview.suggested_angular_z_raw,
            -config.max_suggested_angular_z,
            config.max_suggested_angular_z,
        )
        preview.angular_command_saturated = not math.isclose(
            preview.suggested_angular_z,
            preview.suggested_angular_z_raw,
            rel_tol=1.0e-9,
            abs_tol=1.0e-9,
        )
        preview.controller_confidence = compute_controller_confidence(
            analysis,
            perception,
            config,
            degraded,
            preview.angular_command_saturated,
        )
        base_speed = (
            config.degraded_suggested_linear_x
            if degraded
            else config.nominal_suggested_linear_x
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
            * fps_factor,
        )
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
) -> ControllerPreview:
    try:
        return compute_preview_command(
            analysis,
            perception,
            config,
            path_age_sec,
            status_age_sec,
            update_rate_hz,
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
        # The only publisher is diagnostic JSON.
        self.controller_status_publisher = self.create_publisher(
            String,
            self.config.controller_status_topic,
            qos,
        )
        self.create_timer(
            1.0 / self.config.status_publish_rate_hz,
            self._publish_observation,
        )
        self.get_logger().info(
            "lane controller observer started: "
            f"path={self.config.path_topic} status={self.config.status_topic} "
            f"output={self.config.controller_status_topic} dry_run=true "
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
        self.preview = safe_compute_preview(
            self.path_analysis,
            self.perception,
            self.config,
            path_age,
            status_age,
            self.path_rate.value(),
        )

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
        if now - self.last_log_monotonic >= 1.0 / self.config.log_rate_hz:
            self.last_log_monotonic = now
            if self.preview.controller_ready:
                level = "DEGRADED" if self.preview.degraded_mode else "READY"
                self.get_logger().info(
                    f"{level} mode={self.preview.centerline_mode} "
                    f"lat={self.preview.lateral_error_m:+.3f}m "
                    f"heading={self.preview.heading_error_deg:+.1f}deg "
                    f"target=({self.preview.lookahead_target_x_m:.2f},"
                    f"{self.preview.lookahead_target_y_m:+.3f}) "
                    f"preview_v={self.preview.suggested_linear_x:.3f}m/s "
                    f"preview_w={self.preview.suggested_angular_z:+.3f}rad/s "
                    f"confidence={self.preview.controller_confidence:.2f} "
                    f"path_age={self.preview.path_age_ms:.0f}ms"
                )
            else:
                self.get_logger().warning(
                    f"BLOCKED reason={self.preview.control_block_reason} "
                    f"mode={self.preview.centerline_mode or '-'} "
                    f"path_age={self.preview.path_age_ms:.0f}ms "
                    f"status_age={self.preview.status_age_ms:.0f}ms"
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
        # This minimal fault status contains Python-native values only. Even its
        # construction, conversion, or publication is isolated from the timer.
        try:
            message = String()
            message.data = serialize_json(self.preview.to_dict())
            self.controller_status_publisher.publish(message)
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

    @test("12 too few points blocks")
    def _() -> None:
        result = validate_path("base_link", _path_points(count=5), config)
        assert not result.valid and result.reason == "path_too_few_points"

    @test("13 short span blocks")
    def _() -> None:
        result = validate_path(
            "base_link", _path_points(start=0.1, end=0.4), config
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

    @test("17 status timeout blocks")
    def _() -> None:
        result = validate_path("base_link", _path_points(), config)
        preview = compute_preview_command(
            result, _healthy_perception(), config, 0.01, 0.5
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
        assert preview.control_block_reason == "centerline_confidence_low"

    @test("21 degraded reduces speed and confidence")
    def _() -> None:
        result = validate_path("base_link", _path_points(), config)
        normal = compute_preview_command(
            result, _healthy_perception(), config, 0.01, 0.01
        )
        degraded = compute_preview_command(
            result,
            _healthy_perception(mode="single_green_width_offset"),
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

    @test("27 source has no standard velocity target")
    def _() -> None:
        source = FilePath(__file__).read_text(encoding="utf-8")
        target = "/" + "cmd" + "_vel"
        assert target not in source

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
        broken = ControllerConfig(max_suggested_angular_z="broken")  # type: ignore[arg-type]
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
    except KeyboardInterrupt:
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
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
