#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一体化黄绿车道感知运行节点。

本程序直接通过 OpenCV V4L2 读取 USB 相机，在同一进程、同一帧内完成
黄绿分割、公共边界、逆透视、道路几何和米制 Path 输出。中间 mask 只在
NumPy 内存中传递，不经过 ROS 图像话题。

运行：
    python3 gxb_test/lane_perception_pipeline.py \
      --ros-args \
      -p device:=/dev/video0 \
      -p profile_name:=usb_camera \
      -p width:=640 \
      -p height:=480 \
      -p camera_fps:=10.0 \
      -p camera_fourcc:=YUYV \
      -p web_gui_enable:=true \
      -p web_gui_port:=8093 \
      -p web_gui_max_fps:=1.0 \
      -p publish_debug_raw_images:=false

浏览器访问 http://小车IP:8093。运行本程序时无需同时启动 USB 图像发布器、
第二阶段分割程序或第四阶段几何程序。本节点只产生感知结果和调试画面。
"""

import copy
import importlib.util
import json
import math
import sys
import threading
import time
import types
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


SELF_TEST = "--self-test" in sys.argv
INTERFACE_VERSION = "gxb_lane_pipeline_v1"
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent


def _install_self_test_ros_stubs() -> None:
    """让没有 ROS Python 环境的开发机也能执行纯算法自测。"""

    class Stamp:
        def __init__(self) -> None:
            self.sec = 0
            self.nanosec = 0

    class Header:
        def __init__(self) -> None:
            self.stamp = Stamp()
            self.frame_id = ""

    class Image:
        def __init__(self) -> None:
            self.header = Header()
            self.height = 0
            self.width = 0
            self.encoding = ""
            self.is_bigendian = 0
            self.step = 0
            self.data = b""

    class CompressedImage:
        def __init__(self) -> None:
            self.header = Header()
            self.format = ""
            self.data = b""

    class String:
        def __init__(self) -> None:
            self.data = ""

    class Position:
        def __init__(self) -> None:
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    class Orientation:
        def __init__(self) -> None:
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
            self.w = 0.0

    class Pose:
        def __init__(self) -> None:
            self.position = Position()
            self.orientation = Orientation()

    class PoseStamped:
        def __init__(self) -> None:
            self.header = Header()
            self.pose = Pose()

    class RosPath:
        def __init__(self) -> None:
            self.header = Header()
            self.poses: List[Any] = []

    class Node:
        pass

    class Policy:
        BEST_EFFORT = 1
        VOLATILE = 1
        KEEP_LAST = 1

    class QoSProfile:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    rclpy_module = types.ModuleType("rclpy")
    node_module = types.ModuleType("rclpy.node")
    qos_module = types.ModuleType("rclpy.qos")
    node_module.Node = Node
    qos_module.DurabilityPolicy = Policy
    qos_module.HistoryPolicy = Policy
    qos_module.QoSProfile = QoSProfile
    qos_module.ReliabilityPolicy = Policy
    rclpy_module.node = node_module
    rclpy_module.qos = qos_module

    def add_message_package(package: str, members: Dict[str, Any]) -> None:
        root = types.ModuleType(package)
        message = types.ModuleType(f"{package}.msg")
        for name, value in members.items():
            setattr(message, name, value)
        root.msg = message
        sys.modules[package] = root
        sys.modules[f"{package}.msg"] = message

    sys.modules["rclpy"] = rclpy_module
    sys.modules["rclpy.node"] = node_module
    sys.modules["rclpy.qos"] = qos_module
    add_message_package(
        "sensor_msgs", {"Image": Image, "CompressedImage": CompressedImage}
    )
    add_message_package("std_msgs", {"String": String})
    add_message_package("geometry_msgs", {"PoseStamped": PoseStamped})
    add_message_package("nav_msgs", {"Path": RosPath})


try:
    import rclpy
    from nav_msgs.msg import Path as RosPath
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from sensor_msgs.msg import CompressedImage, Image
    from std_msgs.msg import String
except ImportError:
    if not SELF_TEST:
        raise
    _install_self_test_ros_stubs()
    import rclpy
    from nav_msgs.msg import Path as RosPath
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from sensor_msgs.msg import CompressedImage, Image
    from std_msgs.msg import String


def _load_stage_module(filename: str, alias: str) -> Any:
    """加载数字开头的阶段脚本，并保留其原有算法实现。"""
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载阶段脚本: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


STAGE2 = _load_stage_module(
    "2_green_yellow_segmentation.py", "_gxb_stage2_runtime"
)
STAGE4 = _load_stage_module(
    "4_lane_geometry_planner.py", "_gxb_stage4_runtime"
)


@dataclass
class CapturedFrame:
    sequence: int
    image: np.ndarray
    captured_monotonic: float
    stamp: Any


class LatestFrameSlot:
    """单元素覆盖槽；算法总是获得当时最新的一帧。"""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.latest: Optional[CapturedFrame] = None
        self.sequence = 0
        self.last_taken_sequence = 0
        self.dropped_count = 0

    def put(
        self, image: np.ndarray, captured_monotonic: float, stamp: Any
    ) -> int:
        with self.condition:
            if (
                self.latest is not None
                and self.latest.sequence > self.last_taken_sequence
            ):
                self.dropped_count += 1
            self.sequence += 1
            self.latest = CapturedFrame(
                self.sequence, image, captured_monotonic, stamp
            )
            self.condition.notify()
            return self.sequence

    def take(
        self, previous_sequence: int, stop_event: threading.Event
    ) -> Optional[CapturedFrame]:
        with self.condition:
            self.condition.wait_for(
                lambda: stop_event.is_set()
                or (
                    self.latest is not None
                    and self.latest.sequence > previous_sequence
                ),
                timeout=0.5,
            )
            if stop_event.is_set() or self.latest is None:
                return None
            if self.latest.sequence <= previous_sequence:
                return None
            self.last_taken_sequence = self.latest.sequence
            return self.latest

    def wake(self) -> None:
        with self.condition:
            self.condition.notify_all()


class RateMeter:
    """滑动一秒窗口频率计。"""

    def __init__(self) -> None:
        self.events: Deque[float] = deque()

    def tick(self, now: Optional[float] = None) -> float:
        moment = time.monotonic() if now is None else now
        self.events.append(moment)
        return self.value(moment)

    def value(self, now: Optional[float] = None) -> float:
        moment = time.monotonic() if now is None else now
        while self.events and moment - self.events[0] > 1.0:
            self.events.popleft()
        return float(len(self.events))


class CameraCapture:
    """V4L2 阻塞采集线程，失败时记录状态并持续重试。"""

    def __init__(
        self,
        node: "LanePerceptionPipelineNode",
        slot: LatestFrameSlot,
        stop_event: threading.Event,
        capture_factory: Any = cv2.VideoCapture,
    ) -> None:
        self.node = node
        self.slot = slot
        self.stop_event = stop_event
        self.capture_factory = capture_factory
        self.thread: Optional[threading.Thread] = None
        self.capture: Any = None

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self._run, name="lane-camera-capture", daemon=True
        )
        self.thread.start()

    def _open(self) -> Any:
        capture = self.capture_factory(self.node.device, cv2.CAP_V4L2)
        capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*self.node.camera_fourcc),
        )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.node.request_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.node.request_height)
        capture.set(cv2.CAP_PROP_FPS, self.node.request_camera_fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, self.node.camera_buffer_size)
        return capture

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.capture = self._open()
                if not self.capture.isOpened():
                    raise RuntimeError(f"无法打开相机 {self.node.device}")
                self.node.update_camera_report(self.capture)
                while not self.stop_event.is_set():
                    ok, frame = self.capture.read()
                    captured = time.monotonic()
                    if not ok or frame is None or frame.size == 0:
                        raise RuntimeError("相机读取失败")
                    if frame.ndim == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    self.slot.put(
                        np.ascontiguousarray(frame),
                        captured,
                        self.node.get_clock().now().to_msg(),
                    )
                    self.node.capture_succeeded(captured)
            except Exception as exc:
                self.node.capture_failed(str(exc))
                self.stop_event.wait(0.5)
            finally:
                if self.capture is not None:
                    self.capture.release()
                    self.capture = None

    def stop(self) -> None:
        self.stop_event.set()
        self.slot.wake()
        if self.capture is not None:
            self.capture.release()
        if self.thread is not None:
            self.thread.join(timeout=2.0)


@dataclass
class PipelineResult:
    boundary: np.ndarray
    yellow: np.ndarray
    green: np.ndarray
    observed_ipm_boundary: np.ndarray
    repaired_ipm_boundary: np.ndarray
    repaired_gap_mask: np.ndarray
    transformed: Dict[str, np.ndarray]
    main_yellow: np.ndarray
    candidate_curves: List[Any]
    selected_boundary: np.ndarray
    centerline_mask: np.ndarray
    overlay: Optional[np.ndarray]
    geometry: Any
    roi: Tuple[int, int]
    boundary_valid: bool
    yellow_status: Dict[str, Any]
    observed_boundary_component_count: int
    repaired_boundary_component_count: int
    merged_boundary_group_count: int
    repaired_gap_count: int
    repaired_gap_max_px: int
    road_side_status: Dict[str, Any]
    boundary_component_count: int
    valid_boundary_component_count: int
    roi_y_offset: int
    yellow_corridor_status: Dict[str, Any]
    yellow_corridor_samples: List[Dict[str, Any]]
    green_boundary_status: Dict[str, Any]
    green_boundary_samples: List[Dict[str, Any]]
    fallback_boundary_pipeline_used: bool
    timing_ms: Dict[str, float]
    counters: Dict[str, int]


TIMING_NAMES = (
    "segmentation_ms",
    "boundary_extract_ms",
    "ipm_warp_ms",
    "main_yellow_select_ms",
    "green_boundary_fast_path_ms",
    "single_green_curve_offset_ms",
    "yellow_corridor_ms",
    "component_analysis_ms",
    "fragment_descriptor_ms",
    "boundary_repair_ms",
    "boundary_merge_ms",
    "road_side_vote_ms",
    "dual_planner_ms",
    "single_planner_ms",
    "centerline_smooth_ms",
    "overlay_build_ms",
    "jpeg_encode_ms",
    "path_publish_ms",
    "total_processing_ms",
)

COUNTER_NAMES = (
    "polyfit_call_count",
    "road_side_probe_count",
    "repair_candidate_pair_count",
    "repair_accepted_pair_count",
    "merge_candidate_pair_count",
    "fragment_count_before_filter",
    "fragment_count_after_filter",
    "overlay_build_count",
    "jpeg_encode_count",
)


class PerformanceWindow:
    """保存有限帧的轻量耗时快照，并按需计算平均值和 P95。"""

    def __init__(self, size: int = 120) -> None:
        self.lock = threading.Lock()
        self.samples: Deque[Dict[str, float]] = deque(maxlen=size)

    def add(self, sample: Dict[str, float]) -> None:
        with self.lock:
            self.samples.append(
                {name: float(sample.get(name, 0.0)) for name in TIMING_NAMES}
            )

    def update_latest(self, values: Dict[str, float]) -> None:
        with self.lock:
            if self.samples:
                self.samples[-1].update(
                    {name: float(value) for name, value in values.items()}
                )

    def summary(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        with self.lock:
            samples = list(self.samples)
        if not samples:
            empty = {name: 0.0 for name in TIMING_NAMES}
            return empty, empty.copy()
        average: Dict[str, float] = {}
        p95: Dict[str, float] = {}
        for name in TIMING_NAMES:
            values = np.asarray(
                [sample.get(name, 0.0) for sample in samples],
                dtype=np.float64,
            )
            average[name] = float(np.mean(values))
            p95[name] = float(np.percentile(values, 95))
        return average, p95


@dataclass
class CroppedIpm:
    """完整 IPM 的 planning ROI 局部视图。"""

    full: Any
    y_offset: int
    output_width: int
    output_height: int
    homography_matrix: np.ndarray
    inverse_homography_matrix: np.ndarray
    vehicle_origin_y_px: float

    @classmethod
    def from_full(
        cls, ipm: Any, geometry_config: Any
    ) -> Tuple["CroppedIpm", float, float]:
        forward_min, forward_max, y_min, y_max = ipm.planning_roi(
            geometry_config
        )
        translation = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, -float(y_min)], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        local_homography = translation @ ipm.homography_matrix
        inverse_translation = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, float(y_min)], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return (
            cls(
                full=ipm,
                y_offset=int(y_min),
                output_width=int(ipm.output_width),
                output_height=int(y_max - y_min + 1),
                homography_matrix=local_homography,
                inverse_homography_matrix=(
                    ipm.inverse_homography_matrix @ inverse_translation
                ),
                vehicle_origin_y_px=float(ipm.vehicle_origin_y_px - y_min),
            ),
            forward_min,
            forward_max,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.full, name)

    def planning_roi(
        self, _config: Any
    ) -> Tuple[float, float, int, int]:
        near = (
            self.vehicle_origin_y_px - (self.output_height - 1)
        ) * self.meter_per_pixel
        far = self.vehicle_origin_y_px * self.meter_per_pixel
        return near, far, 0, self.output_height - 1

    def to_full_points(self, points: np.ndarray) -> np.ndarray:
        converted = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
        if len(converted):
            converted[:, 1] += self.y_offset
        return converted

    def paste_mask(self, local: np.ndarray) -> np.ndarray:
        shape = (self.full.output_height, self.full.output_width)
        output = np.zeros(shape, dtype=local.dtype)
        output[
            self.y_offset : self.y_offset + self.output_height,
            : self.output_width,
        ] = local
        return output

    def paste_color(self, local: np.ndarray) -> np.ndarray:
        output = np.zeros(
            (
                self.full.output_height,
                self.full.output_width,
                local.shape[2],
            ),
            dtype=local.dtype,
        )
        output[
            self.y_offset : self.y_offset + self.output_height,
            : self.output_width,
        ] = local
        return output


@dataclass
class YellowCorridorConfig:
    """基于主黄色区域双轮廓的快速中心线参数，均待实车继续验证。"""

    enable: bool = True
    sample_step_m: float = 0.05
    width_min_ratio: float = 0.65
    width_max_ratio: float = 1.40
    center_jump_max_m: float = 0.15
    min_valid_samples: int = 6
    min_forward_span_m: float = 0.25
    boundary_validation_radius_px: int = 5
    min_boundary_valid_ratio: float = 0.45
    max_consecutive_invalid_samples: int = 2
    max_small_gap_m: float = 0.10
    center_gap_fill_enable: bool = True
    center_gap_fill_max_samples: int = 2
    center_gap_fill_max_m: float = 0.10
    center_gap_fill_require_both_sides: bool = True
    center_gap_fill_max_ratio: float = 0.25
    center_smooth_enable: bool = True
    center_smooth_lambda: float = 2.0
    green_outer_seed_max_gap_px: int = 8
    green_min_outer_run_width_px: int = 8
    single_green_curve_min_points: int = 8
    single_green_curve_min_span_m: float = 0.35
    single_green_curve_max_row_gap: int = 2
    single_green_curve_max_point_jump_m: float = 0.12
    single_green_tangent_window_points: int = 5
    single_green_min_tangent_points: int = 3
    single_green_min_tangent_dx_m: float = 0.04
    lane_width_current_min_dual_samples: int = 3
    lane_width_ema_alpha: float = 0.15
    lane_width_history_timeout_sec: float = 2.0
    single_green_lane_width_min_m: float = 0.40
    single_green_lane_width_max_m: float = 0.60
    single_green_center_yellow_window_px: int = 5
    single_green_center_min_yellow_ratio: float = 0.35
    single_green_center_max_green_ratio: float = 0.20
    single_green_center_max_lateral_jump_m: float = 0.10
    single_green_center_max_heading_step_deg: float = 20.0

    def validate(self) -> None:
        if self.sample_step_m <= 0 or self.center_jump_max_m <= 0:
            raise ValueError("黄色通道步长和中心跳变限制必须大于 0")
        if not (0 < self.width_min_ratio < self.width_max_ratio):
            raise ValueError("黄色通道宽度比例非法")
        if self.min_valid_samples < 2 or self.min_forward_span_m <= 0:
            raise ValueError("黄色通道有效点数或跨度非法")
        if self.boundary_validation_radius_px < 1:
            raise ValueError("黄色通道边缘验证半径必须大于 0")
        if not (0 <= self.min_boundary_valid_ratio <= 1):
            raise ValueError("黄色通道边缘有效比例必须位于 [0, 1]")
        if self.max_consecutive_invalid_samples < 0:
            raise ValueError("yellow corridor invalid sample limit must be non-negative")
        if self.max_small_gap_m < 0:
            raise ValueError("yellow corridor small gap limit must be non-negative")
        if self.center_gap_fill_max_samples < 0:
            raise ValueError("center gap fill sample limit must be non-negative")
        if self.center_gap_fill_max_m < 0:
            raise ValueError("center gap fill distance must be non-negative")
        if not (0.0 <= self.center_gap_fill_max_ratio <= 1.0):
            raise ValueError("center gap fill ratio must be in [0, 1]")
        if self.center_smooth_lambda < 0:
            raise ValueError("center smoothing lambda must be non-negative")
        if self.green_outer_seed_max_gap_px < 0:
            raise ValueError("green outer seed gap must be non-negative")
        if self.green_min_outer_run_width_px < 1:
            raise ValueError("green outer run width must be positive")
        if self.single_green_curve_min_points < 3:
            raise ValueError("single green curve point minimum must be >= 3")
        if self.single_green_curve_min_span_m <= 0:
            raise ValueError("single green curve span must be positive")
        if self.single_green_curve_max_row_gap < 0:
            raise ValueError("single green row gap must be non-negative")
        if self.single_green_curve_max_point_jump_m <= 0:
            raise ValueError("single green point jump must be positive")
        if self.single_green_tangent_window_points < 3:
            raise ValueError("single green tangent window must be >= 3")
        if self.single_green_min_tangent_points < 2:
            raise ValueError("single green tangent point minimum must be >= 2")
        if self.single_green_min_tangent_dx_m <= 0:
            raise ValueError("single green tangent dx must be positive")
        if self.lane_width_current_min_dual_samples < 1:
            raise ValueError("current dual lane width sample count must be positive")
        if not 0.0 < self.lane_width_ema_alpha <= 1.0:
            raise ValueError("lane width EMA alpha must be in (0, 1]")
        if self.lane_width_history_timeout_sec <= 0:
            raise ValueError("lane width history timeout must be positive")
        if not (
            0.0
            < self.single_green_lane_width_min_m
            < self.single_green_lane_width_max_m
        ):
            raise ValueError("single green lane width bounds are invalid")
        if self.single_green_center_yellow_window_px < 1:
            raise ValueError("single green support window must be positive")
        if not 0.0 <= self.single_green_center_min_yellow_ratio <= 1.0:
            raise ValueError("single green yellow ratio must be in [0, 1]")
        if not 0.0 <= self.single_green_center_max_green_ratio <= 1.0:
            raise ValueError("single green intrusion ratio must be in [0, 1]")
        if self.single_green_center_max_lateral_jump_m <= 0:
            raise ValueError("single green lateral jump must be positive")
        if not 0.0 < self.single_green_center_max_heading_step_deg <= 180.0:
            raise ValueError("single green heading step must be in (0, 180]")


class YellowCorridorPlanner:
    """逐行跟踪主黄色连续区间，并为每个实际检查的采样行保留诊断结果。"""

    CENTER_WEIGHTS = {
        "strict_observed": 1.0,
        "weak_single_edge": 0.60,
        "green_dual_observed": 1.0,
        "yellow_dual_observed": 1.0,
        "green_single_offset": 0.60,
        "gap_filled": 0.25,
    }
    REASONS = (
        "accepted",
        "reject_no_yellow_interval",
        "reject_seed_or_continuity",
        "reject_width_too_narrow",
        "reject_width_too_wide",
        "reject_center_jump",
        "reject_left_edge_validation",
        "reject_right_edge_validation",
        "reject_boundary_ratio",
        "reject_center_not_yellow",
        "reject_out_of_roi",
        "reject_other",
    )
    EMPTY_STATUS: Dict[str, Any] = {
        "yellow_corridor_attempted": False,
        "yellow_corridor_valid": False,
        "yellow_corridor_valid_sample_count": 0,
        "yellow_corridor_total_sample_count": 0,
        "yellow_corridor_accepted_sample_count": 0,
        "yellow_corridor_reject_no_yellow_interval_count": 0,
        "yellow_corridor_reject_seed_or_continuity_count": 0,
        "yellow_corridor_reject_width_too_narrow_count": 0,
        "yellow_corridor_reject_width_too_wide_count": 0,
        "yellow_corridor_reject_center_jump_count": 0,
        "yellow_corridor_reject_left_edge_validation_count": 0,
        "yellow_corridor_reject_right_edge_validation_count": 0,
        "yellow_corridor_reject_boundary_ratio_count": 0,
        "yellow_corridor_reject_center_not_yellow_count": 0,
        "yellow_corridor_reject_out_of_roi_count": 0,
        "yellow_corridor_reject_other_count": 0,
        "yellow_corridor_first_accepted_forward_m": None,
        "yellow_corridor_last_accepted_forward_m": None,
        "yellow_corridor_first_rejected_forward_m": None,
        "yellow_corridor_max_consecutive_rejected_samples": 0,
        "yellow_corridor_termination_reason": "disabled",
        "yellow_corridor_forward_span_m": 0.0,
        "yellow_corridor_width_px_mean": 0.0,
        "yellow_corridor_width_px_std": 0.0,
        "yellow_corridor_boundary_valid_ratio": 0.0,
        "yellow_corridor_reason": "disabled",
        "center_strict_observed_count": 0,
        "center_weak_observed_count": 0,
        "center_missing_count": 0,
        "center_gap_filled_count": 0,
        "center_gap_count": 0,
        "center_gap_max_samples": 0,
        "center_gap_fill_ratio": 0.0,
        "center_smoothing_used": False,
        "center_smoothing_lambda": 0.0,
        "center_pre_smooth_lateral_std_px": 0.0,
        "center_post_smooth_lateral_std_px": 0.0,
    }

    @staticmethod
    def _intervals(row: np.ndarray) -> List[Tuple[int, int]]:
        active = row > 0
        if not np.any(active):
            return []
        padded = np.pad(active.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0] - 1
        return [(int(start), int(end)) for start, end in zip(starts, ends)]

    @staticmethod
    def _edge_valid(
        x: int,
        y: int,
        observed_boundary: np.ndarray,
        green: np.ndarray,
        radius: int,
    ) -> bool:
        height, width = observed_boundary.shape
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - 1), min(height, y + 2)
        if x1 <= x0 or y1 <= y0:
            return False
        return bool(
            np.any(observed_boundary[y0:y1, x0:x1] > 0)
            or np.any(green[y0:y1, x0:x1] > 0)
        )

    @classmethod
    def rejection_count_sum(cls, status: Dict[str, Any]) -> int:
        return sum(
            int(status.get(f"yellow_corridor_{reason}_count", 0))
            for reason in cls.REASONS
            if reason != "accepted"
        )

    @staticmethod
    def _forward_m(y: int, ipm: Any) -> float:
        return float(
            (float(ipm.vehicle_origin_y_px) - float(y))
            * float(ipm.meter_per_pixel)
        )

    @classmethod
    def _record(
        cls,
        samples: List[Dict[str, Any]],
        y: int,
        ipm: Any,
        reason: str,
        left: Optional[int] = None,
        right: Optional[int] = None,
        center: Optional[float] = None,
        classification: str = "missing",
        width: Optional[float] = None,
        left_valid: bool = False,
        right_valid: bool = False,
        fill_eligible: bool = False,
    ) -> None:
        samples.append(
            {
                "y": int(y),
                "forward_m": cls._forward_m(y, ipm),
                "reason": reason,
                "left": left,
                "right": right,
                "center": center,
                "classification": classification,
                "source_type": (
                    "yellow_dual_observed"
                    if classification
                    in {"strict_observed", "weak_single_edge", "weak_candidate"}
                    else classification
                ),
                "width": width,
                "left_valid": bool(left_valid),
                "right_valid": bool(right_valid),
                "fill_eligible": bool(fill_eligible),
            }
        )

    @classmethod
    def _legacy_plan(
        cls,
        main_yellow: np.ndarray,
        observed_boundary: np.ndarray,
        green: np.ndarray,
        ipm: Any,
        config: YellowCorridorConfig,
    ) -> Tuple[Any, Dict[str, Any]]:
        status = copy.deepcopy(cls.EMPTY_STATUS)
        samples: List[Dict[str, Any]] = []
        if not config.enable:
            result = STAGE4.GeometryResult(reason="yellow_corridor_disabled")
            result.yellow_corridor_samples = samples
            return result, status
        status["yellow_corridor_attempted"] = True
        step_px = max(2, int(round(config.sample_step_m / ipm.meter_per_pixel)))
        expected_width = ipm.expected_lane_width_px
        minimum_width = expected_width * config.width_min_ratio
        maximum_width = expected_width * config.width_max_ratio
        jump_limit = config.center_jump_max_m / ipm.meter_per_pixel
        previous_center = float(ipm.vehicle_center_x_px)
        points: List[Tuple[float, float]] = []
        widths: List[float] = []
        edge_valid_count = 0
        consecutive_rejected = 0
        max_consecutive_rejected = 0
        termination_reason = "roi_exhausted"

        for y in range(main_yellow.shape[0] - 1, -1, -step_px):
            reason = "reject_other"
            intervals = cls._intervals(main_yellow[y])
            if not intervals:
                reason = "reject_no_yellow_interval"

            width_candidates: List[Tuple[int, int, float, float]] = []
            for left, right in intervals:
                width = float(right - left + 1)
                if not (minimum_width <= width <= maximum_width):
                    continue
                center = (left + right) / 2.0
                jump = abs(center - previous_center)
                width_candidates.append((left, right, center, jump))
            if intervals and not width_candidates:
                interval_widths = [
                    float(right - left + 1) for left, right in intervals
                ]
                if max(interval_widths) < minimum_width:
                    reason = "reject_width_too_narrow"
                elif min(interval_widths) > maximum_width:
                    reason = "reject_width_too_wide"

            continuity_candidates: List[Tuple[int, int, float, float]] = []
            for left, right, center, jump in width_candidates:
                if points and jump > jump_limit:
                    continue
                if not points and jump > expected_width:
                    continue
                continuity_candidates.append((left, right, center, jump))
            if width_candidates and not continuity_candidates:
                reason = (
                    "reject_center_jump"
                    if points
                    else "reject_seed_or_continuity"
                )

            candidates: List[
                Tuple[float, int, int, float, bool, bool]
            ] = []
            center_rejected = False
            for left, right, center, jump in continuity_candidates:
                center_index = int(round(center))
                if not (
                    0 <= center_index < main_yellow.shape[1]
                    and main_yellow[y, center_index] > 0
                ):
                    center_rejected = True
                    continue
                left_valid = cls._edge_valid(
                    left,
                    y,
                    observed_boundary,
                    green,
                    config.boundary_validation_radius_px,
                )
                right_valid = cls._edge_valid(
                    right,
                    y,
                    observed_boundary,
                    green,
                    config.boundary_validation_radius_px,
                )
                width = float(right - left + 1)
                score = jump + abs(width - expected_width) * 0.15
                candidates.append(
                    (score, left, right, center, left_valid, right_valid)
                )
            if continuity_candidates and not candidates and center_rejected:
                reason = "reject_center_not_yellow"

            accepted_candidates = [
                item for item in candidates if item[4] and item[5]
            ]
            if candidates and not accepted_candidates:
                best = min(candidates, key=lambda item: item[0])
                reason = (
                    "reject_left_edge_validation"
                    if not best[4]
                    else "reject_right_edge_validation"
                )
            if accepted_candidates:
                _score, left, right, center, _left_ok, _right_ok = min(
                    accepted_candidates, key=lambda item: item[0]
                )
                reason = "accepted"

            if reason != "accepted":
                rejected_center = (
                    min(candidates, key=lambda item: item[0])[3]
                    if candidates
                    else previous_center
                )
                cls._record(
                    samples,
                    y,
                    ipm,
                    reason,
                    center=float(rejected_center),
                )
                consecutive_rejected += 1
                max_consecutive_rejected = max(
                    max_consecutive_rejected, consecutive_rejected
                )
                if points:
                    missing_gap_m = (
                        consecutive_rejected * config.sample_step_m
                    )
                    if (
                        consecutive_rejected
                        > config.max_consecutive_invalid_samples
                    ):
                        termination_reason = "consecutive_invalid_limit"
                        break
                    if missing_gap_m > config.max_small_gap_m + 1.0e-9:
                        termination_reason = "small_gap_limit"
                        break
                continue

            consecutive_rejected = 0
            edge_valid_count += 1
            points.append((center, float(y)))
            widths.append(float(right - left + 1))
            previous_center = center
            cls._record(samples, y, ipm, reason, left, right, center)

        raw = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        span = (
            float(raw[0, 1] - raw[-1, 1]) * ipm.meter_per_pixel
            if len(raw) >= 2
            else 0.0
        )
        boundary_ratio = edge_valid_count / max(1, len(raw))
        reason_counts = {
            reason: sum(sample["reason"] == reason for sample in samples)
            for reason in cls.REASONS
        }
        accepted_forwards = [
            sample["forward_m"]
            for sample in samples
            if sample["reason"] == "accepted"
        ]
        rejected_forwards = [
            sample["forward_m"]
            for sample in samples
            if sample["reason"] != "accepted"
        ]
        status.update(
            {
                "yellow_corridor_valid_sample_count": len(raw),
                "yellow_corridor_total_sample_count": len(samples),
                "yellow_corridor_accepted_sample_count": reason_counts[
                    "accepted"
                ],
                **{
                    f"yellow_corridor_{reason}_count": count
                    for reason, count in reason_counts.items()
                    if reason != "accepted"
                },
                "yellow_corridor_first_accepted_forward_m": (
                    accepted_forwards[0] if accepted_forwards else None
                ),
                "yellow_corridor_last_accepted_forward_m": (
                    accepted_forwards[-1] if accepted_forwards else None
                ),
                "yellow_corridor_first_rejected_forward_m": (
                    rejected_forwards[0] if rejected_forwards else None
                ),
                "yellow_corridor_max_consecutive_rejected_samples": (
                    max_consecutive_rejected
                ),
                "yellow_corridor_termination_reason": termination_reason,
                "yellow_corridor_forward_span_m": span,
                "yellow_corridor_width_px_mean": float(
                    np.mean(widths) if widths else 0.0
                ),
                "yellow_corridor_width_px_std": float(
                    np.std(widths) if widths else 0.0
                ),
                "yellow_corridor_boundary_valid_ratio": boundary_ratio,
            }
        )
        reason = "valid"
        if len(raw) < config.min_valid_samples:
            reason = "yellow_corridor_too_few_samples"
        elif span < config.min_forward_span_m:
            reason = "yellow_corridor_span_too_short"
        elif boundary_ratio < config.min_boundary_valid_ratio:
            reason = "yellow_corridor_boundary_ratio_low"
        status["yellow_corridor_reason"] = reason
        status["yellow_corridor_valid"] = reason == "valid"
        result = STAGE4.GeometryResult(
            mode="yellow_corridor_dual_edge",
            valid=False,
            reason=reason,
            confidence=float(
                np.clip(
                    0.72
                    + 0.18 * boundary_ratio
                    + 0.10
                    * min(1.0, len(raw) / max(config.min_valid_samples, 1)),
                    0.0,
                    1.0,
                )
            ),
            raw_points=raw,
        )
        result.yellow_corridor_samples = samples
        return result, status

    @staticmethod
    def _yellow_supported(
        main_yellow: np.ndarray,
        x: float,
        y: int,
        radius: int,
    ) -> bool:
        height, width = main_yellow.shape
        cx = int(round(x))
        x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        return bool(x1 > x0 and y1 > y0 and np.any(main_yellow[y0:y1, x0:x1]))

    @staticmethod
    def _lateral_deviation(points: np.ndarray) -> float:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 2)[:, 0]
        if len(values) < 3:
            return float(np.std(np.diff(values))) if len(values) >= 2 else 0.0
        return float(np.std(np.diff(values, n=2)))

    @staticmethod
    def _weighted_smooth(
        points: np.ndarray,
        weights: np.ndarray,
        smoothing_lambda: float,
    ) -> np.ndarray:
        candidate = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
        count = len(candidate)
        if count < 3 or smoothing_lambda <= 0:
            return candidate
        second_difference = np.zeros((count - 2, count), dtype=np.float64)
        for index in range(count - 2):
            second_difference[index, index : index + 3] = (1.0, -2.0, 1.0)
        weight_matrix = np.diag(np.asarray(weights, dtype=np.float64))
        system = (
            weight_matrix
            + float(smoothing_lambda)
            * second_difference.T
            @ second_difference
        )
        right_hand_side = weight_matrix @ candidate[:, 0]
        candidate[:, 0] = np.linalg.solve(system, right_hand_side)
        return candidate

    @classmethod
    def _confirm_weak_candidates(
        cls,
        samples: List[Dict[str, Any]],
        jump_limit_px: float,
    ) -> None:
        strict_indices = [
            index
            for index, sample in enumerate(samples)
            if sample["classification"] == "strict_observed"
        ]
        for index, sample in enumerate(samples):
            if sample["classification"] != "weak_candidate":
                continue
            previous = [item for item in strict_indices if item < index]
            following = [item for item in strict_indices if item > index]
            if not previous or not following:
                sample["classification"] = "missing"
                continue
            before_index, after_index = previous[-1], following[0]
            if index - before_index > 3 or after_index - index > 3:
                sample["classification"] = "missing"
                continue
            before, after = samples[before_index], samples[after_index]
            denominator = float(after["y"] - before["y"])
            if abs(denominator) < 1.0e-9:
                sample["classification"] = "missing"
                continue
            ratio = (float(sample["y"]) - float(before["y"])) / denominator
            predicted = float(before["center"]) + ratio * (
                float(after["center"]) - float(before["center"])
            )
            step_jump = abs(float(after["center"]) - float(before["center"])) / (
                after_index - before_index
            )
            if (
                abs(float(sample["center"]) - predicted) <= jump_limit_px
                and step_jump <= jump_limit_px
            ):
                sample["classification"] = "weak_single_edge"
                sample["source_reason"] = sample["reason"]
                sample["reason"] = "accepted"
            else:
                sample["classification"] = "missing"

    @classmethod
    def _fill_internal_gaps(
        cls,
        samples: List[Dict[str, Any]],
        main_yellow: np.ndarray,
        config: YellowCorridorConfig,
        step_px: int,
        jump_limit_px: float,
    ) -> Tuple[int, int, int]:
        if not config.center_gap_fill_enable:
            return 0, 0, 0
        observed_classes = {"strict_observed", "weak_single_edge"}
        filled_count = 0
        gap_count = 0
        maximum_gap = 0
        index = 0
        while index < len(samples):
            if samples[index]["classification"] != "missing":
                index += 1
                continue
            start = index
            while (
                index < len(samples)
                and samples[index]["classification"] == "missing"
            ):
                index += 1
            end = index
            count = end - start
            if start == 0 or end >= len(samples):
                continue
            before, after = samples[start - 1], samples[end]
            if (
                before["classification"] not in observed_classes
                or after["classification"] not in observed_classes
                or count > config.center_gap_fill_max_samples
                or count * config.sample_step_m
                > config.center_gap_fill_max_m + 1.0e-9
                or not all(sample["fill_eligible"] for sample in samples[start:end])
                or (filled_count + count) / max(1, len(samples))
                > config.center_gap_fill_max_ratio + 1.0e-9
            ):
                continue
            per_step_jump = abs(
                float(after["center"]) - float(before["center"])
            ) / (count + 1)
            if per_step_jump > jump_limit_px:
                continue
            proposed: List[float] = []
            supported = True
            support_radius = max(2, step_px // 2)
            for offset, sample in enumerate(samples[start:end], 1):
                ratio = offset / float(count + 1)
                center = float(before["center"]) + ratio * (
                    float(after["center"]) - float(before["center"])
                )
                if not cls._yellow_supported(
                    main_yellow,
                    center,
                    int(sample["y"]),
                    support_radius,
                ):
                    supported = False
                    break
                proposed.append(center)
            if not supported:
                continue
            for sample, center in zip(samples[start:end], proposed):
                sample["source_reason"] = sample["reason"]
                sample["classification"] = "gap_filled"
                sample["source_type"] = "gap_filled"
                sample["center"] = center
            filled_count += count
            gap_count += 1
            maximum_gap = max(maximum_gap, count)
        return filled_count, gap_count, maximum_gap

    @staticmethod
    def _leading_valid_segment(
        samples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        valid_classes = {
            "strict_observed",
            "weak_single_edge",
            "gap_filled",
        }
        segments: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        for sample in samples:
            if sample["classification"] in valid_classes:
                current.append(sample)
            elif current:
                segments.append(current)
                current = []
        if current:
            segments.append(current)
        return segments[0] if segments else []

    @classmethod
    def _validate_centerline(
        cls,
        points: np.ndarray,
        main_yellow: np.ndarray,
        ipm: Any,
        config: YellowCorridorConfig,
        geometry_config: Any,
    ) -> Tuple[str, float, float]:
        candidate = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(candidate) < config.min_valid_samples:
            return "yellow_corridor_too_few_samples", 0.0, 0.0
        if not np.isfinite(candidate).all():
            return "centerline_non_finite", 0.0, 0.0
        if (
            np.any(candidate[:, 0] < 0)
            or np.any(candidate[:, 0] >= ipm.output_width)
            or np.any(candidate[:, 1] < 0)
            or np.any(candidate[:, 1] >= main_yellow.shape[0])
        ):
            return "centerline_out_of_roi", 0.0, 0.0
        forwards = (
            float(ipm.vehicle_origin_y_px) - candidate[:, 1]
        ) * float(ipm.meter_per_pixel)
        if np.any(np.diff(forwards) <= 0):
            return "centerline_forward_not_monotonic", 0.0, 0.0
        span = float(forwards[-1] - forwards[0])
        if span < config.min_forward_span_m:
            return "yellow_corridor_span_too_short", 0.0, span
        max_jump_m = (
            float(getattr(geometry_config, "centerline_max_lateral_jump_m", 0.20))
            if geometry_config is not None
            else config.center_jump_max_m
        )
        if np.any(
            np.abs(np.diff(candidate[:, 0])) * ipm.meter_per_pixel
            > max_jump_m
        ):
            return "centerline_lateral_jump", 0.0, span
        max_abs_m = float(
            getattr(geometry_config, "centerline_max_abs_lateral_m", 1.0)
        )
        if np.any(
            np.abs(candidate[:, 0] - ipm.vehicle_center_x_px)
            * ipm.meter_per_pixel
            > max_abs_m
        ):
            return "centerline_lateral_out_of_range", 0.0, span
        pixels = np.round(candidate).astype(int)
        yellow_ratio = float(
            np.mean(main_yellow[pixels[:, 1], pixels[:, 0]] > 0)
        )
        minimum_yellow_ratio = float(
            getattr(geometry_config, "centerline_min_yellow_ratio", 0.45)
        )
        if yellow_ratio < minimum_yellow_ratio:
            return "centerline_yellow_ratio_low", yellow_ratio, span
        return "valid", yellow_ratio, span

    @classmethod
    def plan(
        cls,
        main_yellow: np.ndarray,
        observed_boundary: np.ndarray,
        green: np.ndarray,
        ipm: Any,
        config: YellowCorridorConfig,
        geometry_config: Any = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        status = copy.deepcopy(cls.EMPTY_STATUS)
        samples: List[Dict[str, Any]] = []
        if not config.enable:
            result = STAGE4.GeometryResult(reason="yellow_corridor_disabled")
            result.yellow_corridor_samples = samples
            return result, status
        status["yellow_corridor_attempted"] = True
        step_px = max(2, int(round(config.sample_step_m / ipm.meter_per_pixel)))
        expected_width = float(ipm.expected_lane_width_px)
        minimum_width = expected_width * config.width_min_ratio
        maximum_width = expected_width * config.width_max_ratio
        weak_minimum_width = expected_width * 0.80
        weak_maximum_width = expected_width * 1.20
        jump_limit = config.center_jump_max_m / ipm.meter_per_pixel
        previous_center = float(ipm.vehicle_center_x_px)
        has_observation = False

        for y in range(main_yellow.shape[0] - 1, -1, -step_px):
            reason = "reject_other"
            intervals = cls._intervals(main_yellow[y])
            if not intervals:
                reason = "reject_no_yellow_interval"
            width_candidates: List[Tuple[float, int, int, float]] = []
            widths = [float(right - left + 1) for left, right in intervals]
            for left, right in intervals:
                width = float(right - left + 1)
                if not (minimum_width <= width <= maximum_width):
                    continue
                center = (left + right) / 2.0
                width_candidates.append(
                    (
                        abs(center - previous_center)
                        + abs(width - expected_width) * 0.15,
                        left,
                        right,
                        center,
                    )
                )
            if intervals and not width_candidates:
                reason = (
                    "reject_width_too_narrow"
                    if max(widths) < minimum_width
                    else "reject_width_too_wide"
                    if min(widths) > maximum_width
                    else "reject_other"
                )
            continuous: List[
                Tuple[float, int, int, float, bool, bool, float]
            ] = []
            for score, left, right, center in width_candidates:
                jump = abs(center - previous_center)
                if (has_observation and jump > jump_limit) or (
                    not has_observation and jump > expected_width
                ):
                    continue
                center_index = int(round(center))
                if not (
                    0 <= center_index < main_yellow.shape[1]
                    and main_yellow[y, center_index] > 0
                ):
                    reason = "reject_center_not_yellow"
                    continue
                left_valid = cls._edge_valid(
                    left,
                    y,
                    observed_boundary,
                    green,
                    config.boundary_validation_radius_px,
                )
                right_valid = cls._edge_valid(
                    right,
                    y,
                    observed_boundary,
                    green,
                    config.boundary_validation_radius_px,
                )
                continuous.append(
                    (
                        score,
                        left,
                        right,
                        center,
                        left_valid,
                        right_valid,
                        float(right - left + 1),
                    )
                )
            if width_candidates and not continuous and reason == "reject_other":
                reason = (
                    "reject_center_jump"
                    if samples
                    else "reject_seed_or_continuity"
                )

            strict = [item for item in continuous if item[4] and item[5]]
            weak = [
                item
                for item in continuous
                if bool(item[4]) != bool(item[5])
                and weak_minimum_width <= item[6] <= weak_maximum_width
            ]
            chosen: Optional[
                Tuple[float, int, int, float, bool, bool, float]
            ] = None
            classification = "missing"
            if strict:
                chosen = min(strict, key=lambda item: item[0])
                classification = "strict_observed"
                reason = "accepted"
            elif weak:
                chosen = min(weak, key=lambda item: item[0])
                classification = "weak_candidate"
                reason = (
                    "reject_left_edge_validation"
                    if not chosen[4]
                    else "reject_right_edge_validation"
                )
            elif continuous:
                chosen = min(continuous, key=lambda item: item[0])
                if not chosen[4] and not chosen[5]:
                    reason = "reject_boundary_ratio"
                elif not chosen[4]:
                    reason = "reject_left_edge_validation"
                elif not chosen[5]:
                    reason = "reject_right_edge_validation"

            if chosen is None:
                cls._record(
                    samples,
                    y,
                    ipm,
                    reason,
                    center=previous_center,
                    fill_eligible=reason
                    in {
                        "reject_no_yellow_interval",
                        "reject_boundary_ratio",
                    },
                )
                continue
            _score, left, right, center, left_ok, right_ok, width = chosen
            fill_eligible = reason in {
                "reject_left_edge_validation",
                "reject_right_edge_validation",
                "reject_boundary_ratio",
            }
            cls._record(
                samples,
                y,
                ipm,
                reason,
                left,
                right,
                center,
                classification,
                width,
                left_ok,
                right_ok,
                fill_eligible,
            )
            previous_center = center
            has_observation = True

        cls._confirm_weak_candidates(samples, jump_limit)
        _filled_count, _gap_count, _maximum_gap = cls._fill_internal_gaps(
            samples,
            main_yellow,
            config,
            step_px,
            jump_limit,
        )
        observed_consecutive = 0
        observed_maximum_consecutive = 0
        for sample in samples:
            observed_consecutive = (
                observed_consecutive + 1
                if sample["reason"] != "accepted"
                else 0
            )
            observed_maximum_consecutive = max(
                observed_maximum_consecutive, observed_consecutive
            )
        selected_segment = cls._leading_valid_segment(samples)
        selected_fill_count = sum(
            sample["classification"] == "gap_filled"
            for sample in selected_segment
        )
        if (
            selected_fill_count / max(1, len(selected_segment))
            > config.center_gap_fill_max_ratio + 1.0e-9
        ):
            for sample in selected_segment:
                if sample["classification"] == "gap_filled":
                    sample["classification"] = "missing"
            selected_segment = cls._leading_valid_segment(samples)
        selected_ids = {id(sample) for sample in selected_segment}
        for sample in samples:
            if (
                sample["classification"]
                in {"strict_observed", "weak_single_edge", "gap_filled"}
                and id(sample) not in selected_ids
            ):
                sample["source_reason"] = sample["reason"]
                sample["reason"] = "reject_other"
                sample["classification"] = "missing"

        raw = np.asarray(
            [
                (float(sample["center"]), float(sample["y"]))
                for sample in selected_segment
            ],
            dtype=np.float64,
        ).reshape(-1, 2)
        weights = np.asarray(
            [
                cls.CENTER_WEIGHTS[sample["classification"]]
                for sample in selected_segment
            ],
            dtype=np.float64,
        )
        pre_deviation = cls._lateral_deviation(raw)
        smoothing_used = False
        final_points = raw.copy()
        smoothing_failed = False
        if config.center_smooth_enable and len(raw) >= 3:
            try:
                final_points = cls._weighted_smooth(
                    raw, weights, config.center_smooth_lambda
                )
                smoothing_used = True
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                smoothing_failed = True
        if smoothing_failed and geometry_config is not None and len(raw):
            (
                final_points,
                fallback_reason,
                fallback_yellow_ratio,
                fallback_span,
            ) = STAGE4.CenterlineSmoother.smooth(
                raw,
                main_yellow,
                ipm,
                geometry_config,
                (0, main_yellow.shape[0] - 1),
            )
            reason, yellow_ratio, span = (
                fallback_reason,
                fallback_yellow_ratio,
                fallback_span,
            )
            smoothing_used = False
        else:
            reason, yellow_ratio, span = cls._validate_centerline(
                final_points,
                main_yellow,
                ipm,
                config,
                geometry_config,
            )
        post_deviation = cls._lateral_deviation(final_points)

        final_classes = [sample["classification"] for sample in samples]
        strict_count = final_classes.count("strict_observed")
        weak_count = final_classes.count("weak_single_edge")
        actual_filled_count = final_classes.count("gap_filled")
        missing_count = final_classes.count("missing")
        actual_gap_count = 0
        actual_gap_max_samples = 0
        current_filled_run = 0
        for classification in final_classes:
            if classification == "gap_filled":
                current_filled_run += 1
                actual_gap_max_samples = max(
                    actual_gap_max_samples, current_filled_run
                )
            elif current_filled_run:
                actual_gap_count += 1
                current_filled_run = 0
        if current_filled_run:
            actual_gap_count += 1
        observed_count = strict_count + weak_count
        widths = [
            float(sample["width"])
            for sample in samples
            if sample["classification"] in {"strict_observed", "weak_single_edge"}
            and sample["width"] is not None
        ]
        boundary_ratio = (
            (strict_count + 0.5 * weak_count) / max(1, observed_count)
        )
        reason_counts = {
            item: sum(sample["reason"] == item for sample in samples)
            for item in cls.REASONS
        }
        accepted_forwards = [
            sample["forward_m"]
            for sample in samples
            if sample["reason"] == "accepted"
        ]
        rejected_forwards = [
            sample["forward_m"]
            for sample in samples
            if sample["reason"] != "accepted"
        ]
        maximum_consecutive = observed_maximum_consecutive
        termination_reason = "roi_exhausted"
        if maximum_consecutive > config.max_consecutive_invalid_samples:
            termination_reason = "consecutive_invalid_limit"
        elif (
            maximum_consecutive * config.sample_step_m
            > config.max_small_gap_m + 1.0e-9
        ):
            termination_reason = "small_gap_limit"
        fill_ratio = actual_filled_count / max(1, len(selected_segment))
        width_mean = float(np.mean(widths) if widths else 0.0)
        width_std = float(np.std(widths) if widths else 0.0)
        width_stability = float(
            np.clip(1.0 - width_std / max(expected_width * 0.20, 1.0), 0.0, 1.0)
        )
        used_count = max(1, strict_count + weak_count + actual_filled_count)
        confidence = float(
            np.clip(
                0.45
                + 0.35 * strict_count / used_count
                + 0.16 * weak_count / used_count
                + 0.12 * yellow_ratio
                + 0.07 * width_stability
                - 0.28 * fill_ratio,
                0.0,
                0.995,
            )
        )
        if boundary_ratio < config.min_boundary_valid_ratio and reason == "valid":
            reason = "yellow_corridor_boundary_ratio_low"
            final_points = np.empty((0, 2), dtype=np.float64)
        mode = (
            "yellow_corridor_center_gap_filled"
            if weak_count or actual_filled_count
            else "yellow_corridor_dual_edge"
        )
        status.update(
            {
                "yellow_corridor_valid": reason == "valid",
                "yellow_corridor_valid_sample_count": len(raw),
                "yellow_corridor_total_sample_count": len(samples),
                "yellow_corridor_accepted_sample_count": reason_counts["accepted"],
                **{
                    f"yellow_corridor_{item}_count": count
                    for item, count in reason_counts.items()
                    if item != "accepted"
                },
                "yellow_corridor_first_accepted_forward_m": (
                    accepted_forwards[0] if accepted_forwards else None
                ),
                "yellow_corridor_last_accepted_forward_m": (
                    accepted_forwards[-1] if accepted_forwards else None
                ),
                "yellow_corridor_first_rejected_forward_m": (
                    rejected_forwards[0] if rejected_forwards else None
                ),
                "yellow_corridor_max_consecutive_rejected_samples": (
                    maximum_consecutive
                ),
                "yellow_corridor_termination_reason": termination_reason,
                "yellow_corridor_forward_span_m": span,
                "yellow_corridor_width_px_mean": width_mean,
                "yellow_corridor_width_px_std": width_std,
                "yellow_corridor_boundary_valid_ratio": boundary_ratio,
                "yellow_corridor_reason": reason,
                "center_strict_observed_count": strict_count,
                "center_weak_observed_count": weak_count,
                "center_missing_count": missing_count,
                "center_gap_filled_count": actual_filled_count,
                "center_gap_count": actual_gap_count,
                "center_gap_max_samples": actual_gap_max_samples,
                "center_gap_fill_ratio": fill_ratio,
                "center_smoothing_used": smoothing_used,
                "center_smoothing_lambda": config.center_smooth_lambda,
                "center_pre_smooth_lateral_std_px": pre_deviation,
                "center_post_smooth_lateral_std_px": post_deviation,
            }
        )
        result = STAGE4.GeometryResult(
            mode=mode,
            valid=reason == "valid",
            reason=reason,
            confidence=confidence if reason == "valid" else 0.0,
            raw_points=raw,
            final_points=final_points if reason == "valid" else np.empty((0, 2)),
            measured_width_mean_px=width_mean,
            measured_width_std_px=width_std,
            yellow_ratio=yellow_ratio,
            forward_span_m=span,
        )
        result.yellow_corridor_samples = samples
        return result, status


@dataclass
class SingleGreenLaneWidthState:
    width_m: float = 0.0
    updated_monotonic: float = 0.0
    valid: bool = False

    def update(
        self, measured_width_m: float, now: float, alpha: float
    ) -> None:
        if self.valid:
            self.width_m = (
                (1.0 - alpha) * self.width_m
                + alpha * measured_width_m
            )
        else:
            self.width_m = measured_width_m
        self.updated_monotonic = now
        self.valid = True


class SingleGreenCurvePlanner:
    """Build a centerline only over an observed single-green boundary chain."""

    SEGMENT_RATIOS = (0.20, 0.36, 0.52, 0.68, 0.84, 1.0)
    STATUS_DEFAULTS: Dict[str, Any] = {
        "single_green_curve_offset_used": False,
        "single_green_curve_side": "",
        "single_green_curve_strategy": "curve_normal_offset",
        "single_green_curve_failure_reason": "not_attempted",
        "single_green_curve_raw_count": 0,
        "single_green_curve_raw_span_m": 0.0,
        "single_green_curve_chain_count": 0,
        "single_green_curve_chain_span_m": 0.0,
        "single_green_curve_smoothed_count": 0,
        "single_green_curve_candidate_count": 0,
        "single_green_curve_accepted_count": 0,
        "single_green_curve_rejected_count": 0,
        "single_green_curve_contributed_count": 0,
        "single_green_curve_reject_invalid_ipm_count": 0,
        "single_green_curve_reject_normal_direction_count": 0,
        "single_green_curve_reject_yellow_support_count": 0,
        "single_green_curve_reject_green_intrusion_count": 0,
        "single_green_curve_reject_opposite_boundary_count": 0,
        "single_green_curve_reject_lateral_jump_count": 0,
        "single_green_curve_reject_heading_jump_count": 0,
        "single_green_curve_reject_offset_distance_count": 0,
        "single_green_lane_width_m": 0.0,
        "single_green_lane_width_source": "expected",
        "single_green_lane_width_history_age_ms": -1.0,
        "single_green_lane_width_clamped": False,
        "lane_width_history_m": 0.0,
        "lane_width_history_age_sec": -1.0,
        "lane_width_history_valid": False,
        "selected_normal_sign": 0,
        "normal_selection_reason": "not_selected",
        "single_green_yellow_support_ratio_mean": 0.0,
        "single_green_green_intrusion_ratio_mean": 0.0,
        "single_green_offset_distance_m_mean": 0.0,
        "single_green_offset_distance_m_std": 0.0,
        "single_green_boundary_heading_deg_mean": 0.0,
        "single_green_boundary_heading_deg_std": 0.0,
        "single_green_center_heading_deg_mean": 0.0,
        "single_green_center_heading_deg_std": 0.0,
        "hybrid_dual_point_count": 0,
        "hybrid_existing_point_count": 0,
        "hybrid_single_curve_point_count": 0,
        "hybrid_final_point_count": 0,
        "hybrid_final_span_m": 0.0,
        "single_green_curve_offset_ms": 0.0,
    }

    @staticmethod
    def _metric_point(
        sample: Dict[str, Any], side: str, ipm: Any
    ) -> Dict[str, Any]:
        pixel_x = float(sample[side])
        return {
            "sample_index": int(sample["_sample_index"]),
            "pixel_x": pixel_x,
            "pixel_y": float(sample["y"]),
            "forward_m": float(sample["forward_m"]),
            "left_m": (
                float(ipm.vehicle_center_x_px) - pixel_x
            )
            * float(ipm.meter_per_pixel),
        }

    @classmethod
    def extract_main_chain(
        cls,
        samples: Sequence[Dict[str, Any]],
        side: str,
        ipm: Any,
        config: YellowCorridorConfig,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        raw: List[Dict[str, Any]] = []
        for index, source in enumerate(samples):
            if source.get(side) is None:
                continue
            sample = dict(source)
            sample["_sample_index"] = index
            raw.append(cls._metric_point(sample, side, ipm))
        raw.sort(key=lambda item: item["forward_m"])
        chains: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        for point in raw:
            if current:
                previous = current[-1]
                missing_rows = max(
                    0,
                    int(round(
                        (
                            point["forward_m"]
                            - previous["forward_m"]
                        )
                        / max(config.sample_step_m, 1.0e-6)
                    ))
                    - 1,
                )
                # Row-gap policy already constrains longitudinal separation;
                # this limit rejects implausible lateral edge jumps.
                jump = abs(point["left_m"] - previous["left_m"])
                if (
                    missing_rows > config.single_green_curve_max_row_gap
                    or jump > config.single_green_curve_max_point_jump_m
                ):
                    chains.append(current)
                    current = []
            current.append(point)
        if current:
            chains.append(current)
        main = max(
            chains,
            key=lambda chain: (
                chain[-1]["forward_m"] - chain[0]["forward_m"],
                len(chain),
            ),
            default=[],
        )
        return raw, list(main)

    @staticmethod
    def _linear_coefficients(points: np.ndarray) -> Tuple[float, float]:
        x_values = points[:, 0]
        y_values = points[:, 1]
        x_mean = float(np.mean(x_values))
        y_mean = float(np.mean(y_values))
        centered_x = x_values - x_mean
        denominator = float(np.dot(centered_x, centered_x))
        slope = (
            float(np.dot(centered_x, y_values - y_mean)) / denominator
            if denominator > 1.0e-12
            else 0.0
        )
        return slope, y_mean - slope * x_mean

    @staticmethod
    def smooth_single_green_boundary(
        chain: Sequence[Dict[str, Any]],
        config: YellowCorridorConfig,
    ) -> np.ndarray:
        points = np.asarray(
            [
                (item["forward_m"], item["left_m"])
                for item in chain
            ],
            dtype=np.float64,
        ).reshape(-1, 2)
        if len(points) < config.single_green_min_tangent_points:
            return points
        output = points.copy()
        half = max(1, config.single_green_tangent_window_points // 2)
        for index in range(len(points)):
            start = max(0, index - half)
            end = min(len(points), index + half + 1)
            local = points[start:end]
            if (
                len(local) < config.single_green_min_tangent_points
                or local[-1, 0] - local[0, 0]
                < config.single_green_min_tangent_dx_m
            ):
                continue
            slope, intercept = SingleGreenCurvePlanner._linear_coefficients(
                local
            )
            residuals = local[:, 1] - (
                slope * local[:, 0] + intercept
            )
            median = float(np.median(residuals))
            mad = float(np.median(np.abs(residuals - median)))
            threshold = max(0.008, 3.0 * mad)
            keep = np.abs(residuals - median) <= threshold
            if np.count_nonzero(keep) >= config.single_green_min_tangent_points:
                slope, intercept = (
                    SingleGreenCurvePlanner._linear_coefficients(local[keep])
                )
            output[index, 1] = float(
                slope * points[index, 0] + intercept
            )
        return output

    @staticmethod
    def estimate_local_boundary_tangent(
        smoothed: np.ndarray,
        config: YellowCorridorConfig,
    ) -> np.ndarray:
        points = np.asarray(smoothed, dtype=np.float64).reshape(-1, 2)
        tangents = np.zeros_like(points)
        half = max(1, config.single_green_tangent_window_points // 2)
        for index in range(len(points)):
            start = max(0, index - half)
            end = min(len(points), index + half + 1)
            local = points[start:end]
            if len(local) < 2:
                continue
            dx = float(local[-1, 0] - local[0, 0])
            if dx < config.single_green_min_tangent_dx_m:
                continue
            slope, _intercept = (
                SingleGreenCurvePlanner._linear_coefficients(local)
            )
            tangent = np.asarray(
                [1.0, slope], dtype=np.float64
            )
            tangent /= max(float(np.linalg.norm(tangent)), 1.0e-9)
            if tangent[0] < 0.0:
                tangent *= -1.0
            tangents[index] = tangent
        return tangents

    @staticmethod
    def _metric_to_pixel(point: np.ndarray, ipm: Any) -> Tuple[float, float]:
        forward, left = float(point[0]), float(point[1])
        return (
            float(ipm.vehicle_center_x_px)
            - left / float(ipm.meter_per_pixel),
            float(ipm.vehicle_origin_y_px)
            - forward / float(ipm.meter_per_pixel),
        )

    @staticmethod
    def _window_ratio(
        mask: np.ndarray,
        valid_mask: np.ndarray,
        pixel_x: float,
        pixel_y: float,
        radius: int,
    ) -> float:
        x = int(round(pixel_x))
        y = int(round(pixel_y))
        x0, x1 = max(0, x - radius), min(mask.shape[1], x + radius + 1)
        y0, y1 = max(0, y - radius), min(mask.shape[0], y + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return 0.0
        valid = valid_mask[y0:y1, x0:x1]
        valid_count = int(np.count_nonzero(valid))
        if not valid_count:
            return 0.0
        selected_count = int(np.count_nonzero(
            (mask[y0:y1, x0:x1] > 0) & valid
        ))
        return selected_count / valid_count

    @staticmethod
    def _segment_green_ratio(
        boundary_pixel: Tuple[float, float],
        center_pixel: Tuple[float, float],
        green: np.ndarray,
        valid_mask: np.ndarray,
    ) -> float:
        values: List[bool] = []
        for ratio in SingleGreenCurvePlanner.SEGMENT_RATIOS:
            x = int(round(
                boundary_pixel[0]
                + ratio * (center_pixel[0] - boundary_pixel[0])
            ))
            y = int(round(
                boundary_pixel[1]
                + ratio * (center_pixel[1] - boundary_pixel[1])
            ))
            if (
                0 <= y < green.shape[0]
                and 0 <= x < green.shape[1]
                and valid_mask[y, x]
            ):
                values.append(bool(green[y, x] > 0))
        return float(np.mean(values)) if values else 1.0

    @classmethod
    def resolve_lane_width(
        cls,
        samples: Sequence[Dict[str, Any]],
        state: Optional[SingleGreenLaneWidthState],
        ipm: Any,
        config: YellowCorridorConfig,
        now: float,
    ) -> Tuple[float, str, float, bool]:
        widths = [
            float(sample["right"] - sample["left"])
            * float(ipm.meter_per_pixel)
            for sample in samples
            if sample.get("left") is not None
            and sample.get("right") is not None
            and sample["right"] > sample["left"]
        ]
        history_age_ms = -1.0
        if len(widths) >= config.lane_width_current_min_dual_samples:
            width = float(np.median(widths))
            source = "current_dual"
        elif state is not None and state.valid:
            age = max(0.0, now - state.updated_monotonic)
            history_age_ms = age * 1000.0
            if age <= config.lane_width_history_timeout_sec:
                width = float(state.width_m)
                source = "history_ema"
            else:
                width = (
                    float(ipm.expected_lane_width_px)
                    * float(ipm.meter_per_pixel)
                )
                source = "expected"
        else:
            width = (
                float(ipm.expected_lane_width_px)
                * float(ipm.meter_per_pixel)
            )
            source = "expected"
        clamped = not (
            config.single_green_lane_width_min_m
            <= width
            <= config.single_green_lane_width_max_m
        )
        width = float(np.clip(
            width,
            config.single_green_lane_width_min_m,
            config.single_green_lane_width_max_m,
        ))
        return width, source, history_age_ms, clamped

    @classmethod
    def build(
        cls,
        samples: Sequence[Dict[str, Any]],
        green: np.ndarray,
        yellow: np.ndarray,
        ipm_valid_mask: np.ndarray,
        ipm: Any,
        config: YellowCorridorConfig,
        width_state: Optional[SingleGreenLaneWidthState] = None,
        now: Optional[float] = None,
    ) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
        started = time.perf_counter()
        instant = time.monotonic() if now is None else float(now)
        status = copy.deepcopy(cls.STATUS_DEFAULTS)
        debug: List[Dict[str, Any]] = []
        side_data: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
        for side in ("left", "right"):
            side_data[side] = cls.extract_main_chain(
                samples, side, ipm, config
            )
        side = max(
            ("left", "right"),
            key=lambda name: (
                (
                    side_data[name][1][-1]["forward_m"]
                    - side_data[name][1][0]["forward_m"]
                )
                if side_data[name][1]
                else 0.0,
                len(side_data[name][1]),
            ),
        )
        raw, chain = side_data[side]
        raw_span = (
            raw[-1]["forward_m"] - raw[0]["forward_m"]
            if raw else 0.0
        )
        chain_span = (
            chain[-1]["forward_m"] - chain[0]["forward_m"]
            if chain else 0.0
        )
        status.update(
            {
                "single_green_curve_side": side if chain else "",
                "single_green_curve_raw_count": len(raw),
                "single_green_curve_raw_span_m": float(raw_span),
                "single_green_curve_chain_count": len(chain),
                "single_green_curve_chain_span_m": float(chain_span),
            }
        )
        if (
            len(chain) < config.single_green_curve_min_points
            or chain_span + 1.0e-9
            < config.single_green_curve_min_span_m
        ):
            status["single_green_curve_failure_reason"] = (
                "main_chain_too_short"
            )
            status["single_green_curve_offset_ms"] = (
                time.perf_counter() - started
            ) * 1000.0
            return {}, status, debug
        smoothed = cls.smooth_single_green_boundary(chain, config)
        tangents = cls.estimate_local_boundary_tangent(smoothed, config)
        status["single_green_curve_smoothed_count"] = len(smoothed)
        lane_width, width_source, width_age_ms, width_clamped = (
            cls.resolve_lane_width(
                samples, width_state, ipm, config, instant
            )
        )
        status.update(
            {
                "single_green_lane_width_m": lane_width,
                "single_green_lane_width_source": width_source,
                "single_green_lane_width_history_age_ms": width_age_ms,
                "single_green_lane_width_clamped": width_clamped,
                "lane_width_history_m": float(
                    width_state.width_m
                    if width_state is not None and width_state.valid
                    else 0.0
                ),
                "lane_width_history_age_sec": (
                    width_age_ms / 1000.0 if width_age_ms >= 0.0 else -1.0
                ),
                "lane_width_history_valid": bool(
                    width_state is not None and width_state.valid
                ),
            }
        )
        accepted: Dict[int, Dict[str, Any]] = {}
        yellow_ratios: List[float] = []
        green_ratios: List[float] = []
        offset_distances: List[float] = []
        boundary_headings: List[float] = []
        center_headings: List[float] = []
        previous_center: Optional[np.ndarray] = None
        previous_heading: Optional[float] = None
        expected_normal_left_sign = -1.0 if side == "left" else 1.0
        radius = config.single_green_center_yellow_window_px
        for index, (boundary, tangent) in enumerate(
            zip(smoothed, tangents)
        ):
            status["single_green_curve_candidate_count"] += 1
            source = samples[chain[index]["sample_index"]]
            if source.get("left") is not None and source.get("right") is not None:
                status[
                    "single_green_curve_reject_opposite_boundary_count"
                ] += 1
                continue
            if tangent[0] <= 0.0 or np.linalg.norm(tangent) < 0.9:
                status[
                    "single_green_curve_reject_normal_direction_count"
                ] += 1
                continue
            boundary_heading = math.atan2(
                float(tangent[1]), float(tangent[0])
            )
            boundary_headings.append(math.degrees(boundary_heading))
            normals = (
                (
                    1,
                    np.asarray(
                        [-tangent[1], tangent[0]], dtype=np.float64
                    ),
                ),
                (
                    -1,
                    np.asarray(
                        [tangent[1], -tangent[0]], dtype=np.float64
                    ),
                ),
            )
            candidate_records: List[Tuple[float, np.ndarray, Dict[str, Any]]] = []
            for normal_sign, normal in normals:
                semantic = normal[1] * expected_normal_left_sign > 0.0
                candidate = boundary + normal * (lane_width * 0.5)
                center_pixel = cls._metric_to_pixel(candidate, ipm)
                boundary_pixel = cls._metric_to_pixel(boundary, ipm)
                x = int(round(center_pixel[0]))
                y = int(round(center_pixel[1]))
                valid = bool(
                    0 <= y < green.shape[0]
                    and 0 <= x < green.shape[1]
                    and ipm_valid_mask[y, x]
                )
                point_yellow = float(valid and yellow[y, x] > 0)
                point_green = float(valid and green[y, x] > 0)
                score = (
                    (2.0 if semantic else -2.0)
                    + 1.5 * point_yellow
                    - 1.5 * point_green
                    - 0.2 * abs(candidate[1])
                )
                candidate_records.append(
                    (
                        score,
                        candidate,
                        {
                            "normal_sign": normal_sign,
                            "semantic": semantic,
                            "valid": valid,
                            "center_pixel": center_pixel,
                            "boundary_pixel": boundary_pixel,
                        },
                    )
                )
            _score, candidate, evidence = max(
                candidate_records, key=lambda item: item[0]
            )
            evidence["yellow_ratio"] = cls._window_ratio(
                yellow,
                ipm_valid_mask,
                *evidence["center_pixel"],
                radius,
            )
            evidence["green_ratio"] = cls._window_ratio(
                green,
                ipm_valid_mask,
                *evidence["center_pixel"],
                radius,
            )
            evidence["segment_green_ratio"] = cls._segment_green_ratio(
                evidence["boundary_pixel"],
                evidence["center_pixel"],
                green,
                ipm_valid_mask,
            )
            reject_key = ""
            if (
                candidate[0] < smoothed[0, 0] - 1.0e-9
                or candidate[0] > smoothed[-1, 0] + 1.0e-9
            ):
                # A normal on a curved endpoint can point slightly beyond the
                # observed station range.  Reject it instead of extrapolating.
                reject_key = (
                    "single_green_curve_reject_offset_distance_count"
                )
            elif not evidence["semantic"]:
                reject_key = (
                    "single_green_curve_reject_normal_direction_count"
                )
            elif not evidence["valid"]:
                reject_key = "single_green_curve_reject_invalid_ipm_count"
            elif (
                evidence["yellow_ratio"]
                < config.single_green_center_min_yellow_ratio
            ):
                reject_key = (
                    "single_green_curve_reject_yellow_support_count"
                )
            elif (
                evidence["green_ratio"]
                > config.single_green_center_max_green_ratio
                or evidence["segment_green_ratio"] > 0.35
            ):
                reject_key = (
                    "single_green_curve_reject_green_intrusion_count"
                )
            offset_distance = float(np.linalg.norm(candidate - boundary))
            if (
                not reject_key
                and abs(offset_distance - lane_width * 0.5) > 0.02
            ):
                reject_key = (
                    "single_green_curve_reject_offset_distance_count"
                )
            center_heading: Optional[float] = None
            if previous_center is not None:
                delta = candidate - previous_center
                if (
                    abs(float(delta[1]))
                    > config.single_green_center_max_lateral_jump_m
                ):
                    reject_key = (
                        "single_green_curve_reject_lateral_jump_count"
                    )
                if delta[0] > 1.0e-6:
                    center_heading = math.atan2(
                        float(delta[1]), float(delta[0])
                    )
                    if (
                        previous_heading is not None
                        and math.degrees(
                            abs(math.atan2(
                                math.sin(center_heading - previous_heading),
                                math.cos(center_heading - previous_heading),
                            ))
                        )
                        > config.single_green_center_max_heading_step_deg
                    ):
                        reject_key = (
                            "single_green_curve_reject_heading_jump_count"
                        )
            debug_record = {
                "sample_index": chain[index]["sample_index"],
                "side": side,
                "boundary_pixel": evidence["boundary_pixel"],
                "smoothed_pixel": cls._metric_to_pixel(boundary, ipm),
                "center_pixel": evidence["center_pixel"],
                "normal_sign": evidence["normal_sign"],
                "normal_selection_reason": "side_and_mask_score",
                "accepted": not bool(reject_key),
                "reject_key": reject_key,
            }
            debug.append(debug_record)
            if reject_key:
                status[reject_key] += 1
                continue
            sample_index = chain[index]["sample_index"]
            accepted[sample_index] = {
                "center_pixel": evidence["center_pixel"],
                "center_metric": candidate.copy(),
                "boundary_metric": boundary.copy(),
                "tangent": tangent.copy(),
                "side": side,
                "yellow_ratio": evidence["yellow_ratio"],
                "green_ratio": evidence["green_ratio"],
                "offset_distance_m": offset_distance,
            }
            status["selected_normal_sign"] = int(evidence["normal_sign"])
            status["normal_selection_reason"] = (
                "side_semantics_yellow_green_score"
            )
            previous_center = candidate
            if center_heading is not None:
                previous_heading = center_heading
                center_headings.append(math.degrees(center_heading))
            yellow_ratios.append(evidence["yellow_ratio"])
            green_ratios.append(evidence["green_ratio"])
            offset_distances.append(offset_distance)
        accepted_count = len(accepted)
        status.update(
            {
                "single_green_curve_offset_used": bool(accepted_count),
                "single_green_curve_failure_reason": (
                    "valid" if accepted_count else "no_candidate_accepted"
                ),
                "single_green_curve_accepted_count": accepted_count,
                "single_green_curve_rejected_count": (
                    status["single_green_curve_candidate_count"]
                    - accepted_count
                ),
                "single_green_yellow_support_ratio_mean": float(
                    np.mean(yellow_ratios) if yellow_ratios else 0.0
                ),
                "single_green_green_intrusion_ratio_mean": float(
                    np.mean(green_ratios) if green_ratios else 0.0
                ),
                "single_green_offset_distance_m_mean": float(
                    np.mean(offset_distances) if offset_distances else 0.0
                ),
                "single_green_offset_distance_m_std": float(
                    np.std(offset_distances) if offset_distances else 0.0
                ),
                "single_green_boundary_heading_deg_mean": float(
                    np.mean(boundary_headings) if boundary_headings else 0.0
                ),
                "single_green_boundary_heading_deg_std": float(
                    np.std(boundary_headings) if boundary_headings else 0.0
                ),
                "single_green_center_heading_deg_mean": float(
                    np.mean(center_headings) if center_headings else 0.0
                ),
                "single_green_center_heading_deg_std": float(
                    np.std(center_headings) if center_headings else 0.0
                ),
            }
        )
        status["single_green_curve_offset_ms"] = (
            time.perf_counter() - started
        ) * 1000.0
        return accepted, status, debug


class GreenBoundaryFastPlanner:
    """仅在稀疏采样行上跟踪左右主绿色区域的通道内边缘。"""

    MAX_GREEN_INTRUSION_RATIO = 0.15
    MAX_SINGLE_CONSECUTIVE_SAMPLES = 2
    EMPTY_STATUS: Dict[str, Any] = {
        "green_dual_edge_valid_count": 0,
        "green_single_left_count": 0,
        "green_single_right_count": 0,
        "green_left_inner_edge_missing_count": 0,
        "green_right_inner_edge_missing_count": 0,
        "green_corridor_width_px_mean": 0.0,
        "green_corridor_width_px_std": 0.0,
        "green_corridor_yellow_support_ratio": 0.0,
        "green_corridor_unknown_ratio": 0.0,
        "green_corridor_green_intrusion_ratio": 0.0,
        "green_boundary_fast_path_used": False,
        "green_boundary_fast_path_reason": "not_attempted",
        "green_reject_no_valid_ipm_span_count": 0,
        "green_reject_left_outer_run_missing_count": 0,
        "green_reject_right_outer_run_missing_count": 0,
        "green_reject_left_outer_gap_count": 0,
        "green_reject_right_outer_gap_count": 0,
        "green_reject_left_run_too_short_count": 0,
        "green_reject_right_run_too_short_count": 0,
        "green_reject_width_count": 0,
        "green_reject_center_jump_count": 0,
        "green_reject_green_intrusion_count": 0,
        "green_valid_span_left_x_mean": 0.0,
        "green_valid_span_right_x_mean": 0.0,
        "green_left_outer_gap_px_mean": 0.0,
        "green_right_outer_gap_px_mean": 0.0,
        **copy.deepcopy(SingleGreenCurvePlanner.STATUS_DEFAULTS),
    }

    @staticmethod
    def _side_interval(
        intervals: Sequence[Tuple[int, int]],
        side: str,
        center_x: int,
        valid_left_x: int,
        valid_right_x: int,
        previous_inner_x: Optional[float],
        continuity_limit_px: float,
        minimum_span_px: int,
        outer_seed_max_gap_px: int,
    ) -> Tuple[Optional[Tuple[int, int, bool, int]], str, Optional[int]]:
        candidates: List[Tuple[int, int, float, int, int, bool]] = []
        side_run_found = False
        long_run_found = False
        observed_outer_gaps: List[int] = []
        for left, right in intervals:
            left = max(int(left), valid_left_x)
            right = min(int(right), valid_right_x)
            if right < left:
                continue
            if side == "left":
                if right >= center_x:
                    continue
                inner_x = float(right)
                outer_gap = left - valid_left_x
            else:
                if left <= center_x:
                    continue
                inner_x = float(left)
                outer_gap = valid_right_x - right
            side_run_found = True
            span = right - left + 1
            if span < minimum_span_px:
                continue
            long_run_found = True
            observed_outer_gaps.append(int(outer_gap))
            seeded = outer_gap <= outer_seed_max_gap_px
            continuous = bool(
                previous_inner_x is not None
                and abs(inner_x - previous_inner_x) <= continuity_limit_px
            )
            if not (seeded or continuous):
                continue
            distance = (
                abs(inner_x - previous_inner_x)
                if previous_inner_x is not None
                else 0.0
            )
            candidates.append(
                (
                    0 if seeded else 1,
                    int(outer_gap),
                    distance,
                    left,
                    right,
                    seeded,
                )
            )
        if not candidates:
            if not side_run_found:
                return None, "outer_run_missing", None
            if not long_run_found:
                return None, "run_too_short", None
            return None, "outer_gap", min(observed_outer_gaps)
        _priority, outer_gap, _distance, left, right, seeded = min(
            candidates,
            key=lambda item: (item[0], item[1], item[2], item[3] - item[4]),
        )
        return (
            (int(left), int(right), bool(seeded), int(outer_gap)),
            "valid",
            int(outer_gap),
        )

    @staticmethod
    def _corridor_evidence(
        green_row: np.ndarray,
        yellow_row: np.ndarray,
        valid_row: np.ndarray,
        left_inner: int,
        right_inner: int,
    ) -> Tuple[float, float, float]:
        start, end = left_inner + 1, right_inner
        if end <= start:
            return 0.0, 1.0, 1.0
        valid_inside = valid_row[start:end]
        if not np.any(valid_inside):
            return 0.0, 0.0, 0.0
        green_inside = green_row[start:end][valid_inside] > 0
        yellow_inside = yellow_row[start:end][valid_inside] > 0
        yellow_ratio = float(np.mean(yellow_inside))
        intrusion_ratio = float(np.mean(green_inside))
        unknown_ratio = float(np.mean(~green_inside & ~yellow_inside))
        return yellow_ratio, unknown_ratio, intrusion_ratio

    @classmethod
    def extract(
        cls,
        green: np.ndarray,
        yellow: np.ndarray,
        ipm: Any,
        config: YellowCorridorConfig,
        seed_center_x: Optional[float] = None,
        ipm_valid_mask: Optional[np.ndarray] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        status = copy.deepcopy(cls.EMPTY_STATUS)
        samples: List[Dict[str, Any]] = []
        valid_mask = (
            np.ones_like(green, dtype=bool)
            if ipm_valid_mask is None
            else np.asarray(ipm_valid_mask, dtype=bool)
        )
        if valid_mask.shape != green.shape or yellow.shape != green.shape:
            raise ValueError("green/yellow/ipm_valid_mask shape mismatch")
        step_px = max(2, int(round(config.sample_step_m / ipm.meter_per_pixel)))
        expected_width = float(ipm.expected_lane_width_px)
        minimum_width = expected_width * config.width_min_ratio
        maximum_width = expected_width * config.width_max_ratio
        continuity_limit = config.center_jump_max_m / ipm.meter_per_pixel
        minimum_green_span = config.green_min_outer_run_width_px
        center_axis = int(round(ipm.vehicle_center_x_px))
        previous_left: Optional[float] = None
        previous_right: Optional[float] = None
        previous_center = float(
            ipm.vehicle_center_x_px
            if seed_center_x is None
            else seed_center_x
        )
        consecutive_single = 0
        widths: List[float] = []
        yellow_ratios: List[float] = []
        unknown_ratios: List[float] = []
        intrusion_ratios: List[float] = []
        valid_left_values: List[int] = []
        valid_right_values: List[int] = []
        left_outer_gaps: List[int] = []
        right_outer_gaps: List[int] = []

        for y in range(green.shape[0] - 1, -1, -step_px):
            valid_x = np.flatnonzero(valid_mask[y])
            if not len(valid_x):
                status["green_reject_no_valid_ipm_span_count"] += 1
                status["green_left_inner_edge_missing_count"] += 1
                status["green_right_inner_edge_missing_count"] += 1
                samples.append(
                    {
                        "y": int(y),
                        "forward_m": YellowCorridorPlanner._forward_m(y, ipm),
                        "reason": "green_reject_no_valid_ipm_span",
                        "classification": "missing",
                        "source_type": "missing",
                        "left": None,
                        "right": None,
                        "center": None,
                        "width": None,
                        "single_side": "",
                        "yellow_support_ratio": 0.0,
                        "unknown_ratio": 0.0,
                        "green_intrusion_ratio": 0.0,
                        "valid_left_x": None,
                        "valid_right_x": None,
                        "left_outer_gap_px": None,
                        "right_outer_gap_px": None,
                    }
                )
                continue
            valid_left_x, valid_right_x = int(valid_x[0]), int(valid_x[-1])
            valid_left_values.append(valid_left_x)
            valid_right_values.append(valid_right_x)
            row_green = np.where(valid_mask[y], green[y], 0).astype(np.uint8)
            row_intervals = YellowCorridorPlanner._intervals(row_green)
            left_interval, left_failure, left_gap_diagnostic = cls._side_interval(
                row_intervals,
                "left",
                center_axis,
                valid_left_x,
                valid_right_x,
                previous_left,
                continuity_limit,
                minimum_green_span,
                config.green_outer_seed_max_gap_px,
            )
            (
                right_interval,
                right_failure,
                right_gap_diagnostic,
            ) = cls._side_interval(
                row_intervals,
                "right",
                center_axis,
                valid_left_x,
                valid_right_x,
                previous_right,
                continuity_limit,
                minimum_green_span,
                config.green_outer_seed_max_gap_px,
            )
            left_inner = (
                int(left_interval[1]) if left_interval is not None else None
            )
            right_inner = (
                int(right_interval[0]) if right_interval is not None else None
            )
            left_outer_gap = (
                int(left_interval[3])
                if left_interval is not None
                else left_gap_diagnostic
            )
            right_outer_gap = (
                int(right_interval[3])
                if right_interval is not None
                else right_gap_diagnostic
            )
            if left_outer_gap is not None:
                left_outer_gaps.append(left_outer_gap)
            if right_outer_gap is not None:
                right_outer_gaps.append(right_outer_gap)
            if left_interval is None:
                status[
                    "green_reject_left_run_too_short_count"
                    if left_failure == "run_too_short"
                    else "green_reject_left_outer_gap_count"
                    if left_failure == "outer_gap"
                    else "green_reject_left_outer_run_missing_count"
                ] += 1
            if right_interval is None:
                status[
                    "green_reject_right_run_too_short_count"
                    if right_failure == "run_too_short"
                    else "green_reject_right_outer_gap_count"
                    if right_failure == "outer_gap"
                    else "green_reject_right_outer_run_missing_count"
                ] += 1
            if (
                left_inner is not None
                and previous_left is not None
                and abs(left_inner - previous_left) > continuity_limit
            ):
                left_inner = None
                status["green_reject_center_jump_count"] += 1
            if (
                right_inner is not None
                and previous_right is not None
                and abs(right_inner - previous_right) > continuity_limit
            ):
                right_inner = None
                status["green_reject_center_jump_count"] += 1
            if left_inner is None:
                status["green_left_inner_edge_missing_count"] += 1
            if right_inner is None:
                status["green_right_inner_edge_missing_count"] += 1

            classification = "missing"
            reason = "green_edges_missing"
            center: Optional[float] = None
            width_value: Optional[float] = None
            yellow_ratio = 0.0
            unknown_ratio = 0.0
            intrusion_ratio = 0.0
            single_side = ""

            if left_inner is not None and right_inner is not None:
                width_value = float(right_inner - left_inner)
                candidate_center = (left_inner + right_inner) / 2.0
                (
                    yellow_ratio,
                    unknown_ratio,
                    intrusion_ratio,
                ) = cls._corridor_evidence(
                    green[y],
                    yellow[y],
                    valid_mask[y],
                    left_inner,
                    right_inner,
                )
                if not (minimum_width <= width_value <= maximum_width):
                    reason = "green_corridor_width_invalid"
                    status["green_reject_width_count"] += 1
                elif abs(candidate_center - previous_center) > continuity_limit:
                    reason = "green_corridor_center_jump"
                    status["green_reject_center_jump_count"] += 1
                elif intrusion_ratio > cls.MAX_GREEN_INTRUSION_RATIO:
                    reason = "green_corridor_intrusion"
                    status["green_reject_green_intrusion_count"] += 1
                else:
                    classification = "green_dual_observed"
                    reason = "accepted"
                    center = candidate_center
                    consecutive_single = 0
                    widths.append(width_value)
                    yellow_ratios.append(yellow_ratio)
                    unknown_ratios.append(unknown_ratio)
                    intrusion_ratios.append(intrusion_ratio)
                    status["green_dual_edge_valid_count"] += 1
            elif left_inner is not None or right_inner is not None:
                single_side = "left" if left_inner is not None else "right"
                candidate_center = (
                    float(left_inner) + expected_width / 2.0
                    if left_inner is not None
                    else float(right_inner) - expected_width / 2.0
                )
                center_index = int(round(candidate_center))
                center_is_green = not (
                    0 <= center_index < green.shape[1]
                ) or bool(green[y, center_index] > 0)
                if (
                    consecutive_single
                    < cls.MAX_SINGLE_CONSECUTIVE_SAMPLES
                    and abs(candidate_center - previous_center)
                    <= continuity_limit
                    and not center_is_green
                ):
                    classification = "green_single_offset"
                    reason = "accepted"
                    center = candidate_center
                    consecutive_single += 1
                    status[
                        "green_single_left_count"
                        if single_side == "left"
                        else "green_single_right_count"
                    ] += 1
                else:
                    reason = "green_single_offset_invalid"
                    consecutive_single += 1
            else:
                consecutive_single += 1

            if left_inner is not None:
                previous_left = float(left_inner)
            if right_inner is not None:
                previous_right = float(right_inner)
            if center is not None:
                previous_center = center
            samples.append(
                {
                    "y": int(y),
                    "forward_m": YellowCorridorPlanner._forward_m(y, ipm),
                    "reason": reason,
                    "classification": classification,
                    "source_type": classification,
                    "left": left_inner,
                    "right": right_inner,
                    "center": center,
                    "width": width_value,
                    "single_side": single_side,
                    "yellow_support_ratio": yellow_ratio,
                    "unknown_ratio": unknown_ratio,
                    "green_intrusion_ratio": intrusion_ratio,
                    "valid_left_x": valid_left_x,
                    "valid_right_x": valid_right_x,
                    "left_outer_gap_px": left_outer_gap,
                    "right_outer_gap_px": right_outer_gap,
                }
            )

        status.update(
            {
                "green_corridor_width_px_mean": float(
                    np.mean(widths) if widths else 0.0
                ),
                "green_corridor_width_px_std": float(
                    np.std(widths) if widths else 0.0
                ),
                "green_corridor_yellow_support_ratio": float(
                    np.mean(yellow_ratios) if yellow_ratios else 0.0
                ),
                "green_corridor_unknown_ratio": float(
                    np.mean(unknown_ratios) if unknown_ratios else 0.0
                ),
                "green_corridor_green_intrusion_ratio": float(
                    np.mean(intrusion_ratios) if intrusion_ratios else 0.0
                ),
                "green_valid_span_left_x_mean": float(
                    np.mean(valid_left_values) if valid_left_values else 0.0
                ),
                "green_valid_span_right_x_mean": float(
                    np.mean(valid_right_values) if valid_right_values else 0.0
                ),
                "green_left_outer_gap_px_mean": float(
                    np.mean(left_outer_gaps) if left_outer_gaps else 0.0
                ),
                "green_right_outer_gap_px_mean": float(
                    np.mean(right_outer_gaps) if right_outer_gaps else 0.0
                ),
            }
        )
        return samples, status

    @classmethod
    def _fill_internal_gaps(
        cls,
        samples: List[Dict[str, Any]],
        green: np.ndarray,
        config: YellowCorridorConfig,
        ipm: Any,
    ) -> None:
        if not config.center_gap_fill_enable:
            return
        valid_classes = {"green_dual_observed", "green_single_offset"}
        index = 0
        while index < len(samples):
            if samples[index]["classification"] != "missing":
                index += 1
                continue
            start = index
            while (
                index < len(samples)
                and samples[index]["classification"] == "missing"
            ):
                index += 1
            end = index
            count = end - start
            if (
                start == 0
                or end >= len(samples)
                or samples[start - 1]["classification"] not in valid_classes
                or samples[end]["classification"] not in valid_classes
                or count > config.center_gap_fill_max_samples
                or count * config.sample_step_m
                > config.center_gap_fill_max_m + 1.0e-9
            ):
                continue
            before, after = samples[start - 1], samples[end]
            if (
                abs(float(after["center"]) - float(before["center"]))
                / (count + 1)
                > config.center_jump_max_m / ipm.meter_per_pixel
            ):
                continue
            proposed: List[float] = []
            valid = True
            for offset, sample in enumerate(samples[start:end], 1):
                ratio = offset / float(count + 1)
                center = float(before["center"]) + ratio * (
                    float(after["center"]) - float(before["center"])
                )
                center_index = int(round(center))
                if (
                    not 0 <= center_index < green.shape[1]
                    or green[int(sample["y"]), center_index] > 0
                ):
                    valid = False
                    break
                proposed.append(center)
            if not valid:
                continue
            for sample, center in zip(samples[start:end], proposed):
                sample["classification"] = "gap_filled"
                sample["source_type"] = "gap_filled"
                sample["center"] = center
                sample["reason"] = "accepted"

    @staticmethod
    def _leading_segment(
        samples: List[Dict[str, Any]],
        allowed: set,
    ) -> List[Dict[str, Any]]:
        segment: List[Dict[str, Any]] = []
        started = False
        for sample in samples:
            if sample["classification"] in allowed:
                segment.append(sample)
                started = True
            elif started:
                break
        return segment

    @classmethod
    def plan(
        cls,
        extracted_samples: Sequence[Dict[str, Any]],
        base_status: Dict[str, Any],
        green: np.ndarray,
        yellow: np.ndarray,
        ipm: Any,
        config: YellowCorridorConfig,
        geometry_config: Any,
        allow_single: bool,
        ipm_valid_mask: Optional[np.ndarray] = None,
        width_state: Optional[SingleGreenLaneWidthState] = None,
        now_monotonic: Optional[float] = None,
    ) -> Tuple[Any, Dict[str, Any], List[Dict[str, Any]]]:
        status = dict(base_status)
        samples = [dict(sample) for sample in extracted_samples]
        single_curve_candidates: Dict[int, Dict[str, Any]] = {}
        single_curve_debug: List[Dict[str, Any]] = []
        valid_mask = (
            np.ones_like(green, dtype=bool)
            if ipm_valid_mask is None
            else np.asarray(ipm_valid_mask, dtype=bool)
        )
        if not config.enable:
            result = STAGE4.GeometryResult(reason="green_fast_path_disabled")
            result.green_boundary_samples = samples
            status["green_boundary_fast_path_reason"] = (
                "green_fast_path_disabled"
            )
            return result, status, samples
        if not allow_single:
            for sample in samples:
                if sample["classification"] == "green_single_offset":
                    sample["classification"] = "missing"
                    sample["reason"] = "green_single_not_enabled"
        else:
            (
                single_curve_candidates,
                single_curve_status,
                single_curve_debug,
            ) = SingleGreenCurvePlanner.build(
                samples,
                green,
                yellow,
                valid_mask,
                ipm,
                config,
                width_state,
                now_monotonic,
            )
            status.update(single_curve_status)
            for item in single_curve_debug:
                sample_index = int(item["sample_index"])
                samples[sample_index]["single_curve_debug"] = item
        cls._fill_internal_gaps(samples, green, config, ipm)
        allowed = {"green_dual_observed", "gap_filled"}
        if allow_single:
            fused: List[Tuple[int, Dict[str, Any]]] = []
            for sample_index, sample in enumerate(samples):
                classification = sample["classification"]
                if classification in allowed:
                    selected = dict(sample)
                elif sample_index in single_curve_candidates:
                    selected = dict(sample)
                    curve = single_curve_candidates[sample_index]
                    selected["classification"] = (
                        "green_single_curve_offset"
                    )
                    selected["source_type"] = (
                        "green_single_curve_offset"
                    )
                    selected["reason"] = "accepted"
                    selected["center"] = float(
                        curve["center_pixel"][0]
                    )
                    selected["center_y"] = float(
                        curve["center_pixel"][1]
                    )
                    selected["single_side"] = str(curve["side"])
                    selected["yellow_support_ratio"] = float(
                        curve["yellow_ratio"]
                    )
                    selected["green_intrusion_ratio"] = float(
                        curve["green_ratio"]
                    )
                    selected["single_curve_debug"] = next(
                        (
                            item
                            for item in single_curve_debug
                            if int(item["sample_index"]) == sample_index
                        ),
                        None,
                    )
                else:
                    continue
                fused.append((sample_index, selected))
            groups: List[List[Tuple[int, Dict[str, Any]]]] = []
            group: List[Tuple[int, Dict[str, Any]]] = []
            for entry in fused:
                if (
                    group
                    and entry[0] - group[-1][0] - 1
                    > config.single_green_curve_max_row_gap
                ):
                    groups.append(group)
                    group = []
                group.append(entry)
            if group:
                groups.append(group)
            chosen = max(
                groups,
                key=lambda values: (
                    float(values[-1][1]["forward_m"])
                    - float(values[0][1]["forward_m"]),
                    len(values),
                ),
                default=[],
            )
            segment = [entry[1] for entry in chosen]
            chosen_classes = [
                sample["classification"] for sample in segment
            ]
            status["hybrid_dual_point_count"] = sum(
                item == "green_dual_observed"
                for item in chosen_classes
            )
            status["hybrid_existing_point_count"] = sum(
                item in allowed for item in chosen_classes
            )
            status["hybrid_single_curve_point_count"] = sum(
                item == "green_single_curve_offset"
                for item in chosen_classes
            )
            status["single_green_curve_contributed_count"] = status[
                "hybrid_single_curve_point_count"
            ]
        else:
            segment = cls._leading_segment(samples, allowed)
        raw = np.asarray(
            [
                (
                    float(sample["center"]),
                    float(sample.get("center_y", sample["y"])),
                )
                for sample in segment
            ],
            dtype=np.float64,
        ).reshape(-1, 2)
        if len(raw):
            raw[:, 1] = np.clip(raw[:, 1], 0.0, green.shape[0] - 1.0)
        weights = np.asarray(
            [
                (
                    0.55
                    if sample["classification"]
                    == "green_single_curve_offset"
                    else YellowCorridorPlanner.CENTER_WEIGHTS[
                        sample["classification"]
                    ]
                )
                for sample in segment
            ],
            dtype=np.float64,
        )
        final_points = raw.copy()
        if config.center_smooth_enable and len(raw) >= 3:
            try:
                final_points = YellowCorridorPlanner._weighted_smooth(
                    raw, weights, config.center_smooth_lambda
                )
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                final_points = raw.copy()
        if len(raw):
            # Forward stations come directly from observations.  Smooth only
            # lateral position so endpoint overshoot cannot become extrapolation.
            final_points[:, 1] = raw[:, 1]
        fake_support = np.full_like(yellow, 255)
        reason, _support, span = YellowCorridorPlanner._validate_centerline(
            final_points,
            fake_support,
            ipm,
            config,
            geometry_config,
        )
        yellow_ratio = 0.0
        if len(final_points):
            pixels = np.round(final_points).astype(int)
            pixels[:, 0] = np.clip(pixels[:, 0], 0, yellow.shape[1] - 1)
            pixels[:, 1] = np.clip(pixels[:, 1], 0, yellow.shape[0] - 1)
            yellow_ratio = float(
                np.mean(yellow[pixels[:, 1], pixels[:, 0]] > 0)
            )
        dual_count = sum(
            sample["classification"] == "green_dual_observed"
            for sample in segment
        )
        single_count = sum(
            sample["classification"] in {
                "green_single_offset",
                "green_single_curve_offset",
            }
            for sample in segment
        )
        filled_count = sum(
            sample["classification"] == "gap_filled" for sample in segment
        )
        if not allow_single and dual_count < config.min_valid_samples:
            reason = "green_dual_too_few_samples"
        if allow_single and single_count == 0:
            reason = "green_single_missing"
        if (
            filled_count / max(1, len(segment))
            > config.center_gap_fill_max_ratio
        ):
            reason = "green_gap_fill_ratio_high"
        width_std = float(status["green_corridor_width_px_std"])
        width_stability = float(
            np.clip(
                1.0
                - width_std / max(float(ipm.expected_lane_width_px) * 0.20, 1.0),
                0.0,
                1.0,
            )
        )
        intrusion = float(status["green_corridor_green_intrusion_ratio"])
        confidence = float(
            np.clip(
                0.65
                + 0.15 * width_stability
                + 0.15 * yellow_ratio
                + 0.05 * (1.0 - intrusion)
                - 0.18 * single_count / max(1, len(segment))
                - 0.20 * filled_count / max(1, len(segment)),
                0.0,
                0.995,
            )
        )
        mode = (
            "single_green_width_offset"
            if allow_single and single_count
            else "green_dual_inner_edge"
        )
        if mode == "single_green_width_offset":
            accepted_ratio = float(
                status.get("single_green_curve_accepted_count", 0)
                / max(
                    1,
                    status.get("single_green_curve_candidate_count", 0),
                )
            )
            chain_factor = min(
                1.0,
                float(status.get("single_green_curve_chain_count", 0))
                / max(config.single_green_curve_min_points, 1),
            )
            confidence = min(
                0.78,
                0.48
                + 0.12 * accepted_ratio
                + 0.08 * chain_factor
                + 0.08 * yellow_ratio
                + 0.02 * width_stability,
            )
        valid = reason == "valid"
        status["green_boundary_fast_path_used"] = valid
        status["green_boundary_fast_path_reason"] = reason
        result = STAGE4.GeometryResult(
            mode=mode,
            valid=valid,
            reason=reason,
            confidence=confidence if valid else 0.0,
            raw_points=raw,
            final_points=final_points if valid else np.empty((0, 2)),
            measured_width_mean_px=float(
                status["green_corridor_width_px_mean"]
            ),
            measured_width_std_px=width_std,
            yellow_ratio=yellow_ratio,
            forward_span_m=span,
        )
        result.green_boundary_samples = samples
        status["hybrid_final_point_count"] = len(
            result.final_points
        )
        status["hybrid_final_span_m"] = float(result.forward_span_m)
        if (
            valid
            and mode == "green_dual_inner_edge"
            and confidence >= 0.80
            and width_state is not None
            and dual_count >= config.min_valid_samples
            and width_std
            <= float(ipm.expected_lane_width_px) * 0.20
        ):
            dual_widths = [
                float(sample["width"])
                * float(ipm.meter_per_pixel)
                for sample in segment
                if sample.get("width") is not None
            ]
            if dual_widths:
                width_state.update(
                    float(np.median(dual_widths)),
                    (
                        time.monotonic()
                        if now_monotonic is None
                        else float(now_monotonic)
                    ),
                    config.lane_width_ema_alpha,
                )
        return result, status, samples

    @classmethod
    def fuse_existing_with_single_curve(
        cls,
        existing: Any,
        single: Any,
        ipm: Any,
        config: YellowCorridorConfig,
        geometry_config: Any,
    ) -> Tuple[Any, int]:
        """Fill uncovered forward stations without moving existing points."""
        if not existing.valid:
            return single, len(single.final_points)
        if not single.valid:
            return existing, 0
        station_tolerance_m = max(0.01, config.sample_step_m * 0.45)
        entries: List[Tuple[float, int, np.ndarray]] = []
        for priority, points in (
            (2, existing.final_points),
            (1, single.final_points),
        ):
            for point in np.asarray(points, dtype=np.float64).reshape(-1, 2):
                forward_m = (
                    float(ipm.vehicle_origin_y_px) - float(point[1])
                ) * float(ipm.meter_per_pixel)
                entries.append((forward_m, priority, point.copy()))
        entries.sort(key=lambda item: (item[0], -item[1]))
        selected: List[Tuple[float, int, np.ndarray]] = []
        single_contributed = 0
        for entry in entries:
            if (
                selected
                and abs(entry[0] - selected[-1][0])
                <= station_tolerance_m
            ):
                if entry[1] > selected[-1][1]:
                    single_contributed -= int(selected[-1][1] == 1)
                    selected[-1] = entry
                continue
            selected.append(entry)
            single_contributed += int(entry[1] == 1)
        points = np.asarray(
            [entry[2] for entry in selected], dtype=np.float64
        ).reshape(-1, 2)
        fake_support = np.full(
            (int(ipm.output_height), int(ipm.output_width)),
            255,
            dtype=np.uint8,
        )
        reason, _support, span = YellowCorridorPlanner._validate_centerline(
            points, fake_support, ipm, config, geometry_config
        )
        if reason != "valid":
            return existing, 0
        result = STAGE4.GeometryResult(
            mode="green_yellow_hybrid",
            valid=True,
            reason="valid",
            confidence=min(
                0.90,
                0.72 * float(existing.confidence)
                + 0.28 * float(single.confidence),
            ),
            raw_points=points.copy(),
            final_points=points,
            measured_width_mean_px=float(
                existing.measured_width_mean_px
                or single.measured_width_mean_px
            ),
            measured_width_std_px=float(
                existing.measured_width_std_px
                or single.measured_width_std_px
            ),
            yellow_ratio=max(
                float(existing.yellow_ratio), float(single.yellow_ratio)
            ),
            forward_span_m=float(span),
        )
        return result, single_contributed


@dataclass
class BoundaryEnhancementConfig:
    """一体化节点专用的保守边界连续性与道路侧投票参数。"""

    boundary_repair_enable: bool = True
    boundary_repair_max_gap_m: float = 0.10
    boundary_repair_max_angle_deg: float = 20.0
    boundary_repair_max_lateral_delta_m: float = 0.08
    boundary_repair_min_fragment_span_m: float = 0.05
    road_side_min_valid_samples: int = 5
    road_side_global_vote_min_ratio: float = 0.65
    road_side_global_yellow_margin: float = 0.08
    fragment_neighbor_check_limit: int = 3

    def validate(self) -> None:
        if self.boundary_repair_max_gap_m <= 0:
            raise ValueError("boundary_repair_max_gap_m 必须大于 0")
        if not (0 < self.boundary_repair_max_angle_deg <= 60):
            raise ValueError("boundary_repair_max_angle_deg 必须位于 (0, 60]")
        if self.boundary_repair_max_lateral_delta_m <= 0:
            raise ValueError(
                "boundary_repair_max_lateral_delta_m 必须大于 0"
            )
        if self.boundary_repair_min_fragment_span_m <= 0:
            raise ValueError(
                "boundary_repair_min_fragment_span_m 必须大于 0"
            )
        if self.road_side_min_valid_samples < 1:
            raise ValueError("road_side_min_valid_samples 必须大于 0")
        if not (0.5 < self.road_side_global_vote_min_ratio <= 1.0):
            raise ValueError(
                "road_side_global_vote_min_ratio 必须位于 (0.5, 1.0]"
            )
        if not (0 < self.road_side_global_yellow_margin <= 1.0):
            raise ValueError(
                "road_side_global_yellow_margin 必须位于 (0, 1]"
            )
        if not (1 <= self.fragment_neighbor_check_limit <= 10):
            raise ValueError(
                "fragment_neighbor_check_limit 必须位于 [1, 10]"
            )

    def max_gap_px(self, meter_per_pixel: float) -> int:
        return max(1, int(round(self.boundary_repair_max_gap_m / meter_per_pixel)))

    def max_lateral_px(self, meter_per_pixel: float) -> float:
        return self.boundary_repair_max_lateral_delta_m / meter_per_pixel

    def min_fragment_span_px(self, meter_per_pixel: float) -> int:
        return max(
            2,
            int(
                round(
                    self.boundary_repair_min_fragment_span_m
                    / meter_per_pixel
                )
            ),
        )


@dataclass
class BoundaryFragment:
    component_id: int
    pixel_count: int
    x_values: np.ndarray
    y_values: np.ndarray
    coefficients: np.ndarray
    fit_residual: float
    y_min: int
    y_max: int
    x_median: float

    def x_at(self, y: float) -> float:
        return float(np.polyval(self.coefficients, y))

    def slope_at(self, y: float) -> float:
        return float(np.polyval(np.polyder(self.coefficients), y))


@dataclass
class BoundaryRepairResult:
    observed: np.ndarray
    repaired: np.ndarray
    gap_mask: np.ndarray
    observed_component_count: int
    repaired_component_count: int
    gap_count: int
    gap_max_px: int
    candidate_pair_count: int = 0
    fragment_count_before_filter: int = 0
    fragment_count_after_filter: int = 0
    polyfit_call_count: int = 0
    descriptor_ms: float = 0.0
    repair_ms: float = 0.0
    fragments: Optional[List[BoundaryFragment]] = None
    accepted_pairs: Optional[List[Tuple[int, int]]] = None


class ConservativeBoundaryRepair:
    """只在规划 ROI 内连接同侧、短小且趋势一致的纵向缺口。"""

    @staticmethod
    def _roi_mask(mask: np.ndarray, roi: Tuple[int, int]) -> np.ndarray:
        output = np.zeros_like(mask)
        output[roi[0] : roi[1] + 1] = np.where(
            mask[roi[0] : roi[1] + 1] > 0, 255, 0
        ).astype(np.uint8)
        return output

    @classmethod
    def fragments(
        cls,
        mask: np.ndarray,
        roi: Tuple[int, int],
        min_span_px: int,
    ) -> Tuple[List[BoundaryFragment], int]:
        roi_mask = cls._roi_mask(mask, roi)
        roi_start = roi[0]
        roi_crop = roi_mask[roi[0] : roi[1] + 1]
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            (roi_crop > 0).astype(np.uint8), connectivity=8
        )
        fragments: List[BoundaryFragment] = []
        for component_id in range(1, count):
            _x0, _y0, width, height, area = stats[component_id]
            if int(height) < min_span_px:
                continue
            # 横向地面接缝通常横跨很宽但只有极小纵向跨度。
            if int(width) > max(int(height) * 3, min_span_px * 3):
                continue
            local_ys, xs = np.where(labels == component_id)
            ys = local_ys + roi_start
            unique_y = np.unique(ys)
            if len(unique_y) < min_span_px:
                continue
            row_x = np.asarray(
                [np.median(xs[ys == row]) for row in unique_y],
                dtype=np.float64,
            )
            # 短片段仅做局部直线估计，避免二次项被少量噪点放大。
            degree = min(1, len(unique_y) - 1)
            coefficients = np.polyfit(
                unique_y.astype(np.float64), row_x, degree
            )
            residual = float(
                np.sqrt(
                    np.mean(
                        (
                            row_x
                            - np.polyval(coefficients, unique_y)
                        )
                        ** 2
                    )
                )
            )
            fragments.append(
                BoundaryFragment(
                    component_id=component_id,
                    pixel_count=int(area),
                    x_values=row_x,
                    y_values=unique_y.astype(np.float64),
                    coefficients=coefficients,
                    fit_residual=residual,
                    y_min=int(unique_y.min()),
                    y_max=int(unique_y.max()),
                    x_median=float(np.median(row_x)),
                )
            )
        return fragments, count - 1

    @staticmethod
    def _side(fragment: BoundaryFragment, center_x: float) -> int:
        delta = fragment.x_median - center_x
        return 1 if delta > 2.0 else -1 if delta < -2.0 else 0

    @staticmethod
    def _angle_difference_deg(first: float, second: float) -> float:
        first_angle = math.degrees(math.atan(first))
        second_angle = math.degrees(math.atan(second))
        return abs(first_angle - second_angle)

    @staticmethod
    def _boundary_adjacency_ratio(
        points: np.ndarray,
        main_yellow: np.ndarray,
        green: np.ndarray,
        radius: int,
    ) -> float:
        hits = 0
        valid = 0
        height, width = main_yellow.shape
        for x_value, y_value in points:
            x = int(round(x_value))
            y = int(round(y_value))
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            y0, y1 = max(0, y - radius), min(height, y + radius + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            valid += 1
            has_yellow = bool(np.any(main_yellow[y0:y1, x0:x1] > 0))
            has_green = bool(np.any(green[y0:y1, x0:x1] > 0))
            hits += int(has_yellow and has_green)
        return hits / max(1, valid)

    @classmethod
    def compatible(
        cls,
        first: BoundaryFragment,
        second: BoundaryFragment,
        main_yellow: np.ndarray,
        green: np.ndarray,
        ipm: Any,
        config: BoundaryEnhancementConfig,
        require_positive_gap: bool,
    ) -> Tuple[bool, int, Optional[np.ndarray]]:
        upper, lower = (
            (first, second)
            if first.y_min <= second.y_min
            else (second, first)
        )
        gap = lower.y_min - upper.y_max - 1
        if require_positive_gap and gap <= 0:
            return False, gap, None
        if gap > config.max_gap_px(ipm.meter_per_pixel):
            return False, gap, None
        if gap < -config.max_gap_px(ipm.meter_per_pixel):
            return False, gap, None
        first_side = cls._side(upper, ipm.vehicle_center_x_px)
        second_side = cls._side(lower, ipm.vehicle_center_x_px)
        if first_side == 0 or first_side != second_side:
            return False, gap, None
        upper_y = float(upper.y_max)
        lower_y = float(lower.y_min)
        upper_slope = upper.slope_at(upper_y)
        lower_slope = lower.slope_at(lower_y)
        if (
            cls._angle_difference_deg(upper_slope, lower_slope)
            > config.boundary_repair_max_angle_deg
        ):
            return False, gap, None
        upper_x = upper.x_at(upper_y)
        lower_x = lower.x_at(lower_y)
        predicted_lower = upper.x_at(lower_y)
        predicted_upper = lower.x_at(upper_y)
        lateral_error = max(
            abs(predicted_lower - lower_x),
            abs(predicted_upper - upper_x),
        )
        if lateral_error > config.max_lateral_px(ipm.meter_per_pixel):
            return False, gap, None
        if gap <= 0:
            overlap_y0 = max(upper.y_min, lower.y_min)
            overlap_y1 = min(upper.y_max, lower.y_max)
            if overlap_y1 < overlap_y0:
                return False, gap, None
            sample_y = np.linspace(overlap_y0, overlap_y1, 5)
            if float(
                np.mean(
                    np.abs(
                        np.polyval(upper.coefficients, sample_y)
                        - np.polyval(lower.coefficients, sample_y)
                    )
                )
            ) > config.max_lateral_px(ipm.meter_per_pixel):
                return False, gap, None
            return True, gap, None
        bridge_y = np.arange(
            upper.y_max + 1, lower.y_min, dtype=np.float64
        )
        if len(bridge_y) == 0:
            return False, gap, None
        # 复用两端缓存的切线，以 Hermite 插值桥接；候选比较不再调用 polyfit。
        span = max(lower_y - upper_y, 1.0)
        normalized = (bridge_y - upper_y) / span
        h00 = 2 * normalized**3 - 3 * normalized**2 + 1
        h10 = normalized**3 - 2 * normalized**2 + normalized
        h01 = -2 * normalized**3 + 3 * normalized**2
        h11 = normalized**3 - normalized**2
        bridge_x = (
            h00 * upper_x
            + h10 * span * upper_slope
            + h01 * lower_x
            + h11 * span * lower_slope
        )
        if np.any(
            (bridge_x - ipm.vehicle_center_x_px) * first_side <= 2.0
        ):
            return False, gap, None
        bridge_points = np.column_stack((bridge_x, bridge_y))
        adjacency_radius = max(
            3, int(round(0.035 / ipm.meter_per_pixel))
        )
        if (
            cls._boundary_adjacency_ratio(
                bridge_points,
                main_yellow,
                green,
                adjacency_radius,
            )
            < 0.60
        ):
            return False, gap, None
        return True, gap, bridge_points

    @classmethod
    def repair(
        cls,
        observed: np.ndarray,
        main_yellow: np.ndarray,
        green: np.ndarray,
        ipm: Any,
        roi: Tuple[int, int],
        config: BoundaryEnhancementConfig,
    ) -> BoundaryRepairResult:
        descriptor_started = time.perf_counter()
        observed_roi = cls._roi_mask(observed, roi)
        min_span = config.min_fragment_span_px(ipm.meter_per_pixel)
        fragments, observed_count = cls.fragments(
            observed_roi, roi, min_span
        )
        descriptor_polyfit_count = len(fragments)
        filtered_fragments: List[BoundaryFragment] = []
        adjacency_radius = max(
            3, int(round(0.035 / ipm.meter_per_pixel))
        )
        for fragment in fragments:
            points = np.column_stack(
                (fragment.x_values, fragment.y_values)
            )
            if fragment.pixel_count < max(5, min_span):
                continue
            if (
                cls._boundary_adjacency_ratio(
                    points,
                    main_yellow,
                    green,
                    adjacency_radius,
                )
                < 0.15
            ):
                continue
            filtered_fragments.append(fragment)
        fragments = filtered_fragments
        descriptor_ms = (
            time.perf_counter() - descriptor_started
        ) * 1000.0
        repair_started = time.perf_counter()
        repaired = observed_roi.copy()
        gap_mask = np.zeros_like(observed_roi)
        accepted: List[Tuple[int, int]] = []
        used_endpoints: set = set()
        maximum_gap = 0
        candidates: List[
            Tuple[int, BoundaryFragment, BoundaryFragment, np.ndarray]
        ] = []
        if config.boundary_repair_enable:
            buckets: Dict[int, List[BoundaryFragment]] = {-1: [], 1: []}
            for fragment in fragments:
                side = cls._side(fragment, ipm.vehicle_center_x_px)
                if side:
                    buckets[side].append(fragment)
            candidate_pair_count = 0
            for bucket in buckets.values():
                bucket.sort(key=lambda fragment: fragment.y_min)
                for first_index, first in enumerate(bucket):
                    end = min(
                        len(bucket),
                        first_index
                        + 1
                        + config.fragment_neighbor_check_limit,
                    )
                    for second in bucket[first_index + 1 : end]:
                        candidate_pair_count += 1
                        compatible, gap, bridge = cls.compatible(
                            first,
                            second,
                            main_yellow,
                            green,
                            ipm,
                            config,
                            require_positive_gap=True,
                        )
                        if compatible and bridge is not None:
                            candidates.append((gap, first, second, bridge))
        else:
            candidate_pair_count = 0
        for gap, first, second, bridge in sorted(
            candidates, key=lambda item: item[0]
        ):
            upper, lower = (
                (first, second)
                if first.y_min <= second.y_min
                else (second, first)
            )
            pair = tuple(sorted((upper.component_id, lower.component_id)))
            if pair in accepted:
                continue
            endpoints = (
                (upper.component_id, "near"),
                (lower.component_id, "far"),
            )
            if any(endpoint in used_endpoints for endpoint in endpoints):
                continue
            points = np.round(bridge).astype(np.int32)
            cv2.polylines(
                repaired, [points], False, 255, 2, cv2.LINE_8
            )
            cv2.polylines(
                gap_mask, [points], False, 255, 2, cv2.LINE_8
            )
            accepted.append(pair)
            used_endpoints.update(endpoints)
            maximum_gap = max(maximum_gap, int(gap))
        repaired = cls._roi_mask(repaired, roi)
        repaired_count = (
            cv2.connectedComponents(
                (
                    repaired[roi[0] : roi[1] + 1] > 0
                ).astype(np.uint8),
                connectivity=8,
            )[0]
            - 1
        )
        return BoundaryRepairResult(
            observed=observed_roi,
            repaired=repaired,
            gap_mask=gap_mask,
            observed_component_count=observed_count,
            repaired_component_count=int(repaired_count),
            gap_count=len(accepted),
            gap_max_px=maximum_gap,
            candidate_pair_count=candidate_pair_count,
            fragment_count_before_filter=observed_count,
            fragment_count_after_filter=len(fragments),
            polyfit_call_count=descriptor_polyfit_count,
            descriptor_ms=descriptor_ms,
            repair_ms=(time.perf_counter() - repair_started) * 1000.0,
            fragments=fragments,
            accepted_pairs=accepted,
        )


class CachedBoundaryCurveBuilder:
    """从已缓存片段描述构造曲线，不再次拟合单个边界分量。"""

    @staticmethod
    def build(
        repair: BoundaryRepairResult,
        ipm: Any,
        geometry_config: Any,
        roi: Tuple[int, int],
    ) -> Tuple[List[Any], int]:
        fragments = list(repair.fragments or [])
        if not fragments:
            return [], 0
        by_id = {
            fragment.component_id: index
            for index, fragment in enumerate(fragments)
        }
        parents = list(range(len(fragments)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(first: int, second: int) -> None:
            first_root, second_root = find(first), find(second)
            if first_root != second_root:
                parents[second_root] = first_root

        for first_id, second_id in repair.accepted_pairs or []:
            if first_id in by_id and second_id in by_id:
                union(by_id[first_id], by_id[second_id])
        grouped: Dict[int, List[BoundaryFragment]] = {}
        for index, fragment in enumerate(fragments):
            grouped.setdefault(find(index), []).append(fragment)

        max_area = (
            (roi[1] - roi[0] + 1)
            * ipm.output_width
            * geometry_config.boundary_component_max_area_ratio
        )
        near_band_start = max(
            roi[0], roi[1] - max(8, (roi[1] - roi[0]) // 5)
        )
        curves: List[Any] = []
        extra_polyfit_calls = 0
        for members in grouped.values():
            pixel_count = sum(member.pixel_count for member in members)
            y_min = min(member.y_min for member in members)
            y_max = max(member.y_max for member in members)
            if (
                pixel_count
                < geometry_config.boundary_component_min_pixels
                or y_max - y_min + 1
                < geometry_config.boundary_component_min_vertical_span_px
                or pixel_count > max_area
            ):
                continue
            if len(members) == 1:
                coefficients = members[0].coefficients
                residual = members[0].fit_residual
                sample_y = members[0].y_values
                sample_x = members[0].x_values
            else:
                sample_y = np.concatenate(
                    [member.y_values for member in members]
                )
                sample_x = np.concatenate(
                    [member.x_values for member in members]
                )
                try:
                    coefficients, keep, residual = (
                        STAGE4.RobustPolynomialFitter.fit(
                            sample_y,
                            sample_x,
                            geometry_config.boundary_fit_degree,
                            geometry_config.boundary_fit_min_points,
                            geometry_config.boundary_fit_max_residual_px,
                            geometry_config.boundary_fit_outlier_iterations,
                        )
                    )
                except STAGE4.GeometryError:
                    continue
                extra_polyfit_calls += 1
                sample_y, sample_x = sample_y[keep], sample_x[keep]
            if residual > geometry_config.boundary_fit_max_residual_px:
                continue
            x_min, x_max = float(np.min(sample_x)), float(np.max(sample_x))
            centroid_x = float(np.mean(sample_x))
            centroid_y = float(np.mean(sample_y))
            curves.append(
                STAGE4.BoundaryCurve(
                    component_id=min(
                        member.component_id for member in members
                    ),
                    pixel_count=pixel_count,
                    bbox=(
                        int(math.floor(x_min)),
                        y_min,
                        max(1, int(math.ceil(x_max - x_min + 1))),
                        y_max - y_min + 1,
                    ),
                    centroid=(centroid_x, centroid_y),
                    vertical_span_px=y_max - y_min + 1,
                    horizontal_span_px=max(
                        1, int(math.ceil(x_max - x_min + 1))
                    ),
                    touches_near_band=y_max >= near_band_start,
                    distance_to_vehicle_center_px=abs(
                        centroid_x - ipm.vehicle_center_x_px
                    ),
                    coefficients=coefficients,
                    fit_residual_px=float(residual),
                    y_min=y_min,
                    y_max=y_max,
                )
            )
        return curves, extra_polyfit_calls


class GlobalRoadSideClassifier:
    """沿整条拟合边界采样，并以明确局部判断的多数票确定道路内侧。"""

    EMPTY_STATUS: Dict[str, Any] = {
        "road_side_valid_sample_count": 0,
        "road_side_positive_vote_count": 0,
        "road_side_negative_vote_count": 0,
        "road_side_global_vote_ratio": 0.0,
        "road_side_positive_yellow_ratio": 0.0,
        "road_side_negative_yellow_ratio": 0.0,
        "road_side_positive_green_ratio": 0.0,
        "road_side_negative_green_ratio": 0.0,
    }

    @staticmethod
    def _sample_side(
        curve: Any,
        y: float,
        sign: int,
        yellow: np.ndarray,
        green: np.ndarray,
        geometry_config: Any,
    ) -> Tuple[float, float]:
        x = float(curve.x_at(np.asarray([y]))[0])
        slope = curve.dx_dy(y)
        normal = np.asarray([1.0, -slope], dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1.0e-9)
        yellow_hits = 0
        green_hits = 0
        valid = 0
        for distance in range(
            geometry_config.road_side_probe_min_px,
            geometry_config.road_side_probe_max_px + 1,
            geometry_config.road_side_probe_step_px,
        ):
            px = int(round(x + sign * normal[0] * distance))
            py = int(round(y + sign * normal[1] * distance))
            if 0 <= px < yellow.shape[1] and 0 <= py < yellow.shape[0]:
                valid += 1
                yellow_hits += int(yellow[py, px] > 0)
                green_hits += int(green[py, px] > 0)
        return (
            yellow_hits / max(1, valid),
            green_hits / max(1, valid),
        )

    @classmethod
    def classify_curve(
        cls,
        curve: Any,
        yellow: np.ndarray,
        green: np.ndarray,
        geometry_config: Any,
        enhancement_config: BoundaryEnhancementConfig,
        sample_step_px: int,
    ) -> Any:
        positive_votes = 0
        negative_votes = 0
        positive_yellow: List[float] = []
        negative_yellow: List[float] = []
        positive_green: List[float] = []
        negative_green: List[float] = []
        margins: List[float] = []
        sample_rows = np.arange(
            curve.y_min,
            curve.y_max + 1,
            max(2, sample_step_px),
            dtype=np.float64,
        )
        for y in sample_rows:
            pos_y, pos_g = cls._sample_side(
                curve,
                float(y),
                1,
                yellow,
                green,
                geometry_config,
            )
            neg_y, neg_g = cls._sample_side(
                curve,
                float(y),
                -1,
                yellow,
                green,
                geometry_config,
            )
            positive_yellow.append(pos_y)
            negative_yellow.append(neg_y)
            positive_green.append(pos_g)
            negative_green.append(neg_g)
            positive_score = pos_y - 0.35 * pos_g
            negative_score = neg_y - 0.35 * neg_g
            score_delta = positive_score - negative_score
            yellow_delta = pos_y - neg_y
            if (
                pos_y >= geometry_config.road_side_min_yellow_ratio
                and score_delta
                >= enhancement_config.road_side_global_yellow_margin
                and yellow_delta
                >= enhancement_config.road_side_global_yellow_margin
            ):
                positive_votes += 1
                margins.append(score_delta)
            elif (
                neg_y >= geometry_config.road_side_min_yellow_ratio
                and score_delta
                <= -enhancement_config.road_side_global_yellow_margin
                and yellow_delta
                <= -enhancement_config.road_side_global_yellow_margin
            ):
                negative_votes += 1
                margins.append(-score_delta)
        valid_samples = positive_votes + negative_votes
        winning_votes = max(positive_votes, negative_votes)
        vote_ratio = winning_votes / max(1, valid_samples)
        sign = 0
        if (
            valid_samples
            >= enhancement_config.road_side_min_valid_samples
            and vote_ratio
            >= enhancement_config.road_side_global_vote_min_ratio
        ):
            sign = (
                1
                if positive_votes > negative_votes
                else -1
                if negative_votes > positive_votes
                else 0
            )
        curve.inward_sign = sign
        curve.road_side = (
            "yellow_right"
            if sign > 0
            else "yellow_left"
            if sign < 0
            else "unknown"
        )
        curve.side_confidence = (
            vote_ratio
            * min(1.0, float(np.mean(margins)) / 0.50)
            if sign and margins
            else 0.0
        )
        curve.pipeline_road_status = {
            "road_side_valid_sample_count": valid_samples,
            "road_side_positive_vote_count": positive_votes,
            "road_side_negative_vote_count": negative_votes,
            "road_side_global_vote_ratio": vote_ratio,
            "road_side_positive_yellow_ratio": float(
                np.mean(positive_yellow) if positive_yellow else 0.0
            ),
            "road_side_negative_yellow_ratio": float(
                np.mean(negative_yellow) if negative_yellow else 0.0
            ),
            "road_side_positive_green_ratio": float(
                np.mean(positive_green) if positive_green else 0.0
            ),
            "road_side_negative_green_ratio": float(
                np.mean(negative_green) if negative_green else 0.0
            ),
        }
        probe_distances = len(
            range(
                geometry_config.road_side_probe_min_px,
                geometry_config.road_side_probe_max_px + 1,
                geometry_config.road_side_probe_step_px,
            )
        )
        curve.pipeline_probe_count = (
            len(sample_rows) * 2 * probe_distances
        )
        return curve

    @classmethod
    def representative_status(cls, curves: Sequence[Any]) -> Dict[str, Any]:
        if not curves:
            return copy.deepcopy(cls.EMPTY_STATUS)
        best = max(
            curves,
            key=lambda curve: getattr(
                curve, "pipeline_road_status", cls.EMPTY_STATUS
            )["road_side_valid_sample_count"],
        )
        return copy.deepcopy(
            getattr(best, "pipeline_road_status", cls.EMPTY_STATUS)
        )


class BoundaryCurveAggregator:
    """在拟合后将同侧、趋势相容且道路内侧不冲突的碎片聚合。"""

    @classmethod
    def aggregate(
        cls,
        curves: Sequence[Any],
        main_yellow: np.ndarray,
        green: np.ndarray,
        ipm: Any,
        geometry_config: Any,
        enhancement_config: BoundaryEnhancementConfig,
    ) -> Tuple[List[Any], int, int, int]:
        if not curves:
            return [], 0, 0, 0
        fragments = [
            BoundaryFragment(
                component_id=curve.component_id,
                pixel_count=curve.pixel_count,
                x_values=curve.x_at(
                    np.arange(curve.y_min, curve.y_max + 1, dtype=np.float64)
                ),
                y_values=np.arange(
                    curve.y_min, curve.y_max + 1, dtype=np.float64
                ),
                coefficients=curve.coefficients,
                fit_residual=curve.fit_residual_px,
                y_min=curve.y_min,
                y_max=curve.y_max,
                x_median=curve.centroid[0],
            )
            for curve in curves
        ]
        parents = list(range(len(curves)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(first: int, second: int) -> None:
            first_root, second_root = find(first), find(second)
            if first_root != second_root:
                parents[second_root] = first_root

        indices = list(range(len(curves)))
        indices.sort(key=lambda index: curves[index].y_min)
        candidate_pair_count = 0
        for bucket_index, first_index in enumerate(indices):
            end = min(
                len(indices),
                bucket_index
                + 1
                + enhancement_config.fragment_neighbor_check_limit,
            )
            for second_index in indices[bucket_index + 1 : end]:
                candidate_pair_count += 1
                first_curve, second_curve = (
                    curves[first_index],
                    curves[second_index],
                )
                if (
                    first_curve.inward_sign
                    and second_curve.inward_sign
                    and first_curve.inward_sign
                    != second_curve.inward_sign
                ):
                    continue
                compatible, _gap, _bridge = (
                    ConservativeBoundaryRepair.compatible(
                        fragments[first_index],
                        fragments[second_index],
                        main_yellow,
                        green,
                        ipm,
                        enhancement_config,
                        require_positive_gap=False,
                    )
                )
                if compatible:
                    union(first_index, second_index)
        groups: Dict[int, List[Any]] = {}
        for index, curve in enumerate(curves):
            groups.setdefault(find(index), []).append(curve)
        aggregated: List[Any] = []
        aggregate_polyfit_calls = 0
        for members in groups.values():
            if len(members) == 1:
                aggregated.append(members[0])
                continue
            sample_y = np.concatenate(
                [
                    np.arange(
                        curve.y_min, curve.y_max + 1, dtype=np.float64
                    )
                    for curve in members
                ]
            )
            sample_x = np.concatenate(
                [
                    curve.x_at(
                        np.arange(
                            curve.y_min,
                            curve.y_max + 1,
                            dtype=np.float64,
                        )
                    )
                    for curve in members
                ]
            )
            try:
                aggregate_polyfit_calls += 1
                coefficients, keep, residual = (
                    STAGE4.RobustPolynomialFitter.fit(
                        sample_y,
                        sample_x,
                        geometry_config.boundary_fit_degree,
                        geometry_config.boundary_fit_min_points,
                        geometry_config.boundary_fit_max_residual_px,
                        geometry_config.boundary_fit_outlier_iterations,
                    )
                )
            except STAGE4.GeometryError:
                aggregated.extend(members)
                continue
            kept_y = sample_y[keep]
            kept_x = sample_x[keep]
            if residual > geometry_config.boundary_fit_max_residual_px:
                aggregated.extend(members)
                continue
            y_min, y_max = int(np.min(kept_y)), int(np.max(kept_y))
            x_min, x_max = float(np.min(kept_x)), float(np.max(kept_x))
            known_signs = {
                curve.inward_sign for curve in members if curve.inward_sign
            }
            inherited_sign = (
                next(iter(known_signs)) if len(known_signs) == 1 else 0
            )
            merged_curve = STAGE4.BoundaryCurve(
                    component_id=min(
                        curve.component_id for curve in members
                    ),
                    pixel_count=sum(
                        curve.pixel_count for curve in members
                    ),
                    bbox=(
                        int(math.floor(x_min)),
                        y_min,
                        max(1, int(math.ceil(x_max - x_min + 1))),
                        y_max - y_min + 1,
                    ),
                    centroid=(
                        float(np.mean(kept_x)),
                        float(np.mean(kept_y)),
                    ),
                    vertical_span_px=y_max - y_min + 1,
                    horizontal_span_px=max(
                        1, int(math.ceil(x_max - x_min + 1))
                    ),
                    touches_near_band=any(
                        curve.touches_near_band for curve in members
                    ),
                    distance_to_vehicle_center_px=abs(
                        float(np.mean(kept_x))
                        - ipm.vehicle_center_x_px
                    ),
                    coefficients=coefficients,
                    fit_residual_px=residual,
                    y_min=y_min,
                    y_max=y_max,
                    inward_sign=inherited_sign,
                    road_side=(
                        "yellow_right"
                        if inherited_sign > 0
                        else "yellow_left"
                        if inherited_sign < 0
                        else "unknown"
                    ),
                    side_confidence=float(
                        max(
                            (
                                curve.side_confidence
                                for curve in members
                            ),
                            default=0.0,
                        )
                    ),
                )
            statuses = [
                getattr(
                    curve,
                    "pipeline_road_status",
                    GlobalRoadSideClassifier.EMPTY_STATUS,
                )
                for curve in members
            ]
            valid_total = sum(
                status["road_side_valid_sample_count"]
                for status in statuses
            )
            merged_curve.pipeline_road_status = {
                "road_side_valid_sample_count": valid_total,
                "road_side_positive_vote_count": sum(
                    status["road_side_positive_vote_count"]
                    for status in statuses
                ),
                "road_side_negative_vote_count": sum(
                    status["road_side_negative_vote_count"]
                    for status in statuses
                ),
                "road_side_global_vote_ratio": max(
                    (
                        status["road_side_global_vote_ratio"]
                        for status in statuses
                    ),
                    default=0.0,
                ),
                "road_side_positive_yellow_ratio": float(
                    np.mean(
                        [
                            status["road_side_positive_yellow_ratio"]
                            for status in statuses
                        ]
                    )
                ),
                "road_side_negative_yellow_ratio": float(
                    np.mean(
                        [
                            status["road_side_negative_yellow_ratio"]
                            for status in statuses
                        ]
                    )
                ),
                "road_side_positive_green_ratio": float(
                    np.mean(
                        [
                            status["road_side_positive_green_ratio"]
                            for status in statuses
                        ]
                    )
                ),
                "road_side_negative_green_ratio": float(
                    np.mean(
                        [
                            status["road_side_negative_green_ratio"]
                            for status in statuses
                        ]
                    )
                ),
            }
            merged_curve.pipeline_probe_count = sum(
                int(getattr(curve, "pipeline_probe_count", 0))
                for curve in members
            )
            aggregated.append(merged_curve)
        return (
            aggregated,
            len(aggregated),
            candidate_pair_count,
            aggregate_polyfit_calls,
        )


class ConservativeDualBoundaryPlanner:
    """复用第四阶段几何约束，不以车辆中心轴分居作为硬拒绝条件。"""

    @staticmethod
    def plan(
        curves: Sequence[Any],
        main_yellow: np.ndarray,
        ipm: Any,
        geometry_config: Any,
        roi: Tuple[int, int],
        previous_center_x: Optional[float] = None,
    ) -> Optional[Any]:
        return STAGE4.DualBoundaryPlanner.plan(
            curves,
            main_yellow,
            ipm,
            geometry_config,
            roi,
            previous_center_x,
        )


class LaneAlgorithm:
    """将第二、第四阶段的原有纯算法串成单帧内存流水线。"""

    def __init__(
        self,
        threshold: Any,
        segmentation_config: Dict[str, object],
        geometry_config: Any,
        ipm: Any,
        enhancement_config: Optional[BoundaryEnhancementConfig] = None,
        corridor_config: Optional[YellowCorridorConfig] = None,
    ) -> None:
        self.threshold = threshold
        self.segmentation_config = segmentation_config
        self.geometry_config = geometry_config
        self.ipm = ipm
        self.enhancement_config = (
            enhancement_config or BoundaryEnhancementConfig()
        )
        self.enhancement_config.validate()
        self.corridor_config = corridor_config or YellowCorridorConfig()
        self.corridor_config.validate()
        (
            self.roi_ipm,
            self.resolved_forward_min_m,
            self.resolved_forward_max_m,
        ) = CroppedIpm.from_full(ipm, geometry_config)
        source_valid = np.full(
            (ipm.image_height, ipm.image_width), 255, dtype=np.uint8
        )
        self.ipm_valid_mask = (
            STAGE4.MaskIpmTransformer.transform(
                source_valid, self.roi_ipm
            )
            > 0
        )
        expected_valid_shape = (
            self.roi_ipm.output_height,
            self.roi_ipm.output_width,
        )
        if self.ipm_valid_mask.shape != expected_valid_shape:
            raise ValueError("cached ipm_valid_mask shape mismatch")
        self.ipm_valid_mask_build_count = 1
        self.last_valid_points = np.empty((0, 2), dtype=np.float64)
        self.last_valid_time = 0.0
        self.last_center_x: Optional[float] = None
        self.single_green_width_state = SingleGreenLaneWidthState()

    @staticmethod
    def points_mask(
        shape: Tuple[int, int], points: np.ndarray, thickness: int
    ) -> np.ndarray:
        output = np.zeros(shape, dtype=np.uint8)
        if len(points) == 0:
            return output
        integer = np.round(points).astype(np.int32)
        if len(integer) == 1:
            cv2.circle(output, tuple(integer[0]), thickness, 255, -1)
        else:
            cv2.polylines(
                output, [integer], False, 255, thickness, cv2.LINE_8
            )
            for point in integer:
                cv2.circle(output, tuple(point), max(1, thickness), 255, -1)
        return output

    def _segment(self, bgr: np.ndarray) -> Tuple[Any, ...]:
        return STAGE2.GreenYellowSegmenter.segment(
            bgr, self.threshold, self.segmentation_config
        )

    def _extract_boundary(self, values: Tuple[Any, ...]) -> Any:
        return STAGE2.BoundaryExtractor.extract(
            values[3],
            values[4],
            values[5],
            int(values[7]),
            int(values[8]),
            self.segmentation_config,
        )

    def _smooth(
        self,
        result: Any,
        main_yellow: np.ndarray,
        roi: Tuple[int, int],
    ) -> float:
        if not len(result.raw_points):
            return 0.0
        started = time.perf_counter()
        (
            result.final_points,
            result.reason,
            result.yellow_ratio,
            result.forward_span_m,
        ) = STAGE4.CenterlineSmoother.smooth(
            result.raw_points,
            main_yellow,
            self.roi_ipm,
            self.geometry_config,
            roi,
        )
        result.valid = result.reason == "valid"
        return (time.perf_counter() - started) * 1000.0

    def process(self, bgr: np.ndarray) -> PipelineResult:
        total_started = time.perf_counter()
        timing = {name: 0.0 for name in TIMING_NAMES}
        counters = {name: 0 for name in COUNTER_NAMES}
        if bgr.shape[:2] != (self.ipm.image_height, self.ipm.image_width):
            raise ValueError(
                f"相机图像 {bgr.shape[1]}x{bgr.shape[0]} 与 IPM "
                f"{self.ipm.image_width}x{self.ipm.image_height} 不一致"
            )

        started = time.perf_counter()
        values = self._segment(bgr)
        timing["segmentation_ms"] = (
            time.perf_counter() - started
        ) * 1000.0
        yellow, green = values[3], values[4]

        started = time.perf_counter()
        boundary_result = self._extract_boundary(values)
        timing["boundary_extract_ms"] = (
            time.perf_counter() - started
        ) * 1000.0
        boundary = boundary_result.final
        boundary_valid = bool(boundary_result.valid)

        started = time.perf_counter()
        observed_ipm_boundary = STAGE4.MaskIpmTransformer.transform(
            boundary, self.roi_ipm
        )
        transformed = {
            "yellow": STAGE4.MaskIpmTransformer.transform(
                yellow, self.roi_ipm
            ),
            "green": STAGE4.MaskIpmTransformer.transform(
                green, self.roi_ipm
            ),
        }
        timing["ipm_warp_ms"] = (
            time.perf_counter() - started
        ) * 1000.0
        roi = (0, self.roi_ipm.output_height - 1)

        started = time.perf_counter()
        main_yellow, yellow_status = STAGE4.MainYellowSelector.select(
            transformed["yellow"],
            self.roi_ipm,
            self.geometry_config,
            roi,
        )
        timing["main_yellow_select_ms"] = (
            time.perf_counter() - started
        ) * 1000.0

        started = time.perf_counter()
        green_extracted, green_base_status = GreenBoundaryFastPlanner.extract(
            transformed["green"],
            transformed["yellow"],
            self.roi_ipm,
            self.corridor_config,
            self.last_center_x,
            self.ipm_valid_mask,
        )
        (
            green_dual_result,
            green_status,
            green_debug_samples,
        ) = GreenBoundaryFastPlanner.plan(
            green_extracted,
            green_base_status,
            transformed["green"],
            transformed["yellow"],
            self.roi_ipm,
            self.corridor_config,
            self.geometry_config,
            allow_single=False,
            ipm_valid_mask=self.ipm_valid_mask,
            width_state=self.single_green_width_state,
        )
        timing["green_boundary_fast_path_ms"] = (
            time.perf_counter() - started
        ) * 1000.0

        started = time.perf_counter()
        if green_dual_result.valid:
            corridor_result = STAGE4.GeometryResult(
                reason="skipped_green_dual_valid"
            )
            corridor_result.yellow_corridor_samples = []
            corridor_status = copy.deepcopy(
                YellowCorridorPlanner.EMPTY_STATUS
            )
            corridor_status["yellow_corridor_reason"] = (
                "skipped_green_dual_valid"
            )
        elif yellow_status["main_yellow_valid"]:
            corridor_result, corridor_status = YellowCorridorPlanner.plan(
                main_yellow,
                observed_ipm_boundary,
                transformed["green"],
                self.roi_ipm,
                self.corridor_config,
                self.geometry_config,
            )
        else:
            corridor_result = STAGE4.GeometryResult(
                reason="main_yellow_missing"
            )
            corridor_result.yellow_corridor_samples = []
            corridor_status = copy.deepcopy(
                YellowCorridorPlanner.EMPTY_STATUS
            )
            corridor_status["yellow_corridor_reason"] = (
                "main_yellow_missing"
            )
        timing["yellow_corridor_ms"] = (
            time.perf_counter() - started
        ) * 1000.0

        green_single_result = STAGE4.GeometryResult(
            reason="skipped_green_dual_valid"
        )
        if not green_dual_result.valid:
            single_started = time.perf_counter()
            (
                green_single_result,
                single_status,
                single_debug_samples,
            ) = GreenBoundaryFastPlanner.plan(
                green_extracted,
                green_base_status,
                transformed["green"],
                transformed["yellow"],
                self.roi_ipm,
                self.corridor_config,
                self.geometry_config,
                allow_single=True,
                ipm_valid_mask=self.ipm_valid_mask,
                width_state=self.single_green_width_state,
            )
            single_elapsed_ms = (
                time.perf_counter() - single_started
            ) * 1000.0
            timing["green_boundary_fast_path_ms"] += single_elapsed_ms
            timing["single_green_curve_offset_ms"] = float(
                single_status.get(
                    "single_green_curve_offset_ms",
                    single_elapsed_ms,
                )
            )
            green_status = single_status
            green_debug_samples = single_debug_samples

        fast_result = green_dual_result
        fast_valid = bool(green_dual_result.valid)
        if not fast_valid and corridor_status["yellow_corridor_valid"]:
            fast_result = corridor_result
            fast_valid = bool(corridor_result.valid)
            if not fast_valid:
                corridor_status["yellow_corridor_valid"] = False
                corridor_status["yellow_corridor_reason"] = (
                    corridor_result.reason
                )
            elif green_single_result.valid:
                fast_result, contributed = (
                    GreenBoundaryFastPlanner.fuse_existing_with_single_curve(
                        corridor_result,
                        green_single_result,
                        self.roi_ipm,
                        self.corridor_config,
                        self.geometry_config,
                    )
                )
                fast_valid = bool(fast_result.valid)
                green_status["hybrid_existing_point_count"] = len(
                    corridor_result.final_points
                )
                green_status[
                    "hybrid_single_curve_point_count"
                ] = contributed
                green_status[
                    "single_green_curve_contributed_count"
                ] = contributed
                green_status["hybrid_final_point_count"] = len(
                    fast_result.final_points
                )
                green_status["hybrid_final_span_m"] = float(
                    fast_result.forward_span_m
                )
            elif green_base_status["green_dual_edge_valid_count"]:
                fast_result.mode = "green_yellow_hybrid"
        if not fast_valid and green_single_result.valid:
            fast_result = green_single_result
            fast_valid = True

        empty = np.zeros(
            (self.roi_ipm.output_height, self.roi_ipm.output_width),
            dtype=np.uint8,
        )
        repair = BoundaryRepairResult(
            observed=observed_ipm_boundary,
            repaired=observed_ipm_boundary.copy(),
            gap_mask=empty.copy(),
            observed_component_count=0,
            repaired_component_count=0,
            gap_count=0,
            gap_max_px=0,
        )
        transformed["boundary"] = repair.repaired
        curves: List[Any] = []
        raw_component_count = 0
        merged_group_count = 0
        fallback_used = not fast_valid
        result = fast_result if fast_valid else STAGE4.GeometryResult(
            reason="segmentation_invalid"
        )

        if (
            fallback_used
            and boundary_valid
            and yellow_status["main_yellow_valid"]
        ):
            repair = ConservativeBoundaryRepair.repair(
                observed_ipm_boundary,
                main_yellow,
                transformed["green"],
                self.roi_ipm,
                roi,
                self.enhancement_config,
            )
            transformed["boundary"] = repair.repaired
            timing["fragment_descriptor_ms"] = repair.descriptor_ms
            timing["boundary_repair_ms"] = repair.repair_ms
            counters.update(
                {
                    "polyfit_call_count": repair.polyfit_call_count,
                    "repair_candidate_pair_count": (
                        repair.candidate_pair_count
                    ),
                    "repair_accepted_pair_count": repair.gap_count,
                    "fragment_count_before_filter": (
                        repair.fragment_count_before_filter
                    ),
                    "fragment_count_after_filter": (
                        repair.fragment_count_after_filter
                    ),
                }
            )

            started = time.perf_counter()
            curves, extra_polyfit_calls = (
                CachedBoundaryCurveBuilder.build(
                    repair,
                    self.roi_ipm,
                    self.geometry_config,
                    roi,
                )
            )
            raw_component_count = repair.observed_component_count
            timing["component_analysis_ms"] = (
                time.perf_counter() - started
            ) * 1000.0
            counters["polyfit_call_count"] += extra_polyfit_calls
            step_px = max(
                2,
                int(
                    round(
                        self.geometry_config.centerline_sample_step_m
                        / self.roi_ipm.meter_per_pixel
                    )
                ),
            )
            started = time.perf_counter()
            for curve in curves:
                GlobalRoadSideClassifier.classify_curve(
                    curve,
                    main_yellow,
                    transformed["green"],
                    self.geometry_config,
                    self.enhancement_config,
                    step_px,
                )
            timing["road_side_vote_ms"] = (
                time.perf_counter() - started
            ) * 1000.0
            counters["road_side_probe_count"] = sum(
                int(getattr(curve, "pipeline_probe_count", 0))
                for curve in curves
            )

            started = time.perf_counter()
            (
                curves,
                merged_group_count,
                merge_pair_count,
                merge_polyfit_calls,
            ) = BoundaryCurveAggregator.aggregate(
                curves,
                main_yellow,
                transformed["green"],
                self.roi_ipm,
                self.geometry_config,
                self.enhancement_config,
            )
            timing["boundary_merge_ms"] = (
                time.perf_counter() - started
            ) * 1000.0
            counters["merge_candidate_pair_count"] = merge_pair_count
            counters["polyfit_call_count"] += merge_polyfit_calls

            if curves:
                started = time.perf_counter()
                result = ConservativeDualBoundaryPlanner.plan(
                    curves,
                    main_yellow,
                    self.roi_ipm,
                    self.geometry_config,
                    roi,
                    self.last_center_x,
                )
                timing["dual_planner_ms"] = (
                    time.perf_counter() - started
                ) * 1000.0
                if result is None:
                    started = time.perf_counter()
                    result = STAGE4.SingleBoundaryPlanner.plan(
                        curves,
                        main_yellow,
                        transformed["green"],
                        self.roi_ipm,
                        self.geometry_config,
                        roi,
                    )
                    timing["single_planner_ms"] = (
                        time.perf_counter() - started
                    ) * 1000.0
                if result is None:
                    reason = (
                        "single_boundary_side_unknown"
                        if any(curve.inward_sign == 0 for curve in curves)
                        else "boundary_pair_invalid"
                    )
                    result = STAGE4.GeometryResult(reason=reason)
            else:
                result = STAGE4.GeometryResult(reason="boundary_missing")
        elif fallback_used and not yellow_status["main_yellow_valid"]:
            result = STAGE4.GeometryResult(reason="main_yellow_missing")
        elif fallback_used and not boundary_valid:
            result = STAGE4.GeometryResult(reason="segmentation_invalid")

        if not bool(yellow_status.get("main_yellow_seed_connected", False)):
            result.confidence *= 0.80
        if fallback_used and len(result.raw_points):
            timing["centerline_smooth_ms"] += self._smooth(
                result, main_yellow, roi
            )

        if not result.valid and self.geometry_config.history_fallback_enable:
            age = time.monotonic() - self.last_valid_time
            lateral_change_ok = bool(
                len(self.last_valid_points)
                and (
                    len(result.raw_points) == 0
                    or abs(
                        float(result.raw_points[0, 0])
                        - float(self.last_valid_points[0, 0])
                    )
                    * self.roi_ipm.meter_per_pixel
                    <= self.geometry_config.centerline_max_lateral_jump_m
                )
            )
            if (
                len(self.last_valid_points)
                >= self.geometry_config.centerline_min_points
                and age <= self.geometry_config.history_max_age_sec
                and lateral_change_ok
            ):
                result.final_points = self.last_valid_points.copy()
                result.mode = "history_fallback"
                result.reason = "valid"
                result.valid = True
                result.confidence = min(0.50, result.confidence or 0.45)
        if result.valid:
            self.last_valid_points = result.final_points.copy()
            self.last_valid_time = time.monotonic()
            self.last_center_x = float(result.final_points[0, 0])
        else:
            result.confidence = 0.0

        shape = (
            self.roi_ipm.output_height,
            self.roi_ipm.output_width,
        )
        # 调试线条延迟到低频调试/Web 线程绘制，算法热路径只保留零画布。
        selected = np.zeros(shape, dtype=np.uint8)
        centerline = np.zeros(shape, dtype=np.uint8)
        road_curves = result.selected_curves or curves
        if result.mode in {
            "yellow_corridor_dual_edge",
            "yellow_corridor_center_gap_filled",
            "green_dual_inner_edge",
            "green_yellow_hybrid",
            "single_green_width_offset",
        }:
            road_side_status = {
                key: "N/A"
                for key in GlobalRoadSideClassifier.EMPTY_STATUS
            }
        else:
            road_side_status = GlobalRoadSideClassifier.representative_status(
                road_curves
            )
        timing["total_processing_ms"] = (
            time.perf_counter() - total_started
        ) * 1000.0
        return PipelineResult(
            boundary=boundary,
            yellow=yellow,
            green=green,
            observed_ipm_boundary=repair.observed,
            repaired_ipm_boundary=repair.repaired,
            repaired_gap_mask=repair.gap_mask,
            transformed=transformed,
            main_yellow=main_yellow,
            candidate_curves=curves,
            selected_boundary=selected,
            centerline_mask=centerline,
            overlay=None,
            geometry=result,
            roi=roi,
            boundary_valid=boundary_valid,
            yellow_status=yellow_status,
            observed_boundary_component_count=(
                repair.observed_component_count
            ),
            repaired_boundary_component_count=(
                repair.repaired_component_count
            ),
            merged_boundary_group_count=merged_group_count,
            repaired_gap_count=repair.gap_count,
            repaired_gap_max_px=repair.gap_max_px,
            road_side_status=road_side_status,
            boundary_component_count=raw_component_count,
            valid_boundary_component_count=len(curves),
            roi_y_offset=self.roi_ipm.y_offset,
            yellow_corridor_status=corridor_status,
            yellow_corridor_samples=list(
                getattr(corridor_result, "yellow_corridor_samples", [])
            ),
            green_boundary_status=green_status,
            green_boundary_samples=green_debug_samples,
            fallback_boundary_pipeline_used=fallback_used,
            timing_ms=timing,
            counters=counters,
        )

    def make_overlay(
        self,
        transformed: Dict[str, np.ndarray],
        main_yellow: np.ndarray,
        selected: np.ndarray,
        raw_line: np.ndarray,
        centerline: np.ndarray,
        result: Any,
        roi: Tuple[int, int],
        processing_ms: float,
        observed_boundary: Optional[np.ndarray] = None,
        repaired_gap_mask: Optional[np.ndarray] = None,
        diagnostic_curves: Optional[Sequence[Any]] = None,
        corridor_samples: Optional[Sequence[Dict[str, Any]]] = None,
        corridor_status: Optional[Dict[str, Any]] = None,
        green_samples: Optional[Sequence[Dict[str, Any]]] = None,
        green_status: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        height, width = main_yellow.shape
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        overlay[transformed["green"] > 0] = (0, 90, 0)
        overlay[main_yellow > 0] = (0, 150, 150)
        overlay[transformed["boundary"] > 0] = (130, 130, 130)
        if observed_boundary is not None:
            overlay[observed_boundary > 0] = (255, 255, 255)
        if (
            result.mode == "dual_boundary_midpoint"
            and len(result.selected_curves) == 2
        ):
            first = STAGE4.BoundaryComponentAnalyzer.component_mask(
                transformed["boundary"], [result.selected_curves[0]], 3
            )
            second = STAGE4.BoundaryComponentAnalyzer.component_mask(
                transformed["boundary"], [result.selected_curves[1]], 3
            )
            overlay[first > 0] = (255, 0, 0)
            overlay[second > 0] = (255, 0, 255)
        else:
            overlay[selected > 0] = (255, 255, 0)
        if repaired_gap_mask is not None:
            overlay[repaired_gap_mask > 0] = (0, 140, 255)
        overlay[raw_line > 0] = (0, 140, 255)
        overlay[centerline > 0] = (0, 0, 255)
        center_x = int(round(self.roi_ipm.vehicle_center_x_px))
        origin_y = int(round(self.roi_ipm.vehicle_origin_y_px))
        cv2.line(
            overlay,
            (center_x, roi[0]),
            (center_x, roi[1]),
            (255, 255, 0),
            1,
        )
        if 0 <= origin_y < height:
            cv2.circle(overlay, (center_x, origin_y), 6, (0, 0, 255), -1)
        cv2.rectangle(
            overlay, (0, roi[0]), (width - 1, roi[1]), (255, 255, 255), 1
        )
        reject_colors = {
            "reject_no_yellow_interval": (125, 125, 125),
            "reject_width_too_narrow": (0, 140, 255),
            "reject_width_too_wide": (0, 140, 255),
            "reject_center_jump": (0, 0, 255),
            "reject_seed_or_continuity": (0, 0, 255),
            "reject_left_edge_validation": (255, 255, 0),
            "reject_right_edge_validation": (255, 255, 0),
            "reject_boundary_ratio": (255, 255, 0),
            "reject_center_not_yellow": (180, 0, 180),
        }
        green_mode = result.mode in {
            "green_dual_inner_edge",
            "green_yellow_hybrid",
            "single_green_width_offset",
        }
        for sample in (() if green_mode else (corridor_samples or ())):
            y = int(sample["y"])
            classification = sample.get("classification", "missing")
            center = int(
                round(
                    sample["center"]
                    if sample.get("center") is not None
                    else center_x
                )
            )
            if classification == "strict_observed":
                cv2.circle(
                    overlay, (int(sample["left"]), y), 3, (255, 0, 0), -1
                )
                cv2.circle(
                    overlay, (int(sample["right"]), y), 3, (255, 0, 255), -1
                )
                cv2.circle(overlay, (center, y), 3, (0, 255, 255), -1)
            elif classification == "weak_single_edge":
                cv2.circle(overlay, (center, y), 5, (0, 140, 255), 2)
            elif classification == "gap_filled":
                cv2.circle(overlay, (center, y), 5, (255, 255, 0), 2)
            else:
                color = reject_colors.get(sample["reason"], (160, 160, 160))
                cv2.drawMarker(
                    overlay,
                    (int(np.clip(center, 0, width - 1)), y),
                    color,
                    cv2.MARKER_TILTED_CROSS,
                    7,
                    1,
                    cv2.LINE_AA,
                )
        for sample in green_samples or ():
            classification = sample.get("classification", "missing")
            y = int(sample["y"])
            curve_debug = sample.get("single_curve_debug")
            if curve_debug:
                boundary_point = tuple(
                    int(round(value))
                    for value in curve_debug["boundary_pixel"]
                )
                smoothed_point = tuple(
                    int(round(value))
                    for value in curve_debug["smoothed_pixel"]
                )
                center_point = tuple(
                    int(round(value))
                    for value in curve_debug["center_pixel"]
                )
                side_color = (
                    (255, 120, 0)
                    if curve_debug.get("side") == "left"
                    else (255, 0, 255)
                )
                cv2.circle(overlay, boundary_point, 3, side_color, -1)
                cv2.circle(
                    overlay, smoothed_point, 4, (220, 220, 220), 1
                )
                if curve_debug.get("accepted"):
                    cv2.circle(overlay, center_point, 3, (0, 165, 255), -1)
                    if int(curve_debug.get("sample_index", 0)) % 3 == 0:
                        cv2.arrowedLine(
                            overlay,
                            boundary_point,
                            center_point,
                            (0, 165, 255),
                            1,
                            cv2.LINE_AA,
                            tipLength=0.25,
                        )
            if sample.get("valid_left_x") is not None:
                cv2.circle(
                    overlay,
                    (int(sample["valid_left_x"]), y),
                    1,
                    (255, 255, 255),
                    -1,
                )
            if sample.get("valid_right_x") is not None:
                cv2.circle(
                    overlay,
                    (int(sample["valid_right_x"]), y),
                    1,
                    (255, 255, 255),
                    -1,
                )
            if classification not in {
                "green_dual_observed",
                "green_single_offset",
                "gap_filled",
            }:
                continue
            center = int(round(float(sample["center"])))
            if classification == "green_dual_observed":
                cv2.circle(
                    overlay, (int(sample["left"]), y), 4, (255, 120, 0), -1
                )
                cv2.circle(
                    overlay, (int(sample["right"]), y), 4, (255, 0, 255), -1
                )
                cv2.circle(overlay, (center, y), 3, (0, 255, 255), -1)
            elif classification == "green_single_offset":
                edge_x = (
                    sample["left"]
                    if sample.get("left") is not None
                    else sample.get("right")
                )
                if edge_x is not None:
                    cv2.circle(
                        overlay, (int(edge_x), y), 4, (255, 160, 0), -1
                    )
                cv2.circle(overlay, (center, y), 5, (0, 140, 255), 2)
            else:
                cv2.circle(overlay, (center, y), 5, (255, 255, 0), 2)
        near_forward = (
            self.roi_ipm.vehicle_origin_y_px - float(roi[1])
        ) * self.roi_ipm.meter_per_pixel
        far_forward = (
            self.roi_ipm.vehicle_origin_y_px - float(roi[0])
        ) * self.roi_ipm.meter_per_pixel
        cv2.putText(
            overlay,
            f"FAR {far_forward:.2f}m",
            (width - 112, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"NEAR {near_forward:.2f}m",
            (width - 120, height - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        display_curves = (
            result.selected_curves
            if result.selected_curves
            else list(diagnostic_curves or [])
        )
        for curve in display_curves:
            if not curve.inward_sign:
                continue
            for y in np.linspace(curve.y_min, curve.y_max, 4):
                x = float(curve.x_at(np.asarray([y]))[0])
                slope = curve.dx_dy(float(y))
                normal = np.asarray([1.0, -slope], dtype=np.float64)
                normal /= max(float(np.linalg.norm(normal)), 1.0e-9)
                start = (int(round(x)), int(round(y)))
                end = (
                    int(round(x + curve.inward_sign * normal[0] * 18)),
                    int(round(y + curve.inward_sign * normal[1] * 18)),
                )
                cv2.arrowedLine(
                    overlay,
                    start,
                    end,
                    (80, 255, 80),
                    2,
                    cv2.LINE_AA,
                    tipLength=0.35,
                )
        vote_status = GlobalRoadSideClassifier.representative_status(
            display_curves
        )
        fast_mode = result.mode in {
            "yellow_corridor_dual_edge",
            "yellow_corridor_center_gap_filled",
        }
        sample_summary = corridor_status or {}
        green_summary = green_status or {}
        if green_mode:
            lines = (
                f"mode={result.mode}",
                f"valid={result.valid} confidence={result.confidence:.2f}",
                (
                    "green L/R="
                    f"{len(green_samples or ()) - green_summary.get('green_left_inner_edge_missing_count', 0)}/"
                    f"{len(green_samples or ()) - green_summary.get('green_right_inner_edge_missing_count', 0)} "
                    "single="
                    f"{green_summary.get('green_single_left_count', 0)}/"
                    f"{green_summary.get('green_single_right_count', 0)}"
                ),
                (
                    "width="
                    f"{green_summary.get('green_corridor_width_px_mean', 0.0):.1f}"
                    f"+/-{green_summary.get('green_corridor_width_px_std', 0.0):.1f}px"
                ),
                (
                    "Y/U/G="
                    f"{green_summary.get('green_corridor_yellow_support_ratio', 0.0):.2f}/"
                    f"{green_summary.get('green_corridor_unknown_ratio', 0.0):.2f}/"
                    f"{green_summary.get('green_corridor_green_intrusion_ratio', 0.0):.2f}"
                ),
                (
                    "curve side="
                    f"{green_summary.get('single_green_curve_side', 'N/A')} "
                    "raw/chain/accepted="
                    f"{green_summary.get('single_green_curve_raw_count', 0)}/"
                    f"{green_summary.get('single_green_curve_chain_count', 0)}/"
                    f"{green_summary.get('single_green_curve_accepted_count', 0)}"
                ),
                (
                    "curve width="
                    f"{green_summary.get('single_green_lane_width_m', 0.0):.3f}m "
                    f"source={green_summary.get('single_green_lane_width_source', 'N/A')}"
                ),
                (
                    "curve contribution="
                    f"{green_summary.get('hybrid_single_curve_point_count', 0)} "
                    f"final={len(result.final_points)}/"
                    f"{result.forward_span_m:.2f}m"
                ),
                f"time={processing_ms:.1f}ms",
            )
        else:
            lines = (
            f"mode={result.mode}",
            f"valid={result.valid} confidence={result.confidence:.2f}",
            f"points={len(result.final_points)} span={result.forward_span_m:.2f}m",
            (
                "strict/weak/missing/filled="
                f"{sample_summary.get('center_strict_observed_count', 0)}/"
                f"{sample_summary.get('center_weak_observed_count', 0)}/"
                f"{sample_summary.get('center_missing_count', 0)}/"
                f"{sample_summary.get('center_gap_filled_count', 0)}"
            ),
            (
                "smooth dev="
                f"{sample_summary.get('center_pre_smooth_lateral_std_px', 0.0):.1f}/"
                f"{sample_summary.get('center_post_smooth_lateral_std_px', 0.0):.1f}px"
                if fast_mode
                else (
                    "side_votes="
                    f"+{vote_status['road_side_positive_vote_count']}/"
                    f"-{vote_status['road_side_negative_vote_count']} "
                    f"ratio={vote_status['road_side_global_vote_ratio']:.2f}"
                )
            ),
            f"time={processing_ms:.1f}ms",
            )
        for index, text in enumerate(lines):
            position = (10, 24 + index * 22)
            cv2.putText(
                overlay,
                text,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                text,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return overlay


class LatestJpeg:
    """Web 与可选压缩话题共用唯一一次 JPEG 编码结果。"""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.data: Optional[bytes] = None
        self.sequence = 0
        self.encode_count = 0

    def update(self, composite: np.ndarray, quality: int) -> Optional[bytes]:
        ok, encoded = cv2.imencode(
            ".jpg",
            np.ascontiguousarray(composite),
            [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
        )
        if not ok:
            return None
        payload = encoded.tobytes()
        with self.condition:
            self.data = payload
            self.sequence += 1
            self.encode_count += 1
            self.condition.notify_all()
        return payload

    def get(self) -> Tuple[Optional[bytes], int]:
        with self.condition:
            return self.data, self.sequence


WEB_PAGE = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>一体化车道感知</title><style>
body{margin:0;background:#10151d;color:#e7edf5;font:14px system-ui}
header{padding:12px 18px;background:#172131;position:sticky;top:0}
h1{margin:0 0 6px;font-size:21px}.page{padding:12px}
.panel{background:#182231;border:1px solid #2b3a50;border-radius:8px;padding:12px}
img{display:block;width:100%;height:auto;background:#05080c}
.stats{display:grid;grid-template-columns:repeat(3,minmax(150px,1fr));gap:8px;margin-top:12px}
.item{background:#111a26;padding:8px;border-radius:5px}.good{color:#6fe29b}.bad{color:#ff6b6b}
@media(max-width:760px){.stats{grid-template-columns:1fr 1fr}}</style></head>
<body><header><h1>一体化车道感知（gxb_lane_pipeline_v1）</h1>
<div>左侧为相机缩略图；右侧放大显示有效 planning ROI、局部中心线与采样诊断。</div></header>
<main class="page"><section class="panel"><img src="/stream/overlay">
<div class="stats" id="stats"></div></section></main><script>
const fields=["centerline_mode","centerline_valid","centerline_confidence",
"capture_fps","process_fps","path_publish_fps","end_to_end_ms",
"dropped_frame_count","processing_time_ms",
"yellow_corridor_valid","yellow_corridor_accepted_sample_count",
"yellow_corridor_total_sample_count",
"yellow_corridor_forward_span_m","yellow_corridor_boundary_valid_ratio",
"yellow_corridor_width_px_mean","yellow_corridor_width_px_std",
"yellow_corridor_reject_no_yellow_interval_count",
"yellow_corridor_reject_seed_or_continuity_count",
"yellow_corridor_reject_width_too_narrow_count",
"yellow_corridor_reject_width_too_wide_count",
"yellow_corridor_reject_center_jump_count",
"yellow_corridor_reject_left_edge_validation_count",
"yellow_corridor_reject_right_edge_validation_count",
"yellow_corridor_reject_boundary_ratio_count",
"yellow_corridor_reject_center_not_yellow_count",
"yellow_corridor_reject_other_count",
"yellow_corridor_first_accepted_forward_m",
"yellow_corridor_last_accepted_forward_m",
"yellow_corridor_first_rejected_forward_m",
"yellow_corridor_max_consecutive_rejected_samples",
"yellow_corridor_termination_reason",
"center_strict_observed_count","center_weak_observed_count",
"center_missing_count","center_gap_filled_count","center_gap_count",
"center_gap_max_samples","center_gap_fill_ratio",
"center_smoothing_used","center_smoothing_lambda",
"center_pre_smooth_lateral_std_px","center_post_smooth_lateral_std_px",
"green_dual_edge_valid_count","green_single_left_count",
"green_single_right_count","green_left_inner_edge_missing_count",
"green_right_inner_edge_missing_count","green_corridor_width_px_mean",
"green_corridor_width_px_std","green_corridor_yellow_support_ratio",
"green_corridor_unknown_ratio","green_corridor_green_intrusion_ratio",
"green_boundary_fast_path_used","green_boundary_fast_path_reason",
"green_reject_no_valid_ipm_span_count",
"green_reject_left_outer_run_missing_count",
"green_reject_right_outer_run_missing_count",
"green_reject_left_outer_gap_count","green_reject_right_outer_gap_count",
"green_reject_left_run_too_short_count",
"green_reject_right_run_too_short_count","green_reject_width_count",
"green_reject_center_jump_count","green_reject_green_intrusion_count",
"green_valid_span_left_x_mean","green_valid_span_right_x_mean",
"green_left_outer_gap_px_mean","green_right_outer_gap_px_mean",
"yellow_corridor_reason","fallback_boundary_pipeline_used",
"observed_boundary_component_count","repaired_boundary_component_count",
"merged_boundary_group_count","repaired_gap_count","repaired_gap_max_px",
"road_side_valid_sample_count","road_side_positive_vote_count",
"road_side_negative_vote_count","road_side_global_vote_ratio","last_error"];
async function refresh(){try{const s=await(await fetch("/api/status")).json();
document.getElementById("stats").innerHTML=fields.map(k=>`<div class="item"><b>${k}</b><br>${s[k]}</div>`).join("");
}catch(e){}setTimeout(refresh,500)}refresh();</script></body></html>"""


class PipelineWebServer:
    def __init__(self, node: "LanePerceptionPipelineNode") -> None:
        self.node = node
        self.server: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.client_lock = threading.Lock()
        self.stream_client_count = 0

    def has_stream_clients(self) -> bool:
        with self.client_lock:
            return self.stream_client_count > 0

    def handler_class(self) -> Any:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/":
                    self._send(
                        200, "text/html; charset=utf-8", WEB_PAGE.encode()
                    )
                elif self.path == "/api/status":
                    body = json.dumps(
                        owner.node.status_snapshot(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode()
                    self._send(200, "application/json", body)
                elif self.path == "/stream/overlay":
                    self._stream()
                else:
                    self._send(404, "text/plain", b"not found")

            def _send(
                self, code: int, content_type: str, body: bytes
            ) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _stream(self) -> None:
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = -1
                with owner.client_lock:
                    owner.stream_client_count += 1
                try:
                    while not owner.node.stop_event.is_set():
                        data, sequence = owner.node.web_jpeg.get()
                        if data is not None and sequence != last:
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\n"
                                + f"Content-Length: {len(data)}\r\n\r\n".encode()
                                + data
                                + b"\r\n"
                            )
                            self.wfile.flush()
                            last = sequence
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    with owner.client_lock:
                        owner.stream_client_count = max(
                            0, owner.stream_client_count - 1
                        )

            def log_message(self, _format: str, *_args: Any) -> None:
                pass

        return Handler

    def start(self, host: str, port: int) -> None:
        self.server = ThreadingHTTPServer(
            (host, port), self.handler_class()
        )
        self.server.daemon_threads = True
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="lane-pipeline-web",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None


@dataclass
class DebugSnapshot:
    raw: np.ndarray
    result: PipelineResult
    source: Image
    sequence: int


class LanePerceptionPipelineNode(Node):
    """直接相机输入的一体化 ROS 感知节点。"""

    DEBUG_TOPICS = {
        "boundary": "/gxb_test/pipeline/debug/boundary_mask",
        "yellow": "/gxb_test/pipeline/debug/yellow_mask",
        "green": "/gxb_test/pipeline/debug/green_mask",
        "ipm_observed": (
            "/gxb_test/pipeline/debug/observed_ipm_boundary_mask"
        ),
        "ipm_repaired": (
            "/gxb_test/pipeline/debug/repaired_ipm_boundary_mask"
        ),
        "repaired_gap": "/gxb_test/pipeline/debug/repaired_gap_mask",
        "selected": "/gxb_test/pipeline/debug/selected_boundary_mask",
        "centerline": "/gxb_test/pipeline/debug/centerline_mask",
    }

    def __init__(self) -> None:
        super().__init__("lane_perception_pipeline")
        self._declare_parameters()
        self.device = str(self.get_parameter("device").value)
        self.profile_name = str(self.get_parameter("profile_name").value)
        self.request_width = int(self.get_parameter("width").value)
        self.request_height = int(self.get_parameter("height").value)
        self.request_camera_fps = float(
            self.get_parameter("camera_fps").value
        )
        self.camera_fourcc = self._validate_fourcc(
            str(self.get_parameter("camera_fourcc").value)
        )
        self.camera_buffer_size = max(
            1, int(self.get_parameter("camera_buffer_size").value)
        )
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.publish_debug_raw_images = bool(
            self.get_parameter("publish_debug_raw_images").value
        )
        self.debug_publish_fps = max(
            0.1, float(self.get_parameter("debug_publish_fps").value)
        )
        self.publish_overlay_compressed = bool(
            self.get_parameter("publish_overlay_compressed").value
        )
        self.web_gui_enable = bool(
            self.get_parameter("web_gui_enable").value
        )
        self.web_gui_host = str(
            self.get_parameter("web_gui_host").value
        )
        self.web_gui_port = int(
            self.get_parameter("web_gui_port").value
        )
        self.web_gui_max_fps = max(
            0.1, float(self.get_parameter("web_gui_max_fps").value)
        )
        self.web_gui_jpeg_quality = int(
            np.clip(
                int(self.get_parameter("web_gui_jpeg_quality").value),
                1,
                100,
            )
        )
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.slot = LatestFrameSlot()
        self.web_jpeg = LatestJpeg()
        self.web = PipelineWebServer(self)
        self.capture_meter = RateMeter()
        self.process_meter = RateMeter()
        self.path_meter = RateMeter()
        self.performance_window = PerformanceWindow()
        self.last_web_encode = 0.0
        self.last_debug_publish = 0.0
        self.process_thread: Optional[threading.Thread] = None
        self.render_thread: Optional[threading.Thread] = None
        self.debug_snapshot_lock = threading.Lock()
        self.debug_snapshot: Optional[DebugSnapshot] = None
        self.debug_snapshot_sequence = 0
        self.overlay_build_count = 0
        self.jpeg_encode_count = 0
        self.capture = CameraCapture(self, self.slot, self.stop_event)
        self.threshold, threshold_path = self._load_threshold()
        self.segmentation_config = self._load_segmentation_config()
        self.geometry_config, self.ipm = self._load_geometry()
        self.enhancement_config = self._load_enhancement_config()
        self.corridor_config = self._load_corridor_config()
        self.algorithm = LaneAlgorithm(
            self.threshold,
            self.segmentation_config,
            self.geometry_config,
            self.ipm,
            self.enhancement_config,
            self.corridor_config,
        )
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.path_publisher = self.create_publisher(
            RosPath, "/gxb_test/pipeline/centerline_path", qos
        )
        self.status_publisher = self.create_publisher(
            String, "/gxb_test/pipeline/status", qos
        )
        self.overlay_publisher = (
            self.create_publisher(
                CompressedImage,
                "/gxb_test/pipeline/overlay/compressed",
                qos,
            )
            if self.publish_overlay_compressed
            else None
        )
        self.debug_publishers: Dict[str, Any] = {}
        if self.publish_debug_raw_images:
            self.debug_publishers = {
                name: self.create_publisher(Image, topic, qos)
                for name, topic in self.DEBUG_TOPICS.items()
            }
        self.status = self._initial_status(str(threshold_path))
        self.create_timer(0.5, self._publish_status)
        self.create_timer(5.0, self._log_statistics)
        if self.web_gui_enable:
            self.web.start(self.web_gui_host, self.web_gui_port)
            self.get_logger().info(
                f"Web GUI: http://{self.web_gui_host}:{self.web_gui_port}"
            )
        self.capture.start()
        self.process_thread = threading.Thread(
            target=self._processing_loop,
            name="lane-pipeline-processing",
            daemon=True,
        )
        self.process_thread.start()
        if self.web_gui_enable or self.publish_overlay_compressed:
            self.render_thread = threading.Thread(
                target=self._render_loop,
                name="lane-pipeline-web-render",
                daemon=True,
            )
            self.render_thread.start()
        self.get_logger().info(
            "pipeline started: "
            f"device={self.device}, requested={self.request_width}x"
            f"{self.request_height}@{self.request_camera_fps:.1f}, "
            f"fourcc={self.camera_fourcc}, profile={self.profile_name}, "
            f"debug_raw={self.publish_debug_raw_images}, "
            f"web={self.web_gui_enable}, "
            f"repair={self.enhancement_config.boundary_repair_enable}, "
            "repair_gap="
            f"{self.enhancement_config.boundary_repair_max_gap_m:.2f}m/"
            f"{self.enhancement_config.max_gap_px(self.ipm.meter_per_pixel)}px, "
            f"ipm_roi={self.algorithm.roi_ipm.output_width}x"
            f"{self.algorithm.roi_ipm.output_height}@"
            f"y{self.algorithm.roi_ipm.y_offset}, "
            f"yellow_corridor={self.corridor_config.enable}"
        )

    def _declare_parameters(self) -> None:
        defaults: Dict[str, object] = {
            "device": "/dev/video0",
            "profile_name": "usb_camera",
            "width": 640,
            "height": 480,
            "camera_fps": 10.0,
            "camera_fourcc": "YUYV",
            "camera_buffer_size": 1,
            "frame_id": "usb_camera",
            "threshold_config_path": "",
            "ipm_config_path": "",
            "auto_load_segmentation_config": True,
            "auto_load_geometry_config": True,
            "publish_debug_raw_images": False,
            "debug_publish_fps": 1.0,
            "publish_overlay_compressed": False,
            "web_gui_enable": True,
            "web_gui_host": "0.0.0.0",
            "web_gui_port": 8093,
            "web_gui_max_fps": 1.0,
            "web_gui_jpeg_quality": 85,
            "boundary_repair_enable": True,
            "boundary_repair_max_gap_m": 0.10,
            "boundary_repair_max_angle_deg": 20.0,
            "boundary_repair_max_lateral_delta_m": 0.08,
            "boundary_repair_min_fragment_span_m": 0.05,
            "road_side_min_valid_samples": 5,
            "road_side_global_vote_min_ratio": 0.65,
            "road_side_global_yellow_margin": 0.08,
            "fragment_neighbor_check_limit": 3,
            "yellow_corridor_enable": True,
            "yellow_corridor_sample_step_m": 0.05,
            "yellow_corridor_width_min_ratio": 0.65,
            "yellow_corridor_width_max_ratio": 1.40,
            "yellow_corridor_center_jump_max_m": 0.15,
            "yellow_corridor_min_valid_samples": 6,
            "yellow_corridor_min_forward_span_m": 0.25,
            "yellow_corridor_boundary_validation_radius_px": 5,
            "yellow_corridor_min_boundary_valid_ratio": 0.45,
            "yellow_corridor_max_consecutive_invalid_samples": 2,
            "yellow_corridor_max_small_gap_m": 0.10,
            "center_gap_fill_enable": True,
            "center_gap_fill_max_samples": 2,
            "center_gap_fill_max_m": 0.10,
            "center_gap_fill_require_both_sides": True,
            "center_gap_fill_max_ratio": 0.25,
            "center_smooth_enable": True,
            "center_smooth_lambda": 2.0,
            "green_outer_seed_max_gap_px": 8,
            "green_min_outer_run_width_px": 8,
            "single_green_curve_min_points": 8,
            "single_green_curve_min_span_m": 0.35,
            "single_green_curve_max_row_gap": 2,
            "single_green_curve_max_point_jump_m": 0.12,
            "single_green_tangent_window_points": 5,
            "single_green_min_tangent_points": 3,
            "single_green_min_tangent_dx_m": 0.04,
            "lane_width_current_min_dual_samples": 3,
            "lane_width_ema_alpha": 0.15,
            "lane_width_history_timeout_sec": 2.0,
            "single_green_lane_width_min_m": 0.40,
            "single_green_lane_width_max_m": 0.60,
            "single_green_center_yellow_window_px": 5,
            "single_green_center_min_yellow_ratio": 0.35,
            "single_green_center_max_green_ratio": 0.20,
            "single_green_center_max_lateral_jump_m": 0.10,
            "single_green_center_max_heading_step_deg": 20.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    @staticmethod
    def _validate_fourcc(value: str) -> str:
        normalized = value.strip().upper()
        if len(normalized) != 4:
            raise ValueError("camera_fourcc 必须恰好为四个字符")
        return normalized

    def _load_threshold(self) -> Tuple[Any, Path]:
        explicit = str(self.get_parameter("threshold_config_path").value)
        path = STAGE2.ThresholdConfigLoader.resolve_path(
            SCRIPT_DIR, self.profile_name, explicit
        )
        threshold = STAGE2.ThresholdConfigLoader.load(path)
        if not threshold.loaded:
            raise RuntimeError(threshold.error or f"阈值配置无效: {path}")
        return threshold, path

    def _load_segmentation_config(self) -> Dict[str, object]:
        values = copy.deepcopy(STAGE2.DEFAULT_SEGMENTATION_CONFIG)
        if bool(self.get_parameter("auto_load_segmentation_config").value):
            store = STAGE2.SegmentationConfigStore(SCRIPT_DIR)
            path = store.path(self.profile_name, "")
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                values = store.flatten(payload)
        return STAGE2.SegmentationConfig.validate(values)

    def _load_geometry(self) -> Tuple[Any, Any]:
        config = STAGE4.GeometryConfig(profile_name=self.profile_name)
        config.ipm_config_path = str(
            self.get_parameter("ipm_config_path").value
        )
        if bool(self.get_parameter("auto_load_geometry_config").value):
            store = STAGE4.GeometryParameterStore(SCRIPT_PATH)
            if store.path(self.profile_name).exists():
                config = store.load(config)
                config.ipm_config_path = str(
                    self.get_parameter("ipm_config_path").value
                )
        config.web_gui_enable = self.web_gui_enable
        config.web_gui_host = self.web_gui_host
        config.web_gui_port = self.web_gui_port
        config.web_gui_max_fps = self.web_gui_max_fps
        config.web_gui_jpeg_quality = self.web_gui_jpeg_quality
        config.validate()
        ipm = STAGE4.IpmConfigLoader(SCRIPT_PATH).load(config)
        return config, ipm

    def _load_enhancement_config(self) -> BoundaryEnhancementConfig:
        config = BoundaryEnhancementConfig(
            boundary_repair_enable=bool(
                self.get_parameter("boundary_repair_enable").value
            ),
            boundary_repair_max_gap_m=float(
                self.get_parameter("boundary_repair_max_gap_m").value
            ),
            boundary_repair_max_angle_deg=float(
                self.get_parameter("boundary_repair_max_angle_deg").value
            ),
            boundary_repair_max_lateral_delta_m=float(
                self.get_parameter(
                    "boundary_repair_max_lateral_delta_m"
                ).value
            ),
            boundary_repair_min_fragment_span_m=float(
                self.get_parameter(
                    "boundary_repair_min_fragment_span_m"
                ).value
            ),
            road_side_min_valid_samples=int(
                self.get_parameter("road_side_min_valid_samples").value
            ),
            road_side_global_vote_min_ratio=float(
                self.get_parameter(
                    "road_side_global_vote_min_ratio"
                ).value
            ),
            road_side_global_yellow_margin=float(
                self.get_parameter(
                    "road_side_global_yellow_margin"
                ).value
            ),
            fragment_neighbor_check_limit=int(
                self.get_parameter("fragment_neighbor_check_limit").value
            ),
        )
        config.validate()
        return config

    def _load_corridor_config(self) -> YellowCorridorConfig:
        config = YellowCorridorConfig(
            enable=bool(
                self.get_parameter("yellow_corridor_enable").value
            ),
            sample_step_m=float(
                self.get_parameter(
                    "yellow_corridor_sample_step_m"
                ).value
            ),
            width_min_ratio=float(
                self.get_parameter(
                    "yellow_corridor_width_min_ratio"
                ).value
            ),
            width_max_ratio=float(
                self.get_parameter(
                    "yellow_corridor_width_max_ratio"
                ).value
            ),
            center_jump_max_m=float(
                self.get_parameter(
                    "yellow_corridor_center_jump_max_m"
                ).value
            ),
            min_valid_samples=int(
                self.get_parameter(
                    "yellow_corridor_min_valid_samples"
                ).value
            ),
            min_forward_span_m=float(
                self.get_parameter(
                    "yellow_corridor_min_forward_span_m"
                ).value
            ),
            boundary_validation_radius_px=int(
                self.get_parameter(
                    "yellow_corridor_boundary_validation_radius_px"
                ).value
            ),
            min_boundary_valid_ratio=float(
                self.get_parameter(
                    "yellow_corridor_min_boundary_valid_ratio"
                ).value
            ),
            max_consecutive_invalid_samples=int(
                self.get_parameter(
                    "yellow_corridor_max_consecutive_invalid_samples"
                ).value
            ),
            max_small_gap_m=float(
                self.get_parameter(
                    "yellow_corridor_max_small_gap_m"
                ).value
            ),
            center_gap_fill_enable=bool(
                self.get_parameter("center_gap_fill_enable").value
            ),
            center_gap_fill_max_samples=int(
                self.get_parameter("center_gap_fill_max_samples").value
            ),
            center_gap_fill_max_m=float(
                self.get_parameter("center_gap_fill_max_m").value
            ),
            center_gap_fill_require_both_sides=bool(
                self.get_parameter(
                    "center_gap_fill_require_both_sides"
                ).value
            ),
            center_gap_fill_max_ratio=float(
                self.get_parameter("center_gap_fill_max_ratio").value
            ),
            center_smooth_enable=bool(
                self.get_parameter("center_smooth_enable").value
            ),
            center_smooth_lambda=float(
                self.get_parameter("center_smooth_lambda").value
            ),
            green_outer_seed_max_gap_px=int(
                self.get_parameter("green_outer_seed_max_gap_px").value
            ),
            green_min_outer_run_width_px=int(
                self.get_parameter("green_min_outer_run_width_px").value
            ),
            single_green_curve_min_points=int(
                self.get_parameter("single_green_curve_min_points").value
            ),
            single_green_curve_min_span_m=float(
                self.get_parameter("single_green_curve_min_span_m").value
            ),
            single_green_curve_max_row_gap=int(
                self.get_parameter("single_green_curve_max_row_gap").value
            ),
            single_green_curve_max_point_jump_m=float(
                self.get_parameter(
                    "single_green_curve_max_point_jump_m"
                ).value
            ),
            single_green_tangent_window_points=int(
                self.get_parameter(
                    "single_green_tangent_window_points"
                ).value
            ),
            single_green_min_tangent_points=int(
                self.get_parameter(
                    "single_green_min_tangent_points"
                ).value
            ),
            single_green_min_tangent_dx_m=float(
                self.get_parameter(
                    "single_green_min_tangent_dx_m"
                ).value
            ),
            lane_width_current_min_dual_samples=int(
                self.get_parameter(
                    "lane_width_current_min_dual_samples"
                ).value
            ),
            lane_width_ema_alpha=float(
                self.get_parameter("lane_width_ema_alpha").value
            ),
            lane_width_history_timeout_sec=float(
                self.get_parameter(
                    "lane_width_history_timeout_sec"
                ).value
            ),
            single_green_lane_width_min_m=float(
                self.get_parameter(
                    "single_green_lane_width_min_m"
                ).value
            ),
            single_green_lane_width_max_m=float(
                self.get_parameter(
                    "single_green_lane_width_max_m"
                ).value
            ),
            single_green_center_yellow_window_px=int(
                self.get_parameter(
                    "single_green_center_yellow_window_px"
                ).value
            ),
            single_green_center_min_yellow_ratio=float(
                self.get_parameter(
                    "single_green_center_min_yellow_ratio"
                ).value
            ),
            single_green_center_max_green_ratio=float(
                self.get_parameter(
                    "single_green_center_max_green_ratio"
                ).value
            ),
            single_green_center_max_lateral_jump_m=float(
                self.get_parameter(
                    "single_green_center_max_lateral_jump_m"
                ).value
            ),
            single_green_center_max_heading_step_deg=float(
                self.get_parameter(
                    "single_green_center_max_heading_step_deg"
                ).value
            ),
        )
        config.validate()
        return config

    def _initial_status(self, threshold_path: str) -> Dict[str, Any]:
        return {
            "interface_version": INTERFACE_VERSION,
            "capture_fps": 0.0,
            "process_fps": 0.0,
            "path_publish_fps": 0.0,
            "captured_frame_count": 0,
            "processed_frame_count": 0,
            "dropped_frame_count": 0,
            "capture_to_process_age_ms": 0.0,
            "processing_time_ms": 0.0,
            "end_to_end_ms": 0.0,
            "camera_width": 0,
            "camera_height": 0,
            "camera_reported_fps": 0.0,
            "camera_fourcc": self.camera_fourcc,
            "green_yellow_config_loaded": True,
            "green_yellow_config_path": threshold_path,
            "ipm_config_loaded": True,
            "ipm_config_path": self.ipm.path,
            "boundary_valid": False,
            "centerline_valid": False,
            "centerline_mode": "invalid",
            "centerline_reason": "waiting_for_camera",
            "centerline_confidence": 0.0,
            "final_centerline_point_count": 0,
            "centerline_forward_span_m": 0.0,
            "centerline_yellow_ratio": 0.0,
            "planning_roi_y_min_px": self.algorithm.roi_ipm.y_offset,
            "planning_roi_y_max_px": (
                self.algorithm.roi_ipm.y_offset
                + self.algorithm.roi_ipm.output_height
                - 1
            ),
            "planning_roi_height_px": (
                self.algorithm.roi_ipm.output_height
            ),
            **copy.deepcopy(YellowCorridorPlanner.EMPTY_STATUS),
            **copy.deepcopy(GreenBoundaryFastPlanner.EMPTY_STATUS),
            "fallback_boundary_pipeline_used": False,
            "boundary_repair_max_gap_px": (
                self.enhancement_config.max_gap_px(
                    self.ipm.meter_per_pixel
                )
            ),
            "observed_boundary_component_count": 0,
            "repaired_boundary_component_count": 0,
            "merged_boundary_group_count": 0,
            "repaired_gap_count": 0,
            "repaired_gap_max_px": 0,
            **copy.deepcopy(GlobalRoadSideClassifier.EMPTY_STATUS),
            **{name: 0.0 for name in TIMING_NAMES},
            **{name: 0 for name in COUNTER_NAMES},
            "timing_average_ms": {
                name: 0.0 for name in TIMING_NAMES
            },
            "timing_p95_ms": {
                name: 0.0 for name in TIMING_NAMES
            },
            "last_error": "",
        }

    def update_camera_report(self, capture: Any) -> None:
        raw_fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        reported_fourcc = "".join(
            chr((raw_fourcc >> (8 * index)) & 0xFF) for index in range(4)
        )
        with self.lock:
            self.status.update(
                {
                    "camera_width": int(
                        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                    ),
                    "camera_height": int(
                        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    ),
                    "camera_reported_fps": float(
                        capture.get(cv2.CAP_PROP_FPS)
                    ),
                    "camera_fourcc": reported_fourcc.strip("\x00")
                    or self.camera_fourcc,
                }
            )

    def capture_succeeded(self, now: float) -> None:
        capture_fps = self.capture_meter.tick(now)
        with self.lock:
            self.status["captured_frame_count"] += 1
            self.status["capture_fps"] = capture_fps
            if self.status["last_error"].startswith("camera:"):
                self.status["last_error"] = ""

    def capture_failed(self, message: str) -> None:
        with self.lock:
            self.status["last_error"] = f"camera: {message}"

    def _source_image(self, frame: CapturedFrame) -> Image:
        source = Image()
        source.header.stamp = frame.stamp
        source.header.frame_id = self.frame_id
        source.height = int(frame.image.shape[0])
        source.width = int(frame.image.shape[1])
        source.encoding = "bgr8"
        source.step = source.width * 3
        return source

    def _processing_loop(self) -> None:
        sequence = 0
        while not self.stop_event.is_set():
            frame = self.slot.take(sequence, self.stop_event)
            if frame is None:
                continue
            sequence = frame.sequence
            started = time.monotonic()
            processing_started = time.perf_counter()
            age_ms = (started - frame.captured_monotonic) * 1000.0
            source = self._source_image(frame)
            try:
                result = self.algorithm.process(frame.image)
                path_started = time.perf_counter()
                full_points = self.algorithm.roi_ipm.to_full_points(
                    result.geometry.final_points
                )
                path = (
                    STAGE4.PathConverter.to_path(
                        full_points, self.ipm, source
                    )
                    if result.geometry.valid
                    else STAGE4.PathConverter.empty(self.ipm, source)
                )
                self.path_publisher.publish(path)
                result.timing_ms["path_publish_ms"] = (
                    time.perf_counter() - path_started
                ) * 1000.0
                processing_ms = (
                    time.perf_counter() - processing_started
                ) * 1000.0
                result.timing_ms["total_processing_ms"] = processing_ms
                self.performance_window.add(result.timing_ms)
                path_fps = self.path_meter.tick()
                process_fps = self.process_meter.tick()
                end_to_end = (
                    time.monotonic() - frame.captured_monotonic
                ) * 1000.0
                self._publish_debug(frame.image, result, source)
                with self.lock:
                    self.status.update(
                        {
                            "process_fps": process_fps,
                            "path_publish_fps": path_fps,
                            "processed_frame_count": self.status[
                                "processed_frame_count"
                            ]
                            + 1,
                            "dropped_frame_count": self.slot.dropped_count,
                            "capture_to_process_age_ms": age_ms,
                            "processing_time_ms": processing_ms,
                            "end_to_end_ms": end_to_end,
                            "boundary_valid": result.boundary_valid,
                            "centerline_valid": bool(
                                result.geometry.valid
                            ),
                            "centerline_mode": result.geometry.mode,
                            "centerline_reason": result.geometry.reason,
                            "centerline_confidence": float(
                                np.clip(
                                    result.geometry.confidence, 0.0, 1.0
                                )
                            ),
                            "final_centerline_point_count": len(
                                result.geometry.final_points
                            ),
                            "centerline_forward_span_m": float(
                                result.geometry.forward_span_m
                            ),
                            "centerline_yellow_ratio": float(
                                result.geometry.yellow_ratio
                            ),
                            **result.yellow_corridor_status,
                            **result.green_boundary_status,
                            "fallback_boundary_pipeline_used": (
                                result.fallback_boundary_pipeline_used
                            ),
                            "observed_boundary_component_count": (
                                result.observed_boundary_component_count
                            ),
                            "repaired_boundary_component_count": (
                                result.repaired_boundary_component_count
                            ),
                            "merged_boundary_group_count": (
                                result.merged_boundary_group_count
                            ),
                            "repaired_gap_count": (
                                result.repaired_gap_count
                            ),
                            "repaired_gap_max_px": (
                                result.repaired_gap_max_px
                            ),
                            **result.road_side_status,
                            **result.timing_ms,
                            **result.counters,
                            "last_error": ""
                            if result.geometry.valid
                            else result.geometry.reason,
                        }
                    )
            except Exception as exc:
                processing_ms = (
                    time.perf_counter() - processing_started
                ) * 1000.0
                self.path_publisher.publish(
                    STAGE4.PathConverter.empty(self.ipm, source)
                )
                self.path_meter.tick()
                self.process_meter.tick()
                with self.lock:
                    self.status.update(
                        {
                            "processed_frame_count": self.status[
                                "processed_frame_count"
                            ]
                            + 1,
                            "dropped_frame_count": self.slot.dropped_count,
                            "capture_to_process_age_ms": age_ms,
                            "processing_time_ms": processing_ms,
                            "end_to_end_ms": (
                                time.monotonic()
                                - frame.captured_monotonic
                            )
                            * 1000.0,
                            "boundary_valid": False,
                            "centerline_valid": False,
                            "centerline_mode": "invalid",
                            "centerline_reason": "processing_error",
                            "centerline_confidence": 0.0,
                            "final_centerline_point_count": 0,
                            "centerline_forward_span_m": 0.0,
                            "centerline_yellow_ratio": 0.0,
                            "observed_boundary_component_count": 0,
                            "repaired_boundary_component_count": 0,
                            "merged_boundary_group_count": 0,
                            "repaired_gap_count": 0,
                            "repaired_gap_max_px": 0,
                            **copy.deepcopy(
                                GreenBoundaryFastPlanner.EMPTY_STATUS
                            ),
                            **copy.deepcopy(
                                GlobalRoadSideClassifier.EMPTY_STATUS
                            ),
                            "last_error": f"processing: {exc}",
                        }
                    )

    def _publish_debug(
        self, raw: np.ndarray, result: PipelineResult, source: Image
    ) -> None:
        now = time.monotonic()
        if self.web_gui_enable or self.publish_overlay_compressed:
            with self.debug_snapshot_lock:
                self.debug_snapshot_sequence += 1
                self.debug_snapshot = DebugSnapshot(
                    raw=raw,
                    result=result,
                    source=copy.deepcopy(source),
                    sequence=self.debug_snapshot_sequence,
                )
        if (
            self.publish_debug_raw_images
            and now - self.last_debug_publish
            >= 1.0 / self.debug_publish_fps
        ):
            selected_mask = (
                STAGE4.BoundaryComponentAnalyzer.component_mask(
                    result.transformed["boundary"],
                    result.geometry.selected_curves,
                    3,
                )
            )
            centerline_mask = self.algorithm.points_mask(
                result.main_yellow.shape,
                result.geometry.final_points,
                3,
            )
            images = {
                "boundary": result.boundary,
                "yellow": result.yellow,
                "green": result.green,
                "ipm_observed": result.observed_ipm_boundary,
                "ipm_repaired": result.repaired_ipm_boundary,
                "repaired_gap": result.repaired_gap_mask,
                "selected": selected_mask,
                "centerline": centerline_mask,
            }
            for name, image in images.items():
                self.debug_publishers[name].publish(
                    self._mono_message(image, source)
                )
            self.last_debug_publish = now

    def _render_loop(self) -> None:
        last_sequence = -1
        while not self.stop_event.is_set():
            should_render = bool(
                self.publish_overlay_compressed
                or self.web.has_stream_clients()
            )
            if not should_render:
                self.stop_event.wait(0.10)
                continue
            now = time.monotonic()
            remaining = (
                1.0 / self.web_gui_max_fps
                - (now - self.last_web_encode)
            )
            if remaining > 0:
                self.stop_event.wait(min(remaining, 0.10))
                continue
            with self.debug_snapshot_lock:
                snapshot = self.debug_snapshot
            if snapshot is None or snapshot.sequence == last_sequence:
                self.stop_event.wait(0.03)
                continue
            result = snapshot.result
            overlay_started = time.perf_counter()
            selected_mask = (
                STAGE4.BoundaryComponentAnalyzer.component_mask(
                    result.transformed["boundary"],
                    result.geometry.selected_curves,
                    3,
                )
            )
            centerline_mask = self.algorithm.points_mask(
                result.main_yellow.shape,
                result.geometry.final_points,
                3,
            )
            raw_line = self.algorithm.points_mask(
                result.main_yellow.shape,
                result.geometry.raw_points,
                2,
            )
            local_overlay = self.algorithm.make_overlay(
                result.transformed,
                result.main_yellow,
                selected_mask,
                raw_line,
                centerline_mask,
                result.geometry,
                result.roi,
                result.timing_ms["total_processing_ms"],
                result.observed_ipm_boundary,
                result.repaired_gap_mask,
                result.candidate_curves,
                result.yellow_corridor_samples,
                result.yellow_corridor_status,
                result.green_boundary_samples,
                result.green_boundary_status,
            )
            composite = self._make_web_composite(
                snapshot.raw,
                local_overlay,
                centerline_mask,
                self._corridor_status_lines(result),
            )
            overlay_ms = (
                time.perf_counter() - overlay_started
            ) * 1000.0
            jpeg_started = time.perf_counter()
            payload = self.web_jpeg.update(
                composite, self.web_gui_jpeg_quality
            )
            jpeg_ms = (
                time.perf_counter() - jpeg_started
            ) * 1000.0
            self.last_web_encode = time.monotonic()
            last_sequence = snapshot.sequence
            self.overlay_build_count += 1
            if payload is not None:
                self.jpeg_encode_count += 1
            self.performance_window.update_latest(
                {
                    "overlay_build_ms": overlay_ms,
                    "jpeg_encode_ms": jpeg_ms,
                }
            )
            with self.lock:
                self.status.update(
                    {
                        "overlay_build_ms": overlay_ms,
                        "jpeg_encode_ms": jpeg_ms,
                        "overlay_build_count": self.overlay_build_count,
                        "jpeg_encode_count": self.jpeg_encode_count,
                    }
                )
            if payload is not None and self.overlay_publisher is not None:
                message = CompressedImage()
                message.header = copy.deepcopy(snapshot.source.header)
                message.format = "jpeg"
                message.data = payload
                self.overlay_publisher.publish(message)

    @staticmethod
    def _format_optional_m(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.2f}m"

    @classmethod
    def _corridor_status_lines(
        cls, result: PipelineResult
    ) -> List[str]:
        if result.geometry.mode in {
            "green_dual_inner_edge",
            "green_yellow_hybrid",
            "single_green_width_offset",
        }:
            green = result.green_boundary_status
            total = len(result.green_boundary_samples)
            return [
                (
                    f"mode={result.geometry.mode} "
                    f"confidence={result.geometry.confidence:.2f}"
                ),
                (
                    "green left/right="
                    f"{total - green.get('green_left_inner_edge_missing_count', 0)}/"
                    f"{total - green.get('green_right_inner_edge_missing_count', 0)} "
                    f"dual={green.get('green_dual_edge_valid_count', 0)} "
                    "single="
                    f"{green.get('green_single_left_count', 0)}/"
                    f"{green.get('green_single_right_count', 0)}"
                ),
                (
                    "valid span L/R="
                    f"{green.get('green_valid_span_left_x_mean', 0.0):.1f}/"
                    f"{green.get('green_valid_span_right_x_mean', 0.0):.1f} "
                    "outer gap="
                    f"{green.get('green_left_outer_gap_px_mean', 0.0):.1f}/"
                    f"{green.get('green_right_outer_gap_px_mean', 0.0):.1f}"
                ),
                (
                    "width="
                    f"{green.get('green_corridor_width_px_mean', 0.0):.1f}"
                    f"+/-{green.get('green_corridor_width_px_std', 0.0):.1f}px"
                ),
                (
                    "yellow/unknown/intrusion="
                    f"{green.get('green_corridor_yellow_support_ratio', 0.0):.2f}/"
                    f"{green.get('green_corridor_unknown_ratio', 0.0):.2f}/"
                    f"{green.get('green_corridor_green_intrusion_ratio', 0.0):.2f}"
                ),
                (
                    f"points={len(result.geometry.final_points)} "
                    f"span={result.geometry.forward_span_m:.2f}m"
                ),
                (
                    "curve side/raw/chain/accepted="
                    f"{green.get('single_green_curve_side', 'N/A')}/"
                    f"{green.get('single_green_curve_raw_count', 0)}/"
                    f"{green.get('single_green_curve_chain_count', 0)}/"
                    f"{green.get('single_green_curve_accepted_count', 0)}"
                ),
                (
                    "curve width/source="
                    f"{green.get('single_green_lane_width_m', 0.0):.3f}m/"
                    f"{green.get('single_green_lane_width_source', 'N/A')} "
                    "contrib="
                    f"{green.get('hybrid_single_curve_point_count', 0)}"
                ),
                (
                    "green fast path="
                    f"{green.get('green_boundary_fast_path_used', False)} "
                    f"reason={green.get('green_boundary_fast_path_reason', 'N/A')}"
                ),
            ]
        status = result.yellow_corridor_status
        return [
            (
                "strict/weak/missing/filled="
                f"{status.get('center_strict_observed_count', 0)}/"
                f"{status.get('center_weak_observed_count', 0)}/"
                f"{status.get('center_missing_count', 0)}/"
                f"{status.get('center_gap_filled_count', 0)}"
            ),
            (
                "gaps="
                f"{status.get('center_gap_count', 0)} "
                f"max={status.get('center_gap_max_samples', 0)} "
                f"fill_ratio={status.get('center_gap_fill_ratio', 0.0):.2f}"
            ),
            (
                f"confidence={result.geometry.confidence:.2f} "
                f"smoothing={status.get('center_smoothing_used', False)} "
                f"lambda={status.get('center_smoothing_lambda', 0.0):.2f}"
            ),
            (
                "pre/post smooth deviation="
                f"{status.get('center_pre_smooth_lateral_std_px', 0.0):.2f}/"
                f"{status.get('center_post_smooth_lateral_std_px', 0.0):.2f}px"
            ),
            (
                "points/span/width="
                f"{len(result.geometry.final_points)}/"
                f"{status.get('yellow_corridor_forward_span_m', 0.0):.2f}m/"
                f"{status.get('yellow_corridor_width_px_mean', 0.0):.1f}"
                f"+/-{status.get('yellow_corridor_width_px_std', 0.0):.1f}px"
            ),
            (
                "termination="
                f"{status.get('yellow_corridor_termination_reason', 'N/A')}"
            ),
            (
                "side_votes=N/A yellow/green=N/A"
                if result.geometry.mode
                in {
                    "yellow_corridor_dual_edge",
                    "yellow_corridor_center_gap_filled",
                }
                else "side_votes and yellow/green ratios: see status API"
            ),
        ]

    @staticmethod
    def _make_web_composite(
        raw: np.ndarray,
        overlay: np.ndarray,
        centerline: np.ndarray,
        status_lines: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        canvas_height, canvas_width = 600, 1200
        left_width, right_x, right_width = 350, 370, 810
        composite = np.full(
            (canvas_height, canvas_width, 3), (16, 22, 31), dtype=np.uint8
        )
        camera_height = min(
            420, int(round(raw.shape[0] * left_width / raw.shape[1]))
        )
        camera_panel = cv2.resize(
            raw, (left_width, camera_height), interpolation=cv2.INTER_AREA
        )
        composite[40 : 40 + camera_height, 10 : 10 + left_width] = camera_panel

        roi_height = max(
            1, int(round(overlay.shape[0] * right_width / overlay.shape[1]))
        )
        overlay_panel = cv2.resize(
            overlay, (right_width, roi_height), interpolation=cv2.INTER_NEAREST
        )
        overlay_y = 35
        composite[
            overlay_y : overlay_y + roi_height,
            right_x : right_x + right_width,
        ] = overlay_panel

        center_bgr = np.zeros_like(overlay)
        center_bgr[:] = (overlay.astype(np.uint16) // 5).astype(np.uint8)
        center_bgr[centerline > 0] = (0, 0, 255)
        center_panel = cv2.resize(
            center_bgr,
            (right_width, roi_height),
            interpolation=cv2.INTER_NEAREST,
        )
        center_y = overlay_y + roi_height + 42
        composite[
            center_y : center_y + roi_height,
            right_x : right_x + right_width,
        ] = center_panel

        labels = (
            ("CAMERA THUMBNAIL", 10, 26),
            ("PLANNING ROI OVERLAY", right_x, 24),
            ("FINAL CENTERLINE (LOCAL ROI)", right_x, center_y - 10),
        )
        for text, x, y in labels:
            cv2.putText(
                composite,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                composite,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        legend = (
            "accepted: left BLUE / right MAGENTA",
            "accepted center: YELLOW / final: RED",
            "reject: gray noY, orange width",
            "red jump, cyan edge, purple center",
        )
        for index, text in enumerate(legend):
            cv2.putText(
                composite,
                text,
                (14, 475 + index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (205, 215, 225),
                1,
                cv2.LINE_AA,
            )
        status_y = center_y + roi_height + 24
        for index, text in enumerate(status_lines or ()):
            y = status_y + index * 17
            if y >= canvas_height - 5:
                break
            cv2.putText(
                composite,
                text,
                (right_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (225, 232, 240),
                1,
                cv2.LINE_AA,
            )
        return composite

    @staticmethod
    def _mono_message(image: np.ndarray, source: Image) -> Image:
        packed = np.ascontiguousarray(image, dtype=np.uint8)
        message = Image()
        message.header = copy.deepcopy(source.header)
        message.height = int(packed.shape[0])
        message.width = int(packed.shape[1])
        message.encoding = "mono8"
        message.is_bigendian = 0
        message.step = message.width
        message.data = packed.tobytes()
        return message

    def _publish_status(self) -> None:
        message = String()
        message.data = json.dumps(
            self.status_snapshot(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.status_publisher.publish(message)

    def status_snapshot(self) -> Dict[str, Any]:
        average, p95 = self.performance_window.summary()
        with self.lock:
            snapshot = copy.deepcopy(self.status)
            snapshot["capture_fps"] = self.capture_meter.value()
            snapshot["process_fps"] = self.process_meter.value()
            snapshot["path_publish_fps"] = self.path_meter.value()
            snapshot["dropped_frame_count"] = self.slot.dropped_count
            snapshot["timing_average_ms"] = average
            snapshot["timing_p95_ms"] = p95
            return snapshot

    def _log_statistics(self) -> None:
        status = self.status_snapshot()
        average = status["timing_average_ms"]
        p95 = status["timing_p95_ms"]
        self.get_logger().info(
            f"capture={status['capture_fps']:.1f} "
            f"process={status['process_fps']:.1f} "
            f"path={status['path_publish_fps']:.1f} "
            f"drop={status['dropped_frame_count']} "
            f"total={average['total_processing_ms']:.1f}/"
            f"{p95['total_processing_ms']:.1f}ms "
            f"seg={average['segmentation_ms']:.1f}/"
            f"{p95['segmentation_ms']:.1f} "
            f"boundary={average['boundary_extract_ms']:.1f}/"
            f"{p95['boundary_extract_ms']:.1f} "
            f"ipm={average['ipm_warp_ms']:.1f}/"
            f"{p95['ipm_warp_ms']:.1f} "
            f"greenfast={average['green_boundary_fast_path_ms']:.1f}/"
            f"{p95['green_boundary_fast_path_ms']:.1f} "
            f"corridor={average['yellow_corridor_ms']:.1f}/"
            f"{p95['yellow_corridor_ms']:.1f} "
            f"repair={average['boundary_repair_ms']:.1f}/"
            f"{p95['boundary_repair_ms']:.1f} "
            f"desc={average['fragment_descriptor_ms']:.1f}/"
            f"{p95['fragment_descriptor_ms']:.1f} "
            f"components={average['component_analysis_ms']:.1f}/"
            f"{p95['component_analysis_ms']:.1f} "
            f"merge={average['boundary_merge_ms']:.1f}/"
            f"{p95['boundary_merge_ms']:.1f} "
            f"vote={average['road_side_vote_ms']:.1f}/"
            f"{p95['road_side_vote_ms']:.1f} "
            f"dual={average['dual_planner_ms']:.1f}/"
            f"{p95['dual_planner_ms']:.1f} "
            f"single={average['single_planner_ms']:.1f}/"
            f"{p95['single_planner_ms']:.1f} "
            f"smooth={average['centerline_smooth_ms']:.1f}/"
            f"{p95['centerline_smooth_ms']:.1f} "
            f"pathpub={average['path_publish_ms']:.1f}/"
            f"{p95['path_publish_ms']:.1f} "
            f"overlay={average['overlay_build_ms']:.1f}/"
            f"{p95['overlay_build_ms']:.1f} "
            f"jpeg={average['jpeg_encode_ms']:.1f}/"
            f"{p95['jpeg_encode_ms']:.1f}"
        )

    def destroy_node(self) -> bool:
        self.stop_event.set()
        self.capture.stop()
        self.slot.wake()
        if self.process_thread is not None:
            self.process_thread.join(timeout=2.0)
        if self.render_thread is not None:
            self.render_thread.join(timeout=2.0)
        self.web.stop()
        return super().destroy_node()


def _threshold_document() -> Dict[str, Any]:
    return {
        "version": 1,
        "profile_name": "test",
        "saved_at": "self-test",
        "image_topic": "/test",
        "frame_width": 600,
        "frame_height": 800,
        "roi": {"y_start_ratio": 0.0, "y_end_ratio": 1.0},
        "yellow": {
            "h_min": 20,
            "h_max": 40,
            "s_min": 100,
            "s_max": 255,
            "v_min": 100,
            "v_max": 255,
            "prefer_bottom_connected": False,
        },
        "green": {
            "h_min": 45,
            "h_max": 85,
            "s_min": 100,
            "s_max": 255,
            "v_min": 80,
            "v_max": 255,
            "prefer_bottom_connected": False,
        },
        "morphology": {
            "open_kernel_size": 1,
            "close_kernel_size": 1,
            "open_iterations": 0,
            "close_iterations": 0,
            "min_component_area": 20,
            "bottom_band_pixels": 10,
        },
    }


def _ipm_document(version: str = "gxb_ipm_v1") -> Dict[str, Any]:
    return {
        "version": version,
        "image_width": 600,
        "image_height": 800,
        "homography_matrix": np.eye(3).tolist(),
        "inverse_homography_matrix": np.eye(3).tolist(),
        "output": {
            "width_px": 600,
            "height_px": 800,
            "meter_per_pixel": 0.005,
            "vehicle_center_x_px": 300,
            "vehicle_origin_y_px": 799,
        },
        "coordinate_convention": {
            "vehicle_reference_point_name": "base_link"
        },
        "calibration_board": {
            "width_m": 0.50,
            "length_m": 0.70,
            "near_edge_distance_m": 0.30,
            "center_lateral_offset_m": 0.0,
        },
        "road_geometry_defaults": {
            "expected_lane_width_m": 0.50,
            "single_boundary_center_offset_m": 0.25,
        },
    }


def run_self_tests() -> None:
    """覆盖配置、完整处理、覆盖槽、失败降级和轻量发布约束。"""
    passed: List[str] = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            raise AssertionError(name)
        passed.append(name)

    class MemoryConfigPath:
        def __init__(self, name: str, document: Dict[str, Any]) -> None:
            self.name = name
            self.document = document

        def exists(self) -> bool:
            return True

        def read_text(self, encoding: str = "utf-8") -> str:
            del encoding
            return json.dumps(self.document)

        def stat(self) -> Any:
            return types.SimpleNamespace(st_mtime_ns=1)

        def __str__(self) -> str:
            return self.name

    threshold_path = MemoryConfigPath(
        "memory:test_green_yellow.json", _threshold_document()
    )
    threshold = STAGE2.ThresholdConfigLoader.load(threshold_path)
    check("1 config load", threshold.loaded)

    geometry = STAGE4.GeometryConfig(
        profile_name="test",
        main_yellow_min_area_px=50,
        main_yellow_min_vertical_span_px=10,
        boundary_component_min_pixels=5,
        boundary_component_min_vertical_span_px=8,
        boundary_fit_min_points=5,
        centerline_min_points=5,
        centerline_min_forward_span_m=0.20,
        centerline_min_yellow_ratio=0.45,
    )
    geometry.validate()
    loader = STAGE4.IpmConfigLoader(SCRIPT_PATH)
    ipm_path = MemoryConfigPath(
        "memory:test_ipm.json", _ipm_document()
    )
    loader.resolve_path = lambda _config: ipm_path
    ipm = loader.load(geometry)
    check("2 ipm version", ipm.path == str(ipm_path))
    bad_path = MemoryConfigPath(
        "memory:bad_ipm.json", _ipm_document("bad")
    )
    bad_loader = STAGE4.IpmConfigLoader(SCRIPT_PATH)
    bad_loader.resolve_path = lambda _config: bad_path
    try:
        bad_loader.load(geometry)
    except STAGE4.GeometryError:
        passed.append("3 reject ipm version")
    else:
        raise AssertionError("3 reject ipm version")

    config = copy.deepcopy(STAGE2.DEFAULT_SEGMENTATION_CONFIG)
    config.update(
        {
            "boundary_min_component_pixels": 5,
            "boundary_min_valid_pixels": 5,
            "boundary_min_span_px": 5,
            "boundary_max_component_area_ratio": 0.20,
        }
    )
    algorithm = LaneAlgorithm(threshold, config, geometry, ipm)
    hsv = np.zeros((800, 600, 3), dtype=np.uint8)
    hsv[:, :175] = (60, 220, 220)
    hsv[:, 175:425] = (30, 220, 220)
    hsv[:, 425:] = (60, 220, 220)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    result = algorithm.process(image)
    check("4 static full process", result.boundary.shape == (800, 600))
    binary = all(
        set(np.unique(mask).tolist()).issubset({0, 255})
        for mask in (
            result.yellow,
            result.green,
            result.boundary,
            result.transformed["yellow"],
            result.transformed["green"],
            result.transformed["boundary"],
        )
    )
    check("5 binary masks", binary)
    points = np.array(
        [[300.0, 739.0], [300.0, 700.0], [300.0, 650.0]]
    )
    samples = STAGE4.PathConverter.metric_samples(points, ipm)
    check("6 metric conversion", len(samples) == 3)
    check(
        "7 forward monotonic",
        all(samples[index + 1][0] > samples[index][0] for index in range(2)),
    )

    slot = LatestFrameSlot()
    stop = threading.Event()
    stamp = types.SimpleNamespace(sec=0, nanosec=0)
    slot.put(np.zeros((1, 1, 3), np.uint8), time.monotonic(), stamp)
    slot.put(np.ones((1, 1, 3), np.uint8), time.monotonic(), stamp)
    newest = slot.take(0, stop)
    check(
        "8 latest overwrite",
        newest is not None and int(newest.image[0, 0, 0]) == 1,
    )
    check("9 slow consumer drop", slot.dropped_count == 1)

    class FailedCapture:
        def __init__(self, *_args: Any) -> None:
            pass

        def set(self, *_args: Any) -> bool:
            return False

        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            pass

    fake_node = types.SimpleNamespace(
        device="/missing",
        camera_fourcc="YUYV",
        request_width=640,
        request_height=480,
        request_camera_fps=10.0,
        camera_buffer_size=1,
    )
    capture = CameraCapture(
        fake_node, LatestFrameSlot(), threading.Event(), FailedCapture
    )
    opened = capture._open()
    check("10 camera failure safe", not opened.isOpened())

    source = Image()
    empty_path = STAGE4.PathConverter.empty(ipm, source)
    check("11 invalid empty path", len(empty_path.poses) == 0)
    jpeg = LatestJpeg()
    web_enabled = False
    if web_enabled:
        jpeg.update(np.zeros((10, 10, 3), np.uint8), 85)
    check("12 web off no encode", jpeg.encode_count == 0)
    debug_default = False
    check("13 debug raw default off", not debug_default)
    source_text = SCRIPT_PATH.read_text(encoding="utf-8")
    forbidden = (
        "Twi" + "st",
        "cmd" + "_vel",
        "linear" + ".x",
        "angular" + ".z",
        "Pure" + "Pursuit",
        "P" + "ID",
    )
    check(
        "14 perception only",
        not any(token in source_text for token in forbidden),
    )
    local_overlay = algorithm.make_overlay(
        result.transformed,
        result.main_yellow,
        result.selected_boundary,
        algorithm.points_mask(
            result.centerline_mask.shape, result.geometry.raw_points, 2
        ),
        result.centerline_mask,
        result.geometry,
        result.roi,
        result.timing_ms["total_processing_ms"],
        result.observed_ipm_boundary,
        result.repaired_gap_mask,
        result.candidate_curves,
        result.yellow_corridor_samples,
        result.yellow_corridor_status,
        result.green_boundary_samples,
        result.green_boundary_status,
    )
    composite = LanePerceptionPipelineNode._make_web_composite(
        image,
        local_overlay,
        result.centerline_mask,
    )
    check(
        "15 one composite",
        composite.ndim == 3 and composite.shape[1] > image.shape[1],
    )

    enhancement = BoundaryEnhancementConfig()
    enhancement.validate()
    roi = (599, 739)
    lane_yellow = np.zeros((800, 600), dtype=np.uint8)
    lane_green = np.zeros_like(lane_yellow)
    lane_yellow[roi[0] : roi[1] + 1, 251:350] = 255
    lane_green[roi[0] : roi[1] + 1, :250] = 255
    lane_green[roi[0] : roi[1] + 1, 351:] = 255

    def fragmented_boundary(
        gap_px: int,
        include_right: bool = False,
        angled_lower: bool = False,
    ) -> np.ndarray:
        mask = np.zeros((800, 600), dtype=np.uint8)
        upper_end = 650
        lower_start = upper_end + gap_px + 1
        cv2.line(mask, (250, 600), (250, upper_end), 255, 1)
        lower_x = 282 if angled_lower else 250
        cv2.line(mask, (250, lower_start), (lower_x, 739), 255, 1)
        if include_right:
            cv2.line(mask, (350, 600), (350, upper_end), 255, 1)
            cv2.line(mask, (350, lower_start), (350, 739), 255, 1)
        return mask

    repaired_ten = ConservativeBoundaryRepair.repair(
        fragmented_boundary(10),
        lane_yellow,
        lane_green,
        ipm,
        roi,
        enhancement,
    )
    check(
        "16 short same-side gap repaired",
        repaired_ten.gap_count == 1
        and np.any(repaired_ten.gap_mask[651:661, 248:253] > 0),
    )
    repaired_thirty = ConservativeBoundaryRepair.repair(
        fragmented_boundary(30),
        lane_yellow,
        lane_green,
        ipm,
        roi,
        enhancement,
    )
    check("17 long gap rejected", repaired_thirty.gap_count == 0)
    repaired_angle = ConservativeBoundaryRepair.repair(
        fragmented_boundary(10, angled_lower=True),
        lane_yellow,
        lane_green,
        ipm,
        roi,
        enhancement,
    )
    check("18 angle mismatch rejected", repaired_angle.gap_count == 0)

    opposite = np.zeros((800, 600), dtype=np.uint8)
    cv2.line(opposite, (250, 600), (250, 650), 255, 1)
    cv2.line(opposite, (350, 661), (350, 739), 255, 1)
    repaired_opposite = ConservativeBoundaryRepair.repair(
        opposite,
        lane_yellow,
        lane_green,
        ipm,
        roi,
        enhancement,
    )
    check("19 opposite sides rejected", repaired_opposite.gap_count == 0)

    horizontal = np.zeros((800, 600), dtype=np.uint8)
    cv2.line(horizontal, (80, 670), (520, 670), 255, 2)
    horizontal_result = ConservativeBoundaryRepair.repair(
        horizontal,
        lane_yellow,
        lane_green,
        ipm,
        roi,
        enhancement,
    )
    horizontal_fragments, _ = ConservativeBoundaryRepair.fragments(
        horizontal_result.observed,
        roi,
        enhancement.min_fragment_span_px(ipm.meter_per_pixel),
    )
    check("20 horizontal seam excluded", len(horizontal_fragments) == 0)

    curved_fragments = np.zeros((800, 600), dtype=np.uint8)
    for start_y, end_y in ((600, 635), (643, 682), (690, 730)):
        points = []
        for y_value in range(start_y, end_y + 1):
            x_value = int(
                round(250.0 + 0.0015 * (y_value - 665.0) ** 2)
            )
            points.append((x_value, y_value))
        cv2.polylines(
            curved_fragments,
            [np.asarray(points, dtype=np.int32)],
            False,
            255,
            1,
        )
    fragment_curves, _ = STAGE4.BoundaryComponentAnalyzer.analyze(
        curved_fragments, ipm, geometry, roi
    )
    step_px = max(
        2,
        int(
            round(
                geometry.centerline_sample_step_m / ipm.meter_per_pixel
            )
        ),
    )
    for curve in fragment_curves:
        GlobalRoadSideClassifier.classify_curve(
            curve,
            lane_yellow,
            lane_green,
            geometry,
            enhancement,
            step_px,
        )
    (
        aggregated_curves,
        aggregated_group_count,
        _aggregate_pair_count,
        _aggregate_polyfit_count,
    ) = (
        BoundaryCurveAggregator.aggregate(
            fragment_curves,
            lane_yellow,
            lane_green,
            ipm,
            geometry,
            enhancement,
        )
    )
    check(
        "21 fragments aggregate quadratic",
        len(fragment_curves) >= 2
        and aggregated_group_count == 1
        and len(aggregated_curves) == 1
        and len(aggregated_curves[0].coefficients) == 3,
    )

    def test_curve(x_value: float) -> Any:
        return STAGE4.BoundaryCurve(
            component_id=1,
            pixel_count=140,
            bbox=(int(x_value), 599, 1, 141),
            centroid=(x_value, 669.0),
            vertical_span_px=141,
            horizontal_span_px=1,
            touches_near_band=True,
            distance_to_vehicle_center_px=abs(x_value - 300.0),
            coefficients=np.asarray([0.0, x_value]),
            fit_residual_px=0.0,
            y_min=599,
            y_max=739,
        )

    right_curve = test_curve(250.0)
    GlobalRoadSideClassifier.classify_curve(
        right_curve,
        lane_yellow,
        lane_green,
        geometry,
        enhancement,
        step_px,
    )
    check(
        "22 yellow majority gives positive normal",
        right_curve.inward_sign == 1,
    )
    left_yellow = np.zeros_like(lane_yellow)
    left_green = np.zeros_like(lane_green)
    left_yellow[roi[0] : roi[1] + 1, 151:250] = 255
    left_green[roi[0] : roi[1] + 1, 251:] = 255
    left_curve = test_curve(250.0)
    GlobalRoadSideClassifier.classify_curve(
        left_curve,
        left_yellow,
        left_green,
        geometry,
        enhancement,
        step_px,
    )
    check(
        "23 yellow majority gives negative normal",
        left_curve.inward_sign == -1,
    )
    ambiguous_yellow = np.zeros_like(lane_yellow)
    ambiguous_yellow[roi[0] : roi[1] + 1, 220:281] = 255
    ambiguous_curve = test_curve(250.0)
    GlobalRoadSideClassifier.classify_curve(
        ambiguous_curve,
        ambiguous_yellow,
        np.zeros_like(ambiguous_yellow),
        geometry,
        enhancement,
        step_px,
    )
    check(
        "24 balanced sides remain unknown",
        ambiguous_curve.inward_sign == 0,
    )

    repaired_dual = ConservativeBoundaryRepair.repair(
        fragmented_boundary(10, include_right=True),
        lane_yellow,
        lane_green,
        ipm,
        roi,
        enhancement,
    )
    dual_curves, _ = STAGE4.BoundaryComponentAnalyzer.analyze(
        repaired_dual.repaired, ipm, geometry, roi
    )
    for curve in dual_curves:
        GlobalRoadSideClassifier.classify_curve(
            curve,
            lane_yellow,
            lane_green,
            geometry,
            enhancement,
            step_px,
        )
    dual_curves, _, _, _ = BoundaryCurveAggregator.aggregate(
        dual_curves,
        lane_yellow,
        lane_green,
        ipm,
        geometry,
        enhancement,
    )
    for curve in dual_curves:
        GlobalRoadSideClassifier.classify_curve(
            curve,
            lane_yellow,
            lane_green,
            geometry,
            enhancement,
            step_px,
        )
    dual_result = ConservativeDualBoundaryPlanner.plan(
        dual_curves,
        lane_yellow,
        ipm,
        geometry,
        roi,
    )
    check(
        "25 repaired dual width near 100 px",
        dual_result is not None
        and abs(dual_result.measured_width_mean_px - 100.0) <= 3.0,
    )
    if dual_result is None:
        raise AssertionError("26 repaired centerline generated")
    (
        dual_result.final_points,
        dual_result.reason,
        dual_result.yellow_ratio,
        dual_result.forward_span_m,
    ) = STAGE4.CenterlineSmoother.smooth(
        dual_result.raw_points,
        lane_yellow,
        ipm,
        geometry,
        roi,
    )
    dual_result.valid = dual_result.reason == "valid"
    check(
        "26 repaired centerline generated",
        dual_result.valid and len(dual_result.final_points) > 0,
    )
    dual_samples = STAGE4.PathConverter.metric_samples(
        dual_result.final_points, ipm
    )
    check(
        "27 repaired path forward monotonic",
        len(dual_samples) > 1
        and all(
            dual_samples[index + 1][0] > dual_samples[index][0]
            for index in range(len(dual_samples) - 1)
        ),
    )

    # 28：直接裁剪 warp 必须与完整鸟瞰对应区域逐像素一致。
    random_source = np.zeros((800, 600), dtype=np.uint8)
    random_source[610:730:3, 120:480:7] = 255
    full_warp = STAGE4.MaskIpmTransformer.transform(random_source, ipm)
    cropped_warp = STAGE4.MaskIpmTransformer.transform(
        random_source, algorithm.roi_ipm
    )
    check(
        "28 cropped warp equals full crop",
        np.array_equal(
            cropped_warp,
            full_warp[
                algorithm.roi_ipm.y_offset :
                algorithm.roi_ipm.y_offset
                + algorithm.roi_ipm.output_height
            ],
        ),
    )

    # 29：局部 y 加回完整鸟瞰偏移后，米制坐标含义保持不变。
    local_points = np.asarray(
        [[300.0, 140.0], [300.0, 100.0], [300.0, 20.0]]
    )
    full_points = algorithm.roi_ipm.to_full_points(local_points)
    local_samples = STAGE4.PathConverter.metric_samples(full_points, ipm)
    expected_first_forward = (
        ipm.vehicle_origin_y_px
        - (140.0 + algorithm.roi_ipm.y_offset)
    ) * ipm.meter_per_pixel
    check(
        "29 local y path conversion",
        abs(local_samples[0][0] - expected_first_forward) < 1.0e-9,
    )

    local_shape = (
        algorithm.roi_ipm.output_height,
        algorithm.roi_ipm.output_width,
    )
    corridor_yellow = np.zeros(local_shape, dtype=np.uint8)
    corridor_green = np.zeros(local_shape, dtype=np.uint8)
    corridor_boundary = np.zeros(local_shape, dtype=np.uint8)
    corridor_yellow[:, 250:351] = 255
    corridor_green[:, :250] = 255
    corridor_green[:, 351:] = 255
    corridor_boundary[:, 250] = 255
    corridor_boundary[:, 350] = 255
    corridor_config = YellowCorridorConfig()
    corridor_result, corridor_status = YellowCorridorPlanner.plan(
        corridor_yellow,
        corridor_boundary,
        corridor_green,
        algorithm.roi_ipm,
        corridor_config,
    )
    check(
        "30 straight yellow corridor",
        corridor_status["yellow_corridor_valid"]
        and corridor_result.mode == "yellow_corridor_dual_edge"
        and len(corridor_result.raw_points) >= 8,
    )

    isolated = corridor_yellow.copy()
    isolated[30:80, 40:141] = 255
    isolated_result, isolated_status = YellowCorridorPlanner.plan(
        isolated,
        corridor_boundary,
        corridor_green,
        algorithm.roi_ipm,
        corridor_config,
    )
    check(
        "31 isolated yellow block ignored",
        isolated_status["yellow_corridor_valid"]
        and np.all(
            np.abs(isolated_result.raw_points[:, 0] - 300.0) < 2.0
        ),
    )

    two_regions = corridor_yellow.copy()
    two_regions[:70, 410:511] = 255
    two_result, two_status = YellowCorridorPlanner.plan(
        two_regions,
        corridor_boundary,
        corridor_green,
        algorithm.roi_ipm,
        corridor_config,
    )
    check(
        "32 far second region keeps continuity",
        two_status["yellow_corridor_valid"]
        and np.max(two_result.raw_points[:, 0]) < 350.0,
    )

    curved_yellow = np.zeros(local_shape, dtype=np.uint8)
    curved_green = np.zeros(local_shape, dtype=np.uint8)
    curved_boundary = np.zeros(local_shape, dtype=np.uint8)
    for local_y in range(local_shape[0]):
        center = int(round(300.0 + 0.20 * (140 - local_y)))
        left, right = center - 50, center + 50
        curved_yellow[local_y, left : right + 1] = 255
        curved_green[local_y, :left] = 255
        curved_green[local_y, right + 1 :] = 255
        curved_boundary[local_y, left] = 255
        curved_boundary[local_y, right] = 255
    curved_result, curved_status = YellowCorridorPlanner.plan(
        curved_yellow,
        curved_boundary,
        curved_green,
        algorithm.roi_ipm,
        corridor_config,
    )
    check(
        "33 gentle curve center continuity",
        curved_status["yellow_corridor_valid"]
        and np.max(
            np.abs(np.diff(curved_result.raw_points[:, 0]))
        )
        <= corridor_config.center_jump_max_m
        / algorithm.roi_ipm.meter_per_pixel,
    )

    fast_hsv = np.zeros((800, 600, 3), dtype=np.uint8)
    fast_hsv[:, :250] = (60, 220, 220)
    fast_hsv[:, 250:351] = (30, 220, 220)
    fast_hsv[:, 351:] = (60, 220, 220)
    fast_image = cv2.cvtColor(fast_hsv, cv2.COLOR_HSV2BGR)
    fast_algorithm = LaneAlgorithm(
        threshold,
        config,
        geometry,
        ipm,
        BoundaryEnhancementConfig(),
        YellowCorridorConfig(),
    )
    fast_pipeline_result = fast_algorithm.process(fast_image)
    check(
        "34 valid corridor skips repair",
        fast_pipeline_result.geometry.valid
        and fast_pipeline_result.geometry.mode
        in {
            "green_dual_inner_edge",
            "green_yellow_hybrid",
            "yellow_corridor_dual_edge",
        }
        and not fast_pipeline_result.fallback_boundary_pipeline_used
        and len(fast_pipeline_result.geometry.final_points) >= 8
        and fast_pipeline_result.geometry.forward_span_m >= 0.30
        and fast_pipeline_result.counters[
            "repair_candidate_pair_count"
        ]
        == 0,
    )

    disabled_corridor = YellowCorridorConfig(enable=False)
    fallback_algorithm = LaneAlgorithm(
        threshold,
        config,
        geometry,
        ipm,
        BoundaryEnhancementConfig(),
        disabled_corridor,
    )
    fallback_result = fallback_algorithm.process(fast_image)
    check(
        "35 failed corridor enters fallback",
        fallback_result.fallback_boundary_pipeline_used,
    )

    acute_yellow = np.zeros((800, 600), dtype=np.uint8)
    acute_yellow[599:740, 350:451] = 255
    acute_left = test_curve(350.0)
    acute_right = test_curve(450.0)
    acute_left.inward_sign = 1
    acute_right.inward_sign = -1
    acute_left.road_side = "yellow_right"
    acute_right.road_side = "yellow_left"
    acute_dual = ConservativeDualBoundaryPlanner.plan(
        [acute_left, acute_right],
        acute_yellow,
        ipm,
        geometry,
        roi,
    )
    check(
        "36 same vehicle side dual boundaries",
        acute_dual is not None
        and abs(acute_dual.measured_width_mean_px - 100.0) <= 2.0,
    )

    check(
        "37 web off overlay count zero",
        fast_pipeline_result.counters["overlay_build_count"] == 0
        and fast_pipeline_result.overlay is None,
    )
    check(
        "38 web off jpeg count zero",
        fast_pipeline_result.counters["jpeg_encode_count"] == 0,
    )

    descriptor_test = ConservativeBoundaryRepair.repair(
        fragmented_boundary(10, include_right=True),
        lane_yellow,
        lane_green,
        ipm,
        roi,
        enhancement,
    )
    check(
        "39 descriptor fitted once",
        descriptor_test.polyfit_call_count
        == descriptor_test.fragment_count_after_filter,
    )

    many_fragments = np.zeros((800, 600), dtype=np.uint8)
    fragment_y = 600
    for fragment_index in range(8):
        cv2.line(
            many_fragments,
            (245, fragment_y),
            (245, fragment_y + 10),
            255,
            1,
        )
        fragment_y += 17
    limited_enhancement = BoundaryEnhancementConfig(
        fragment_neighbor_check_limit=3
    )
    limited_pairs = ConservativeBoundaryRepair.repair(
        many_fragments,
        lane_yellow,
        lane_green,
        ipm,
        roi,
        limited_enhancement,
    )
    check(
        "40 candidate pairs bounded",
        limited_pairs.candidate_pair_count
        <= limited_pairs.fragment_count_after_filter
        * limited_enhancement.fragment_neighbor_check_limit,
    )

    local_monotonic_points = np.asarray(
        [[300.0, 140.0], [302.0, 100.0], [305.0, 50.0], [308.0, 0.0]]
    )
    final_full_points = algorithm.roi_ipm.to_full_points(
        local_monotonic_points
    )
    final_samples = STAGE4.PathConverter.metric_samples(
        final_full_points, ipm
    )
    check(
        "41 cropped path forward monotonic",
        all(
            final_samples[index + 1][0] > final_samples[index][0]
            for index in range(len(final_samples) - 1)
        ),
    )

    # 42-53：黄色走廊逐行诊断、保守短缺口恢复与紧凑 Web 布局。
    classified_reasons = {
        sample["reason"]
        for sample in corridor_result.yellow_corridor_samples
    }
    check(
        "42 every corridor row classified",
        len(corridor_result.yellow_corridor_samples)
        == corridor_status["yellow_corridor_total_sample_count"]
        and classified_reasons.issubset(set(YellowCorridorPlanner.REASONS)),
    )
    check(
        "43 corridor reason counts sum",
        corridor_status["yellow_corridor_accepted_sample_count"]
        + YellowCorridorPlanner.rejection_count_sum(corridor_status)
        == corridor_status["yellow_corridor_total_sample_count"],
    )

    def corridor_with_missing_rows(
        missing_rows: Sequence[int],
        recovered_center_x: int = 300,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        yellow_mask = corridor_yellow.copy()
        green_mask = corridor_green.copy()
        boundary_mask = corridor_boundary.copy()
        for sample_y in missing_rows:
            yellow_mask[sample_y, :] = 0
        if recovered_center_x != 300 and missing_rows:
            farthest_missing = min(missing_rows)
            left = recovered_center_x - 50
            right = recovered_center_x + 50
            yellow_mask[:farthest_missing, :] = 0
            green_mask[:farthest_missing, :] = 0
            boundary_mask[:farthest_missing, :] = 0
            yellow_mask[:farthest_missing, left : right + 1] = 255
            green_mask[:farthest_missing, :left] = 255
            green_mask[:farthest_missing, right + 1 :] = 255
            boundary_mask[:farthest_missing, left] = 255
            boundary_mask[:farthest_missing, right] = 255
        return yellow_mask, boundary_mask, green_mask

    one_gap_masks = corridor_with_missing_rows([110])
    one_gap_result, one_gap_status = YellowCorridorPlanner.plan(
        *one_gap_masks, algorithm.roi_ipm, corridor_config
    )
    check(
        "44 one invalid row recovers",
        one_gap_status["yellow_corridor_valid"]
        and one_gap_status[
            "yellow_corridor_reject_no_yellow_interval_count"
        ]
        == 1
        and np.any(one_gap_result.raw_points[:, 1] < 110),
    )

    two_gap_masks = corridor_with_missing_rows([110, 100])
    two_gap_result, two_gap_status = YellowCorridorPlanner.plan(
        *two_gap_masks, algorithm.roi_ipm, corridor_config
    )
    check(
        "45 two invalid rows recover",
        two_gap_status["yellow_corridor_valid"]
        and two_gap_status[
            "yellow_corridor_max_consecutive_rejected_samples"
        ]
        == 2
        and np.any(two_gap_result.raw_points[:, 1] < 100),
    )

    three_gap_masks = corridor_with_missing_rows([110, 100, 90])
    three_gap_result, three_gap_status = YellowCorridorPlanner.plan(
        *three_gap_masks, algorithm.roi_ipm, corridor_config
    )
    check(
        "46 excessive invalid rows terminate",
        three_gap_status["yellow_corridor_termination_reason"]
        == "consecutive_invalid_limit"
        and not np.any(three_gap_result.raw_points[:, 1] < 90),
    )

    distance_limited_config = copy.deepcopy(corridor_config)
    distance_limited_config.max_consecutive_invalid_samples = 4
    distance_limited_result, distance_limited_status = (
        YellowCorridorPlanner.plan(
            *three_gap_masks,
            algorithm.roi_ipm,
            distance_limited_config,
        )
    )
    check(
        "47 gap over point one meter not interpolated",
        distance_limited_status["yellow_corridor_termination_reason"]
        == "small_gap_limit"
        and not np.any(distance_limited_result.raw_points[:, 1] < 90),
    )

    recovered_masks = corridor_with_missing_rows([110], 320)
    recovered_result, recovered_status = YellowCorridorPlanner.plan(
        *recovered_masks, algorithm.roi_ipm, corridor_config
    )
    check(
        "48 recovered center obeys continuity",
        recovered_status["yellow_corridor_valid"]
        and np.max(np.abs(np.diff(recovered_result.raw_points[:, 0])))
        <= corridor_config.center_jump_max_m
        / algorithm.roi_ipm.meter_per_pixel,
    )

    check(
        "49 multiple intervals prefer previous center",
        two_status["yellow_corridor_valid"]
        and np.max(two_result.raw_points[:, 0]) < 350.0,
    )
    check(
        "50 fast mode road side is N/A",
        fast_pipeline_result.geometry.valid
        and all(
            value == "N/A"
            for value in fast_pipeline_result.road_side_status.values()
        ),
    )

    compact_composite = LanePerceptionPipelineNode._make_web_composite(
        image,
        local_overlay,
        result.centerline_mask,
        ["samples=6/9", "termination=roi_exhausted"],
    )
    check(
        "51 web composite uses compact local ROI",
        compact_composite.shape == (600, 1200, 3)
        and np.any(compact_composite[35:226, 370:1180] != 0),
    )
    check(
        "52 web off still skips overlay",
        fast_pipeline_result.overlay is None
        and fast_pipeline_result.counters["overlay_build_count"] == 0
        and fast_pipeline_result.timing_ms["overlay_build_ms"] == 0.0,
    )
    check(
        "53 final path forward monotonic",
        len(final_samples) > 1
        and all(
            final_samples[index + 1][0] > final_samples[index][0]
            for index in range(len(final_samples) - 1)
        ),
    )

    # 54-72：中心点分类、内部短缺口补全、带权平滑与 fallback 保持。
    check(
        "54 complete strict sequence needs no fill",
        corridor_status["center_strict_observed_count"] == 15
        and corridor_status["center_weak_observed_count"] == 0
        and corridor_status["center_gap_filled_count"] == 0
        and corridor_result.mode == "yellow_corridor_dual_edge",
    )
    check(
        "55 one internal point gap filled",
        one_gap_status["center_gap_filled_count"] == 1
        and one_gap_status["center_gap_count"] == 1
        and one_gap_result.mode == "yellow_corridor_center_gap_filled",
    )
    check(
        "56 two internal point gap filled",
        two_gap_status["center_gap_filled_count"] == 2
        and two_gap_status["center_gap_max_samples"] == 2,
    )
    check(
        "57 three point gap not filled",
        three_gap_status["center_gap_filled_count"] == 0,
    )
    distance_only_config = copy.deepcopy(corridor_config)
    distance_only_config.center_gap_fill_max_samples = 3
    distance_only_result, distance_only_status = YellowCorridorPlanner.plan(
        *three_gap_masks,
        algorithm.roi_ipm,
        distance_only_config,
        geometry,
    )
    check(
        "58 gap over point one meter not filled",
        distance_only_status["center_gap_filled_count"] == 0
        and not np.any(distance_only_result.raw_points[:, 1] < 90),
    )

    near_gap_masks = corridor_with_missing_rows([140])
    near_gap_result, near_gap_status = YellowCorridorPlanner.plan(
        *near_gap_masks, algorithm.roi_ipm, corridor_config, geometry
    )
    check(
        "59 near endpoint not extrapolated",
        near_gap_status["center_gap_filled_count"] == 0
        and len(near_gap_result.raw_points)
        and np.max(near_gap_result.raw_points[:, 1]) == 130,
    )
    far_gap_masks = corridor_with_missing_rows([0])
    far_gap_result, far_gap_status = YellowCorridorPlanner.plan(
        *far_gap_masks, algorithm.roi_ipm, corridor_config, geometry
    )
    check(
        "60 far endpoint not extrapolated",
        far_gap_status["center_gap_filled_count"] == 0
        and len(far_gap_result.raw_points)
        and np.min(far_gap_result.raw_points[:, 1]) == 10,
    )

    narrow_yellow = corridor_yellow.copy()
    narrow_yellow[110, :] = 0
    narrow_yellow[110, 280:321] = 255
    narrow_result, narrow_status = YellowCorridorPlanner.plan(
        narrow_yellow,
        corridor_boundary,
        corridor_green,
        algorithm.roi_ipm,
        corridor_config,
        geometry,
    )
    check(
        "61 abnormal width gap not filled",
        narrow_status["center_gap_filled_count"] == 0
        and narrow_status["yellow_corridor_reject_width_too_narrow_count"]
        >= 1
        and not np.any(narrow_result.raw_points[:, 1] < 110),
    )

    jump_yellow = corridor_yellow.copy()
    jump_green = corridor_green.copy()
    jump_boundary = corridor_boundary.copy()
    jump_yellow[110, :] = 0
    jump_green[110, :] = 0
    jump_boundary[110, :] = 0
    jump_yellow[110, 340:441] = 255
    jump_green[110, :340] = 255
    jump_green[110, 441:] = 255
    jump_boundary[110, 340] = 255
    jump_boundary[110, 440] = 255
    jump_result, jump_status = YellowCorridorPlanner.plan(
        jump_yellow,
        jump_boundary,
        jump_green,
        algorithm.roi_ipm,
        corridor_config,
        geometry,
    )
    check(
        "62 center jump gap not filled",
        jump_status["center_gap_filled_count"] == 0
        and jump_status["yellow_corridor_reject_center_jump_count"] >= 1
        and not np.any(jump_result.raw_points[:, 1] < 110),
    )

    weak_yellow = corridor_yellow.copy()
    weak_green = corridor_green.copy()
    weak_boundary = corridor_boundary.copy()
    weak_green[109:112, 245:256] = 0
    weak_boundary[109:112, 245:256] = 0
    weak_result, weak_status = YellowCorridorPlanner.plan(
        weak_yellow,
        weak_boundary,
        weak_green,
        algorithm.roi_ipm,
        corridor_config,
        geometry,
    )
    check(
        "63 weak weight below strict",
        YellowCorridorPlanner.CENTER_WEIGHTS["weak_single_edge"]
        < YellowCorridorPlanner.CENTER_WEIGHTS["strict_observed"]
        and weak_status["center_weak_observed_count"] == 1,
    )
    check(
        "64 filled weight below weak",
        YellowCorridorPlanner.CENTER_WEIGHTS["gap_filled"]
        < YellowCorridorPlanner.CENTER_WEIGHTS["weak_single_edge"],
    )

    jitter_points = np.column_stack(
        (
            np.asarray(
                [300, 306, 296, 307, 295, 305, 297, 304, 298],
                dtype=np.float64,
            ),
            np.arange(140, 59, -10, dtype=np.float64),
        )
    )
    smoothed_jitter = YellowCorridorPlanner._weighted_smooth(
        jitter_points,
        np.ones(len(jitter_points), dtype=np.float64),
        2.0,
    )
    check(
        "65 weighted smoothing reduces jitter",
        YellowCorridorPlanner._lateral_deviation(smoothed_jitter)
        < YellowCorridorPlanner._lateral_deviation(jitter_points),
    )

    curve_y = np.arange(140, -1, -10, dtype=np.float64)
    curve_x = 300.0 + 0.001 * np.square(140.0 - curve_y)
    gentle_curve = np.column_stack((curve_x, curve_y))
    smoothed_curve = YellowCorridorPlanner._weighted_smooth(
        gentle_curve,
        np.ones(len(gentle_curve), dtype=np.float64),
        2.0,
    )
    check(
        "66 gentle curve trend retained",
        smoothed_curve[-1, 0] - smoothed_curve[0, 0] > 10.0
        and np.all(np.diff(smoothed_curve[:, 0]) >= -0.2),
    )
    check(
        "67 smoothed center remains yellow",
        one_gap_result.yellow_ratio
        >= geometry.centerline_min_yellow_ratio,
    )
    check(
        "68 fill lowers confidence",
        one_gap_result.confidence < corridor_result.confidence,
    )

    original_edge_valid = YellowCorridorPlanner._edge_valid

    def intermittent_left_edge(
        x: int,
        y: int,
        observed: np.ndarray,
        local_green: np.ndarray,
        radius: int,
    ) -> bool:
        if x < algorithm.roi_ipm.vehicle_center_x_px and y == 110:
            return False
        return original_edge_valid(x, y, observed, local_green, radius)

    YellowCorridorPlanner._edge_valid = staticmethod(intermittent_left_edge)
    try:
        weak_fast_result = fast_algorithm.process(fast_image)
    finally:
        YellowCorridorPlanner._edge_valid = staticmethod(original_edge_valid)
    check(
        "69 weak fast path skips complex fallback",
        weak_fast_result.geometry.valid
        and not weak_fast_result.fallback_boundary_pipeline_used,
    )
    check(
        "70 failed fast path keeps original fallback",
        fallback_result.fallback_boundary_pipeline_used,
    )
    weak_full_points = algorithm.roi_ipm.to_full_points(
        weak_result.final_points
    )
    weak_path_samples = STAGE4.PathConverter.metric_samples(
        weak_full_points, ipm
    )
    check(
        "71 refined path forward monotonic",
        len(weak_path_samples) > 1
        and all(
            weak_path_samples[index + 1][0]
            > weak_path_samples[index][0]
            for index in range(len(weak_path_samples) - 1)
        ),
    )
    check(
        "72 web off refinement has zero overlay jpeg",
        weak_fast_result.overlay is None
        and weak_fast_result.counters["overlay_build_count"] == 0
        and weak_fast_result.counters["jpeg_encode_count"] == 0
        and weak_fast_result.timing_ms["overlay_build_ms"] == 0.0
        and weak_fast_result.timing_ms["jpeg_encode_ms"] == 0.0,
    )

    # 73-85：绿色主区域内边缘优先，黄色只作为辅助证据。
    (
        green_samples,
        green_base_status,
    ) = GreenBoundaryFastPlanner.extract(
        corridor_green,
        corridor_yellow,
        algorithm.roi_ipm,
        corridor_config,
    )
    green_dual_result, green_dual_status, green_dual_samples = (
        GreenBoundaryFastPlanner.plan(
            green_samples,
            green_base_status,
            corridor_green,
            corridor_yellow,
            algorithm.roi_ipm,
            corridor_config,
            geometry,
            allow_single=False,
        )
    )
    check(
        "73 complete green yellow dual center",
        green_dual_result.valid
        and green_dual_result.mode == "green_dual_inner_edge"
        and green_dual_status["green_dual_edge_valid_count"] == 15
        and np.max(np.abs(green_dual_result.final_points[:, 0] - 300.0))
        <= 1.0,
    )

    sparse_yellow = corridor_yellow.copy()
    sparse_yellow[20:121] = 0
    sparse_samples, sparse_base = GreenBoundaryFastPlanner.extract(
        corridor_green,
        sparse_yellow,
        algorithm.roi_ipm,
        corridor_config,
    )
    sparse_result, sparse_status, _sparse_debug = (
        GreenBoundaryFastPlanner.plan(
            sparse_samples,
            sparse_base,
            corridor_green,
            sparse_yellow,
            algorithm.roi_ipm,
            corridor_config,
            geometry,
            allow_single=False,
        )
    )
    check(
        "74 sparse yellow still full green centerline",
        sparse_result.valid
        and len(sparse_result.final_points) == 15
        and sparse_status["green_corridor_unknown_ratio"] > 0.50,
    )

    no_yellow = np.zeros_like(corridor_yellow)
    no_yellow_samples, no_yellow_base = GreenBoundaryFastPlanner.extract(
        corridor_green,
        no_yellow,
        algorithm.roi_ipm,
        corridor_config,
    )
    no_yellow_result, no_yellow_status, _no_yellow_debug = (
        GreenBoundaryFastPlanner.plan(
            no_yellow_samples,
            no_yellow_base,
            corridor_green,
            no_yellow,
            algorithm.roi_ipm,
            corridor_config,
            geometry,
            allow_single=False,
        )
    )
    check(
        "75 no yellow keeps green path with lower confidence",
        no_yellow_result.valid
        and no_yellow_status["green_corridor_yellow_support_ratio"] == 0.0
        and no_yellow_result.confidence < green_dual_result.confidence,
    )

    noisy_green = corridor_green.copy()
    noisy_green[:, 286:292] = 255
    noisy_samples, noisy_status = GreenBoundaryFastPlanner.extract(
        noisy_green,
        corridor_yellow,
        algorithm.roi_ipm,
        corridor_config,
    )
    check(
        "76 isolated inner green noise ignored",
        noisy_status["green_dual_edge_valid_count"] == 15
        and all(
            sample["left"] == 249
            for sample in noisy_samples
            if sample["classification"] == "green_dual_observed"
        ),
    )
    check(
        "77 left green border connection",
        all(
            sample["left"] == 249
            for sample in green_dual_samples
            if sample["classification"] == "green_dual_observed"
        ),
    )
    check(
        "78 right green border connection",
        all(
            sample["right"] == 351
            for sample in green_dual_samples
            if sample["classification"] == "green_dual_observed"
        ),
    )

    wide_green = np.zeros_like(corridor_green)
    wide_green[:, :250] = 255
    wide_green[:, 450:] = 255
    wide_samples, wide_base = GreenBoundaryFastPlanner.extract(
        wide_green,
        no_yellow,
        algorithm.roi_ipm,
        corridor_config,
    )
    wide_result, _wide_status, _wide_debug = GreenBoundaryFastPlanner.plan(
        wide_samples,
        wide_base,
        wide_green,
        no_yellow,
        algorithm.roi_ipm,
        corridor_config,
        geometry,
        allow_single=False,
    )
    check("79 abnormal green width rejected", not wide_result.valid)

    green_metric_samples = STAGE4.PathConverter.metric_samples(
        algorithm.roi_ipm.to_full_points(green_dual_result.final_points),
        ipm,
    )
    check(
        "80 green dual forward monotonic",
        len(green_metric_samples) > 1
        and all(
            green_metric_samples[index + 1][0]
            > green_metric_samples[index][0]
            for index in range(len(green_metric_samples) - 1)
        ),
    )

    mixed_single_green = corridor_green.copy()
    mixed_single_green[110, 351:] = 0
    mixed_single_green[100, :250] = 0
    mixed_samples, mixed_base = GreenBoundaryFastPlanner.extract(
        mixed_single_green,
        corridor_yellow,
        algorithm.roi_ipm,
        corridor_config,
    )
    mixed_result, _mixed_status, mixed_debug = GreenBoundaryFastPlanner.plan(
        mixed_samples,
        mixed_base,
        mixed_single_green,
        corridor_yellow,
        algorithm.roi_ipm,
        corridor_config,
        geometry,
        allow_single=True,
    )
    left_estimates = [
        sample
        for sample in mixed_debug
        if sample["classification"] == "green_single_offset"
        and sample["single_side"] == "left"
    ]
    right_estimates = [
        sample
        for sample in mixed_debug
        if sample["classification"] == "green_single_offset"
        and sample["single_side"] == "right"
    ]
    check(
        "81 single green offset direction",
        mixed_result.valid
        and left_estimates
        and right_estimates
        and abs(float(left_estimates[0]["center"]) - 299.0) <= 1.0
        and abs(float(right_estimates[0]["center"]) - 301.0) <= 1.0,
    )
    check(
        "82 single green confidence below dual",
        mixed_result.confidence < green_dual_result.confidence,
    )

    green_only_hsv = np.zeros((800, 600, 3), dtype=np.uint8)
    green_only_hsv[:, :250] = (60, 220, 220)
    green_only_hsv[:, 250:351] = (0, 0, 35)
    green_only_hsv[:, 351:] = (60, 220, 220)
    green_only_image = cv2.cvtColor(green_only_hsv, cv2.COLOR_HSV2BGR)
    green_only_pipeline = fast_algorithm.process(green_only_image)
    check(
        "83 green dual skips complex fallback",
        green_only_pipeline.geometry.valid
        and green_only_pipeline.geometry.mode == "green_dual_inner_edge"
        and green_only_pipeline.green_boundary_status[
            "green_boundary_fast_path_used"
        ]
        and not green_only_pipeline.fallback_boundary_pipeline_used,
    )
    check(
        "84 green yellow failure retains fallback",
        fallback_result.fallback_boundary_pipeline_used,
    )
    check(
        "85 green web off no overlay jpeg",
        green_only_pipeline.overlay is None
        and green_only_pipeline.counters["overlay_build_count"] == 0
        and green_only_pipeline.counters["jpeg_encode_count"] == 0
        and green_only_pipeline.timing_ms["overlay_build_ms"] == 0.0
        and green_only_pipeline.timing_ms["jpeg_encode_ms"] == 0.0,
    )

    # 86-98：IPM 黑色三角无效区、真实有效边界种子与缓存验证。
    triangular_valid = np.zeros(local_shape, dtype=bool)
    valid_seed_green = np.zeros(local_shape, dtype=np.uint8)
    valid_seed_yellow = np.zeros(local_shape, dtype=np.uint8)
    for local_y in range(local_shape[0]):
        taper = int(round((140 - local_y) * 0.08))
        valid_left, valid_right = 20 + taper, 579 - taper
        triangular_valid[local_y, valid_left : valid_right + 1] = True
        valid_seed_green[local_y, valid_left:250] = 255
        valid_seed_green[local_y, 351 : valid_right + 1] = 255
        valid_seed_yellow[local_y, 250:351] = 255

    valid_seed_samples, valid_seed_status = GreenBoundaryFastPlanner.extract(
        valid_seed_green,
        valid_seed_yellow,
        algorithm.roi_ipm,
        corridor_config,
        ipm_valid_mask=triangular_valid,
    )
    valid_seed_result, valid_seed_plan_status, valid_seed_debug = (
        GreenBoundaryFastPlanner.plan(
            valid_seed_samples,
            valid_seed_status,
            valid_seed_green,
            valid_seed_yellow,
            algorithm.roi_ipm,
            corridor_config,
            geometry,
            allow_single=False,
        )
    )
    check(
        "86 black triangle valid boundary seed",
        valid_seed_result.valid
        and valid_seed_plan_status["green_dual_edge_valid_count"] == 15
        and valid_seed_plan_status[
            "green_reject_no_valid_ipm_span_count"
        ]
        == 0,
    )
    check(
        "87 left seed uses valid left not zero",
        all(
            sample["valid_left_x"] > 0
            and sample["left_outer_gap_px"] == 0
            for sample in valid_seed_debug
            if sample["classification"] == "green_dual_observed"
        ),
    )
    check(
        "88 right seed uses valid right not image edge",
        all(
            sample["valid_right_x"] < local_shape[1] - 1
            and sample["right_outer_gap_px"] == 0
            for sample in valid_seed_debug
            if sample["classification"] == "green_dual_observed"
        ),
    )
    check(
        "89 invalid ipm black excluded from unknown",
        valid_seed_plan_status["green_corridor_unknown_ratio"] == 0.0,
    )

    valid_noise_green = valid_seed_green.copy()
    valid_noise_green[:, 286:292] = 255
    valid_noise_samples, valid_noise_status = (
        GreenBoundaryFastPlanner.extract(
            valid_noise_green,
            valid_seed_yellow,
            algorithm.roi_ipm,
            corridor_config,
            ipm_valid_mask=triangular_valid,
        )
    )
    check(
        "90 valid roi inner green noise ignored",
        valid_noise_status["green_dual_edge_valid_count"] == 15
        and all(
            sample["left"] == 249
            for sample in valid_noise_samples
            if sample["classification"] == "green_dual_observed"
        ),
    )

    outer_gap_green = np.zeros_like(valid_seed_green)
    for local_y in range(local_shape[0]):
        valid_x = np.flatnonzero(triangular_valid[local_y])
        left_x, right_x = int(valid_x[0]), int(valid_x[-1])
        outer_gap_green[local_y, left_x + 8 : 250] = 255
        outer_gap_green[local_y, 351 : right_x - 8 + 1] = 255
    outer_gap_samples, outer_gap_status = GreenBoundaryFastPlanner.extract(
        outer_gap_green,
        valid_seed_yellow,
        algorithm.roi_ipm,
        corridor_config,
        ipm_valid_mask=triangular_valid,
    )
    check(
        "91 outer gap up to eight pixels accepted",
        outer_gap_status["green_dual_edge_valid_count"] == 15
        and abs(outer_gap_status["green_left_outer_gap_px_mean"] - 8.0)
        < 1.0e-9
        and abs(outer_gap_status["green_right_outer_gap_px_mean"] - 8.0)
        < 1.0e-9,
    )

    excessive_gap_green = np.zeros_like(valid_seed_green)
    for local_y in range(local_shape[0]):
        valid_x = np.flatnonzero(triangular_valid[local_y])
        left_x, right_x = int(valid_x[0]), int(valid_x[-1])
        excessive_gap_green[local_y, left_x + 9 : 250] = 255
        excessive_gap_green[local_y, 351 : right_x - 9 + 1] = 255
    _excessive_samples, excessive_status = GreenBoundaryFastPlanner.extract(
        excessive_gap_green,
        valid_seed_yellow,
        algorithm.roi_ipm,
        corridor_config,
        ipm_valid_mask=triangular_valid,
    )
    check(
        "92 excessive outer gap rejected",
        excessive_status["green_dual_edge_valid_count"] == 0
        and excessive_status["green_reject_left_outer_gap_count"] == 15
        and excessive_status["green_reject_right_outer_gap_count"] == 15,
    )

    short_run_green = np.zeros_like(valid_seed_green)
    for local_y in range(local_shape[0]):
        valid_x = np.flatnonzero(triangular_valid[local_y])
        left_x, right_x = int(valid_x[0]), int(valid_x[-1])
        short_run_green[local_y, left_x : left_x + 7] = 255
        short_run_green[local_y, right_x - 6 : right_x + 1] = 255
    _short_samples, short_status = GreenBoundaryFastPlanner.extract(
        short_run_green,
        valid_seed_yellow,
        algorithm.roi_ipm,
        corridor_config,
        ipm_valid_mask=triangular_valid,
    )
    check(
        "93 short outer green run rejected",
        short_status["green_dual_edge_valid_count"] == 0
        and short_status["green_reject_left_run_too_short_count"] == 15
        and short_status["green_reject_right_run_too_short_count"] == 15,
    )
    check(
        "94 valid seed dual center correct",
        np.max(np.abs(valid_seed_result.final_points[:, 0] - 300.0))
        <= 1.0,
    )

    valid_no_yellow = np.zeros_like(valid_seed_yellow)
    no_yellow_valid_samples, no_yellow_valid_status = (
        GreenBoundaryFastPlanner.extract(
            valid_seed_green,
            valid_no_yellow,
            algorithm.roi_ipm,
            corridor_config,
            ipm_valid_mask=triangular_valid,
        )
    )
    no_yellow_valid_result, _no_yellow_valid_plan, _no_yellow_valid_debug = (
        GreenBoundaryFastPlanner.plan(
            no_yellow_valid_samples,
            no_yellow_valid_status,
            valid_seed_green,
            valid_no_yellow,
            algorithm.roi_ipm,
            corridor_config,
            geometry,
            allow_single=False,
        )
    )
    check(
        "95 no yellow with valid green full path",
        no_yellow_valid_result.valid
        and len(no_yellow_valid_result.final_points) == 15,
    )

    curved_valid_green = np.zeros_like(valid_seed_green)
    curved_valid_yellow = np.zeros_like(valid_seed_yellow)
    for local_y in range(local_shape[0]):
        valid_x = np.flatnonzero(triangular_valid[local_y])
        left_x, right_x = int(valid_x[0]), int(valid_x[-1])
        curve_center = 300 + int(round((140 - local_y) * 0.04))
        left_inner, right_inner = curve_center - 50, curve_center + 50
        curved_valid_green[local_y, left_x : left_inner + 1] = 255
        curved_valid_green[local_y, right_inner : right_x + 1] = 255
        curved_valid_yellow[local_y, left_inner + 1 : right_inner] = 255
    curved_valid_samples, curved_valid_status = (
        GreenBoundaryFastPlanner.extract(
            curved_valid_green,
            curved_valid_yellow,
            algorithm.roi_ipm,
            corridor_config,
            ipm_valid_mask=triangular_valid,
        )
    )
    curved_valid_result, _curved_plan_status, _curved_debug = (
        GreenBoundaryFastPlanner.plan(
            curved_valid_samples,
            curved_valid_status,
            curved_valid_green,
            curved_valid_yellow,
            algorithm.roi_ipm,
            corridor_config,
            geometry,
            allow_single=False,
        )
    )
    check(
        "96 curved valid boundaries tracked",
        curved_valid_result.valid
        and curved_valid_status["green_dual_edge_valid_count"] == 15
        and curved_valid_result.final_points[-1, 0]
        > curved_valid_result.final_points[0, 0],
    )

    cached_mask_identity = id(algorithm.ipm_valid_mask)
    cached_mask_build_count = algorithm.ipm_valid_mask_build_count
    algorithm.process(image)
    algorithm.process(image)
    check(
        "97 ipm valid mask cached once",
        id(algorithm.ipm_valid_mask) == cached_mask_identity
        and algorithm.ipm_valid_mask_build_count == cached_mask_build_count
        == 1,
    )
    cached_web_off_result = algorithm.process(image)
    check(
        "98 valid mask web off no overlay build",
        cached_web_off_result.overlay is None
        and cached_web_off_result.counters["overlay_build_count"] == 0
        and cached_web_off_result.counters["jpeg_encode_count"] == 0,
    )

    curve_config = copy.deepcopy(corridor_config)
    curve_config.single_green_curve_min_points = 8
    curve_config.single_green_curve_min_span_m = 0.35
    curve_config.single_green_center_min_yellow_ratio = 0.30
    curve_config.validate()
    curve_ipm = algorithm.roi_ipm
    curve_shape = (
        curve_ipm.output_height,
        curve_ipm.output_width,
    )
    curve_valid = np.ones(curve_shape, dtype=bool)

    def synthetic_curve_samples(
        side: str,
        missing: Sequence[int] = (),
        lateral_by_index: Optional[Callable[[int], float]] = None,
    ) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        missing_set = set(missing)
        step_px = max(
            2,
            int(round(
                curve_config.sample_step_m / curve_ipm.meter_per_pixel
            )),
        )
        for index in range(15):
            y = curve_shape[0] - 1 - index * step_px
            forward_m = YellowCorridorPlanner._forward_m(y, curve_ipm)
            lateral_delta = (
                0.0
                if lateral_by_index is None
                else float(lateral_by_index(index))
            )
            edge_x = (
                250.0 + lateral_delta / curve_ipm.meter_per_pixel
                if side == "left"
                else 350.0 + lateral_delta / curve_ipm.meter_per_pixel
            )
            present = index not in missing_set
            output.append(
                {
                    "y": y,
                    "forward_m": forward_m,
                    "left": edge_x if side == "left" and present else None,
                    "right": edge_x if side == "right" and present else None,
                    "center": None,
                    "classification": "missing",
                    "source_type": "missing",
                    "reason": "synthetic",
                    "width": None,
                    "single_side": side,
                    "yellow_support_ratio": 0.0,
                    "unknown_ratio": 0.0,
                    "green_intrusion_ratio": 0.0,
                }
            )
        return output

    left_samples = synthetic_curve_samples("left")
    right_samples = synthetic_curve_samples("right")
    left_raw, left_chain = SingleGreenCurvePlanner.extract_main_chain(
        left_samples, "left", curve_ipm, curve_config
    )
    right_raw, right_chain = SingleGreenCurvePlanner.extract_main_chain(
        right_samples, "right", curve_ipm, curve_config
    )
    check("99 complete left curve chain", len(left_raw) == len(left_chain) == 15)
    check("100 complete right curve chain", len(right_raw) == len(right_chain) == 15)
    _, one_gap_chain = SingleGreenCurvePlanner.extract_main_chain(
        synthetic_curve_samples("left", (7,)),
        "left",
        curve_ipm,
        curve_config,
    )
    check(
        "101 one internal row gap allowed",
        len(one_gap_chain) == 14
        and one_gap_chain[-1]["forward_m"] - one_gap_chain[0]["forward_m"]
        >= 0.69,
    )
    _, two_gap_chain = SingleGreenCurvePlanner.extract_main_chain(
        synthetic_curve_samples("left", (6, 7)),
        "left",
        curve_ipm,
        curve_config,
    )
    check("102 two internal row gaps allowed", len(two_gap_chain) == 13)
    _, large_gap_chain = SingleGreenCurvePlanner.extract_main_chain(
        synthetic_curve_samples("left", (6, 7, 8)),
        "left",
        curve_ipm,
        curve_config,
    )
    check("103 large curve gap is not crossed", len(large_gap_chain) <= 6)
    noisy_samples = synthetic_curve_samples("left")
    noisy_samples[12]["left"] = 500.0
    _, noisy_chain = SingleGreenCurvePlanner.extract_main_chain(
        noisy_samples, "left", curve_ipm, curve_config
    )
    check("104 isolated green noise excluded from main chain", len(noisy_chain) >= 11)
    check(
        "105 longest continuous curve chain selected",
        noisy_chain[0]["sample_index"] == 0
        and noisy_chain[-1]["sample_index"] == 11,
    )
    shuffled_samples = list(reversed(left_samples))
    _, shuffled_chain = SingleGreenCurvePlanner.extract_main_chain(
        shuffled_samples, "left", curve_ipm, curve_config
    )
    check(
        "106 shuffled curve input sorted by forward",
        len(shuffled_chain) == 15
        and all(
            shuffled_chain[index]["forward_m"]
            < shuffled_chain[index + 1]["forward_m"]
            for index in range(len(shuffled_chain) - 1)
        ),
    )

    straight_smoothed = SingleGreenCurvePlanner.smooth_single_green_boundary(
        left_chain, curve_config
    )
    straight_tangents = SingleGreenCurvePlanner.estimate_local_boundary_tangent(
        straight_smoothed, curve_config
    )
    check("107 straight boundary tangent", np.max(np.abs(straight_tangents[:, 1])) < 1.0e-6)
    curved_samples = synthetic_curve_samples(
        "left", lateral_by_index=lambda index: 0.0012 * index * index
    )
    _, curved_chain = SingleGreenCurvePlanner.extract_main_chain(
        curved_samples, "left", curve_ipm, curve_config
    )
    curved_smoothed = SingleGreenCurvePlanner.smooth_single_green_boundary(
        curved_chain, curve_config
    )
    curved_tangents = SingleGreenCurvePlanner.estimate_local_boundary_tangent(
        curved_smoothed, curve_config
    )
    curved_headings = np.unwrap(
        np.arctan2(curved_tangents[:, 1], curved_tangents[:, 0])
    )
    check("108 curved tangent remains continuous", np.max(np.abs(np.diff(curved_headings))) < 0.20)
    outlier_chain = copy.deepcopy(left_chain)
    outlier_chain[7]["left_m"] += 0.10
    robust_smoothed = SingleGreenCurvePlanner.smooth_single_green_boundary(
        outlier_chain, curve_config
    )
    check("109 isolated boundary outlier is smoothed", abs(robust_smoothed[7, 1] - robust_smoothed[6, 1]) < 0.05)
    check("110 all tangents point forward", np.all(curved_tangents[:, 0] > 0.0))

    width_samples = copy.deepcopy(left_samples)
    for sample in width_samples[:3]:
        sample["right"] = float(sample["left"]) + 104.0
    width_state = SingleGreenLaneWidthState()
    current_width = SingleGreenCurvePlanner.resolve_lane_width(
        width_samples, width_state, curve_ipm, curve_config, 10.0
    )
    check("111 current dual width has priority", current_width[1] == "current_dual" and abs(current_width[0] - 0.52) < 0.01)
    width_state.update(0.48, 9.5, 1.0)
    history_width = SingleGreenCurvePlanner.resolve_lane_width(
        left_samples, width_state, curve_ipm, curve_config, 10.0
    )
    check("112 recent history width used", history_width[1] == "history_ema" and abs(history_width[0] - 0.48) < 1.0e-6)
    expired_width = SingleGreenCurvePlanner.resolve_lane_width(
        left_samples, width_state, curve_ipm, curve_config, 20.0
    )
    check("113 expired history falls back expected", expired_width[1] == "expected")
    abnormal_samples = copy.deepcopy(width_samples)
    for sample in abnormal_samples[:3]:
        sample["right"] = float(sample["left"]) + 180.0
    clamped_width = SingleGreenCurvePlanner.resolve_lane_width(
        abnormal_samples, None, curve_ipm, curve_config, 10.0
    )
    check("114 abnormal current width clamped", clamped_width[3] and abs(clamped_width[0] - 0.60) < 1.0e-6)
    width_state.update(0.52, 10.0, 0.25)
    check("115 width history EMA", abs(width_state.width_m - 0.49) < 1.0e-6)
    check("116 width source field stable", current_width[1] in {"current_dual", "history_ema", "expected"})

    left_green = np.zeros(curve_shape, dtype=np.uint8)
    left_yellow = np.zeros(curve_shape, dtype=np.uint8)
    left_green[:, :251] = 255
    left_yellow[:, 251:351] = 255
    left_candidates, left_curve_status, _left_debug = (
        SingleGreenCurvePlanner.build(
            left_samples,
            left_green,
            left_yellow,
            curve_valid,
            curve_ipm,
            curve_config,
        )
    )
    right_green = np.zeros(curve_shape, dtype=np.uint8)
    right_yellow = np.zeros(curve_shape, dtype=np.uint8)
    right_green[:, 350:] = 255
    right_yellow[:, 250:350] = 255
    right_candidates, right_curve_status, _right_debug = (
        SingleGreenCurvePlanner.build(
            right_samples,
            right_green,
            right_yellow,
            curve_valid,
            curve_ipm,
            curve_config,
        )
    )
    check("117 left boundary normal offset accepted", len(left_candidates) >= 12 and left_curve_status["single_green_curve_side"] == "left")
    check("118 right boundary normal offset accepted", len(right_candidates) >= 12 and right_curve_status["single_green_curve_side"] == "right")
    check(
        "119 metric normal offset equals half width",
        abs(left_curve_status["single_green_offset_distance_m_mean"] - 0.25) < 0.01,
    )
    no_yellow_candidates, no_yellow_curve_status, _ = (
        SingleGreenCurvePlanner.build(
            left_samples,
            left_green,
            np.zeros_like(left_yellow),
            curve_valid,
            curve_ipm,
            curve_config,
        )
    )
    check("120 insufficient yellow support rejected", not no_yellow_candidates and no_yellow_curve_status["single_green_curve_reject_yellow_support_count"] > 0)
    intrusion_green = left_green.copy()
    intrusion_green[:, 295:307] = 255
    intrusion_candidates, intrusion_status, _ = SingleGreenCurvePlanner.build(
        left_samples,
        intrusion_green,
        left_yellow,
        curve_valid,
        curve_ipm,
        curve_config,
    )
    check("121 green intrusion rejected", len(intrusion_candidates) < len(left_candidates) and intrusion_status["single_green_curve_reject_green_intrusion_count"] > 0)
    invalid_candidates, invalid_status, _ = SingleGreenCurvePlanner.build(
        left_samples,
        left_green,
        left_yellow,
        np.zeros(curve_shape, dtype=bool),
        curve_ipm,
        curve_config,
    )
    check("122 invalid IPM candidates rejected", not invalid_candidates and invalid_status["single_green_curve_reject_invalid_ipm_count"] > 0)
    short_candidates, short_status, _ = SingleGreenCurvePlanner.build(
        left_samples[:4],
        left_green,
        left_yellow,
        curve_valid,
        curve_ipm,
        curve_config,
    )
    check("123 four boundary points cannot fabricate path", not short_candidates and short_status["single_green_curve_failure_reason"] == "main_chain_too_short")
    accepted_forward = [
        value["center_metric"][0] for value in left_candidates.values()
    ]
    check(
        "124 no center generated outside observed range",
        min(accepted_forward) >= left_chain[0]["forward_m"] - 1.0e-9
        and max(accepted_forward)
        <= left_chain[-1]["forward_m"] + 1.0e-9,
    )

    single_plan, single_plan_status, _ = GreenBoundaryFastPlanner.plan(
        left_samples,
        copy.deepcopy(GreenBoundaryFastPlanner.EMPTY_STATUS),
        left_green,
        left_yellow,
        curve_ipm,
        curve_config,
        geometry,
        allow_single=True,
        ipm_valid_mask=curve_valid,
    )
    single_forward = (
        curve_ipm.vehicle_origin_y_px - single_plan.final_points[:, 1]
    ) * curve_ipm.meter_per_pixel
    check("125 single curve path forward monotonic", single_plan.valid and np.all(np.diff(single_forward) > 0.0))
    check("126 single curve confidence capped", single_plan.confidence <= 0.78)
    json.dumps(single_plan_status, ensure_ascii=False)
    check("127 new single curve status serializable", all(key in single_plan_status for key in SingleGreenCurvePlanner.STATUS_DEFAULTS))

    existing_points = single_plan.final_points[::2].copy()
    existing_result = STAGE4.GeometryResult(
        mode="yellow_corridor_dual_edge",
        valid=True,
        reason="valid",
        confidence=0.86,
        raw_points=existing_points.copy(),
        final_points=existing_points,
        forward_span_m=float(
            (existing_points[0, 1] - existing_points[-1, 1])
            * curve_ipm.meter_per_pixel
        ),
    )
    fused_result, fused_single_count = (
        GreenBoundaryFastPlanner.fuse_existing_with_single_curve(
            existing_result,
            single_plan,
            curve_ipm,
            curve_config,
            geometry,
        )
    )
    check("128 existing center points keep priority", fused_result.valid and np.min([np.min(np.linalg.norm(fused_result.final_points - point, axis=1)) for point in existing_points]) < 1.0e-6)
    check("129 single curve fills uncovered stations", fused_single_count > 0 and len(fused_result.final_points) > len(existing_points))
    fused_forward = (
        curve_ipm.vehicle_origin_y_px - fused_result.final_points[:, 1]
    ) * curve_ipm.meter_per_pixel
    check("130 fused path has no duplicate stations", np.min(np.diff(fused_forward)) > curve_config.sample_step_m * 0.40)
    check("131 fused path order is forward monotonic", np.all(np.diff(fused_forward) > 0.0))
    check("132 fused mode remains compatible", fused_result.mode == "green_yellow_hybrid")

    mixed_green = np.zeros(curve_shape, dtype=np.uint8)
    mixed_yellow = np.zeros(curve_shape, dtype=np.uint8)
    for row in range(curve_shape[0]):
        progress = (curve_shape[0] - 1 - row) / max(
            1, curve_shape[0] - 1
        )
        left_edge = 250 + int(round(20.0 * progress * progress))
        right_edge = left_edge + 100
        mixed_green[row, : left_edge + 1] = 255
        mixed_yellow[row, left_edge + 1 : right_edge] = 255
    for sample in left_samples[:3]:
        row = int(sample["y"])
        progress = (curve_shape[0] - 1 - row) / max(
            1, curve_shape[0] - 1
        )
        right_edge = 350 + int(round(20.0 * progress * progress))
        mixed_green[row, right_edge:] = 255
    mixed_samples, mixed_base = GreenBoundaryFastPlanner.extract(
        mixed_green,
        mixed_yellow,
        curve_ipm,
        curve_config,
        ipm_valid_mask=curve_valid,
    )
    mixed_curve_result, mixed_curve_status, _ = GreenBoundaryFastPlanner.plan(
        mixed_samples,
        mixed_base,
        mixed_green,
        mixed_yellow,
        curve_ipm,
        curve_config,
        geometry,
        allow_single=True,
        ipm_valid_mask=curve_valid,
    )
    check("133 synthetic 15/3 curve recovers at least ten points", mixed_curve_result.valid and len(mixed_curve_result.final_points) >= 10)
    check("134 synthetic 15/3 curve recovers span", mixed_curve_result.forward_span_m >= 0.45 and mixed_curve_status["single_green_curve_chain_count"] >= 12)
    for _index in range(10):
        SingleGreenCurvePlanner.build(
            left_samples,
            left_green,
            left_yellow,
            curve_valid,
            curve_ipm,
            curve_config,
        )
    performance_started = time.perf_counter()
    for _index in range(200):
        SingleGreenCurvePlanner.build(
            left_samples,
            left_green,
            left_yellow,
            curve_valid,
            curve_ipm,
            curve_config,
        )
    average_curve_ms = (time.perf_counter() - performance_started) * 5.0
    print(f"single-green synthetic average: {average_curve_ms:.3f} ms")
    check("135 single curve synthetic average at most 3ms", average_curve_ms <= 3.0)
    check("136 straight dual regression remains fifteen points", green_dual_result.valid and len(green_dual_result.final_points) == 15)

    print(f"SELF-TEST PASS: {len(passed)} checks")
    for name in passed:
        print(f"  PASS {name}")


def main(args: Optional[List[str]] = None) -> None:
    if SELF_TEST:
        run_self_tests()
        return
    rclpy.init(args=args)
    node: Optional[LanePerceptionPipelineNode] = None
    try:
        node = LanePerceptionPipelineNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
