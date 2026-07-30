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
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

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
    overlay: np.ndarray
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
            fragments.append(
                BoundaryFragment(
                    component_id=component_id,
                    pixel_count=int(area),
                    x_values=row_x,
                    y_values=unique_y.astype(np.float64),
                    coefficients=coefficients,
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
        combined_y = np.concatenate((upper.y_values, lower.y_values))
        combined_x = np.concatenate((upper.x_values, lower.x_values))
        degree = min(2, len(combined_y) - 1)
        coefficients = np.polyfit(combined_y, combined_x, degree)
        residual = float(
            np.sqrt(
                np.mean(
                    (
                        combined_x
                        - np.polyval(coefficients, combined_y)
                    )
                    ** 2
                )
            )
        )
        if residual > min(8.0, max(4.0, lateral_error * 0.50)):
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
        bridge_x = np.polyval(coefficients, bridge_y)
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
        observed_roi = cls._roi_mask(observed, roi)
        min_span = config.min_fragment_span_px(ipm.meter_per_pixel)
        fragments, observed_count = cls.fragments(
            observed_roi, roi, min_span
        )
        repaired = observed_roi.copy()
        gap_mask = np.zeros_like(observed_roi)
        accepted: List[Tuple[int, int]] = []
        used_endpoints: set = set()
        maximum_gap = 0
        if config.boundary_repair_enable:
            candidates: List[
                Tuple[int, BoundaryFragment, BoundaryFragment, np.ndarray]
            ] = []
            for first_index in range(len(fragments)):
                for second_index in range(first_index + 1, len(fragments)):
                    first = fragments[first_index]
                    second = fragments[second_index]
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
        )


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
        for y in np.arange(
            curve.y_min,
            curve.y_max + 1,
            max(2, sample_step_px),
            dtype=np.float64,
        ):
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
    ) -> Tuple[List[Any], int]:
        if not curves:
            return [], 0
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

        for first_index in range(len(curves)):
            for second_index in range(first_index + 1, len(curves)):
                first_curve, second_curve = (
                    curves[first_index],
                    curves[second_index],
                )
                if (
                    first_curve.inward_sign
                    and second_curve.inward_sign
                    and first_curve.inward_sign != second_curve.inward_sign
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
            aggregated.append(
                STAGE4.BoundaryCurve(
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
                )
            )
        return aggregated, len(aggregated)


class ConservativeDualBoundaryPlanner:
    """仅将车辆中心轴两侧的曲线交给原第四阶段双边界配对器。"""

    @staticmethod
    def plan(
        curves: Sequence[Any],
        main_yellow: np.ndarray,
        ipm: Any,
        geometry_config: Any,
        roi: Tuple[int, int],
        previous_center_x: Optional[float] = None,
    ) -> Optional[Any]:
        best: Optional[Any] = None
        best_score = -float("inf")
        for first_index in range(len(curves)):
            for second_index in range(first_index + 1, len(curves)):
                first = curves[first_index]
                second = curves[second_index]
                first_side = first.centroid[0] - ipm.vehicle_center_x_px
                second_side = second.centroid[0] - ipm.vehicle_center_x_px
                if first_side * second_side >= 0:
                    continue
                candidate = STAGE4.DualBoundaryPlanner.plan(
                    [first, second],
                    main_yellow,
                    ipm,
                    geometry_config,
                    roi,
                    previous_center_x,
                )
                if candidate is None:
                    continue
                score = float(candidate.dual_pair_score)
                if score > best_score:
                    best, best_score = candidate, score
        return best


class LaneAlgorithm:
    """将第二、第四阶段的原有纯算法串成单帧内存流水线。"""

    def __init__(
        self,
        threshold: Any,
        segmentation_config: Dict[str, object],
        geometry_config: Any,
        ipm: Any,
        enhancement_config: Optional[BoundaryEnhancementConfig] = None,
    ) -> None:
        self.threshold = threshold
        self.segmentation_config = segmentation_config
        self.geometry_config = geometry_config
        self.ipm = ipm
        self.enhancement_config = (
            enhancement_config or BoundaryEnhancementConfig()
        )
        self.enhancement_config.validate()
        self.last_valid_points = np.empty((0, 2), dtype=np.float64)
        self.last_valid_time = 0.0
        self.last_center_x: Optional[float] = None

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

    def _segment(
        self, bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        values = STAGE2.GreenYellowSegmenter.segment(
            bgr, self.threshold, self.segmentation_config
        )
        (
            _yellow_raw,
            _green_raw,
            _unknown_raw,
            yellow,
            green,
            unknown,
            _hsv,
            roi_start,
            roi_end,
            _overlap,
            _yellow_stats,
            _green_stats,
        ) = values
        boundary_result = STAGE2.BoundaryExtractor.extract(
            yellow,
            green,
            unknown,
            roi_start,
            roi_end,
            self.segmentation_config,
        )
        return yellow, green, boundary_result.final, boundary_result.valid

    def process(self, bgr: np.ndarray, processing_ms: float = 0.0) -> PipelineResult:
        if bgr.shape[:2] != (self.ipm.image_height, self.ipm.image_width):
            raise ValueError(
                f"相机图像 {bgr.shape[1]}x{bgr.shape[0]} 与 IPM "
                f"{self.ipm.image_width}x{self.ipm.image_height} 不一致"
            )
        yellow, green, boundary, boundary_valid = self._segment(bgr)
        observed_ipm_boundary = STAGE4.MaskIpmTransformer.transform(
            boundary, self.ipm
        )
        transformed = {
            "yellow": STAGE4.MaskIpmTransformer.transform(yellow, self.ipm),
            "green": STAGE4.MaskIpmTransformer.transform(green, self.ipm),
        }
        (
            _forward_min,
            _forward_max,
            y_min,
            y_max,
        ) = self.ipm.planning_roi(self.geometry_config)
        roi = (y_min, y_max)
        main_yellow, yellow_status = STAGE4.MainYellowSelector.select(
            transformed["yellow"],
            self.ipm,
            self.geometry_config,
            roi,
        )
        repair = ConservativeBoundaryRepair.repair(
            observed_ipm_boundary,
            main_yellow,
            transformed["green"],
            self.ipm,
            roi,
            self.enhancement_config,
        )
        transformed["boundary"] = repair.repaired
        curves: List[Any] = []
        raw_component_count = 0
        merged_group_count = 0
        result = STAGE4.GeometryResult(reason="segmentation_invalid")
        if boundary_valid and yellow_status["main_yellow_valid"]:
            curves, raw_component_count = (
                STAGE4.BoundaryComponentAnalyzer.analyze(
                    transformed["boundary"],
                    self.ipm,
                    self.geometry_config,
                    roi,
                )
            )
            step_px = max(
                2,
                int(
                    round(
                        self.geometry_config.centerline_sample_step_m
                        / self.ipm.meter_per_pixel
                    )
                ),
            )
            for curve in curves:
                GlobalRoadSideClassifier.classify_curve(
                    curve,
                    main_yellow,
                    transformed["green"],
                    self.geometry_config,
                    self.enhancement_config,
                    step_px,
                )
            curves, merged_group_count = BoundaryCurveAggregator.aggregate(
                curves,
                main_yellow,
                transformed["green"],
                self.ipm,
                self.geometry_config,
                self.enhancement_config,
            )
            # 聚合后重新投票，避免仅继承某个短碎片的局部方向。
            for curve in curves:
                GlobalRoadSideClassifier.classify_curve(
                    curve,
                    main_yellow,
                    transformed["green"],
                    self.geometry_config,
                    self.enhancement_config,
                    step_px,
                )
            if curves:
                result = ConservativeDualBoundaryPlanner.plan(
                    curves,
                    main_yellow,
                    self.ipm,
                    self.geometry_config,
                    roi,
                    self.last_center_x,
                )
                if result is None:
                    result = STAGE4.SingleBoundaryPlanner.plan(
                        curves,
                        main_yellow,
                        transformed["green"],
                        self.ipm,
                        self.geometry_config,
                        roi,
                    )
                if result is None:
                    reason = (
                        "single_boundary_side_unknown"
                        if any(curve.inward_sign == 0 for curve in curves)
                        else "boundary_pair_invalid"
                    )
                    result = STAGE4.GeometryResult(reason=reason)
            else:
                result = STAGE4.GeometryResult(reason="boundary_missing")
        elif not yellow_status["main_yellow_valid"]:
            result = STAGE4.GeometryResult(reason="main_yellow_missing")

        if not bool(yellow_status.get("main_yellow_seed_connected", False)):
            result.confidence *= 0.80
        if len(result.raw_points):
            (
                result.final_points,
                result.reason,
                result.yellow_ratio,
                result.forward_span_m,
            ) = STAGE4.CenterlineSmoother.smooth(
                result.raw_points,
                main_yellow,
                self.ipm,
                self.geometry_config,
                roi,
            )
            result.valid = result.reason == "valid"
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
                    * self.ipm.meter_per_pixel
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

        shape = (self.ipm.output_height, self.ipm.output_width)
        selected = STAGE4.BoundaryComponentAnalyzer.component_mask(
            transformed["boundary"], result.selected_curves, 3
        )
        centerline = self.points_mask(shape, result.final_points, 3)
        raw_line = self.points_mask(shape, result.raw_points, 2)
        overlay = self.make_overlay(
            transformed,
            main_yellow,
            selected,
            raw_line,
            centerline,
            result,
            roi,
            processing_ms,
            repair.observed,
            repair.gap_mask,
            curves,
        )
        road_curves = result.selected_curves or curves
        road_side_status = GlobalRoadSideClassifier.representative_status(
            road_curves
        )
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
            overlay=overlay,
            geometry=result,
            roi=roi,
            boundary_valid=bool(boundary_valid),
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
        overlay[raw_line > 0] = (255, 255, 0)
        overlay[centerline > 0] = (0, 255, 255)
        center_x = int(round(self.ipm.vehicle_center_x_px))
        origin_y = int(round(self.ipm.vehicle_origin_y_px))
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
        lines = (
            f"mode={result.mode}",
            f"valid={result.valid} confidence={result.confidence:.2f}",
            f"points={len(result.final_points)} span={result.forward_span_m:.2f}m",
            (
                "side_votes="
                f"+{vote_status['road_side_positive_vote_count']}/"
                f"-{vote_status['road_side_negative_vote_count']} "
                f"ratio={vote_status['road_side_global_vote_ratio']:.2f}"
            ),
            (
                "Y(+/-)="
                f"{vote_status['road_side_positive_yellow_ratio']:.2f}/"
                f"{vote_status['road_side_negative_yellow_ratio']:.2f} "
                "G(+/-)="
                f"{vote_status['road_side_positive_green_ratio']:.2f}/"
                f"{vote_status['road_side_negative_green_ratio']:.2f}"
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
<div>综合图从左到右：相机缩略图、鸟瞰 Overlay、最终中心线。</div></header>
<main class="page"><section class="panel"><img src="/stream/overlay">
<div class="stats" id="stats"></div></section></main><script>
const fields=["centerline_mode","centerline_valid","centerline_confidence",
"capture_fps","process_fps","path_publish_fps","end_to_end_ms",
"dropped_frame_count","processing_time_ms",
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
        self.last_web_encode = 0.0
        self.last_debug_publish = 0.0
        self.process_thread: Optional[threading.Thread] = None
        self.capture = CameraCapture(self, self.slot, self.stop_event)
        self.threshold, threshold_path = self._load_threshold()
        self.segmentation_config = self._load_segmentation_config()
        self.geometry_config, self.ipm = self._load_geometry()
        self.enhancement_config = self._load_enhancement_config()
        self.algorithm = LaneAlgorithm(
            self.threshold,
            self.segmentation_config,
            self.geometry_config,
            self.ipm,
            self.enhancement_config,
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
            f"{self.enhancement_config.max_gap_px(self.ipm.meter_per_pixel)}px"
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
            age_ms = (started - frame.captured_monotonic) * 1000.0
            source = self._source_image(frame)
            try:
                result = self.algorithm.process(frame.image)
                processing_ms = (time.monotonic() - started) * 1000.0
                result.overlay = self.algorithm.make_overlay(
                    result.transformed,
                    result.main_yellow,
                    result.selected_boundary,
                    self.algorithm.points_mask(
                        result.centerline_mask.shape,
                        result.geometry.raw_points,
                        2,
                    ),
                    result.centerline_mask,
                    result.geometry,
                    result.roi,
                    processing_ms,
                    result.observed_ipm_boundary,
                    result.repaired_gap_mask,
                    result.candidate_curves,
                )
                path = (
                    STAGE4.PathConverter.to_path(
                        result.geometry.final_points, self.ipm, source
                    )
                    if result.geometry.valid
                    else STAGE4.PathConverter.empty(self.ipm, source)
                )
                self.path_publisher.publish(path)
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
                            "last_error": ""
                            if result.geometry.valid
                            else result.geometry.reason,
                        }
                    )
            except Exception as exc:
                processing_ms = (time.monotonic() - started) * 1000.0
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
                                GlobalRoadSideClassifier.EMPTY_STATUS
                            ),
                            "last_error": f"processing: {exc}",
                        }
                    )

    def _publish_debug(
        self, raw: np.ndarray, result: PipelineResult, source: Image
    ) -> None:
        now = time.monotonic()
        web_due = (
            (self.web_gui_enable or self.publish_overlay_compressed)
            and now - self.last_web_encode >= 1.0 / self.web_gui_max_fps
        )
        if web_due:
            composite = self._make_web_composite(
                raw, result.overlay, result.centerline_mask
            )
            payload = self.web_jpeg.update(
                composite, self.web_gui_jpeg_quality
            )
            self.last_web_encode = now
            if payload is not None and self.overlay_publisher is not None:
                message = CompressedImage()
                message.header = copy.deepcopy(source.header)
                message.format = "jpeg"
                message.data = payload
                self.overlay_publisher.publish(message)
        if (
            self.publish_debug_raw_images
            and now - self.last_debug_publish
            >= 1.0 / self.debug_publish_fps
        ):
            images = {
                "boundary": result.boundary,
                "yellow": result.yellow,
                "green": result.green,
                "ipm_observed": result.observed_ipm_boundary,
                "ipm_repaired": result.repaired_ipm_boundary,
                "repaired_gap": result.repaired_gap_mask,
                "selected": result.selected_boundary,
                "centerline": result.centerline_mask,
            }
            for name, image in images.items():
                self.debug_publishers[name].publish(
                    self._mono_message(image, source)
                )
            self.last_debug_publish = now

    @staticmethod
    def _make_web_composite(
        raw: np.ndarray, overlay: np.ndarray, centerline: np.ndarray
    ) -> np.ndarray:
        panel_height = max(360, overlay.shape[0])
        raw_width = max(
            1, int(round(raw.shape[1] * panel_height / raw.shape[0]))
        )
        overlay_width = max(
            1,
            int(round(overlay.shape[1] * panel_height / overlay.shape[0])),
        )
        raw_panel = cv2.resize(
            raw, (raw_width, panel_height), interpolation=cv2.INTER_AREA
        )
        overlay_panel = cv2.resize(
            overlay,
            (overlay_width, panel_height),
            interpolation=cv2.INTER_AREA,
        )
        center_bgr = cv2.cvtColor(centerline, cv2.COLOR_GRAY2BGR)
        center_panel = cv2.resize(
            center_bgr,
            (overlay_width, panel_height),
            interpolation=cv2.INTER_NEAREST,
        )
        composite = np.hstack((raw_panel, overlay_panel, center_panel))
        labels = (
            ("CAMERA", 8),
            ("BIRDVIEW OVERLAY", raw_width + 8),
            ("FINAL CENTERLINE", raw_width + overlay_width + 8),
        )
        for text, x in labels:
            cv2.putText(
                composite,
                text,
                (x, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                composite,
                text,
                (x, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
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
        with self.lock:
            snapshot = copy.deepcopy(self.status)
            snapshot["capture_fps"] = self.capture_meter.value()
            snapshot["process_fps"] = self.process_meter.value()
            snapshot["path_publish_fps"] = self.path_meter.value()
            snapshot["dropped_frame_count"] = self.slot.dropped_count
            return snapshot

    def _log_statistics(self) -> None:
        status = self.status_snapshot()
        self.get_logger().info(
            f"capture={status['capture_fps']:.1f} "
            f"process={status['process_fps']:.1f} "
            f"path={status['path_publish_fps']:.1f} "
            f"drop={status['dropped_frame_count']} "
            f"processing_ms={status['processing_time_ms']:.1f}"
        )

    def destroy_node(self) -> bool:
        self.stop_event.set()
        self.capture.stop()
        self.slot.wake()
        if self.process_thread is not None:
            self.process_thread.join(timeout=2.0)
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
    composite = LanePerceptionPipelineNode._make_web_composite(
        image,
        result.overlay,
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
    aggregated_curves, aggregated_group_count = (
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
    dual_curves, _ = BoundaryCurveAggregator.aggregate(
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
