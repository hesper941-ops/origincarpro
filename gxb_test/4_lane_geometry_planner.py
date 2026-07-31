#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鸟瞰道路几何分析与中心线生成工具。

本程序只负责鸟瞰道路几何分析、当前道路边界选择和中心线生成，不包含速度控制、
转向控制和底盘运动功能。程序读取第三阶段生成的 ``gxb_ipm_v1`` JSON，并消费
第二阶段 ``gxb_boundary_v1`` 的同步黄、绿、分界线 mask；不会修改前三阶段配置。

启动示例：

    python3 gxb_test/4_lane_geometry_planner.py \
      --ros-args \
      -p profile_name:=usb_camera \
      -p ipm_config_path:=gxb_test/config/usb_camera_ipm.json \
      -p web_gui_enable:=true \
      -p web_gui_port:=8092 \
      -p history_fallback_enable:=false

浏览器访问 ``http://小车IP:8092``。第一轮实车测试只观察几何输出，不让车辆运动。
纯算法合成自测可在 ROS 环境中运行：

    python3 gxb_test/4_lane_geometry_planner.py --self-test
"""

import copy
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
import urllib.parse
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as RosPath
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import String


GEOMETRY_INTERFACE_VERSION = "gxb_geometry_v1"
GEOMETRY_CONFIG_VERSION = "gxb_geometry_config_v1"
IPM_INTERFACE_VERSION = "gxb_ipm_v1"
SEGMENTATION_INTERFACE_VERSION = "gxb_boundary_v1"
PROFILE_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
MASK_NAMES = ("boundary", "yellow", "green")

BOUNDARY_TOPIC = "/gxb_test/final_boundary_mask"
YELLOW_TOPIC = "/gxb_test/yellow_mask_final"
GREEN_TOPIC = "/gxb_test/green_mask_final"
SEGMENTATION_STATUS_TOPIC = "/gxb_test/segmentation_status"

OUTPUT_TOPICS = {
    "ipm_boundary": "/gxb_test/geometry/ipm_boundary_mask",
    "ipm_yellow": "/gxb_test/geometry/ipm_yellow_mask",
    "ipm_green": "/gxb_test/geometry/ipm_green_mask",
    "main_yellow": "/gxb_test/geometry/main_yellow_mask",
    "selected_boundary": "/gxb_test/geometry/selected_boundary_mask",
    "centerline_raw": "/gxb_test/geometry/centerline_raw_mask",
    "centerline": "/gxb_test/geometry/centerline_mask",
    "overlay": "/gxb_test/geometry/overlay",
    "path": "/gxb_test/geometry/centerline_path",
    "status": "/gxb_test/geometry/status",
}


class GeometryError(ValueError):
    """可安全呈现在状态和 Web 页面的几何错误。"""


@dataclass
class GeometryConfig:
    """第四阶段参数；带“待实车验证”的阈值均集中在这里。"""

    profile_name: str = "usb_camera"
    ipm_config_path: str = ""
    ipm_config_required: bool = True
    auto_reload_ipm_config: bool = False

    boundary_topic: str = BOUNDARY_TOPIC
    yellow_topic: str = YELLOW_TOPIC
    green_topic: str = GREEN_TOPIC
    segmentation_status_topic: str = SEGMENTATION_STATUS_TOPIC

    sync_cache_size: int = 6
    sync_max_age_sec: float = 0.50

    planning_forward_min_m_override: float = -1.0
    planning_forward_max_m_override: float = -1.0

    near_seed_forward_min_m: float = 0.30
    near_seed_forward_max_m: float = 0.45
    near_seed_half_width_m: float = 0.40
    main_yellow_min_area_px: int = 300
    main_yellow_min_vertical_span_px: int = 30

    boundary_component_min_pixels: int = 15
    boundary_component_min_vertical_span_px: int = 12
    boundary_component_max_area_ratio: float = 0.08
    boundary_fit_degree: int = 2
    boundary_fit_min_points: int = 8
    boundary_fit_max_residual_px: float = 8.0
    boundary_fit_outlier_iterations: int = 3

    lane_width_min_ratio: float = 0.70
    lane_width_max_ratio: float = 1.30

    road_side_probe_min_px: int = 4
    road_side_probe_max_px: int = 24
    road_side_probe_step_px: int = 4
    road_side_min_yellow_ratio: float = 0.35
    road_side_ratio_margin: float = 0.12

    centerline_sample_step_m: float = 0.05
    center_yellow_validation_radius_px: int = 8
    single_boundary_min_valid_ratio: float = 0.60
    single_boundary_min_samples: int = 6
    max_small_gap_forward_m: float = 0.10

    centerline_fit_degree: int = 2
    centerline_min_points: int = 6
    centerline_max_lateral_jump_m: float = 0.20
    centerline_max_abs_lateral_m: float = 1.20
    centerline_min_forward_span_m: float = 0.25
    centerline_min_yellow_ratio: float = 0.60

    history_fallback_enable: bool = False
    history_max_frames: int = 3
    history_max_age_sec: float = 0.30

    web_gui_enable: bool = True
    web_gui_host: str = "0.0.0.0"
    web_gui_port: int = 8092
    web_gui_jpeg_quality: int = 92
    web_gui_max_fps: float = 6.0

    def validate(self) -> None:
        """完整校验组合参数，非法值不能进入图像回调。"""
        if not PROFILE_PATTERN.fullmatch(self.profile_name):
            raise GeometryError("profile_name 只允许字母、数字、下划线和短横线")
        if not (1 <= self.sync_cache_size <= 30):
            raise GeometryError("sync_cache_size 必须位于 1 到 30")
        if not (0.05 <= self.sync_max_age_sec <= 5.0):
            raise GeometryError("sync_max_age_sec 必须位于 0.05 到 5.0")
        if (
            self.planning_forward_min_m_override >= 0
            and self.planning_forward_max_m_override >= 0
            and self.planning_forward_max_m_override
            <= self.planning_forward_min_m_override
        ):
            raise GeometryError("规划远端距离必须大于近端距离")
        if not (
            0 <= self.near_seed_forward_min_m
            < self.near_seed_forward_max_m
        ):
            raise GeometryError("近场黄色种子距离范围非法")
        positive = (
            "near_seed_half_width_m",
            "main_yellow_min_area_px",
            "main_yellow_min_vertical_span_px",
            "boundary_component_min_pixels",
            "boundary_component_min_vertical_span_px",
            "boundary_fit_min_points",
            "boundary_fit_max_residual_px",
            "boundary_fit_outlier_iterations",
            "road_side_probe_min_px",
            "road_side_probe_max_px",
            "road_side_probe_step_px",
            "centerline_sample_step_m",
            "center_yellow_validation_radius_px",
            "single_boundary_min_valid_ratio",
            "single_boundary_min_samples",
            "max_small_gap_forward_m",
            "centerline_min_points",
            "centerline_max_lateral_jump_m",
            "centerline_max_abs_lateral_m",
            "centerline_min_forward_span_m",
            "centerline_min_yellow_ratio",
        )
        for name in positive:
            if float(getattr(self, name)) <= 0:
                raise GeometryError(f"{name} 必须大于零")
        if self.road_side_probe_max_px < self.road_side_probe_min_px:
            raise GeometryError("道路侧探测最大距离不能小于最小距离")
        if not (1 <= self.boundary_fit_degree <= 3):
            raise GeometryError("boundary_fit_degree 必须位于 1 到 3")
        if not (1 <= self.centerline_fit_degree <= 3):
            raise GeometryError("centerline_fit_degree 必须位于 1 到 3")
        if not (0 < self.lane_width_min_ratio < self.lane_width_max_ratio):
            raise GeometryError("车道宽度比例范围非法")
        for name in (
            "boundary_component_max_area_ratio",
            "road_side_min_yellow_ratio",
            "road_side_ratio_margin",
            "single_boundary_min_valid_ratio",
            "centerline_min_yellow_ratio",
        ):
            value = float(getattr(self, name))
            if not (0 < value <= 1):
                raise GeometryError(f"{name} 必须位于 0 到 1")
        if not (1 <= self.web_gui_port <= 65535):
            raise GeometryError("Web 端口非法")
        if not (1 <= self.web_gui_jpeg_quality <= 100):
            raise GeometryError("JPEG 质量必须位于 1 到 100")
        if self.web_gui_max_fps <= 0:
            raise GeometryError("Web 最大帧率必须大于零")
        if self.history_max_frames < 1 or self.history_max_age_sec <= 0:
            raise GeometryError("历史帧数量和有效时间必须大于零")


@dataclass
class IpmConfig:
    """从第三阶段正式 JSON 提取出的稳定几何接口。"""

    path: str
    image_width: int
    image_height: int
    homography_matrix: np.ndarray
    inverse_homography_matrix: np.ndarray
    output_width: int
    output_height: int
    meter_per_pixel: float
    vehicle_center_x_px: float
    vehicle_origin_y_px: float
    vehicle_reference_point_name: str
    board_width_m: float
    board_length_m: float
    board_near_edge_distance_m: float
    board_center_lateral_offset_m: float
    expected_lane_width_m: float
    single_boundary_center_offset_m: float

    @property
    def expected_lane_width_px(self) -> float:
        return self.expected_lane_width_m / self.meter_per_pixel

    @property
    def single_boundary_offset_px(self) -> float:
        return self.single_boundary_center_offset_m / self.meter_per_pixel

    def planning_roi(
        self, config: GeometryConfig
    ) -> Tuple[float, float, int, int]:
        """由标定覆盖范围动态解析近远距离和鸟瞰 y 范围。"""
        forward_min = (
            config.planning_forward_min_m_override
            if config.planning_forward_min_m_override >= 0
            else self.board_near_edge_distance_m
        )
        forward_max = (
            config.planning_forward_max_m_override
            if config.planning_forward_max_m_override >= 0
            else self.board_near_edge_distance_m + self.board_length_m
        )
        if forward_min < 0 or forward_max <= forward_min:
            raise GeometryError("解析后的规划距离范围非法")
        y_near = int(
            round(self.vehicle_origin_y_px - forward_min / self.meter_per_pixel)
        )
        y_far = int(
            round(self.vehicle_origin_y_px - forward_max / self.meter_per_pixel)
        )
        y_min = max(0, min(self.output_height - 1, y_far))
        y_max = max(0, min(self.output_height - 1, y_near))
        if y_max <= y_min:
            raise GeometryError("规划 ROI 超出鸟瞰图或高度不足")
        return forward_min, forward_max, y_min, y_max


@dataclass
class MaskPacket:
    """一张已解码 mask 及其精确同步元数据。"""

    mask: np.ndarray
    message: Image
    arrival_time: float


@dataclass
class BoundaryCurve:
    """一个有效边界分量的统计、拟合和道路内侧判断。"""

    component_id: int
    pixel_count: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    vertical_span_px: int
    horizontal_span_px: int
    touches_near_band: bool
    distance_to_vehicle_center_px: float
    coefficients: np.ndarray
    fit_residual_px: float
    y_min: int
    y_max: int
    inward_sign: int = 0
    road_side: str = "unknown"
    side_confidence: float = 0.0

    def x_at(self, y: np.ndarray) -> np.ndarray:
        return np.polyval(self.coefficients, y)

    def dx_dy(self, y: float) -> float:
        derivative = np.polyder(self.coefficients)
        return float(np.polyval(derivative, y))

    def as_status(self) -> Dict[str, Any]:
        return {
            "id": self.component_id,
            "pixel_count": self.pixel_count,
            "bbox": list(self.bbox),
            "centroid": list(self.centroid),
            "vertical_span_px": self.vertical_span_px,
            "horizontal_span_px": self.horizontal_span_px,
            "touches_near_band": self.touches_near_band,
            "distance_to_vehicle_center_px": self.distance_to_vehicle_center_px,
            "fit_residual_px": self.fit_residual_px,
            "road_side": self.road_side,
            "side_confidence": self.side_confidence,
        }


@dataclass
class GeometryResult:
    """一帧规划的纯算法结果。"""

    mode: str = "invalid"
    valid: bool = False
    reason: str = "processing_error"
    confidence: float = 0.0
    raw_points: np.ndarray = None  # type: ignore[assignment]
    final_points: np.ndarray = None  # type: ignore[assignment]
    selected_curves: List[BoundaryCurve] = None  # type: ignore[assignment]
    selected_boundary_mask: np.ndarray = None  # type: ignore[assignment]
    measured_width_mean_px: float = 0.0
    measured_width_std_px: float = 0.0
    dual_overlap_count: int = 0
    dual_pair_score: float = 0.0
    single_road_side: str = "unknown"
    single_valid_ratio: float = 0.0
    yellow_ratio: float = 0.0
    forward_span_m: float = 0.0

    def __post_init__(self) -> None:
        if self.raw_points is None:
            self.raw_points = np.empty((0, 2), dtype=np.float64)
        if self.final_points is None:
            self.final_points = np.empty((0, 2), dtype=np.float64)
        if self.selected_curves is None:
            self.selected_curves = []


class ImageCodec:
    """不依赖图像桥接库的 mono8/8UC1 解码和 ROS Image 封装。"""

    @staticmethod
    def to_binary_mask(message: Image) -> np.ndarray:
        encoding = str(message.encoding).lower()
        if encoding not in ("mono8", "8uc1"):
            raise GeometryError(f"mask 编码必须为 mono8/8UC1，收到 {message.encoding}")
        width, height, step = (
            int(message.width),
            int(message.height),
            int(message.step),
        )
        if width <= 0 or height <= 0 or step < width:
            raise GeometryError(
                f"非法 mask 尺寸或 step: {width}x{height}, step={step}"
            )
        raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
        if raw.size < step * height:
            raise GeometryError("mask 数据长度不足")
        image = raw[: step * height].reshape(height, step)[:, :width]
        return np.where(image > 127, 255, 0).astype(np.uint8)

    @staticmethod
    def to_message(
        image: np.ndarray, encoding: str, source: Image, frame_id: str
    ) -> Image:
        output = Image()
        output.header = copy.deepcopy(source.header)
        output.header.frame_id = frame_id
        contiguous = np.ascontiguousarray(image)
        output.height, output.width = contiguous.shape[:2]
        output.encoding = encoding
        output.is_bigendian = 0
        output.step = output.width * (
            1 if contiguous.ndim == 2 else contiguous.shape[2]
        )
        output.data = contiguous.tobytes()
        return output


class ExactStampSynchronizer:
    """三路小容量精确时间戳缓存，不依赖额外同步包。"""

    def __init__(self, cache_size: int, max_age_sec: float) -> None:
        self.cache_size = cache_size
        self.max_age_sec = max_age_sec
        self.caches: Dict[
            str, "OrderedDict[Tuple[int, int, str], MaskPacket]"
        ] = {name: OrderedDict() for name in MASK_NAMES}
        self.processed_limit = cache_size * 4
        self.processed: Deque[Tuple[int, int, str]] = deque()
        self.processed_set: set = set()
        self.drop_count = 0

    @staticmethod
    def key(message: Image) -> Tuple[int, int, str]:
        return (
            int(message.header.stamp.sec),
            int(message.header.stamp.nanosec),
            str(message.header.frame_id),
        )

    def add(
        self, name: str, packet: MaskPacket
    ) -> Optional[Tuple[Tuple[int, int, str], Dict[str, MaskPacket]]]:
        if name not in self.caches:
            raise GeometryError(f"未知同步通道: {name}")
        key = self.key(packet.message)
        if not key[2]:
            self.drop_count += 1
            raise GeometryError("mask frame_id 为空")
        if key in self.processed_set:
            return None
        cache = self.caches[name]
        cache[key] = packet
        cache.move_to_end(key)
        while len(cache) > self.cache_size:
            cache.popitem(last=False)
            self.drop_count += 1
        self.cleanup(packet.arrival_time)
        if not all(key in self.caches[channel] for channel in MASK_NAMES):
            return None
        group = {
            channel: self.caches[channel].pop(key) for channel in MASK_NAMES
        }
        if len(self.processed) >= self.processed_limit:
            oldest = self.processed.popleft()
            self.processed_set.discard(oldest)
        self.processed.append(key)
        self.processed_set.add(key)
        return key, group

    def cleanup(self, now: float) -> int:
        removed = 0
        for cache in self.caches.values():
            for key, packet in list(cache.items()):
                if now - packet.arrival_time > self.max_age_sec:
                    cache.pop(key, None)
                    removed += 1
        self.drop_count += removed
        return removed

    def sizes(self) -> Dict[str, int]:
        return {name: len(cache) for name, cache in self.caches.items()}


class IpmConfigLoader:
    """严格读取第三阶段 JSON，不写入或修改该文件。"""

    def __init__(self, script_path: Path) -> None:
        self.config_dir = script_path.resolve().parent / "config"

    def resolve_path(self, config: GeometryConfig) -> Path:
        if config.ipm_config_path:
            return Path(config.ipm_config_path).expanduser().resolve()
        if not PROFILE_PATTERN.fullmatch(config.profile_name):
            raise GeometryError("非法 profile_name")
        return self.config_dir / f"{config.profile_name}_ipm.json"

    def load(self, config: GeometryConfig) -> IpmConfig:
        path = self.resolve_path(config)
        if not path.exists():
            raise GeometryError(f"IPM CONFIG MISSING: {path}")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise GeometryError(f"IPM CONFIG INVALID: {exc}") from exc
        if document.get("version") != IPM_INTERFACE_VERSION:
            raise GeometryError(
                f"IPM 接口版本应为 {IPM_INTERFACE_VERSION}，收到 {document.get('version')}"
            )
        try:
            output = document["output"]
            board = document["calibration_board"]
            road = document["road_geometry_defaults"]
            convention = document["coordinate_convention"]
            matrix = np.asarray(document["homography_matrix"], dtype=np.float64)
            inverse = np.asarray(
                document["inverse_homography_matrix"], dtype=np.float64
            )
            loaded = IpmConfig(
                path=str(path),
                image_width=int(document["image_width"]),
                image_height=int(document["image_height"]),
                homography_matrix=matrix,
                inverse_homography_matrix=inverse,
                output_width=int(output["width_px"]),
                output_height=int(output["height_px"]),
                meter_per_pixel=float(output["meter_per_pixel"]),
                vehicle_center_x_px=float(output["vehicle_center_x_px"]),
                vehicle_origin_y_px=float(output["vehicle_origin_y_px"]),
                vehicle_reference_point_name=str(
                    convention["vehicle_reference_point_name"]
                ),
                board_width_m=float(board["width_m"]),
                board_length_m=float(board["length_m"]),
                board_near_edge_distance_m=float(
                    board["near_edge_distance_m"]
                ),
                board_center_lateral_offset_m=float(
                    board["center_lateral_offset_m"]
                ),
                expected_lane_width_m=float(road["expected_lane_width_m"]),
                single_boundary_center_offset_m=float(
                    road["single_boundary_center_offset_m"]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GeometryError(f"IPM CONFIG INVALID: 缺少或非法字段 {exc}") from exc
        if matrix.shape != (3, 3) or inverse.shape != (3, 3):
            raise GeometryError("IPM CONFIG INVALID: 矩阵必须为 3x3")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(inverse)):
            raise GeometryError("IPM CONFIG INVALID: 矩阵包含 NaN/Inf")
        if abs(float(np.linalg.det(matrix))) < 1.0e-12:
            raise GeometryError("IPM CONFIG INVALID: Homography 奇异")
        if (
            loaded.image_width <= 0
            or loaded.image_height <= 0
            or loaded.output_width <= 0
            or loaded.output_height <= 0
            or loaded.meter_per_pixel <= 0
            or not loaded.vehicle_reference_point_name
            or not (0 <= loaded.vehicle_center_x_px < loaded.output_width)
            or not (0 <= loaded.vehicle_origin_y_px < loaded.output_height)
            or loaded.board_width_m <= 0
            or loaded.board_length_m <= 0
            or loaded.board_near_edge_distance_m < 0
            or loaded.expected_lane_width_m <= 0
            or loaded.single_boundary_center_offset_m <= 0
        ):
            raise GeometryError("IPM CONFIG INVALID: 尺寸、比例或坐标语义非法")
        loaded.planning_roi(config)
        return loaded


class MaskIpmTransformer:
    """使用最近邻插值变换二值 mask，并强制恢复 0/255。"""

    @staticmethod
    def transform(mask: np.ndarray, ipm: IpmConfig) -> np.ndarray:
        warped = cv2.warpPerspective(
            mask,
            ipm.homography_matrix,
            (ipm.output_width, ipm.output_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return np.where(warped > 127, 255, 0).astype(np.uint8)


def validate_mask_group_size(
    group: Dict[str, MaskPacket], ipm: IpmConfig
) -> None:
    """同步组中任一分辨率不匹配时整组拒绝。"""
    expected = (ipm.image_height, ipm.image_width)
    for name, packet in group.items():
        if packet.mask.shape != expected:
            raise GeometryError(
                f"{name} mask 尺寸 {packet.mask.shape[::-1]} 与 IPM "
                f"{ipm.image_width}x{ipm.image_height} 不一致"
            )


class MainYellowSelector:
    """从规划 ROI 中选择与车辆近场最相关的黄色连通域。"""

    @staticmethod
    def select(
        yellow: np.ndarray,
        ipm: IpmConfig,
        config: GeometryConfig,
        roi: Tuple[int, int],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        y_min, y_max = roi
        roi_mask = np.zeros_like(yellow)
        roi_mask[y_min : y_max + 1] = yellow[y_min : y_max + 1]
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (roi_mask > 0).astype(np.uint8), connectivity=8
        )
        seed_y_near = int(
            round(
                ipm.vehicle_origin_y_px
                - config.near_seed_forward_min_m / ipm.meter_per_pixel
            )
        )
        seed_y_far = int(
            round(
                ipm.vehicle_origin_y_px
                - config.near_seed_forward_max_m / ipm.meter_per_pixel
            )
        )
        seed_y0 = max(y_min, min(seed_y_near, seed_y_far))
        seed_y1 = min(y_max, max(seed_y_near, seed_y_far))
        half_width = config.near_seed_half_width_m / ipm.meter_per_pixel
        seed_x0 = max(0, int(round(ipm.vehicle_center_x_px - half_width)))
        seed_x1 = min(
            yellow.shape[1] - 1,
            int(round(ipm.vehicle_center_x_px + half_width)),
        )
        candidates: List[Tuple[float, int, bool]] = []
        valid_count = 0
        for component_id in range(1, count):
            x, y, width, height, area = stats[component_id]
            if (
                int(area) < config.main_yellow_min_area_px
                or int(height) < config.main_yellow_min_vertical_span_px
            ):
                continue
            valid_count += 1
            seed_connected = bool(
                np.any(
                    labels[seed_y0 : seed_y1 + 1, seed_x0 : seed_x1 + 1]
                    == component_id
                )
            )
            center_distance = abs(
                float(centroids[component_id, 0])
                - ipm.vehicle_center_x_px
            )
            score = (
                (1000000.0 if seed_connected else 0.0)
                + float(area) * 2.0
                + float(height) * 100.0
                - center_distance * 20.0
            )
            candidates.append((score, component_id, seed_connected))
        result = np.zeros_like(yellow)
        if not candidates:
            return result, {
                "yellow_component_count": valid_count,
                "main_yellow_component_id": -1,
                "main_yellow_pixels": 0,
                "main_yellow_seed_connected": False,
                "main_yellow_valid": False,
            }
        _, selected_id, seed_connected = max(candidates, key=lambda item: item[0])
        result[labels == selected_id] = 255
        return result, {
            "yellow_component_count": valid_count,
            "main_yellow_component_id": int(selected_id),
            "main_yellow_pixels": int(np.count_nonzero(result)),
            "main_yellow_seed_connected": bool(seed_connected),
            "main_yellow_valid": True,
        }


class RobustPolynomialFitter:
    """仅使用 NumPy 的 MAD 迭代稳健多项式拟合。"""

    @staticmethod
    def fit(
        y: np.ndarray,
        x: np.ndarray,
        degree: int,
        min_points: int,
        max_residual: float,
        iterations: int,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        y = np.asarray(y, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        y, x = y[finite], x[finite]
        if len(y) < max(2, min_points):
            raise GeometryError("拟合点数不足")
        keep = np.ones(len(y), dtype=bool)
        coefficients = np.polyfit(y, x, min(degree, len(y) - 1))
        for _ in range(max(1, iterations)):
            if np.count_nonzero(keep) < max(2, min_points):
                break
            coefficients = np.polyfit(
                y[keep], x[keep], min(degree, np.count_nonzero(keep) - 1)
            )
            residuals = np.abs(x - np.polyval(coefficients, y))
            median = float(np.median(residuals[keep]))
            mad = float(np.median(np.abs(residuals[keep] - median)))
            robust_limit = median + 3.5 * max(1.4826 * mad, 0.5)
            limit = max(max_residual, robust_limit)
            new_keep = residuals <= limit
            if np.array_equal(new_keep, keep):
                break
            if np.count_nonzero(new_keep) < max(2, min_points):
                break
            keep = new_keep
        coefficients = np.polyfit(
            y[keep], x[keep], min(degree, np.count_nonzero(keep) - 1)
        )
        residual = float(
            np.sqrt(
                np.mean(
                    (
                        x[keep]
                        - np.polyval(coefficients, y[keep])
                    )
                    ** 2
                )
            )
        )
        return coefficients, keep, residual


class BoundaryComponentAnalyzer:
    """在规划 ROI 内统计边界分量并稳健拟合 x=f(y)。"""

    @staticmethod
    def analyze(
        boundary: np.ndarray,
        ipm: IpmConfig,
        config: GeometryConfig,
        roi: Tuple[int, int],
    ) -> Tuple[List[BoundaryCurve], int]:
        y_min, y_max = roi
        roi_binary = np.zeros_like(boundary)
        roi_binary[y_min : y_max + 1] = boundary[y_min : y_max + 1]
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (roi_binary > 0).astype(np.uint8), connectivity=8
        )
        max_area = (
            (y_max - y_min + 1)
            * boundary.shape[1]
            * config.boundary_component_max_area_ratio
        )
        curves: List[BoundaryCurve] = []
        near_band_start = max(y_min, y_max - max(8, (y_max - y_min) // 5))
        for component_id in range(1, count):
            x0, y0, width, height, area = stats[component_id]
            if (
                int(area) < config.boundary_component_min_pixels
                or int(height) < config.boundary_component_min_vertical_span_px
                or float(area) > max_area
            ):
                continue
            ys, xs = np.where(labels == component_id)
            unique_y = np.unique(ys)
            row_x = np.asarray(
                [np.median(xs[ys == row]) for row in unique_y],
                dtype=np.float64,
            )
            try:
                coefficients, _, residual = RobustPolynomialFitter.fit(
                    unique_y,
                    row_x,
                    config.boundary_fit_degree,
                    config.boundary_fit_min_points,
                    config.boundary_fit_max_residual_px,
                    config.boundary_fit_outlier_iterations,
                )
            except GeometryError:
                continue
            if residual > config.boundary_fit_max_residual_px:
                continue
            touches_near = bool(np.any(ys >= near_band_start))
            curves.append(
                BoundaryCurve(
                    component_id=component_id,
                    pixel_count=int(area),
                    bbox=(int(x0), int(y0), int(width), int(height)),
                    centroid=(
                        float(centroids[component_id, 0]),
                        float(centroids[component_id, 1]),
                    ),
                    vertical_span_px=int(height),
                    horizontal_span_px=int(width),
                    touches_near_band=touches_near,
                    distance_to_vehicle_center_px=abs(
                        float(centroids[component_id, 0])
                        - ipm.vehicle_center_x_px
                    ),
                    coefficients=coefficients,
                    fit_residual_px=residual,
                    y_min=int(np.min(unique_y)),
                    y_max=int(np.max(unique_y)),
                )
            )
        return curves, count - 1

    @staticmethod
    def component_mask(
        boundary: np.ndarray, curves: Sequence[BoundaryCurve], thickness: int = 2
    ) -> np.ndarray:
        result = np.zeros_like(boundary)
        for curve in curves:
            ys = np.arange(curve.y_min, curve.y_max + 1, dtype=np.float64)
            xs = curve.x_at(ys)
            points = np.column_stack((xs, ys)).round().astype(np.int32)
            valid = (
                (points[:, 0] >= 0)
                & (points[:, 0] < result.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < result.shape[0])
            )
            if np.any(valid):
                cv2.polylines(
                    result, [points[valid]], False, 255, thickness, cv2.LINE_8
                )
        return result


class BoundaryRoadSideClassifier:
    """沿局部法线两侧采样黄绿 mask，判断黄色道路内部方向。"""

    @staticmethod
    def classify_at(
        curve: BoundaryCurve,
        y: float,
        yellow: np.ndarray,
        green: np.ndarray,
        config: GeometryConfig,
    ) -> Tuple[int, float, float, float]:
        x = float(curve.x_at(np.asarray([y]))[0])
        slope = curve.dx_dy(y)
        normal = np.asarray([1.0, -slope], dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1.0e-9)
        distances = range(
            config.road_side_probe_min_px,
            config.road_side_probe_max_px + 1,
            config.road_side_probe_step_px,
        )
        scores: List[float] = []
        yellow_ratios: List[float] = []
        for sign in (1, -1):
            yellow_hits = 0
            green_hits = 0
            valid = 0
            for distance in distances:
                px = int(round(x + sign * normal[0] * distance))
                py = int(round(y + sign * normal[1] * distance))
                if 0 <= px < yellow.shape[1] and 0 <= py < yellow.shape[0]:
                    valid += 1
                    yellow_hits += int(yellow[py, px] > 0)
                    green_hits += int(green[py, px] > 0)
            ratio_y = yellow_hits / max(1, valid)
            ratio_g = green_hits / max(1, valid)
            yellow_ratios.append(ratio_y)
            scores.append(ratio_y - 0.35 * ratio_g)
        best = int(np.argmax(scores))
        margin = scores[best] - scores[1 - best]
        if (
            yellow_ratios[best] < config.road_side_min_yellow_ratio
            or margin < config.road_side_ratio_margin
        ):
            return 0, margin, yellow_ratios[0], yellow_ratios[1]
        return (1 if best == 0 else -1), margin, yellow_ratios[0], yellow_ratios[1]

    @classmethod
    def classify_curve(
        cls,
        curve: BoundaryCurve,
        yellow: np.ndarray,
        green: np.ndarray,
        config: GeometryConfig,
        sample_step_px: int,
    ) -> BoundaryCurve:
        signs: List[int] = []
        margins: List[float] = []
        for y in np.arange(
            curve.y_min, curve.y_max + 1, max(2, sample_step_px)
        ):
            sign, margin, _, _ = cls.classify_at(
                curve, float(y), yellow, green, config
            )
            if sign:
                signs.append(sign)
                margins.append(max(0.0, margin))
        if not signs:
            curve.inward_sign = 0
            curve.road_side = "unknown"
            curve.side_confidence = 0.0
            return curve
        positive = sum(sign > 0 for sign in signs)
        negative = sum(sign < 0 for sign in signs)
        majority = 1 if positive > negative else -1 if negative > positive else 0
        agreement = max(positive, negative) / len(signs)
        curve.inward_sign = majority if agreement >= 0.60 else 0
        curve.road_side = (
            "yellow_right"
            if curve.inward_sign > 0
            else "yellow_left"
            if curve.inward_sign < 0
            else "mixed"
        )
        curve.side_confidence = agreement * min(
            1.0, float(np.mean(margins)) / 0.5
        )
        return curve


class DualBoundaryPlanner:
    """为左右边界配对并以中点生成双边界 raw centerline。"""

    @staticmethod
    def plan(
        curves: Sequence[BoundaryCurve],
        main_yellow: np.ndarray,
        ipm: IpmConfig,
        config: GeometryConfig,
        roi: Tuple[int, int],
        previous_center_x: Optional[float] = None,
    ) -> Optional[GeometryResult]:
        if len(curves) < 2:
            return None
        step_px = max(
            2, int(round(config.centerline_sample_step_m / ipm.meter_per_pixel))
        )
        expected = ipm.expected_lane_width_px
        best: Optional[Tuple[float, GeometryResult]] = None
        for first_index in range(len(curves)):
            for second_index in range(first_index + 1, len(curves)):
                first, second = curves[first_index], curves[second_index]
                common_min = max(roi[0], first.y_min, second.y_min)
                common_max = min(roi[1], first.y_max, second.y_max)
                if common_max - common_min < step_px * max(
                    2, config.centerline_min_points - 1
                ):
                    continue
                ys = np.arange(common_max, common_min - 1, -step_px, dtype=float)
                x_first, x_second = first.x_at(ys), second.x_at(ys)
                if np.mean(x_first) <= np.mean(x_second):
                    left, right = first, second
                    x_left, x_right = x_first, x_second
                else:
                    left, right = second, first
                    x_left, x_right = x_second, x_first
                # 左边界道路内部应总体向右，右边界总体向左。
                if left.inward_sign <= 0 or right.inward_sign >= 0:
                    continue
                widths = x_right - x_left
                width_valid = (
                    (widths >= expected * config.lane_width_min_ratio)
                    & (widths <= expected * config.lane_width_max_ratio)
                )
                valid_indices: List[int] = []
                between_ratios: List[float] = []
                center_hits: List[float] = []
                for index, y in enumerate(ys):
                    if not width_valid[index]:
                        continue
                    x0 = max(0, int(round(x_left[index])))
                    x1 = min(
                        main_yellow.shape[1] - 1,
                        int(round(x_right[index])),
                    )
                    yi = int(round(y))
                    if x1 <= x0 or not (0 <= yi < main_yellow.shape[0]):
                        continue
                    ratio = float(np.mean(main_yellow[yi, x0 : x1 + 1] > 0))
                    center_x = int(round((x_left[index] + x_right[index]) / 2))
                    center_hit = float(
                        0 <= center_x < main_yellow.shape[1]
                        and main_yellow[yi, center_x] > 0
                    )
                    if ratio >= 0.50 and center_hit > 0:
                        valid_indices.append(index)
                        between_ratios.append(ratio)
                        center_hits.append(center_hit)
                if len(valid_indices) < config.centerline_min_points:
                    continue
                widths_valid = widths[valid_indices]
                center_x = (
                    x_left[valid_indices] + x_right[valid_indices]
                ) / 2.0
                raw = np.column_stack((center_x, ys[valid_indices]))
                width_score = max(
                    0.0,
                    1.0
                    - abs(float(np.mean(widths_valid)) - expected)
                    / max(expected * 0.30, 1.0),
                )
                yellow_score = float(np.mean(between_ratios))
                overlap_score = min(
                    1.0,
                    len(valid_indices)
                    / max(config.centerline_min_points * 2.0, 1.0),
                )
                continuity = (
                    1.0
                    if previous_center_x is None
                    else max(
                        0.0,
                        1.0
                        - abs(float(raw[0, 0]) - previous_center_x)
                        / max(expected, 1.0),
                    )
                )
                near_score = float(
                    left.touches_near_band and right.touches_near_band
                )
                score = (
                    0.32 * width_score
                    + 0.28 * yellow_score
                    + 0.18 * overlap_score
                    + 0.12 * continuity
                    + 0.10 * near_score
                )
                result = GeometryResult(
                    mode="dual_boundary_midpoint",
                    valid=False,
                    reason="centerline_too_few_points",
                    confidence=min(1.0, 0.78 + 0.22 * score),
                    raw_points=raw,
                    selected_curves=[left, right],
                    measured_width_mean_px=float(np.mean(widths_valid)),
                    measured_width_std_px=float(np.std(widths_valid)),
                    dual_overlap_count=len(valid_indices),
                    dual_pair_score=score,
                )
                if best is None or score > best[0]:
                    best = score, result
        return None if best is None else best[1]


class SingleBoundaryPlanner:
    """沿局部法线朝黄色道路内部偏移，而非固定水平平移。"""

    @staticmethod
    def _near_yellow(
        x: float,
        y: float,
        yellow: np.ndarray,
        distance_to_yellow: np.ndarray,
        radius: int,
    ) -> bool:
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < yellow.shape[1] and 0 <= yi < yellow.shape[0]):
            return False
        return bool(yellow[yi, xi] > 0 or distance_to_yellow[yi, xi] <= radius)

    @classmethod
    def plan(
        cls,
        curves: Sequence[BoundaryCurve],
        main_yellow: np.ndarray,
        green: np.ndarray,
        ipm: IpmConfig,
        config: GeometryConfig,
        roi: Tuple[int, int],
    ) -> Optional[GeometryResult]:
        reliable = [curve for curve in curves if curve.inward_sign != 0]
        if not reliable:
            return None
        curve = max(
            reliable,
            key=lambda item: (
                int(item.touches_near_band),
                item.side_confidence,
                item.vertical_span_px,
                -item.distance_to_vehicle_center_px,
            ),
        )
        step_px = max(
            2, int(round(config.centerline_sample_step_m / ipm.meter_per_pixel))
        )
        y_min, y_max = max(roi[0], curve.y_min), min(roi[1], curve.y_max)
        ys = np.arange(y_max, y_min - 1, -step_px, dtype=float)
        outside = np.where(main_yellow > 0, 0, 255).astype(np.uint8)
        distance_to_yellow = cv2.distanceTransform(outside, cv2.DIST_L2, 3)
        points: List[Tuple[float, float]] = []
        side_votes: List[int] = []
        attempted = 0
        for y in ys:
            sign, _, _, _ = BoundaryRoadSideClassifier.classify_at(
                curve, y, main_yellow, green, config
            )
            if sign == 0:
                sign = curve.inward_sign
            if sign == 0:
                continue
            attempted += 1
            x = float(curve.x_at(np.asarray([y]))[0])
            slope = curve.dx_dy(y)
            normal = np.asarray([1.0, -slope], dtype=np.float64)
            normal /= max(float(np.linalg.norm(normal)), 1.0e-9)
            center = np.asarray([x, y]) + (
                sign * normal * ipm.single_boundary_offset_px
            )
            if (
                roi[0] <= center[1] <= roi[1]
                and cls._near_yellow(
                    center[0],
                    center[1],
                    main_yellow,
                    distance_to_yellow,
                    config.center_yellow_validation_radius_px,
                )
            ):
                points.append((float(center[0]), float(center[1])))
                side_votes.append(sign)
        valid_ratio = len(points) / max(1, attempted)
        if (
            len(points) < config.single_boundary_min_samples
            or valid_ratio < config.single_boundary_min_valid_ratio
        ):
            return GeometryResult(
                mode="single_boundary_normal_offset",
                valid=False,
                reason="single_boundary_side_unknown"
                if curve.inward_sign == 0
                else "centerline_too_few_points",
                selected_curves=[curve],
                raw_points=np.asarray(points, dtype=np.float64).reshape(-1, 2),
                single_road_side=curve.road_side,
                single_valid_ratio=valid_ratio,
            )
        confidence = min(
            0.85,
            0.50
            + 0.20 * valid_ratio
            + 0.15 * curve.side_confidence,
        )
        return GeometryResult(
            mode="single_boundary_normal_offset",
            valid=False,
            reason="centerline_too_few_points",
            confidence=confidence,
            selected_curves=[curve],
            raw_points=np.asarray(points, dtype=np.float64),
            single_road_side=curve.road_side,
            single_valid_ratio=valid_ratio,
        )


class CenterlineSmoother:
    """过滤、拟合、重采样并验证最终中心线。"""

    @staticmethod
    def smooth(
        raw_points: np.ndarray,
        main_yellow: np.ndarray,
        ipm: IpmConfig,
        config: GeometryConfig,
        roi: Tuple[int, int],
    ) -> Tuple[np.ndarray, str, float, float]:
        points = np.asarray(raw_points, dtype=np.float64).reshape(-1, 2)
        if len(points) < config.centerline_min_points:
            return np.empty((0, 2)), "centerline_too_few_points", 0.0, 0.0
        points = points[np.isfinite(points).all(axis=1)]
        # 同一 y 只保留横向中值，然后按近到远（y 递减）排列。
        grouped: List[Tuple[float, float]] = []
        for y in sorted(set(np.round(points[:, 1]).astype(int)), reverse=True):
            grouped.append((float(np.median(points[np.round(points[:, 1]) == y, 0])), float(y)))
        points = np.asarray(grouped, dtype=np.float64)
        # 超过允许的小缺口不做长距离补线，只保留最长连续采样段。
        max_gap_px = config.max_small_gap_forward_m / ipm.meter_per_pixel
        split_indices = np.where(np.abs(np.diff(points[:, 1])) > max_gap_px)[0] + 1
        segments = np.split(points, split_indices)
        points = max(segments, key=len)
        if len(points) < config.centerline_min_points:
            return np.empty((0, 2)), "centerline_too_few_points", 0.0, 0.0
        max_jump_px = config.centerline_max_lateral_jump_m / ipm.meter_per_pixel
        keep = np.ones(len(points), dtype=bool)
        for index in range(1, len(points)):
            if abs(points[index, 0] - points[index - 1, 0]) > max_jump_px:
                keep[index] = False
        points = points[keep]
        if len(points) < config.centerline_min_points:
            return np.empty((0, 2)), "centerline_too_few_points", 0.0, 0.0
        try:
            coefficients, _, _ = RobustPolynomialFitter.fit(
                points[:, 1],
                points[:, 0],
                config.centerline_fit_degree,
                config.centerline_min_points,
                max(2.0, max_jump_px * 0.35),
                3,
            )
        except GeometryError:
            return np.empty((0, 2)), "centerline_too_few_points", 0.0, 0.0
        step_px = max(
            2, int(round(config.centerline_sample_step_m / ipm.meter_per_pixel))
        )
        y_near, y_far = int(np.max(points[:, 1])), int(np.min(points[:, 1]))
        ys = np.arange(y_near, y_far - 1, -step_px, dtype=np.float64)
        xs = np.polyval(coefficients, ys)
        candidate = np.column_stack((xs, ys))
        valid = (
            (candidate[:, 0] >= 0)
            & (candidate[:, 0] < ipm.output_width)
            & (candidate[:, 1] >= roi[0])
            & (candidate[:, 1] <= roi[1])
            & (
                np.abs(candidate[:, 0] - ipm.vehicle_center_x_px)
                * ipm.meter_per_pixel
                <= config.centerline_max_abs_lateral_m
            )
        )
        candidate = candidate[valid]
        if len(candidate) < config.centerline_min_points:
            return np.empty((0, 2)), "centerline_too_few_points", 0.0, 0.0
        pixel_indices = np.round(candidate).astype(int)
        yellow_ratio = float(
            np.mean(main_yellow[pixel_indices[:, 1], pixel_indices[:, 0]] > 0)
        )
        span_m = (
            float(candidate[0, 1] - candidate[-1, 1])
            * ipm.meter_per_pixel
        )
        if span_m < config.centerline_min_forward_span_m:
            return np.empty((0, 2)), "centerline_span_too_short", yellow_ratio, span_m
        if yellow_ratio < config.centerline_min_yellow_ratio:
            return np.empty((0, 2)), "centerline_yellow_ratio_low", yellow_ratio, span_m
        forwards = (
            ipm.vehicle_origin_y_px - candidate[:, 1]
        ) * ipm.meter_per_pixel
        if np.any(np.diff(forwards) <= 0):
            return np.empty((0, 2)), "processing_error", yellow_ratio, span_m
        return candidate, "valid", yellow_ratio, span_m


class PathConverter:
    """鸟瞰像素转换为车辆参考坐标和带航向的 ROS Path。"""

    @staticmethod
    def metric_samples(
        points: np.ndarray, ipm: IpmConfig
    ) -> List[Tuple[float, float, float]]:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(points) == 0:
            return []
        forward = (
            ipm.vehicle_origin_y_px - points[:, 1]
        ) * ipm.meter_per_pixel
        left = (
            ipm.vehicle_center_x_px - points[:, 0]
        ) * ipm.meter_per_pixel
        if np.any(np.diff(forward) <= 0):
            raise GeometryError("Path forward 必须单调增加")
        samples: List[Tuple[float, float, float]] = []
        for index in range(len(points)):
            if index == len(points) - 1:
                reference = max(0, index - 1)
                delta_forward = forward[index] - forward[reference]
                delta_left = left[index] - left[reference]
            else:
                delta_forward = forward[index + 1] - forward[index]
                delta_left = left[index + 1] - left[index]
            yaw = math.atan2(float(delta_left), float(delta_forward))
            samples.append((float(forward[index]), float(left[index]), yaw))
        return samples

    @classmethod
    def to_path(
        cls, points: np.ndarray, ipm: IpmConfig, source: Image
    ) -> RosPath:
        path = RosPath()
        path.header = copy.deepcopy(source.header)
        path.header.frame_id = ipm.vehicle_reference_point_name
        for forward, left, yaw in cls.metric_samples(points, ipm):
            pose = PoseStamped()
            pose.header = copy.deepcopy(path.header)
            pose.pose.position.x = forward
            pose.pose.position.y = left
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            path.poses.append(pose)
        return path

    @staticmethod
    def empty(ipm: Optional[IpmConfig], source: Optional[Image]) -> RosPath:
        path = RosPath()
        if source is not None:
            path.header = copy.deepcopy(source.header)
        if ipm is not None:
            path.header.frame_id = ipm.vehicle_reference_point_name
        return path


class GeometryParameterStore:
    """第四阶段参数独立原子保存，不触碰前三阶段 JSON。"""

    def __init__(self, script_path: Path) -> None:
        self.config_dir = script_path.resolve().parent / "config"

    def path(self, profile_name: str) -> Path:
        if not PROFILE_PATTERN.fullmatch(profile_name):
            raise GeometryError("非法 profile_name")
        return self.config_dir / f"{profile_name}_geometry.json"

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def save(self, config: GeometryConfig, ipm_path: str) -> Path:
        document = {
            "version": GEOMETRY_CONFIG_VERSION,
            "profile_name": config.profile_name,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "ipm_config_path": ipm_path,
            "planning_roi": {
                "forward_min_m_override": config.planning_forward_min_m_override,
                "forward_max_m_override": config.planning_forward_max_m_override,
            },
            "main_yellow": {
                "near_seed_forward_min_m": config.near_seed_forward_min_m,
                "near_seed_forward_max_m": config.near_seed_forward_max_m,
                "near_seed_half_width_m": config.near_seed_half_width_m,
                "min_area_px": config.main_yellow_min_area_px,
                "min_vertical_span_px": config.main_yellow_min_vertical_span_px,
            },
            "boundary": {
                "component_min_pixels": config.boundary_component_min_pixels,
                "component_min_vertical_span_px": config.boundary_component_min_vertical_span_px,
                "fit_degree": config.boundary_fit_degree,
                "fit_min_points": config.boundary_fit_min_points,
                "fit_max_residual_px": config.boundary_fit_max_residual_px,
                "fit_outlier_iterations": config.boundary_fit_outlier_iterations,
            },
            "lane_width": {
                "min_ratio": config.lane_width_min_ratio,
                "max_ratio": config.lane_width_max_ratio,
            },
            "road_side": {
                "probe_min_px": config.road_side_probe_min_px,
                "probe_max_px": config.road_side_probe_max_px,
                "probe_step_px": config.road_side_probe_step_px,
                "min_yellow_ratio": config.road_side_min_yellow_ratio,
                "ratio_margin": config.road_side_ratio_margin,
            },
            "centerline": {
                "sample_step_m": config.centerline_sample_step_m,
                "fit_degree": config.centerline_fit_degree,
                "min_points": config.centerline_min_points,
                "min_forward_span_m": config.centerline_min_forward_span_m,
                "min_yellow_ratio": config.centerline_min_yellow_ratio,
                "max_lateral_jump_m": config.centerline_max_lateral_jump_m,
                "max_abs_lateral_m": config.centerline_max_abs_lateral_m,
            },
            "history": {
                "enable": config.history_fallback_enable,
                "max_frames": config.history_max_frames,
                "max_age_sec": config.history_max_age_sec,
            },
        }
        path = self.path(config.profile_name)
        self._atomic_write(
            path, json.dumps(document, ensure_ascii=False, indent=2) + "\n"
        )
        return path

    def load(self, config: GeometryConfig) -> GeometryConfig:
        path = self.path(config.profile_name)
        if not path.exists():
            raise GeometryError(f"第四阶段参数文件不存在: {path}")
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("version") != GEOMETRY_CONFIG_VERSION:
            raise GeometryError("第四阶段参数版本不匹配")
        candidate = copy.deepcopy(config)
        mapping = {
            "planning_forward_min_m_override": document["planning_roi"]["forward_min_m_override"],
            "planning_forward_max_m_override": document["planning_roi"]["forward_max_m_override"],
            "near_seed_forward_min_m": document["main_yellow"]["near_seed_forward_min_m"],
            "near_seed_forward_max_m": document["main_yellow"]["near_seed_forward_max_m"],
            "near_seed_half_width_m": document["main_yellow"]["near_seed_half_width_m"],
            "main_yellow_min_area_px": document["main_yellow"]["min_area_px"],
            "main_yellow_min_vertical_span_px": document["main_yellow"]["min_vertical_span_px"],
            "boundary_component_min_pixels": document["boundary"]["component_min_pixels"],
            "boundary_component_min_vertical_span_px": document["boundary"]["component_min_vertical_span_px"],
            "boundary_fit_degree": document["boundary"]["fit_degree"],
            "boundary_fit_min_points": document["boundary"]["fit_min_points"],
            "boundary_fit_max_residual_px": document["boundary"]["fit_max_residual_px"],
            "boundary_fit_outlier_iterations": document["boundary"]["fit_outlier_iterations"],
            "lane_width_min_ratio": document["lane_width"]["min_ratio"],
            "lane_width_max_ratio": document["lane_width"]["max_ratio"],
            "road_side_probe_min_px": document["road_side"]["probe_min_px"],
            "road_side_probe_max_px": document["road_side"]["probe_max_px"],
            "road_side_probe_step_px": document["road_side"]["probe_step_px"],
            "road_side_min_yellow_ratio": document["road_side"]["min_yellow_ratio"],
            "road_side_ratio_margin": document["road_side"]["ratio_margin"],
            "centerline_sample_step_m": document["centerline"]["sample_step_m"],
            "centerline_fit_degree": document["centerline"]["fit_degree"],
            "centerline_min_points": document["centerline"]["min_points"],
            "centerline_min_forward_span_m": document["centerline"]["min_forward_span_m"],
            "centerline_min_yellow_ratio": document["centerline"]["min_yellow_ratio"],
            "centerline_max_lateral_jump_m": document["centerline"]["max_lateral_jump_m"],
            "centerline_max_abs_lateral_m": document["centerline"]["max_abs_lateral_m"],
            "history_fallback_enable": document["history"]["enable"],
            "history_max_frames": document["history"]["max_frames"],
            "history_max_age_sec": document["history"]["max_age_sec"],
        }
        for name, value in mapping.items():
            current = getattr(candidate, name)
            if isinstance(current, bool):
                converted = bool(value)
            elif isinstance(current, int):
                converted = int(value)
            else:
                converted = float(value)
            setattr(candidate, name, converted)
        candidate.validate()
        return candidate


class GeometryFrameStore:
    """Web 只保留每种图像的最新 JPEG。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frames: Dict[str, bytes] = {}
        self.sequence: Dict[str, int] = {}

    def update(self, name: str, image: np.ndarray, quality: int) -> None:
        ok, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not ok:
            return
        with self.lock:
            self.frames[name] = encoded.tobytes()
            self.sequence[name] = self.sequence.get(name, 0) + 1

    def get(self, name: str) -> Tuple[Optional[bytes], int]:
        with self.lock:
            return self.frames.get(name), self.sequence.get(name, 0)


WEB_PAGE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>第四阶段道路几何</title>
<style>
body{margin:0;background:#10151d;color:#e7edf5;font:14px system-ui}.top{padding:12px 18px;background:#172131;position:sticky;top:0;z-index:2}
h1{margin:0 0 6px;font-size:21px}.good{color:#6fe29b}.bad{color:#ff6b6b}.warn{color:#ffcc66}
.page{padding:12px}.panel{background:#182231;border:1px solid #2b3a50;border-radius:8px;padding:12px;margin-bottom:12px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
img{display:block;width:100%;height:auto;background:#05080c}h3{margin:4px 0}.layout{display:grid;grid-template-columns:minmax(0,2fr) minmax(330px,1fr);gap:12px}
label{display:grid;grid-template-columns:1fr 120px;gap:8px;margin:5px 0}input{background:#0f1722;color:#eef;border:1px solid #40516a;border-radius:4px;padding:5px}
button{padding:7px 10px;background:#31577f;color:white;border:0;border-radius:5px;margin:3px;cursor:pointer}pre{white-space:pre-wrap;background:#0d131d;padding:8px;max-height:360px;overflow:auto}
@media(max-width:1000px){.layout,.grid2,.grid3{grid-template-columns:1fr}}</style></head><body>
<div class="top"><h1>第四阶段：鸟瞰道路几何（gxb_geometry_v1）</h1><div id="summary">等待同步图像……</div>
<div class="warn">本页面仅观察和调整几何参数，不提供车辆运动操作。</div></div>
<div class="page layout"><main>
<section class="panel grid2"><div><h3>鸟瞰综合 Overlay</h3><img src="/stream/overlay"></div><div><h3>最终中心线</h3><img src="/stream/centerline"></div></section>
<section class="panel grid3"><div><h3>IPM boundary</h3><img src="/stream/ipm_boundary"></div><div><h3>IPM yellow</h3><img src="/stream/ipm_yellow"></div><div><h3>IPM green</h3><img src="/stream/ipm_green"></div></section>
<section class="panel grid3"><div><h3>Main yellow</h3><img src="/stream/main_yellow"></div><div><h3>Selected boundary</h3><img src="/stream/selected_boundary"></div><div><h3>Raw centerline</h3><img src="/stream/centerline_raw"></div></section>
</main><aside class="panel"><h3>第四阶段参数</h3><div id="params"></div>
<button onclick="post('/api/reload_ipm')">重新加载 IPM</button><button onclick="post('/api/save')">保存参数</button>
<button onclick="post('/api/load')">加载参数</button><button onclick="post('/api/reset_defaults')">恢复默认</button>
<pre id="status"></pre></aside></div>
<script>
const fields=['planning_forward_min_m_override','planning_forward_max_m_override','near_seed_forward_min_m','near_seed_forward_max_m','near_seed_half_width_m','boundary_component_min_pixels','boundary_component_min_vertical_span_px','boundary_fit_degree','boundary_fit_max_residual_px','lane_width_min_ratio','lane_width_max_ratio','road_side_probe_min_px','road_side_probe_max_px','road_side_min_yellow_ratio','road_side_ratio_margin','centerline_sample_step_m','centerline_fit_degree','centerline_min_points','centerline_min_forward_span_m','centerline_min_yellow_ratio','centerline_max_lateral_jump_m','history_fallback_enable'];
function el(id){return document.getElementById(id)}
function build(){el('params').innerHTML=fields.map(k=>`<label>${k}<input id="p_${k}"></label>`).join('');fields.forEach(k=>el('p_'+k).onchange=()=>post('/api/set',{[k]:el('p_'+k).value}))}
async function post(url,data={}){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});const j=await r.json();if(!r.ok)alert(j.error||'操作失败');await refresh()}
async function refresh(){try{const s=await(await fetch('/api/status',{cache:'no-store'})).json();const c=await(await fetch('/api/config',{cache:'no-store'})).json();fields.forEach(k=>{if(document.activeElement!==el('p_'+k))el('p_'+k).value=c[k]});el('summary').innerHTML=`模式=${s.centerline_mode} | <span class="${s.centerline_valid?'good':'bad'}">${s.centerline_valid?'有效':'无效: '+s.centerline_reason}</span> | 置信度=${Number(s.centerline_confidence).toFixed(2)} | 边界=${s.selected_boundary_count} | 点数=${s.final_centerline_point_count} | 宽度=${Number(s.measured_lane_width_px_mean).toFixed(1)} px | FPS=${Number(s.fps).toFixed(1)} | ${s.last_error||''}`;el('status').textContent=JSON.stringify(s,null,2)}catch(e){el('summary').textContent='状态读取失败: '+e}}
build();setInterval(refresh,800);refresh();</script></body></html>"""


class GeometryWebServer:
    """标准库 HTTP 服务；慢客户端只读取最新 JPEG。"""

    def __init__(self, node: "LaneGeometryPlannerNode") -> None:
        self.node = node
        self.server: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def _json(self, code: int, data: Dict[str, Any]) -> None:
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                path = urllib.parse.urlparse(self.path).path
                if path == "/":
                    body = WEB_PAGE.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/api/status":
                    self._json(200, owner.node.api_status())
                elif path == "/api/config":
                    self._json(200, asdict(owner.node.config))
                elif path.startswith("/stream/"):
                    self._stream(path.removeprefix("/stream/"))
                else:
                    self.send_error(404)

            def _stream(self, name: str) -> None:
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last_sequence = -1
                try:
                    while True:
                        frame, sequence = owner.node.frames.get(name)
                        if frame is not None and sequence != last_sequence:
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\n"
                            )
                            self.wfile.write(
                                f"Content-Length: {len(frame)}\r\n\r\n".encode()
                            )
                            self.wfile.write(frame + b"\r\n")
                            self.wfile.flush()
                            last_sequence = sequence
                        time.sleep(
                            max(
                                0.05,
                                1.0 / owner.node.config.web_gui_max_fps,
                            )
                        )
                except (BrokenPipeError, ConnectionResetError):
                    return

            def do_POST(self) -> None:
                path = urllib.parse.urlparse(self.path).path
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length > 65536:
                        raise GeometryError("请求体过大")
                    body = self.rfile.read(length) if length else b"{}"
                    payload = json.loads(body.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise GeometryError("请求必须是 JSON 对象")
                    self._json(200, owner.node.handle_api(path, payload))
                except (GeometryError, ValueError, TypeError, KeyError) as exc:
                    self._json(400, {"ok": False, "error": str(exc)})
                except Exception as exc:
                    owner.node.set_error(f"Web 内部错误: {exc}")
                    self._json(500, {"ok": False, "error": "内部错误"})

        self.server = ThreadingHTTPServer(
            (self.node.config.web_gui_host, self.node.config.web_gui_port),
            Handler,
        )
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="geometry-web",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()


class LaneGeometryPlannerNode(Node):
    """协调精确同步、纯几何算法、ROS 输出和 Web 参数。"""

    CONFIG_FIELDS = {item.name for item in fields(GeometryConfig)}
    WEB_EDITABLE = {
        "planning_forward_min_m_override",
        "planning_forward_max_m_override",
        "near_seed_forward_min_m",
        "near_seed_forward_max_m",
        "near_seed_half_width_m",
        "boundary_component_min_pixels",
        "boundary_component_min_vertical_span_px",
        "boundary_fit_degree",
        "boundary_fit_max_residual_px",
        "lane_width_min_ratio",
        "lane_width_max_ratio",
        "road_side_probe_min_px",
        "road_side_probe_max_px",
        "road_side_min_yellow_ratio",
        "road_side_ratio_margin",
        "centerline_sample_step_m",
        "centerline_fit_degree",
        "centerline_min_points",
        "centerline_min_forward_span_m",
        "centerline_min_yellow_ratio",
        "centerline_max_lateral_jump_m",
        "history_fallback_enable",
    }

    def __init__(self) -> None:
        super().__init__("gxb_lane_geometry_planner")
        self.lock = threading.RLock()
        defaults = GeometryConfig()
        for name in self.CONFIG_FIELDS:
            self.declare_parameter(name, getattr(defaults, name))
        self.config = GeometryConfig(
            **{name: self.get_parameter(name).value for name in self.CONFIG_FIELDS}
        )
        self.config.validate()
        self.defaults = copy.deepcopy(self.config)
        self.ipm_loader = IpmConfigLoader(Path(__file__))
        self.parameter_store = GeometryParameterStore(Path(__file__))
        self.ipm: Optional[IpmConfig] = None
        self.ipm_error = ""
        self.ipm_mtime = 0.0
        self.last_ipm_reload_check = 0.0
        self._reload_ipm(required=self.config.ipm_config_required)
        self.synchronizer = ExactStampSynchronizer(
            self.config.sync_cache_size, self.config.sync_max_age_sec
        )
        self.pending_groups: "OrderedDict[Tuple[int, int, str], Tuple[Dict[str, MaskPacket], float]]" = OrderedDict()
        self.segmentation_status_cache: "OrderedDict[Tuple[int, int, str], Tuple[Dict[str, Any], float]]" = OrderedDict()
        self.latest_segmentation_status: Dict[str, Any] = {}
        self.frame_sequence = 0
        self.last_frame_times: Deque[float] = deque(maxlen=30)
        self.last_valid_points = np.empty((0, 2), dtype=np.float64)
        self.last_valid_time = 0.0
        self.last_center_x: Optional[float] = None
        self.history: Deque[Tuple[float, np.ndarray]] = deque(
            maxlen=self.config.history_max_frames
        )
        self.frames = GeometryFrameStore()
        self.last_web_encode_time = 0.0
        self._initialize_frames()
        self.status = self._initial_status()

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            Image,
            self.config.boundary_topic,
            lambda msg: self._mask_callback("boundary", msg),
            image_qos,
        )
        self.create_subscription(
            Image,
            self.config.yellow_topic,
            lambda msg: self._mask_callback("yellow", msg),
            image_qos,
        )
        self.create_subscription(
            Image,
            self.config.green_topic,
            lambda msg: self._mask_callback("green", msg),
            image_qos,
        )
        self.create_subscription(
            String,
            self.config.segmentation_status_topic,
            self._segmentation_status_callback,
            image_qos,
        )
        self.image_publishers = {
            name: self.create_publisher(Image, topic, image_qos)
            for name, topic in OUTPUT_TOPICS.items()
            if name not in ("path", "status")
        }
        self.path_publisher = self.create_publisher(
            RosPath, OUTPUT_TOPICS["path"], reliable_qos
        )
        self.status_publisher = self.create_publisher(
            String, OUTPUT_TOPICS["status"], reliable_qos
        )
        self.create_timer(0.1, self._maintenance)
        self.create_timer(0.5, self._publish_status)
        self.web = GeometryWebServer(self)
        if self.config.web_gui_enable:
            self.web.start()
        self.get_logger().info(
            f"第四阶段启动: IPM={'loaded' if self.ipm else 'missing'}, "
            f"web=http://{self.config.web_gui_host}:{self.config.web_gui_port}, "
            f"history={self.config.history_fallback_enable}"
        )

    def _initial_status(self) -> Dict[str, Any]:
        if self.ipm is not None:
            initial_reason = "sync_timeout"
        elif "MISSING" in self.ipm_error.upper():
            initial_reason = "ipm_config_missing"
        else:
            initial_reason = "ipm_config_invalid"
        if self.ipm is not None:
            forward_min, forward_max, roi_y_min, roi_y_max = (
                self.ipm.planning_roi(self.config)
            )
        else:
            forward_min = forward_max = 0.0
            roi_y_min = roi_y_max = 0
        return {
            "interface_version": GEOMETRY_INTERFACE_VERSION,
            "ipm_interface_version": IPM_INTERFACE_VERSION,
            "segmentation_interface_version": "",
            "ipm_config_loaded": self.ipm is not None,
            "ipm_config_path": self.ipm.path if self.ipm else str(self.ipm_loader.resolve_path(self.config)),
            "ipm_config_error": self.ipm_error,
            "image_width": self.ipm.image_width if self.ipm else 0,
            "image_height": self.ipm.image_height if self.ipm else 0,
            "output_width": self.ipm.output_width if self.ipm else 0,
            "output_height": self.ipm.output_height if self.ipm else 0,
            "meter_per_pixel": self.ipm.meter_per_pixel if self.ipm else 0.0,
            "vehicle_center_x_px": self.ipm.vehicle_center_x_px if self.ipm else 0.0,
            "vehicle_origin_y_px": self.ipm.vehicle_origin_y_px if self.ipm else 0.0,
            "vehicle_reference_point_name": self.ipm.vehicle_reference_point_name if self.ipm else "",
            "source_stamp_sec": 0,
            "source_stamp_nanosec": 0,
            "source_frame_id": "",
            "frame_sequence": 0,
            "resolved_forward_min_m": forward_min,
            "resolved_forward_max_m": forward_max,
            "planning_roi_y_min_px": roi_y_min,
            "planning_roi_y_max_px": roi_y_max,
            "boundary_pixels": 0,
            "yellow_pixels": 0,
            "green_pixels": 0,
            "main_yellow_pixels": 0,
            "yellow_component_count": 0,
            "main_yellow_component_id": -1,
            "main_yellow_seed_connected": False,
            "main_yellow_valid": False,
            "boundary_component_count": 0,
            "valid_boundary_component_count": 0,
            "selected_boundary_count": 0,
            "selected_boundary_ids": [],
            "selected_boundary_components": [],
            "centerline_mode": "invalid",
            "centerline_valid": False,
            "centerline_reason": initial_reason,
            "centerline_confidence": 0.0,
            "expected_lane_width_m": self.ipm.expected_lane_width_m if self.ipm else 0.0,
            "expected_lane_width_px": self.ipm.expected_lane_width_px if self.ipm else 0.0,
            "measured_lane_width_m_mean": 0.0,
            "measured_lane_width_px_mean": 0.0,
            "measured_lane_width_px_std": 0.0,
            "dual_overlap_sample_count": 0,
            "dual_pair_score": 0.0,
            "single_boundary_center_offset_m": self.ipm.single_boundary_center_offset_m if self.ipm else 0.0,
            "single_boundary_offset_px": self.ipm.single_boundary_offset_px if self.ipm else 0.0,
            "single_boundary_road_side": "unknown",
            "single_boundary_valid_sample_ratio": 0.0,
            "raw_centerline_point_count": 0,
            "final_centerline_point_count": 0,
            "centerline_forward_span_m": 0.0,
            "centerline_yellow_ratio": 0.0,
            "sync_drop_count": 0,
            "sync_cache_boundary_size": 0,
            "sync_cache_yellow_size": 0,
            "sync_cache_green_size": 0,
            "processing_time_ms": 0.0,
            "fps": 0.0,
            "last_error": self.ipm_error,
        }

    def _initialize_frames(self) -> None:
        height = self.ipm.output_height if self.ipm else 800
        width = self.ipm.output_width if self.ipm else 600
        blank = np.zeros((height, width), dtype=np.uint8)
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            overlay,
            "WAITING FOR SYNCHRONIZED MASKS",
            (35, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        for name in (
            "ipm_boundary",
            "ipm_yellow",
            "ipm_green",
            "main_yellow",
            "selected_boundary",
            "centerline_raw",
            "centerline",
        ):
            self.frames.update(name, blank, self.config.web_gui_jpeg_quality)
        self.frames.update("overlay", overlay, self.config.web_gui_jpeg_quality)

    def _reload_ipm(self, required: bool = True) -> None:
        try:
            self.ipm = self.ipm_loader.load(self.config)
            self.ipm_error = ""
            self.ipm_mtime = Path(self.ipm.path).stat().st_mtime
        except GeometryError as exc:
            self.ipm = None
            self.ipm_error = str(exc)
            self.ipm_mtime = 0.0
            if required:
                self.get_logger().error(self.ipm_error)
        if hasattr(self, "history"):
            self.history.clear()
            self.last_valid_points = np.empty((0, 2), dtype=np.float64)
            self.last_valid_time = 0.0
            self.last_center_x = None

    def set_error(self, message: str) -> None:
        with self.lock:
            self.status["last_error"] = message

    def _mask_callback(self, name: str, message: Image) -> None:
        try:
            mask = ImageCodec.to_binary_mask(message)
            packet = MaskPacket(mask, message, time.monotonic())
            with self.lock:
                synchronized = self.synchronizer.add(name, packet)
                if synchronized is not None:
                    key, group = synchronized
                    self.pending_groups[key] = (group, time.monotonic())
                    while len(self.pending_groups) > self.config.sync_cache_size:
                        self.pending_groups.popitem(last=False)
                        self.synchronizer.drop_count += 1
                    self._try_process(key)
                self._update_sync_status()
        except Exception as exc:
            self.set_error(f"{name} mask 解码/同步失败: {exc}")

    @staticmethod
    def _status_key(document: Dict[str, Any]) -> Optional[Tuple[int, int, str]]:
        try:
            return (
                int(document["source_stamp_sec"]),
                int(document["source_stamp_nanosec"]),
                str(document["source_frame_id"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _segmentation_status_callback(self, message: String) -> None:
        try:
            document = json.loads(message.data)
            if not isinstance(document, dict):
                raise GeometryError("第二阶段状态不是 JSON 对象")
            key = self._status_key(document)
            with self.lock:
                self.latest_segmentation_status = document
                if key is not None:
                    self.segmentation_status_cache[key] = (
                        document,
                        time.monotonic(),
                    )
                    while len(self.segmentation_status_cache) > self.config.sync_cache_size * 2:
                        self.segmentation_status_cache.popitem(last=False)
                    self._try_process(key)
        except Exception as exc:
            self.set_error(f"第二阶段状态解析失败: {exc}")

    def _try_process(self, key: Tuple[int, int, str]) -> None:
        pending = self.pending_groups.get(key)
        status_item = self.segmentation_status_cache.get(key)
        if pending is None:
            return
        # 旧版状态若没有源时间戳，允许使用最新状态；当前接口正常走精确 key。
        if status_item is None:
            if self._status_key(self.latest_segmentation_status) is None and self.latest_segmentation_status:
                status_document = self.latest_segmentation_status
            else:
                return
        else:
            status_document = status_item[0]
        group = self.pending_groups.pop(key)[0]
        self.segmentation_status_cache.pop(key, None)
        self._process_group(key, group, status_document)

    def _maintenance(self) -> None:
        now = time.monotonic()
        expired_source: Optional[Image] = None
        with self.lock:
            if (
                self.config.auto_reload_ipm_config
                and now - self.last_ipm_reload_check >= 1.0
            ):
                self.last_ipm_reload_check = now
                try:
                    path = self.ipm_loader.resolve_path(self.config)
                    mtime = path.stat().st_mtime if path.exists() else 0.0
                    if mtime != self.ipm_mtime:
                        self._reload_ipm(required=False)
                        self.status = self._initial_status()
                        self._initialize_frames()
                except Exception as exc:
                    self.status["last_error"] = f"自动重载 IPM 失败: {exc}"
            sync_dropped = self.synchronizer.cleanup(now)
            pending_dropped = 0
            for key, (_group, arrival) in list(self.pending_groups.items()):
                if now - arrival > self.config.sync_max_age_sec:
                    expired = self.pending_groups.pop(key, None)
                    if expired is not None:
                        expired_source = expired[0]["boundary"].message
                    pending_dropped += 1
            for key, (_status, arrival) in list(self.segmentation_status_cache.items()):
                if now - arrival > self.config.sync_max_age_sec:
                    self.segmentation_status_cache.pop(key, None)
            if sync_dropped or pending_dropped:
                self.synchronizer.drop_count += pending_dropped
                self.status["centerline_valid"] = False
                self.status["centerline_reason"] = "sync_timeout"
                self.status["last_error"] = "同步缓存存在超时数据"
            self._update_sync_status()
            if expired_source is not None:
                self.frame_sequence += 1
                key = ExactStampSynchronizer.key(expired_source)
                self._publish_invalid(
                    expired_source,
                    "sync_timeout",
                    {
                        "source_stamp_sec": key[0],
                        "source_stamp_nanosec": key[1],
                        "source_frame_id": key[2],
                        "frame_sequence": self.frame_sequence,
                    },
                    time.perf_counter(),
                )

    def _update_sync_status(self) -> None:
        sizes = self.synchronizer.sizes()
        self.status.update(
            {
                "sync_drop_count": self.synchronizer.drop_count,
                "sync_cache_boundary_size": sizes["boundary"],
                "sync_cache_yellow_size": sizes["yellow"],
                "sync_cache_green_size": sizes["green"],
            }
        )

    def _segmentation_reason(self, document: Dict[str, Any]) -> Optional[str]:
        if document.get("interface_version") != SEGMENTATION_INTERFACE_VERSION:
            return "segmentation_interface_mismatch"
        if not bool(document.get("config_loaded", False)):
            return "segmentation_invalid"
        if not bool(document.get("boundary_valid", False)):
            return "segmentation_invalid"
        return None

    def _process_group(
        self,
        key: Tuple[int, int, str],
        group: Dict[str, MaskPacket],
        segmentation_status: Dict[str, Any],
    ) -> None:
        start = time.perf_counter()
        self.frame_sequence += 1
        source = group["boundary"].message
        base_update = {
            "source_stamp_sec": key[0],
            "source_stamp_nanosec": key[1],
            "source_frame_id": key[2],
            "frame_sequence": self.frame_sequence,
            "segmentation_interface_version": str(
                segmentation_status.get("interface_version", "")
            ),
        }
        if self.ipm is None:
            reason = (
                "ipm_config_missing"
                if "MISSING" in self.ipm_error.upper()
                else "ipm_config_invalid"
            )
            self._publish_invalid(
                source, reason, base_update, start
            )
            return
        ipm = self.ipm
        try:
            validate_mask_group_size(group, ipm)
        except GeometryError as exc:
            self._publish_invalid(
                source, "mask_size_mismatch", base_update, start
            )
            self.set_error(str(exc))
            return
        try:
            transformed = {
                name: MaskIpmTransformer.transform(packet.mask, ipm)
                for name, packet in group.items()
            }
            forward_min, forward_max, y_min, y_max = ipm.planning_roi(
                self.config
            )
            roi = (y_min, y_max)
            segmentation_reason = self._segmentation_reason(
                segmentation_status
            )
            if segmentation_reason:
                self._publish_invalid(
                    source,
                    segmentation_reason,
                    {
                        **base_update,
                        "resolved_forward_min_m": forward_min,
                        "resolved_forward_max_m": forward_max,
                        "planning_roi_y_min_px": y_min,
                        "planning_roi_y_max_px": y_max,
                    },
                    start,
                    transformed,
                )
                return
            main_yellow, yellow_status = MainYellowSelector.select(
                transformed["yellow"], ipm, self.config, roi
            )
            if not yellow_status["main_yellow_valid"]:
                self._publish_invalid(
                    source,
                    "main_yellow_missing",
                    {
                        **base_update,
                        **yellow_status,
                        "resolved_forward_min_m": forward_min,
                        "resolved_forward_max_m": forward_max,
                        "planning_roi_y_min_px": y_min,
                        "planning_roi_y_max_px": y_max,
                    },
                    start,
                    transformed,
                    main_yellow,
                )
                return
            curves, raw_component_count = BoundaryComponentAnalyzer.analyze(
                transformed["boundary"], ipm, self.config, roi
            )
            step_px = max(
                2,
                int(
                    round(
                        self.config.centerline_sample_step_m
                        / ipm.meter_per_pixel
                    )
                ),
            )
            for curve in curves:
                BoundaryRoadSideClassifier.classify_curve(
                    curve,
                    main_yellow,
                    transformed["green"],
                    self.config,
                    step_px,
                )
            if not curves:
                self._publish_invalid(
                    source,
                    "boundary_missing",
                    {
                        **base_update,
                        **yellow_status,
                        "boundary_component_count": raw_component_count,
                        "resolved_forward_min_m": forward_min,
                        "resolved_forward_max_m": forward_max,
                        "planning_roi_y_min_px": y_min,
                        "planning_roi_y_max_px": y_max,
                    },
                    start,
                    transformed,
                    main_yellow,
                )
                return
            result = DualBoundaryPlanner.plan(
                curves,
                main_yellow,
                ipm,
                self.config,
                roi,
                self.last_center_x,
            )
            if result is None:
                result = SingleBoundaryPlanner.plan(
                    curves,
                    main_yellow,
                    transformed["green"],
                    ipm,
                    self.config,
                    roi,
                )
            if result is None:
                reason = (
                    "single_boundary_side_unknown"
                    if any(curve.inward_sign == 0 for curve in curves)
                    else "boundary_pair_invalid"
                )
                result = GeometryResult(reason=reason)
            if not bool(yellow_status.get("main_yellow_seed_connected", False)):
                result.confidence *= 0.80
            if len(result.raw_points):
                (
                    result.final_points,
                    result.reason,
                    result.yellow_ratio,
                    result.forward_span_m,
                ) = CenterlineSmoother.smooth(
                    result.raw_points,
                    main_yellow,
                    ipm,
                    self.config,
                    roi,
                )
                result.valid = result.reason == "valid"
            if not result.valid and self.config.history_fallback_enable:
                age = time.monotonic() - self.last_valid_time
                lateral_change_ok = bool(
                    len(self.last_valid_points)
                    and (
                        len(result.raw_points) == 0
                        or abs(
                            float(result.raw_points[0, 0])
                            - float(self.last_valid_points[0, 0])
                        )
                        * ipm.meter_per_pixel
                        <= self.config.centerline_max_lateral_jump_m
                    )
                )
                if (
                    len(self.last_valid_points) >= self.config.centerline_min_points
                    and age <= self.config.history_max_age_sec
                    and lateral_change_ok
                ):
                    result.final_points = self.last_valid_points.copy()
                    result.mode = "history_fallback"
                    result.reason = "valid"
                    result.valid = True
                    result.confidence = min(0.50, result.confidence or 0.45)
            if not result.valid:
                result.confidence = 0.0
            if result.valid:
                self.last_valid_points = result.final_points.copy()
                self.last_valid_time = time.monotonic()
                self.last_center_x = float(result.final_points[0, 0])
                self.history.append(
                    (self.last_valid_time, result.final_points.copy())
                )
            self._publish_result(
                source,
                transformed,
                main_yellow,
                curves,
                result,
                {
                    **base_update,
                    **yellow_status,
                    "boundary_component_count": raw_component_count,
                    "resolved_forward_min_m": forward_min,
                    "resolved_forward_max_m": forward_max,
                    "planning_roi_y_min_px": y_min,
                    "planning_roi_y_max_px": y_max,
                },
                start,
                roi,
            )
        except Exception as exc:
            self.set_error(f"几何处理异常: {exc}")
            self._publish_invalid(
                source, "processing_error", base_update, start
            )

    @staticmethod
    def _points_mask(
        shape: Tuple[int, int],
        points: np.ndarray,
        thickness: int,
    ) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        if len(points) == 0:
            return mask
        integer = np.round(points).astype(np.int32)
        if len(integer) == 1:
            cv2.circle(mask, tuple(integer[0]), thickness, 255, -1)
        else:
            cv2.polylines(mask, [integer], False, 255, thickness, cv2.LINE_8)
            for point in integer:
                cv2.circle(mask, tuple(point), max(1, thickness), 255, -1)
        return mask

    def _make_overlay(
        self,
        transformed: Dict[str, np.ndarray],
        main_yellow: np.ndarray,
        selected_mask: np.ndarray,
        raw_mask: np.ndarray,
        center_mask: np.ndarray,
        result: GeometryResult,
        roi: Tuple[int, int],
        processing_ms: float,
    ) -> np.ndarray:
        height, width = main_yellow.shape
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        green = transformed["green"] > 0
        yellow = main_yellow > 0
        overlay[green] = (0, 90, 0)
        overlay[yellow] = (0, 150, 150)
        overlay[transformed["boundary"] > 0] = (200, 200, 200)
        if result.mode == "dual_boundary_midpoint" and len(result.selected_curves) == 2:
            left_mask = BoundaryComponentAnalyzer.component_mask(
                transformed["boundary"], [result.selected_curves[0]], 3
            )
            right_mask = BoundaryComponentAnalyzer.component_mask(
                transformed["boundary"], [result.selected_curves[1]], 3
            )
            overlay[left_mask > 0] = (255, 0, 0)
            overlay[right_mask > 0] = (255, 0, 255)
        else:
            overlay[selected_mask > 0] = (255, 255, 0)
        overlay[raw_mask > 0] = (0, 140, 255)
        overlay[center_mask > 0] = (0, 0, 255)
        cx = int(round(self.ipm.vehicle_center_x_px)) if self.ipm else width // 2
        oy = int(round(self.ipm.vehicle_origin_y_px)) if self.ipm else height - 1
        cv2.line(overlay, (cx, roi[0]), (cx, roi[1]), (255, 255, 0), 1)
        if 0 <= oy < height:
            cv2.circle(overlay, (cx, oy), 6, (0, 0, 255), -1)
        cv2.rectangle(
            overlay, (0, roi[0]), (width - 1, roi[1]), (255, 255, 255), 1
        )
        lines = (
            f"mode={result.mode}",
            f"valid={result.valid} confidence={result.confidence:.2f}",
            f"boundaries={len(result.selected_curves)} width={result.measured_width_mean_px:.1f}px",
            f"points={len(result.final_points)} time={processing_ms:.1f}ms",
        )
        for index, text in enumerate(lines):
            cv2.putText(
                overlay,
                text,
                (10, 24 + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                text,
                (10, 24 + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return overlay

    def _publish_result(
        self,
        source: Image,
        transformed: Dict[str, np.ndarray],
        main_yellow: np.ndarray,
        curves: Sequence[BoundaryCurve],
        result: GeometryResult,
        status_update: Dict[str, Any],
        start: float,
        roi: Tuple[int, int],
    ) -> None:
        assert self.ipm is not None
        ipm = self.ipm
        shape = (ipm.output_height, ipm.output_width)
        selected_mask = BoundaryComponentAnalyzer.component_mask(
            transformed["boundary"], result.selected_curves, 3
        )
        raw_mask = self._points_mask(shape, result.raw_points, 2)
        center_mask = self._points_mask(shape, result.final_points, 3)
        processing_ms = (time.perf_counter() - start) * 1000.0
        overlay = self._make_overlay(
            transformed,
            main_yellow,
            selected_mask,
            raw_mask,
            center_mask,
            result,
            roi,
            processing_ms,
        )
        images = {
            "ipm_boundary": transformed["boundary"],
            "ipm_yellow": transformed["yellow"],
            "ipm_green": transformed["green"],
            "main_yellow": main_yellow,
            "selected_boundary": selected_mask,
            "centerline_raw": raw_mask,
            "centerline": center_mask,
            "overlay": overlay,
        }
        encode_web = (
            time.monotonic() - self.last_web_encode_time
            >= 1.0 / self.config.web_gui_max_fps
        )
        for name, image in images.items():
            encoding = "bgr8" if name == "overlay" else "mono8"
            self.image_publishers[name].publish(
                ImageCodec.to_message(
                    image, encoding, source, ipm.vehicle_reference_point_name
                )
            )
            if encode_web:
                self.frames.update(
                    name, image, self.config.web_gui_jpeg_quality
                )
        if encode_web:
            self.last_web_encode_time = time.monotonic()
        path = (
            PathConverter.to_path(result.final_points, ipm, source)
            if result.valid
            else PathConverter.empty(ipm, source)
        )
        self.path_publisher.publish(path)
        now = time.monotonic()
        self.last_frame_times.append(now)
        recent = [stamp for stamp in self.last_frame_times if now - stamp <= 1.0]
        measured_m = result.measured_width_mean_px * ipm.meter_per_pixel
        with self.lock:
            self.status.update(
                {
                    **status_update,
                    "ipm_config_loaded": True,
                    "ipm_config_path": ipm.path,
                    "ipm_config_error": "",
                    "image_width": ipm.image_width,
                    "image_height": ipm.image_height,
                    "output_width": ipm.output_width,
                    "output_height": ipm.output_height,
                    "meter_per_pixel": ipm.meter_per_pixel,
                    "vehicle_center_x_px": ipm.vehicle_center_x_px,
                    "vehicle_origin_y_px": ipm.vehicle_origin_y_px,
                    "vehicle_reference_point_name": ipm.vehicle_reference_point_name,
                    "boundary_pixels": int(np.count_nonzero(transformed["boundary"])),
                    "yellow_pixels": int(np.count_nonzero(transformed["yellow"])),
                    "green_pixels": int(np.count_nonzero(transformed["green"])),
                    "main_yellow_pixels": int(np.count_nonzero(main_yellow)),
                    "valid_boundary_component_count": len(curves),
                    "selected_boundary_count": len(result.selected_curves),
                    "selected_boundary_ids": [curve.component_id for curve in result.selected_curves],
                    "selected_boundary_components": [curve.as_status() for curve in result.selected_curves],
                    "centerline_mode": result.mode,
                    "centerline_valid": result.valid,
                    "centerline_reason": result.reason,
                    "centerline_confidence": float(np.clip(result.confidence, 0.0, 1.0)),
                    "expected_lane_width_m": ipm.expected_lane_width_m,
                    "expected_lane_width_px": ipm.expected_lane_width_px,
                    "measured_lane_width_m_mean": measured_m,
                    "measured_lane_width_px_mean": result.measured_width_mean_px,
                    "measured_lane_width_px_std": result.measured_width_std_px,
                    "dual_overlap_sample_count": result.dual_overlap_count,
                    "dual_pair_score": result.dual_pair_score,
                    "single_boundary_center_offset_m": ipm.single_boundary_center_offset_m,
                    "single_boundary_offset_px": ipm.single_boundary_offset_px,
                    "single_boundary_road_side": result.single_road_side,
                    "single_boundary_valid_sample_ratio": result.single_valid_ratio,
                    "raw_centerline_point_count": len(result.raw_points),
                    "final_centerline_point_count": len(result.final_points),
                    "centerline_forward_span_m": result.forward_span_m,
                    "centerline_yellow_ratio": result.yellow_ratio,
                    "processing_time_ms": processing_ms,
                    "fps": float(len(recent)),
                    "last_error": "" if result.valid else result.reason,
                }
            )
            self._update_sync_status()
        self._publish_status()

    def _publish_invalid(
        self,
        source: Optional[Image],
        reason: str,
        status_update: Dict[str, Any],
        start: float,
        transformed: Optional[Dict[str, np.ndarray]] = None,
        main_yellow: Optional[np.ndarray] = None,
    ) -> None:
        invalid_defaults = {
            "yellow_component_count": 0,
            "main_yellow_component_id": -1,
            "main_yellow_seed_connected": False,
            "main_yellow_valid": False,
            "boundary_component_count": 0,
        }
        status_update = {**invalid_defaults, **status_update}
        ipm = self.ipm
        width = ipm.output_width if ipm else 600
        height = ipm.output_height if ipm else 800
        empty = np.zeros((height, width), dtype=np.uint8)
        if transformed is None:
            transformed = {name: empty.copy() for name in MASK_NAMES}
        if main_yellow is None:
            main_yellow = empty.copy()
        result = GeometryResult(reason=reason)
        if ipm is not None and source is not None:
            try:
                roi_data = ipm.planning_roi(self.config)
                roi = (roi_data[2], roi_data[3])
            except GeometryError:
                roi = (0, height - 1)
            self._publish_result(
                source,
                transformed,
                main_yellow,
                [],
                result,
                status_update,
                start,
                roi,
            )
        else:
            if source is not None:
                frame_id = str(source.header.frame_id)
                overlay = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(
                    overlay,
                    reason,
                    (30, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                empty_images = {
                    "ipm_boundary": empty,
                    "ipm_yellow": empty,
                    "ipm_green": empty,
                    "main_yellow": empty,
                    "selected_boundary": empty,
                    "centerline_raw": empty,
                    "centerline": empty,
                    "overlay": overlay,
                }
                for name, image in empty_images.items():
                    encoding = "bgr8" if name == "overlay" else "mono8"
                    self.image_publishers[name].publish(
                        ImageCodec.to_message(
                            image, encoding, source, frame_id
                        )
                    )
                    self.frames.update(
                        name, image, self.config.web_gui_jpeg_quality
                    )
            self.path_publisher.publish(PathConverter.empty(ipm, source))
            with self.lock:
                self.status.update(
                    {
                        **status_update,
                        "centerline_mode": "invalid",
                        "centerline_valid": False,
                        "centerline_reason": reason,
                        "centerline_confidence": 0.0,
                        "raw_centerline_point_count": 0,
                        "final_centerline_point_count": 0,
                        "processing_time_ms": (time.perf_counter() - start) * 1000.0,
                        "last_error": self.ipm_error or reason,
                    }
                )
            self._publish_status()

    def _publish_status(self) -> None:
        message = String()
        message.data = json.dumps(
            self.api_status(), ensure_ascii=False, separators=(",", ":")
        )
        self.status_publisher.publish(message)

    def api_status(self) -> Dict[str, Any]:
        with self.lock:
            return copy.deepcopy(self.status)

    def _apply_web_values(self, values: Dict[str, Any]) -> None:
        candidate = copy.deepcopy(self.config)
        for name, raw in values.items():
            if name not in self.WEB_EDITABLE:
                raise GeometryError(f"不允许修改参数: {name}")
            current = getattr(candidate, name)
            if isinstance(current, bool):
                if isinstance(raw, str):
                    value = raw.lower() in ("true", "1", "yes", "on")
                else:
                    value = bool(raw)
            elif isinstance(current, int):
                value = int(float(raw))
            else:
                value = float(raw)
            setattr(candidate, name, value)
        candidate.validate()
        if self.ipm is not None:
            self.ipm.planning_roi(candidate)
        self.config = candidate

    def handle_api(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            if path == "/api/set":
                self._apply_web_values(payload)
            elif path == "/api/reload_ipm":
                self._reload_ipm(required=False)
                self.status = self._initial_status()
                self._initialize_frames()
            elif path == "/api/save":
                ipm_path = self.ipm.path if self.ipm else ""
                saved = self.parameter_store.save(self.config, ipm_path)
                return {"ok": True, "message": "参数保存成功", "path": str(saved)}
            elif path == "/api/load":
                self.config = self.parameter_store.load(self.config)
                self.history = deque(
                    list(self.history)[-self.config.history_max_frames :],
                    maxlen=self.config.history_max_frames,
                )
                if self.ipm is not None:
                    self.ipm.planning_roi(self.config)
            elif path == "/api/reset_defaults":
                profile = self.config.profile_name
                ipm_path = self.config.ipm_config_path
                self.config = copy.deepcopy(self.defaults)
                self.config.profile_name = profile
                self.config.ipm_config_path = ipm_path
                self.config.validate()
            else:
                raise GeometryError(f"未知 API: {path}")
            return {"ok": True, "message": "操作成功"}

    def destroy_node(self) -> bool:
        self.web.stop()
        return super().destroy_node()


def _synthetic_ipm() -> IpmConfig:
    """自测使用的 600x800 单位 Homography 配置。"""
    return IpmConfig(
        path="synthetic",
        image_width=600,
        image_height=800,
        homography_matrix=np.eye(3),
        inverse_homography_matrix=np.eye(3),
        output_width=600,
        output_height=800,
        meter_per_pixel=0.005,
        vehicle_center_x_px=300.0,
        vehicle_origin_y_px=799.0,
        vehicle_reference_point_name="base_link",
        board_width_m=0.50,
        board_length_m=0.70,
        board_near_edge_distance_m=0.30,
        board_center_lateral_offset_m=0.0,
        expected_lane_width_m=0.50,
        single_boundary_center_offset_m=0.25,
    )


def run_self_tests() -> None:
    """覆盖同步、IPM、黄区、单双边界、拟合和米制路径的合成测试。"""
    config = GeometryConfig(
        main_yellow_min_area_px=50,
        main_yellow_min_vertical_span_px=10,
        centerline_min_forward_span_m=0.20,
    )
    config.validate()
    ipm = _synthetic_ipm()
    forward_min, forward_max, y_min, y_max = ipm.planning_roi(config)
    assert abs(forward_min - 0.30) < 1.0e-9
    assert abs(forward_max - 1.00) < 1.0e-9
    assert (y_min, y_max) == (599, 739)

    # 1/2：IPM 配置加载和版本拒绝。
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "test_ipm.json"
        document = {
            "version": IPM_INTERFACE_VERSION,
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
        path.write_text(json.dumps(document), encoding="utf-8")
        loader_config = copy.deepcopy(config)
        loader_config.ipm_config_path = str(path)
        loaded = IpmConfigLoader(Path(__file__)).load(loader_config)
        assert loaded.vehicle_reference_point_name == "base_link"
        document["version"] = "wrong"
        path.write_text(json.dumps(document), encoding="utf-8")
        try:
            IpmConfigLoader(Path(__file__)).load(loader_config)
        except GeometryError:
            pass
        else:
            raise AssertionError("错误 IPM 版本未被拒绝")

    # 3/4：三路精确同步、不同 stamp 不混合、缓存有界。
    def _message(sec: int, nanosec: int = 2, frame: str = "camera") -> Any:
        return type(
            "Message",
            (),
            {
                "header": type(
                    "Header",
                    (),
                    {
                        "stamp": type(
                            "Stamp",
                            (),
                            {"sec": sec, "nanosec": nanosec},
                        )(),
                        "frame_id": frame,
                    },
                )()
            },
        )()

    synchronizer = ExactStampSynchronizer(2, 0.5)
    packet = MaskPacket(
        np.zeros((2, 2), np.uint8), _message(1), time.monotonic()
    )
    assert synchronizer.add("boundary", packet) is None
    assert synchronizer.add("yellow", packet) is None
    synchronized = synchronizer.add("green", packet)
    assert synchronized is not None and len(synchronized[1]) == 3
    mismatch_sync = ExactStampSynchronizer(2, 0.5)
    assert mismatch_sync.add(
        "boundary",
        MaskPacket(np.zeros((2, 2), np.uint8), _message(1), time.monotonic()),
    ) is None
    assert mismatch_sync.add(
        "yellow",
        MaskPacket(np.zeros((2, 2), np.uint8), _message(2), time.monotonic()),
    ) is None
    assert mismatch_sync.add(
        "green",
        MaskPacket(np.zeros((2, 2), np.uint8), _message(1), time.monotonic()),
    ) is None
    for sec in (3, 4, 5):
        mismatch_sync.add(
            "boundary",
            MaskPacket(
                np.zeros((2, 2), np.uint8),
                _message(sec),
                time.monotonic(),
            ),
        )
    assert mismatch_sync.sizes()["boundary"] <= 2
    assert mismatch_sync.drop_count >= 1

    # 5：最近邻二值变换只产生 0/255。
    binary = np.zeros((800, 600), np.uint8)
    binary[620:700, 250:350] = 255
    transformed = MaskIpmTransformer.transform(binary, ipm)
    assert set(np.unique(transformed)).issubset({0, 255})

    # 6：主黄色优先选择近场中心种子连接分量。
    yellow = np.zeros((800, 600), np.uint8)
    yellow[599:740, 250:351] = 255
    yellow[599:680, 20:90] = 255
    main_yellow, selection = MainYellowSelector.select(
        yellow, ipm, config, (y_min, y_max)
    )
    assert selection["main_yellow_valid"]
    assert main_yellow[700, 300] == 255 and main_yellow[630, 50] == 0

    # 构造相距 100 px 的双直线边界和黄色道路。
    boundary = np.zeros_like(yellow)
    cv2.line(boundary, (250, 599), (250, 739), 255, 2)
    cv2.line(boundary, (350, 599), (350, 739), 255, 2)
    green = np.zeros_like(yellow)
    curves, _ = BoundaryComponentAnalyzer.analyze(
        boundary, ipm, config, (y_min, y_max)
    )
    for curve in curves:
        BoundaryRoadSideClassifier.classify_curve(
            curve, main_yellow, green, config, 10
        )
    # 7/8/11：宽度约 100，中心为 300，黄色侧方向相对。
    dual = DualBoundaryPlanner.plan(
        curves, main_yellow, ipm, config, (y_min, y_max)
    )
    assert dual is not None
    assert abs(dual.measured_width_mean_px - 100.0) < 3.0
    assert abs(float(np.mean(dual.raw_points[:, 0])) - 300.0) < 2.0

    # 9：单条竖直边界向黄色侧沿法线偏移约 50 px。
    vertical = [curve for curve in curves if curve.centroid[0] < 300]
    single = SingleBoundaryPlanner.plan(
        vertical, main_yellow, green, ipm, config, (y_min, y_max)
    )
    assert single is not None and len(single.raw_points) >= 6
    assert abs(float(np.mean(single.raw_points[:, 0])) - 300.0) < 3.0

    # 10：斜边界偏移同时改变 x/y，证明不是固定水平平移。
    diagonal_boundary = np.zeros_like(yellow)
    cv2.line(diagonal_boundary, (240, 599), (270, 739), 255, 2)
    diagonal_yellow = np.zeros_like(yellow)
    polygon = np.asarray([[240, 599], [340, 599], [370, 739], [270, 739]])
    cv2.fillPoly(diagonal_yellow, [polygon], 255)
    diagonal_curves, _ = BoundaryComponentAnalyzer.analyze(
        diagonal_boundary, ipm, config, (y_min, y_max)
    )
    for curve in diagonal_curves:
        BoundaryRoadSideClassifier.classify_curve(
            curve, diagonal_yellow, green, config, 10
        )
    diagonal_single = SingleBoundaryPlanner.plan(
        diagonal_curves,
        diagonal_yellow,
        green,
        ipm,
        config,
        (y_min, y_max),
    )
    assert diagonal_single is not None and len(diagonal_single.raw_points) >= 6
    source_y = diagonal_single.raw_points[:, 1]
    source_x = diagonal_curves[0].x_at(source_y)
    assert np.std(diagonal_single.raw_points[:, 1] - np.round(source_y)) < 1.0
    assert np.mean(diagonal_single.raw_points[:, 0] - source_x) > 40.0
    diagonal_slope = diagonal_curves[0].dx_dy(670.0)
    diagonal_normal = np.asarray([1.0, -diagonal_slope])
    diagonal_normal /= np.linalg.norm(diagonal_normal)
    assert abs(diagonal_slope) > 0.05
    assert abs(diagonal_normal[1] * ipm.single_boundary_offset_px) > 1.0

    # 12：两侧黄色相同则道路侧必须不确定。
    full_yellow = np.full_like(yellow, 255)
    uncertain = copy.deepcopy(curves[0])
    BoundaryRoadSideClassifier.classify_curve(
        uncertain, full_yellow, green, config, 10
    )
    assert uncertain.inward_sign == 0

    # 13/14：二次拟合并过滤离群点。
    fit_y = np.arange(0, 100, dtype=float)
    fit_x = 0.01 * fit_y**2 + 2.0
    fit_x[20] += 80
    coefficients, keep, residual = RobustPolynomialFitter.fit(
        fit_y, fit_x, 2, 8, 5.0, 3
    )
    assert not keep[20] and residual < 1.0
    assert abs(coefficients[0] - 0.01) < 1.0e-3

    # 15/16：像素转 base_link 米制坐标，forward 单调。
    points = np.asarray([[300, 739], [300, 699], [300, 659]], dtype=float)
    metric = PathConverter.metric_samples(points, ipm)
    assert abs(metric[0][0] - 0.30) < 1.0e-9
    assert all(metric[i + 1][0] > metric[i][0] for i in range(len(metric) - 1))

    # 17：空输入必须 invalid；18：不同分辨率整组拒绝。
    final, reason, _, _ = CenterlineSmoother.smooth(
        np.empty((0, 2)), main_yellow, ipm, config, (y_min, y_max)
    )
    assert len(final) == 0 and reason == "centerline_too_few_points"
    wrong_group = {
        name: MaskPacket(
            np.zeros((10, 10), np.uint8),
            _message(10),
            time.monotonic(),
        )
        for name in MASK_NAMES
    }
    try:
        validate_mask_group_size(wrong_group, ipm)
    except GeometryError:
        pass
    else:
        raise AssertionError("不同分辨率未被拒绝")
    print("SELF-TEST PASSED: 18 geometry checks")


def main(args: Optional[List[str]] = None) -> None:
    """运行自测或启动第四阶段 ROS 节点。"""
    if "--self-test" in sys.argv:
        run_self_tests()
        return
    rclpy.init(args=args)
    node: Optional[LaneGeometryPlannerNode] = None
    try:
        node = LaneGeometryPlannerNode()
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
