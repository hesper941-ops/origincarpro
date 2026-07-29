#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
黄绿道路区域分割与公共边界提取工具。

本程序只负责黄绿道路区域分割与分界线提取，不包含中心线规划、路径规划和运动控制功能。
程序独立读取第一阶段保存的 JSON，不导入第一阶段脚本，也不会修改阈值 JSON。

USB 相机启动示例：

    python3 tools/usb_camera_publisher.py \
      --ros-args \
      -p device:=/dev/video0 \
      -p image_topic:=/image_out \
      -p width:=640 \
      -p height:=480 \
      -p fps:=15.0

第二阶段启动示例：

    python3 gxb_test/2_green_yellow_segmentation.py \
      --ros-args \
      -p image_topic:=/image_out \
      -p profile_name:=usb_camera \
      -p web_gui_enable:=true \
      -p web_gui_host:=0.0.0.0 \
      -p web_gui_port:=8090

浏览器访问 http://小车IP:8090。黄绿 HSV 需要修改时，请回到第一阶段工具
重新标定、保存，再在本页面点击“重新加载阈值配置”。

中心黄色种子、边界搜索距离、缝隙宽度和拓扑修复参数均需要现场验证。
"""

import copy
import json
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from sensor_msgs.msg import Image
from std_msgs.msg import String


SEGMENTATION_CONFIG_VERSION = 1
PROFILE_ALLOWED_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
)
BOUNDARY_OVERLAY_COLOR = (255, 0, 255)
INTERFACE_VERSION = "gxb_boundary_v1"
THIRD_STAGE_BOUNDARY_TOPIC = "/gxb_test/final_boundary_mask"
THIRD_STAGE_YELLOW_TOPIC = "/gxb_test/yellow_mask_final"
THIRD_STAGE_GREEN_TOPIC = "/gxb_test/green_mask_final"
THIRD_STAGE_STATUS_TOPIC = "/gxb_test/segmentation_status"
BOUNDARY_REGION_TOPIC = "/gxb_test/boundary_region_mask"


# 第二阶段参数不包含 HSV；HSV 始终来自第一阶段 JSON。
DEFAULT_SEGMENTATION_CONFIG: Dict[str, object] = {
    "boundary_search_radius_px": 10,
    "boundary_contour_thickness_px": 2,
    "gap_centerline_enable": True,
    "gap_max_distance_to_color_px": 14,
    "gap_distance_balance_tolerance_px": 2.5,
    "boundary_close_kernel_size": 3,
    "boundary_close_iterations": 1,
    "boundary_min_component_pixels": 30,
    "boundary_max_component_area_ratio": 0.08,
    "boundary_min_span_px": 20,
    "boundary_min_valid_pixels": 40,
    "boundary_overlay_thickness": 3,
    "final_boundary_thickness_px": 1,
    "prefer_center_connected_yellow": True,
    "center_seed_x_min_ratio": 0.30,
    "center_seed_x_max_ratio": 0.70,
    "center_seed_y_min_ratio": 0.75,
    "center_seed_y_max_ratio": 1.00,
    "topology_repair_enable": False,
    "repair_max_component_area_px": 800,
    "repair_max_area_ratio": 0.005,
    "same_color_contact_ratio": 0.80,
    "opposite_color_max_contact_ratio": 0.10,
    "preserve_dark_unknown": True,
    "dark_unknown_v_max": 40,
    "repair_max_iterations": 1,
}


PARAMETER_SCHEMA: Dict[str, Tuple[str, object, object, object, str]] = {
    "boundary_search_radius_px": ("公共边界", 1, 50, 1, "number"),
    "boundary_contour_thickness_px": ("公共边界", 1, 10, 1, "number"),
    "gap_centerline_enable": ("缝隙中线", False, True, None, "bool"),
    "gap_max_distance_to_color_px": ("缝隙中线", 1, 50, 1, "number"),
    "gap_distance_balance_tolerance_px": (
        "缝隙中线", 0.0, 20.0, 0.5, "number"
    ),
    "boundary_close_kernel_size": ("边界清理", 1, 9, 2, "number"),
    "boundary_close_iterations": ("边界清理", 0, 5, 1, "number"),
    "boundary_min_component_pixels": ("边界清理", 1, 5000, 5, "number"),
    "boundary_max_component_area_ratio": (
        "边界清理", 0.001, 0.5, 0.005, "number"
    ),
    "boundary_min_span_px": ("边界清理", 1, 500, 1, "number"),
    "boundary_min_valid_pixels": ("边界清理", 1, 10000, 5, "number"),
    "boundary_overlay_thickness": ("边界清理", 1, 10, 1, "number"),
    "final_boundary_thickness_px": ("边界清理", 1, 3, 1, "number"),
    "prefer_center_connected_yellow": (
        "主黄色区域", False, True, None, "bool"
    ),
    "center_seed_x_min_ratio": ("主黄色区域", 0.0, 0.95, 0.01, "number"),
    "center_seed_x_max_ratio": ("主黄色区域", 0.05, 1.0, 0.01, "number"),
    "center_seed_y_min_ratio": ("主黄色区域", 0.0, 0.95, 0.01, "number"),
    "center_seed_y_max_ratio": ("主黄色区域", 0.05, 1.0, 0.01, "number"),
    "topology_repair_enable": ("拓扑修复", False, True, None, "bool"),
    "repair_max_component_area_px": ("拓扑修复", 1, 10000, 10, "number"),
    "repair_max_area_ratio": ("拓扑修复", 0.0001, 0.05, 0.0005, "number"),
    "same_color_contact_ratio": ("拓扑修复", 0.0, 1.0, 0.05, "number"),
    "opposite_color_max_contact_ratio": (
        "拓扑修复", 0.0, 1.0, 0.05, "number"
    ),
    "preserve_dark_unknown": ("拓扑修复", False, True, None, "bool"),
    "dark_unknown_v_max": ("拓扑修复", 0, 255, 1, "number"),
    "repair_max_iterations": ("拓扑修复", 1, 3, 1, "number"),
}


class APIError(ValueError):
    """带参数名和 HTTP 状态码的接口错误。"""

    def __init__(
        self, message: str, parameter: str = "", status: int = 400
    ) -> None:
        super().__init__(message)
        self.parameter = parameter
        self.status = status


@dataclass
class SegmentationConfig:
    """第二阶段边界和可选拓扑修复参数。"""

    values: Dict[str, object] = field(
        default_factory=lambda: copy.deepcopy(
            DEFAULT_SEGMENTATION_CONFIG
        )
    )

    @staticmethod
    def _odd(value: object) -> int:
        size = max(1, int(round(float(value))))
        return size if size % 2 else size + 1

    @classmethod
    def validate(cls, candidate: Dict[str, object]) -> Dict[str, object]:
        """完整校验候选参数，防止图像回调读取非法组合。"""
        cfg = copy.deepcopy(candidate)
        cfg["boundary_close_kernel_size"] = cls._odd(
            cfg["boundary_close_kernel_size"]
        )

        def require(condition: bool, message: str, parameter: str) -> None:
            if not condition:
                raise APIError(message, parameter)

        positive_names = (
            "boundary_search_radius_px",
            "boundary_contour_thickness_px",
            "gap_max_distance_to_color_px",
            "boundary_min_component_pixels",
            "boundary_min_span_px",
            "boundary_min_valid_pixels",
            "boundary_overlay_thickness",
            "final_boundary_thickness_px",
            "repair_max_component_area_px",
            "repair_max_iterations",
        )
        for name in positive_names:
            require(float(cfg[name]) > 0, f"{name} 必须大于 0", name)
        require(
            int(cfg["boundary_close_kernel_size"]) <= 9,
            "boundary_close_kernel_size 不能超过 9",
            "boundary_close_kernel_size",
        )
        require(
            0 <= int(cfg["boundary_close_iterations"]) <= 5,
            "boundary_close_iterations 必须位于 0 到 5",
            "boundary_close_iterations",
        )
        require(
            float(cfg["gap_distance_balance_tolerance_px"]) >= 0.0,
            "缝隙距离平衡容差不能为负数",
            "gap_distance_balance_tolerance_px",
        )
        require(
            0.0 < float(cfg["boundary_max_component_area_ratio"]) <= 1.0,
            "boundary_max_component_area_ratio 必须位于 (0, 1]",
            "boundary_max_component_area_ratio",
        )
        for axis in ("x", "y"):
            minimum = float(cfg[f"center_seed_{axis}_min_ratio"])
            maximum = float(cfg[f"center_seed_{axis}_max_ratio"])
            require(
                0.0 <= minimum < maximum <= 1.0,
                f"中心种子 {axis} 范围必须满足 0 <= min < max <= 1",
                f"center_seed_{axis}_min_ratio",
            )
        require(
            0.0 < float(cfg["repair_max_area_ratio"]) <= 1.0,
            "repair_max_area_ratio 必须位于 (0, 1]",
            "repair_max_area_ratio",
        )
        for name in (
            "same_color_contact_ratio",
            "opposite_color_max_contact_ratio",
        ):
            require(
                0.0 <= float(cfg[name]) <= 1.0,
                f"{name} 必须位于 [0, 1]",
                name,
            )
        require(
            0 <= int(cfg["dark_unknown_v_max"]) <= 255,
            "dark_unknown_v_max 必须位于 0 到 255",
            "dark_unknown_v_max",
        )
        return cfg

    def snapshot(self) -> Dict[str, object]:
        return copy.deepcopy(self.values)


@dataclass
class ThresholdConfig:
    """第一阶段 JSON 中第二阶段真正需要的只读阈值。"""

    loaded: bool = False
    path: str = ""
    profile_name: str = ""
    image_topic: str = ""
    frame_width: int = 0
    frame_height: int = 0
    saved_at: str = ""
    version: object = None
    roi_y_start_ratio: float = 0.0
    roi_y_end_ratio: float = 1.0
    yellow: Dict[str, object] = field(default_factory=dict)
    green: Dict[str, object] = field(default_factory=dict)
    morphology: Dict[str, object] = field(default_factory=dict)
    error: str = ""
    mtime_ns: int = -1

    def snapshot(self) -> "ThresholdConfig":
        return copy.deepcopy(self)


@dataclass
class ComponentStats:
    """连通域筛选统计。"""

    raw_count: int = 0
    kept_count: int = 0
    main_found: bool = False


@dataclass
class BoundaryResult:
    """公共边界中间结果和统计。"""

    yellow_edge_near_green: np.ndarray
    green_edge_near_yellow: np.ndarray
    gap_centerline: np.ndarray
    candidate: np.ndarray
    boundary_region: np.ndarray
    final: np.ndarray
    component_count: int
    components: List[Dict[str, object]]
    estimated_mean_thickness_px: float
    valid: bool
    reason: str


@dataclass
class SegmentationResult:
    """单帧分割、边界和 overlay。"""

    raw_bgr: np.ndarray
    overlay: np.ndarray
    yellow_raw: np.ndarray
    green_raw: np.ndarray
    unknown_raw: np.ndarray
    yellow_final: np.ndarray
    green_final: np.ndarray
    unknown_final: np.ndarray
    boundary: BoundaryResult
    roi_start: int
    roi_end: int
    overlap_pixels: int
    yellow_stats: ComponentStats
    green_stats: ComponentStats


class ImageCodec:
    """不依赖 cv_bridge 的 ROS Image 编解码。"""

    CHANNELS = {
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
    }

    @classmethod
    def to_bgr(cls, msg: Image) -> np.ndarray:
        """处理 msg.step 行填充并统一转换到 OpenCV BGR。"""
        encoding = str(msg.encoding).strip().lower()
        if encoding not in cls.CHANNELS:
            raise ValueError(f"不支持的图像编码：{msg.encoding}")
        width = int(msg.width)
        height = int(msg.height)
        channels = cls.CHANNELS[encoding]
        row_bytes = width * channels
        step = int(msg.step) or row_bytes
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        required = step * height
        if width <= 0 or height <= 0:
            raise ValueError(f"图像尺寸无效：{width}x{height}")
        if step < row_bytes or raw.size < required:
            raise ValueError(
                f"图像缓冲区无效：step={step}, bytes={raw.size}"
            )
        packed = raw[:required].reshape(height, step)[:, :row_bytes]
        if channels == 1:
            mono = np.ascontiguousarray(packed.reshape(height, width))
            return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
        image = np.ascontiguousarray(
            packed.reshape(height, width, channels)
        )
        conversion = {
            "rgb8": cv2.COLOR_RGB2BGR,
            "bgra8": cv2.COLOR_BGRA2BGR,
            "rgba8": cv2.COLOR_RGBA2BGR,
        }.get(encoding)
        return cv2.cvtColor(image, conversion) if conversion else image

    @staticmethod
    def to_message(
        image: np.ndarray, encoding: str, source: Image
    ) -> Image:
        """手工封装 mono8/bgr8 sensor_msgs Image。"""
        packed = np.ascontiguousarray(image, dtype=np.uint8)
        output = Image()
        output.header = source.header
        output.height = int(packed.shape[0])
        output.width = int(packed.shape[1])
        output.encoding = encoding
        output.is_bigendian = 0
        channels = 1 if packed.ndim == 2 else int(packed.shape[2])
        output.step = output.width * channels
        output.data = packed.tobytes()
        return output

    @staticmethod
    def encode_jpeg(image: np.ndarray, quality: int) -> bytes:
        """保持原始分辨率编码 Web JPEG。"""
        ok, encoded = cv2.imencode(
            ".jpg",
            np.ascontiguousarray(image),
            [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(quality, 1, 100))],
        )
        if not ok:
            raise ValueError("JPEG 编码失败")
        return encoded.tobytes()


class ThresholdConfigLoader:
    """读取并校验第一阶段实际保存的 JSON 结构。"""

    REQUIRED_COLOR_KEYS = (
        "h_min",
        "h_max",
        "s_min",
        "s_max",
        "v_min",
        "v_max",
        "prefer_bottom_connected",
    )
    REQUIRED_MORPH_KEYS = (
        "open_kernel_size",
        "close_kernel_size",
        "open_iterations",
        "close_iterations",
        "min_component_area",
        "bottom_band_pixels",
    )

    @staticmethod
    def validate_profile(profile_name: str) -> str:
        """限制 profile 为安全文件名。"""
        if (
            not profile_name
            or ".." in profile_name
            or any(
                char not in PROFILE_ALLOWED_CHARS for char in profile_name
            )
        ):
            raise APIError(
                "profile_name 只允许字母、数字、下划线和短横线",
                "profile_name",
            )
        return profile_name

    @staticmethod
    def resolve_path(
        script_dir: Path, profile_name: str, explicit_path: str
    ) -> Path:
        """显式 JSON 优先，否则使用第一阶段 profile 路径。"""
        if explicit_path:
            return Path(explicit_path).expanduser().resolve()
        safe_profile = ThresholdConfigLoader.validate_profile(profile_name)
        return (
            script_dir
            / "config"
            / f"{safe_profile}_green_yellow.json"
        )

    @classmethod
    def load(cls, path: Path) -> ThresholdConfig:
        """加载阈值；任何错误都包装为 config_loaded=false 状态。"""
        if not path.exists():
            return ThresholdConfig(
                loaded=False,
                path=str(path),
                error=f"CONFIG MISSING：{path}",
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON 顶层必须是对象")
            roi = payload.get("roi")
            yellow = payload.get("yellow")
            green = payload.get("green")
            morphology = payload.get("morphology")
            if not all(
                isinstance(value, dict)
                for value in (roi, yellow, green, morphology)
            ):
                raise ValueError("缺少 roi/yellow/green/morphology 对象")
            for key in ("y_start_ratio", "y_end_ratio"):
                if key not in roi:
                    raise ValueError(f"roi 缺少 {key}")
            for color_name, color in (
                ("yellow", yellow),
                ("green", green),
            ):
                missing = [
                    key
                    for key in cls.REQUIRED_COLOR_KEYS
                    if key not in color
                ]
                if missing:
                    raise ValueError(
                        f"{color_name} 缺少字段：{missing}"
                    )
                for channel, maximum in (
                    ("h", 179),
                    ("s", 255),
                    ("v", 255),
                ):
                    low = int(color[f"{channel}_min"])
                    high = int(color[f"{channel}_max"])
                    if not 0 <= low <= high <= maximum:
                        raise ValueError(
                            f"{color_name} {channel} 范围非法"
                        )
            missing_morph = [
                key
                for key in cls.REQUIRED_MORPH_KEYS
                if key not in morphology
            ]
            if missing_morph:
                raise ValueError(
                    f"morphology 缺少字段：{missing_morph}"
                )
            roi_start = float(roi["y_start_ratio"])
            roi_end = float(roi["y_end_ratio"])
            if not (
                0.0 <= roi_start < roi_end <= 1.0
                and roi_end - roi_start >= 0.10 - 1e-9
            ):
                raise ValueError("ROI 范围非法")
            for kernel_name in (
                "open_kernel_size",
                "close_kernel_size",
            ):
                kernel = int(morphology[kernel_name])
                if kernel < 1:
                    raise ValueError(f"{kernel_name} 必须大于 0")
                morphology[kernel_name] = (
                    kernel if kernel % 2 else kernel + 1
                )
            if (
                int(morphology["open_iterations"]) < 0
                or int(morphology["close_iterations"]) < 0
                or int(morphology["min_component_area"]) < 0
                or int(morphology["bottom_band_pixels"]) < 1
            ):
                raise ValueError("形态学参数非法")
            return ThresholdConfig(
                loaded=True,
                path=str(path),
                profile_name=str(payload.get("profile_name", "")),
                image_topic=str(payload.get("image_topic", "")),
                frame_width=int(payload.get("frame_width", 0)),
                frame_height=int(payload.get("frame_height", 0)),
                saved_at=str(payload.get("saved_at", "")),
                version=payload.get("version"),
                roi_y_start_ratio=roi_start,
                roi_y_end_ratio=roi_end,
                yellow=copy.deepcopy(yellow),
                green=copy.deepcopy(green),
                morphology=copy.deepcopy(morphology),
                error="",
                mtime_ns=path.stat().st_mtime_ns,
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            return ThresholdConfig(
                loaded=False,
                path=str(path),
                error=f"CONFIG INVALID：{exc}",
            )


class GreenYellowSegmenter:
    """按第一阶段阈值执行基础分割与保守拓扑修复。"""

    @staticmethod
    def _morph(
        mask: np.ndarray, morphology: Dict[str, object]
    ) -> np.ndarray:
        output = mask
        open_iterations = int(morphology["open_iterations"])
        close_iterations = int(morphology["close_iterations"])
        if open_iterations > 0:
            size = int(morphology["open_kernel_size"])
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (size, size)
            )
            output = cv2.morphologyEx(
                output,
                cv2.MORPH_OPEN,
                kernel,
                iterations=open_iterations,
            )
        if close_iterations > 0:
            size = int(morphology["close_kernel_size"])
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (size, size)
            )
            output = cv2.morphologyEx(
                output,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=close_iterations,
            )
        return output

    @staticmethod
    def _filter_yellow(
        mask: np.ndarray,
        morphology: Dict[str, object],
        cfg: Dict[str, object],
    ) -> Tuple[np.ndarray, ComponentStats]:
        """优先保留触及 ROI 底部中央种子的主黄色连通域。"""
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        raw_count = max(0, count - 1)
        minimum_area = int(morphology["min_component_area"])
        qualified = [
            label
            for label in range(1, count)
            if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area
        ]
        if not qualified:
            return np.zeros_like(mask), ComponentStats(raw_count, 0, False)

        height, width = mask.shape
        x0 = int(width * float(cfg["center_seed_x_min_ratio"]))
        x1 = int(width * float(cfg["center_seed_x_max_ratio"]))
        y0 = int(height * float(cfg["center_seed_y_min_ratio"]))
        y1 = int(height * float(cfg["center_seed_y_max_ratio"]))
        seed_labels = set(
            int(value)
            for value in np.unique(labels[y0:y1, x0:x1])
            if int(value) in qualified
        )
        main_found = bool(seed_labels)
        if bool(cfg["prefer_center_connected_yellow"]) and seed_labels:
            kept = list(seed_labels)
            main_area = max(
                int(stats[label, cv2.CC_STAT_AREA]) for label in kept
            )
            # 额外保留与主区同量级的大区域，支持多条真实黄绿接触；
            # 小于主区 35% 的远处黄色杂物仍会被删除。
            kept.extend(
                label
                for label in qualified
                if label not in seed_labels
                and int(stats[label, cv2.CC_STAT_AREA])
                >= 0.35 * main_area
            )
        else:
            main_label = max(
                qualified,
                key=lambda label: int(
                    stats[label, cv2.CC_STAT_AREA]
                ),
            )
            main_area = int(stats[main_label, cv2.CC_STAT_AREA])
            kept = [
                label
                for label in qualified
                if int(stats[label, cv2.CC_STAT_AREA])
                >= 0.50 * main_area
            ]
        output = np.zeros_like(mask)
        for label in kept:
            output[labels == label] = 255
        return output, ComponentStats(raw_count, len(kept), main_found)

    @staticmethod
    def _filter_green(
        mask: np.ndarray, morphology: Dict[str, object]
    ) -> Tuple[np.ndarray, ComponentStats]:
        """绿色不要求触底，保留所有达到面积要求的主要区域。"""
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        raw_count = max(0, count - 1)
        minimum_area = int(morphology["min_component_area"])
        kept = [
            label
            for label in range(1, count)
            if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area
        ]
        output = np.zeros_like(mask)
        for label in kept:
            output[labels == label] = 255
        return output, ComponentStats(raw_count, len(kept), False)

    @staticmethod
    def _repair_unknown(
        yellow: np.ndarray,
        green: np.ndarray,
        unknown: np.ndarray,
        hsv_roi: np.ndarray,
        cfg: Dict[str, object],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """只填充被同色强包围的小孔洞，保留暗色和不确定区域。"""
        repaired_yellow = yellow.copy()
        repaired_green = green.copy()
        roi_area = max(1, unknown.shape[0] * unknown.shape[1])
        maximum_area = min(
            int(cfg["repair_max_component_area_px"]),
            int(roi_area * float(cfg["repair_max_area_ratio"])),
        )
        if maximum_area <= 0:
            return repaired_yellow, repaired_green
        contact_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (3, 3)
        )
        for _ in range(int(cfg["repair_max_iterations"])):
            changed = False
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                unknown, connectivity=8
            )
            for label in range(1, count):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area <= 0 or area > maximum_area:
                    continue
                component = np.where(labels == label, 255, 0).astype(
                    np.uint8
                )
                if bool(cfg["preserve_dark_unknown"]):
                    mean_v = cv2.mean(
                        hsv_roi[:, :, 2], mask=component
                    )[0]
                    if mean_v <= float(cfg["dark_unknown_v_max"]):
                        continue
                ring = cv2.subtract(
                    cv2.dilate(component, contact_kernel, iterations=1),
                    component,
                )
                ring_pixels = max(1, cv2.countNonZero(ring))
                yellow_ratio = (
                    cv2.countNonZero(
                        cv2.bitwise_and(ring, repaired_yellow)
                    )
                    / ring_pixels
                )
                green_ratio = (
                    cv2.countNonZero(
                        cv2.bitwise_and(ring, repaired_green)
                    )
                    / ring_pixels
                )
                same_min = float(cfg["same_color_contact_ratio"])
                opposite_max = float(
                    cfg["opposite_color_max_contact_ratio"]
                )
                if yellow_ratio >= same_min and green_ratio <= opposite_max:
                    repaired_yellow[component > 0] = 255
                    unknown[component > 0] = 0
                    changed = True
                elif green_ratio >= same_min and yellow_ratio <= opposite_max:
                    repaired_green[component > 0] = 255
                    unknown[component > 0] = 0
                    changed = True
            if not changed:
                break
        return repaired_yellow, repaired_green

    @classmethod
    def segment(
        cls,
        bgr: np.ndarray,
        threshold: ThresholdConfig,
        cfg: Dict[str, object],
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
        int,
        int,
        ComponentStats,
        ComponentStats,
    ]:
        """在 ROI 内产生 raw/final 三分类 mask。"""
        height, width = bgr.shape[:2]
        roi_start = int(height * threshold.roi_y_start_ratio)
        roi_end = int(height * threshold.roi_y_end_ratio)
        roi_start = min(max(roi_start, 0), height - 1)
        roi_end = min(max(roi_end, roi_start + 1), height)
        hsv_roi = cv2.cvtColor(
            bgr[roi_start:roi_end, :], cv2.COLOR_BGR2HSV
        )

        def range_mask(color: Dict[str, object]) -> np.ndarray:
            lower = np.array(
                (color["h_min"], color["s_min"], color["v_min"]),
                dtype=np.uint8,
            )
            upper = np.array(
                (color["h_max"], color["s_max"], color["v_max"]),
                dtype=np.uint8,
            )
            return cv2.inRange(hsv_roi, lower, upper)

        yellow_base = cls._morph(
            range_mask(threshold.yellow), threshold.morphology
        )
        green_base = cls._morph(
            range_mask(threshold.green), threshold.morphology
        )
        yellow_roi, yellow_stats = cls._filter_yellow(
            yellow_base, threshold.morphology, cfg
        )
        green_roi, green_stats = cls._filter_green(
            green_base, threshold.morphology
        )
        overlap = cv2.bitwise_and(yellow_roi, green_roi)
        not_overlap = cv2.bitwise_not(overlap)
        yellow_roi = cv2.bitwise_and(yellow_roi, not_overlap)
        green_roi = cv2.bitwise_and(green_roi, not_overlap)
        unknown_roi = cv2.bitwise_not(
            cv2.bitwise_or(yellow_roi, green_roi)
        )
        yellow_final_roi = yellow_roi.copy()
        green_final_roi = green_roi.copy()
        if bool(cfg["topology_repair_enable"]):
            yellow_final_roi, green_final_roi = cls._repair_unknown(
                yellow_final_roi,
                green_final_roi,
                unknown_roi.copy(),
                hsv_roi,
                cfg,
            )
        unknown_final_roi = cv2.bitwise_not(
            cv2.bitwise_or(yellow_final_roi, green_final_roi)
        )

        def full(roi_mask: np.ndarray) -> np.ndarray:
            output = np.zeros((height, width), dtype=np.uint8)
            output[roi_start:roi_end, :] = roi_mask
            return output

        return (
            full(yellow_roi),
            full(green_roi),
            full(unknown_roi),
            full(yellow_final_roi),
            full(green_final_roi),
            full(unknown_final_roi),
            cv2.cvtColor(bgr[roi_start:roi_end, :], cv2.COLOR_BGR2HSV),
            roi_start,
            roi_end,
            int(cv2.countNonZero(overlap)),
            yellow_stats,
            green_stats,
        )


class BoundaryExtractor:
    """双向轮廓邻接与 unknown 等距中线公共边界提取。"""

    @staticmethod
    def _contour_mask(mask: np.ndarray, thickness: int) -> np.ndarray:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        output = np.zeros_like(mask)
        cv2.drawContours(output, contours, -1, 255, thickness)
        return output

    @staticmethod
    def _distance_to(mask: np.ndarray) -> np.ndarray:
        """distanceTransform 输入非零区域，返回到目标颜色的距离。"""
        return cv2.distanceTransform(
            cv2.bitwise_not(mask), cv2.DIST_L2, 5
        )

    @staticmethod
    def _close(mask: np.ndarray, cfg: Dict[str, object]) -> np.ndarray:
        """使用同一个小核参数连接轻微断裂。"""
        iterations = int(cfg["boundary_close_iterations"])
        if iterations <= 0:
            return mask.copy()
        size = int(cfg["boundary_close_kernel_size"])
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (size, size)
        )
        return cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=iterations,
        )

    @staticmethod
    def _filter_components(
        mask: np.ndarray,
        roi_area: int,
        cfg: Dict[str, object],
    ) -> Tuple[np.ndarray, List[Dict[str, object]]]:
        """只按像素、跨度和异常面积过滤，不选择道路位置。"""
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        output = np.zeros_like(mask)
        components: List[Dict[str, object]] = []
        maximum_area = roi_area * float(
            cfg["boundary_max_component_area_ratio"]
        )
        for label in range(1, count):
            pixel_count = int(stats[label, cv2.CC_STAT_AREA])
            bbox_x = int(stats[label, cv2.CC_STAT_LEFT])
            bbox_y = int(stats[label, cv2.CC_STAT_TOP])
            bbox_width = int(stats[label, cv2.CC_STAT_WIDTH])
            bbox_height = int(stats[label, cv2.CC_STAT_HEIGHT])
            span = max(bbox_width, bbox_height)
            if pixel_count < int(cfg["boundary_min_component_pixels"]):
                continue
            if pixel_count > maximum_area:
                continue
            if span < int(cfg["boundary_min_span_px"]):
                continue
            output[labels == label] = 255
            components.append({
                "id": len(components) + 1,
                "pixel_count": pixel_count,
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "centroid_x": round(float(centroids[label, 0]), 2),
                "centroid_y": round(float(centroids[label, 1]), 2),
            })
        return output, components

    @classmethod
    def extract(
        cls,
        yellow: np.ndarray,
        green: np.ndarray,
        unknown: np.ndarray,
        roi_start: int,
        roi_end: int,
        cfg: Dict[str, object],
    ) -> BoundaryResult:
        """分别生成宽候选区域和供第三阶段使用的窄分界线。"""
        thickness = int(cfg["boundary_contour_thickness_px"])
        final_thickness = int(cfg["final_boundary_thickness_px"])
        search_radius = float(cfg["boundary_search_radius_px"])
        yellow_contour = cls._contour_mask(yellow, thickness)
        green_contour = cls._contour_mask(green, thickness)
        thin_yellow_contour = cls._contour_mask(
            yellow, final_thickness
        )
        distance_to_yellow = cls._distance_to(yellow)
        distance_to_green = cls._distance_to(green)
        yellow_near_green = np.where(
            (yellow_contour > 0) & (distance_to_green <= search_radius),
            255,
            0,
        ).astype(np.uint8)
        green_near_yellow = np.where(
            (green_contour > 0) & (distance_to_yellow <= search_radius),
            255,
            0,
        ).astype(np.uint8)
        thin_yellow_near_green = np.where(
            (thin_yellow_contour > 0)
            & (distance_to_green <= search_radius),
            255,
            0,
        ).astype(np.uint8)

        gap = np.zeros_like(yellow)
        if bool(cfg["gap_centerline_enable"]):
            maximum = float(cfg["gap_max_distance_to_color_px"])
            tolerance = float(
                cfg["gap_distance_balance_tolerance_px"]
            )
            balance = np.abs(distance_to_yellow - distance_to_green)
            gap_region = (
                (unknown > 0)
                & (distance_to_yellow <= maximum)
                & (distance_to_green <= maximum)
                & (balance <= tolerance)
            )
            # 在 3x3 邻域中只保留距离差的局部最小值，把缝隙带压成
            # 一至数像素的近似中线，不依赖 opencv-contrib 骨架函数。
            local_minimum = cv2.erode(
                balance.astype(np.float32),
                np.ones((3, 3), dtype=np.uint8),
            )
            gap = np.where(
                gap_region & (balance <= local_minimum + 0.25),
                255,
                0,
            ).astype(np.uint8)

        candidate = cv2.bitwise_or(
            cv2.bitwise_or(yellow_near_green, green_near_yellow), gap
        )
        region_cleaned = cls._close(candidate, cfg)
        roi_area = max(1, (roi_end - roi_start) * yellow.shape[1])
        boundary_region, _region_components = cls._filter_components(
            region_cleaned, roi_area, cfg
        )

        # 正式接口只使用黄色道路侧细轮廓和缝隙中线；绿色侧轮廓不并入，
        # 从源头避免两条平行白线。
        support_size = max(3, int(round(search_radius)) * 2 + 1)
        yellow_support = cv2.dilate(
            thin_yellow_near_green,
            cv2.getStructuringElement(
                # 主要沿横向覆盖同一截面的黄色侧轮廓，避免把平行的
                # gap 中线重复并入；纵向只扩 3 像素，仍允许中线补断。
                cv2.MORPH_RECT, (support_size, 3)
            ),
            iterations=1,
        )
        gap_supplement = cv2.bitwise_and(
            gap, cv2.bitwise_not(yellow_support)
        )
        final_candidate = cv2.bitwise_or(
            thin_yellow_near_green, gap_supplement
        )
        final_cleaned = cls._close(final_candidate, cfg)
        final, components = cls._filter_components(
            final_cleaned, roi_area, cfg
        )
        component_count = len(components)
        final_pixels = int(cv2.countNonZero(final))
        if cv2.countNonZero(yellow) == 0:
            valid = False
            reason = "yellow_missing"
        elif cv2.countNonZero(green) == 0:
            valid = False
            reason = "green_missing"
        elif cv2.countNonZero(final_candidate) == 0:
            valid = False
            reason = "boundary_missing"
        elif component_count < 1:
            valid = False
            reason = "boundary_too_short"
        elif final_pixels < int(cfg["boundary_min_valid_pixels"]):
            valid = False
            reason = "boundary_too_short"
        else:
            valid = True
            reason = "valid"
        approximate_length = sum(
            max(
                int(component["bbox_width"]),
                int(component["bbox_height"]),
            )
            for component in components
        )
        mean_thickness = (
            float(final_pixels) / approximate_length
            if approximate_length > 0
            else 0.0
        )
        return BoundaryResult(
            yellow_edge_near_green=yellow_near_green,
            green_edge_near_yellow=green_near_yellow,
            gap_centerline=gap,
            candidate=candidate,
            boundary_region=boundary_region,
            final=final,
            component_count=component_count,
            components=components[:20],
            estimated_mean_thickness_px=round(mean_thickness, 3),
            valid=valid,
            reason=reason,
        )


class OverlayRenderer:
    """只在最终显示阶段叠加颜色和紫红色边界。"""

    @staticmethod
    def render(
        raw_bgr: np.ndarray,
        yellow: np.ndarray,
        green: np.ndarray,
        boundary: np.ndarray,
        roi_start: int,
        roi_end: int,
        cfg: Dict[str, object],
    ) -> np.ndarray:
        """所有识别完成后再绘图，overlay 永不回灌分割。"""
        overlay = raw_bgr.copy()
        tint = overlay.copy()
        tint[yellow > 0] = (0, 255, 255)
        tint[green > 0] = (0, 210, 0)
        cv2.addWeighted(tint, 0.38, overlay, 0.62, 0.0, overlay)
        height, width = overlay.shape[:2]
        cv2.line(
            overlay, (0, roi_start), (width - 1, roi_start),
            (255, 255, 0), 1
        )
        cv2.line(
            overlay,
            (0, min(roi_end, height - 1)),
            (width - 1, min(roi_end, height - 1)),
            (255, 255, 0),
            1,
        )
        thickness = int(cfg["boundary_overlay_thickness"])
        display_boundary = boundary
        if thickness > 1:
            size = max(1, thickness)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (size, size)
            )
            display_boundary = cv2.dilate(
                boundary, kernel, iterations=1
            )
        overlay[display_boundary > 0] = BOUNDARY_OVERLAY_COLOR
        return overlay


@dataclass
class RuntimeStatus:
    """ROS JSON 状态与 Web 状态的统一来源。"""

    values: Dict[str, object] = field(default_factory=dict)

    @staticmethod
    def defaults() -> Dict[str, object]:
        return {
            "interface_version": INTERFACE_VERSION,
            "boundary_topic": THIRD_STAGE_BOUNDARY_TOPIC,
            "yellow_topic": THIRD_STAGE_YELLOW_TOPIC,
            "green_topic": THIRD_STAGE_GREEN_TOPIC,
            "image_received": False,
            "image_topic": "/image_out",
            "frame_width": 0,
            "frame_height": 0,
            "encoding": "",
            "fps": 0.0,
            "last_frame_age_sec": None,
            "source_stamp_sec": 0,
            "source_stamp_nanosec": 0,
            "source_frame_id": "",
            "frame_sequence": 0,
            "segmentation_valid": False,
            "config_loaded": False,
            "threshold_config_path": "",
            "profile_name": "",
            "config_saved_at": "",
            "config_version": None,
            "config_image_topic": "",
            "config_frame_width": 0,
            "config_frame_height": 0,
            "config_error": "",
            "hsv_range_overlap": False,
            "hue_range_overlap_size": 0,
            "roi_y_start_ratio": 0.0,
            "roi_y_end_ratio": 1.0,
            "yellow_pixels_raw": 0,
            "green_pixels_raw": 0,
            "overlap_pixels": 0,
            "unknown_pixels_raw": 0,
            "yellow_pixels_final": 0,
            "green_pixels_final": 0,
            "unknown_pixels_final": 0,
            "yellow_component_count_raw": 0,
            "yellow_component_count_kept": 0,
            "green_component_count_raw": 0,
            "green_component_count_kept": 0,
            "main_yellow_found": False,
            "yellow_edge_near_green_pixels": 0,
            "green_edge_near_yellow_pixels": 0,
            "gap_centerline_pixels": 0,
            "boundary_candidate_pixels": 0,
            "boundary_region_pixels": 0,
            "final_boundary_pixels": 0,
            "estimated_boundary_mean_thickness_px": 0.0,
            "boundary_component_count": 0,
            "boundary_components": [],
            "boundary_valid": False,
            "boundary_reason": "no_image",
            "topology_repair_enable": False,
            "repaired_yellow_pixels": 0,
            "repaired_green_pixels": 0,
            "processing_time_ms": 0.0,
            "last_error": "",
        }

    def __post_init__(self) -> None:
        defaults = self.defaults()
        defaults.update(self.values)
        self.values = defaults


class SegmentationConfigStore:
    """第二阶段参数独立 JSON 持久化。"""

    def __init__(self, script_dir: Path) -> None:
        self.config_dir = script_dir / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def path(self, profile_name: str, explicit_path: str) -> Path:
        if explicit_path:
            path = Path(explicit_path).expanduser().resolve()
            if path.suffix.lower() != ".json":
                raise APIError(
                    "segmentation_config_path 必须是 JSON 文件"
                )
            return path
        profile = ThresholdConfigLoader.validate_profile(profile_name)
        return self.config_dir / f"{profile}_segmentation.json"

    @staticmethod
    def payload(
        profile_name: str,
        threshold_path: str,
        cfg: Dict[str, object],
    ) -> Dict[str, object]:
        return {
            "version": SEGMENTATION_CONFIG_VERSION,
            "profile_name": profile_name,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "threshold_config_path": threshold_path,
            "boundary": {
                "search_radius_px": cfg["boundary_search_radius_px"],
                "contour_thickness_px": cfg[
                    "boundary_contour_thickness_px"
                ],
                "close_kernel_size": cfg["boundary_close_kernel_size"],
                "close_iterations": cfg["boundary_close_iterations"],
                "min_component_pixels": cfg[
                    "boundary_min_component_pixels"
                ],
                "max_component_area_ratio": cfg[
                    "boundary_max_component_area_ratio"
                ],
                "min_span_px": cfg["boundary_min_span_px"],
                "min_valid_pixels": cfg["boundary_min_valid_pixels"],
                "overlay_color_bgr": list(BOUNDARY_OVERLAY_COLOR),
                "overlay_thickness": cfg["boundary_overlay_thickness"],
                "final_boundary_thickness_px": cfg[
                    "final_boundary_thickness_px"
                ],
            },
            "gap_centerline": {
                "enable": cfg["gap_centerline_enable"],
                "max_distance_to_color_px": cfg[
                    "gap_max_distance_to_color_px"
                ],
                "distance_balance_tolerance_px": cfg[
                    "gap_distance_balance_tolerance_px"
                ],
            },
            "main_yellow": {
                "prefer_center_connected": cfg[
                    "prefer_center_connected_yellow"
                ],
                "center_seed_x_min_ratio": cfg[
                    "center_seed_x_min_ratio"
                ],
                "center_seed_x_max_ratio": cfg[
                    "center_seed_x_max_ratio"
                ],
                "center_seed_y_min_ratio": cfg[
                    "center_seed_y_min_ratio"
                ],
                "center_seed_y_max_ratio": cfg[
                    "center_seed_y_max_ratio"
                ],
            },
            "topology_repair": {
                "enable": cfg["topology_repair_enable"],
                "repair_max_component_area_px": cfg[
                    "repair_max_component_area_px"
                ],
                "repair_max_area_ratio": cfg["repair_max_area_ratio"],
                "same_color_contact_ratio": cfg[
                    "same_color_contact_ratio"
                ],
                "opposite_color_max_contact_ratio": cfg[
                    "opposite_color_max_contact_ratio"
                ],
                "preserve_dark_unknown": cfg[
                    "preserve_dark_unknown"
                ],
                "dark_unknown_v_max": cfg["dark_unknown_v_max"],
                "max_iterations": cfg["repair_max_iterations"],
            },
        }

    @staticmethod
    def flatten(payload: Dict[str, object]) -> Dict[str, object]:
        cfg = copy.deepcopy(DEFAULT_SEGMENTATION_CONFIG)
        boundary = payload.get("boundary", {})
        gap = payload.get("gap_centerline", {})
        main = payload.get("main_yellow", {})
        repair = payload.get("topology_repair", {})
        if not all(
            isinstance(section, dict)
            for section in (boundary, gap, main, repair)
        ):
            raise APIError("第二阶段配置结构错误")
        mapping = {
            "boundary_search_radius_px": (boundary, "search_radius_px"),
            "boundary_contour_thickness_px": (
                boundary, "contour_thickness_px"
            ),
            "boundary_close_kernel_size": (
                boundary, "close_kernel_size"
            ),
            "boundary_close_iterations": (
                boundary, "close_iterations"
            ),
            "boundary_min_component_pixels": (
                boundary, "min_component_pixels"
            ),
            "boundary_max_component_area_ratio": (
                boundary, "max_component_area_ratio"
            ),
            "boundary_min_span_px": (boundary, "min_span_px"),
            "boundary_min_valid_pixels": (
                boundary, "min_valid_pixels"
            ),
            "boundary_overlay_thickness": (
                boundary, "overlay_thickness"
            ),
            "final_boundary_thickness_px": (
                boundary, "final_boundary_thickness_px"
            ),
            "gap_centerline_enable": (gap, "enable"),
            "gap_max_distance_to_color_px": (
                gap, "max_distance_to_color_px"
            ),
            "gap_distance_balance_tolerance_px": (
                gap, "distance_balance_tolerance_px"
            ),
            "prefer_center_connected_yellow": (
                main, "prefer_center_connected"
            ),
            "center_seed_x_min_ratio": (
                main, "center_seed_x_min_ratio"
            ),
            "center_seed_x_max_ratio": (
                main, "center_seed_x_max_ratio"
            ),
            "center_seed_y_min_ratio": (
                main, "center_seed_y_min_ratio"
            ),
            "center_seed_y_max_ratio": (
                main, "center_seed_y_max_ratio"
            ),
            "topology_repair_enable": (repair, "enable"),
            "repair_max_component_area_px": (
                repair, "repair_max_component_area_px"
            ),
            "repair_max_area_ratio": (
                repair, "repair_max_area_ratio"
            ),
            "same_color_contact_ratio": (
                repair, "same_color_contact_ratio"
            ),
            "opposite_color_max_contact_ratio": (
                repair, "opposite_color_max_contact_ratio"
            ),
            "preserve_dark_unknown": (
                repair, "preserve_dark_unknown"
            ),
            "dark_unknown_v_max": (repair, "dark_unknown_v_max"),
            "repair_max_iterations": (repair, "max_iterations"),
        }
        for name, (section, key) in mapping.items():
            cfg[name] = section.get(key, cfg[name])
        return SegmentationConfig.validate(cfg)


class GreenYellowSegmentationNode(Node):
    """协调配置、ROS 分割发布和 Web 调试。"""

    IMAGE_TOPICS = (
        ("yellow_raw", "/gxb_test/yellow_mask_raw"),
        ("green_raw", "/gxb_test/green_mask_raw"),
        ("unknown_raw", "/gxb_test/unknown_mask_raw"),
        ("yellow_final", THIRD_STAGE_YELLOW_TOPIC),
        ("green_final", THIRD_STAGE_GREEN_TOPIC),
        ("unknown_final", "/gxb_test/unknown_mask_final"),
        ("gap", "/gxb_test/gap_centerline_mask"),
        ("candidate", "/gxb_test/boundary_candidate_mask"),
        ("region", BOUNDARY_REGION_TOPIC),
        ("boundary", THIRD_STAGE_BOUNDARY_TOPIC),
    )

    def __init__(self) -> None:
        super().__init__("green_yellow_segmentation")
        defaults: Dict[str, object] = {
            "image_topic": "/image_out",
            "profile_name": "usb_camera",
            "threshold_config_path": "",
            "threshold_config_required": True,
            "auto_reload_threshold_config": False,
            "segmentation_config_path": "",
            "auto_load_segmentation_config": True,
            "web_gui_enable": True,
            "web_gui_host": "0.0.0.0",
            "web_gui_port": 8090,
            "web_gui_jpeg_quality": 92,
            "web_gui_max_fps": 6.0,
            **DEFAULT_SEGMENTATION_CONFIG,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        params = {
            name: self.get_parameter(name).value for name in defaults
        }

        self.script_dir = Path(__file__).resolve().parent
        self.image_topic = str(params["image_topic"])
        self.profile_name = ThresholdConfigLoader.validate_profile(
            str(params["profile_name"])
        )
        self.threshold_explicit_path = str(
            params["threshold_config_path"]
        )
        self.threshold_required = bool(
            params["threshold_config_required"]
        )
        self.auto_reload_threshold = bool(
            params["auto_reload_threshold_config"]
        )
        self.segmentation_explicit_path = str(
            params["segmentation_config_path"]
        )
        self.web_enabled = bool(params["web_gui_enable"])
        self.web_host = str(params["web_gui_host"])
        self.web_port = int(params["web_gui_port"])
        self.jpeg_quality = int(
            np.clip(params["web_gui_jpeg_quality"], 1, 100)
        )
        self.web_max_fps = max(
            0.1, float(params["web_gui_max_fps"])
        )

        self.cfg_lock = threading.RLock()
        self.threshold_lock = threading.RLock()
        self.status_lock = threading.RLock()
        self.frame_condition = threading.Condition(self.status_lock)
        self.cfg = SegmentationConfig(
            SegmentationConfig.validate({
                name: params[name]
                for name in DEFAULT_SEGMENTATION_CONFIG
            })
        )
        self.threshold = ThresholdConfig()
        self.status = RuntimeStatus({
            "image_topic": self.image_topic,
            "profile_name": self.profile_name,
        })
        self.latest_frames: Dict[str, bytes] = {}
        self.last_frame_time = 0.0
        self.last_callback_time = 0.0
        self.last_encode_time = 0.0
        self.frame_sequence = 0
        self.latest_raw_bgr: Optional[np.ndarray] = None
        self.http_server: Optional[ThreadingHTTPServer] = None
        self.http_thread: Optional[threading.Thread] = None
        self.store = SegmentationConfigStore(self.script_dir)

        self.reload_threshold_config()
        if bool(params["auto_load_segmentation_config"]):
            try:
                self.load_segmentation_config(require_exists=False)
            except APIError as exc:
                with self.status_lock:
                    self.status.values["last_error"] = str(exc)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.image_publishers = {
            name: self.create_publisher(Image, topic, qos)
            for name, topic in self.IMAGE_TOPICS
        }
        self.overlay_publisher = self.create_publisher(
            Image, "/gxb_test/segmentation_overlay", qos
        )
        self.status_publisher = self.create_publisher(
            String, THIRD_STAGE_STATUS_TOPIC, qos
        )
        self.create_subscription(Image, self.image_topic, self._on_image, qos)
        if self.web_enabled:
            self._start_web()
        self.get_logger().info(
            f"第二阶段已启动：image_topic={self.image_topic}, "
            f"profile={self.profile_name}"
        )

    def cfg_snapshot(self) -> Dict[str, object]:
        with self.cfg_lock:
            return self.cfg.snapshot()

    def threshold_snapshot(self) -> ThresholdConfig:
        with self.threshold_lock:
            return self.threshold.snapshot()

    def threshold_path(self) -> Path:
        return ThresholdConfigLoader.resolve_path(
            self.script_dir,
            self.profile_name,
            self.threshold_explicit_path,
        )

    def reload_threshold_config(self) -> Dict[str, object]:
        """重新读取第一阶段 JSON，失败时清空有效分割状态。"""
        path = self.threshold_path()
        loaded = ThresholdConfigLoader.load(path)
        with self.threshold_lock:
            self.threshold = loaded
        with self.status_lock:
            missing_reason = (
                "config_missing"
                if "MISSING" in loaded.error
                else "config_invalid"
            )
            self.status.values.update({
                "config_loaded": loaded.loaded,
                "segmentation_valid": False,
                "threshold_config_path": loaded.path,
                "config_saved_at": loaded.saved_at,
                "config_version": loaded.version,
                "config_image_topic": loaded.image_topic,
                "config_frame_width": loaded.frame_width,
                "config_frame_height": loaded.frame_height,
                "config_error": loaded.error,
                "boundary_valid": False,
                "boundary_reason": (
                    "no_image" if loaded.loaded else missing_reason
                ),
                "last_error": loaded.error,
            })
        if loaded.loaded:
            self.get_logger().info(f"已加载阈值配置：{loaded.path}")
        else:
            log_message = loaded.error
            if self.threshold_required:
                self.get_logger().error(log_message)
            else:
                self.get_logger().warning(log_message)
        return {
            "ok": loaded.loaded,
            "config_loaded": loaded.loaded,
            "path": loaded.path,
            "error": loaded.error,
        }

    def _maybe_auto_reload(self) -> None:
        if not self.auto_reload_threshold:
            return
        current = self.threshold_snapshot()
        path = Path(current.path) if current.path else self.threshold_path()
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            mtime = -1
        if mtime != current.mtime_ns:
            self.reload_threshold_config()

    def update_parameter(self, name: str, raw_value: str) -> Dict[str, object]:
        """原子更新第二阶段参数，不提供 HSV 修改入口。"""
        if name not in PARAMETER_SCHEMA:
            raise APIError(f"第二阶段参数不可调：{name}", name)
        schema = PARAMETER_SCHEMA[name]
        with self.cfg_lock:
            candidate = self.cfg.snapshot()
            old = candidate[name]
            try:
                if schema[4] == "bool":
                    lowered = str(raw_value).strip().lower()
                    if lowered not in ("true", "false", "1", "0"):
                        raise ValueError
                    value: object = lowered in ("true", "1")
                elif isinstance(old, int) and not isinstance(old, bool):
                    value = int(round(float(raw_value)))
                else:
                    value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise APIError(f"{name} 的值无效", name) from exc
            if schema[1] is not None and (
                float(value) < float(schema[1])
                or float(value) > float(schema[2])
            ):
                raise APIError(
                    f"{name} 必须位于 {schema[1]} 到 {schema[2]}",
                    name,
                )
            candidate[name] = value
            validated = SegmentationConfig.validate(candidate)
            self.cfg = SegmentationConfig(validated)
        return {"ok": True, "name": name, "value": validated[name]}

    def _empty_result(self, bgr: np.ndarray) -> SegmentationResult:
        """配置无效时生成明确的全零 mask，不伪造分割。"""
        height, width = bgr.shape[:2]
        zero = np.zeros((height, width), dtype=np.uint8)
        boundary = BoundaryResult(
            yellow_edge_near_green=zero.copy(),
            green_edge_near_yellow=zero.copy(),
            gap_centerline=zero.copy(),
            candidate=zero.copy(),
            boundary_region=zero.copy(),
            final=zero.copy(),
            component_count=0,
            components=[],
            estimated_mean_thickness_px=0.0,
            valid=False,
            reason="config_missing",
        )
        return SegmentationResult(
            bgr,
            bgr.copy(),
            zero.copy(),
            zero.copy(),
            zero.copy(),
            zero.copy(),
            zero.copy(),
            zero.copy(),
            boundary,
            0,
            height,
            0,
            ComponentStats(),
            ComponentStats(),
        )

    def _process(
        self,
        bgr: np.ndarray,
        threshold: ThresholdConfig,
        cfg: Dict[str, object],
    ) -> SegmentationResult:
        (
            yellow_raw,
            green_raw,
            unknown_raw,
            yellow_final,
            green_final,
            unknown_final,
            _hsv_roi,
            roi_start,
            roi_end,
            overlap_pixels,
            yellow_stats,
            green_stats,
        ) = GreenYellowSegmenter.segment(bgr, threshold, cfg)
        boundary = BoundaryExtractor.extract(
            yellow_final,
            green_final,
            unknown_final,
            roi_start,
            roi_end,
            cfg,
        )
        overlay = OverlayRenderer.render(
            bgr,
            yellow_final,
            green_final,
            boundary.final,
            roi_start,
            roi_end,
            cfg,
        )
        return SegmentationResult(
            raw_bgr=bgr,
            overlay=overlay,
            yellow_raw=yellow_raw,
            green_raw=green_raw,
            unknown_raw=unknown_raw,
            yellow_final=yellow_final,
            green_final=green_final,
            unknown_final=unknown_final,
            boundary=boundary,
            roi_start=roi_start,
            roi_end=roi_end,
            overlap_pixels=overlap_pixels,
            yellow_stats=yellow_stats,
            green_stats=green_stats,
        )

    def _on_image(self, msg: Image) -> None:
        """一帧使用同一配置快照，异常时等待下一帧。"""
        start = time.monotonic()
        try:
            bgr = ImageCodec.to_bgr(msg)
            self.frame_sequence += 1
            self._maybe_auto_reload()
            threshold = self.threshold_snapshot()
            cfg = self.cfg_snapshot()
            if threshold.loaded:
                result = self._process(bgr, threshold, cfg)
            else:
                result = self._empty_result(bgr)
            now = time.monotonic()
            elapsed = now - self.last_callback_time
            instantaneous_fps = 1.0 / elapsed if elapsed > 1e-6 else 0.0
            self.last_callback_time = now
            self.last_frame_time = now
            processing_ms = (now - start) * 1000.0
            self._update_status(
                msg, result, threshold, cfg, instantaneous_fps,
                processing_ms
            )
            self._publish_result(msg, result)
            if (
                self.web_enabled
                and now - self.last_encode_time >= 1.0 / self.web_max_fps
            ):
                self._encode_web(result)
                self.last_encode_time = now
        except Exception as exc:
            with self.status_lock:
                self.status.values["last_error"] = (
                    f"图像处理失败：{exc}"
                )
                self.status.values["segmentation_valid"] = False
                self.status.values["boundary_valid"] = False
                self.status.values["boundary_reason"] = "processing_error"
            self._publish_status()

    def _update_status(
        self,
        msg: Image,
        result: SegmentationResult,
        threshold: ThresholdConfig,
        cfg: Dict[str, object],
        instantaneous_fps: float,
        processing_ms: float,
    ) -> None:
        with self.status_lock:
            previous_fps = float(self.status.values["fps"])
            fps = (
                instantaneous_fps
                if previous_fps <= 0.0
                else 0.15 * instantaneous_fps + 0.85 * previous_fps
            )
            hue_overlap = 0
            if threshold.loaded:
                hue_overlap = max(
                    0,
                    int(threshold.yellow["h_max"])
                    - int(threshold.green["h_min"])
                    + 1,
                )
            self.latest_raw_bgr = result.raw_bgr.copy()
            header = getattr(msg, "header", None)
            stamp = getattr(header, "stamp", None)
            self.status.values.update({
                "interface_version": INTERFACE_VERSION,
                "boundary_topic": THIRD_STAGE_BOUNDARY_TOPIC,
                "yellow_topic": THIRD_STAGE_YELLOW_TOPIC,
                "green_topic": THIRD_STAGE_GREEN_TOPIC,
                "image_received": True,
                "frame_width": int(result.raw_bgr.shape[1]),
                "frame_height": int(result.raw_bgr.shape[0]),
                "encoding": str(msg.encoding),
                "fps": fps,
                "last_frame_age_sec": 0.0,
                "source_stamp_sec": int(
                    getattr(stamp, "sec", 0)
                ),
                "source_stamp_nanosec": int(
                    getattr(stamp, "nanosec", 0)
                ),
                "source_frame_id": str(
                    getattr(header, "frame_id", "")
                ),
                "frame_sequence": self.frame_sequence,
                "config_loaded": threshold.loaded,
                "segmentation_valid": threshold.loaded,
                "threshold_config_path": threshold.path,
                "profile_name": self.profile_name,
                "config_saved_at": threshold.saved_at,
                "config_error": threshold.error,
                "roi_y_start_ratio": threshold.roi_y_start_ratio,
                "roi_y_end_ratio": threshold.roi_y_end_ratio,
                "hsv_range_overlap": hue_overlap > 0,
                "hue_range_overlap_size": hue_overlap,
                "yellow_pixels_raw": int(
                    cv2.countNonZero(result.yellow_raw)
                ),
                "green_pixels_raw": int(
                    cv2.countNonZero(result.green_raw)
                ),
                "overlap_pixels": result.overlap_pixels,
                "unknown_pixels_raw": int(
                    cv2.countNonZero(result.unknown_raw)
                ),
                "yellow_pixels_final": int(
                    cv2.countNonZero(result.yellow_final)
                ),
                "green_pixels_final": int(
                    cv2.countNonZero(result.green_final)
                ),
                "unknown_pixels_final": int(
                    cv2.countNonZero(result.unknown_final)
                ),
                "yellow_component_count_raw":
                    result.yellow_stats.raw_count,
                "yellow_component_count_kept":
                    result.yellow_stats.kept_count,
                "green_component_count_raw":
                    result.green_stats.raw_count,
                "green_component_count_kept":
                    result.green_stats.kept_count,
                "main_yellow_found":
                    result.yellow_stats.main_found,
                "yellow_edge_near_green_pixels": int(
                    cv2.countNonZero(
                        result.boundary.yellow_edge_near_green
                    )
                ),
                "green_edge_near_yellow_pixels": int(
                    cv2.countNonZero(
                        result.boundary.green_edge_near_yellow
                    )
                ),
                "gap_centerline_pixels": int(
                    cv2.countNonZero(result.boundary.gap_centerline)
                ),
                "boundary_candidate_pixels": int(
                    cv2.countNonZero(result.boundary.candidate)
                ),
                "boundary_region_pixels": int(
                    cv2.countNonZero(
                        result.boundary.boundary_region
                    )
                ),
                "final_boundary_pixels": int(
                    cv2.countNonZero(result.boundary.final)
                ),
                "estimated_boundary_mean_thickness_px":
                    result.boundary.estimated_mean_thickness_px,
                "boundary_component_count":
                    result.boundary.component_count,
                "boundary_components":
                    copy.deepcopy(result.boundary.components),
                "boundary_valid": (
                    threshold.loaded and result.boundary.valid
                ),
                "boundary_reason": (
                    result.boundary.reason
                    if threshold.loaded
                    else (
                        "config_missing"
                        if "MISSING" in threshold.error
                        else "config_invalid"
                    )
                ),
                "topology_repair_enable": bool(
                    cfg["topology_repair_enable"]
                ),
                "repaired_yellow_pixels": max(
                    0,
                    int(cv2.countNonZero(result.yellow_final))
                    - int(cv2.countNonZero(result.yellow_raw)),
                ),
                "repaired_green_pixels": max(
                    0,
                    int(cv2.countNonZero(result.green_final))
                    - int(cv2.countNonZero(result.green_raw)),
                ),
                "processing_time_ms": processing_ms,
                "last_error": threshold.error,
            })

    def _publish_result(
        self, source: Image, result: SegmentationResult
    ) -> None:
        masks = {
            "yellow_raw": result.yellow_raw,
            "green_raw": result.green_raw,
            "unknown_raw": result.unknown_raw,
            "yellow_final": result.yellow_final,
            "green_final": result.green_final,
            "unknown_final": result.unknown_final,
            "gap": result.boundary.gap_centerline,
            # 兼容旧调试话题：candidate 与新 region 都发布清理后的
            # 候选分界区域；正式接口始终是 boundary。
            "candidate": result.boundary.boundary_region,
            "region": result.boundary.boundary_region,
            "boundary": result.boundary.final,
        }
        for name, mask in masks.items():
            self.image_publishers[name].publish(
                ImageCodec.to_message(mask, "mono8", source)
            )
        self.overlay_publisher.publish(
            ImageCodec.to_message(result.overlay, "bgr8", source)
        )
        self._publish_status()

    def _publish_status(self) -> None:
        message = String()
        message.data = json.dumps(
            self.api_status(), ensure_ascii=False, separators=(",", ":")
        )
        self.status_publisher.publish(message)

    def _encode_web(self, result: SegmentationResult) -> None:
        frames = {
            "raw": ImageCodec.encode_jpeg(
                result.raw_bgr, self.jpeg_quality
            ),
            "overlay": ImageCodec.encode_jpeg(
                result.overlay, self.jpeg_quality
            ),
            "yellow": ImageCodec.encode_jpeg(
                cv2.cvtColor(
                    result.yellow_final, cv2.COLOR_GRAY2BGR
                ),
                100,
            ),
            "green": ImageCodec.encode_jpeg(
                cv2.cvtColor(
                    result.green_final, cv2.COLOR_GRAY2BGR
                ),
                100,
            ),
            "unknown": ImageCodec.encode_jpeg(
                cv2.cvtColor(
                    result.unknown_final, cv2.COLOR_GRAY2BGR
                ),
                100,
            ),
            "gap": ImageCodec.encode_jpeg(
                cv2.cvtColor(
                    result.boundary.gap_centerline,
                    cv2.COLOR_GRAY2BGR,
                ),
                100,
            ),
            "candidate": ImageCodec.encode_jpeg(
                cv2.cvtColor(
                    result.boundary.boundary_region,
                    cv2.COLOR_GRAY2BGR,
                ),
                100,
            ),
            "region": ImageCodec.encode_jpeg(
                cv2.cvtColor(
                    result.boundary.boundary_region,
                    cv2.COLOR_GRAY2BGR,
                ),
                100,
            ),
            "boundary": ImageCodec.encode_jpeg(
                cv2.cvtColor(
                    result.boundary.final, cv2.COLOR_GRAY2BGR
                ),
                100,
            ),
        }
        with self.frame_condition:
            self.latest_frames = frames
            self.frame_condition.notify_all()

    def api_status(self) -> Dict[str, object]:
        with self.status_lock:
            output = copy.deepcopy(self.status.values)
        if self.last_frame_time > 0.0:
            output["last_frame_age_sec"] = max(
                0.0, time.monotonic() - self.last_frame_time
            )
        return output

    def api_config(self) -> Dict[str, object]:
        threshold = self.threshold_snapshot()
        cfg = self.cfg_snapshot()
        return {
            "threshold_source": {
                "loaded": threshold.loaded,
                "path": threshold.path,
                "profile_name": threshold.profile_name,
                "saved_at": threshold.saved_at,
                "version": threshold.version,
                "roi": {
                    "y_start_ratio": threshold.roi_y_start_ratio,
                    "y_end_ratio": threshold.roi_y_end_ratio,
                },
                "yellow": threshold.yellow,
                "green": threshold.green,
                "morphology": threshold.morphology,
                "error": threshold.error,
                "note": (
                    "来源：第一阶段 JSON。需要修改阈值时返回第一阶段"
                    "重新保存，然后点击重新加载。"
                ),
            },
            "segmentation_params": cfg,
            "schema": {
                name: list(schema)
                for name, schema in PARAMETER_SCHEMA.items()
            },
        }

    def save_segmentation_config(self) -> Dict[str, object]:
        path = self.store.path(
            self.profile_name, self.segmentation_explicit_path
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.store.payload(
            self.profile_name,
            self.threshold_snapshot().path,
            self.cfg_snapshot(),
        )
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return {"ok": True, "path": str(path)}

    def load_segmentation_config(
        self, require_exists: bool = True
    ) -> Dict[str, object]:
        path = self.store.path(
            self.profile_name, self.segmentation_explicit_path
        )
        if not path.exists():
            if require_exists:
                raise APIError(f"第二阶段配置不存在：{path}")
            return {"ok": True, "loaded": False, "path": str(path)}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise APIError("第二阶段 JSON 顶层必须是对象")
            loaded = self.store.flatten(payload)
        except (OSError, json.JSONDecodeError, APIError) as exc:
            raise APIError(f"加载第二阶段配置失败：{exc}") from exc
        with self.cfg_lock:
            self.cfg = SegmentationConfig(loaded)
        return {"ok": True, "loaded": True, "path": str(path)}

    def _start_web(self) -> None:
        web = SegmentationWebServer(self)
        self.http_server = ThreadingHTTPServer(
            (self.web_host, self.web_port), web.handler_class()
        )
        self.http_server.daemon_threads = True
        self.http_thread = threading.Thread(
            target=self.http_server.serve_forever,
            name="segmentation-web-gui",
            daemon=True,
        )
        self.http_thread.start()
        self.get_logger().info(
            f"Web GUI: http://{self.web_host}:{self.web_port}"
        )
        if self.web_host == "0.0.0.0":
            self.get_logger().warning(
                "Web GUI 无身份认证，请仅在可信机器人网络中使用。"
            )

    def shutdown(self) -> None:
        server = self.http_server
        self.http_server = None
        if server is not None:
            server.shutdown()
            server.server_close()
        with self.frame_condition:
            self.frame_condition.notify_all()


WEB_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>黄绿道路分割</title>
<style>
body{margin:0;background:#111;color:#eee;font:14px system-ui}
header{position:sticky;top:0;z-index:3;background:#222;padding:9px;display:flex;
gap:10px;align-items:center;flex-wrap:wrap}.warn{color:#ffb347}.bad{color:#ff6969}
.good{color:#72db84}button{padding:7px 11px}.layout{display:grid;
grid-template-columns:minmax(520px,2fr) minmax(320px,1fr);gap:10px;padding:10px}
.card{background:#202020;padding:9px;border-radius:6px;margin-bottom:9px}
.grid2{display:grid;grid-template-columns:repeat(2,minmax(240px,1fr));gap:9px}
.grid3{display:grid;grid-template-columns:repeat(3,minmax(180px,1fr));gap:9px}
img{width:100%;height:auto;object-fit:contain;background:#181818}
.row{display:grid;grid-template-columns:1fr minmax(165px,1fr);gap:6px;margin:5px 0}
.ctrl{display:grid;grid-template-columns:1fr 72px;gap:5px}
input{width:100%;box-sizing:border-box;background:#333;color:#fff}
pre{white-space:pre-wrap;word-break:break-word}
@media(max-width:1000px){.layout{grid-template-columns:1fr}.grid2,.grid3{grid-template-columns:1fr}}
</style></head><body>
<header><b>第二阶段：黄绿道路分割与公共边界</b><span id="topic"></span>
<span id="size">等待相机图像</span><span id="fps"></span><span id="profile"></span>
<b id="configState"></b><b id="boundaryState"></b>
<button onclick="action('/api/reload_config')">重新加载阈值配置</button>
<button onclick="action('/api/save_segmentation_config')">保存第二阶段参数</button>
<button onclick="action('/api/load_segmentation_config')">加载第二阶段参数</button>
<span class="warn">Web GUI 无身份认证，请仅在可信机器人网络中使用。</span></header>
<div class="layout"><section>
<div class="grid2"><div class="card"><b>完整原图</b><img src="/stream/raw"></div>
<div class="card"><b>最终 Overlay（紫红色公共边界）</b><img src="/stream/overlay"></div></div>
<div class="grid3"><div class="card"><b>Yellow mask final</b><img src="/stream/yellow"></div>
<div class="card"><b>Green mask final</b><img src="/stream/green"></div>
<div class="card"><b>Unknown mask final</b><img src="/stream/unknown"></div></div>
<div class="grid2"><div class="card"><b>Boundary region（调试候选区域）</b>
<img src="/stream/region"></div>
<div class="card"><b>Final boundary line（第三阶段正式接口）</b>
<img src="/stream/boundary"></div></div>
<pre id="status" class="card">等待相机图像</pre></section>
<aside><div class="card"><b>第一阶段阈值（只读）</b>
<p>来源：第一阶段 JSON 配置。需要修改 HSV 时，请运行
1_auto_green_yellow_threshold.py 保存后重新加载。</p><pre id="threshold"></pre></div>
<div id="controls"></div><pre id="message" class="card"></pre></aside></div>
<script>
let cfg={},schema={};
async function jf(url,opt){let r=await fetch(url,opt),x=await r.json();
if(!r.ok)throw Error(x.error||r.statusText);return x}
function field(n,v,s){let row=document.createElement('div');row.className='row';
let label=document.createElement('label');label.textContent=n;row.appendChild(label);
let c;if(s[4]==='bool'){c=document.createElement('input');c.type='checkbox';c.checked=!!v;
c.onchange=()=>setv(n,c.checked)}else{c=document.createElement('div');c.className='ctrl';
let range=document.createElement('input'),number=document.createElement('input');
range.type='range';number.type='number';[range,number].forEach(e=>{e.min=s[1];e.max=s[2];
e.step=s[3];e.value=v});range.oninput=()=>number.value=range.value;
number.oninput=()=>range.value=number.value;range.onchange=()=>setv(n,range.value);
number.onchange=()=>setv(n,number.value);c.append(range,number)}row.appendChild(c);return row}
async function loadConfig(){let x=await jf('/api/config');cfg=x.segmentation_params;schema=x.schema;
document.getElementById('threshold').textContent=JSON.stringify(x.threshold_source,null,2);
let groups={};Object.keys(schema).forEach(n=>(groups[schema[n][0]]??=[]).push(n));
let root=document.getElementById('controls');root.innerHTML='';
Object.keys(groups).forEach(g=>{let box=document.createElement('div');box.className='card';
box.innerHTML='<b>'+g+'</b>';groups[g].forEach(n=>box.appendChild(field(n,cfg[n],schema[n])));
root.appendChild(box)})}
async function setv(n,v){try{show(JSON.stringify(await jf('/api/set?name='+
encodeURIComponent(n)+'&value='+encodeURIComponent(v),{method:'POST'})))}
catch(e){show(e.message,true)}await loadConfig()}
async function action(url){try{show(JSON.stringify(await jf(url,{method:'POST'}),null,2));
await loadConfig()}catch(e){show(e.message,true)}}
function show(t,bad=false){let e=document.getElementById('message');e.textContent=t;
e.className='card '+(bad?'bad':'good')}
async function tick(){try{let s=await jf('/api/status');document.getElementById('topic').
textContent=s.image_topic;document.getElementById('size').textContent=s.image_received?
s.frame_width+'×'+s.frame_height:'等待相机图像';document.getElementById('fps').
textContent=(s.fps||0).toFixed(1)+' FPS';document.getElementById('profile').
textContent=s.profile_name;let c=document.getElementById('configState');
c.textContent=s.config_loaded?'CONFIG LOADED':(s.config_error||'CONFIG MISSING');
c.className=s.config_loaded?'good':'bad';let b=document.getElementById('boundaryState');
b.textContent=s.boundary_valid?'BOUNDARY VALID':s.boundary_reason;
b.className=s.boundary_valid?'good':'bad';document.getElementById('status').
textContent=JSON.stringify(s,null,2)}catch(e){}}
loadConfig();tick();setInterval(tick,500);
</script></body></html>"""


class SegmentationWebServer:
    """Python 标准库 Web/MJPEG 服务。"""

    STREAM_NAMES = {
        "raw", "overlay", "yellow", "green", "unknown",
        "gap", "candidate", "region", "boundary",
    }

    def __init__(self, node: GreenYellowSegmentationNode) -> None:
        self.node = node

    def handler_class(self):
        node = self.node

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args) -> None:
                return

            def send_bytes(
                self, status: int, body: bytes, content_type: str
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def send_json(self, value: object, status: int = 200) -> None:
                self.send_bytes(
                    status,
                    json.dumps(value, ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                )

            def stream(self, name: str) -> None:
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last_frame: Optional[bytes] = None
                while node.http_server is not None:
                    with node.frame_condition:
                        node.frame_condition.wait(timeout=1.0)
                        frame = node.latest_frames.get(name)
                    if frame is None or frame is last_frame:
                        continue
                    last_frame = frame
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(frame)}\r\n\r\n".
                        encode("ascii")
                    )
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")

            def route(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                query = urllib.parse.parse_qs(parsed.query)
                try:
                    if parsed.path == "/":
                        self.send_bytes(
                            200,
                            WEB_HTML.encode("utf-8"),
                            "text/html; charset=utf-8",
                        )
                    elif parsed.path.startswith("/stream/"):
                        name = parsed.path.rsplit("/", 1)[-1]
                        if name not in SegmentationWebServer.STREAM_NAMES:
                            raise APIError("未知图像流", status=404)
                        self.stream(name)
                    elif parsed.path == "/api/status":
                        self.send_json(node.api_status())
                    elif parsed.path == "/api/config":
                        self.send_json(node.api_config())
                    elif parsed.path == "/api/reload_config":
                        self.send_json(node.reload_threshold_config())
                    elif parsed.path == "/api/set":
                        self.send_json(
                            node.update_parameter(
                                query.get("name", [""])[0],
                                query.get("value", [""])[0],
                            )
                        )
                    elif parsed.path == "/api/save_segmentation_config":
                        self.send_json(node.save_segmentation_config())
                    elif parsed.path == "/api/load_segmentation_config":
                        self.send_json(
                            node.load_segmentation_config(True)
                        )
                    else:
                        raise APIError("接口不存在", status=404)
                except (BrokenPipeError, ConnectionResetError):
                    return
                except APIError as exc:
                    self.send_json(
                        {
                            "ok": False,
                            "error": str(exc),
                            "parameter": exc.parameter,
                        },
                        exc.status,
                    )
                except Exception as exc:
                    self.send_json(
                        {
                            "ok": False,
                            "error": str(exc),
                            "parameter": "",
                        },
                        400,
                    )

            def do_GET(self) -> None:
                self.route()

            def do_POST(self) -> None:
                self.route()

        return Handler


def main(args=None) -> int:
    """启动独立 ROS2 第二阶段节点。"""
    rclpy.init(args=args)
    node: Optional[GreenYellowSegmentationNode] = None
    exit_code = 1
    try:
        node = GreenYellowSegmentationNode()
        rclpy.spin(node)
        exit_code = 0
    except KeyboardInterrupt:
        exit_code = 0
    except Exception as exc:
        if node is not None:
            node.get_logger().error(f"第二阶段异常：{exc}")
        else:
            print(f"第二阶段启动失败：{exc}", flush=True)
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
